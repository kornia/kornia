# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

from kornia.feature.aliked import ALIKED, ALIKEDFeatures
from kornia.feature.aliked.deform_conv2d import deform_conv2d

from testing.base import BaseTester


def _has_torchvision() -> bool:
    try:
        import torchvision.ops  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# deform_conv2d tests
# ---------------------------------------------------------------------------


class TestDeformConv2d:
    """Tests for the pure-PyTorch deform_conv2d implementation."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_zero_offset_matches_regular_conv(self, device, dtype):
        """With all-zero offsets, deform_conv2d should equal regular conv2d."""
        B, C_in, H, W = 1, 4, 8, 8
        C_out, kH, kW = 8, 3, 3
        K = kH * kW

        x = torch.randn(B, C_in, H, W, device=device, dtype=dtype)
        weight = torch.randn(C_out, C_in, kH, kW, device=device, dtype=dtype)
        offset = torch.zeros(B, 2 * K, H, W, device=device, dtype=dtype)

        out_dcn = deform_conv2d(x, offset, weight, padding=1)
        out_reg = torch.nn.functional.conv2d(x, weight, padding=1)

        assert out_dcn.shape == out_reg.shape
        assert torch.allclose(out_dcn, out_reg, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_output_shape(self, device, dtype):
        """Check output spatial dimensions for various settings."""
        B, C_in, H, W = 2, 3, 16, 16
        C_out, kH, kW = 6, 3, 3
        K = kH * kW
        padding, stride = 1, 1

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1

        x = torch.randn(B, C_in, H, W, device=device, dtype=dtype)
        weight = torch.randn(C_out, C_in, kH, kW, device=device, dtype=dtype)
        offset = torch.zeros(B, 2 * K, H_out, W_out, device=device, dtype=dtype)

        out = deform_conv2d(x, offset, weight, padding=padding, stride=stride)
        assert out.shape == (B, C_out, H_out, W_out)

    @pytest.mark.skipif(not _has_torchvision(), reason="torchvision not installed")
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_matches_torchvision(self, device, dtype, use_mask):
        """Pure-PyTorch implementation should match torchvision reference."""
        import torchvision.ops as tvops

        B, C_in, H, W = 2, 4, 10, 10
        C_out, kH, kW = 8, 3, 3
        K = kH * kW
        padding = 1

        H_out = H  # padding=1, stride=1, 3x3 kernel
        W_out = W

        torch.manual_seed(0)
        x = torch.randn(B, C_in, H, W, device=device, dtype=dtype)
        weight = torch.randn(C_out, C_in, kH, kW, device=device, dtype=dtype)
        bias = torch.randn(C_out, device=device, dtype=dtype)
        offset = torch.randn(B, 2 * K, H_out, W_out, device=device, dtype=dtype) * 0.5
        mask = torch.rand(B, K, H_out, W_out, device=device, dtype=dtype) if use_mask else None

        out_ours = deform_conv2d(x, offset, weight, bias=bias, padding=padding, mask=mask)
        out_tv = tvops.deform_conv2d(x, offset, weight, bias=bias, padding=padding, mask=mask)

        assert torch.allclose(out_ours, out_tv, atol=1e-4), f"Max diff: {(out_ours - out_tv).abs().max():.2e}"

    def test_gradients(self, device):
        """Gradients should flow through the pure-PyTorch implementation."""
        B, C_in, H, W = 1, 2, 6, 6
        C_out, kH, kW = 4, 3, 3
        K = kH * kW

        x = torch.randn(B, C_in, H, W, device=device, requires_grad=True)
        weight = torch.randn(C_out, C_in, kH, kW, device=device, requires_grad=True)
        offset = torch.randn(B, 2 * K, H, W, device=device, requires_grad=True) * 0.1

        out = deform_conv2d(x, offset, weight, padding=1)
        out.sum().backward()
        assert x.grad is not None
        assert weight.grad is not None
        assert offset.grad is not None


# ---------------------------------------------------------------------------
# ALIKED tests
# ---------------------------------------------------------------------------


class TestALIKED(BaseTester):
    def test_smoke(self, dtype, device):
        aliked = ALIKED().to(device, dtype)
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        output = aliked(inp)
        assert isinstance(output, list)
        assert len(output) == 1
        assert isinstance(output[0], ALIKEDFeatures)

    def test_smoke_batch(self, dtype, device):
        aliked = ALIKED().to(device, dtype)
        inp = torch.rand(2, 3, 64, 64, device=device, dtype=dtype)
        output = aliked(inp)
        assert len(output) == 2
        assert all(isinstance(f, ALIKEDFeatures) for f in output)

    def test_grayscale_input(self, dtype, device):
        """1-channel input should be broadcast to 3 channels automatically."""
        aliked = ALIKED().to(device, dtype)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        output = aliked(inp)
        assert isinstance(output[0], ALIKEDFeatures)

    def test_output_shapes(self, dtype, device):
        """Keypoints should be (N,2), descriptors (N,D), scores (N,)."""
        aliked = ALIKED(model_name="aliked-t16").to(device, dtype)
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        output = aliked(inp)
        feat = output[0]
        N = feat.n
        assert feat.keypoints.shape == (N, 2)
        assert feat.descriptors.shape[0] == N
        assert feat.keypoint_scores.shape == (N,)

    def test_descriptor_normalized(self, dtype, device):
        """Descriptors should be L2-normalised."""
        aliked = ALIKED(model_name="aliked-t16").to(device, dtype)
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        output = aliked(inp)
        feat = output[0]
        if feat.n > 0:
            norms = feat.descriptors.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_keypoints_in_image(self, dtype, device):
        """Keypoints should lie within the image boundaries."""
        H, W = 64, 64
        aliked = ALIKED(model_name="aliked-t16").to(device, dtype)
        inp = torch.rand(1, 3, H, W, device=device, dtype=dtype)
        output = aliked(inp)
        feat = output[0]
        if feat.n > 0:
            assert (feat.keypoints[:, 0] >= 0).all()
            assert (feat.keypoints[:, 0] <= W - 1).all()
            assert (feat.keypoints[:, 1] >= 0).all()
            assert (feat.keypoints[:, 1] <= H - 1).all()

    @pytest.mark.parametrize("model_name", ["aliked-t16", "aliked-n16", "aliked-n32"])
    def test_model_configs(self, device, model_name):
        """All model configurations should run without error."""
        aliked = ALIKED(model_name=model_name).to(device)
        inp = torch.rand(1, 3, 64, 64, device=device)
        output = aliked(inp)
        assert len(output) == 1

    def test_aliked_features_to(self, device):
        """ALIKEDFeatures.to() should move all tensors."""
        feat = ALIKEDFeatures(
            keypoints=torch.rand(10, 2),
            descriptors=torch.rand(10, 64),
            keypoint_scores=torch.rand(10),
        )
        feat_dev = feat.to(device)
        assert feat_dev.keypoints.device.type == device.type
        assert feat_dev.descriptors.device.type == device.type

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", ["aliked-n16"])
    def test_pretrained(self, device, model_name):
        """Pretrained model should load and produce reasonable detections."""
        aliked = ALIKED.from_pretrained(model_name=model_name, device=device)
        inp = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            output = aliked(inp)
        assert len(output) == 1
        assert output[0].n > 0

    def test_gradcheck(self, device):
        """Gradients should flow through ALIKED."""
        aliked = ALIKED(model_name="aliked-t16").to(device, torch.float64)
        inp = torch.rand(1, 3, 32, 32, device=device, dtype=torch.float64, requires_grad=True)
        output = aliked(inp)
        if output[0].n > 0:
            loss = output[0].keypoints.sum() + output[0].descriptors.sum()
            loss.backward()
            assert inp.grad is not None

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
        offset = (torch.randn(B, 2 * K, H, W, device=device) * 0.1).requires_grad_(True)

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
        assert feat_dev.keypoint_scores.device.type == device.type

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
        """gradcheck on the fully-differentiable extract_dense_map sub-graph.

        The full pipeline includes discrete NMS/argmax steps that are not
        differentiable, so gradcheck is run on extract_dense_map which covers
        the backbone, feature pyramid, and score head.  The model is placed
        in eval mode so BatchNorm uses fixed running statistics.
        """
        aliked = ALIKED(model_name="aliked-t16").to(device, torch.float64).eval()
        inp = torch.rand(1, 3, 32, 32, device=device, dtype=torch.float64, requires_grad=True)

        def fn(x: torch.Tensor) -> torch.Tensor:
            fm, sm = aliked.extract_dense_map(x)
            return fm, sm

        assert torch.autograd.gradcheck(fn, [inp], eps=1e-4, atol=1e-3, rtol=1e-3, fast_mode=True)


# ---------------------------------------------------------------------------
# forward_laf tests
# ---------------------------------------------------------------------------


class TestALIKEDForwardLAF(BaseTester):
    """Tests for ALIKED.forward_laf returning kornia LAF-format outputs."""

    def test_smoke(self, device, dtype):
        """forward_laf should return the three expected tensors."""
        aliked = ALIKED(model_name="aliked-t16", max_num_keypoints=32).to(device, dtype)
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        lafs, responses, descs = aliked.forward_laf(inp)
        assert lafs.ndim == 4  # (B, N, 2, 3)
        assert responses.ndim == 3  # (B, N, 1)
        assert descs.ndim == 3  # (B, N, D)

    def test_output_shapes(self, device, dtype):
        """Shapes should be (B, N, 2, 3), (B, N, 1), (B, N, D)."""
        B, H, W = 2, 64, 64
        top_k = 20
        aliked = ALIKED(model_name="aliked-t16", max_num_keypoints=top_k).to(device, dtype)
        inp = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        lafs, responses, descs = aliked.forward_laf(inp)
        N = lafs.shape[1]
        assert lafs.shape == (B, N, 2, 3)
        assert responses.shape == (B, N, 1)
        assert descs.shape[0] == B
        assert descs.shape[1] == N

    def test_laf_center_in_image(self, device, dtype):
        """LAF centres (col 2) should lie within image bounds."""
        H, W = 64, 64
        top_k = 30
        aliked = ALIKED(model_name="aliked-t16", max_num_keypoints=top_k).to(device, dtype)
        inp = torch.rand(1, 3, H, W, device=device, dtype=dtype)
        lafs, responses, _ = aliked.forward_laf(inp)
        # Only check keypoints with positive response (non-padding).
        valid = responses[0, :, 0] > 0
        if valid.any():
            centers = lafs[0, valid, :, 2]  # (N_valid, 2) — [x, y]
            assert (centers[:, 0] >= 0).all()
            assert (centers[:, 0] <= W - 1).all()
            assert (centers[:, 1] >= 0).all()
            assert (centers[:, 1] <= H - 1).all()

    def test_batch_padding(self, device, dtype):
        """All images in the batch should share the same N (padded to max)."""
        B = 3
        aliked = ALIKED(model_name="aliked-t16", max_num_keypoints=25).to(device, dtype)
        inp = torch.rand(B, 3, 64, 64, device=device, dtype=dtype)
        lafs, responses, descs = aliked.forward_laf(inp)
        assert lafs.shape[0] == B
        assert responses.shape[0] == B
        assert descs.shape[0] == B

    def test_laf_with_mask(self, device, dtype):
        """Providing a mask should suppress keypoints in the masked region."""
        H, W = 64, 64
        aliked = ALIKED(model_name="aliked-t16", max_num_keypoints=40).to(device, dtype)
        inp = torch.rand(1, 3, H, W, device=device, dtype=dtype)
        # Suppress the entire image except the bottom-right quarter.
        mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
        mask[:, :, H // 2 :, W // 2 :] = 1.0
        lafs_masked, _, _ = aliked.forward_laf(inp, mask=mask)
        assert lafs_masked.shape[0] == 1

    def test_laf_affine_shape(self, device, dtype):
        """The 2x2 affine part of each LAF should be non-degenerate for valid kpts."""
        top_k = 20
        aliked = ALIKED(model_name="aliked-t16", max_num_keypoints=top_k).to(device, dtype)
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        lafs, responses, _ = aliked.forward_laf(inp)
        valid = responses[0, :, 0] > 0
        if valid.any():
            A = lafs[0, valid, :, :2]  # (N_valid, 2, 2)
            # det(A) should be non-zero (non-degenerate)
            dets = torch.det(A)
            assert (dets.abs() > 1e-6).all()

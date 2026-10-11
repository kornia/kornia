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

import math

import pytest
import torch

from kornia.feature import get_laf_center
from kornia.feature.aliked import ALIKED, ALIKEDFeatures
from kornia.feature.aliked.deform_conv2d import deform_conv2d

from testing.base import BaseTester


def _score_map_with_maxima_chains(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a (1, 1, 64, 96) score map of isolated peaks, chains of maxima 2 px apart and a tied pair.

    Background 0.01. Isolated: 0.9 at (x, y) = (21, 13); 0.6 at (60, 20) and 0.5 at (63, 20), 3 px apart. Row 30:
    0.9 falling to 0.6 in steps of 0.05 at x = 10, 12, ..., 22. Column 70: 0.7 / 0.62 / 0.55 at y = 40 / 42 / 44.
    Row 52: 0.5 at both x = 40 and x = 41, then 0.45 / 0.4 / 0.35 / 0.3 at x = 43 / 45 / 47 / 49.
    """
    s = torch.full((1, 1, 64, 96), 0.01, device=device, dtype=dtype)
    s[0, 0, 13, 21] = 0.9
    s[0, 0, 20, 60], s[0, 0, 20, 63] = 0.6, 0.5
    for i, value in enumerate((0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)):
        s[0, 0, 30, 10 + 2 * i] = value
    s[0, 0, 40, 70], s[0, 0, 42, 70], s[0, 0, 44, 70] = 0.7, 0.62, 0.55
    s[0, 0, 52, 40], s[0, 0, 52, 41] = 0.5, 0.5
    for i, value in enumerate((0.45, 0.4, 0.35, 0.3)):
        s[0, 0, 52, 43 + 2 * i] = value
    return s


def _aliked_with_score_map(score_map: torch.Tensor, nms_radius: int = 2) -> ALIKED:
    """Return an untrained ``aliked-t16`` whose score head yields ``score_map`` after its sigmoid."""
    aliked = ALIKED(model_name="aliked-t16", nms_radius=nms_radius).to(score_map.device, score_map.dtype).eval()
    aliked.score_head.register_forward_hook(lambda module, inputs, output: torch.logit(score_map))
    return aliked


# ---------------------------------------------------------------------------
# deform_conv2d tests
# ---------------------------------------------------------------------------


class TestDeformConv2d(BaseTester):
    """Tests for the pure-PyTorch deform_conv2d implementation."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_zero_offset_matches_regular_conv(self, device, dtype):
        """With all-zero offsets, deform_conv2d should equal regular conv2d."""
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
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

    @staticmethod
    def _torchvision_reference_inputs():
        """Deterministic float64 CPU inputs shared by the hardcoded literals below.

        B=1, C_in=2, H=W=4, C_out=2, kH=kW=3, padding=1, stride=1 -- small enough that the
        (1, 2, 4, 4) = 32-element output fits in a hardcoded literal, per AGENTS.md's guidance to
        prefer a literal plus its generation snippet over a live optional-dependency comparison.
        """
        B, C_in, H, W = 1, 2, 4, 4
        C_out, kH, kW = 2, 3, 3
        x = torch.linspace(-1, 1, B * C_in * H * W, dtype=torch.float64).reshape(B, C_in, H, W)
        weight = torch.linspace(-0.5, 0.5, C_out * C_in * kH * kW, dtype=torch.float64).reshape(C_out, C_in, kH, kW)
        bias = torch.tensor([0.1, -0.2], dtype=torch.float64)
        offset = 0.3 * torch.sin(torch.arange(B * 2 * kH * kW * H * W, dtype=torch.float64)).reshape(B, 18, H, W)
        mask = torch.sigmoid(torch.cos(torch.arange(B * kH * kW * H * W, dtype=torch.float64))).reshape(B, 9, H, W)
        return x, weight, bias, offset, mask

    # Generated once with torchvision (not a kornia dependency, so unavailable in CI -- this is
    # why the test this replaces, `test_matches_torchvision`, never ran there) in a throwaway venv:
    #   uv venv /tmp/tv && uv pip install --python /tmp/tv/bin/python torch torchvision
    #   -> torch==2.14.0, torchvision==0.29.0 (CPU wheels, Python 3.11.0)
    # Standalone snippet for that venv (which has neither kornia nor pytest, so the inputs are inlined
    # rather than imported; keep them identical to `_torchvision_reference_inputs` above):
    #   import torch, torchvision.ops as tvops
    #   torch.set_printoptions(precision=10)
    #   B, C_in, H, W, C_out, kH, kW = 1, 2, 4, 4, 2, 3, 3
    #   x = torch.linspace(-1, 1, B * C_in * H * W, dtype=torch.float64).reshape(B, C_in, H, W)
    #   weight = torch.linspace(-0.5, 0.5, C_out * C_in * kH * kW, dtype=torch.float64).reshape(C_out, C_in, kH, kW)
    #   bias = torch.tensor([0.1, -0.2], dtype=torch.float64)
    #   offset = 0.3 * torch.sin(torch.arange(B * 2 * kH * kW * H * W, dtype=torch.float64)).reshape(B, 18, H, W)
    #   mask = torch.sigmoid(torch.cos(torch.arange(B * kH * kW * H * W, dtype=torch.float64))).reshape(B, 9, H, W)
    #   for use_mask in (False, True):
    #       out = tvops.deform_conv2d(
    #           x, offset, weight, bias=bias, padding=1, stride=1, mask=mask if use_mask else None
    #       )
    #       print(use_mask, out)
    _EXPECTED_NO_MASK = torch.tensor(
        [
            1.3524878443,
            1.9642850189,
            1.4678019606,
            0.8534719288,
            1.4020842470,
            1.9136044871,
            1.8863619178,
            1.2793386143,
            0.8920304159,
            1.2077669486,
            0.9345327788,
            0.4964185813,
            0.2459445647,
            0.0819180146,
            -0.1149317217,
            -0.0837964278,
            -0.4751185738,
            -0.4331255309,
            -0.1060449678,
            0.0392646305,
            0.2247573785,
            0.6240284088,
            0.7780789016,
            0.4968604144,
            0.9902952558,
            1.4242185698,
            1.5622926375,
            1.1953252054,
            0.6699741701,
            1.4466645925,
            1.5290271874,
            0.7818486934,
        ],
        dtype=torch.float64,
    ).reshape(1, 2, 4, 4)
    _EXPECTED_WITH_MASK = torch.tensor(
        [
            0.7464577357,
            1.0163278244,
            0.7994783630,
            0.4874494644,
            0.7544586797,
            1.0291377431,
            1.0361463498,
            0.7093859204,
            0.4840123409,
            0.6248190328,
            0.4886002027,
            0.2803572404,
            0.1663333364,
            0.0952250511,
            -0.0103768928,
            -0.0005014362,
            -0.3440748865,
            -0.3220334021,
            -0.1632313152,
            -0.0853091101,
            0.0239935903,
            0.2204502075,
            0.2789350116,
            0.1407094840,
            0.3924910576,
            0.6243384778,
            0.7212428510,
            0.4743984538,
            0.2333181531,
            0.6454519918,
            0.6817443629,
            0.3121810405,
        ],
        dtype=torch.float64,
    ).reshape(1, 2, 4, 4)

    @pytest.mark.parametrize("use_mask", [False, True])
    def test_matches_torchvision_reference(self, device, dtype, use_mask):
        """Pure-PyTorch implementation should match a hardcoded torchvision-generated literal.

        Replaces a live comparison against ``torchvision.ops.deform_conv2d`` (skipif-guarded on
        torchvision, which no CI environment installs, so it never ran there) with a literal
        generated once offline -- see the comment above ``_EXPECTED_NO_MASK``.
        """
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")

        x64, weight64, bias64, offset64, mask64 = self._torchvision_reference_inputs()
        expected64 = self._EXPECTED_WITH_MASK if use_mask else self._EXPECTED_NO_MASK

        x = x64.to(device=device, dtype=dtype)
        weight = weight64.to(device=device, dtype=dtype)
        bias = bias64.to(device=device, dtype=dtype)
        offset = offset64.to(device=device, dtype=dtype)
        mask = mask64.to(device=device, dtype=dtype) if use_mask else None

        out = deform_conv2d(x, offset, weight, bias=bias, padding=1, stride=1, mask=mask)
        expected = expected64.to(device=device, dtype=dtype)

        if dtype == torch.float64 and device.type == "cpu":
            # The literal is quoted to 10 decimals (its own rounding is <= 5e-11) and kornia's float64
            # CPU result lands within 5e-11 of it, so an absolute bound of 1e-9 still has 20x headroom;
            # a looser bound would only hide float64 arithmetic bugs.
            self.assert_close(out, expected, rtol=0, atol=1e-9)
        else:
            self.assert_close(out, expected)

        # Guard a float32 precision regression against the float64 reference, regardless of the
        # dtype this test instance is parametrized over.
        mask32 = mask64.float() if use_mask else None
        out_f32 = deform_conv2d(
            x64.float(), offset64.float(), weight64.float(), bias=bias64.float(), padding=1, stride=1, mask=mask32
        )
        assert torch.allclose(out_f32.double(), expected64, rtol=0, atol=1e-5)

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
            # A unit vector stored in `dtype` cannot have a norm closer to 1 than the grid allows, and
            # `norm` itself rounds into that same grid, so the deviation saturates at exactly one eps
            # (measured over 2400+ descriptors on torch 2.9.1 and 2.14.0; taking the norm in float32
            # instead gives 0.55 eps, i.e. the descriptors really are unit to the storage limit).
            # atol=1e-4 is unreachable for float16 (eps 9.8e-4) and bfloat16 (eps 7.8e-3); 2 eps keeps
            # a 2x margin there and leaves float32/float64 on the original bound.
            atol = max(1e-4, 2 * torch.finfo(dtype).eps)
            assert torch.allclose(norms, torch.ones_like(norms), atol=atol)

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

    @pytest.mark.parametrize(
        "nms_radius, expected",
        [
            (
                2,
                [
                    [21.0, 13.0],
                    [60.0, 20.0],
                    [63.0, 20.0],
                    [10.75344036, 30.0],
                    [13.36128702, 30.0],
                    [17.36401829, 30.0],
                    [70.0, 40.60892188],
                    [70.0, 42.7110507],
                    [40.45713005, 52.0],
                    [41.07421962, 52.0],
                    [44.4367411, 52.0],
                    [48.18569541, 52.0],
                ],
            ),
            (
                3,
                [
                    [21.0, 13.0],
                    [60.73192275, 20.0],
                    [10.75191131, 30.0],
                    [13.36302377, 30.0],
                    [17.36867735, 30.0],
                    [70.0, 40.5990816],
                    [70.0, 42.7543607],
                    [40.94599394, 52.0],
                    [41.06972181, 52.0],
                    [44.50210413, 52.0],
                    [48.39111553, 52.0],
                ],
            ),
        ],
    )
    def test_nms_keeps_maxima_outside_suppressed_windows_5730(self, device, dtype, nms_radius, expected):
        """ALIKED's NMS is the reference's, which re-admits maxima outside the suppressed windows in two rounds.

        The expected keypoints are the reference DKD's on the same score map (cvg/LightGlue eb42fee
        ``lightglue/aliked.py``; at radius 2 Shiaoming/ALIKED 683d7c6 ``nets/soft_detect.py`` gives the same), float64::

            dkd = DKD(radius=nms_radius, top_k=-1, scores_th=0.2, n_limit=20000)
            keypoints, _, _ = dkd(score_map)
            torch.tensor([95.0, 63.0]) * (keypoints[0] + 1) / 2

        At radius 2, the maxima at x = 14 and 18 on row 30, y = 44 on column 70 and x = 45 and 49 on row 52 are
        outscored only by a neighbour that a kept maximum suppresses (x = 14 only by x = 12, which x = 10
        suppresses); one strict pass drops them. The reference re-admits such maxima for two rounds, so x = 22 on
        row 30, whose stronger neighbour x = 20 only the second-round x = 18 suppresses, stays suppressed. Both
        tied maxima on row 52 are kept. At radius 3, (63, 20) lies in the window of (60, 20) and is suppressed.
        """
        score_map = _score_map_with_maxima_chains(device, dtype)
        aliked = _aliked_with_score_map(score_map, nms_radius)
        with torch.no_grad():
            features = aliked(torch.rand(1, 3, 64, 96, device=device, dtype=dtype))
        # float16 rounds the soft-argmax position to within 1.5e-3 relative, above its 1e-3 default tolerance.
        self.assert_close(
            features[0].keypoints,
            torch.tensor(expected, device=device, dtype=dtype),
            low_tolerance=dtype == torch.float16,
        )

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

        assert torch.autograd.gradcheck(fn, [inp], eps=1e-4, atol=1e-3, rtol=1e-3, fast_mode=True, nondet_tol=1e-3)


# ---------------------------------------------------------------------------
# forward_laf tests
# ---------------------------------------------------------------------------


class TestALIKEDForwardLAF(BaseTester):
    """Tests for ALIKED.forward_laf returning kornia LAF-format outputs."""

    def test_smoke(self, device, dtype):
        """forward_laf should return the three expected tensors."""
        aliked = ALIKED(model_name="aliked-t16", max_num_keypoints=32).to(device, dtype)
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        lafs, responses, descs = aliked.forward_laf(inp, compute_affine=False)
        assert lafs.ndim == 4  # (B, N, 2, 3)
        assert responses.ndim == 3  # (B, N, 1)
        assert descs.ndim == 3  # (B, N, D)

    def test_output_shapes(self, device, dtype):
        """Shapes should be (B, N, 2, 3), (B, N, 1), (B, N, D)."""
        B, H, W = 2, 64, 64
        top_k = 20
        aliked = ALIKED(model_name="aliked-t16", max_num_keypoints=top_k).to(device, dtype)
        inp = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        lafs, responses, descs = aliked.forward_laf(inp, compute_affine=False)
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
        lafs, responses, _ = aliked.forward_laf(inp, compute_affine=False)
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
        lafs, responses, descs = aliked.forward_laf(inp, compute_affine=False)
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
        lafs_masked, _, _ = aliked.forward_laf(inp, mask=mask, compute_affine=False)
        assert lafs_masked.shape[0] == 1

    def test_laf_affine_shape(self, device, dtype):
        """The 2x2 block of each LAF is a positive multiple of the identity: upright and not mirrored."""
        top_k = 20
        aliked = ALIKED(model_name="aliked-t16", max_num_keypoints=top_k).to(device, dtype)
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        lafs, responses, _ = aliked.forward_laf(inp, compute_affine=True)
        valid = responses[0, :, 0] > 0
        assert valid.any()
        A = lafs[0, valid, :, :2]  # (N_valid, 2, 2)
        assert (A[:, 0, 0] > 0).all()
        self.assert_close(A, A[:, :1, :1] * torch.eye(2, device=device, dtype=dtype))

    def test_frame_scale_is_descriptor_sampling_radius_5730(self, device, dtype):
        """Each LAF is ``s * I``, ``s`` the RMS distance of the descriptor's sampling positions from the keypoint.

        The offset head is set so that every keypoint's descriptor samples 8 positions at (3, 4) px and 8 at
        (-6, 8) px from it: RMS distance ``sqrt((25 + 100) / 2)``, against a mean distance of 7.5, a largest
        distance of 10 and a spread about the pattern's centroid of ``sqrt(24.25)``. The LAF centres are the
        keypoints of :meth:`ALIKED.forward`; ``compute_affine=False`` gives identity blocks.
        """
        score_map = _score_map_with_maxima_chains(device, dtype)
        aliked = _aliked_with_score_map(score_map)
        offset_head = aliked.desc_head.offset_conv[2]
        n_pos = aliked.desc_head.n_pos
        half = n_pos // 2
        # bias layout (x offsets of every sampling position, then y offsets)
        offsets_xy = torch.tensor([3.0] * half + [-6.0] * half + [4.0] * half + [8.0] * half)
        with torch.no_grad():
            offset_head.weight.zero_()
            offset_head.bias.copy_(offsets_xy)
            img = torch.rand(1, 3, 64, 96, device=device, dtype=dtype)
            lafs, _, _ = aliked.forward_laf(img)
            lafs_unit, _, _ = aliked.forward_laf(img, compute_affine=False)
            keypoints = aliked(img)[0].keypoints
        n = keypoints.shape[0]
        assert n > 0
        assert lafs.shape == (1, n, 2, 3)
        eye = torch.eye(2, device=device, dtype=dtype).expand(1, n, 2, 2)
        self.assert_close(lafs[..., :2], math.sqrt(62.5) * eye)
        self.assert_close(get_laf_center(lafs)[0], keypoints)
        self.assert_close(lafs_unit[..., :2], eye)
        self.assert_close(lafs_unit[..., 2], lafs[..., 2])

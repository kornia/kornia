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

from kornia.feature.affine_shape import LAFAffineShapeEstimator, LAFAffNetShapeEstimator, PatchAffineShapeEstimator
from kornia.feature.laf import make_upright
from kornia.filters import get_gaussian_kernel2d

from testing.base import BaseTester


class DegenerateShape(torch.nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        shape = patches.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1) * 0
        return torch.cat([shape, shape, torch.ones_like(shape)], dim=-1)


class OverflowShape(torch.nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        one = patches.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1) * 0 + 1
        tiny = one * 1e-38
        return torch.cat([tiny, one, tiny], dim=-1)


class SingularAffNetOutput(torch.nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        zero = patches.mean(dim=(-3, -2, -1), keepdim=True) * 0
        return torch.cat([zero - 1, zero, zero - 1], dim=1)


class TestPatchAffineShapeEstimator(BaseTester):
    def test_zero_patch_uses_circular_shape(self, device, dtype):
        patch = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        out = PatchAffineShapeEstimator(32).to(device, dtype)(patch)
        expected = torch.tensor([[[1.0, 0.0, 1.0]]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_half_precision_retains_anisotropic_shape(self, device, dtype):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision regression test")
        patch = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        patch[:, :, 15:17, 9:23] = 1
        expected = PatchAffineShapeEstimator(32).to(device)(patch.float()).to(dtype)
        patch.requires_grad_()

        out = PatchAffineShapeEstimator(32).to(device, dtype)(patch)

        assert out.dtype == dtype
        self.assert_close(out, expected)
        out.sum().backward()
        assert patch.grad is not None
        assert torch.isfinite(patch.grad).all()

    def test_half_precision_uses_float32_weighting(self, device, dtype):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision regression test")
        patch = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        patch[:, :, 2:4, 14:16] = 1
        reference = PatchAffineShapeEstimator(32).to(device)
        sigma = 32.0 / math.sqrt(2.0)
        reference.weighting = get_gaussian_kernel2d((32, 32), (sigma, sigma), True, device=device, dtype=torch.float32)
        expected = reference(patch.float()).to(dtype)

        out = PatchAffineShapeEstimator(32).to(device, dtype)(patch)

        assert torch.equal(out, expected)

    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = PatchAffineShapeEstimator(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([1, 1, 3])

    def test_shape_batch(self, device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        ori = PatchAffineShapeEstimator(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([2, 1, 3])

    def test_print(self, device):
        sift = PatchAffineShapeEstimator(32)
        sift.__repr__()

    def test_toy(self, device):
        aff = PatchAffineShapeEstimator(19).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, 5:-5, 1:-1] = 1
        abc = aff(inp)
        expected = torch.tensor([[[0.4146, 0.0000, 1.0000]]], device=device)
        self.assert_close(abc, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 13, 13
        ori = PatchAffineShapeEstimator(width).to(device)
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(ori, (patches,), nondet_tol=1e-4)


class TestAffineShapeKernelBuffer(BaseTester):
    """`weighting` must be a real buffer so `.to()` moves it, and `forward` must not rebind it (#4069)."""

    def test_to_moves_the_kernel(self, device):
        """`.to()` must carry the kernel along with the parameters.

        float16 rather than the default float32, or the dtype assertion would hold
        vacuously; float16 also works on MPS, where float64 is unavailable.
        """
        mod = PatchAffineShapeEstimator(19).to(device, torch.float16)
        assert mod.weighting.dtype == torch.float16
        assert mod.weighting.device == torch.empty(0, device=device).device

    def test_kernel_stays_out_of_state_dict(self, device):
        """Registered non-persistent, so existing checkpoints keep loading with strict=True."""
        assert "weighting" not in PatchAffineShapeEstimator(19).state_dict()

    def test_forward_does_not_mutate_the_module(self, device):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        mod = PatchAffineShapeEstimator(19)
        before = (mod.weighting.dtype, mod.weighting.device)
        mod(torch.rand(2, 1, 19, 19, dtype=torch.float64))
        assert (mod.weighting.dtype, mod.weighting.device) == before


class TestLAFAffineShapeEstimator(BaseTester):
    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffineShapeEstimator().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = LAFAffineShapeEstimator().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        sift = LAFAffineShapeEstimator()
        sift.__repr__()

    def test_toy(self, device, dtype):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=False).to(device, dtype)
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[35.078, 0.0, 16.0], [0.0, 11.403, 16.0]]]], device=device, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            # Use the repository's dtype-specific tolerances for the newly supported half-precision path.
            self.assert_close(new_laf, expected)
        else:
            self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_toy_preserve(self, device, dtype):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=True).to(device, dtype)
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[0.0, 20.0, 16.0], [-20.0, 0.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[0.0, 35.078, 16.0], [-11.403, 0, 16.0]]]], device=device, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            # Orientation recovery adds small absolute noise to entries whose ideal value is zero.
            self.assert_close(new_laf, expected, atol=2e-2, rtol=1e-3 if dtype == torch.float16 else 7.8e-3)
        else:
            self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_toy_not_preserve(self, device):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=False).to(device)
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[0.0, 20.0, 16.0], [-20.0, 0.0, 16.0]]]], device=device)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[35.078, 0, 16.0], [0, 11.403, 16.0]]]], device=device)
        self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_degenerate_ellipse_falls_back_to_input_laf_float16(self, device, dtype):
        if dtype != torch.float16:
            pytest.skip("float16 regression test")
        # The default patch estimator now catches this nearly one-dimensional patch at the source.
        # Keep the end-to-end assertion to pin the public finite-value and finite-gradient boundary.
        if device.type == "mps":
            pytest.skip("MPS autocast changes the effective dtype")
        y = torch.linspace(0, 1, 32, device=device, dtype=dtype).view(32, 1).expand(32, 32)
        x = torch.linspace(0, 1e-4, 32, device=device, dtype=dtype).view(1, 32).expand(32, 32)
        img = (y + x).view(1, 1, 32, 32).clone().requires_grad_()
        laf = torch.tensor([[[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0]]]], device=device, dtype=dtype, requires_grad=True)
        out = LAFAffineShapeEstimator(32).to(device, dtype)(laf, img)
        assert torch.isfinite(out).all()
        self.assert_close(out, laf)
        out.sum().backward()
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()
        assert laf.grad is not None
        assert torch.isfinite(laf.grad).all()

    @pytest.mark.parametrize("preserve_orientation", [False, True])
    def test_zero_scale_input_has_finite_fallback(self, device, dtype, preserve_orientation):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision regression test")
        if device.type == "mps":
            pytest.skip("MPS autocast changes the effective dtype")
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype, requires_grad=True)
        laf = torch.tensor([[[[0.0, 0.0, 16.0], [0.0, 0.0, 16.0]]]], device=device, dtype=dtype, requires_grad=True)
        aff = LAFAffineShapeEstimator(32, DegenerateShape(), preserve_orientation=preserve_orientation).to(
            device, dtype
        )

        out = aff(laf, img)

        assert torch.isfinite(out).all()
        self.assert_close(out, laf)
        out.sum().backward()
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()
        assert laf.grad is not None
        assert torch.isfinite(laf.grad).all()

    def test_degenerate_ellipse_boundary_has_finite_backward(self, device):
        dtype = torch.float32
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype, requires_grad=True)
        laf = torch.tensor([[[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0]]]], device=device, dtype=dtype, requires_grad=True)
        out = LAFAffineShapeEstimator(32, DegenerateShape(), preserve_orientation=True).to(device, dtype)(laf, img)
        assert torch.isfinite(out).all()
        self.assert_close(out, laf)
        out.sum().backward()
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()
        assert laf.grad is not None
        assert torch.isfinite(laf.grad).all()

    def test_non_positive_definite_shape_falls_back_with_finite_backward(self, device):
        dtype = torch.float32
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype, requires_grad=True)
        laf = torch.tensor([[[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0]]]], device=device, dtype=dtype, requires_grad=True)
        out = LAFAffineShapeEstimator(32, OverflowShape(), preserve_orientation=True).to(device, dtype)(laf, img)
        assert torch.isfinite(out).all()
        self.assert_close(out, laf)
        out.sum().backward()
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()
        assert laf.grad is not None
        assert torch.isfinite(laf.grad).all()

    def test_degenerate_ellipse_fallback_respects_upright_contract(self, device):
        dtype = torch.float32
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        laf = torch.tensor([[[[0.0, 8.0, 16.0], [-8.0, 0.0, 16.0]]]], device=device, dtype=dtype)
        out = LAFAffineShapeEstimator(32, DegenerateShape(), preserve_orientation=False).to(device, dtype)(laf, img)
        self.assert_close(out, make_upright(laf))

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 40, 40
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        laf = torch.tensor([[[[5.0, 0.0, 26.0], [0.0, 5.0, 26.0]]]], device=device, dtype=torch.float64)
        self.gradcheck(
            LAFAffineShapeEstimator(11).to(device),
            (laf, patches),
            rtol=1e-3,
            atol=1e-3,
            nondet_tol=1e-4,
        )


class TestLAFAffNetShapeEstimator(BaseTester):
    def test_singular_prediction_falls_back_upright(self, device, dtype):
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        laf = torch.tensor([[[[0.0, 8.0, 16.0], [-8.0, 0.0, 16.0]]]], device=device, dtype=dtype)
        aff = LAFAffNetShapeEstimator(preserve_orientation=False).to(device, dtype)
        aff.features = SingularAffNetOutput()
        out = aff(laf, img)
        self.assert_close(out, make_upright(laf))

    @pytest.mark.parametrize("preserve_orientation", [False, True])
    def test_zero_scale_input_has_finite_fallback(self, device, dtype, preserve_orientation):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision regression test")
        if device.type == "mps":
            pytest.skip("MPS autocast changes the effective dtype")
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype, requires_grad=True)
        laf = torch.tensor([[[[0.0, 0.0, 16.0], [0.0, 0.0, 16.0]]]], device=device, dtype=dtype, requires_grad=True)
        aff = LAFAffNetShapeEstimator(pretrained=False, preserve_orientation=preserve_orientation).to(device, dtype)
        aff.features = SingularAffNetOutput()

        out = aff(laf, img)

        assert torch.isfinite(out).all()
        self.assert_close(out, laf)
        out.sum().backward()
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()
        assert laf.grad is not None
        assert torch.isfinite(laf.grad).all()

    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator(False).to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    @pytest.mark.slow
    def test_pretrained(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator(True).to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 5, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator().to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        sift = LAFAffNetShapeEstimator()
        sift.__repr__()

    def test_toy(self, device, dtype):
        aff = LAFAffNetShapeEstimator(True).to(device, dtype).eval()
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[33.2073, 0.0, 16.0], [-1.3766, 12.0456, 16.0]]]], device=device, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            # AffNet's convolutions carry reduced-precision noise proportional to the LAF's own scale
            # (~33 px), not to each entry's magnitude, so the small a21 entry (-1.38) cannot be held to a
            # relative bound: it misses by 0.024 (float16) and 0.031 (bfloat16), bitwise identical on
            # torch 2.9.1 and 2.14.0. The 1e-1 bound is ~3 float16 ULP at that scale, so it is set by the
            # dtype rather than by the observed miss and does not need re-tuning per platform. It still
            # discriminates: a genuine shape regression moves these entries by O(1).
            self.assert_close(new_laf, expected, atol=1e-1, rtol=1e-3 if dtype == torch.float16 else 7.8e-3)
        else:
            atol = 5e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-4
            self.assert_close(new_laf, expected, atol=atol, rtol=1e-4)

    @pytest.mark.slow
    def test_gradcheck(self, device):
        torch.manual_seed(0)
        batch_size, channels, height, width = 1, 1, 35, 35
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        laf = torch.tensor([[[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0]]]], device=device, dtype=torch.float64)
        self.gradcheck(
            LAFAffNetShapeEstimator(True).to(device, dtype=patches.dtype),
            (laf, patches),
            requires_grad=[False, True],
            rtol=1e-3,
            atol=1e-3,
            nondet_tol=1e-3,
        )

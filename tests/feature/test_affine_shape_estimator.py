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
from kornia.feature.laf import (
    get_laf_orientation,
    get_laf_scale,
    laf_from_center_scale_ori,
    make_upright,
)
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


class NearlySingularShape(torch.nn.Module):
    # In float32, a * c - b * b is positive (3e-8) but a - (b / sqrt(c)) ** 2, the square of the first diagonal
    # entry of the inverse LAF that ellipse_to_laf forms, rounds to 0. c is a perfect square, so sqrt(c) is exact on
    # every backend and only the correctly rounded multiplications and division decide where the two paths land.
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        zero = patches.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1) * 0
        return torch.cat([zero + 0.75, zero + 0.6495190262794495, zero + 0.5625], dim=-1)


class SingularAffNetOutput(torch.nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        zero = patches.mean(dim=(-3, -2, -1), keepdim=True) * 0
        return torch.cat([zero - 1, zero, zero - 1], dim=1)


class ConstantShape(torch.nn.Module):
    def __init__(self, abc: tuple[float, float, float]) -> None:
        super().__init__()
        self.abc = abc

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        zero = patches.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1) * 0
        return torch.cat([zero + v for v in self.abc], dim=-1)


class ConstantAffNetOutput(torch.nn.Module):
    def __init__(self, xy: tuple[float, float, float]) -> None:
        super().__init__()
        self.xy = xy

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        zero = patches.mean(dim=(-3, -2, -1), keepdim=True) * 0
        return torch.cat([zero + v for v in self.xy], dim=1)


# A LAF [[14, 6], [-3, 17]] is neither upright nor isotropic; its sampling frame make_upright(laf) does not commute with
# the lower-triangular shapes below, so composing in the wrong order changes the region.
_SKEWED_SHAPE = [[14.0, 6.0], [-3.0, 17.0]]

# Entry tolerances of a unit-determinant ellipse matrix computed from a LAF of each dtype.
_ELLIPSE_ATOL = {torch.float64: 1e-6, torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 5e-2}

# A LAF of scale ~292 px: products of two of its entries (det 300 * 280 + 40 * 25 = 85000) exceed float16's 65504.
_LARGE_SHAPE = [[300.0, 40.0], [-25.0, 280.0]]

# Relative tolerances of the scale of an output LAF whose entries are rounded to each dtype.
_SCALE_RTOL = {torch.float64: 1e-6, torch.float32: 1e-5, torch.float16: 4e-3, torch.bfloat16: 3e-2}


def _gaussian_blob(
    h: int, w: int, center: tuple[float, float], axes: tuple[float, float], angle_deg: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a Gaussian blob image and its covariance scaled to unit determinant (float64, CPU).

    The blob's standard deviations are ``axes``, the first along the direction ``angle_deg`` (x right, y down).
    """
    t = math.radians(angle_deg)
    rot = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=torch.float64)
    cov = rot @ torch.diag(torch.tensor(axes, dtype=torch.float64).square()) @ rot.T
    inv = torch.linalg.inv(cov)
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    dx, dy = xx - center[0], yy - center[1]
    img = torch.exp(-0.5 * (inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy))
    return img.view(1, 1, h, w).to(device, dtype), cov / cov.det().sqrt()


def _unit_ellipse(laf: torch.Tensor) -> torch.Tensor:
    """Return the region's ellipse matrix ``A @ A.T`` of the first LAF, scaled to unit determinant (float64, CPU)."""
    a = laf[0, 0, :, :2].cpu().double()
    m = a @ a.T
    return m / m.det().sqrt()


def _laf_with_shape(shape, center: tuple[float, float], device, dtype) -> torch.Tensor:
    return torch.tensor([[[shape[0] + [center[0]], shape[1] + [center[1]]]]], device=device, dtype=dtype)


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
        # The bar (rows 15-16, columns 9-22) is centred on pixel (15.5, 15.5), where the LAF sits.
        laf = torch.tensor([[[[20.0, 0.0, 15.5], [0.0, 20.0, 15.5]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[36.6428, 0.0, 15.5], [0.0, 10.9162, 15.5]]]], device=device, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            # Use the repository's dtype-specific tolerances for the newly supported half-precision path.
            self.assert_close(new_laf, expected)
        else:
            self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_toy_preserve(self, device, dtype):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=True).to(device, dtype)
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        # The bar (rows 15-16, columns 9-22) is centred on pixel (15.5, 15.5), where the LAF sits.
        laf = torch.tensor([[[[0.0, 20.0, 15.5], [-20.0, 0.0, 15.5]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[0.0, 36.6428, 15.5], [-10.9162, 0, 15.5]]]], device=device, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            # Orientation recovery adds small absolute noise to entries whose ideal value is zero.
            self.assert_close(new_laf, expected, atol=2e-2, rtol=1e-3 if dtype == torch.float16 else 7.8e-3)
        else:
            self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_toy_not_preserve(self, device):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=False).to(device)
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        # The bar (rows 15-16, columns 9-22) is centred on pixel (15.5, 15.5), where the LAF sits.
        laf = torch.tensor([[[[0.0, 20.0, 15.5], [-20.0, 0.0, 15.5]]]], device=device)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[36.6428, 0, 15.5], [0, 10.9162, 15.5]]]], device=device)
        self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        ("blob_axes", "input_shape"),
        [
            # An isotropic blob seen through an anisotropic frame: the patch shows the inverse anisotropy.
            ((4.0, 4.0), [[24.0, 0.0], [0.0, 12.0]]),
            # The input frame matches the blob, so the patch is isotropic and the input shape is kept.
            ((6.0, 3.0), "matched"),
            ((6.0, 3.0), _SKEWED_SHAPE),
        ],
    )
    def test_estimate_is_composed_with_the_input_shape_5733(self, device, dtype, blob_axes, input_shape):
        img, blob_ellipse = _gaussian_blob(64, 80, (41.0, 30.0), blob_axes, 25.0, device, dtype)
        if input_shape == "matched":
            input_shape = (16.0 * torch.linalg.cholesky(blob_ellipse)).tolist()
        laf = _laf_with_shape(input_shape, (41.0, 30.0), device, dtype)
        out = LAFAffineShapeEstimator(32).to(device, dtype)(laf, img)
        # Measured: at most 0.015 from the blob's ellipse in every dtype; a patch estimate written over the input
        # shape is 0.45 or more away.
        self.assert_close(_unit_ellipse(out), blob_ellipse, rtol=0.0, atol=0.05)
        self.assert_close(out[..., 2], laf[..., 2])

    def test_repeated_application_converges_5733(self, device, dtype):
        img, blob_ellipse = _gaussian_blob(64, 80, (41.0, 30.0), (6.0, 3.0), 25.0, device, dtype)
        laf = laf_from_center_scale_ori(
            torch.tensor([[[41.0, 30.0]]], device=device, dtype=dtype),
            torch.full((1, 1, 1, 1), 16.0, device=device, dtype=dtype),
            torch.full((1, 1, 1), 40.0, device=device, dtype=dtype),
        )
        est = LAFAffineShapeEstimator(32).to(device, dtype)
        outs = []
        for _ in range(4):
            laf = est(laf, img)
            outs.append(laf)
            self.assert_close(_unit_ellipse(laf), blob_ellipse, rtol=0.0, atol=0.05)
        self.assert_close(outs[3], outs[2], rtol=0.0, atol=0.05)

    @pytest.mark.parametrize("preserve_orientation", [False, True])
    def test_patch_shape_maps_through_the_sampling_frame_5733(self, device, dtype, preserve_orientation):
        # The patch is sampled through make_upright(laf), so a shape estimated in patch coordinates is the image region
        # frame @ shape, whose ellipse matrix is frame @ inverse(E) @ frame.T for the patch ellipse E.
        abc = (1.0, 0.3, 0.5)
        img = torch.rand(1, 1, 48, 64, device=device, dtype=dtype)
        laf = _laf_with_shape(_SKEWED_SHAPE, (30.0, 22.0), device, dtype)
        aff = LAFAffineShapeEstimator(32, ConstantShape(abc), preserve_orientation=preserve_orientation)
        out = aff.to(device, dtype)(laf, img)
        frame = make_upright(laf)[0, 0, :, :2].cpu().double()
        patch_ellipse = torch.tensor([[abc[0], abc[1]], [abc[1], abc[2]]], dtype=torch.float64)
        expected = frame @ torch.linalg.inv(patch_ellipse) @ frame.T
        self.assert_close(_unit_ellipse(out), expected / expected.det().sqrt(), rtol=0.0, atol=_ELLIPSE_ATOL[dtype])
        self.assert_close(out[..., 2], laf[..., 2])
        self.assert_close(get_laf_scale(out), get_laf_scale(laf))
        if preserve_orientation:
            self.assert_close(get_laf_orientation(out), get_laf_orientation(laf))
        else:
            assert (out[..., 0, 1] == 0).all()

    @pytest.mark.parametrize(
        ("preserve_orientation", "expected_shape"),
        [
            (True, [[16.15094, 13.55225], [-2.50039, 13.75239]]),
            (False, [[21.08356, 0.0], [6.92445, 12.14216]]),
        ],
    )
    def test_isotropic_input_gets_the_patch_estimate_5733(self, device, dtype, preserve_orientation, expected_shape):
        # An isotropic input is sampled through a scaled identity frame (up to rounding), so its result is the patch
        # estimate at the input's scale. Values from torch 2.14, CPU float64.
        img, _ = _gaussian_blob(64, 80, (41.0, 30.0), (6.0, 3.0), 25.0, device, dtype)
        laf = laf_from_center_scale_ori(
            torch.tensor([[[41.0, 30.0]]], device=device, dtype=dtype),
            torch.full((1, 1, 1, 1), 16.0, device=device, dtype=dtype),
            torch.full((1, 1, 1), 40.0, device=device, dtype=dtype),
        )
        out = LAFAffineShapeEstimator(32, preserve_orientation=preserve_orientation).to(device, dtype)(laf, img)
        expected = _laf_with_shape(expected_shape, (41.0, 30.0), device, dtype)
        if dtype in (torch.float16, torch.bfloat16):
            # The half-precision patch differs from the float64 one by rounding: up to one ULP of the output, which is
            # 0.125 in bfloat16 between 16 and 32.
            self.assert_close(out, expected, rtol=0.0, atol=2e-2 if dtype == torch.float16 else 1.3e-1)
        else:
            self.assert_close(out, expected, rtol=1e-4, atol=1e-4)

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

    def test_nearly_singular_shape_falls_back_with_finite_backward(self, device):
        dtype = torch.float32
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype, requires_grad=True)
        laf = torch.tensor([[[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0]]]], device=device, dtype=dtype, requires_grad=True)
        a, b, c = NearlySingularShape()(img).unbind(-1)
        assert a * c - b * b > 0
        assert a - (b / c.sqrt()).square() <= 0
        out = LAFAffineShapeEstimator(32, NearlySingularShape(), preserve_orientation=True).to(device, dtype)(laf, img)
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
        # Off the pixel lattice: a sample exactly on a pixel centre sits on a kink of bilinear interpolation.
        laf = torch.tensor([[[[5.0, 0.0, 26.3], [0.0, 5.0, 26.3]]]], device=device, dtype=torch.float64)
        self.gradcheck(
            LAFAffineShapeEstimator(11).to(device),
            (laf, patches),
            rtol=1e-3,
            atol=1e-3,
            nondet_tol=1e-4,
        )

    @pytest.mark.parametrize("preserve_orientation", [False, True])
    def test_large_half_precision_laf_is_estimated(self, device, dtype, preserve_orientation):
        # Products of two entries of this LAF overflow float16 (det 85000 > 65504). It used to be judged invalid in
        # float16, and the estimate was dropped: a zero-shape frame without preserve_orientation, the input with it.
        abc = (1.0, 0.3, 0.5)
        img = torch.rand(1, 1, 96, 128, device=device, dtype=dtype)
        laf = _laf_with_shape(_LARGE_SHAPE, (70.0, 41.0), device, dtype)
        aff = LAFAffineShapeEstimator(32, ConstantShape(abc), preserve_orientation=preserve_orientation)
        out = aff.to(device, dtype)(laf, img)
        assert out.dtype == dtype
        frame = make_upright(laf.cpu().double())[0, 0, :, :2]
        patch_ellipse = torch.tensor([[abc[0], abc[1]], [abc[1], abc[2]]], dtype=torch.float64)
        expected = frame @ torch.linalg.inv(patch_ellipse) @ frame.T
        self.assert_close(_unit_ellipse(out), expected / expected.det().sqrt(), rtol=0.0, atol=_ELLIPSE_ATOL[dtype])
        self.assert_close(out[..., 2], laf[..., 2])
        actual_scale, expected_scale = get_laf_scale(out.cpu().double()), get_laf_scale(laf.cpu().double())
        self.assert_close(actual_scale, expected_scale, rtol=_SCALE_RTOL[dtype], atol=0.0)


class TestLAFAffNetShapeEstimator(BaseTester):
    def test_singular_prediction_falls_back_upright(self, device, dtype):
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        laf = torch.tensor([[[[0.0, 8.0, 16.0], [-8.0, 0.0, 16.0]]]], device=device, dtype=dtype)
        aff = LAFAffNetShapeEstimator(preserve_orientation=False).to(device, dtype)
        aff.features = SingularAffNetOutput()
        out = aff(laf, img)
        self.assert_close(out, make_upright(laf))

    @pytest.mark.parametrize("preserve_orientation", [False, True])
    @pytest.mark.parametrize("xy", [(0.0, 0.0, 0.0), (0.2, -0.3, -0.4)])
    def test_prediction_maps_through_the_sampling_frame_5733(self, device, dtype, xy, preserve_orientation):
        # AffNet predicts P = [[1 + x0, 0], [x1, 1 + x2]] in the coordinates of a patch sampled through
        # make_upright(laf), so the image region is frame @ P: an identity prediction keeps the input shape.
        img = torch.rand(1, 1, 48, 64, device=device, dtype=dtype)
        laf = _laf_with_shape(_SKEWED_SHAPE, (30.0, 22.0), device, dtype)
        aff = LAFAffNetShapeEstimator(preserve_orientation=preserve_orientation).to(device, dtype)
        aff.features = ConstantAffNetOutput(xy)
        out = aff(laf, img)
        frame = make_upright(laf)[0, 0, :, :2].cpu().double()
        prediction = torch.tensor([[1.0 + xy[0], 0.0], [xy[1], 1.0 + xy[2]]], dtype=torch.float64)
        expected = frame @ prediction @ prediction.T @ frame.T
        self.assert_close(_unit_ellipse(out), expected / expected.det().sqrt(), rtol=0.0, atol=_ELLIPSE_ATOL[dtype])
        self.assert_close(out[..., 2], laf[..., 2])
        self.assert_close(get_laf_scale(out), get_laf_scale(laf))
        if preserve_orientation:
            self.assert_close(get_laf_orientation(out), get_laf_orientation(laf))
        else:
            assert (out[..., 0, 1] == 0).all()

    @pytest.mark.parametrize(
        ("blob_axes", "input_shape"),
        [((4.0, 4.0), [[24.0, 0.0], [0.0, 12.0]]), ((6.0, 3.0), "matched")],
    )
    def test_pretrained_estimate_is_composed_with_the_input_shape_5733(self, device, dtype, blob_axes, input_shape):
        img, blob_ellipse = _gaussian_blob(64, 80, (41.0, 30.0), blob_axes, 25.0, device, dtype)
        if input_shape == "matched":
            input_shape = (16.0 * torch.linalg.cholesky(blob_ellipse)).tolist()
        laf = _laf_with_shape(input_shape, (41.0, 30.0), device, dtype)
        out = LAFAffNetShapeEstimator(True).to(device, dtype).eval()(laf, img)
        # AffNet is learned, not a fixed-point estimator: measured at most 0.062 from the blob's ellipse in every
        # dtype; a prediction written over the input shape is 0.76 or more away.
        self.assert_close(_unit_ellipse(out), blob_ellipse, rtol=0.0, atol=0.15)

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
        # The bar (rows 15-16, columns 9-22) is centred on pixel (15.5, 15.5), where the LAF sits.
        laf = torch.tensor([[[[20.0, 0.0, 15.5], [0.0, 20.0, 15.5]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[40.87576, 0.0, 15.5], [-0.38244, 9.78575, 15.5]]]], device=device, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            # AffNet's convolutions carry reduced-precision noise proportional to the LAF's own scale
            # (~41 px), not to each entry's magnitude, so the small a21 entry (-0.38) cannot be held to a
            # relative bound: the largest miss is 0.032 in float16 and 0.149 in bfloat16 (torch 2.14.0).
            # The bounds are set by the dtype, not by the observed miss: ~3 float16 ULP (1e-1) and one
            # bfloat16 ULP (0.25) at that scale. They still discriminate: a genuine shape regression
            # moves these entries by O(1).
            atol = 1e-1 if dtype == torch.float16 else 0.25
            self.assert_close(new_laf, expected, atol=atol, rtol=1e-3 if dtype == torch.float16 else 7.8e-3)
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

    @pytest.mark.parametrize("preserve_orientation", [False, True])
    def test_large_half_precision_laf_is_estimated(self, device, dtype, preserve_orientation):
        # Products of two entries of this LAF overflow float16 (det 85000 > 65504). It used to be judged invalid in
        # float16, and the prediction was dropped: a zero-shape frame without preserve_orientation, the input with it.
        xy = (0.2, -0.3, -0.4)
        img = torch.rand(1, 1, 96, 128, device=device, dtype=dtype)
        laf = _laf_with_shape(_LARGE_SHAPE, (70.0, 41.0), device, dtype)
        aff = LAFAffNetShapeEstimator(preserve_orientation=preserve_orientation).to(device, dtype)
        aff.features = ConstantAffNetOutput(xy)
        out = aff(laf, img)
        assert out.dtype == dtype
        frame = make_upright(laf.cpu().double())[0, 0, :, :2]
        prediction = torch.tensor([[1.0 + xy[0], 0.0], [xy[1], 1.0 + xy[2]]], dtype=torch.float64)
        expected = frame @ prediction @ prediction.T @ frame.T
        self.assert_close(_unit_ellipse(out), expected / expected.det().sqrt(), rtol=0.0, atol=_ELLIPSE_ATOL[dtype])
        self.assert_close(out[..., 2], laf[..., 2])
        actual_scale, expected_scale = get_laf_scale(out.cpu().double()), get_laf_scale(laf.cpu().double())
        self.assert_close(actual_scale, expected_scale, rtol=_SCALE_RTOL[dtype], atol=0.0)


class TestConventionsAffineShape(BaseTester):
    def test_convention_patch_affine_shape_is_the_normalised_gradient_second_moment(self, device, dtype):
        # The output (a, b, c) is (<gx^2>, <gx gy>, <gy^2>) of the Gaussian-weighted patch gradients (x right, y down),
        # divided by its largest entry. On the ramp I = x / 4 - y / 8 the gradient is (1/4, -1/8), so (a, b, c) is
        # (1, -1/2, 1/4) up to the replicate-padded border rows and columns: (1.0, -0.4903902, 0.25), CPU float64.
        # The ramp is exact in every dtype.
        yy, xx = torch.meshgrid(
            torch.arange(19, dtype=torch.float64), torch.arange(19, dtype=torch.float64), indexing="ij"
        )
        patch = (0.25 * xx - 0.125 * yy).view(1, 1, 19, 19).to(device, dtype)
        estimator = PatchAffineShapeEstimator(19).to(device, dtype)
        expected = torch.tensor([[[1.0, -0.4903902, 0.25]]], device=device, dtype=dtype)
        self.assert_close(estimator(patch), expected)
        # relabelling: the transposed patch swaps a and c and keeps b
        self.assert_close(estimator(patch.transpose(-1, -2)), expected.flip(-1))

    def test_wart_patch_affine_shape_counts_a_negative_b_as_below_eps_5824(self, device, dtype):
        # The degeneracy test counts the entries below eps with their sign, so a negative b always counts (#5824). The
        # ramp 0.25 x + 1e-3 y gives the thin ellipse (1, +0.0039, 1.6e-5) (CPU float64), while its vertical mirror
        # 0.25 x - 1e-3 y, whose b is negative, gives the circle (1, 0, 1) instead of (1, -0.0039, 1.6e-5).
        yy, xx = torch.meshgrid(
            torch.arange(19, dtype=torch.float64), torch.arange(19, dtype=torch.float64), indexing="ij"
        )
        estimator = PatchAffineShapeEstimator(19).to(device, dtype)
        shape = estimator((0.25 * xx + 1e-3 * yy).view(1, 1, 19, 19).to(device, dtype))[0, 0]
        mirror = estimator((0.25 * xx - 1e-3 * yy).view(1, 1, 19, 19).to(device, dtype))[0, 0]
        assert shape[1] > 0
        assert shape[2] < 1e-3
        self.assert_close(mirror, torch.tensor([1.0, 0.0, 1.0], device=device, dtype=dtype), rtol=0.0, atol=0.0)

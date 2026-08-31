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

from kornia.feature.affine_shape import LAFAffineShapeEstimator, LAFAffNetShapeEstimator, PatchAffineShapeEstimator
from kornia.feature.laf import make_upright

from testing.base import BaseTester, supports_conv2d, supports_grid_sample, supports_replicate_padding


class DegenerateShape(torch.nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        shape = patches.mean(dim=(-2, -1), keepdim=False).unsqueeze(-1) * 0
        return torch.cat([shape, shape, torch.ones_like(shape)], dim=-1)


class SingularAffNetOutput(torch.nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        values = patches.new_tensor([-1.0, 0.0, -1.0]).view(1, 3, 1, 1)
        return values.expand(patches.shape[0], -1, -1, -1)


class TestPatchAffineShapeEstimator(BaseTester):
    def test_zero_patch_uses_circular_shape(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16) and not (
            supports_conv2d(device, dtype) and supports_replicate_padding(device, dtype)
        ):
            pytest.skip(f"no {dtype} Sobel kernels on {device.type}")
        patch = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        out = PatchAffineShapeEstimator(32).to(device, dtype)(patch)
        expected = torch.tensor([[[1.0, 0.0, 1.0]]], device=device, dtype=dtype)
        self.assert_close(out, expected)

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
        if dtype == torch.float16:
            # The half-precision moment matrix is degenerate and uses the source-level circular fallback.
            expected = laf
        else:
            expected = torch.tensor([[[[35.078, 0.0, 16.0], [0.0, 11.403, 16.0]]]], device=device, dtype=dtype)
        self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_toy_preserve(self, device, dtype):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=True).to(device, dtype)
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[0.0, 20.0, 16.0], [-20.0, 0.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        if dtype == torch.float16:
            # The circular fallback preserves the original orientation up to half-precision angle rounding.
            expected = laf
        else:
            expected = torch.tensor([[[[0.0, 35.078, 16.0], [-11.403, 0, 16.0]]]], device=device, dtype=dtype)
        self.assert_close(new_laf, expected, atol=1e-2 if dtype == torch.float16 else 1e-4, rtol=1e-4)

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
        if not (supports_conv2d(device, dtype) and supports_grid_sample(device, dtype)):
            # Patch extraction needs `grid_sample`, the moment matrix needs the Sobel convolution;
            # older torch lacks the float16 CPU kernels (see testing/base.py).
            pytest.skip(f"no float16 conv2d/grid_sample kernels on {device.type}")
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
        if dtype in (torch.float16, torch.bfloat16) and not supports_grid_sample(device, dtype):
            pytest.skip(f"no {dtype} grid_sample kernel on {device.type}")
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        laf = torch.tensor([[[[0.0, 8.0, 16.0], [-8.0, 0.0, 16.0]]]], device=device, dtype=dtype)
        aff = LAFAffNetShapeEstimator(preserve_orientation=False).to(device, dtype)
        aff.features = SingularAffNetOutput()
        out = aff(laf, img)
        self.assert_close(out, make_upright(laf))

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

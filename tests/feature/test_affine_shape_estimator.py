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

from testing.base import BaseTester


class TestPatchAffineShapeEstimator(BaseTester):
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

    @pytest.mark.parametrize("patch_size", [19, 32])
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_precision(self, device, dtype, patch_size, batch_size):
        """Flat and anisotropic patches retain their shape and finite input gradients."""
        patches = torch.zeros(batch_size, 1, patch_size, patch_size, device=device, dtype=dtype)
        middle = patch_size // 2
        patches[:, :, middle - 1 : middle + 1, 2:-2] = 1
        if batch_size == 3:
            patches[0].zero_()
            patches[2] = patches[1].transpose(-2, -1)

        expected = PatchAffineShapeEstimator(patch_size).to(device)(patches.float()).to(dtype)
        module = PatchAffineShapeEstimator(patch_size).to(device, dtype)
        weighting = module.weighting.clone()
        patches.requires_grad_()

        actual = module(patches)

        assert actual.shape == (batch_size, 1, 3)
        assert actual.dtype == dtype
        assert actual.device == device
        self.assert_close(actual, expected)
        self.assert_close(module.weighting, weighting, atol=0, rtol=0)
        actual.sum().backward()
        assert patches.grad is not None and torch.isfinite(patches.grad).all()
        if batch_size == 3:
            assert (patches.grad[0] == 0).all()

    @pytest.mark.parametrize("half_dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_cpu(self, device, half_dtype):
        """Keep the half-precision regression covered in ordinary CPU CI jobs."""
        if device.type != "cpu":
            pytest.skip("Explicit half-precision CI guard is CPU-only. Other devices use the dtype fixture.")
        patches = torch.zeros(2, 1, 32, 32, device=device, dtype=half_dtype)
        patches[1, :, 15:17, 9:23] = 1
        actual = PatchAffineShapeEstimator(32).to(device, half_dtype)(patches)
        # The anisotropic value is the float32 result for the rectangle in #4123.
        expected = torch.tensor([[[1.0, 0.0, 1.0]], [[0.0883344, 0.0, 1.0]]], device=device, dtype=half_dtype)
        assert actual.dtype == half_dtype
        self.assert_close(actual, expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        """Compilation preserves the public result and output dtype."""
        module = PatchAffineShapeEstimator(32).to(device, dtype)
        patches = torch.zeros(2, 1, 32, 32, device=device, dtype=dtype)
        patches[1, :, 15:17, 9:23] = 1
        actual = torch_optimizer(module)(patches)
        assert actual.dtype == dtype
        self.assert_close(actual, module(patches))


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
        self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_toy_preserve(self, device, dtype):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=True).to(device, dtype)
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[0.0, 20.0, 16.0], [-20.0, 0.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[0.0, 35.078, 16.0], [-11.403, 0, 16.0]]]], device=device, dtype=dtype)
        self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

    def test_toy_not_preserve(self, device):
        aff = LAFAffineShapeEstimator(32, preserve_orientation=False).to(device)
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[0.0, 20.0, 16.0], [-20.0, 0.0, 16.0]]]], device=device)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[35.078, 0, 16.0], [0, 11.403, 16.0]]]], device=device)
        self.assert_close(new_laf, expected, atol=1e-4, rtol=1e-4)

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

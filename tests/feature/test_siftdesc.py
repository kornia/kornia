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

from kornia.feature.siftdesc import (
    DenseSIFTDescriptor,
    SIFTDescriptor,
    get_sift_bin_ksize_stride_pad,
    get_sift_pooling_kernel,
)

from testing.base import BaseTester, supports_replicate_padding


@pytest.mark.parametrize("ksize", [5, 13, 25])
def test_get_sift_pooling_kernel(ksize):
    kernel = get_sift_pooling_kernel(ksize)
    assert kernel.shape == (ksize, ksize)


@pytest.mark.parametrize("ps,n_bins,ksize,stride,pad", [(41, 3, 20, 13, 5), (32, 4, 12, 8, 3)])
def test_get_sift_bin_ksize_stride_pad(ps, n_bins, ksize, stride, pad):
    out = get_sift_bin_ksize_stride_pad(ps, n_bins)
    assert out == (ksize, stride, pad)


class TestSIFTDescriptor(BaseTester):
    def test_shape(self, device, dtype):
        inp = torch.ones(1, 1, 32, 32, device=device, dtype=dtype)
        sift = SIFTDescriptor(32).to(device, dtype)
        out = sift(inp)
        assert out.shape == (1, 128)

    def test_batch_shape(self, device, dtype):
        inp = torch.ones(2, 1, 15, 15, device=device, dtype=dtype)
        sift = SIFTDescriptor(15).to(device, dtype)
        out = sift(inp)
        assert out.shape == (2, 128)

    def test_batch_shape_non_std(self, device, dtype):
        inp = torch.ones(3, 1, 19, 19, device=device, dtype=dtype)
        sift = SIFTDescriptor(19, 5, 3).to(device, dtype)
        out = sift(inp)
        assert out.shape == (3, (3**2) * 5)

    def test_toy(self, device, dtype):
        patch = torch.ones(1, 1, 6, 6, device=device, dtype=dtype)
        patch[0, 0, :, 3:] = 0
        sift = SIFTDescriptor(6, num_ang_bins=4, num_spatial_bins=1, clipval=0.2, rootsift=False).to(device, dtype)
        out = sift(patch)
        expected = torch.tensor([[0, 0, 1.0, 0]], device=device, dtype=dtype)
        self.assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        dtype = torch.float64
        batch_size, channels, height, width = 1, 1, 15, 15
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        sift = SIFTDescriptor(15).to(device, dtype)
        self.gradcheck(sift, (patches,), nondet_tol=1e-4)

    @pytest.mark.skip("Compiled functions can't take variable number")
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = SIFTDescriptor(41).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(SIFTDescriptor(41).to(patches.device, patches.dtype).eval())
        self.assert_close(model(patches), model_jit(patches))


class TestSIFTDescriptorKernelBuffer(BaseTester):
    """`gk` must be a real buffer so `.to()` moves it, and `forward` must not rebind it (#4069)."""

    def test_to_moves_the_kernel(self, device):
        """`.to()` must carry the kernel along with the parameters.

        float16 rather than the default float32, or the dtype assertion would hold
        vacuously; float16 also works on MPS, where float64 is unavailable.
        """
        mod = SIFTDescriptor(32).to(device, torch.float16)
        assert mod.gk.dtype == torch.float16
        assert mod.gk.device == torch.empty(0, device=device).device

    def test_kernel_stays_out_of_state_dict(self, device):
        """Registered non-persistent, so existing checkpoints keep loading with strict=True."""
        assert "gk" not in SIFTDescriptor(32).state_dict()

    def test_forward_does_not_mutate_the_module(self, device):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        mod = SIFTDescriptor(32)
        before = (mod.gk.dtype, mod.gk.device)
        mod(torch.rand(2, 1, 32, 32, dtype=torch.float64))
        assert (mod.gk.dtype, mod.gk.device) == before


class TestSIFTConstantPatchIsFinite(BaseTester):
    """A constant patch has a zero-norm descriptor; normalising it must not give NaN.

    `F.normalize`'s default eps of 1e-12 is not representable in float16, so the clamp that
    exists to stop `0 / 0` was itself zero there. A detector that pads a short result hands the
    descriptor a zero LAF, which samples one point repeatedly and produces exactly this patch.
    """

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("rootsift", [False, True])
    def test_sift(self, device, desc_dtype, rootsift):
        if not supports_replicate_padding(device, desc_dtype):
            # `spatial_gradient` pads with mode="replicate"; torch 2.5.1 has no float16 CPU kernel.
            pytest.skip(f"no replicate-padding kernel for {desc_dtype} on {device.type}")
        patches = torch.zeros(2, 1, 16, 16, device=device, dtype=desc_dtype)
        out = SIFTDescriptor(16, rootsift=rootsift).to(device, desc_dtype)(patches)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("rootsift", [False, True])
    def test_gradients_are_finite_on_ordinary_patches(self, device, desc_dtype, rootsift):
        # Not only the forward: `sqrt` and `atan2` have an undefined backward at a zero gradient,
        # and the 1e-10 guard is zero in float16, so two equal neighbouring pixels -- ordinary
        # with a 10-bit mantissa -- put NaN into the input gradient (9 of 4096 on this seed).
        if not supports_replicate_padding(device, desc_dtype):
            pytest.skip(f"no replicate-padding kernel for {desc_dtype} on {device.type}")
        torch.manual_seed(0)
        patches = torch.rand(4, 1, 32, 32, device=device, dtype=desc_dtype, requires_grad=True)
        out = SIFTDescriptor(32, rootsift=rootsift).to(device, desc_dtype)(patches)
        out.sum().backward()
        assert torch.isfinite(out).all()
        assert patches.grad is not None and torch.isfinite(patches.grad).all()

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dense_sift(self, device, desc_dtype):
        if not supports_replicate_padding(device, desc_dtype):
            # `spatial_gradient` pads with mode="replicate"; torch 2.5.1 has no float16 CPU kernel.
            pytest.skip(f"no replicate-padding kernel for {desc_dtype} on {device.type}")
        img = torch.zeros(1, 1, 32, 32, device=device, dtype=desc_dtype)
        out = DenseSIFTDescriptor().to(device, desc_dtype)(img)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16])
    def test_rootsift_half_precision_matches_float32(self, device, desc_dtype):
        # The float16-representable `eps` is 6.1e-5, and `sqrt(6.1e-5)` in every empty bin pushed
        # the norm to ~1.004; the RootSIFT step therefore runs in float32 for a float16 input.
        if not supports_replicate_padding(device, desc_dtype):
            pytest.skip(f"no replicate-padding kernel for {desc_dtype} on {device.type}")
        torch.manual_seed(0)
        patches = torch.rand(4, 1, 41, 41, device=device)
        ref = SIFTDescriptor(41, rootsift=True).to(device)(patches)
        out = SIFTDescriptor(41, rootsift=True).to(device, desc_dtype)(patches.to(desc_dtype))
        assert out.dtype == desc_dtype
        self.assert_close(out.float().norm(dim=1), torch.ones(4, device=device), atol=2e-3, rtol=0)
        self.assert_close(out.float(), ref, atol=2e-2, rtol=0)

    def test_the_float16_eps_is_the_smallest_normal(self):
        # `_normalize_eps` spells it as a literal, because TorchScript cannot read `torch.finfo`.
        from kornia.core.utils import _normalize_eps

        assert _normalize_eps(torch.zeros(1, dtype=torch.float16)) == torch.finfo(torch.float16).tiny
        for dt in (torch.bfloat16, torch.float32, torch.float64):
            assert _normalize_eps(torch.zeros(1, dtype=dt)) == 1e-12


class TestDenseSIFTDescriptor(BaseTester):
    def test_shape_default(self, device, dtype):
        bs, h, w = 1, 20, 15
        inp = torch.rand(1, 1, h, w, device=device, dtype=dtype)
        sift = DenseSIFTDescriptor().to(device, dtype)
        out = sift(inp)
        assert out.shape == torch.Size([bs, 128, h, w])

    def test_batch_shape(self, device, dtype):
        bs, h, w = 2, 32, 15
        inp = torch.rand(bs, 1, h, w, device=device, dtype=dtype)
        sift = DenseSIFTDescriptor().to(device, dtype)
        out = sift(inp)
        assert out.shape == torch.Size([bs, 128, h, w])

    def test_batch_shape_custom(self, device, dtype):
        bs, h, w = 2, 40, 30
        inp = torch.rand(bs, 1, h, w, device=device, dtype=dtype)
        sift = DenseSIFTDescriptor(5, 3, 3, padding=1, stride=2).to(device, dtype)
        out = sift(inp)
        assert out.shape == torch.Size([bs, 45, h // 2, w // 2])

    def test_print(self, device):
        sift = DenseSIFTDescriptor()
        sift.__repr__()

    def test_instances_do_not_share_buffer_storage(self, device):
        """Instances must not share storage for `_poolingconv_weight` (see #4068).

        `_get_reshape_kernel` used to memoise `torch.eye(numel)` and hand out a view of it, so
        every `DenseSIFTDescriptor` with the same configuration shared one storage and a single
        in-place write leaked into all of them.
        """
        a = DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4)
        b = DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4)
        assert a._poolingconv_weight.untyped_storage().data_ptr() != (
            b._poolingconv_weight.untyped_storage().data_ptr()
        )

    def test_load_state_dict_does_not_leak_into_other_instances(self, device):
        """`load_state_dict` copies into buffers in place; that must stay local to one module."""
        a = DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4)
        b = DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4)
        expected = a._poolingconv_weight.clone()

        state = b.state_dict()
        state["_poolingconv_weight"] = torch.zeros_like(state["_poolingconv_weight"])
        b.load_state_dict(state)

        self.assert_close(a._poolingconv_weight, expected)
        # and a module built afterwards is still healthy
        c = DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4)
        self.assert_close(c._poolingconv_weight, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 16, 16
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(DenseSIFTDescriptor(4, 2, 2), (patches), nondet_tol=1e-4)

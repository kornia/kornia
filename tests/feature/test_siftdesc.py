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

from kornia.core.exceptions import BaseError, ShapeError
from kornia.feature import SIFTDescriptorFromPyramid, convert_sift_descriptor_layout, laf_from_center_scale_ori
from kornia.feature.siftdesc import (
    DenseSIFTDescriptor,
    SIFTDescriptor,
    _dense_sift_histograms_from_gradients,
    get_sift_bin_ksize_stride_pad,
    get_sift_pooling_kernel,
)
from kornia.filters import gaussian_blur2d

from testing.base import BaseTester, supports_conv2d, supports_replicate_padding


@pytest.mark.parametrize("ksize", [5, 13, 25])
def test_get_sift_pooling_kernel(ksize):
    kernel = get_sift_pooling_kernel(ksize)
    assert kernel.shape == (ksize, ksize)


# Cell centres -pad + (ksize - 1) / 2 + k * stride: (41, 3) 7 / 20 / 33, (32, 4) 3.5 / 11.5 / 19.5 / 27.5,
# (41, 4) 5 / 15 / 25 / 35 and (11, 4) 2 / 4 / 6 / 8, each grid centred on (ps - 1) / 2. The off-centre helper of
# #5691 rejected (11, 4): its 4-pixel kernel with pad 1 gave five cells.
@pytest.mark.parametrize(
    "ps,n_bins,ksize,stride,pad", [(41, 3, 21, 13, 3), (32, 4, 12, 8, 2), (41, 4, 17, 10, 3), (11, 4, 5, 2, 0)]
)
def test_get_sift_bin_ksize_stride_pad(ps, n_bins, ksize, stride, pad):
    out = get_sift_bin_ksize_stride_pad(ps, n_bins)
    assert out == (ksize, stride, pad)


@pytest.mark.parametrize("ps,n_bins", [(3, 4), (0, 1)])
def test_get_sift_bin_ksize_stride_pad_rejects_a_patch_smaller_than_the_bins_5691(ps, n_bins):
    # A zero cell spacing used to raise ZeroDivisionError from the output-size check.
    with pytest.raises(ValueError, match="incompatible"):
        get_sift_bin_ksize_stride_pad(ps, n_bins)


# Every (patch_size, num_spatial_bins) the suite builds a SIFTDescriptor with, odd/even pairs (31/32/33, 40/41,
# 64/65), the 41-pixel default of SIFTFeature and SIFTFeatureScaleSpace, and (11, 4), which the off-centre helper
# rejected.
_GRID_SIZES = [
    (6, 1), (11, 4), (15, 4), (16, 4), (19, 3), (31, 4), (32, 4), (33, 4), (40, 4), (41, 3), (41, 4), (64, 4), (65, 4),
]  # fmt: skip


def _flip_lr(desc: torch.Tensor, num_ang_bins: int, num_spatial_bins: int) -> torch.Tensor:
    """The descriptor of the left-right mirrored patch, predicted from ``desc``: an angle a becomes pi - a."""
    d = desc.view(desc.shape[0], num_ang_bins, num_spatial_bins, num_spatial_bins)
    perm = [(num_ang_bins // 2 - a) % num_ang_bins for a in range(num_ang_bins)]
    return d[:, perm].flip(-1).reshape(desc.shape)


def _flip_ud(desc: torch.Tensor, num_ang_bins: int, num_spatial_bins: int) -> torch.Tensor:
    """The descriptor of the upside-down mirrored patch, predicted from ``desc``: an angle a becomes -a."""
    d = desc.view(desc.shape[0], num_ang_bins, num_spatial_bins, num_spatial_bins)
    perm = [(-a) % num_ang_bins for a in range(num_ang_bins)]
    return d[:, perm].flip(-2).reshape(desc.shape)


def _rot90(desc: torch.Tensor, num_ang_bins: int, num_spatial_bins: int) -> torch.Tensor:
    """The descriptor of ``torch.rot90(patch, 1, (-2, -1))``, predicted from ``desc``: an angle a becomes a - pi/2."""
    d = desc.view(desc.shape[0], num_ang_bins, num_spatial_bins, num_spatial_bins)
    return torch.rot90(d, 1, (-2, -1)).roll(-(num_ang_bins // 4), 1).reshape(desc.shape)


@pytest.mark.parametrize("ps,n_bins", _GRID_SIZES)
def test_get_sift_bin_ksize_stride_pad_centres_the_grid_5691(ps, n_bins):
    ksize, stride, pad = get_sift_bin_ksize_stride_pad(ps, n_bins)
    # The pooling triangle is centred at (ksize - 1) / 2 of its window; output k's window starts at k * stride - pad.
    centres = [-pad + (ksize - 1) / 2 + k * stride for k in range(n_bins)]
    # Mirror images about the patch centre (ps - 1) / 2, where the Gaussian weighting window is centred. Half-pixel
    # values are exact in binary floating point.
    assert [centres[k] + centres[n_bins - 1 - k] for k in range(n_bins)] == [ps - 1.0] * n_bins
    assert pad >= 0
    assert (ps + 2 * pad - ksize) // stride + 1 == n_bins


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

    def test_float64_axis_ramps_vote_in_one_bin_5127(self, device):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        # Every gradient of a ramp along +x (-x) points at angle 0 (pi), the centre of angular bin 0 (4).
        axis = torch.arange(32, device=device, dtype=torch.float64)
        patches = torch.stack([axis.expand(32, 32), -axis.expand(32, 32)]).unsqueeze(1)
        descriptors = SIFTDescriptor(32, rootsift=False).to(device, torch.float64)(patches).view(2, 8, 16)
        expected = torch.zeros_like(descriptors)
        expected[0, 0], expected[1, 4] = descriptors[0, 0], descriptors[1, 4]
        # The float32 pi in the bin scale leaked every vote into the bin below (4.3e-8 after normalisation).
        self.assert_close(descriptors, expected, rtol=0.0, atol=1e-12)
        assert (descriptors[0, 0] > 0.1).all()
        assert (descriptors[1, 4] > 0.1).all()

    @pytest.mark.parametrize("ps,n_bins", _GRID_SIZES)
    def test_output_has_num_spatial_bins_cells_5691(self, device, dtype, ps, n_bins):
        sift = SIFTDescriptor(ps, num_ang_bins=8, num_spatial_bins=n_bins).to(device, dtype)
        assert sift(torch.rand(2, 1, ps, ps, device=device, dtype=dtype)).shape == (2, 8 * n_bins**2)

    @pytest.mark.parametrize("ps,n_bins", _GRID_SIZES)
    def test_cell_footprints_mirror_about_the_patch_centre_5691(self, device, ps, n_bins):
        # What each output cell pools, read off the module's own pooling convolution: the gradient of one cell
        # with respect to its input map. Mirroring the patch has to map every cell's footprint onto the mirrored cell's.
        sift = SIFTDescriptor(ps, num_spatial_bins=n_bins).to(device)
        x = torch.zeros(1, 1, ps, ps, device=device, requires_grad=True)
        pooled = torch.nn.functional.conv2d(x, sift.pk.weight, None, sift.pk.stride, sift.pk.padding)
        assert pooled.shape[-2:] == (n_bins, n_bins)
        footprints = torch.stack(
            [
                torch.autograd.grad(pooled[0, 0, i, j], x, retain_graph=True)[0][0, 0]
                for i in range(n_bins)
                for j in range(n_bins)
            ]
        ).view(n_bins, n_bins, ps, ps)
        self.assert_close(footprints.flip(-1), footprints.flip(1), rtol=0.0, atol=0.0)
        self.assert_close(footprints.flip(-2), footprints.flip(0), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("ps", [32, 41, 65])
    def test_centred_blob_mirrors_to_a_permutation_5691(self, device, dtype, ps):
        # An isotropic blob at the patch centre is its own mirror image, so its descriptor must be a fixed point of
        # the predicted permutation. The off-centre grid of #5691 left a residual of 0.1235 at 41 pixels.
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        axis = torch.arange(ps, dtype=torch.float64) - (ps - 1) / 2
        blob = torch.exp(-(axis.view(-1, 1) ** 2 + axis.view(1, -1) ** 2) / (2 * (ps / 6) ** 2))
        patch = blob.to(device, dtype)[None, None]
        desc = SIFTDescriptor(ps, rootsift=False).to(device, dtype)(patch)
        self.assert_close(_flip_lr(desc, 8, 4), desc)
        self.assert_close(_flip_ud(desc, 8, 4), desc)
        self.assert_close(_rot90(desc, 8, 4), desc)

    @pytest.mark.parametrize("ps,n_bins", [(19, 3), (32, 4), (41, 4), (65, 4)])
    @pytest.mark.parametrize("transform", ["fliplr", "flipud", "rot90"])
    def test_mirrored_or_rotated_patch_permutes_the_descriptor_5691(self, device, dtype, ps, n_bins, transform):
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        patches = torch.rand(8, 1, ps, ps, device=device, dtype=dtype)
        sift = SIFTDescriptor(ps, num_ang_bins=8, num_spatial_bins=n_bins).to(device, dtype)
        moved, predict = {
            "fliplr": (patches.flip(-1), _flip_lr),
            "flipud": (patches.flip(-2), _flip_ud),
            "rot90": (torch.rot90(patches, 1, (-2, -1)), _rot90),
        }[transform]
        # A 90-degree rotation of a random 41-pixel patch matched the permuted descriptor to a mean cosine of 0.989.
        self.assert_close(sift(moved), predict(sift(patches), 8, n_bins))

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

    @pytest.mark.parametrize("module_dtype", [torch.float32, torch.float16])
    def test_loads_a_state_dict_saved_before_the_grid_was_centred_5691(self, device, module_dtype):
        """A checkpoint saved with the former 16-pixel kernel of the 41-pixel patch still loads with strict=True."""
        mod = SIFTDescriptor(41).to(device, module_dtype)
        expected = mod.get_pooling_kernel()
        assert expected.shape == (1, 1, 17, 17)
        former = get_sift_pooling_kernel(16).reshape(1, 1, 16, 16).to(device, module_dtype)
        outer = torch.nn.Sequential(mod)
        outer.load_state_dict({"0.pk.weight": former})
        assert torch.equal(mod.get_pooling_kernel(), expected)
        # The untrained kernel is rebuilt, not copied from the destination, whose kernel may have been trained.
        with torch.no_grad():
            mod.pk.weight.zero_()
        outer.load_state_dict({"0.pk.weight": former})
        assert torch.equal(mod.get_pooling_kernel(), expected)
        # A kernel of the former shape with other values is a trained one: it still fails on its shape.
        with pytest.raises(RuntimeError, match="size mismatch"):
            outer.load_state_dict({"0.pk.weight": 2 * former})

    def test_forward_does_not_mutate_the_module(self, device):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        mod = SIFTDescriptor(32)
        before = (mod.gk.dtype, mod.gk.device)
        mod(torch.rand(2, 1, 32, 32, dtype=torch.float64))
        assert (mod.gk.dtype, mod.gk.device) == before

    def test_kernel_accessors_return_copies(self, device):
        """Accessor writes must not mutate SIFT's pooling parameter or weighting buffer (#4083)."""
        mod = SIFTDescriptor(32).to(device)
        pooling = mod.get_pooling_kernel()
        weighting = mod.get_weighting_kernel()
        expected_pooling = pooling.clone()
        expected_weighting = weighting.clone()
        assert not pooling.requires_grad
        assert not weighting.requires_grad

        pooling.zero_()
        weighting.zero_()

        self.assert_close(mod.pk.weight, expected_pooling)
        self.assert_close(mod.gk, expected_weighting)

    def test_forward_uses_input_dtype_without_mutating_pooling_layer(self, device):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        torch.manual_seed(0)
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float64)
        mod = SIFTDescriptor(32).to(device)
        ref = SIFTDescriptor(32).to(device, torch.float64)
        ref.load_state_dict(mod.state_dict())

        out = mod(patches)
        with torch.no_grad():
            expected = ref(patches)

        assert out.dtype == torch.float64
        assert torch.equal(out, expected)
        assert mod.pk.weight.dtype == torch.float32
        assert mod.pk.weight.device == device
        out.sum().backward()
        assert mod.pk.weight.grad is not None

    def test_forward_uses_input_device_without_mutating_pooling_layer(self, device):
        if device.type == "cpu":
            pytest.skip("device mismatch requires a non-CPU test device")
        if not (supports_replicate_padding(device, torch.float32) and supports_conv2d(device, torch.float32)):
            pytest.skip(f"no float32 SIFT kernels on {device.type}")
        patches = torch.rand(2, 1, 32, 32, device=device)
        mod = SIFTDescriptor(32)

        out = mod(patches)

        assert out.device == device
        assert mod.pk.weight.device == torch.device("cpu")


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
        assert patches.grad is not None
        assert torch.isfinite(patches.grad).all()

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("rootsift", [False, True])
    def test_flat_patch_backward_is_finite(self, device, desc_dtype, rootsift):
        # The flat forward and the random-patch backward were each pinned, never their intersection:
        # on an exactly flat patch -- what a zero-LAF padding slot samples -- the zero-norm descriptor
        # had the `eps` clamp's `1 / eps` gradient, `1e12` in float32 and `inf` once cast to float16,
        # and the orientation's `atan2` a `1 / eps` of its own; the input gradient was NaN in float16
        # (256 of 256) and ~1e8 in float32. A zero gradient has no orientation and a zero vector no
        # direction, so both branches now have a zero derivative.
        if not supports_replicate_padding(device, desc_dtype):
            pytest.skip(f"no replicate-padding kernel for {desc_dtype} on {device.type}")
        patches = torch.zeros(2, 1, 16, 16, device=device, dtype=desc_dtype, requires_grad=True)
        out = SIFTDescriptor(16, rootsift=rootsift).to(device, desc_dtype)(patches)
        out.sum().backward()
        assert torch.isfinite(out).all()
        assert patches.grad is not None, patches.grad.abs().max()
        assert torch.isfinite(patches.grad).all(), patches.grad.abs().max()
        assert bool((patches.grad == 0).all()), f"flat patch has a gradient of {patches.grad.abs().max().item()}"

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_l2_normalize_zero_vector_has_a_zero_gradient(self, device, desc_dtype):
        from kornia.core.utils import _l2_normalize

        if device.type == "mps" and desc_dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        x = torch.zeros(3, 8, device=device, dtype=desc_dtype, requires_grad=True)
        _l2_normalize(x, dim=1).sum().backward()
        assert x.grad is not None, x.grad
        assert bool((x.grad == 0).all()), x.grad
        # and a non-zero vector keeps `F.normalize`'s value and gradient
        torch.manual_seed(0)
        y = torch.rand(3, 8, device=device, dtype=desc_dtype, requires_grad=True)
        y_ref = y.detach().clone().requires_grad_(True)
        _l2_normalize(y, dim=1).sum().backward()
        torch.nn.functional.normalize(y_ref.float(), dim=1, eps=1e-12).to(desc_dtype).sum().backward()
        self.assert_close(y.grad, y_ref.grad)

    def test_l2_normalize_propagates_nan(self, device):
        from kornia.core.utils import _l2_normalize

        nan, inf = float("nan"), float("inf")
        # [-3, 0]: a non-zero vector whose largest entry is 0, so the zero test must run on magnitudes
        x = torch.tensor([[nan, 1.0], [inf, 0.0], [0.0, 0.0], [3.0, 4.0], [-3.0, 0.0]], device=device)
        out = _l2_normalize(x, dim=1)
        # a vector holding a NaN is not a zero vector: it keeps `F.normalize`'s NaN instead of becoming zeros
        assert bool(torch.isnan(out[0]).all()), out[0]
        torch.testing.assert_close(out[1:], torch.nn.functional.normalize(x[1:], dim=1), equal_nan=True)

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

    def test_float16_subnormal_norm_still_normalises_to_one(self, device):
        # A float16 vector whose norm sits below the smallest normal (6.1e-5) is representable
        # and torch's fp16 `norm` accumulates in fp32, so clamping the norm to 6.1e-5 shrank the
        # result to 0.55 instead of 1. The normalisation runs in float32 for a float16 input.
        from kornia.core.utils import _l2_normalize

        x = torch.full((1, 128), 3e-6, device=device, dtype=torch.float16)
        out = _l2_normalize(x, dim=1)
        assert out.dtype == torch.float16
        self.assert_close(out.float().norm(dim=1), torch.ones(1, device=device), atol=2e-3, rtol=0)
        assert (_l2_normalize(torch.zeros_like(x), dim=1) == 0).all()
        comparison_dtypes = (torch.bfloat16, torch.float32)
        if device.type != "mps":
            comparison_dtypes += (torch.float64,)
        for dt in comparison_dtypes:
            y = torch.rand(3, 8, device=device, dtype=dt)
            assert torch.equal(_l2_normalize(y, dim=1), torch.nn.functional.normalize(y, dim=1, eps=1e-12))


class TestDenseSIFTDescriptor(BaseTester):
    def test_shape_default(self, device, dtype):
        bs, h, w = 1, 20, 15
        inp = torch.rand(1, 1, h, w, device=device, dtype=dtype)
        sift = DenseSIFTDescriptor().to(device, dtype)
        out = sift(inp)
        # (20 + 1 + 2 - 12 - 1) // 1 + 1 and (15 + 1 + 2 - 12 - 1) // 1 + 1
        assert out.shape == torch.Size([bs, 128, 11, 6])

    def test_batch_shape(self, device, dtype):
        bs, h, w = 2, 32, 15
        inp = torch.rand(bs, 1, h, w, device=device, dtype=dtype)
        sift = DenseSIFTDescriptor().to(device, dtype)
        out = sift(inp)
        # (32 + 1 + 2 - 12 - 1) // 1 + 1 and (15 + 1 + 2 - 12 - 1) // 1 + 1
        assert out.shape == torch.Size([bs, 128, 23, 6])

    def test_batch_shape_custom(self, device, dtype):
        bs, h, w = 2, 40, 30
        inp = torch.rand(bs, 1, h, w, device=device, dtype=dtype)
        sift = DenseSIFTDescriptor(5, 3, 3, padding=1, stride=2).to(device, dtype)
        out = sift(inp)
        # (40 + 0 + 2 - 6 - 1) // 2 + 1 and (30 + 0 + 2 - 6 - 1) // 2 + 1
        assert out.shape == torch.Size([bs, 45, 18, 13])

    @pytest.mark.parametrize(
        "kwargs,h,w,expected",
        [
            ({"padding": 6}, 20, 15, (21, 16)),  # even bin size: one more row and column than the input
            ({"spatial_bin_size": 5, "padding": 7}, 20, 15, (19, 14)),  # odd bin size: no extra row or column
            ({"spatial_bin_size": 3, "num_spatial_bins": 3, "padding": 3, "stride": 2}, 40, 30, (20, 15)),
            ({"padding": 0}, 40, 53, (29, 42)),
        ],
    )
    def test_output_size_rule_5733(self, device, dtype, kwargs, h, w, expected):
        out = DenseSIFTDescriptor(**kwargs).to(device, dtype)(torch.rand(1, 1, h, w, device=device, dtype=dtype))
        assert out.shape[-2:] == torch.Size(expected)

    def test_print(self, device):
        sift = DenseSIFTDescriptor()
        sift.__repr__()

    def test_pooling_kernel_accessor_is_isolated_and_tracks_to(self, device):
        """Dense SIFT's accessor must return a copy of the kernel used in forward (#4083)."""
        mod = DenseSIFTDescriptor()
        expected = mod.bin_pooling_kernel.weight.detach().clone()
        mod.get_pooling_kernel().zero_()
        self.assert_close(mod.bin_pooling_kernel.weight, expected)

        target_dtype = torch.float32 if device.type == "mps" else torch.float64
        mod.to(device, target_dtype)
        kernel = mod.get_pooling_kernel()
        assert not kernel.requires_grad
        assert kernel.dtype == target_dtype
        assert kernel.device == device

        if device.type != "mps":
            mod = DenseSIFTDescriptor()
            before = (mod.bin_pooling_kernel.weight.dtype, mod.bin_pooling_kernel.weight.device)
            mod(torch.rand(1, 1, 16, 16, device=device, dtype=torch.float64))
            kernel = mod.get_pooling_kernel()
            assert (kernel.dtype, kernel.device) == before
            self.assert_close(kernel, mod.bin_pooling_kernel.weight)

    def test_forward_uses_input_dtype_without_mutating_convolution_layers(self, device):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        torch.manual_seed(0)
        img = torch.rand(1, 1, 16, 16, device=device, dtype=torch.float64)
        mod = DenseSIFTDescriptor().to(device)
        ref = DenseSIFTDescriptor().to(device, torch.float64)
        ref.load_state_dict(mod.state_dict())

        out = mod(img)
        with torch.no_grad():
            expected = ref(img)

        assert out.dtype == torch.float64
        assert torch.equal(out, expected)
        assert mod.bin_pooling_kernel.weight.dtype == torch.float32
        assert mod.PoolingConv.weight.dtype == torch.float32
        assert mod.bin_pooling_kernel.weight.device == device
        assert mod.PoolingConv.weight.device == device
        out.sum().backward()
        assert mod.bin_pooling_kernel.weight.grad is not None
        assert mod.PoolingConv.weight.grad is not None

    def test_forward_uses_input_device_without_mutating_convolution_layers(self, device):
        if device.type == "cpu":
            pytest.skip("device mismatch requires a non-CPU test device")
        if not (supports_replicate_padding(device, torch.float32) and supports_conv2d(device, torch.float32)):
            pytest.skip(f"no float32 DenseSIFT kernels on {device.type}")
        img = torch.rand(1, 1, 16, 16, device=device)
        mod = DenseSIFTDescriptor()

        out = mod(img)

        assert out.device == device
        assert mod.bin_pooling_kernel.weight.device == torch.device("cpu")
        assert mod.PoolingConv.weight.device == torch.device("cpu")

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

    def test_receptive_field_is_the_cell_grid_5733(self, device, dtype):
        """One output depends on `num_spatial_bins * spatial_bin_size` pixels (plus the gradient stencil)."""
        torch.manual_seed(0)
        img = torch.rand(1, 1, 40, 53, device=device, dtype=dtype, requires_grad=True)
        mod = DenseSIFTDescriptor(padding=0, rootsift=False, clipval=1.0).to(device, dtype)
        out = mod(img)
        assert out.shape[-2:] == torch.Size([29, 42])
        weights = torch.rand(128, device=device, dtype=dtype) + 0.5
        (out[0, :, 10, 20] * weights).sum().backward()
        ys, xs = img.grad[0, 0].nonzero(as_tuple=True)
        # the centre is (10 + 6 - 0.5, 20 + 6 - 0.5); 16 px of cells, one more pixel each side for the gradient
        assert (int(ys.min()), int(ys.max())) == (7, 24)
        assert (int(xs.min()), int(xs.max())) == (18 - 1, 33 + 1)

    def test_vertical_edge_lights_one_cell_column_5733(self, device, dtype):
        """An edge inside a cell's band is seen by that cell column only, not by its neighbours."""
        h, w, j = 40, 53, 20
        mod = DenseSIFTDescriptor(padding=0, rootsift=False, clipval=1.0).to(device, dtype)
        for edge, cell in [(j + 3, 1), (j + 8, 2), (j + 12, 3), (j + 4, 1)]:
            img = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
            img[..., edge:] = 1.0
            out = mod(img)[0, :, 10, j]
            # (angle, cell row, cell column) -> energy of each cell column
            per_col = out.view(8, 4, 4).pow(2).sum(dim=(0, 1))
            expected = torch.zeros(4, device=device, dtype=dtype)
            expected[cell] = 1.0
            self.assert_close(per_col / per_col.sum(), expected, atol=1e-3, rtol=1e-3)

    def test_matches_sift_descriptor_cell_layout_5733(self, device, dtype):
        """The dense descriptor at a pixel is close to `SIFTDescriptor` of the 16 x 16 patch around that pixel."""
        torch.manual_seed(0)
        img = gaussian_blur2d(torch.rand(1, 1, 60, 70, device=device, dtype=dtype), (5, 5), (1.5, 1.5))
        dense = DenseSIFTDescriptor(padding=0).to(device, dtype)(img)[0]
        patch_desc = SIFTDescriptor(16, 8, 4).to(device, dtype)
        for y0, x0 in [(5, 7), (20, 30), (33, 11), (10, 45)]:
            ref = patch_desc(img[:, :, y0 : y0 + 16, x0 : x0 + 16])[0]
            # the patch centre (y0 + 7.5, x0 + 7.5) is the dense centre of output (y0 + 2, x0 + 2)
            cos = torch.nn.functional.cosine_similarity(dense[:, y0 + 2, x0 + 2], ref, dim=0)
            assert float(cos.detach()) > 0.95
            window = dense[:, y0 : y0 + 5, x0 : x0 + 5]
            cos_map = torch.nn.functional.cosine_similarity(window, ref[:, None, None], dim=0)
            assert int(cos_map.detach().argmax()) == 2 * 5 + 2

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 16, 16
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(DenseSIFTDescriptor(4, 2, 2), (patches), nondet_tol=1e-4)

    @pytest.mark.parametrize("bins", [8, 36])
    def test_float64_bin_centre_gradient_votes_in_one_bin_5127(self, device, bins):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        angle = torch.arange(bins, device=device, dtype=torch.float64) * (2 * torch.pi / bins)
        # Unit gradients at the bin centres, plus one zero gradient that votes nowhere.
        gx = torch.cat([torch.cos(angle), angle.new_zeros(1)]).view(1, 1, 1, bins + 1)
        gy = torch.cat([torch.sin(angle), angle.new_zeros(1)]).view(1, 1, 1, bins + 1)
        histograms = _dense_sift_histograms_from_gradients(gx, gy, bins, eps=0.0)
        expected = torch.zeros(bins, bins + 1, device=device, dtype=torch.float64)
        expected[:, :bins] = torch.eye(bins, device=device, dtype=torch.float64)
        # The float32 pi in the orientation offset and the bin scale leaked up to 1e-7 into the lower bin.
        self.assert_close(histograms[0, :, 0], expected, rtol=0.0, atol=1e-12)


class TestConvertSIFTDescriptorLayout(BaseTester):
    def test_smoke(self, device, dtype):
        descriptors = torch.rand(2, 5, 128, device=device, dtype=dtype)
        converted = convert_sift_descriptor_layout(descriptors, "kornia", "opencv")
        assert converted.shape == descriptors.shape
        assert converted.dtype == dtype
        assert converted.device == descriptors.device

    def test_index_mapping(self, device, dtype):
        # kornia stores (angle, row, column) and OpenCV (row, column, angle), with the angle bins counted the other
        # way round: kornia index a * 16 + y * 4 + x is OpenCV index (y * 4 + x) * 8 + (-a mod 8).
        one_hot = torch.eye(128, device=device, dtype=dtype)
        converted = convert_sift_descriptor_layout(one_hot, "kornia", "opencv")
        assert torch.equal(converted.sum(0), torch.ones(128, device=device, dtype=dtype))
        for kornia_index, opencv_index in ((0, 0), (18, 23), (13, 104), (32, 6), (127, 121)):
            assert converted[kornia_index].argmax().item() == opencv_index

    @pytest.mark.parametrize("num_ang_bins, num_spatial_bins", [(8, 4), (6, 3), (4, 1)])
    def test_round_trip(self, device, dtype, num_ang_bins, num_spatial_bins):
        descriptors = torch.rand(3, 2, num_ang_bins * num_spatial_bins**2, device=device, dtype=dtype)
        bins = {"num_ang_bins": num_ang_bins, "num_spatial_bins": num_spatial_bins}
        opencv = convert_sift_descriptor_layout(descriptors, "kornia", "opencv", **bins)
        assert not torch.equal(opencv, descriptors)
        assert torch.equal(convert_sift_descriptor_layout(opencv, "opencv", "kornia", **bins), descriptors)
        assert torch.equal(convert_sift_descriptor_layout(descriptors, "kornia", "kornia", **bins), descriptors)

    def test_angle_bins_follow_each_convention(self, device, dtype):
        # Intensity grows downwards, so every gradient points along +y. kornia measures angles from +x towards +y
        # (image rows run downwards), which puts it in bin 2 of 8. OpenCV measures them towards -y: bin 6.
        ramp = torch.arange(32, device=device, dtype=torch.float32).view(32, 1).expand(32, 32)
        descriptor = SIFTDescriptor(32, rootsift=False).to(device)(ramp[None, None]).to(dtype)
        assert descriptor.view(8, 16).sum(-1).argmax().item() == 2
        opencv = convert_sift_descriptor_layout(descriptor, "kornia", "opencv")
        assert opencv.view(16, 8).sum(0).argmax().item() == 6

    def test_exception(self, device, dtype):
        descriptors = torch.rand(2, 128, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="layout"):
            convert_sift_descriptor_layout(descriptors, "kornia", "vlfeat")
        with pytest.raises(ValueError, match="layout"):
            convert_sift_descriptor_layout(descriptors, "colmap", "kornia")
        with pytest.raises(BaseError):
            convert_sift_descriptor_layout(torch.rand(2, 100, device=device, dtype=dtype), "kornia", "opencv")
        # -4 spatial bins square to the size of 4, and 0 angle bins match an empty descriptor.
        with pytest.raises(ValueError, match="bin counts must be positive"):
            convert_sift_descriptor_layout(descriptors, "kornia", "opencv", num_spatial_bins=-4)
        with pytest.raises(ValueError, match="bin counts must be positive"):
            convert_sift_descriptor_layout(descriptors[:, :0], "kornia", "opencv", num_ang_bins=0)

    def test_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        descriptors = torch.rand(2, 128, device=device, dtype=torch.float64)
        self.gradcheck(lambda d: convert_sift_descriptor_layout(d, "kornia", "opencv"), (descriptors,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        descriptors = torch.rand(2, 128, device=device, dtype=dtype)

        def op(d: torch.Tensor) -> torch.Tensor:
            return convert_sift_descriptor_layout(d, "kornia", "opencv")

        self.assert_close(torch_optimizer(op, fullgraph=True)(descriptors), op(descriptors))


def _sift_family_member(kind, device, dtype, image, **kwargs):
    """Describe ``image`` with the ``kind`` SIFT descriptor; the descriptor axis is returned last."""
    if kind == "patch":
        return SIFTDescriptor(image.shape[-1], **kwargs).to(device, dtype)(image)
    if kind == "dense":
        return DenseSIFTDescriptor(**kwargs).to(device, dtype)(image).movedim(1, -1)
    # two oriented, sheared frames off the image centre, both straddling the step edge of the fixtures below
    lafs = laf_from_center_scale_ori(
        torch.tensor([[[30.0, 20.0], [33.0, 31.0]]], device=device, dtype=dtype),
        torch.tensor([[[[6.0]], [[5.0]]]], device=device, dtype=dtype),
        torch.tensor([[[20.0], [-40.0]]], device=device, dtype=dtype),
    )
    lafs[..., 0, 1] += 1.0
    return SIFTDescriptorFromPyramid(**kwargs).to(device, dtype)(image, lafs)


class TestConventionsSIFTDescriptors(BaseTester):
    """Pins for the SIFT family's patch size, clipping, RootSIFT default, flat-patch output and dense channel layout."""

    def test_convention_sift_descriptor_takes_41_pixel_patches_by_default(self, device, dtype):
        # Unlike the 32-pixel learned descriptors and the patch_size=32 default of LAFDescriptor / get_laf_descriptors;
        # the size is validated.
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        sift = SIFTDescriptor().to(device, dtype)
        assert sift.patch_size == 41
        assert sift(torch.rand(2, 1, 41, 41, device=device, dtype=dtype)).shape == (2, 128)
        with pytest.raises(ShapeError):
            sift(torch.rand(2, 1, 32, 32, device=device, dtype=dtype))

    def test_convention_sift_clips_at_clipval_between_two_l2_normalisations(self, device, dtype):
        # A vertical step between columns 24 and 25 of a 41-pixel patch: every gradient points along +x (angle bin 0)
        # and lies in cell column 2 only (cells centred at 5, 15, 25 and 35, each pooling 17 pixels). Index
        # a * 16 + cy * 4 + cx: entries 2, 6, 10 and 14. clipval=1.0 leaves the unit descriptor unclipped.
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        patch = torch.zeros(1, 1, 41, 41, device=device, dtype=dtype)
        patch[..., 25:] = 1.0
        # Relabelling: the transposed patch has a horizontal step, angle bin 2 (+y, image rows grow downwards) in cell
        # row 2, entries 32 + 8 + cx.
        for image, entries in ((patch, [2, 6, 10, 14]), (patch.transpose(-1, -2), [40, 41, 42, 43])):
            raw = SIFTDescriptor(rootsift=False, clipval=1.0).to(device, dtype)(image)[0]
            out = SIFTDescriptor(rootsift=False).to(device, dtype)(image)[0]
            assert (raw > 0.1).nonzero().flatten().tolist() == entries
            self.assert_close(raw.norm(), torch.ones((), device=device, dtype=dtype))
            assert raw[entries].min() > 0.2  # every entry is clipped
            # the cells mirror about the patch centre, as the grid is centred there
            self.assert_close(raw[entries[:2]], raw[entries[::-1][:2]])
            # unequal before the clip (the window weights the middle cells more), so the 0.5 below is the clip's doing
            assert (raw[entries] - 0.5).abs().max() > 0.01
            # normalise, clip every entry at clipval = 0.2, normalise again: the four equal clipped entries become
            # 0.5, so an output entry can exceed clipval
            assert (out > 0.1).nonzero().flatten().tolist() == entries
            self.assert_close(out[entries], torch.full((4,), 0.5, device=device, dtype=dtype))

    @pytest.mark.parametrize("kind", ["patch", "dense", "pyramid"])
    def test_convention_sift_family_defaults_to_rootsift_of_the_clipped_descriptor(self, device, dtype, kind):
        # rootsift=True and clipval=0.2 by default in all three (pytorch-sift and OpenCV return plain SIFT): the
        # default output is sqrt(L1-normalise(clipped SIFT) + 1e-10), where clipped SIFT is the unit descriptor
        # clipped at 0.2 and renormalised. clipval=1.0 leaves a unit descriptor unclipped.
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        size = {"patch": (3, 1, 41, 41), "dense": (1, 1, 30, 37), "pyramid": (1, 1, 48, 64)}[kind]
        image = 0.2 * torch.rand(*size, device=device, dtype=dtype)
        image[..., size[-1] // 2 + 4 :] += 1.0  # a step edge, so that the clipping acts
        raw = _sift_family_member(kind, device, dtype, image, rootsift=False, clipval=1.0)
        raw = raw.to(torch.promote_types(dtype, torch.float32))
        assert raw.max() > 0.25
        clipped = torch.nn.functional.normalize(raw.clamp(0.0, 0.2), dim=-1)
        expected = (clipped / clipped.sum(dim=-1, keepdim=True) + 1e-10).sqrt()
        self.assert_close(_sift_family_member(kind, device, dtype, image), expected.to(dtype))

    @pytest.mark.parametrize("kind", ["patch", "dense", "pyramid"])
    def test_convention_sift_family_describes_a_flat_patch_by_a_zero_vector(self, device, dtype, kind):
        # A flat patch has no gradient, hence no orientation: its descriptor is zero rather than a unit vector, and
        # its RootSIFT is sqrt(1e-10) = 1e-5 in every bin. A zero-LAF padding slot hands SIFTDescriptor (behind
        # LAFDescriptor) such a patch. SIFTDescriptorFromPyramid instead treats a zero LAF as invalid and returns an
        # exact zero descriptor whether or not rootsift is set.
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        size = {"patch": (3, 1, 41, 41), "dense": (1, 1, 30, 37), "pyramid": (1, 1, 48, 64)}[kind]
        image = torch.full(size, 0.5, device=device, dtype=dtype)
        plain = _sift_family_member(kind, device, dtype, image, rootsift=False)
        assert torch.equal(plain, torch.zeros_like(plain))
        root = _sift_family_member(kind, device, dtype, image)
        self.assert_close(root, torch.full_like(root, 1e-5), rtol=1e-2, atol=0.0)
        if kind == "pyramid":
            # a zero LAF next to a valid one, so that the invalid slot goes through the same pass as a real frame
            lafs = torch.zeros(1, 2, 2, 3, device=device, dtype=dtype)
            lafs[0, 0] = torch.tensor([[6.0, 1.0, 30.0], [0.0, 6.0, 20.0]], device=device, dtype=dtype)
            for rootsift in (False, True):
                desc = SIFTDescriptorFromPyramid(rootsift=rootsift).to(device, dtype)(image, lafs)[0]
                assert torch.equal(desc[1], torch.zeros_like(desc[1]))
                assert bool((desc[0] > 0).all()) == rootsift  # the valid slot of the flat image: 1e-5 or 0

    def test_convention_dense_sift_channels_are_angle_major_like_sift_descriptor(self, device, dtype):
        # Channel a * 16 + cy * 4 + cx, as SIFTDescriptor's entries. With the defaults output (i, j) is centred on
        # input (i + 4.5, j + 4.5) and its four cell columns sit 4 pixels apart, each pooling 4 pixels: at j = 25 the
        # cells pool columns 22-25, 26-29, 30-33 and 34-37. A vertical step between columns 35 and 36 has its
        # gradients (angle bin 0, +x) in cell column 3 only: channels 3, 7, 11 and 15, equal (no Gaussian weight), 0.5.
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        image = torch.zeros(1, 1, 40, 53, device=device, dtype=dtype)
        image[..., 36:] = 1.0
        dense = DenseSIFTDescriptor(rootsift=False).to(device, dtype)
        # Relabelling: transposing the image transposes the output map and turns the step horizontal, angle bin 2
        # (+y) in cell row 3: channels 32 + 12 + cx.
        for out, entries in (
            (dense(image)[0, :, 12, 25], [3, 7, 11, 15]),
            (dense(image.transpose(-1, -2))[0, :, 25, 12], [44, 45, 46, 47]),
        ):
            assert (out > 0.1).nonzero().flatten().tolist() == entries
            self.assert_close(out[entries], torch.full((4,), 0.5, device=device, dtype=dtype))

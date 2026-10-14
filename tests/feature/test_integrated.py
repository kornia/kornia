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

import copy
import sys
import warnings

import pytest
import torch
from torch import nn

import kornia
from kornia.core._compat import torch_version_le
from kornia.core.exceptions import BaseError
from kornia.feature import (
    DescriptorMatcher,
    GFTTAffNetHardNet,
    HardNet,
    KeyNetHardNet,
    LAFDescriptor,
    LocalFeature,
    ScaleSpaceDetector,
    SIFTDescriptor,
    SIFTFeature,
    extract_patches_from_pyramid,
    get_laf_center,
    get_laf_descriptors,
    get_laf_orientation,
    get_laf_scale,
    laf_is_filled,
)
from kornia.feature.integrated import LocalFeatureMatcher
from kornia.feature.scale_space_detector import get_default_detector_config
from kornia.geometry import RANSAC, resize, transform_points

from testing.base import (
    DYNAMO_UNAVAILABLE_REASON,
    BaseTester,
    dynamo_is_available,
    supports_reflect_padding,
    supports_replicate_padding,
)
from testing.casts import dict_to


class TestGetLAFDescriptors(BaseTester):
    def test_same(self, device, dtype):
        B, C, H, W = 1, 3, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        img_gray = kornia.color.rgb_to_grayscale(img)
        centers = torch.tensor([[H / 3.0, W / 3.0], [2.0 * H / 3.0, W / 2.0]], device=device, dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        sift = SIFTDescriptor(PS).to(device, dtype)
        descs_test_from_rgb = get_laf_descriptors(img, lafs, sift, PS, True)
        descs_test_from_gray = get_laf_descriptors(img_gray, lafs, sift, PS, True)

        patches = extract_patches_from_pyramid(img_gray, lafs, PS)
        B1, N1, CH1, H1, W1 = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs_reference = sift(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        self.assert_close(descs_test_from_rgb, descs_reference)
        self.assert_close(descs_test_from_gray, descs_reference)

    def test_gradcheck(self, device):
        dtype = torch.float64
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        # Off the pixel lattice: a sample exactly on a pixel centre sits on a kink of bilinear interpolation, where the
        # central difference and the one-sided analytical derivative disagree.
        centers = torch.tensor(
            [[H / 2.0 + 0.3, W / 2.0 + 0.3], [2.0 * H / 3.0, W / 2.0 + 0.3]], device=device, dtype=dtype
        ).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 5.0, (H + W) / 6.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)

        class _MeanPatch(nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(2, 3))

        desc = _MeanPatch()
        self.gradcheck(
            get_laf_descriptors,
            (img, lafs, desc, PS, True),
            eps=1e-3,
            atol=1e-3,
            nondet_tol=1e-3,
        )

    @pytest.mark.parametrize("layout", ["transposed", "channels_last"])
    def test_non_contiguous_descriptor_output(self, device, dtype, layout):
        # The per-patch output is flattened into (B, N, D). A `view` there refused an output whose trailing dimensions
        # cannot be merged without a copy, such as a transposed block or a channels-last convolution.
        B, N, PS = 2, 3, 16
        img = torch.rand(B, 1, 48, 64, device=device, dtype=dtype)
        lafs = torch.tensor(
            [
                [
                    [[6.0, 2.0, 20.0], [-1.0, 5.0, 30.0]],
                    [[4.0, -1.5, 41.0], [0.5, 7.0, 17.0]],
                    [[8.0, 1.0, 33.0], [-3.0, 6.0, 25.0]],
                ],
                [
                    [[5.0, 1.0, 12.0], [-2.0, 4.0, 36.0]],
                    [[7.0, 2.5, 50.0], [1.5, 6.0, 22.0]],
                    [[3.0, -0.5, 27.0], [1.0, 9.0, 14.0]],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        patches = extract_patches_from_pyramid(img, lafs, PS).view(B * N, 1, PS, PS)
        if layout == "transposed":

            class _TransposedBlock(nn.Module):
                def forward(self, x):
                    return x[..., :3, :5].transpose(-1, -2)

            desc = _TransposedBlock()
            expected = patches[..., :3, :5].transpose(-1, -2).reshape(B, N, 15)
        else:

            class _ChannelsLast(nn.Module):
                def forward(self, x):
                    return x.contiguous(memory_format=torch.channels_last)

            conv = nn.Conv2d(1, 4, 5, stride=4).to(device, dtype)
            desc = nn.Sequential(conv, _ChannelsLast())
            expected = conv(patches).reshape(B, N, -1)
        self.assert_close(get_laf_descriptors(img, lafs, desc, PS), expected)


class TestGetLAFDescriptorsEmptyPath(BaseTester):
    """The empty-LAF result must agree with the non-empty one (see #4065)."""

    @pytest.mark.parametrize("output_dims", [64, 128, 238])
    def test_empty_matches_the_descriptor_width(self, device, dtype, output_dims):
        """The width came from a hardcoded 128 rather than from the descriptor module."""
        img = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        desc = kornia.feature.MKDDescriptor(patch_size=32, output_dims=output_dims).to(device, dtype)
        laf = kornia.feature.laf_from_center_scale_ori(
            torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype),
            torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype),
        )
        non_empty = kornia.feature.get_laf_descriptors(img, laf, desc, 32)
        with pytest.warns(UserWarning, match=r"LAF contains no keypoints"):
            empty = kornia.feature.get_laf_descriptors(
                img, torch.zeros(1, 0, 2, 3, device=device, dtype=dtype), desc, 32
            )
        assert empty.shape == (1, 0, non_empty.shape[-1])
        assert empty.dtype == non_empty.dtype
        assert empty.device.type == non_empty.device.type

    @pytest.mark.parametrize("descriptor", ["four_dim", "dense_sift"])
    def test_empty_matches_a_non_2d_descriptor(self, device, dtype, descriptor):
        """The width must be what `.reshape(B, N, -1)` produces, not the probe's last dimension.

        The non-empty path flattens *everything* the descriptor produced per patch, so taking
        `probe.shape[-1]` agreed only for descriptors whose output is 2-D. `patch_descriptor` is
        typed as a bare `nn.Module`; `DenseSIFTDescriptor` returns `(N, C, H, W)` and so does any
        user module of that shape, and those got a narrower empty result that could not be
        concatenated with a non-empty one -- the very defect this path exists to remove (#4065).
        """

        class FourDimDescriptor(nn.Module):
            """Minimal descriptor whose output is not 2-D."""

            def forward(self, patches: torch.Tensor) -> torch.Tensor:
                return patches.new_zeros(patches.shape[0], 3, 2, 5)

        desc = (
            FourDimDescriptor()
            if descriptor == "four_dim"
            else kornia.feature.DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4)
        )
        desc = desc.to(device, dtype)
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        laf = kornia.feature.laf_from_center_scale_ori(
            torch.tensor([[[16.0, 16.0]]], device=device, dtype=dtype),
            torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype),
        )
        non_empty = kornia.feature.get_laf_descriptors(img, laf, desc, 32)
        with pytest.warns(UserWarning, match=r"LAF contains no keypoints"):
            empty = kornia.feature.get_laf_descriptors(
                img, torch.zeros(1, 0, 2, 3, device=device, dtype=dtype), desc, 32
            )
        assert empty.shape == (1, 0, non_empty.shape[-1])
        # which is the point: an empty batch has to be concatenable with a non-empty one
        assert torch.cat([non_empty, empty], dim=1).shape == non_empty.shape


class TestLocalFeatureMatcherEmptyPath(BaseTester):
    """`no_match_output` must keep the rank the success path uses (see #4065)."""

    def test_empty_lafs_keep_the_batch_dimension(self, device, dtype):
        matcher = kornia.feature.LocalFeatureMatcher(
            kornia.feature.SIFTFeature(50), kornia.feature.DescriptorMatcher("snn", 0.9)
        )
        out = matcher.no_match_output(device, dtype)
        # the success path returns `.view(1, -1, 2, 3)`
        assert out["lafs0"].shape == (1, 0, 2, 3)
        assert out["lafs1"].shape == (1, 0, 2, 3)
        # so indexing the batch works either way
        assert out["lafs0"][0].shape == (0, 2, 3)
        assert out["keypoints0"].shape == (0, 2)

    def test_no_match_keypoints_keep_the_laf_dtype(self, device, dtype):
        # Kornia's detectors return float32 LAFs for a float16 or bfloat16 image; a pair without matches returns
        # keypoints and LAFs in the dtype a matched pair has, so the two concatenate without a cast.
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS has no float64")
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        matcher = kornia.feature.LocalFeatureMatcher(
            kornia.feature.SIFTFeature(50), kornia.feature.DescriptorMatcher("nn")
        ).to(device, dtype)
        g = torch.Generator().manual_seed(0)
        img = torch.nn.functional.avg_pool2d(torch.rand(1, 1, 64, 80, generator=g), 3, 1, 1).to(device, dtype)
        with torch.no_grad():
            matched = matcher({"image0": img, "image1": img})
            empty = matcher({"image0": torch.zeros_like(img), "image1": torch.zeros_like(img)})
        assert matched["keypoints0"].shape[0] > 0
        assert empty["keypoints0"].shape[0] == 0
        for key in ("keypoints0", "keypoints1", "lafs0", "lafs1"):
            assert matched[key].dtype == empty[key].dtype == torch.promote_types(dtype, torch.float32)
        assert empty["confidence"].dtype == matched["confidence"].dtype == dtype

    def test_an_empty_side_never_reaches_the_matcher(self, device, dtype):
        # Filtering the zero-LAF padding can leave a side with no descriptor at all. `matcher` is an
        # arbitrary module with no obligation to accept a `(0, D)` input, so a textureless pair called
        # it with one and could raise; the pair is skipped instead, which is what `no_match_output` is for.
        class RejectsEmpty(torch.nn.Module):
            def forward(self, d0: torch.Tensor, d1: torch.Tensor):
                if d0.shape[0] == 0 or d1.shape[0] == 0:
                    raise RuntimeError("empty descriptors")
                return kornia.feature.match_snn(d0, d1, 0.9)

        matcher = kornia.feature.LocalFeatureMatcher(kornia.feature.SIFTFeature(50), RejectsEmpty()).to(device, dtype)
        img = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        out = matcher({"image0": img, "image1": img})
        assert out["keypoints0"].shape == (0, 2)
        assert out["lafs0"].shape == (1, 0, 2, 3)


class TestLAFDescriptor(BaseTester):
    def test_same(self, device, dtype):
        B, C, H, W = 1, 3, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        img_gray = kornia.color.rgb_to_grayscale(img)
        centers = torch.tensor([[H / 3.0, W / 3.0], [2.0 * H / 3.0, W / 2.0]], device=device, dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        sift = SIFTDescriptor(PS).to(device, dtype)
        lafsift = LAFDescriptor(sift, PS)
        descs_test = lafsift(img, lafs)
        patches = extract_patches_from_pyramid(img_gray, lafs, PS)
        B1, N1, CH1, H1, W1 = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs_reference = sift(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        self.assert_close(descs_test, descs_reference)

    def test_empty(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        lafs = torch.zeros(B, 0, 2, 3, device=device)
        sift = SIFTDescriptor(PS).to(device)
        lafsift = LAFDescriptor(sift, PS)
        descs_test = lafsift(img, lafs)
        assert descs_test.shape == (B, 0, 128)

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        # Off the pixel lattice: a sample exactly on a pixel centre sits on a kink of bilinear interpolation, where the
        # central difference and the one-sided analytical derivative disagree.
        centers = torch.tensor([[H / 2.0 + 0.3, W / 2.0 + 0.3], [2.0 * H / 3.0, W / 2.0 + 0.3]], device=device).view(
            1, 2, 2
        )
        scales = torch.tensor([(H + W) / 5.0, (H + W) / 6.0], device=device).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)

        class _MeanPatch(nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(2, 3))

        lafdesc = LAFDescriptor(_MeanPatch(), PS)
        self.gradcheck(lafdesc, (img, lafs), eps=1e-3, atol=1e-3, nondet_tol=1e-3)


class _TrainModeProbe(nn.Module):
    """Patch descriptor with dropout and a parameter, so mode and dtype changes are observable."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16 * 16, 5)
        self.drop = nn.Dropout(0.5)

    def forward(self, patches):
        return self.drop(self.proj(patches.flatten(1)))


class TestDescriptorModuleIsNotModified(BaseTester):
    @staticmethod
    def _inputs(device, dtype):
        img = torch.rand(1, 1, 48, 64, device=device, dtype=dtype)
        centers = torch.tensor([[20.0, 30.0], [30.0, 40.0]], device=device, dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([10.0, 8.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        return img, kornia.feature.laf_from_center_scale_ori(centers, scales, ori)

    @staticmethod
    def _state(module):
        return (
            [m.training for m in module.modules()],
            [(p.dtype, p.device) for p in module.parameters()],
            list(module.parameters()),
            [p.data_ptr() for p in module.parameters()],
        )

    def _assert_state_equal(self, before, after):
        assert before[0] == after[0]
        assert before[1] == after[1]
        assert all(a is b for a, b in zip(before[2], after[2]))
        assert before[3] == after[3]

    @staticmethod
    def _reference(module, img, lafs, ps):
        # what the descriptor gives when it is cast to the image and put in eval mode explicitly
        ref = copy.deepcopy(module).to(img).eval()
        patches = extract_patches_from_pyramid(img, lafs, ps)
        B, N, C, H, W = patches.shape
        return ref(patches.view(B * N, C, H, W)).view(B, N, -1)

    @pytest.mark.parametrize("train", [True, False])
    def test_module_is_unchanged_and_output_is_the_eval_cast_copy_5724(self, device, train):
        if torch.device(device).type == "mps":
            pytest.skip("MPS does not support float64")
        img, lafs = self._inputs(device, torch.float64)
        module = _TrainModeProbe().to(device).train(train)  # float32 module, float64 image
        before = self._state(module)
        expected = self._reference(module, img, lafs, 16)
        out = get_laf_descriptors(img, lafs, module, 16)
        self._assert_state_equal(before, self._state(module))
        assert out.dtype == torch.float64
        self.assert_close(out, expected)

    def test_lafdescriptor_keeps_every_submodule_mode_5724(self, device):
        img, lafs = self._inputs(device, torch.float32)
        module = _TrainModeProbe().to(device).train()
        module.drop.eval()  # mixed modes are restored one by one
        before = self._state(module)
        out = LAFDescriptor(module, 16)(img, lafs)
        self._assert_state_equal(before, self._state(module))
        self.assert_close(out, self._reference(module, img, lafs, 16))

    def test_float64_module_with_a_float32_image_5724(self, device):
        if torch.device(device).type == "mps":
            pytest.skip("MPS does not support float64")
        img, lafs = self._inputs(device, torch.float32)
        module = _TrainModeProbe().to(device, torch.float64).train()
        before = self._state(module)
        out = get_laf_descriptors(img, lafs, module, 16)
        self._assert_state_equal(before, self._state(module))
        assert module.proj.weight.dtype == torch.float64
        assert out.dtype == torch.float32
        self.assert_close(out, self._reference(module, img, lafs, 16))

    def test_gradients_reach_the_module_parameters_5724(self, device):
        if torch.device(device).type == "mps":
            pytest.skip("MPS does not support float64")
        img, lafs = self._inputs(device, torch.float64)
        module = _TrainModeProbe().to(device).eval()  # float32 parameters, float64 image
        get_laf_descriptors(img, lafs, module, 16).sum().backward()
        for p in module.parameters():
            assert p.grad is not None
            assert p.grad.dtype == torch.float32
            assert torch.isfinite(p.grad).all()
        assert module.proj.weight.grad.abs().sum() > 0

    @pytest.mark.parametrize("kind", ["script_hardnet", "trace_sift"])
    @pytest.mark.parametrize("image_dtype", [torch.float32, torch.float64])
    def test_torchscript_descriptors_5724(self, device, kind, image_dtype):
        if torch.device(device).type == "mps" and image_dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        img, lafs = self._inputs(device, image_dtype)
        patch = torch.rand(2, 1, 32, 32, device=device)
        if kind == "script_hardnet":
            module = torch.jit.script(HardNet(False).to(device).train())
        else:
            module = torch.jit.trace(SIFTDescriptor(32).to(device), patch)
        before = self._state(module)
        if kind == "trace_sift" and image_dtype == torch.float64:
            # the trace holds float32 constants, so a cast copy fails exactly as the module cast in place did
            with pytest.raises(RuntimeError, match="Double but found Float"):
                get_laf_descriptors(img, lafs, module, 32)
            self._assert_state_equal(before, self._state(module))
            return
        out = get_laf_descriptors(img, lafs, module, 32)
        self._assert_state_equal(before, self._state(module))
        reference = copy.deepcopy(module).to(img).eval()
        patches = extract_patches_from_pyramid(img, lafs, 32)
        B, N, C, H, W = patches.shape
        self.assert_close(out, reference(patches.view(B * N, C, H, W)).view(B, N, -1))
        assert out.dtype == image_dtype

    def test_empty_lafs_leave_the_module_alone_5724(self, device):
        if torch.device(device).type == "mps":
            pytest.skip("MPS does not support float64")
        img, _ = self._inputs(device, torch.float32)
        module = _TrainModeProbe().to(device, torch.float64).train()
        before = self._state(module)
        with pytest.warns(UserWarning, match="no keypoints"):
            out = get_laf_descriptors(img, torch.zeros(1, 0, 2, 3, device=device), module, 16)
        self._assert_state_equal(before, self._state(module))
        assert out.shape == (1, 0, 5)
        assert out.dtype == torch.float32

    def test_torchscript_gradients_reach_the_module_5724(self, device):
        # a float32 CPU module: on the CPU the image is float64 (a dtype cast), elsewhere the module is on another
        # device than the image (a device cast)
        image_dtype = torch.float64 if torch.device(device).type == "cpu" else torch.float32
        img, lafs = self._inputs(device, image_dtype)
        plain = HardNet(False)
        scripted = torch.jit.script(copy.deepcopy(plain))
        before = self._state(scripted)
        get_laf_descriptors(img, lafs, plain, 32).sum().backward()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_laf_descriptors(img, lafs, scripted, 32).sum().backward()
        assert not [w for w in caught if "not a leaf Tensor" in str(w.message)]
        self._assert_state_equal(before, self._state(scripted))
        for p, q in zip(scripted.parameters(), plain.parameters()):
            assert p.grad is not None
            assert p.grad.dtype == torch.float32
            assert p.grad.device == torch.device("cpu")
            self.assert_close(p.grad, q.grad)

    def test_dynamo_casts_a_module_of_another_dtype_5724(self, device, torch_optimizer):
        if torch.device(device).type == "mps":
            pytest.skip("MPS does not support float64")
        img, lafs = self._inputs(device, torch.float64)
        module = _TrainModeProbe().to(device).train()  # float32 module, float64 image
        before = self._state(module)
        expected = self._reference(module, img, lafs, 16)
        out = torch_optimizer(LAFDescriptor(module, 16))(img, lafs)
        self._assert_state_equal(before, self._state(module))
        self.assert_close(out, expected)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_export_casts_a_module_of_another_dtype_5724(self, device):
        if torch.device(device).type == "mps":
            pytest.skip("MPS does not support float64")
        img, lafs = self._inputs(device, torch.float64)
        module = _TrainModeProbe().to(device)  # float32 module, float64 image
        expected = self._reference(module, img, lafs, 16)
        # strict export cannot capture module.eval() on torch 2.5 and 2.6
        exported = torch.export.export(LAFDescriptor(module, 16), (img, lafs), strict=False)
        self.assert_close(exported.module()(img, lafs), expected)


class TestLocalFeature(BaseTester):
    def test_int_scaling_coef_5724(self, device, dtype):
        img = torch.rand(1, 1, 48, 64, device=device, dtype=dtype)
        det = ScaleSpaceDetector(10)
        desc = LAFDescriptor(SIFTDescriptor(32), 32)
        as_int = LocalFeature(det, desc, 2).to(device, dtype)
        as_float = LocalFeature(det, desc, 2.0).to(device, dtype)
        assert isinstance(as_int.scaling_coef, float)
        lafs_int, _, descs_int = as_int(img)
        lafs_float, _, descs_float = as_float(img)
        self.assert_close(lafs_int, lafs_float)
        self.assert_close(descs_int, descs_float)

    @pytest.mark.parametrize("coef", [0, 0.0, -1, -0.5])
    def test_non_positive_scaling_coef_message_5724(self, coef):
        with pytest.raises(ValueError, match="should be > 0"):
            LocalFeature(ScaleSpaceDetector(10), SIFTDescriptor(32), coef)

    def test_smoke(self, device, dtype):
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(32)
        local_feature = LocalFeature(det, desc).to(device, dtype)
        assert local_feature is not None

    def test_same(self, device, dtype):
        B, C, H, W = 1, 1, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(PS)
        local_feature = LocalFeature(det, LAFDescriptor(desc, PS)).to(device, dtype)
        lafs, responses, descs = local_feature(img)
        lafs1, responses1 = det(img)
        self.assert_close(lafs, lafs1)
        self.assert_close(responses, responses1)
        patches = extract_patches_from_pyramid(img, lafs1, PS)
        B1, N1, CH1, H1, W1 = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs1 = desc(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        self.assert_close(descs, descs1)

    def test_scale(self, device, dtype):
        B, C, H, W = 1, 1, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(PS)
        local_feature = LocalFeature(det, LAFDescriptor(desc, PS), 1.0).to(device, dtype)
        local_feature2 = LocalFeature(det, LAFDescriptor(desc, PS), 2.0).to(device, dtype)
        lafs, _responses, _descs = local_feature(img)
        lafs2, _responses2, _descs2 = local_feature2(img)
        self.assert_close(get_laf_center(lafs), get_laf_center(lafs2))
        self.assert_close(get_laf_orientation(lafs), get_laf_orientation(lafs2))
        self.assert_close(2.0 * get_laf_scale(lafs), get_laf_scale(lafs2))

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64)
        local_feature = LocalFeature(ScaleSpaceDetector(2), LAFDescriptor(SIFTDescriptor(PS), PS)).to(device, img.dtype)
        self.gradcheck(local_feature, img, eps=1e-4, atol=1e-4, nondet_tol=1e-8)


class TestSIFTFeature(BaseTester):
    # The real test is in TestLocalFeatureMatcher
    def test_smoke(self, device, dtype):
        sift = SIFTFeature()
        assert sift is not None

    def test_pyramid_sift_opt_in(self, device, dtype):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        lafs, responses, descriptors = SIFTFeature(2, descriptor_backend="pyramid").to(device, dtype)(image)
        assert lafs.shape == (1, 2, 2, 3)
        assert responses.shape == (1, 2)
        assert descriptors.shape == (1, 2, 128)

    @staticmethod
    def _blobs(h, w, blobs, device, dtype, background=0.5):
        # Gaussian blobs (x, y, sigma, amplitude) on a constant background, built in float64 on the CPU.
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
        )
        img = torch.full_like(xx, background)
        for x0, y0, s, a in blobs:
            img = img + a * torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s))
        return img[None, None].to(device=device, dtype=dtype)

    def test_detects_bright_and_dark_blobs_5743(self, device, dtype):
        # The single-scale DoG is negative at a bright blob and positive at a dark one. Like Lowe's, OpenCV's and
        # VLFeat's DoG detectors, SIFTFeature keeps the minima of the DoG as well as its maxima.
        img = self._blobs(120, 160, [(70.0, 52.0, 3.0, 0.4), (40.0, 85.0, 3.0, -0.4)], device, dtype)
        with torch.no_grad():
            lafs, _, _ = SIFTFeature(8, upright=True).to(device, dtype)(img)
        centres = get_laf_center(lafs)[0, kornia.feature.laf_is_filled(lafs)[0]].cpu().double()
        for name, x0, y0 in (("bright", 70.0, 52.0), ("dark", 40.0, 85.0)):
            dist = (centres - torch.tensor([x0, y0], dtype=torch.float64)).norm(dim=-1)
            assert bool((dist <= 3.0).any()), f"no detection at the {name} blob: {centres.tolist()}"

    def test_responses_are_the_dog_magnitude_5743(self, device, dtype):
        # On a single full-resolution level a keypoint sits on a pixel, and its response is the magnitude of the
        # DoG there, for a bright blob (DoG < 0) and a dark one (DoG > 0) alike.
        cfg = get_default_detector_config()
        cfg.update(pyramid_levels=0, up_levels=0)
        img = self._blobs(96, 128, [(40.0, 34.0, 2.5, 0.4), (86.0, 60.0, 3.0, -0.3)], device, dtype)
        with torch.no_grad():
            lafs, responses, _ = SIFTFeature(2, upright=True, config=cfg).to(device, dtype)(img)
            dog = kornia.feature.BlobDoGSingle(1.0, 1.6).to(device, dtype)(img)[0, 0]
        xy = get_laf_center(lafs)[0].round().long()
        assert xy.tolist() == [[40, 34], [86, 60]], xy.tolist()
        signed = dog[xy[:, 1], xy[:, 0]]
        assert signed[0] < 0 < signed[1]
        self.assert_close(responses[0], signed.abs())

    def test_blob_is_detected_without_its_ring_5743(self, device, dtype):
        # The DoG of a blob is surrounded by a weaker ring of the opposite sign. The ring lies within the blob's
        # suppression window and is weaker, so every detection of note is at the blob, whichever its polarity.
        for name, amplitude, background in (("bright", 1.0, 0.0), ("dark", -1.0, 1.0)):
            img = self._blobs(120, 160, [(70.0, 52.0, 3.0, amplitude)], device, dtype, background)
            with torch.no_grad():
                lafs, responses, _ = SIFTFeature(32, upright=True).to(device, dtype)(img)
            # A flat background still has extrema of rounding-error size; they are not features.
            strong = responses[0] > 0.01 * responses[0].max()
            centres = get_laf_center(lafs)[0, strong].cpu().double()
            dist = (centres - torch.tensor([70.0, 52.0], dtype=torch.float64)).norm(dim=-1)
            assert dist.numel() > 0, name
            assert bool((dist <= 3.0).all()), f"{name} blob: detections at distances {dist.tolist()}"

    def test_frames_do_not_depend_on_polarity_5743(self, device, dtype):
        # DoG(1 - x) = -DoG(x): an image and its negative have the same extrema with the same magnitudes, so
        # upright SIFTFeature finds the same frames in both. Every pixel lies in [0.5, 1], where 1 - x is exact.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the DoG is computed in the input dtype, which rounds an image and its negative differently")
        blobs = [(34.0, 30.0, 3.0, 0.2), (88.0, 42.0, 4.0, -0.2), (58.0, 66.0, 2.5, -0.15), (100.0, 70.0, 3.5, 0.12)]
        img = self._blobs(96, 128, blobs, device, dtype, background=0.75)
        feature = SIFTFeature(8, upright=True).to(device, dtype)
        with torch.no_grad():
            lafs, responses, _ = feature(img)
            lafs_neg, responses_neg, _ = feature(1 - img)
        assert bool(kornia.feature.laf_is_filled(lafs).all())
        self.assert_close(lafs_neg, lafs)
        self.assert_close(responses_neg, responses)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64)
        local_feature = SIFTFeature(2, True).to(device)
        self.gradcheck(local_feature, img, eps=1e-4, atol=1e-4, fast_mode=False)


class TestSIFTFeatureHalfPrecision(BaseTester):
    def test_lafs_and_descriptors_are_finite(self, device):
        # The orienter's unguarded `sqrt`/`atan2` gave a NaN LAF at a *filled* slot, which the
        # padding mask cannot cover, and 128 NaN descriptor values in that row with it.
        if device.type == "mps":
            pytest.skip("MPS autocast changes the effective dtype")
        if not supports_replicate_padding(device, torch.float16):
            pytest.skip("no replicate-padding kernel for float16 on this device")
        torch.manual_seed(0)
        img = torch.rand(1, 1, 64, 64, device=device, dtype=torch.float16)
        with torch.no_grad():
            lafs, _, descs = SIFTFeature(50).to(device, torch.float16)(img)
        assert lafs.dtype == torch.float32  # pixel coordinates are float32 for a half-precision image
        assert descs.dtype == torch.float16
        assert int(lafs[0].ne(0).any(dim=-1).any(dim=-1).sum()) > 0
        assert torch.isfinite(lafs).all()
        assert torch.isfinite(descs).all()


class TestKeyNetHardNetFeature(BaseTester):
    # The real test is in TestLocalFeatureMatcher
    def test_smoke(self, device, dtype):
        sift = KeyNetHardNet(2).to(device, dtype)
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        out = sift(img)
        assert out is not None

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64)
        local_feature = KeyNetHardNet(2, True).to(device).to(device)
        self.gradcheck(local_feature, img, eps=1e-4, atol=1e-4, fast_mode=False)


class TestGFTTAffNetHardNet(BaseTester):
    # The real test is in TestLocalFeatureMatcher
    def test_smoke(self, device, dtype):
        feat = GFTTAffNetHardNet().to(device, dtype)
        assert feat is not None

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64)
        local_feature = GFTTAffNetHardNet(2, True).to(device, img.dtype)
        self.gradcheck(local_feature, img, eps=1e-4, atol=1e-4, fast_mode=False)


class TestDoGHardNet(BaseTester):
    @pytest.fixture(autouse=True)
    def _offline_networks(self, monkeypatch):
        # The presets load pretrained HardNet and AffNet weights. The frames do not depend on HardNet's weights, so a
        # random HardNet keeps these tests offline. The AffNet stand-in gives every patch one anisotropic shape,
        # tanh of the head bias, so that a preset which skipped AffNet would return similarity frames instead.
        from kornia.feature import HardNet, LAFAffNetShapeEstimator

        def affnet(pretrained=False, preserve_orientation=True):
            module = LAFAffNetShapeEstimator(False, preserve_orientation)
            head = module.features[-3]
            with torch.no_grad():
                head.weight.zero_()
                head.bias.copy_(torch.tensor([0.5, 0.3, -0.3]))
            return module

        monkeypatch.setattr("kornia.feature.integrated.HardNet", lambda pretrained=False: HardNet(False))
        monkeypatch.setattr("kornia.feature.integrated.LAFAffNetShapeEstimator", affnet)

    @staticmethod
    def _image(device, dtype):
        # Blurred noise has blobs at several scales; at 72 x 96 its DoG detections fill the 16 slots below in every
        # dtype. The noise is drawn in float32 from a seeded CPU generator, because a draw made in half precision
        # differs between torch releases, and the blurred image is then cast to the tested device and dtype.
        noise = torch.rand(1, 1, 72, 96, generator=torch.Generator().manual_seed(0))
        return kornia.filters.gaussian_blur2d(noise, (7, 7), (1.5, 1.5)).to(device, dtype)

    @staticmethod
    def _dog_frames(image, num_features):
        """Return the 6-sigma DoG frames and responses of :class:`SIFTFeatureScaleSpace`'s patch backend."""
        feature = kornia.feature.SIFTFeatureScaleSpace(num_features, descriptor_backend="patch")
        lafs, responses = feature.detector.to(image.device, image.dtype)(image)
        assert int(laf_is_filled(lafs).sum()) >= num_features // 2
        return lafs, responses

    @pytest.mark.parametrize("preset", ["DoGHardNet", "DoGAffNetHardNet"])
    def test_shapes_dtype_and_device(self, device, dtype, preset):
        image = torch.rand(2, 1, 40, 56, device=device, dtype=dtype)
        feature = getattr(kornia.feature, preset)(8, device=device).to(dtype=dtype)
        lafs, responses, descriptors = feature(image)
        assert lafs.shape == (2, 8, 2, 3)
        assert responses.shape == (2, 8)
        assert descriptors.shape == (2, 8, 128)
        assert lafs.dtype == torch.promote_types(dtype, torch.float32)  # float32 coordinates for a half image
        assert responses.dtype == descriptors.dtype == dtype
        for out in (lafs, responses, descriptors):
            assert out.device == image.device

    def test_lafs_are_twice_the_dog_frames(self, device, dtype):
        image = self._image(device, dtype)
        feature = kornia.feature.DoGHardNet(16).to(device, dtype)
        lafs, responses, descriptors = feature(image)
        dog_lafs, dog_responses = self._dog_frames(image, 16)
        self.assert_close(responses, dog_responses)
        self.assert_close(get_laf_center(lafs), get_laf_center(dog_lafs))
        # Orientations are estimated on the 6-sigma detections; only the returned frames are twice as large.
        self.assert_close(get_laf_orientation(lafs), get_laf_orientation(dog_lafs))
        self.assert_close(get_laf_scale(lafs), 2.0 * get_laf_scale(dog_lafs))
        self.assert_close(descriptors, feature.descriptor(image, lafs))

    @pytest.mark.parametrize("preset", ["DoGHardNet", "DoGAffNetHardNet"])
    def test_upright_frames(self, device, dtype, preset):
        image = self._image(device, dtype)
        lafs, _, _ = getattr(kornia.feature, preset)(16, upright=True).to(device, dtype)(image)
        dog_lafs, _ = self._dog_frames(image, 16)
        self.assert_close(get_laf_center(lafs), get_laf_center(dog_lafs))
        self.assert_close(get_laf_orientation(lafs), torch.zeros_like(get_laf_orientation(lafs)))
        affnet = preset == "DoGAffNetHardNet"
        self.assert_close(get_laf_scale(lafs), 2.0 * get_laf_scale(dog_lafs), low_tolerance=affnet)

    def test_affnet_frames_keep_the_dog_centres_and_scale(self, device, dtype):
        image = self._image(device, dtype)
        lafs, responses, _ = kornia.feature.DoGAffNetHardNet(16).to(device, dtype)(image)
        dog_lafs, dog_responses = self._dog_frames(image, 16)
        self.assert_close(responses, dog_responses)
        self.assert_close(get_laf_center(lafs), get_laf_center(dog_lafs))
        # AffNet's shape is rescaled back to the detection scale through its determinant, which rounds in half
        # precision.
        self.assert_close(get_laf_scale(lafs), 2.0 * get_laf_scale(dog_lafs), low_tolerance=True)
        filled = laf_is_filled(lafs)
        a = lafs[filled][:, :, :2].float()
        assert bool((a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0] > 0).all())
        # A scaled rotation has a11 == a22 and a12 == -a21; the stand-in's shape is far from one.
        anisotropy = (a[:, 0, 0] - a[:, 1, 1]).abs() + (a[:, 0, 1] + a[:, 1, 0]).abs()
        assert bool((anisotropy > 0.1 * get_laf_scale(lafs)[filled].float().reshape(-1)).all())

    @pytest.mark.parametrize(
        "preset, selection, wrapped_type",
        [("DoGHardNet", "resp", "BlobDoG"), ("DoGAffNetHardNet", "aff", "LAFAffNetShapeEstimator")],
    )
    def test_detector_receives_the_module_selection(self, monkeypatch, preset, selection, wrapped_type):
        # Records what the detector would wrap instead of compiling it, so the test needs no Dynamo.
        wrapped = []

        def record(module, **kwargs):
            wrapped.append(module)
            return module

        monkeypatch.setattr(torch, "compile", record)
        feature = getattr(kornia.feature, preset)(4, compile_modules=[selection])
        assert [type(module).__name__ for module in wrapped] == [wrapped_type]
        assert wrapped[0] is getattr(feature.detector, selection)

    @pytest.mark.parametrize("preset", ["DoGHardNet", "DoGAffNetHardNet"])
    def test_color_image_raises(self, device, preset):
        feature = getattr(kornia.feature, preset)(4, device=device)
        with pytest.raises(BaseError, match="convert it to grayscale"):
            feature(torch.rand(1, 3, 40, 56, device=device))

    @pytest.mark.parametrize(
        "preset, feature_name", [("DoGHardNet", "doghardnet"), ("DoGAffNetHardNet", "dog_affnet_hardnet")]
    )
    def test_lightglue_matcher_receives_the_keypoint_size(self, device, dtype, monkeypatch, preset, feature_name):
        from kornia.feature import LightGlue, LightGlueMatcher

        scale_coef = LightGlue.features[feature_name]["scale_coef"]
        seen = {}

        class _RecordingLightGlue(nn.Module):
            # Stands in for the network, whose construction downloads weights; the matcher only prepares its inputs.
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, data):
                seen.update(data)
                matches = torch.full((1, data["image0"]["keypoints"].shape[1]), -1, device=device)
                return {"matches0": matches, "matching_scores0": torch.zeros_like(matches, dtype=torch.float32)}

        monkeypatch.setattr("kornia.feature.integrated.LightGlue", _RecordingLightGlue)
        image = self._image(device, dtype)
        lafs, _, descriptors = getattr(kornia.feature, preset)(16).to(device, dtype)(image)
        dog_lafs, _ = self._dog_frames(image, 16)
        valid = laf_is_filled(lafs)[0]
        lafs, descriptors, dog_lafs = lafs[:, valid], descriptors[0, valid], dog_lafs[:, valid]
        LightGlueMatcher(feature_name)(descriptors, descriptors, lafs, lafs)
        # The network takes ``scale_coef`` times the scale it is given. For these frames that is the OpenCV
        # ``KeyPoint.size`` the weights were trained with: 2 sigma, a third of the 6-sigma DoG frame.
        expected = get_laf_scale(dog_lafs).reshape(1, -1) / 3.0
        self.assert_close(seen["image0"]["scales"] * scale_coef, expected, low_tolerance=preset == "DoGAffNetHardNet")


class TestLocalFeatureMatcher(BaseTester):
    def test_smoke(self, device):
        matcher = LocalFeatureMatcher(SIFTFeature(5), DescriptorMatcher("snn", 0.8)).to(device)
        assert matcher is not None

    @pytest.mark.parametrize("match_type", ["nn", "mnn"])
    def test_detector_padding_is_not_matched(self, device, dtype, match_type):
        # A fixed-shape detector with no maxima returns zero LAFs. Their descriptors are finite,
        # but NN/MNN would otherwise report them as false correspondences at the image origin.
        feat = SIFTFeature(8, upright=True, score_threshold=1e6).to(device, dtype)
        matcher = LocalFeatureMatcher(feat, DescriptorMatcher(match_type, 0.8)).to(device, dtype)
        image0 = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        image1 = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        with torch.no_grad():
            out = matcher({"image0": image0, "image1": image1})
        assert out["keypoints0"].shape == (0, 2)
        assert out["keypoints1"].shape == (0, 2)
        assert out["lafs0"].shape == (1, 0, 2, 3)
        assert out["lafs1"].shape == (1, 0, 2, 3)

    def test_laf_and_descriptor_counts_must_agree(self, device, dtype):
        # The padding mask is derived from the LAFs and applied to the descriptors, so a count
        # mismatch used to surface as an opaque boolean-index `IndexError` deep in the loop.
        matcher = LocalFeatureMatcher(SIFTFeature(5), DescriptorMatcher("nn", 0.8)).to(device, dtype)
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        lafs = torch.rand(1, 3, 2, 3, device=device, dtype=dtype)
        descs = torch.rand(1, 5, 128, device=device, dtype=dtype)
        images = {"image0": img, "image1": img}
        data = {**images, "lafs0": lafs, "descriptors0": descs, "lafs1": lafs, "descriptors1": descs[:, :3]}
        with pytest.raises(Exception, match=r"lafs0 and descriptors0"):
            matcher(data)
        data = {**images, "lafs0": lafs, "descriptors0": descs[:, :3], "lafs1": lafs, "descriptors1": descs}
        with pytest.raises(Exception, match=r"lafs1 and descriptors1"):
            matcher(data)

    def test_masks_are_forwarded_to_the_detector(self, device, dtype):
        feat = SIFTFeature(8, upright=True).to(device, dtype)
        matcher = LocalFeatureMatcher(feat, DescriptorMatcher("nn", 0.8)).to(device, dtype)
        image0 = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        image1 = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        # LocalFeatureMatcher historically documents masks without a channel dimension; it adds
        # that singleton axis before forwarding to the detector's (B, 1, H, W) mask contract.
        mask = torch.zeros(1, 64, 64, device=device, dtype=torch.bool)
        with torch.no_grad():
            out = matcher({"image0": image0, "image1": image1, "mask0": mask, "mask1": mask})
        assert out["keypoints0"].shape == (0, 2)
        assert out["keypoints1"].shape == (0, 2)

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_nomatch(self, device, dtype, data):
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(100), DescriptorMatcher("snn", 0.8)).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            out = matcher({"image0": data_dev["image0"], "image1": 0 * data_dev["image0"]})
        assert len(out["keypoints0"]) == 0

    @pytest.mark.skip("Takes too long time (but works)")
    def test_gradcheck(self, device):
        matcher = LocalFeatureMatcher(SIFTFeature(5), DescriptorMatcher("nn", 1.0)).to(device)
        patches = torch.rand(1, 1, 32, 32, device=device, dtype=torch.float64)
        patches05 = resize(patches, (48, 48))

        def proxy_forward(x, y):
            return matcher({"image0": x, "image1": y})["keypoints0"]

        self.gradcheck(proxy_forward, (patches, patches05), eps=1e-4, atol=1e-4)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_sift(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(SIFTFeature(1000), DescriptorMatcher("snn", 0.8)).to(device, dtype)
        ransac = RANSAC("homography", 1.0, 1024, 5).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        with torch.no_grad():
            out = matcher(data_dev)
        homography, inliers = ransac(out["keypoints0"], out["keypoints1"])
        assert inliers.sum().item() > 50  # we have enough inliers
        # Reprojection error of 5px is OK
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_sift_preextract(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        feat = SIFTFeature(1000).to(device, dtype)
        matcher = LocalFeatureMatcher(feat, DescriptorMatcher("snn", 0.8)).to(device)
        ransac = RANSAC("homography", 1.0, 1024, 5).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]

        lafs, _, descs = feat(data_dev["image0"])
        data_dev["lafs0"] = lafs
        data_dev["descriptors0"] = descs

        lafs2, _, descs2 = feat(data_dev["image1"])
        data_dev["lafs1"] = lafs2
        data_dev["descriptors1"] = descs2

        with torch.no_grad():
            out = matcher(data_dev)
        homography, inliers = ransac(out["keypoints0"], out["keypoints1"])
        assert inliers.sum().item() > 50  # we have enough inliers
        # Reprojection error of 5px is OK
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_gftt(self, device, dtype, data):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(1000), DescriptorMatcher("snn", 0.8)).to(device, dtype)
        ransac = RANSAC("homography", 1.0, 1024, 5).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        with torch.no_grad():
            torch.manual_seed(0)
            out = matcher(data_dev)
        homography, inliers = ransac(out["keypoints0"], out["keypoints1"])
        assert inliers.sum().item() > 50  # we have enough inliers
        # Reprojection error of 5px is OK
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_keynet(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(KeyNetHardNet(500), DescriptorMatcher("snn", 0.9)).to(device, dtype)
        ransac = RANSAC("homography", 1.0, 1024, 5).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        with torch.no_grad():
            out = matcher(data_dev)
        homography, inliers = ransac(out["keypoints0"], out["keypoints1"])
        assert inliers.sum().item() > 50  # we have enough inliers
        # Reprojection error of 5px is OK
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)

    @pytest.mark.skip("ScaleSpaceDetector now is not jittable")
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        patches2x = resize(patches, (48, 48))
        inputs = {"image0": patches, "image1": patches2x}
        model = LocalFeatureMatcher(SIFTDescriptor(32), DescriptorMatcher("snn", 0.8)).to(device).eval()
        model_jit = torch.jit.script(model)

        out = model(inputs)
        out_jit = model_jit(inputs)
        for k, v in out.items():
            self.assert_close(v, out_jit[k])


class TestSIFTPyramidBackend(BaseTester):
    def test_sift_pyramid_does_not_extract_image_patches(self, device, dtype, monkeypatch):
        def unexpected_patch(*args, **kwargs):
            raise AssertionError("The pyramid backend must operate on shared maps without image patches")

        monkeypatch.setattr(kornia.feature.LAFOrienter, "forward", unexpected_patch)
        monkeypatch.setattr(kornia.feature.LAFDescriptor, "forward", unexpected_patch)
        original_sample = kornia.feature.SIFTDescriptorFromPyramid._sample
        sampled_channels = []

        def sample_maps(maps, *args):
            assert maps.shape[1] != 1, "Sparse sampling must not extract patches from the grayscale image"
            sampled_channels.append(maps.shape[1])
            return original_sample(maps, *args)

        monkeypatch.setattr(kornia.feature.SIFTDescriptorFromPyramid, "_sample", staticmethod(sample_maps))
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        feature = SIFTFeature(3, descriptor_backend="pyramid").to(device, dtype).eval()
        lafs, _, descriptors = feature(image)
        assert lafs.shape == (1, 3, 2, 3)
        assert torch.isfinite(descriptors).all()
        assert set(sampled_channels) == {2, feature.descriptor._descriptor_bins}

    @pytest.mark.parametrize("preset", [kornia.feature.SIFTFeature, kornia.feature.SIFTFeatureScaleSpace])
    @pytest.mark.parametrize("upright", [False, True])
    def test_sparse_output_and_real_descriptor(self, device, dtype, preset, upright):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        feature = preset(num_features=3, upright=upright, descriptor_backend="pyramid").to(device, dtype).eval()
        if preset is kornia.feature.SIFTFeature:
            assert isinstance(feature.descriptor, kornia.feature.SIFTDescriptorFromPyramid)
        else:
            from kornia.feature.sift.scale_space import _SIFTScaleSpaceDescriptor

            assert isinstance(feature.descriptor, _SIFTScaleSpaceDescriptor)
        lafs, responses, descriptors = feature(image)
        assert lafs.shape == (1, 3, 2, 3)
        assert responses.shape == (1, 3)
        assert descriptors.shape == (1, 3, 128)
        assert torch.isfinite(descriptors).all()
        if preset is kornia.feature.SIFTFeature:
            self.assert_close(descriptors, feature.descriptor(image, lafs))

    @pytest.mark.parametrize("preset", [kornia.feature.SIFTFeature])
    def test_detector_responses_and_centers_are_unchanged(self, device, dtype, preset):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        patch = preset(num_features=3, upright=True).to(device, dtype).eval()
        pyramid = preset(num_features=3, descriptor_backend="pyramid").to(device, dtype).eval()
        patch_lafs, patch_responses, _ = patch(image)
        pyramid_lafs, pyramid_responses, _ = pyramid(image)
        self.assert_close(get_laf_center(patch_lafs), get_laf_center(pyramid_lafs))
        self.assert_close(patch_responses, pyramid_responses)

    @pytest.mark.parametrize("preset", [kornia.feature.SIFTFeature, kornia.feature.SIFTFeatureScaleSpace])
    def test_invalid_backend(self, preset):
        with pytest.raises(ValueError, match="descriptor backend"):
            preset(descriptor_backend="unknown")

    def test_generic_local_feature_preserves_frames_and_forwards_mask(self, device, dtype):
        image = torch.rand(1, 1, 40, 40, device=device, dtype=dtype)
        lafs = torch.tensor([[[[6.0, 2.0, 20.0], [-2.0, 6.0, 20.0]]]], device=device, dtype=dtype)
        mask = torch.ones_like(image)

        class Detector(nn.Module):
            def forward(self, image, received_mask):
                assert received_mask is mask
                return lafs, image.new_ones(1, 1)

        descriptor = kornia.feature.SIFTDescriptorFromPyramid().to(device, dtype)
        feature = LocalFeature(Detector(), descriptor)
        returned, responses, descriptors = feature(image, mask)
        self.assert_close(returned, lafs)
        self.assert_close(responses, image.new_ones(1, 1))
        self.assert_close(descriptors, descriptor(image, lafs))


class TestLightGlueKeypointConventions(BaseTester):
    def _lafs(self, device, dtype, orientations):
        centers = torch.zeros(1, len(orientations), 2, device=device, dtype=dtype)
        scales = torch.full((1, len(orientations), 1, 1), 12.0, device=device, dtype=dtype)
        angles = torch.tensor(orientations, device=device, dtype=dtype).view(1, -1, 1)
        return kornia.feature.laf_from_center_scale_ori(centers, scales, angles)

    def test_sift_uses_the_colmap_convention_of_its_training_keypoints(self, device, dtype):
        from kornia.feature.integrated import _lightglue_keypoint_scale_ori

        # kornia orientation +30 deg is COLMAP's -30 deg, -90 is +90, and 180 stays at pi in (-pi, pi]; the
        # 12-pixel LAF is a 6-sigma DoG frame, so sigma is 2.
        lafs = self._lafs(device, dtype, [30.0, -90.0, 180.0])
        scales, oris = _lightglue_keypoint_scale_ori(lafs, "sift")
        self.assert_close(scales, torch.full((1, 3), 2.0, device=device, dtype=dtype))
        expected = torch.tensor([[-torch.pi / 6, torch.pi / 2, torch.pi]], device=device, dtype=dtype)
        self.assert_close(oris, expected)

    def test_other_features_keep_the_laf_scale_and_orientation(self, device, dtype):
        from kornia.feature.integrated import _lightglue_keypoint_scale_ori

        lafs = self._lafs(device, dtype, [30.0, -90.0])
        scales, oris = _lightglue_keypoint_scale_ori(lafs, "doghardnet")
        self.assert_close(scales, torch.full((1, 2), 12.0, device=device, dtype=dtype))
        expected = torch.tensor([[torch.pi / 6, 3 * torch.pi / 2]], device=device, dtype=dtype)
        self.assert_close(oris, expected)


class TestSIFTDescriptorLayoutArgument(BaseTester):
    @pytest.mark.parametrize("preset", [kornia.feature.SIFTFeature, kornia.feature.SIFTFeatureScaleSpace])
    @pytest.mark.parametrize("backend", ["patch", "pyramid"])
    def test_opencv_layout_reorders_only_the_descriptors(self, device, dtype, preset, backend):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        default = preset(num_features=8, descriptor_backend=backend).to(device, dtype).eval()
        opencv = preset(num_features=8, descriptor_backend=backend, descriptor_layout="opencv").to(device, dtype).eval()
        assert default.descriptor_layout == "kornia"
        assert opencv.descriptor_layout == "opencv"
        lafs, responses, descriptors = default(image)
        opencv_lafs, opencv_responses, opencv_descriptors = opencv(image)
        self.assert_close(opencv_lafs, lafs)
        self.assert_close(opencv_responses, responses)
        self.assert_close(
            opencv_descriptors, kornia.feature.convert_sift_descriptor_layout(descriptors, "kornia", "opencv")
        )

    @pytest.mark.parametrize("preset", [kornia.feature.SIFTFeature, kornia.feature.SIFTFeatureScaleSpace])
    def test_invalid_layout(self, preset):
        with pytest.raises(ValueError, match="layout"):
            preset(descriptor_layout="vlfeat")


class TestConventionsSIFTScaleSpaceKeypointSize(BaseTester):
    @pytest.mark.parametrize("backend, backend_rtol", [("pyramid", 1e-3), ("patch", 2e-2)])
    def test_convention_sift_scale_space_frames_are_three_times_the_opencv_keypoint_size(
        self, device, dtype, backend, backend_rtol
    ):
        from testing.base import supports_reflect_padding

        if backend == "patch" and not supports_reflect_padding(device, dtype):
            pytest.skip("the patch backend's scale pyramid needs a reflection-padding kernel for this dtype")
        # Four Gaussian blobs (sigma 2.5, 3.5, 5 and 4.2 px) at non-integer centres x != y of a 150 x 230 uint8 image.
        # OpenCV 5.0.0 finds one keypoint on the sigma-5 blob, of size 2 sigma for its detection scale sigma:
        #   cv2.SIFT_create(enable_precise_upscale=True).detect(img_u8, None) -> size 8.871075630187988
        # (the same image built with numpy; pixel sum 97820). SIFTFeatureScaleSpace frames have scale 6 sigma, so its
        # frame on that blob has scale 3 * size. The sigma-5 blob is the one both backends find in every dtype.
        h, w, cx, cy = 150, 230, 186.25, 101.4
        blobs = [(41.3, 37.6, 2.5), (118.7, 44.2, 3.5), (cx, cy, 5.0), (60.5, 110.8, 4.2)]
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
        )
        img = sum(torch.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2.0 * s * s)) for bx, by, s in blobs)
        img = (torch.round(img.clamp(0.0, 1.0) * 255.0) / 255.0)[None, None].to(device, dtype)
        feature = kornia.feature.SIFTFeatureScaleSpace(64, descriptor_backend=backend).to(device, dtype).eval()
        with torch.no_grad():
            lafs, _, _ = feature(img)
        centres = kornia.feature.get_laf_center(lafs)[0].cpu().double()
        distance = (centres - torch.tensor([cx, cy], dtype=torch.float64)).norm(dim=-1)
        nearest = int(distance.argmin())
        assert distance[nearest] < 1.0
        scale = kornia.feature.get_laf_scale(lafs)[0, nearest, 0, 0].cpu().double()
        # the frame's own rounding on top of the backend's detection error
        rtol = backend_rtol + 2.0 * torch.finfo(dtype).eps
        self.assert_close(scale, torch.tensor(3.0 * 8.871075630187988, dtype=torch.float64), rtol=rtol, atol=0.0)


class _PatchMeanDescriptor(nn.Module):
    """A descriptor that does not normalise: the patch mean and a constant 1, so no patch maps to the zero vector."""

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        mean = patches.mean(dim=(1, 2, 3))
        return torch.stack([mean, torch.ones_like(mean)], dim=-1)


class TestConventionsLocalFeature(BaseTester):
    @staticmethod
    def _blobs(blobs, device, dtype, h=72, w=96):
        # Bright Gaussian blobs (x, y, sigma) on a black H != W image, built in float64 on the CPU, then cast.
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
        )
        img = sum(torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * s * s)) for x0, y0, s in blobs)
        return img[None, None].to(device, dtype)

    def test_convention_local_feature_pads_every_image_to_num_features_with_zero_frames(self, device, dtype):
        # Blobs at (30, 22) sigma 3 and (61, 47) sigma 4; the second image has the second blob only.
        # ScaleSpaceDetector(8) finds one frame per blob, so the images fill 2 and 1 of the 8 slots: N is the detector's
        # num_features for every image, not the largest count in the batch. The centres are checked to 0.1 px, and to
        # 0.25 px in half precision, half of the 0.5 px offset of an edge-aligned frame.
        if not (supports_reflect_padding(device, dtype) and supports_replicate_padding(device, dtype)):
            pytest.skip("ScaleSpaceDetector and SIFTDescriptor need reflection and replication padding for this dtype")
        img = torch.cat(
            [
                self._blobs([(30.0, 22.0, 3.0), (61.0, 47.0, 4.0)], device, dtype),
                self._blobs([(61.0, 47.0, 4.0)], device, dtype),
            ]
        )
        descriptor = LAFDescriptor(SIFTDescriptor(32), patch_size=32)
        feature = LocalFeature(ScaleSpaceDetector(8), descriptor).to(device, dtype)
        with torch.no_grad():
            lafs, responses, descs = feature(img)
            lafs_t, _, _ = feature(img.transpose(-1, -2).contiguous())
            zero_frame = descriptor(img, lafs.new_zeros(2, 1, 2, 3))
        assert lafs.shape == (2, 8, 2, 3)
        assert responses.shape == (2, 8)
        assert descs.shape == (2, 8, 128)
        filled = laf_is_filled(lafs)
        assert filled.sum(dim=-1).tolist() == [2, 1]
        tol = 0.25 if dtype in (torch.float16, torch.bfloat16) else 0.1
        blobs = torch.tensor([[30.0, 22.0], [61.0, 47.0]], dtype=torch.float64)
        centres = get_laf_center(lafs).cpu().double()
        self.assert_close(centres[0, filled[0].cpu()], blobs, rtol=0.0, atol=tol)
        self.assert_close(centres[1, filled[1].cpu()], blobs[1:], rtol=0.0, atol=tol)
        # relabelling: the transposed image gives the transposed centres
        centres_t = get_laf_center(lafs_t).cpu().double()[0, laf_is_filled(lafs_t)[0].cpu()]
        self.assert_close(centres_t, blobs.flip(-1), rtol=0.0, atol=tol)
        # an unfilled slot is a zero LAF with a zero response ...
        assert lafs[~filled].abs().max() == 0
        assert responses[~filled].abs().max() == 0
        # ... and every one of them carries the descriptor of a patch sampled at the origin, one vector per image.
        # SIFT maps that constant patch to almost zero, so a descriptor that does not normalise shows it is described.
        for b in range(2):
            assert (descs[b, ~filled[b]] == descs[b, ~filled[b]][:1]).all()
        self.assert_close(descs[0, ~filled[0]], zero_frame[0].expand(6, 128))
        self.assert_close(descs[1, ~filled[1]], zero_frame[1].expand(7, 128))
        # The same frames described on a ramp whose origin pixel differs from every other pixel and between the
        # images: a zero LAF samples every patch pixel at the origin, so the patch mean is ramp[b, 0, 0, 0].
        yy, xx = torch.meshgrid(torch.arange(72.0), torch.arange(96.0), indexing="ij")
        ramp = ((xx + 100.0 * yy) / 8000.0 + torch.tensor([0.25, 0.5]).view(2, 1, 1, 1)).to(device, dtype)
        mean_feature = LocalFeature(lambda im, mask=None: (lafs, responses), LAFDescriptor(_PatchMeanDescriptor(), 8))
        with torch.no_grad():
            _, _, mean_descs = mean_feature.to(device, dtype)(ramp)
        origin = torch.stack([ramp[:, 0, 0, 0], torch.ones_like(ramp[:, 0, 0, 0])], dim=-1)
        self.assert_close(mean_descs[0, ~filled[0]], origin[0].expand(6, 2))
        self.assert_close(mean_descs[1, ~filled[1]], origin[1].expand(7, 2))

    # preset: (detector, num_features, mr_size, response, orientation estimator and its patch size, affine shape,
    # descriptor, patch size). mr_size multiplies the detection scale into the LAF scale: sigma for ScaleSpaceDetector,
    # the pyramid level's pixel size for the multi-resolution detectors. The upright variants orient with PassLAF.
    _PRESETS = {
        "SIFTFeature": ("MultiResolutionDetector", 8000, 22.0, None, "PatchDominantGradientOrientation", 19, "PassLAF",
                        "SIFTDescriptor", 41),
        "SIFTFeatureScaleSpace": ("ScaleSpaceDetector", 8000, 6.0, "BlobDoG", "PatchDominantGradientOrientation", 19,
                                  "PassLAF", "SIFTDescriptor", 41),
        "GFTTAffNetHardNet": ("ScaleSpaceDetector", 8000, 6.0, "CornerGFTT", "PatchDominantGradientOrientation", 19,
                              "LAFAffNetShapeEstimator", "HardNet", 32),
        "HesAffNetHardNet": ("ScaleSpaceDetector", 2048, 6.0, "BlobHessian", "PatchDominantGradientOrientation", 19,
                             "LAFAffNetShapeEstimator", "HardNet", 32),
        "KeyNetHardNet": ("KeyNetDetector", 8000, 22.0, "KeyNet", "OriNet", 32, "PassLAF", "HardNet", 32),
        "KeyNetAffNetHardNet": ("KeyNetDetector", 8000, 22.0, "KeyNet", "OriNet", 32, "LAFAffNetShapeEstimator",
                                "HardNet", 32),
    }  # fmt: skip

    @pytest.mark.parametrize("preset", list(_PRESETS))
    def test_convention_presets_compose_their_documented_modules(self, monkeypatch, preset):
        # The presets load pretrained HardNet, AffNet, OriNet and KeyNet weights; random networks keep this offline.
        # SIFTFeature's response, the magnitude of BlobDoGSingle(1.0, 1.6), is checked by TestSIFTFeature.
        from kornia.feature import KeyNetDetector, LAFAffNetShapeEstimator, OriNet

        integrated = "kornia.feature.integrated"
        monkeypatch.setattr(f"{integrated}.HardNet", lambda pretrained=False: HardNet(False))
        monkeypatch.setattr(
            f"{integrated}.LAFAffNetShapeEstimator",
            lambda pretrained=False, preserve_orientation=True: LAFAffNetShapeEstimator(False, preserve_orientation),
        )
        monkeypatch.setattr(f"{integrated}.OriNet", lambda pretrained=False, eps=1e-8: OriNet(False))
        monkeypatch.setattr(f"{integrated}.KeyNetDetector", lambda pretrained=False, **kw: KeyNetDetector(False, **kw))
        spec = self._PRESETS[preset]
        detector, num_features, mr_size, response, orienter, orienter_size, affine, descriptor, patch_size = spec
        cls = getattr(kornia.feature, preset)
        extra = {"scale_laf": 2.0} if preset.startswith("KeyNet") else {}
        feature = cls(**extra)
        det = feature.detector
        assert type(det).__name__ == detector
        assert det.num_features == num_features
        assert det.mr_size == mr_size
        if response is not None:
            assert type(getattr(det, "resp", None) or det.model).__name__ == response
        assert type(det.ori).__name__ == "LAFOrienter"
        assert type(det.ori.angle_detector).__name__ == orienter
        assert det.ori.patch_size == orienter_size
        assert type(det.aff).__name__ == affine
        if affine != "PassLAF":
            # the detectors estimate the affine shape before the orientation, so AffNet's upright shape gets turned
            assert det.aff.preserve_orientation is False
        assert type(feature.descriptor).__name__ == "LAFDescriptor"
        assert type(feature.descriptor.descriptor).__name__ == descriptor
        assert feature.descriptor.patch_size == patch_size
        # the KeyNet presets' scale_laf is LocalFeature's scaling_coef; every other preset keeps the detector frames
        assert feature.scaling_coef == (2.0 if extra else 1.0)
        assert type(cls(upright=True).detector.ori).__name__ == "PassLAF"

    def test_convention_local_feature_ratio_tests_keep_a_single_padded_pair(self, device, dtype):
        # A detector that filled num_features - 1 slots leaves one padded slot: a zero LAF with the descriptor of the
        # origin patch. Image 0 has 4 real slots and 1 padded one, image 1 has 3 real and 1 padded. The padded pair is
        # at distance 0 and each padded descriptor's second-nearest is a real one, so the ratio is 0: match_snn and
        # match_smnn both report the pair, and filtering with laf_is_filled removes it.
        gen = torch.Generator().manual_seed(3)
        real = torch.nn.functional.normalize(torch.rand(7, 8, generator=gen, dtype=torch.float64), dim=-1)
        pad = torch.nn.functional.normalize(
            torch.tensor([[0.9, 0.1, 0.4, 0.2, 0.7, 0.3, 0.5, 0.6]], dtype=torch.float64), dim=-1
        )
        desc0 = torch.cat([real[:4], pad]).to(device, dtype)
        desc1 = torch.cat([real[4:], pad]).to(device, dtype)
        centres = torch.tensor([[[23.0, 11.0], [61.0, 17.0], [9.0, 40.0], [75.0, 33.0]]])
        scales, oris = torch.full((1, 4, 1, 1), 6.0), torch.full((1, 4, 1), 17.0)
        lafs = kornia.feature.laf_from_center_scale_ori(centres, scales, oris)
        lafs0 = torch.cat([lafs, torch.zeros(1, 1, 2, 3)], dim=1).to(device, dtype)
        lafs1 = torch.cat([lafs[:, :3], torch.zeros(1, 1, 2, 3)], dim=1).to(device, dtype)
        filled0, filled1 = laf_is_filled(lafs0)[0], laf_is_filled(lafs1)[0]
        assert filled0.tolist() == [True, True, True, True, False]
        assert filled1.tolist() == [True, True, True, False]
        for matcher, th in ((kornia.feature.match_snn, 0.8), (kornia.feature.match_smnn, 0.95)):
            dists, idxs = matcher(desc0, desc1, th)
            pairs = [tuple(pair) for pair in idxs.tolist()]
            assert (4, 3) in pairs
            assert dists[pairs.index((4, 3))].item() == 0.0
            _, idxs_filled = matcher(desc0[filled0], desc1[filled1], th)
            assert (idxs_filled[:, 0] < 4).all()
            assert (idxs_filled[:, 1] < 3).all()

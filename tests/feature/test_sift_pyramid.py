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
# limitations under the License.

from __future__ import annotations

import math

import pytest
import torch

from kornia.feature import LAFOrienter, SIFTDescriptorFromPyramid, get_laf_orientation, laf_from_center_scale_ori
from kornia.filters import gaussian_blur2d

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding


class TestSIFTDescriptorFromPyramid(BaseTester):
    def test_orientation_rejects_pixel_oscillation_5748(self, device, dtype):
        # A horizontal ramp with a four-pixel vertical oscillation. Integrating
        # smoothed gradients preserves the ramp; the former sparse histogram
        # sampling aliases the oscillation into an orientation about -51 degrees.
        axis = torch.arange(65, device=device, dtype=torch.float32)
        image = (0.2 * axis[None, :] + 0.25 * torch.sin(torch.pi / 2 * (axis[:, None] - 32)))[None, None]
        image = image.to(dtype)
        lafs = torch.tensor([[[[8.0, 0.0, 32.0], [0.0, 8.0, 32.0]]]], device=device, dtype=dtype)
        oriented, _ = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs)
        self.assert_close(
            get_laf_orientation(oriented), torch.zeros(1, 1, 1, device=device, dtype=dtype), rtol=0.0, atol=5.0
        )

    @pytest.mark.parametrize("size", [1, 3, 4, 8, 9])
    @pytest.mark.parametrize("inference", [False, True])
    def test_separable_pooling_matches_full_kernel(self, device, dtype, size, inference):
        feature = SIFTDescriptorFromPyramid(spatial_bin_size=size).to(device, dtype)
        # Cross the CPU pooling tile boundary and leave a partial final tile.
        hist = torch.rand(2, 36, 37, 19, device=device, dtype=dtype)
        # The public descriptor promotes reduced-precision images before pooling.
        if dtype in (torch.float16, torch.bfloat16):
            hist = hist.float()
        pooled = torch.nn.functional.conv2d(
            hist.reshape(-1, 1, 37, 19), feature.descriptor_pooling_kernel.to(hist), padding=size // 2
        )
        if size % 2 == 0:
            pooled = torch.nn.functional.avg_pool2d(pooled, 2, stride=1)
        with torch.set_grad_enabled(not inference):
            actual = feature._pool_histograms(hist)
        self.assert_close(actual.to(dtype), pooled.reshape_as(hist).to(dtype))

    def test_orientation_bins_do_not_change_supplied_frame_descriptors(self, device, dtype):
        image = torch.rand(1, 1, 48, 48, device=device, dtype=dtype)
        lafs = torch.tensor([[[[8.0, 2.0, 24.0], [-2.0, 8.0, 24.0]]]], device=device, dtype=dtype)
        coarse = SIFTDescriptorFromPyramid(orientation_bins=18).to(device, dtype)
        fine = SIFTDescriptorFromPyramid(orientation_bins=72).to(device, dtype)
        self.assert_close(coarse(image, lafs), fine(image, lafs))

    def test_shape_norm_and_empty(self, device, dtype):
        image = torch.rand(2, 1, 64, 80, device=device, dtype=dtype)
        xy = torch.tensor([[[24.0, 24.0], [48.0, 40.0]], [[20.0, 30.0], [52.0, 36.0]]], device=device, dtype=dtype)
        scale = torch.full((2, 2, 1, 1), 8.0, device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(xy, scale)
        feature = SIFTDescriptorFromPyramid().to(device, dtype)
        oriented, descriptors = feature.orient_and_describe(image, lafs)
        assert oriented.shape == lafs.shape
        assert descriptors.shape == (2, 2, 128)
        self.assert_close(descriptors.norm(dim=-1), torch.ones(2, 2, device=device, dtype=dtype))
        empty_lafs, empty_descriptors = feature.orient_and_describe(image, lafs[:, :0])
        assert empty_lafs.shape == (2, 0, 2, 3)
        assert empty_descriptors.shape == (2, 0, 128)

    def test_float64_identity_remap_keeps_histogram_5127(self, device):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        histograms = torch.rand(1, 3, 8, 5, device=device, dtype=torch.float64)
        affine = torch.eye(2, device=device, dtype=torch.float64).expand(1, 3, 2, 2)
        remapped = SIFTDescriptorFromPyramid._remap_angles(histograms, affine, 8)
        # The float32 pi moved 2.2e-7 of every bin past pi into its lower neighbour.
        self.assert_close(remapped, histograms, rtol=0.0, atol=1e-12)

    def test_horizontal_gradient_keeps_orientation(self, device, dtype):
        image = torch.arange(64, device=device, dtype=dtype).reshape(1, 1, 1, 64).expand(1, 1, 64, 64)
        xy = torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(xy, torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype))
        oriented, _ = SIFTDescriptorFromPyramid(rootsift=False).to(device, dtype).orient_and_describe(image, lafs)
        self.assert_close(
            get_laf_orientation(oriented), torch.zeros(1, 1, 1, device=device, dtype=dtype), rtol=0.0, atol=1e-2
        )

    def test_float64_vertical_gradient_orientation_5127(self, device):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        # The peak lands on orientation bin 9 of 36, so the bin-to-angle scale decides the angle; the
        # horizontal case above sits on bin 0 and cannot see it.
        image = torch.arange(64, device=device, dtype=torch.float64).reshape(1, 1, 64, 1).expand(1, 1, 64, 64)
        xy = torch.tensor([[[32.0, 32.0]]], device=device, dtype=torch.float64)
        lafs = laf_from_center_scale_ori(xy, torch.full((1, 1, 1, 1), 8.0, device=device, dtype=torch.float64))
        oriented, _ = (
            SIFTDescriptorFromPyramid(rootsift=False).to(device, torch.float64).orient_and_describe(image, lafs)
        )
        angle = torch.deg2rad(get_laf_orientation(oriented)).flatten()
        # The float32 pi put this angle 1.6e-8 rad off; the residual here is about 6e-11.
        self.assert_close(angle, torch.full_like(angle, -torch.pi / 2), rtol=0.0, atol=1e-9)

    def test_flat_input_has_finite_backward(self, device, dtype):
        image = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype, requires_grad=True)
        xy = torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype)
        # This scale selects the second octave, exercising the in-place updates
        # on _laf_at_level's cloned LAF tensor during backward.
        lafs = laf_from_center_scale_ori(
            xy, torch.full((1, 1, 1, 1), 20.0, device=device, dtype=dtype, requires_grad=True)
        )
        _, descriptors = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs)
        descriptors.sum().backward()
        assert torch.isfinite(image.grad).all()

    def test_odd_spatial_pooling(self, device, dtype):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        xy = torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(xy, torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype))
        _, descriptors = (
            SIFTDescriptorFromPyramid(spatial_bin_size=3).to(device, dtype).orient_and_describe(image, lafs)
        )
        assert descriptors.shape == (1, 1, 128)

    def test_invalid_laf_and_mixed_laf_dtype(self, device, dtype):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        laf_dtype = torch.float32 if device.type == "mps" else torch.float64
        lafs = torch.zeros(1, 2, 2, 3, device=device, dtype=laf_dtype)
        lafs[0, 1] = torch.tensor([[8.0, 0.0, 32.0], [0.0, 8.0, 32.0]], device=device, dtype=laf_dtype)
        oriented, descriptors = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs)
        assert oriented.dtype == lafs.dtype
        assert descriptors.dtype == dtype
        self.assert_close(descriptors[0, 0], torch.zeros(128, device=device, dtype=dtype))
        assert torch.isfinite(descriptors).all()

    def test_all_invalid_lafs_keep_zero_backward(self, device, dtype):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype, requires_grad=True)
        lafs = torch.zeros(1, 2, 2, 3, device=device, dtype=dtype, requires_grad=True)
        descriptors = SIFTDescriptorFromPyramid().to(device, dtype)(image, lafs)
        descriptors.sum().backward()
        self.assert_close(descriptors, torch.zeros_like(descriptors))
        self.assert_close(image.grad, torch.zeros_like(image.grad))
        self.assert_close(lafs.grad, torch.zeros_like(lafs.grad))

    def test_unused_upper_octaves_do_not_build_histograms(self, device, dtype, monkeypatch):
        import kornia.feature.sift.pyramid as implementation

        image = torch.rand(1, 1, 80, 80, device=device, dtype=dtype)
        lafs = torch.tensor([[[[6.0, 0, 40], [0, 6.0, 40]]]], device=device, dtype=dtype)
        calls = []
        pyrdown_calls = []
        original = implementation.spatial_gradient
        original_pyrdown = implementation.pyrdown

        def gradient(level, *args, **kwargs):
            calls.append(level.shape)
            return original(level, *args, **kwargs)

        def counted_pyrdown(level, *args, **kwargs):
            pyrdown_calls.append(level.shape)
            return original_pyrdown(level, *args, **kwargs)

        monkeypatch.setattr(implementation, "spatial_gradient", gradient)
        monkeypatch.setattr(implementation, "pyrdown", counted_pyrdown)
        actual = SIFTDescriptorFromPyramid().to(device, dtype)(image, lafs)
        assert calls == [image.shape]
        assert pyrdown_calls == []
        single_level = SIFTDescriptorFromPyramid().to(device, dtype)
        monkeypatch.setattr(single_level, "_pyramid", lambda image, num_levels=None: [image])
        self.assert_close(actual, single_level(image, lafs))

    def test_unused_lower_octaves_do_not_build_histograms(self, device, dtype, monkeypatch):
        import kornia.feature.sift.pyramid as implementation

        image = torch.rand(1, 1, 160, 160, device=device, dtype=dtype)
        # A large frame selects octave 2. Octaves 0 and 1 are still resampled to
        # reach it, but their histogram maps are the largest allocations here and
        # nothing ever samples them.
        lafs = torch.tensor([[[[40.0, 0.0, 80.0], [0.0, 40.0, 80.0]]]], device=device, dtype=dtype)
        feature = SIFTDescriptorFromPyramid().to(device, dtype)
        levels = feature._select_levels(lafs, feature._num_pyramid_levels(image))
        assert levels.flatten().tolist() == [2]

        shapes = []
        original = implementation.spatial_gradient

        def gradient(level, *args, **kwargs):
            shapes.append(tuple(level.shape[-2:]))
            return original(level, *args, **kwargs)

        monkeypatch.setattr(implementation, "spatial_gradient", gradient)
        oriented, descriptors = feature.orient_and_describe(image, lafs)
        # Description and orientation use distinct gradient maps at the used octave.
        assert shapes == [(40, 40), (40, 40)]

        # Skipping the unused octaves must not change a single output value.
        # Only CPU can be held to bitwise equality: the histogram scatter uses
        # atomics on an accelerator, so it is not run-to-run reproducible there.
        reference = SIFTDescriptorFromPyramid().to(device, dtype)
        monkeypatch.setattr(reference, "_level_occupancy", lambda levels, num_levels: [True] * num_levels)
        expected_lafs, expected_descriptors = reference.orient_and_describe(image, lafs)
        exact = {"rtol": 0.0, "atol": 0.0} if device.type == "cpu" else {}
        self.assert_close(oriented, expected_lafs, **exact)
        self.assert_close(descriptors, expected_descriptors, **exact)

    def test_all_invalid_lafs_skip_pyramid_construction(self, device, dtype, monkeypatch):
        image = torch.rand(1, 1, 80, 80, device=device, dtype=dtype)
        lafs = torch.zeros(1, 2, 2, 3, device=device, dtype=dtype)
        feature = SIFTDescriptorFromPyramid().to(device, dtype)

        def unexpected_pyramid(*args, **kwargs):
            raise AssertionError("all-invalid LAFs must not build an image pyramid")

        monkeypatch.setattr(feature, "_pyramid", unexpected_pyramid)
        descriptors = feature(image, lafs)
        self.assert_close(descriptors, torch.zeros_like(descriptors))

    def test_upright_preserves_laf_orientation(self, device, dtype):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        xy = torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(
            xy,
            torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype),
            torch.full((1, 1, 1), 25.0, device=device, dtype=dtype),
        )
        oriented, _ = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs, upright=True)
        self.assert_close(get_laf_orientation(oriented), get_laf_orientation(lafs))

    def test_affine_gradient_is_canonicalized(self, device, dtype):
        # I(x,y)=x+y has a known gradient, so A^T(1,1) tests shear,
        # anisotropic magnitude, nonzero orientation, and the LAF rotation sign.
        axis = torch.arange(65, device=device, dtype=dtype)
        image = (axis[:, None] + axis[None, :])[None, None]
        lafs = torch.tensor([[[[8.0, 2.0, 32.0], [0.0, 6.0, 32.0]]]], device=device, dtype=dtype)
        oriented, descriptors = (
            SIFTDescriptorFromPyramid(rootsift=False).to(device, dtype).orient_and_describe(image, lafs)
        )
        gradient = oriented[..., :2, :2].transpose(-1, -2) @ torch.ones(2, 1, device=device, dtype=dtype)
        # Histogram discretization interpolates a 45-degree input between bins.
        assert gradient[..., 0, 0].min() > 0
        assert (gradient[..., 1, 0].abs() / gradient[..., 0, 0]).max() < 0.06
        assert descriptors.reshape(1, 1, 8, 16).sum(-1).argmax(-1).item() == 0

    def test_odd_pyramid_coordinates(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("direct pyramid helper requires native reflect padding; the descriptor promotes half inputs")
        # Linear images remain linear away from borders. Sampling any octave at
        # the transformed frame centre must return the original pixel coordinate.
        h, w = 65, 67
        x = torch.arange(w, device=device, dtype=dtype)[None].expand(h, w)
        y = torch.arange(h, device=device, dtype=dtype)[:, None].expand(h, w)
        feature = SIFTDescriptorFromPyramid().to(device, dtype)
        lafs = torch.tensor([[[[20.0, 0.0, 27.0], [0.0, 20.0, 31.0]]]], device=device, dtype=dtype)
        pyramid = feature._pyramid(torch.stack([x, y])[None])
        level_lafs = feature._laf_at_level(lafs, pyramid, 1)
        sampled = feature._sample(pyramid[1], pyramid[1], level_lafs, 1)
        self.assert_close(sampled.flatten(), lafs[0, 0, :, 2])

    def test_nonfinite_frame_backward(self, device, dtype):
        image = torch.rand(1, 1, 40, 40, device=device, dtype=dtype, requires_grad=True)
        lafs = torch.tensor(
            [[[[8.0, 0.0, float("nan")], [0.0, 8.0, 20.0]], [[8.0, 0.0, 20.0], [0.0, 8.0, 20.0]]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        _, desc = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs)
        desc.sum().backward()
        self.assert_close(desc[0, 0], torch.zeros_like(desc[0, 0]))
        assert torch.isfinite(image.grad).all()
        assert torch.isfinite(lafs.grad).all()

    @pytest.mark.parametrize("upright", [False, True])
    def test_gradcheck(self, device, upright):
        image = torch.rand(1, 1, 8, 8, device=device, dtype=torch.float64)
        lafs = torch.tensor([[[[2.0, 0.0, 3.0], [0.0, 2.0, 3.0]]]], device=device, dtype=torch.float64)
        feature = SIFTDescriptorFromPyramid(spatial_bin_size=3).to(device, torch.float64)
        self.gradcheck(lambda image: feature.orient_and_describe(image, lafs, upright=upright)[1], (image,))

    @pytest.mark.parametrize("inference", [False, True])
    def test_dynamo(self, device, dtype, torch_optimizer, inference):
        image = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        lafs = torch.tensor([[[[6.0, 0.0, 16.0], [0.0, 6.0, 16.0]]]], device=device, dtype=dtype)
        feature = SIFTDescriptorFromPyramid().to(device, dtype)
        with torch.set_grad_enabled(not inference):
            expected_lafs, expected_descriptors = feature.orient_and_describe(image, lafs)
            actual_lafs, actual_descriptors = torch_optimizer(feature.orient_and_describe)(image, lafs)
        self.assert_close(actual_lafs, expected_lafs)
        self.assert_close(actual_descriptors, expected_descriptors)

    def test_empty_batch(self, device, dtype):
        image = torch.empty(0, 1, 40, 40, device=device, dtype=dtype)
        lafs = torch.empty(0, 3, 2, 3, device=device, dtype=dtype)
        oriented, desc = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs)
        assert oriented.shape == (0, 3, 2, 3)
        assert desc.shape == (0, 3, 128)

    @pytest.mark.parametrize("batch_size,num_lafs", [(1, 0), (0, 3)])
    def test_empty_inputs_keep_zero_backward(self, device, dtype, batch_size, num_lafs):
        image = torch.empty(batch_size, 1, 40, 40, device=device, dtype=dtype, requires_grad=True)
        lafs = torch.empty(batch_size, num_lafs, 2, 3, device=device, dtype=dtype, requires_grad=True)
        descriptors = SIFTDescriptorFromPyramid().to(device, dtype)(image, lafs)
        descriptors.sum().backward()
        self.assert_close(image.grad, torch.zeros_like(image.grad))
        self.assert_close(lafs.grad, torch.zeros_like(lafs.grad))

    def test_descriptor_forward_does_not_reorient(self, device, dtype, monkeypatch):
        image = torch.rand(1, 1, 40, 40, device=device, dtype=dtype)
        lafs = torch.tensor([[[[6.0, 2.0, 20.0], [-2.0, 6.0, 20.0]]]], device=device, dtype=dtype)
        original = lafs.clone()
        descriptor = SIFTDescriptorFromPyramid().to(device, dtype)

        def unexpected_orientation(*args):
            raise AssertionError("Descriptor forward must respect the supplied orientation")

        monkeypatch.setattr(descriptor, "_orientation", unexpected_orientation)
        descriptors = descriptor(image, lafs)
        assert descriptors.shape == (1, 1, 128)
        self.assert_close(lafs, original)

    def test_orientation_and_description_share_one_pyramid(self, device, dtype, monkeypatch):
        image = torch.rand(1, 1, 40, 40, device=device, dtype=dtype)
        lafs = torch.tensor([[[[8.0, 0.0, 20.0], [0.0, 8.0, 20.0]]]], device=device, dtype=dtype)
        descriptor = SIFTDescriptorFromPyramid().to(device, dtype)
        original = descriptor._pyramid
        calls = []

        def counted_pyramid(image, num_levels=None):
            calls.append(image.shape)
            return original(image, num_levels)

        monkeypatch.setattr(descriptor, "_pyramid", counted_pyramid)
        descriptor.orient_and_describe(image, lafs)
        assert len(calls) == 1


class TestConventionsSIFTDescriptorFromPyramid(BaseTester):
    """Pins for the frame in which SIFTDescriptorFromPyramid reads LAFs and the orientation it assigns."""

    @staticmethod
    def _flip_lr(desc: torch.Tensor) -> torch.Tensor:
        """The descriptor of the left-right mirrored neighbourhood: angle bin a becomes 4 - a, cell column cx 3 - cx."""
        d = desc.reshape(-1, 8, 4, 4)
        return d[:, [(4 - a) % 8 for a in range(8)]].flip(-1).reshape(desc.shape)

    @staticmethod
    def _transpose(desc: torch.Tensor) -> torch.Tensor:
        """The descriptor of the transposed neighbourhood: angle bin a becomes 2 - a, cells (cy, cx) become (cx, cy)."""
        d = desc.reshape(-1, 8, 4, 4)
        return d[:, [(2 - a) % 8 for a in range(8)]].transpose(-1, -2).reshape(desc.shape)

    def test_convention_pyramid_sift_reads_lafs_in_pixel_coordinates(self, device, dtype):
        # A LAF centre (x, y) is the pixel centre of column x and row y, as everywhere in kornia.feature. A Gaussian
        # blob centred on pixel (41, 22) of a 60 x 75 image, described by an axis-aligned LAF at that pixel, is its own
        # mirror image, so its descriptor is a fixed point of the mirror permutation; the same LAF half a pixel to the
        # right is not (the residual is 0.058).
        ys, xs = torch.arange(60.0).view(-1, 1), torch.arange(75.0).view(1, -1)
        blob = torch.exp(-((xs - 41) ** 2 + (ys - 22) ** 2) / (2 * 4.0**2))[None, None].to(device, dtype)
        lafs = torch.tensor(
            [[[[6.0, 0.0, 41.0], [0.0, 6.0, 22.0]], [[6.0, 0.0, 41.5], [0.0, 6.0, 22.0]]]], device=device, dtype=dtype
        )
        desc = SIFTDescriptorFromPyramid(rootsift=False).to(device, dtype)(blob, lafs)[0]
        self.assert_close(self._flip_lr(desc[0]), desc[0])
        assert (self._flip_lr(desc[1]) - desc[1]).abs().max() > 0.04
        # Relabelling: transposing the image and the frame (A -> T A T, centre (x, y) -> (y, x)) transposes the
        # described neighbourhood, so the descriptor permutes; a sheared frame on a smooth random image.
        torch.manual_seed(0)
        image = gaussian_blur2d(torch.rand(1, 1, 60, 75, device=device), (7, 7), (1.5, 1.5)).to(dtype)
        laf = torch.tensor([[[[5.0, 1.5, 41.0], [-0.8, 6.0, 22.0]]]], device=device, dtype=dtype)
        laf_t = torch.tensor([[[[6.0, -0.8, 22.0], [1.5, 5.0, 41.0]]]], device=device, dtype=dtype)
        sift = SIFTDescriptorFromPyramid(rootsift=False).to(device, dtype)
        self.assert_close(sift(image.transpose(-1, -2), laf_t)[0], self._transpose(sift(image, laf)[0]))

    def test_convention_orient_and_describe_turns_the_laf_x_axis_onto_the_dominant_gradient(self, device, dtype):
        # The assigned orientation is minus the gradient's angle (measured from +x towards +y, image rows growing
        # downwards), whatever orientation the input LAF had, so the oriented frame's x axis lies along the gradient and
        # the descriptor's dominant angle bin is 0. LAFOrienter assigns the same orientation, so SIFTFeature's patch and
        # pyramid descriptor backends agree.
        ys, xs = torch.arange(60.0).view(-1, 1), torch.arange(75.0).view(1, -1)
        lafs = laf_from_center_scale_ori(
            torch.tensor([[[41.0, 22.0], [30.0, 35.0], [52.0, 30.0]]], device=device, dtype=dtype),
            torch.full((1, 3, 1, 1), 6.0, device=device, dtype=dtype),
            torch.tensor([[[0.0], [50.0], [-120.0]]], device=device, dtype=dtype),
        )
        atol = 0.05 if dtype in (torch.float32, torch.float64) else 1.0  # degrees; bfloat16 reads 135 to a degree
        sift = SIFTDescriptorFromPyramid(rootsift=False).to(device, dtype)
        for gradient_angle in (30.0, -60.0, 135.0):
            g = math.radians(gradient_angle)
            ramp = (0.01 * (math.cos(g) * xs + math.sin(g) * ys))[None, None].to(device, dtype)
            oriented, desc = sift.orient_and_describe(ramp, lafs)
            expected = torch.full((1, 3, 1), -gradient_angle, device=device, dtype=dtype)
            self.assert_close(get_laf_orientation(oriented), expected, rtol=0.0, atol=atol)
            assert desc.reshape(3, 8, 16).sum(-1).argmax(-1).tolist() == [0, 0, 0]
            if supports_replicate_padding(device, dtype):  # LAFOrienter's patch gradients pad with mode="replicate"
                orienter = LAFOrienter(19).to(device, dtype)
                self.assert_close(get_laf_orientation(orienter(lafs, ramp)), expected, rtol=0.0, atol=2 * atol)

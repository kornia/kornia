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

from __future__ import annotations

import pytest
import torch

import kornia.augmentation as K
from kornia.constants import Resample, SamplePadding

from testing.base import BaseTester


class TestConventionGeometricMatrices(BaseTester):
    def test_nearest_resize_uses_nearest_exact_and_half_pixel_matrix(self):
        image = torch.arange(49, dtype=torch.float64).view(1, 1, 7, 7)
        augmentation = K.Resize((14, 21), resample="nearest")

        output = augmentation(image)
        expected = torch.nn.functional.interpolate(image, size=(14, 21), mode="nearest-exact")
        self.assert_close(output, expected)
        self.assert_close(
            augmentation.transform_matrix,
            image.new_tensor([[[3.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]]),
        )

    @pytest.mark.parametrize("input_size,output_size", [((7, 7), (21, 21)), ((12, 12), (9, 9)), ((5, 5), (8, 8))])
    def test_nearest_resize_inverse_roundtrip(self, input_size, output_size):
        image = torch.arange(input_size[0] * input_size[1], dtype=torch.float32).view(1, 1, *input_size)
        augmentation = K.AugmentationSequential(K.Resize(output_size, resample="nearest"))
        restored = augmentation.inverse(augmentation(image))
        assert restored.shape == image.shape
        # Nearest downsampling is lossy, but inverse resampling must still cover both image edges.
        self.assert_close(restored[..., 0, 0], image[..., 0, 0])
        self.assert_close(restored[..., -1, -1], image[..., -1, -1])

    @pytest.mark.parametrize(
        "augmentation", [K.LongestMaxSize(21, resample="nearest"), K.SmallestMaxSize(9, resample="nearest")]
    )
    def test_nearest_max_size_matches_nearest_exact(self, augmentation):
        image = torch.arange(35, dtype=torch.float64).view(1, 1, 5, 7)
        output = augmentation(image)
        expected = torch.nn.functional.interpolate(image, size=output.shape[-2:], mode="nearest-exact")
        self.assert_close(output, expected)

    @pytest.mark.parametrize("align_corners", [False, True])
    def test_nearest_slice_resized_crop_uses_nearest_exact(self, align_corners):
        image = torch.arange(11, dtype=torch.float64).view(1, 1, 1, 11).expand(1, 1, 5, -1)
        augmentation = K.RandomResizedCrop(
            (6, 21), resample="nearest", align_corners=align_corners, cropping_mode="slice"
        )
        params = augmentation.forward_parameters(image.shape)
        params["src"] = image.new_tensor([[[2, 1], [8, 1], [8, 3], [2, 3]]])

        output = augmentation(image, params=params)
        expected = torch.nn.functional.interpolate(image[..., 1:4, 2:9], size=(6, 21), mode="nearest-exact")
        self.assert_close(output, expected)
        self.assert_close(
            augmentation.transform_matrix,
            image.new_tensor([[[3.0, 0.0, -5.0], [0.0, 2.0, -1.5], [0.0, 0.0, 1.0]]]),
        )

    @pytest.mark.parametrize(
        "augmentation,output_size",
        [
            (K.Resize((6, 9), resample="nearest"), (6, 9)),
            (K.LongestMaxSize(9, resample="nearest"), (7, 9)),
            (K.SmallestMaxSize(6, resample="nearest"), (6, 7)),
        ],
    )
    def test_nearest_resize_handles_empty_batch(self, augmentation, output_size):
        image = torch.empty(0, 1, 4, 5)
        output = augmentation(image)
        assert output.shape == (0, 1, *output_size)
        assert augmentation.transform_matrix.shape == (0, 3, 3)

    def test_convention_flips_use_inclusive_pixel_coordinates(self, device, dtype):
        x = torch.zeros(1, 1, 5, 7, device=device, dtype=dtype)
        x[..., 1, 2] = 1
        horizontal = K.RandomHorizontalFlip(p=1.0)
        vertical = K.RandomVerticalFlip(p=1.0)

        # x' = W - 1 - x and y' = H - 1 - y, in the image and in the matrix alike.
        assert horizontal(x)[0, 0].argmax().item() == 1 * 7 + 4
        assert vertical(x)[0, 0].argmax().item() == 3 * 7 + 2
        self.assert_close(
            horizontal.transform_matrix,
            torch.tensor([[[-1.0, 0.0, 6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype),
        )
        self.assert_close(
            vertical.transform_matrix,
            torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 4.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype),
        )

    @pytest.mark.device_agnostic
    def test_convention_affine_translate_and_shear_sample_pixel_parameters(self):
        shape = (128, 1, 5, 7)
        torch.manual_seed(0)
        affine = K.RandomAffine(
            degrees=(0.0, 0.0), translate=(0.2, 0.3), scale=(2.0, 2.0), shear=(10.0, 10.0, 20.0, 20.0), p=1.0
        ).forward_parameters(shape)
        shear = K.RandomShear((10.0, 10.0, 20.0, 20.0), p=1.0).forward_parameters(shape)
        translate = K.RandomTranslate((0.2, 0.2), (-0.3, -0.3), p=1.0).forward_parameters(shape)

        for params in (affine, shear):
            self.assert_close(params["center"], params["center"].new_tensor([3.0, 2.0]).expand(128, 2))
            self.assert_close(params["shear_x"], params["shear_x"].new_full((128,), 10.0))
            self.assert_close(params["shear_y"], params["shear_y"].new_full((128,), 20.0))
        self.assert_close(affine["scale"], affine["scale"].new_full((128, 2), 2.0))
        offsets = affine["translations"]
        extent = offsets.new_tensor([0.2 * 7, 0.3 * 5])
        assert (offsets.abs() <= extent).all()
        # Both ends must be approached: swapping W/H must fail even if all samples fit one bound.
        assert (offsets.amin(dim=0) < -0.9 * extent).all()
        assert (offsets.amax(dim=0) > 0.9 * extent).all()
        self.assert_close(translate["translate_x"], translate["translate_x"].new_full((128,), 1.4))
        self.assert_close(translate["translate_y"], translate["translate_y"].new_full((128,), -1.5))

    @pytest.mark.parametrize("kind", ["affine", "translate"])
    def test_convention_positive_translation_moves_content_right_and_down(self, device, dtype, kind):
        # Corner-aligned 5x9 grids have exactly representable steps even in half precision.
        image = torch.zeros(1, 1, 5, 9, device=device, dtype=dtype)
        image[..., 1, 2] = 1
        if kind == "affine":
            aug = K.RandomAffine(0.0, translate=(0.2, 0.3), align_corners=True, p=1.0)
            params = aug.forward_parameters(image.shape)
            params["translations"] = image.new_tensor([[1.0, 2.0]])
        else:
            aug = K.RandomTranslate((1 / 9, 1 / 9), (2 / 5, 2 / 5), align_corners=True, p=1.0)
            params = aug.forward_parameters(image.shape)
        expected = torch.zeros_like(image)
        expected[..., 3, 3] = 1
        self.assert_close(aug(image, params=params), expected)
        self.assert_close(aug.transform_matrix[0, :2, 2], image.new_tensor([1.0, 2.0]))

    def test_convention_perspective_uses_inclusive_source_corners(self, device, dtype):
        x = torch.zeros(1, 1, 5, 7, device=device, dtype=dtype)
        augmentation = K.RandomPerspective(distortion_scale=0.0, p=1.0)
        augmentation(x)

        for name in ("start_points", "end_points"):
            points = augmentation._params[name]
            expected = points.new_tensor([[[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]])
            self.assert_close(points, expected)
        self.assert_close(augmentation.transform_matrix, torch.eye(3, device=device, dtype=dtype)[None])

    @pytest.mark.device_agnostic
    def test_convention_perspective_sampling_extents_and_directions(self):
        shape = (64, 1, 5, 7)
        torch.manual_seed(0)
        basic = K.RandomPerspective(0.5, p=1.0).forward_parameters(shape)
        area = K.RandomPerspective(0.5, p=1.0, sampling_method="area_preserving").forward_parameters(shape)
        basic_offset = basic["end_points"] - basic["start_points"]
        area_offset = area["end_points"] - area["start_points"]
        extent = basic_offset.new_tensor([1.75, 1.25])

        assert (basic_offset[:, 0] >= 0).all()
        assert (basic_offset[:, 1, 0] <= 0).all()
        assert (basic_offset[:, 1, 1] >= 0).all()
        assert (basic_offset[:, 2] <= 0).all()
        assert (basic_offset[:, 3, 0] >= 0).all()
        assert (basic_offset[:, 3, 1] <= 0).all()
        assert (basic_offset.abs() <= extent).all()
        assert (basic_offset.abs().amax(dim=(0, 1)) > 0.9 * extent).all()
        assert (area_offset.abs() <= extent).all()
        assert (area_offset.abs().amax(dim=(0, 1)) > 0.9 * extent).all()
        assert (area_offset.amin(dim=(0, 1)) < 0).all()
        assert (area_offset.amax(dim=(0, 1)) > 0).all()

    @pytest.mark.device_agnostic
    def test_convention_perspective_same_on_batch_reuses_destination_corners(self):
        params = K.RandomPerspective(0.5, p=1.0, same_on_batch=True).forward_parameters((3, 1, 5, 7))

        self.assert_close(params["end_points"], params["end_points"][:1].expand_as(params["end_points"]))

    @pytest.mark.parametrize("constant", [False, True])
    @pytest.mark.parametrize("align_corners", [False, True])
    def test_convention_random_perspective_identity_warp_is_identity_4411(self, device, dtype, constant, align_corners):
        image = torch.ones(1, 1, 5, 7, device=device, dtype=dtype)
        if not constant:
            image = torch.arange(35, device=device, dtype=dtype).reshape_as(image)
        self.assert_close(
            K.RandomPerspective(0.0, align_corners=align_corners, p=1.0)(image), image, low_tolerance=True
        )

    @pytest.mark.parametrize("align_corners", [False, True])
    def test_random_perspective_float64_identity_is_exact_4776(self, device, align_corners):
        # #4776: warp_perspective builds its grid in the input dtype, so a float64 identity warp is exact to
        # float64 roundoff, like RandomAffine's warp_affine on the same input.
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        image = torch.arange(35, device=device, dtype=torch.float64).reshape(1, 1, 5, 7) / 35
        output = K.RandomPerspective(0.0, align_corners=align_corners, p=1.0)(image)
        assert output.dtype == torch.float64
        assert (output - image).abs().max() < 1e-12

    @pytest.mark.parametrize("size", [(1, 7), (5, 1), (1, 1)])
    @pytest.mark.parametrize("align_corners", [False, True])
    def test_convention_random_perspective_singleton_dimensions_4787(self, device, dtype, size, align_corners):
        # #4787: a size-1 axis gets a unit source extent and no offset, so the solve is not degenerate and
        # zero distortion is the identity.
        height, width = size
        image = torch.arange(height * width, device=device, dtype=dtype).reshape(1, 1, height, width)
        augmentation = K.RandomPerspective(0.0, align_corners=align_corners, p=1.0)

        output = augmentation(image)

        assert torch.isfinite(augmentation.transform_matrix).all()
        self.assert_close(output, image, low_tolerance=True)

    @pytest.mark.parametrize("sampling_method", ["basic", "area_preserving"])
    @pytest.mark.parametrize("size", [(1, 7), (5, 1)])
    def test_convention_random_perspective_does_not_distort_a_singleton_axis_4787(
        self, device, dtype, size, sampling_method
    ):
        height, width = size
        torch.manual_seed(0)
        image = torch.rand(4, 1, height, width, device=device, dtype=dtype)
        augmentation = K.RandomPerspective(1.0, p=1.0, sampling_method=sampling_method)

        output = augmentation(image)

        assert torch.isfinite(output).all()
        # The singleton axis spans a unit source extent (the other axis keeps its inclusive corners).
        start = augmentation._params["start_points"]
        x_end, y_end = (width - 1, 1) if height == 1 else (1, height - 1)
        expected = torch.tensor([[0, 0], [x_end, 0], [x_end, y_end], [0, y_end]]).to(start).expand_as(start)
        assert torch.equal(start, expected)
        matrix = augmentation.transform_matrix.to(torch.float32)
        assert torch.isfinite(matrix).all()
        # The singleton row (or column) maps onto itself, while the other axis is still warped.
        axis = 1 if height == 1 else 0
        extent = width if height == 1 else height
        points = torch.zeros(4, extent, 2, device=device)
        points[..., 1 - axis] = torch.arange(extent, device=device, dtype=torch.float32)
        homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1) @ matrix.transpose(-1, -2)
        mapped = homogeneous[..., :2] / homogeneous[..., 2:]
        self.assert_close(mapped[..., axis], torch.zeros_like(mapped[..., axis]), low_tolerance=True)
        assert not torch.allclose(mapped[..., 1 - axis], points[..., 1 - axis], atol=1e-2)

    def test_wart_random_affine_rotation_sign_4408(self, device, dtype):
        x = torch.zeros(1, 1, 7, 7, device=device, dtype=dtype)
        x[..., 1, 4] = 1
        rotation = K.RandomRotation((90.0, 90.0), p=1.0)(x)
        affine = K.RandomAffine((90.0, 90.0), p=1.0)(x)

        # Positive RandomRotation is visually counter-clockwise, while the affine composer is clockwise.
        assert rotation[0, 0].argmax().item() == 2 * 7 + 1
        assert affine[0, 0].argmax().item() == 4 * 7 + 5

    # Rounding a continuous draw on `[lo, hi]` gave the two end points half the weight of the others, so
    # `times=(-1, 1)` drew 0 about twice as often as -1 or 1 (25% / 50% / 25%). The draw is now floored on
    # `[lo, hi + 1)`, so every whole number of a closed range has the same probability.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   drawn = K.RandomRotation90((-1, 1), p=1.0).forward_parameters((9000, 1, 4, 4))["times"]
    #   print(torch.unique(drawn, return_counts=True))
    # The pin allows 15% around the uniform count, 8 sigma for 3000 draws per value, and what it has to separate
    # is a factor of two, not a few percent.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("times", [(-1, 1), (0, 2), (0, 3), (-3, 3)])
    def test_convention_random_rotation90_times_are_closed_and_uniform_4409(self, times):
        low, high = times
        count = high - low + 1
        torch.manual_seed(0)
        drawn = K.RandomRotation90(times, p=1.0).forward_parameters((count * 3000, 1, 4, 4))["times"]

        assert torch.equal(drawn, drawn.round())
        assert (int(drawn.min()), int(drawn.max())) == (low, high)
        counts = torch.bincount((drawn - low).long(), minlength=count)
        expected = drawn.numel() / count
        assert bool(((counts > 0.85 * expected) & (counts < 1.15 * expected)).all()), counts.tolist()

    @pytest.mark.device_agnostic
    def test_convention_random_rotation90_same_on_batch_shares_one_whole_number_4409(self):
        torch.manual_seed(0)
        drawn = K.RandomRotation90((-3, 3), p=1.0, same_on_batch=True).forward_parameters((16, 1, 4, 4))["times"]

        assert drawn.shape == (16,)
        assert bool((drawn == drawn[0]).all())
        assert float(drawn[0]) == float(drawn[0].round())

    # The excluded end point of the half-open sampler range is clamped back, not drawn: for the largest value
    # `torch.rand` can return, `lo + u * (hi + 1 - lo)` rounds up onto `hi + 1` in float32 for these ranges
    # (`1 + (1 - 2 ** -24) * 1` is exactly `2.0`), which `floor` would keep -- about one draw in 2**24, and
    # `times=(1, 1)` would turn a quarter turn into a half turn.
    # Snippet used to generate expected: the same ranges, with `torch.rand` patched to return `1 - 2 ** -24`.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("times", [(1, 1), (-3, -3), (-2, -2), (1, 2)])
    def test_convention_random_rotation90_largest_draw_stays_in_the_closed_range_4409(self, monkeypatch, times):
        largest = 1 - 2**-24
        real_rand = torch.rand
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.full_like(real_rand(*args, **kwargs), largest))
        drawn = K.RandomRotation90(times, p=1.0).forward_parameters((4, 1, 4, 4))["times"]

        assert drawn.tolist() == [float(times[1])] * 4

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        "times,error,match",
        [
            ((2, 1), ValueError, r"times\[0\] should be smaller than times\[1\]"),
            ((-4, 4), ValueError, "times out of bounds"),
            ((float("nan"), 1), ValueError, "must be finite"),
            ((1, 2, 3), TypeError, "length 2"),
        ],
    )
    def test_convention_random_rotation90_times_range_validation_is_unchanged_4409(self, times, error, match):
        with pytest.raises(error, match=match):
            K.RandomRotation90(times, p=1.0)

    # A range with a fractional bound is not a closed range of whole numbers, so it keeps the continuous draw
    # that is rounded when the matrix is built, and an integer tensor range draws its upper bound too.
    @pytest.mark.device_agnostic
    def test_convention_random_rotation90_fractional_and_tensor_ranges_4409(self):
        torch.manual_seed(0)
        fractional = K.RandomRotation90((0.5, 2.5), p=1.0).forward_parameters((6000, 1, 4, 4))["times"]
        assert float(fractional.min()) >= 0.5
        assert float(fractional.max()) <= 2.5
        assert not torch.equal(fractional, fractional.round())

        tensor_range = K.RandomRotation90(torch.tensor([0, 2]), p=1.0).forward_parameters((6000, 1, 4, 4))["times"]
        assert tensor_range.dtype == torch.int64
        assert set(tensor_range.tolist()) == {0, 1, 2}

    def test_convention_random_rotation90_rounds_supplied_times_4409(self, device, dtype):
        # `times` passed through `params`, for instance saved by an earlier release that drew them continuously,
        # is still rounded to a whole number of quarter turns.
        augmentation = K.RandomRotation90((0, 1), p=1.0)
        x = torch.zeros(2, 1, 7, 7, device=device, dtype=dtype)
        x[..., 1, 4] = 1
        params = augmentation.forward_parameters(x.shape)
        params["times"] = x.new_tensor([0.6, 0.4])
        output = augmentation(x, params=params)
        assert output[0, 0].argmax().item() == 2 * 7 + 1
        assert output[1, 0].argmax().item() == 1 * 7 + 4

    @pytest.mark.device_agnostic
    def test_wart_random_rotation90_keeps_non_square_shape_4409(self):
        x = torch.zeros(1, 1, 5, 7)
        assert K.RandomRotation90((1, 1), p=1.0)(x).shape == x.shape

    @pytest.mark.device_agnostic
    def test_wart_align_corners_defaults_split_geometric_augmentations_4412(self):
        assert K.RandomRotation(0.0).flags["align_corners"] is True
        assert K.RandomRotation90((0, 0)).flags["align_corners"] is True
        assert K.RandomAffine(0.0).flags["align_corners"] is False
        assert K.RandomShear(0.0).flags["align_corners"] is False
        assert K.RandomTranslate().flags["align_corners"] is False
        assert K.RandomPerspective().flags["align_corners"] is False

    def test_convention_geometric_resample_and_padding_defaults(self, device, dtype):
        affine = K.RandomAffine(0.0)
        shear = K.RandomShear(0.0)
        translate = K.RandomTranslate()
        perspective = K.RandomPerspective()
        rotation = K.RandomRotation(0.0)
        rotation90 = K.RandomRotation90((0, 0))

        for augmentation in (affine, shear, translate):
            assert augmentation.flags["resample"] is Resample.BILINEAR
            assert augmentation.flags["padding_mode"] is SamplePadding.ZEROS
        for augmentation in (perspective, rotation, rotation90):
            assert augmentation.flags["resample"] is Resample.BILINEAR

        x = torch.ones(1, 1, 5, 7, device=device, dtype=dtype)
        translated = K.RandomTranslate((1.0, 1.0), (0.0, 0.0), p=1.0)(x)
        self.assert_close(translated, torch.zeros_like(x))

        perspective_params = {
            "start_points": x.new_tensor([[[0, 0], [6, 0], [6, 4], [0, 4]]]),
            "end_points": x.new_tensor([[[1, 1], [5, 1], [5, 3], [1, 3]]]),
        }
        perspective_output = K.RandomPerspective(0.5, p=1.0)(x, params=perspective_params)
        rotation_output = K.RandomRotation((45.0, 45.0), p=1.0)(x)
        rotation90_output = K.RandomRotation90((1, 1), p=1.0)(x)
        for border in (perspective_output[..., 0, :], rotation_output[..., 0, 0], rotation90_output[..., :, 0]):
            self.assert_close(border, torch.zeros_like(border))

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
    @pytest.mark.device_agnostic
    def test_convention_flips_use_inclusive_pixel_coordinates(self):
        x = torch.zeros(1, 1, 5, 7)
        x[..., 1, 2] = 1
        horizontal = K.RandomHorizontalFlip(p=1.0)(x)
        vertical = K.RandomVerticalFlip(p=1.0)(x)

        assert horizontal[0, 0].argmax().item() == 1 * 7 + 4
        assert vertical[0, 0].argmax().item() == 3 * 7 + 2

    def test_convention_flips_expose_their_discrete_coordinate_matrices(self, device, dtype):
        x = torch.zeros(1, 1, 5, 7, device=device, dtype=dtype)
        horizontal = K.RandomHorizontalFlip(p=1.0)
        vertical = K.RandomVerticalFlip(p=1.0)
        horizontal(x)
        vertical(x)

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

    def test_convention_perspective_sampling_extents_and_directions(self):
        shape = (64, 1, 5, 7)
        torch.manual_seed(0)
        basic = K.RandomPerspective(0.5, p=1.0).forward_parameters(shape)
        area = K.RandomPerspective(0.5, p=1.0, sampling_method="area_preserving").forward_parameters(shape)
        basic_offset = basic["end_points"] - basic["start_points"]
        area_offset = area["end_points"] - area["start_points"]
        extent = basic_offset.new_tensor([1.75, 1.25])

        assert (basic_offset[:, 0] >= 0).all()
        assert (basic_offset[:, 1, 0] <= 0).all() and (basic_offset[:, 1, 1] >= 0).all()
        assert (basic_offset[:, 2] <= 0).all()
        assert (basic_offset[:, 3, 0] >= 0).all() and (basic_offset[:, 3, 1] <= 0).all()
        assert (basic_offset.abs() <= extent).all()
        assert (area_offset.abs() <= extent).all()
        assert (area_offset.amin(dim=(0, 1)) < 0).all()
        assert (area_offset.amax(dim=(0, 1)) > 0).all()

    def test_convention_perspective_same_on_batch_reuses_destination_corners(self):
        params = K.RandomPerspective(0.5, p=1.0, same_on_batch=True).forward_parameters((3, 1, 5, 7))

        self.assert_close(params["end_points"], params["end_points"][:1].expand_as(params["end_points"]))

    @pytest.mark.parametrize("constant", [False, True])
    @pytest.mark.parametrize(
        "align_corners",
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    strict=True, raises=AssertionError, reason="#4411: identity perspective warp is not identity"
                ),
            ),
        ],
    )
    def test_convention_random_perspective_identity_warp_is_identity_4411(self, device, dtype, constant, align_corners):
        image = torch.ones(1, 1, 5, 7, device=device, dtype=dtype)
        if not constant:
            image = torch.arange(35, device=device, dtype=dtype).reshape_as(image)
        self.assert_close(K.RandomPerspective(0.0, align_corners=align_corners, p=1.0)(image), image)

    def test_wart_random_affine_rotation_sign_4408(self, device, dtype):
        x = torch.zeros(1, 1, 7, 7, device=device, dtype=dtype)
        x[..., 1, 4] = 1
        rotation = K.RandomRotation((90.0, 90.0), p=1.0)(x)
        affine = K.RandomAffine((90.0, 90.0), p=1.0)(x)

        # Positive RandomRotation is visually counter-clockwise, while the affine composer is clockwise.
        assert rotation[0, 0].argmax().item() == 2 * 7 + 1
        assert affine[0, 0].argmax().item() == 4 * 7 + 5

    def test_wart_random_rotation90_continuously_samples_then_rounds_4409(self, device, dtype):
        augmentation = K.RandomRotation90((0, 1), p=1.0)
        torch.manual_seed(1)
        params = augmentation.forward_parameters((64, 1, 5, 7))

        assert ((params["times"] >= 0) & (params["times"] <= 1)).all()
        assert not torch.equal(params["times"], params["times"].round())

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

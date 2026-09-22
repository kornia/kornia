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

import kornia
import kornia.augmentation as K

from testing.base import BaseTester, supports_bilinear_3d_grid_sample, supports_nearest_3d_grid_sample


class Test3DAugmentationConventions(BaseTester):
    def test_convention_rank_keepdim_and_dtype_guard(self, device, dtype):
        # 3D augmentations promote D,H,W and C,D,H,W to B,C,D,H,W. keepdim restores only promoted dimensions.
        augmentation = K.RandomHorizontalFlip3D(p=1.0, keepdim=True)
        unchannelled = torch.arange(24, device=device, dtype=dtype).reshape(2, 3, 4)
        channelled = unchannelled.unsqueeze(0)
        batch = channelled.unsqueeze(0)
        assert augmentation(unchannelled).shape == unchannelled.shape
        assert augmentation(channelled).shape == channelled.shape
        assert augmentation(batch).shape == batch.shape
        with pytest.raises(TypeError, match="float16"):
            augmentation(torch.ones(1, 1, 2, 3, 4, device=device, dtype=torch.int64))

    def test_convention_flip_axes_and_inclusive_matrices(self, device, dtype):
        volume = torch.zeros(1, 1, 3, 4, 5, device=device, dtype=dtype)
        volume[..., 0, 2, 3] = 1
        cases = (
            (K.RandomHorizontalFlip3D, -1, [[-1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            (K.RandomVerticalFlip3D, -2, [[1, 0, 0, 0], [0, -1, 0, 3], [0, 0, 1, 0], [0, 0, 0, 1]]),
            (K.RandomDepthicalFlip3D, -3, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 2], [0, 0, 0, 1]]),
        )
        for cls, axis, matrix in cases:
            augmentation = cls(p=1.0)
            output = augmentation(volume)
            self.assert_close(output, torch.flip(volume, (axis,)))
            self.assert_close(augmentation.transform_matrix, volume.new_tensor(matrix)[None])

    @pytest.mark.device_agnostic
    def test_convention_flip_matrix_is_rounded_in_float16(self):
        # Transform matrices keep the input dtype. float16 cannot represent every integer above 2048.
        volume = torch.zeros(1, 1, 1, 1, 2050, dtype=torch.float16)
        augmentation = K.RandomHorizontalFlip3D(p=1.0)
        augmentation(volume)
        assert augmentation.transform_matrix.dtype is torch.float16
        assert augmentation.transform_matrix[0, 0, 3] == 2048

    @pytest.mark.device_agnostic
    def test_convention_degrees_and_motion_angle_follow_xyz_order(self):
        ranges = ((10.0, 10.0), (20.0, 20.0), (30.0, 30.0))
        shape = (2, 1, 3, 4, 5)
        affine = K.RandomAffine3D(ranges, p=1.0).forward_parameters(shape)
        rotation = K.RandomRotation3D(ranges, p=1.0).forward_parameters(shape)
        motion = K.RandomMotionBlur3D(3, ranges, (0.0, 0.0), p=1.0).forward_parameters(shape)
        expected = torch.tensor([10.0, 20.0, 30.0]).expand(2, 3)
        assert torch.equal(affine["angles"], expected)
        assert torch.equal(torch.stack((rotation["yaw"], rotation["pitch"], rotation["roll"]), dim=-1), expected)
        assert torch.equal(motion["angle_factor"], expected)
        volume = torch.zeros(1, 1, 5, 5, 5)
        for axis, diagonal in enumerate(((0, 0), (1, 1), (2, 2))):
            degrees = [(0.0, 0.0)] * 3
            degrees[axis] = (30.0, 30.0)
            augmentation = K.RandomRotation3D(tuple(degrees), p=1.0, align_corners=True)
            augmentation(volume)
            assert augmentation.transform_matrix[0, diagonal[0], diagonal[1]] == 1

    def test_convention_crop_size_padding_defaults_and_batch_gate(self, device, dtype):
        if not supports_bilinear_3d_grid_sample(device, dtype):
            pytest.skip("bilinear 3D grid_sample is unavailable for this device and dtype")
        volume = torch.arange(120, device=device, dtype=dtype).reshape(1, 1, 4, 5, 6)
        center = K.CenterCrop3D((2, 3, 4), p=1.0)
        crop = K.RandomCrop3D((2, 3, 4), p=1.0, same_on_batch=True)
        assert center.flags["align_corners"] and crop.flags["align_corners"]
        assert center.flags["resample"].name == crop.flags["resample"].name == "BILINEAR"
        assert center(volume).shape == crop(volume).shape == (1, 1, 2, 3, 4)
        self.assert_close(center(volume), volume[..., 1:3, 1:4, 1:5])
        self.assert_close(K.CenterCrop3D((2, 3, 4), p=0.0)(volume), volume, rtol=0, atol=0)
        self.assert_close(K.RandomCrop3D((2, 3, 4), p=0.0)(volume), volume, rtol=0, atol=0)
        padded = K.RandomCrop3D((2, 3, 4), padding=(1, 2, 3, 4, 5, 6), p=1.0).precrop_padding(volume)
        assert padded.shape[-3:] == (15, 12, 9)
        self.assert_close(padded[..., 5:9, 3:8, 1:7], volume)
        padded = K.RandomCrop3D((2, 3, 4), padding=(1, 2, 3), p=1.0).precrop_padding(volume)
        assert padded.shape[-3:] == (10, 9, 8)
        self.assert_close(padded[..., 3:7, 2:7, 1:7], volume)
        padded = K.RandomCrop3D((2, 3, 4), padding=1, p=1.0).precrop_padding(volume)
        assert padded.shape[-3:] == (6, 7, 8)
        self.assert_close(padded[..., 1:5, 1:6, 1:7], volume)
        # The gate is call-wide: it never selects a strict subset of the rows.
        for augmentation in (K.CenterCrop3D((2, 3, 4), p=0.5), K.RandomCrop3D((2, 3, 4), p=0.5)):
            for _ in range(20):
                batch_prob = augmentation.forward_parameters(torch.Size([8, 1, 4, 5, 6]))["batch_prob"]
                assert batch_prob.numel() == 8
                assert bool((batch_prob == batch_prob[0]).all())

    def test_convention_random_crop3d_uses_sampled_nonzero_offset(self, device, dtype):
        if not supports_nearest_3d_grid_sample(device, dtype):
            pytest.skip("nearest 3D grid_sample is unavailable for this device and dtype")
        volume = torch.arange(120, device=device, dtype=dtype).reshape(1, 1, 4, 5, 6)
        augmentation = K.RandomCrop3D((2, 3, 4), resample="nearest", p=1.0)
        params = augmentation.forward_parameters(volume.shape)
        # The source cube starts at (x, y, z) = (1, 1, 1), deliberately excluding the top-left crop.
        params["src"] = torch.tensor(
            [[[1, 1, 1], [4, 1, 1], [4, 3, 1], [1, 3, 1], [1, 1, 2], [4, 1, 2], [4, 3, 2], [1, 3, 2]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(augmentation(volume, params=params), volume[..., 1:3, 1:4, 1:5])

    def test_convention_geometric_defaults_and_identity(self, device, dtype):
        if not supports_bilinear_3d_grid_sample(device, dtype):
            pytest.skip("bilinear 3D grid_sample is unavailable for this device and dtype")
        volume = torch.arange(60, device=device, dtype=dtype).reshape(1, 1, 3, 4, 5) / 64
        affine = K.RandomAffine3D((0.0, 0.0, 0.0), p=1.0)
        rotation = K.RandomRotation3D((0.0, 0.0, 0.0), p=1.0)
        perspective = K.RandomPerspective3D(0.0, p=1.0)
        assert not affine.flags["align_corners"] and not rotation.flags["align_corners"]
        assert not perspective.flags["align_corners"]
        assert (
            affine.flags["resample"].name
            == rotation.flags["resample"].name
            == perspective.flags["resample"].name
            == "BILINEAR"
        )
        self.assert_close(affine(volume), volume)
        self.assert_close(rotation(volume), volume)

    def test_wart_random_perspective3d_identity_align_corners_false_4503(self, device, dtype):
        # #4503: the default false-setting perspective path still mixes coordinate normalizations. A zero
        # distortion matrix is identity, but a non-constant volume is resampled away from its voxel centres.
        if not supports_bilinear_3d_grid_sample(device, dtype):
            pytest.skip("bilinear 3D grid_sample is unavailable for this device and dtype")
        volume = torch.arange(60, device=device, dtype=dtype).reshape(1, 1, 3, 4, 5)
        default = K.RandomPerspective3D(0.0, p=1.0)
        aligned = K.RandomPerspective3D(0.0, p=1.0, align_corners=True)
        assert (default(volume) - volume).abs().max() > 1
        self.assert_close(aligned(volume), volume)

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason="3D normalization ignores align_corners, #4503")
    def test_convention_random_perspective3d_identity_align_corners_false(self, device, dtype):
        if not supports_bilinear_3d_grid_sample(device, dtype):
            pytest.skip("bilinear 3D grid_sample is unavailable for this device and dtype")
        volume = torch.arange(60, device=device, dtype=dtype).reshape(1, 1, 3, 4, 5)
        self.assert_close(K.RandomPerspective3D(0.0, p=1.0)(volume), volume)

    @pytest.mark.device_agnostic
    def test_wart_positive_roll_direction_splits_the_rotation_entry_points_4408(self):
        # An off-centre marker in a 7 x 7 slice: a counter-clockwise quarter turn sends (row 1, col 4) to (2, 1),
        # a clockwise one to (4, 5).
        image = torch.zeros(1, 1, 7, 7)
        image[0, 0, 1, 4] = 1
        volume = image[:, :, None].repeat(1, 1, 7, 1, 1)
        zero, quarter = (0.0, 0.0), (90.0, 90.0)

        def marker(plane):
            return divmod(int(plane.flatten().argmax()), 7)

        assert marker(kornia.geometry.transform.rotate(image, torch.tensor([90.0]))[0, 0]) == (2, 1)
        assert marker(K.RandomRotation(quarter, p=1.0)(image)[0, 0]) == (2, 1)
        assert marker(K.RandomAffine3D((zero, zero, quarter), p=1.0)(volume)[0, 0, 3]) == (2, 1)
        assert marker(K.RandomAffine(quarter, p=1.0)(image)[0, 0]) == (4, 5)
        assert marker(K.RandomRotation3D((zero, zero, quarter), p=1.0)(volume)[0, 0, 3]) == (4, 5)
        angles = [torch.tensor([value]) for value in (0.0, 0.0, 90.0)]
        assert marker(kornia.geometry.transform.rotate3d(volume, *angles)[0, 0, 3]) == (4, 5)

    def test_wart_random_affine3d_rotation_sign_4408(self, device, dtype):
        # #4408: the affine composer negates the angle relative to RandomRotation3D; pixel directions are pinned above.
        if not supports_bilinear_3d_grid_sample(device, dtype):
            pytest.skip("bilinear 3D grid_sample is unavailable for this device and dtype")
        volume = torch.zeros(1, 1, 5, 5, 5, device=device, dtype=dtype)
        affine = K.RandomAffine3D(((0.0, 0.0), (0.0, 0.0), (30.0, 30.0)), p=1.0)
        rotation = K.RandomRotation3D(((0.0, 0.0), (0.0, 0.0), (30.0, 30.0)), p=1.0)
        affine(volume)
        rotation(volume)
        self.assert_close(affine.transform_matrix[:, :3, :3], rotation.transform_matrix[:, :3, :3].transpose(-1, -2))

    @pytest.mark.parametrize(
        ("axis", "rotation_position", "affine_position"),
        [(0, (1, 4, 3), (3, 2, 3)), (1, (3, 2, 3), (1, 2, 5)), (2, (1, 2, 5), (1, 4, 3))],
    )
    def test_convention_affine_and_rotation3d_move_an_asymmetric_marker(
        self, axis, rotation_position, affine_position, device, dtype
    ):
        if not supports_nearest_3d_grid_sample(device, dtype):
            pytest.skip("nearest 3D grid_sample is unavailable for this device and dtype")
        # Odd, unequal D/H/W dimensions put the rotation centre on voxels and make all axes observable.
        volume = torch.zeros(1, 1, 5, 7, 9, device=device, dtype=dtype)
        volume[..., 1, 2, 3] = 1
        degrees = [(0.0, 0.0)] * 3
        degrees[axis] = (90.0, 90.0)
        rotation = K.RandomRotation3D(tuple(degrees), resample="nearest", p=1.0, align_corners=True)
        affine = K.RandomAffine3D(tuple(degrees), resample="nearest", p=1.0, align_corners=True)
        rotated, affined = rotation(volume), affine(volume)
        expected_rotation, expected_affine = torch.zeros_like(volume), torch.zeros_like(volume)
        expected_rotation[(0, 0, *rotation_position)] = 1
        expected_affine[(0, 0, *affine_position)] = 1
        self.assert_close(rotated, expected_rotation)
        self.assert_close(affined, expected_affine)

    def test_convention_equalize_small_volume_roundoff(self, device, dtype):
        small = torch.linspace(0.1, 0.9, 255, device=device, dtype=dtype).reshape(1, 1, 1, 1, 255)
        augmentation = K.RandomEqualize3D(p=1.0)
        self.assert_close(augmentation(small), small, rtol=0, atol=torch.finfo(dtype).eps)
        self.assert_close(augmentation.transform_matrix, torch.eye(4, device=device, dtype=dtype)[None])
        # 256 voxels is the smallest volume the lookup can change: 128 zeros and 128 singly occupied bins.
        ramp = torch.arange(1, 129, device=device, dtype=dtype)
        large = torch.cat([torch.zeros_like(ramp), ramp / 255]).reshape(1, 1, 1, 1, 256)
        expected = torch.cat([torch.zeros_like(ramp), (ramp + 127) / 255]).reshape(1, 1, 1, 1, 256)
        self.assert_close(augmentation(large), expected)

    @pytest.mark.device_agnostic
    def test_convention_equalize_range_check_on_cpu(self):
        with pytest.raises(RuntimeError, match="values in \\[0, 1\\]"):
            K.RandomEqualize3D(p=1.0)(torch.full((1, 1, 1, 1, 2), -0.1))

    @pytest.mark.device_agnostic
    def test_convention_motion_blur_defaults_and_probability(self):
        augmentation = K.RandomMotionBlur3D(3, (0.0, 0.0, 0.0), 0.0, p=0.5)
        assert augmentation.flags["resample"].name == "NEAREST"
        assert augmentation.p_batch == 1.0
        assert augmentation.p == 0.5

    def test_convention_motion_blur3d_fixed_kernel_filters_an_impulse(self, device, dtype):
        volume = torch.zeros(1, 1, 5, 5, 5, device=device, dtype=dtype)
        volume[..., 2, 2, 2] = 1
        output = K.RandomMotionBlur3D(3, (0.0, 0.0, 0.0), 0.0, p=1.0)(volume)
        expected = torch.zeros_like(volume)
        expected[..., 2, 2, 1:4] = 1 / 3
        self.assert_close(output, expected)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        ("axis", "line"),
        [(0, (2, 2, slice(1, 4))), (1, (slice(1, 4), 2, 2)), (2, (2, slice(1, 4), 2))],
        ids=["yaw-about-x", "pitch-about-y", "roll-about-z"],
    )
    def test_convention_motion_blur3d_kernel_uses_the_yaw_pitch_roll_order(self, axis, line):
        # The zero-angle kernel is a line along x (W). A quarter turn about x leaves it in place, one about y
        # stands it along D, and one about z lays it along H -- so the three angle slots are distinguishable.
        volume = torch.zeros(1, 1, 5, 5, 5)
        volume[..., 2, 2, 2] = 1
        angle = [(0.0, 0.0)] * 3
        angle[axis] = (90.0, 90.0)
        output = K.RandomMotionBlur3D(3, tuple(angle), (0.0, 0.0), p=1.0)(volume)
        expected = torch.zeros_like(volume)
        expected[(0, 0, *line)] = 1 / 3
        self.assert_close(output, expected, atol=1e-4, rtol=0)

    @pytest.mark.device_agnostic
    def test_convention_3d_augmentations_have_no_direct_inverse(self):
        augmentations = (
            K.RandomAffine3D((0.0, 0.0, 0.0)),
            K.CenterCrop3D((2, 2, 2)),
            K.RandomCrop3D((2, 2, 2)),
            K.RandomDepthicalFlip3D(),
            K.RandomHorizontalFlip3D(),
            K.RandomPerspective3D(),
            K.RandomRotation3D((0.0, 0.0, 0.0)),
            K.RandomVerticalFlip3D(),
            K.RandomEqualize3D(),
            K.RandomMotionBlur3D(3, (0.0, 0.0, 0.0), 0.0),
        )
        assert all(not hasattr(augmentation, "inverse") for augmentation in augmentations)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_convention_rotation3d_sign_is_literal_and_affine_is_opposite_4408(self, axis):
        # The matrix literal anchors RandomRotation3D's own direction; #4408 is RandomAffine3D being its transpose.
        degrees = [(0.0, 0.0)] * 3
        degrees[axis] = (30.0, 30.0)
        volume = torch.zeros(1, 1, 5, 5, 5)
        rotation = K.RandomRotation3D(tuple(degrees), p=1.0, align_corners=True)
        affine = K.RandomAffine3D(tuple(degrees), p=1.0, align_corners=True)
        rotation(volume)
        affine(volume)
        first, second = [index for index in range(3) if index != axis]
        matrix = rotation.transform_matrix[0, :3, :3]
        self.assert_close(matrix[first, second].abs(), torch.tensor(0.5))
        self.assert_close(matrix[first, second], -matrix[second, first])
        expected_sign = {0: -1.0, 1: 1.0, 2: -1.0}[axis]
        assert torch.sign(matrix[first, second]) == expected_sign
        self.assert_close(affine.transform_matrix[0, :3, :3], matrix.transpose(-1, -2))

    @pytest.mark.device_agnostic
    def test_convention_random_crop3d_returns_the_unpadded_input_when_gated_off(self):
        # Fixed by #4667: a gated-off call used to return the padded volume (#4654).
        volume = torch.rand(1, 1, 4, 5, 6)
        for kwargs in (
            {"size": (2, 3, 4), "padding": 2},
            {"size": (6, 7, 9), "pad_if_needed": True},
            {"size": (2, 3, 4)},
        ):
            augmentation = K.RandomCrop3D(p=0.0, **kwargs)
            self.assert_close(augmentation(volume), volume, rtol=0, atol=0)
            self.assert_close(augmentation.transform_matrix, torch.eye(4)[None])
        assert K.RandomCrop3D((2, 3, 4), padding=2, p=1.0)(volume).shape == (1, 1, 2, 3, 4)

    @pytest.mark.device_agnostic
    def test_convention_crop3d_validates_size_before_a_disabled_gate(self):
        volume = torch.rand(1, 1, 4, 5, 6)
        with pytest.raises(AssertionError, match="Crop size must be smaller"):
            K.CenterCrop3D((5, 6, 7), p=0.0)(volume)
        with pytest.raises(ValueError, match="cannot be smaller than crop size"):
            K.RandomCrop3D((9, 9, 9), padding=1, p=0.0)(volume)
        # (6, 5, 6) fits the padded (6, 7, 8) volume but not the (4, 5, 6) input: validation sees the padded one.
        self.assert_close(K.RandomCrop3D((6, 5, 6), padding=1, p=0.0)(volume), volume, rtol=0, atol=0)
        with pytest.raises(ValueError, match="cannot be smaller than crop size"):
            K.RandomCrop3D((6, 5, 6), p=0.0)(volume)

    @pytest.mark.parametrize(
        "padding,size,marker",
        [(1, (5, 5, 5), (2, 2, 2)), ((1, 2, 3), (9, 7, 5), (4, 3, 2))],
    )
    def test_convention_random_crop3d_matrix_uses_the_padded_source_frame(self, padding, size, marker, device, dtype):
        if not supports_nearest_3d_grid_sample(device, dtype):
            pytest.skip("nearest 3D grid_sample is unavailable for this device and dtype")
        volume = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=dtype)
        volume[..., 1, 1, 1] = 1
        # Nearest sampling isolates the coordinate frame from half-precision interpolation roundoff.
        augmentation = K.RandomCrop3D(size, padding=padding, resample="nearest", p=1.0)
        output = augmentation(volume)
        expected = torch.zeros_like(output)
        expected[..., marker[0], marker[1], marker[2]] = 1
        self.assert_close(output, expected)
        # Taking the whole padded canvas records identity, despite moving the original marker by the padding.
        self.assert_close(augmentation.transform_matrix, torch.eye(4, device=device, dtype=dtype)[None])

    @pytest.mark.device_agnostic
    def test_convention_motion_blur3d_kernel_range_is_drawn_once_per_call_bounds_included(self):
        # Fixed by #4662 and #4674: the size used to be drawn per sample, which raised for B > 1, and never
        # reached the upper bound (#4653).
        volume = torch.rand(6, 1, 4, 5, 6)
        sizes = set()
        for seed in range(40):
            torch.manual_seed(seed)
            augmentation = K.RandomMotionBlur3D((3, 7), 35.0, 0.5, p=1.0)
            assert augmentation(volume).shape == volume.shape
            drawn = augmentation._params["ksize_factor"]
            assert drawn.shape == (6,) and drawn.unique().numel() == 1
            sizes.update(drawn.tolist())
        assert sizes == {3, 5, 7}
        rounded_up = K.RandomMotionBlur3D((4, 4), 35.0, 0.5, p=1.0)
        rounded_up(volume)
        assert rounded_up._params["ksize_factor"].tolist() == [5] * 6
        with pytest.raises(ValueError, match="smaller than or equal to"):
            K.RandomMotionBlur3D((7, 3), 35.0, 0.5)

    @pytest.mark.device_agnostic
    def test_convention_motion_blur3d_positive_roll_is_clockwise(self):
        # #4408's split reaches motion blur too: the 3D roll mirrors the 2D angle.
        def tilt(plane):
            nonzero = (plane.abs() > 1e-4).nonzero().float()
            centred = nonzero - nonzero.mean(0)
            return float((centred[:, 0] * centred[:, 1]).sum())

        volume = torch.zeros(1, 1, 3, 15, 15)
        volume[0, 0, 1, 7, 7] = 1.0
        image = torch.zeros(1, 1, 15, 15)
        image[0, 0, 7, 7] = 1.0
        rolled = K.RandomMotionBlur3D(9, ((0.0, 0.0), (0.0, 0.0), (30.0, 30.0)), 0.0, p=1.0)(volume)[0, 0, 1]
        flat = K.RandomMotionBlur(9, (30.0, 30.0), 0.0, p=1.0)(image)[0, 0]
        assert tilt(rolled) > 0  # rows increase with columns: clockwise as displayed
        assert tilt(flat) < 0  # the 2D class turns the other way
        assert tilt(rolled) == -tilt(flat)

    @pytest.mark.device_agnostic
    def test_convention_only_transplantation3d_exposes_p_batch(self):
        import inspect

        def concrete(cls):
            for sub in cls.__subclasses__():
                # Library classes only: other test modules define their own 3D subclasses in the same process.
                if sub.__module__.startswith("kornia.") and not sub.__name__.endswith("Base3D"):
                    yield sub
                yield from concrete(sub)

        # Enumerate the subclasses rather than a hand-written list or `__all__`: RandomTransplantation3D was
        # missing from kornia.augmentation.__all__ until #4695, and is the one 3D class that does take p_batch.
        classes = set(concrete(K.AugmentationBase3D))
        assert {K.RandomAffine3D, K.CenterCrop3D, K.RandomEqualize3D, K.RandomTransplantation3D} <= classes
        exposing = {cls.__name__ for cls in classes if "p_batch" in inspect.signature(cls.__init__).parameters}
        assert exposing == {"RandomTransplantation3D"}
        with pytest.raises(TypeError, match="p_batch"):
            K.RandomHorizontalFlip3D(p=1.0, p_batch=0.5)
        volume = torch.rand(4, 1, 3, 4, 5)
        mask = torch.randint(0, 3, (4, 3, 4, 5))
        skipped, _ = K.RandomTransplantation3D(p=1.0, p_batch=0.0)(volume, mask, data_keys=["input", "mask"])
        self.assert_close(skipped, volume, rtol=0, atol=0)

    @pytest.mark.device_agnostic
    def test_wart_random_crop3d_accepts_a_one_voxel_oversized_crop_4688(self):
        volume = torch.ones(1, 1, 4, 5, 6)
        output = K.RandomCrop3D((5, 5, 6), p=1.0)(volume)
        assert output.shape == (1, 1, 5, 5, 6)
        # The appended slab carries no input signal. It is compared against a real slice rather than
        # against zero: torch 2.5.1 and 2.9.1 leave ~1e-6 of grid_sample roundoff there, 2.14 leaves 0.
        slices = output.sum(dim=(1, 3, 4))[0]
        assert float(slices[-1]) < 1e-3 * float(slices[0])
        with pytest.raises(ValueError, match="cannot be smaller than crop size"):
            K.RandomCrop3D((6, 5, 6), p=1.0)(volume)
        # CenterCrop3D rejects the same one-voxel request.
        with pytest.raises(AssertionError, match="Crop size must be smaller"):
            K.CenterCrop3D((5, 5, 6), p=1.0)(volume)

    @pytest.mark.device_agnostic
    def test_convention_perspective3d_identity_is_only_float32_grid_precise(self):
        volume = torch.rand(1, 1, 4, 5, 6, dtype=torch.float64)
        rotation = float((K.RandomRotation3D(0.0, p=1.0, align_corners=True)(volume) - volume).abs().max())
        residual = float((K.RandomPerspective3D(0.0, p=1.0, align_corners=True)(volume) - volume).abs().max())
        # The rotation path is exact to float64 roundoff (0 on torch 2.14, ~2e-16 on 2.5.1 and 2.9.1);
        # the perspective path is off by ~1e-7 because its sampling grid is built in float32.
        assert rotation < 1e-12
        # An absolute window rather than a multiple of `rotation`: where the rotation is exactly 0 a relative
        # bound collapses to "> 1e-16", under one float64 ULP. 1e-9 has two orders of headroom on either side.
        assert 1e-9 < residual < 1e-5

    @pytest.mark.device_agnostic
    def test_convention_random_crop3d_offset_reaches_both_ends(self):
        # The offset pin overwrites params["src"], so the sampler's inclusive range is never measured.
        aug = K.RandomCrop3D((2, 3, 4), p=1.0)
        starts = [set(), set(), set()]
        for _ in range(20):
            src = aug.forward_parameters(torch.Size([8, 1, 4, 5, 6]))["src"]
            for axis in range(3):
                starts[axis].update(int(value) for value in src[:, 0, axis])
        # x: 6 - 4, y: 5 - 3, z: 4 - 2 -> the inclusive range [0, 2] on every axis, each with its own bound.
        assert starts == [{0, 1, 2}] * 3

    @pytest.mark.device_agnostic
    def test_convention_random_crop3d_fixed_padding_feeds_pad_if_needed(self):
        # No other pin exercises `padding` and `pad_if_needed` together, which is where the axis
        # bookkeeping for the second padding pass lives.
        volume = torch.rand(1, 1, 4, 5, 6)
        aug = K.RandomCrop3D((12, 3, 3), padding=(0, 0, 0, 0, 3, 3), pad_if_needed=True, p=1.0)
        assert aug._compute_padding(tuple(volume.shape), aug.flags) == [[0, 0, 0, 0, 3, 3], [0, 0, 0, 0, 2, 2]]
        assert aug.precrop_padding(volume).shape[-3:] == (14, 5, 6)
        # Six different paddings and three different deficits, so that each axis has to read its own pair.
        aug = K.RandomCrop3D((16, 14, 12), padding=(1, 2, 3, 4, 5, 6), pad_if_needed=True, p=1.0)
        assert aug._compute_padding(tuple(volume.shape), aug.flags) == [
            [1, 2, 3, 4, 5, 6],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 2, 2, 0, 0],
            [3, 3, 0, 0, 0, 0],
        ]
        assert aug.precrop_padding(volume).shape[-3:] == (17, 16, 15)

    @pytest.mark.device_agnostic
    def test_convention_motion_blur3d_rejects_a_mixed_replayed_kernel_size(self):
        # The kernel-size pin measures the generator; the consumer's rejection of a mixed replay is separate.
        volume = torch.rand(2, 1, 5, 5, 5)
        aug = K.RandomMotionBlur3D((3, 7), 35.0, 0.5, p=1.0)
        params = aug.forward_parameters(volume.shape)
        params["ksize_factor"] = torch.tensor([3, 7], dtype=torch.int32)
        with pytest.raises(RuntimeError):
            aug(volume, params=params)

    @pytest.mark.device_agnostic
    def test_convention_random_affine3d_two_value_scale_is_isotropic_4704(self):
        # A two-value scale draws one factor per sample for all three axes, as the 2D class does; the
        # three-pair form keeps one independent draw per axis.
        flat = K.RandomAffine(0.0, scale=(0.5, 2.0), p=1.0).forward_parameters(torch.Size([64, 1, 6, 7]))["scale"]
        assert bool((flat[:, 0] == flat[:, 1]).all())
        aug = K.RandomAffine3D(0.0, scale=(0.5, 2.0), p=1.0)
        aug(torch.rand(64, 1, 5, 6, 7))
        scale = aug._params["scale"]
        assert scale.shape == (64, 3)
        assert bool((scale[:, 0] == scale[:, 1]).all()) and bool((scale[:, 1] == scale[:, 2]).all())
        assert 0.5 <= float(scale.min()) and float(scale.max()) <= 2.0
        assert len(scale[:, 0].unique()) > 1
        self.assert_close(aug.transform_matrix[:, :3, :3].diagonal(dim1=-2, dim2=-1), scale)
        shared = K.RandomAffine3D(0.0, scale=(0.5, 2.0), same_on_batch=True, p=1.0)
        shared_scale = shared.forward_parameters(torch.Size([64, 1, 5, 6, 7]))["scale"]
        assert len(shared_scale.unique()) == 1
        per_axis = K.RandomAffine3D(0.0, scale=((0.5, 2.0), (0.5, 2.0), (0.5, 2.0)), p=1.0)
        axes = per_axis.forward_parameters(torch.Size([64, 1, 5, 6, 7]))["scale"]
        assert axes.shape == (64, 3)
        assert not bool((axes[:, 0] == axes[:, 1]).any()) and not bool((axes[:, 1] == axes[:, 2]).any())

    @pytest.mark.device_agnostic
    def test_convention_random_affine3d_six_pair_shears_keep_their_lower_bound(self):
        # The Args text used to negate the lower bound of each pair.
        pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12))
        params = K.RandomAffine3D(0.0, shears=pairs, p=1.0).forward_parameters(torch.Size([64, 1, 4, 5, 6]))
        for key, (low, high) in zip(("sxy", "sxz", "syx", "syz", "szx", "szy"), pairs):
            assert low <= float(params[key].min()) and float(params[key].max()) <= high

    @pytest.mark.device_agnostic
    def test_convention_random_crop3d_padding_modes_are_those_of_f_pad(self):
        volume = torch.rand(2, 1, 4, 5, 6)
        for mode in ("constant", "reflect", "replicate", "circular"):
            assert K.RandomCrop3D((5, 6, 7), padding=1, padding_mode=mode, p=1.0)(volume).shape == (2, 1, 5, 6, 7)
        for mode in ("edge", "symmetric"):
            with pytest.raises(NotImplementedError, match="Unrecognised padding mode"):
                K.RandomCrop3D((5, 6, 7), padding=1, padding_mode=mode, p=1.0)(volume)

    @pytest.mark.device_agnostic
    def test_convention_3d_generator_keys_match_their_returns_sections(self):
        from kornia.augmentation import random_generator as rg

        shape = torch.Size([3, 1, 4, 5, 6])
        assert rg.AffineGenerator3D(10.0)(shape)["angles"].shape == (3, 3)
        perspective = rg.PerspectiveGenerator3D(0.3)(shape)
        assert set(perspective) == {"start_points", "end_points"}
        assert perspective["start_points"].shape == perspective["end_points"].shape == (3, 8, 3)
        blur = rg.MotionBlurGenerator3D(3, 35.0, 0.5)(shape)
        assert blur["angle_factor"].shape == (3, 3) and blur["ksize_factor"].shape == (3,)
        # "odd and at least 3": a kernel of one voxel is rejected.
        with pytest.raises(AssertionError, match="must be odd and greater than 3"):
            K.RandomMotionBlur3D(1, 35.0, 0.5, p=1.0)(torch.rand(1, 1, 5, 5, 5))

    @pytest.mark.device_agnostic
    def test_convention_identity_warp_error_grows_with_size_outside_full_precision(self):
        # "Up to roundoff" is a float32 / float64 statement: in half precision the sampling grid is rounded.
        large = torch.rand(1, 1, 32, 48, 96)
        assert float((K.RandomRotation3D(0.0, p=1.0)(large) - large).abs().max()) < 1e-3
        if supports_bilinear_3d_grid_sample(torch.device("cpu"), torch.bfloat16):
            half = large.to(torch.bfloat16)
            for aug in (K.RandomRotation3D(0.0, p=1.0), K.RandomAffine3D(0.0, p=1.0)):
                assert float((aug(half) - half).abs().max()) > 0.1
        # The float32 grid of the perspective path shows at float64 too, and also grows with the volume.
        small64 = torch.rand(1, 1, 4, 5, 6, dtype=torch.float64)
        large64 = torch.rand(1, 1, 8, 16, 32, dtype=torch.float64)
        residuals = [
            float((K.RandomPerspective3D(0.0, p=1.0, align_corners=True)(volume) - volume).abs().max())
            for volume in (small64, large64)
        ]
        assert residuals[0] < residuals[1] < 1e-4

    @pytest.mark.device_agnostic
    def test_convention_center_crop3d_centres_each_axis_with_its_own_offset(self):
        # Offsets (z, y, x) = (1, 2, 4), all different: a (1, 1, 1) fixture cannot tell the axes apart.
        volume = torch.arange(6 * 8 * 12.0).reshape(1, 1, 6, 8, 12)
        output = K.CenterCrop3D((4, 4, 4), resample="nearest", p=1.0)(volume)
        self.assert_close(output, volume[..., 1:5, 2:6, 4:8], rtol=0, atol=0)

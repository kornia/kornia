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

import pytest
import torch

import kornia.augmentation as K

from testing.base import BaseTester, supports_bilinear_3d_grid_sample


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
        volume = torch.arange(120, device=device, dtype=dtype).reshape(1, 1, 4, 5, 6)
        center = K.CenterCrop3D((2, 3, 4), p=1.0)
        crop = K.RandomCrop3D((2, 3, 4), p=1.0, same_on_batch=True)
        assert center.flags["align_corners"] and crop.flags["align_corners"]
        assert center.flags["resample"].name == crop.flags["resample"].name == "BILINEAR"
        assert center(volume).shape == crop(volume).shape == (1, 1, 2, 3, 4)
        assert K.CenterCrop3D((2, 3, 4), p=0.0)(volume).shape == volume.shape
        assert K.RandomCrop3D((2, 3, 4), p=0.0)(volume).shape == volume.shape
        padded = K.RandomCrop3D((2, 3, 4), padding=(1, 2, 3, 4, 5, 6), p=1.0).precrop_padding(volume)
        assert padded.shape[-3:] == (15, 12, 9)
        self.assert_close(padded[..., 5:9, 3:8, 1:7], volume)

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

    def test_wart_random_affine3d_rotation_sign_4408(self, device, dtype):
        # #4408: the affine composer negates the angle relative to RandomRotation3D's displayed convention.
        if not supports_bilinear_3d_grid_sample(device, dtype):
            pytest.skip("bilinear 3D grid_sample is unavailable for this device and dtype")
        volume = torch.zeros(1, 1, 5, 5, 5, device=device, dtype=dtype)
        affine = K.RandomAffine3D(((0.0, 0.0), (0.0, 0.0), (30.0, 30.0)), p=1.0)
        rotation = K.RandomRotation3D(((0.0, 0.0), (0.0, 0.0), (30.0, 30.0)), p=1.0)
        affine(volume)
        rotation(volume)
        self.assert_close(affine.transform_matrix[:, :3, :3], rotation.transform_matrix[:, :3, :3].transpose(-1, -2))

    def test_convention_equalize_small_volume_roundoff(self, device, dtype):
        small = torch.linspace(0.1, 0.9, 255, device=device, dtype=dtype).reshape(1, 1, 1, 1, 255)
        self.assert_close(K.RandomEqualize3D(p=1.0)(small), small, rtol=0, atol=torch.finfo(dtype).eps)

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

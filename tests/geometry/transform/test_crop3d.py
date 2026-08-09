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

import kornia
from kornia.core._compat import torch_version

from testing.base import BaseTester


class TestCropAndResize3D(BaseTester):
    def test_crop(self, device, dtype):
        inp = torch.arange(0.0, 64.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4)

        depth, height, width = 2, 2, 2
        expected = torch.tensor(
            [[[[[25.1667, 27.1667], [30.5000, 32.5000]], [[46.5000, 48.5000], [51.8333, 53.8333]]]]],
            device=device,
            dtype=dtype,
        )

        boxes = torch.tensor(
            [[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        patches = kornia.geometry.transform.crop_and_resize3d(inp, boxes, (depth, height, width))
        self.assert_close(patches, expected)

    def test_crop_batch(self, device, dtype):
        inp = torch.cat(
            [
                torch.arange(0.0, 64.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4),
                torch.arange(0.0, 128.0, step=2, device=device, dtype=dtype).view(1, 1, 4, 4, 4),
            ],
            dim=0,
        )

        depth, height, width = 2, 2, 2
        expected = torch.tensor(
            [
                [[[[16.0000, 19.0000], [24.0000, 27.0000]], [[48.0000, 51.0000], [56.0000, 59.0000]]]],
                [[[[0.0000, 6.0000], [16.0000, 22.0000]], [[64.0000, 70.0000], [80.0000, 86.0000]]]],
            ],
            device=device,
            dtype=dtype,
        )

        boxes = torch.tensor(
            [
                [[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]],
                [[0, 0, 0], [3, 0, 0], [3, 2, 0], [0, 2, 0], [0, 0, 2], [3, 0, 2], [3, 2, 2], [0, 2, 2]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x8x3

        patches = kornia.geometry.transform.crop_and_resize3d(inp, boxes, (depth, height, width), align_corners=True)
        self.assert_close(patches, expected)

    def test_gradcheck(self, device):
        img = torch.arange(0.0, 64.0, device=device, dtype=torch.float64).view(1, 1, 4, 4, 4)

        boxes = torch.tensor(
            [[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]],
            device=device,
            dtype=torch.float64,
        )  # 1x8x3

        self.gradcheck(kornia.geometry.transform.crop_and_resize3d, (img, boxes, (4, 3, 2)))

    def test_dynamo(self, device, dtype, torch_optimizer):
        # Define script
        op = kornia.geometry.transform.crop_and_resize3d
        op_script = torch_optimizer(op)

        img = torch.arange(0.0, 64.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4)

        boxes = torch.tensor(
            [[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        actual = op_script(img, boxes, (4, 3, 2))
        expected = op(img, boxes, (4, 3, 2))
        self.assert_close(actual, expected)


class TestCenterCrop3D(BaseTester):
    @pytest.mark.parametrize("crop_size", [(3, 5, 7), (5, 3, 7), (7, 3, 5)])
    def test_center_crop_357(self, crop_size, device, dtype):
        inp = torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        expected = inp[
            :,
            :,
            (inp.size(2) // 2 - crop_size[0] // 2) : (inp.size(2) // 2 + crop_size[0] // 2 + 1),
            (inp.size(3) // 2 - crop_size[1] // 2) : (inp.size(3) // 2 + crop_size[1] // 2 + 1),
            (inp.size(4) // 2 - crop_size[2] // 2) : (inp.size(4) // 2 + crop_size[2] // 2 + 1),
        ]
        out_crop = kornia.geometry.transform.center_crop3d(inp, crop_size, align_corners=True)
        self.assert_close(out_crop, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("crop_size", [(3, 5, 7), (5, 3, 7), (7, 3, 5)])
    def test_center_crop_357_batch(self, crop_size, device, dtype):
        inp = torch.cat(
            [
                torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7),
                torch.arange(343.0, 686.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7),
            ]
        )
        expected = inp[
            :,
            :,
            (inp.size(2) // 2 - crop_size[0] // 2) : (inp.size(2) // 2 + crop_size[0] // 2 + 1),
            (inp.size(3) // 2 - crop_size[1] // 2) : (inp.size(3) // 2 + crop_size[1] // 2 + 1),
            (inp.size(4) // 2 - crop_size[2] // 2) : (inp.size(4) // 2 + crop_size[2] // 2 + 1),
        ]
        out_crop = kornia.geometry.transform.center_crop3d(inp, crop_size, align_corners=True)
        self.assert_close(out_crop, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        img = torch.arange(0.0, 343.0, device=device, dtype=torch.float64).view(1, 1, 7, 7, 7)

        self.gradcheck(kornia.geometry.transform.center_crop3d, (img, (3, 5, 7)))

    @pytest.mark.skipif(
        torch_version() == "2.1.0",
        reason=(
            "https://github.com/pytorch/pytorch/issues/110680"
            " -  unsupported operand type(s) for @: 'FakeTensor' and 'FakeTensor' on `normalize_homography3d`"
        ),
    )
    def test_dynamo(self, device, dtype, torch_optimizer):
        # Define script
        op = kornia.geometry.transform.center_crop3d
        op_script = torch_optimizer(op)
        img = torch.ones(4, 3, 5, 6, 7, device=device, dtype=dtype)

        actual = op_script(img, (4, 3, 2))
        expected = kornia.geometry.transform.center_crop3d(img, (4, 3, 2))
        self.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_convention_align_corners_default_is_true(self, device, dtype):
        # center_crop3d's align_corners default is True, not merely a value that every existing
        # test happens to pass explicitly: omitting the kwarg must equal align_corners=True and
        # differ from align_corners=False.
        inp = torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        out_default = kornia.geometry.transform.center_crop3d(inp, (3, 3, 3))
        out_true = kornia.geometry.transform.center_crop3d(inp, (3, 3, 3), align_corners=True)
        out_false = kornia.geometry.transform.center_crop3d(inp, (3, 3, 3), align_corners=False)
        self.assert_close(out_default, out_true, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_default, out_false, atol=1e-2, rtol=1e-2)


class TestCropByBoxes3D(BaseTester):
    def test_crop_by_boxes_no_resizing(self, device, dtype):
        inp = torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        src_box = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                    [3.0, 3.0, 1.0],
                    [1.0, 3.0, 1.0],
                    [1.0, 1.0, 2.0],
                    [3.0, 1.0, 2.0],
                    [3.0, 3.0, 2.0],
                    [1.0, 3.0, 2.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1x8x3
        dst_box = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 2.0, 1.0],
                    [0.0, 2.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        expected = inp[:, :, 1:3, 1:4, 1:4]

        patches = kornia.geometry.transform.crop_by_boxes3d(inp, src_box, dst_box, align_corners=True)
        self.assert_close(patches, expected, rtol=1e-4, atol=1e-4)

    def test_crop_by_boxes_resizing(self, device, dtype):
        inp = torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        src_box = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                    [3.0, 3.0, 1.0],
                    [1.0, 3.0, 1.0],
                    [1.0, 1.0, 2.0],
                    [3.0, 1.0, 2.0],
                    [3.0, 3.0, 2.0],
                    [1.0, 3.0, 2.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1x8x3
        dst_box = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        expected = torch.tensor(
            [[[[[57.0000, 59.0000], [71.0000, 73.0000]], [[106.0000, 108.0000], [120.0000, 122.0000]]]]],
            device=device,
            dtype=dtype,
        )

        patches = kornia.geometry.transform.crop_by_boxes3d(inp, src_box, dst_box, align_corners=True)
        self.assert_close(patches, expected, rtol=1e-4, atol=1e-4)

    def test_dynamo(self, device, dtype, torch_optimizer):
        # Define script
        op = kornia.geometry.transform.crop_by_boxes3d
        op_script = torch_optimizer(op)
        # Define input
        inp = torch.randn((1, 1, 7, 7, 7), device=device, dtype=dtype)
        src_box = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                    [3.0, 3.0, 1.0],
                    [1.0, 3.0, 1.0],
                    [1.0, 1.0, 2.0],
                    [3.0, 1.0, 2.0],
                    [3.0, 3.0, 2.0],
                    [1.0, 3.0, 2.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1x8x3
        dst_box = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        actual = op_script(inp, src_box, dst_box, align_corners=True)
        expected = op(inp, src_box, dst_box, align_corners=True)
        self.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        dtype = torch.float64
        inp = torch.randn((1, 1, 7, 7, 7), device=device, dtype=dtype)
        src_box = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                    [3.0, 3.0, 1.0],
                    [1.0, 3.0, 1.0],
                    [1.0, 1.0, 2.0],
                    [3.0, 1.0, 2.0],
                    [3.0, 3.0, 2.0],
                    [1.0, 3.0, 2.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1x8x3
        dst_box = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        self.gradcheck(
            kornia.geometry.transform.crop_by_boxes3d, (inp, src_box, dst_box), requires_grad=(True, False, False)
        )

    def test_convention_align_corners_default_is_false(self, device, dtype):
        # crop_by_boxes3d's align_corners default is False -- every other dedicated test in
        # this class passes align_corners=True explicitly, so the default itself was never
        # exercised until now. Also pins the docstring's "inclusive coordinates" consequence
        # with the same fixture: the box extent (1, 1, 1)..(2, 2, 2) reproduces the exact
        # integer-voxel slice only under align_corners=True -- the align_corners=False
        # default interpolates instead.
        vol = torch.arange(64.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4)
        src_box = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 2.0],
                    [2.0, 1.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [1.0, 2.0, 2.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        dst_box = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        expected_slice = vol[:, :, 1:3, 1:3, 1:3]

        out_default = kornia.geometry.transform.crop_by_boxes3d(vol, src_box, dst_box)
        out_true = kornia.geometry.transform.crop_by_boxes3d(vol, src_box, dst_box, align_corners=True)
        out_false = kornia.geometry.transform.crop_by_boxes3d(vol, src_box, dst_box, align_corners=False)

        self.assert_close(out_default, out_false, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_default, out_true, atol=1e-2, rtol=1e-2)
        self.assert_close(out_true, expected_slice, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_default, expected_slice, atol=1e-2, rtol=1e-2)

    def test_convention_crop_by_transform_mat3d_direct(self, device, dtype):
        # crop_by_transform_mat3d has no direct test anywhere in this file -- it is only ever
        # exercised indirectly through crop_by_boxes3d. Pin it directly: align_corners=True
        # default, and out_size in (depth, height, width) order.
        inp = torch.arange(0.0, 64.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4)
        transform = torch.tensor(
            [[[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        out_default = kornia.geometry.transform.crop_by_transform_mat3d(inp, transform, (2, 2, 2))
        out_true = kornia.geometry.transform.crop_by_transform_mat3d(inp, transform, (2, 2, 2), align_corners=True)
        expected = inp[:, :, 1:3, 1:3, 1:3]
        self.assert_close(out_default, out_true)
        self.assert_close(out_default, expected, rtol=1e-2, atol=1e-2)

    def test_convention_padding_mode_default_zeros_3d(self, device, dtype):
        # crop_by_transform_mat3d's padding_mode default is 'zeros': sampling past the volume
        # bounds fills with 0, not the edge value ('border' would).
        inp = torch.arange(1.0, 65.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4)
        transform = torch.tensor(
            [[[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        out_default = kornia.geometry.transform.crop_by_transform_mat3d(inp, transform, (3, 3, 3), align_corners=True)
        out_zeros = kornia.geometry.transform.crop_by_transform_mat3d(
            inp, transform, (3, 3, 3), align_corners=True, padding_mode="zeros"
        )
        self.assert_close(out_default, out_zeros)

        # Intentionally unguarded on MPS: 3D grid_sample supports padding_mode='border' there;
        # only 2D grid_sample has the MPS 'border' limitation guarded elsewhere (see
        # tests/geometry/transform/conftest.py's supports_2d_border_padding).
        out_border = kornia.geometry.transform.crop_by_transform_mat3d(
            inp, transform, (3, 3, 3), align_corners=True, padding_mode="border"
        )
        assert not torch.allclose(out_default, out_border, atol=1e-2, rtol=1e-2)


class TestCrop3DSizeValidation:
    """Tests that 3D crop functions properly reject invalid size arguments."""

    def test_crop_and_resize3d_rejects_wrong_length(self, device, dtype):
        inp = torch.rand(1, 1, 4, 4, 4, device=device, dtype=dtype)
        boxes = torch.rand(1, 8, 3, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="tuple/list of length 3"):
            kornia.geometry.transform.crop_and_resize3d(inp, boxes, (2, 2))

    def test_crop_and_resize3d_rejects_non_tuple(self, device, dtype):
        inp = torch.rand(1, 1, 4, 4, 4, device=device, dtype=dtype)
        boxes = torch.rand(1, 8, 3, device=device, dtype=dtype)
        with pytest.raises((ValueError, TypeError)):
            kornia.geometry.transform.crop_and_resize3d(inp, boxes, 2)

    def test_center_crop3d_rejects_wrong_length(self, device, dtype):
        inp = torch.rand(1, 1, 4, 4, 4, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="tuple/list of length 3"):
            kornia.geometry.transform.center_crop3d(inp, (2, 2))

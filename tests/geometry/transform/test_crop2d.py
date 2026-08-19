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

from testing.base import BaseTester


class TestCropAndResize(BaseTester):
    def test_align_corners_true(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        height, width = 2, 3

        expected = torch.tensor([[[[6.0000, 6.5000, 7.0000], [10.0000, 10.5000, 11.0000]]]], device=device, dtype=dtype)

        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        # default should use align_coners True
        patches = kornia.geometry.transform.crop_and_resize(inp, boxes, (height, width))
        self.assert_close(patches, expected, rtol=1e-4, atol=1e-4)

    def test_align_corners_false(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        height, width = 2, 3
        # The box corners are pixel coordinates, so they pin the sampling geometry regardless of
        # align_corners: the box spans src x, y in [1, 2] and the 2x3 output samples pixel centers
        # at x = 1, 1.5, 2 and y = 1, 2 -- the same values the align_corners=True case produces.
        expected = torch.tensor([[[[6.0, 6.5, 7.0], [10.0, 10.5, 11.0]]]], device=device, dtype=dtype)

        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        patches = kornia.geometry.transform.crop_and_resize(inp, boxes, (height, width), align_corners=False)
        self.assert_close(patches, expected, rtol=1e-4, atol=1e-4)

    def test_crop_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]],
                [[[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [[[[6.0, 7.0], [10.0, 11.0]]], [[[7.0, 15.0], [8.0, 16.0]]]], device=device, dtype=dtype
        )

        boxes = torch.tensor(
            [[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [3.0, 2.0], [3.0, 3.0], [1.0, 3.0]]],
            device=device,
            dtype=dtype,
        )  # 2x4x2

        patches = kornia.geometry.transform.crop_and_resize(inp, boxes, (2, 2))
        self.assert_close(patches, expected, rtol=1e-4, atol=1e-4)

    def test_crop_non_parallel_box(self, device, dtype):
        # regression test for https://github.com/kornia/kornia/issues/3560:
        # boxes with non-parallel opposite sides need a perspective warp, not affine
        inp = torch.arange(1.0, 37.0, device=device, dtype=dtype).view(1, 1, 6, 6)

        boxes = torch.tensor(
            [[[1.0, 1.0], [4.0, 0.0], [5.0, 4.0], [0.0, 3.0]]], device=device, dtype=dtype
        )  # 1x4x2 trapezoid

        # Snippet used to generate expected (requires numpy only): solve the DLT for the
        # homography mapping output corners to the box corners, then bilinear-sample:
        #   import numpy as np
        #   img = np.arange(1.0, 37.0).reshape(6, 6)
        #   src = np.array([[1.0, 1.0], [4.0, 0.0], [5.0, 4.0], [0.0, 3.0]])
        #   dst = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        #   A, b = [], []
        #   for (xs, ys), (xd, yd) in zip(src, dst):
        #       A += [[xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd], [0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd]]
        #       b += [xs, ys]
        #   H = np.append(np.linalg.solve(np.array(A), np.array(b)), 1.0).reshape(3, 3)
        #   expected = np.zeros((3, 3))
        #   for j in range(3):
        #       for i in range(3):
        #           p = H @ np.array([i, j, 1.0])
        #           x, y = p[0] / p[2], p[1] / p[2]
        #           x0, y0 = int(np.floor(x)), int(np.floor(y))
        #           ax, ay = x - x0, y - y0
        #           x1, y1 = min(x0 + 1, 5), min(y0 + 1, 5)
        #           expected[j, i] = (img[y0, x0] * (1 - ax) * (1 - ay) + img[y0, x1] * ax * (1 - ay)
        #                             + img[y1, x0] * (1 - ax) * ay + img[y1, x1] * ax * ay)
        expected = torch.tensor(
            [[[[8.0, 6.9, 5.0], [12.46875, 12.583333, 12.8125], [19.0, 22.055556, 30.0]]]],
            device=device,
            dtype=dtype,
        )

        patches = kornia.geometry.transform.crop_and_resize(inp, boxes, (3, 3))
        self.assert_close(patches, expected, rtol=1e-4, atol=1e-4)

    def test_crop_batch_broadcast(self, device, dtype):
        inp = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]],
                [[[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [[[[6.0, 7.0], [10.0, 11.0]]], [[[6.0, 10.0], [7.0, 11.0]]]], device=device, dtype=dtype
        )

        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        patches = kornia.geometry.transform.crop_and_resize(inp, boxes, (2, 2))
        self.assert_close(patches, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)

        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=torch.float64)

        self.gradcheck(
            kornia.geometry.transform.crop_and_resize, (img, boxes, (4, 2)), requires_grad=(True, False, False)
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        # Define script
        op = kornia.geometry.transform.crop_and_resize
        op_optimized = torch_optimizer(op)
        # Define input
        img = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )
        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        crop_height, crop_width = 4, 2
        actual = op_optimized(img, boxes, (crop_height, crop_width))
        expected = op(img, boxes, (crop_height, crop_width))
        self.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_convention_single_image_does_not_broadcast_over_boxes(self, device, dtype):
        # A single box broadcasts over a batch of N images (see test_crop_batch_broadcast),
        # but a single image does NOT broadcast over a batch of N boxes -- it raises
        # RuntimeError instead, so the broadcasting crop_and_resize supports is
        # one-directional, not general batch broadcasting.
        inp_one = torch.arange(0.0, 16.0, device=device, dtype=dtype).view(1, 1, 4, 4)
        two_boxes = torch.tensor(
            [[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        with pytest.raises(RuntimeError):
            kornia.geometry.transform.crop_and_resize(inp_one, two_boxes, (2, 2))

    def test_convention_padding_mode_default_zeros(self, device, dtype):
        # crop_and_resize's padding_mode default is 'zeros': a box sampling outside the image
        # bounds fills the out-of-bounds region with 0, not the edge value ('border' would).
        inp = torch.arange(1.0, 17.0, device=device, dtype=dtype).view(1, 1, 4, 4)
        boxes = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]], device=device, dtype=dtype)

        out_default = kornia.geometry.transform.crop_and_resize(inp, boxes, (3, 3))
        # Snippet used to generate expected (independent F.grid_sample call, not through
        # crop_and_resize's own box-to-grid machinery):
        # import torch.nn.functional as F
        # inp = torch.arange(1.0, 17.0).view(1, 1, 4, 4)
        # xs = ys = torch.tensor([-1.0, 0.0, 1.0])
        # gx, gy = 2 * xs / (4 - 1) - 1, 2 * ys / (4 - 1) - 1
        # grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1).unsqueeze(0)
        # expected = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        expected_zeros = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 5.0, 6.0]]]], device=device, dtype=dtype
        )
        self.assert_close(out_default, expected_zeros, rtol=1e-2, atol=1e-2)


class TestCenterCrop(BaseTester):
    def test_center_crop_h2_w4(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor([[[[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]], device=device, dtype=dtype)

        out_crop = kornia.geometry.transform.center_crop(inp, (2, 4))
        self.assert_close(out_crop, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(kornia.geometry.transform.CenterCrop2D((2, 4))(inp), expected, rtol=1e-4, atol=1e-4)

    def test_center_crop_h4_w2(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        height, width = 4, 2
        expected = torch.tensor([[[[2.0, 3.0], [6.0, 7.0], [10.0, 11.0], [14.0, 15.0]]]], device=device, dtype=dtype)

        out_crop = kornia.geometry.transform.center_crop(inp, (height, width))
        self.assert_close(out_crop, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(kornia.geometry.transform.CenterCrop2D((height, width))(inp), expected, rtol=1e-4, atol=1e-4)

    def test_center_crop_h4_w2_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]],
                [[[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [[[2.0, 3.0], [6.0, 7.0], [10.0, 11.0], [14.0, 15.0]]],
                [[[5.0, 9.0], [6.0, 10.0], [7.0, 11.0], [8.0, 12.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        out_crop = kornia.geometry.transform.center_crop(inp, (4, 2))
        self.assert_close(out_crop, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(kornia.geometry.transform.CenterCrop2D((4, 2))(inp), expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)

        self.gradcheck(kornia.geometry.transform.center_crop, (img, (4, 2)))
        self.gradcheck(kornia.geometry.transform.CenterCrop2D((4, 2)), (img))

    def test_dynamo(self, device, dtype, torch_optimizer):
        # Define script
        op = kornia.geometry.transform.center_crop
        op_script = torch_optimizer(op)
        # Define input
        img = torch.ones(1, 2, 5, 4, device=device, dtype=dtype)

        actual = op_script(img, (4, 2))
        expected = op(img, (4, 2))
        self.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_convention_align_corners_ignored_for_an_in_bounds_crop(self, device, dtype):
        # CenterCrop2D's align_corners constructor arg has NO effect under the default
        # cropping_mode='slice' (pure integer-index slicing, no resampling). Under
        # cropping_mode='resample' it also has no effect for a crop that samples strictly
        # in bounds: the crop box is specified in pixel coordinates, so both align_corners
        # conventions must resolve it to the same pixel centers (#3904). align_corners still
        # selects grid_sample's convention, which shows up only in out-of-bounds handling.
        inp = torch.arange(0.0, 16.0, device=device, dtype=dtype).view(1, 1, 4, 4)

        out_slice_true = kornia.geometry.transform.CenterCrop2D((2, 2), align_corners=True, cropping_mode="slice")(inp)
        out_slice_false = kornia.geometry.transform.CenterCrop2D((2, 2), align_corners=False, cropping_mode="slice")(
            inp
        )
        self.assert_close(out_slice_true, out_slice_false)

        out_resample_true = kornia.geometry.transform.CenterCrop2D(
            (2, 2), align_corners=True, cropping_mode="resample"
        )(inp)
        out_resample_false = kornia.geometry.transform.CenterCrop2D(
            (2, 2), align_corners=False, cropping_mode="resample"
        )(inp)
        self.assert_close(out_resample_true, out_resample_false, atol=1e-4, rtol=1e-4)
        # and both agree with the plain integer-index slice of the same region
        self.assert_close(out_resample_false, inp[:, :, 1:3, 1:3], atol=1e-4, rtol=1e-4)


class TestCropByBoxes(BaseTester):
    def test_crop_by_boxes_no_resizing(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        src = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        dst = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)  # 1x4x2

        expected = torch.tensor([[[[6.0, 7.0], [10.0, 11.0]]]], device=device, dtype=dtype)

        patches = kornia.geometry.transform.crop_by_boxes(inp, src, dst)
        self.assert_close(patches, expected)

    def test_crop_by_boxes_resizing(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        src = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        dst = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)  # 1x4x2

        expected = torch.tensor([[[[6.0, 6.5, 7.0], [10.0, 10.5, 11.0]]]], device=device, dtype=dtype)

        patches = kornia.geometry.transform.crop_by_boxes(inp, src, dst)
        self.assert_close(patches, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        dtype = torch.float64
        inp = torch.randn((1, 1, 3, 3), device=device, dtype=dtype)
        src = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]], device=device, dtype=dtype)
        dst = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)

        self.gradcheck(kornia.geometry.transform.crop_by_boxes, (inp, src, dst), requires_grad=(True, False, False))


class TestCropByTransform(BaseTester):
    def test_crop_by_transform_no_resizing(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        transform = torch.tensor(
            [[[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 1x3x3

        expected = torch.tensor([[[[6.0, 7.0], [10.0, 11.0]]]], device=device, dtype=dtype)

        patches = kornia.geometry.transform.crop_by_transform_mat(inp, transform, (2, 2))
        self.assert_close(patches, expected)

    def test_crop_by_boxes_resizing(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        transform = torch.tensor(
            [[[2.0, 0.0, -2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 1x3x3

        expected = torch.tensor([[[[6.0, 6.5, 7.0], [10.0, 10.5, 11.0]]]], device=device, dtype=dtype)

        patches = kornia.geometry.transform.crop_by_transform_mat(inp, transform, (2, 3))
        self.assert_close(patches, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        inp = torch.randn((1, 1, 3, 3), device=device, dtype=torch.float64)
        transform = torch.tensor(
            [[[2.0, 0.0, -2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=torch.float64
        )  # 1x3x3

        self.gradcheck(
            kornia.geometry.transform.crop_by_transform_mat,
            (inp, transform, (2, 2)),
            requires_grad=(True, False, False),
        )


class TestCropByIndices(BaseTester):
    def test_crop_by_indices_no_resizing(self, device, dtype):
        inp = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7, 8, 9]]]], device=device, dtype=dtype)  # 1x3x3

        # provide the indices to crop as 4 points
        indices = torch.tensor([[[0, 0], [1, 0], [1, 1], [0, 1]]], device=device, dtype=torch.int64)
        expected = torch.tensor([[[[1.0, 2.0], [4.0, 5.0]]]], device=device, dtype=dtype)

        self.assert_close(kornia.geometry.transform.crop_by_indices(inp, indices), expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        # Define script
        op = kornia.geometry.transform.crop_by_indices
        op_script = torch_optimizer(op)
        # Define input
        img = torch.ones(1, 2, 5, 4, device=device, dtype=dtype)

        actual = op_script(img, torch.tensor([[[0, 0], [1, 0], [1, 1], [0, 1]]]))
        expected = op(img, torch.tensor([[[0, 0], [1, 0], [1, 1], [0, 1]]]))
        self.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_crop_by_indices_variable_sizes_exception(self, device, dtype):
        img = torch.rand(2, 3, 20, 20, device=device, dtype=dtype)
        src_box = torch.tensor(
            [
                [[0, 0], [4, 0], [4, 4], [0, 4]],  # 5x5 box
                [[0, 0], [9, 0], [9, 9], [0, 9]],  # 10x10 box
            ],
            dtype=dtype,
            device=device,
        )

        with pytest.raises(ValueError, match="All boxes in the batch must have the same height and width"):
            kornia.geometry.transform.crop_by_indices(img, src_box, size=None)

    def test_convention_shape_compensation_pad_vs_resize(self, device, dtype):
        # shape_compensation is pinned per the crop_by_indices Convention block: when
        # src_box is identical across the batch it is ignored (exact integer slice if the
        # slice already matches `size`, resized otherwise); it only takes effect for a
        # non-uniform batch, where 'pad' trims via F.pad's negative padding (keeps the
        # top-left corner, no interpolation) while 'resize' downsamples via interpolation.
        inp = torch.arange(0.0, 32.0, device=device, dtype=dtype).view(1, 1, 4, 8).repeat(2, 1, 1, 1)

        # --- identical src_box across the batch: shape_compensation is ignored ---
        box_3x3 = torch.tensor([[[0, 0], [2, 0], [2, 2], [0, 2]]], device=device, dtype=torch.int64).expand(2, -1, -1)
        box_2x2 = torch.tensor([[[0, 0], [1, 0], [1, 1], [0, 1]]], device=device, dtype=torch.int64).expand(2, -1, -1)
        size = (2, 2)

        # slice (2x2) already matches `size`: exact integer slice under both settings.
        out_resize_match = kornia.geometry.transform.crop_by_indices(
            inp, box_2x2, size=size, shape_compensation="resize"
        )
        out_pad_match = kornia.geometry.transform.crop_by_indices(inp, box_2x2, size=size, shape_compensation="pad")
        expected_slice = inp[..., 0:2, 0:2]
        self.assert_close(out_resize_match, expected_slice, atol=0.0, rtol=0.0)
        self.assert_close(out_pad_match, expected_slice, atol=0.0, rtol=0.0)

        # slice (3x3) differs from `size` (2x2): resized under both settings.
        out_resize_diff = kornia.geometry.transform.crop_by_indices(
            inp, box_3x3, size=size, shape_compensation="resize"
        )
        out_pad_diff = kornia.geometry.transform.crop_by_indices(inp, box_3x3, size=size, shape_compensation="pad")
        self.assert_close(out_resize_diff, out_pad_diff, atol=0.0, rtol=0.0)
        assert out_resize_diff.shape[-2:] == size

        # --- non-uniform batch (box 0 != box 1): shape_compensation genuinely takes effect ---
        src_box = torch.tensor(
            [
                [[0, 0], [2, 0], [2, 2], [0, 2]],  # 3x3 box, LARGER than size=(2, 2)
                [[0, 0], [1, 0], [1, 1], [0, 1]],  # 2x2 box, matches size=(2, 2)
            ],
            device=device,
            dtype=torch.int64,
        )

        out_resize = kornia.geometry.transform.crop_by_indices(inp, src_box, size=size, shape_compensation="resize")
        out_pad = kornia.geometry.transform.crop_by_indices(inp, src_box, size=size, shape_compensation="pad")

        # Snippet used to generate expected (requires torch + kornia.geometry.transform.resize):
        # inp = torch.arange(0.0, 32.0).view(1, 1, 4, 8).repeat(2, 1, 1, 1)
        # slice3x3 = inp[0:1, :, 0:3, 0:3]
        # expected_resize_0 = kornia.geometry.transform.resize(
        #     slice3x3, (2, 2), interpolation="bilinear", align_corners=None, side="short", antialias=False
        # )[0, 0]
        # expected_pad_0 = torch.nn.functional.pad(slice3x3, (0, -1, 0, -1))[0, 0]
        expected_resize_0 = torch.tensor([[2.25, 3.75], [14.25, 15.75]], device=device, dtype=dtype)
        expected_pad_0 = torch.tensor([[0.0, 1.0], [8.0, 9.0]], device=device, dtype=dtype)
        self.assert_close(out_resize[0, 0], expected_resize_0, rtol=1e-2, atol=1e-2)
        self.assert_close(out_pad[0, 0], expected_pad_0, rtol=1e-2, atol=1e-2)
        # sample 1's own cropped slice already matches `size`, so its per-item branch takes
        # the plain-copy path, and 'pad'/'resize' agree there too.
        self.assert_close(out_resize[1], out_pad[1])


class TestCropSizeValidation:
    """Tests that crop functions properly reject invalid size arguments."""

    def test_crop_and_resize_rejects_wrong_length(self, device, dtype):
        inp = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        boxes = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)
        with pytest.raises(ValueError, match="tuple/list of length 2"):
            kornia.geometry.transform.crop_and_resize(inp, boxes, (2, 2, 2))

    def test_crop_and_resize_rejects_non_tuple(self, device, dtype):
        inp = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        boxes = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)
        # Passing an int instead of a tuple can raise either ValueError (from an
        # explicit validation check) or TypeError (from calling len() on an int),
        # depending on which code path runs first inside crop_and_resize.
        with pytest.raises((ValueError, TypeError)):
            kornia.geometry.transform.crop_and_resize(inp, boxes, 2)

    def test_center_crop_rejects_wrong_length(self, device, dtype):
        inp = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="tuple/list of length 2"):
            kornia.geometry.transform.center_crop(inp, (2, 2, 2))

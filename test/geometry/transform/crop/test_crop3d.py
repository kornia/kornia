from typing import Tuple

import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestBoundingBoxInferring3D:
    def test_bounding_boxes_dim_inferring(self, device, dtype):
        boxes = torch.tensor([
            [[0, 1, 2],
             [10, 1, 2],
             [10, 21, 2],
             [0, 21, 2],
             [0, 1, 32],
             [10, 1, 32],
             [10, 21, 32],
             [0, 21, 32]],
            [[3, 4, 5],
             [43, 4, 5],
             [43, 54, 5],
             [3, 54, 5],
             [3, 4, 65],
             [43, 4, 65],
             [43, 54, 65],
             [3, 54, 65]
             ]], device=device, dtype=dtype)  # 2x8x3
        d, h, w = kornia.geometry.transform.crop.infer_box_shape3d(boxes)

        assert_allclose(d, torch.tensor([31., 61.], device=device, dtype=dtype))
        assert_allclose(h, torch.tensor([21., 51.], device=device, dtype=dtype))
        assert_allclose(w, torch.tensor([11., 41.], device=device, dtype=dtype))

    def test_gradcheck(self, device, dtype):
        boxes = torch.tensor([[
            [0, 1, 2],
            [10, 1, 2],
            [10, 21, 2],
            [0, 21, 2],
            [0, 1, 32],
            [10, 1, 32],
            [10, 21, 32],
            [0, 21, 32]
        ]], device=device, dtype=dtype)
        boxes = utils.tensor_to_gradcheck_var(boxes)
        assert gradcheck(kornia.kornia.geometry.transform.crop.infer_box_shape3d,
                         (boxes,), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.infer_box_shape3d
        op_script = torch.jit.script(op)

        boxes = torch.tensor([[
            [0, 0, 1],
            [3, 0, 1],
            [3, 2, 1],
            [0, 2, 1],
            [0, 0, 3],
            [3, 0, 3],
            [3, 2, 3],
            [0, 2, 3],
        ]], device=device, dtype=dtype)  # 1x8x3

        actual = op_script(boxes)
        expected = op(boxes)
        assert_allclose(actual, expected)


class TestCropAndResize3D:
    def test_crop(self, device, dtype):
        inp = torch.arange(0., 64., device=device, dtype=dtype).view(1, 1, 4, 4, 4)

        depth, height, width = 2, 2, 2
        expected = torch.tensor([[[
            [[25.1667, 27.1667],
             [30.5000, 32.5000]],
            [[46.5000, 48.5000],
             [51.8333, 53.8333]]]]], device=device, dtype=dtype)

        boxes = torch.tensor([[
            [0, 0, 1],
            [3, 0, 1],
            [3, 2, 1],
            [0, 2, 1],
            [0, 0, 3],
            [3, 0, 3],
            [3, 2, 3],
            [0, 2, 3],
        ]], device=device, dtype=dtype)  # 1x8x3

        patches = kornia.crop_and_resize3d(inp, boxes, (depth, height, width))
        assert_allclose(patches, expected)

    def test_crop_batch(self, device, dtype):
        inp = torch.cat([
            torch.arange(0., 64., device=device, dtype=dtype).view(1, 1, 4, 4, 4),
            torch.arange(0., 128., step=2, device=device, dtype=dtype).view(1, 1, 4, 4, 4)
        ], dim=0)

        depth, height, width = 2, 2, 2
        expected = torch.tensor([
            [[[[16.0000, 19.0000],
               [24.0000, 27.0000]],
              [[48.0000, 51.0000],
               [56.0000, 59.0000]]]],
            [[[[0.0000, 6.0000],
               [16.0000, 22.0000]],
              [[64.0000, 70.0000],
               [80.0000, 86.0000]]]]
        ], device=device, dtype=dtype)

        boxes = torch.tensor([[
            [0, 0, 1],
            [3, 0, 1],
            [3, 2, 1],
            [0, 2, 1],
            [0, 0, 3],
            [3, 0, 3],
            [3, 2, 3],
            [0, 2, 3],
        ], [
            [0, 0, 0],
            [3, 0, 0],
            [3, 2, 0],
            [0, 2, 0],
            [0, 0, 2],
            [3, 0, 2],
            [3, 2, 2],
            [0, 2, 2],
        ]], device=device, dtype=dtype)  # 2x8x3

        patches = kornia.crop_and_resize3d(inp, boxes, (depth, height, width), align_corners=True)
        assert_allclose(patches, expected)

    def test_gradcheck(self, device, dtype):
        img = torch.arange(0., 64., device=device, dtype=dtype).view(1, 1, 4, 4, 4)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        boxes = torch.tensor([[
            [0, 0, 1],
            [3, 0, 1],
            [3, 2, 1],
            [0, 2, 1],
            [0, 0, 3],
            [3, 0, 3],
            [3, 2, 3],
            [0, 2, 3],
        ]], device=device, dtype=dtype)  # 1x8x3
        boxes = utils.tensor_to_gradcheck_var(boxes, requires_grad=False)  # to var

        assert gradcheck(kornia.crop_and_resize3d, (img, boxes, (4, 3, 2),), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.crop_and_resize3d
        op_script = torch.jit.script(op)

        img = torch.arange(0., 64., device=device, dtype=dtype).view(1, 1, 4, 4, 4)

        boxes = torch.tensor([[
            [0, 0, 1],
            [3, 0, 1],
            [3, 2, 1],
            [0, 2, 1],
            [0, 0, 3],
            [3, 0, 3],
            [3, 2, 3],
            [0, 2, 3],
        ]], device=device, dtype=dtype)  # 1x8x3

        actual = op_script(img, boxes, (4, 3, 2))
        expected = op(img, boxes, (4, 3, 2))
        assert_allclose(actual, expected)


class TestCenterCrop3D:
    @pytest.mark.parametrize("crop_size", [(3, 5, 7), (5, 3, 7), (7, 3, 5)])
    def test_center_crop_357(self, crop_size, device, dtype):
        inp = torch.arange(0., 343., device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        expected = inp[
            :, :,
            (inp.size(2) // 2 - crop_size[0] // 2):(inp.size(2) // 2 + crop_size[0] // 2 + 1),
            (inp.size(3) // 2 - crop_size[1] // 2):(inp.size(3) // 2 + crop_size[1] // 2 + 1),
            (inp.size(4) // 2 - crop_size[2] // 2):(inp.size(4) // 2 + crop_size[2] // 2 + 1)
        ]
        out_crop = kornia.center_crop3d(inp, crop_size, align_corners=True)
        assert_allclose(out_crop, expected)

    @pytest.mark.parametrize("crop_size", [(3, 5, 7), (5, 3, 7), (7, 3, 5)])
    def test_center_crop_357_batch(self, crop_size, device, dtype):
        inp = torch.cat([
            torch.arange(0., 343., device=device, dtype=dtype).view(1, 1, 7, 7, 7),
            torch.arange(343., 686., device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        ])
        expected = inp[
            :, :,
            (inp.size(2) // 2 - crop_size[0] // 2):(inp.size(2) // 2 + crop_size[0] // 2 + 1),
            (inp.size(3) // 2 - crop_size[1] // 2):(inp.size(3) // 2 + crop_size[1] // 2 + 1),
            (inp.size(4) // 2 - crop_size[2] // 2):(inp.size(4) // 2 + crop_size[2] // 2 + 1)
        ]
        out_crop = kornia.center_crop3d(inp, crop_size, align_corners=True)
        assert_allclose(out_crop, expected)

    def test_gradcheck(self, device, dtype):
        img = torch.arange(0., 343., device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        assert gradcheck(kornia.center_crop3d, (img, (3, 5, 7),), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.center_crop3d
        op_script = torch.jit.script(op)
        img = torch.ones(4, 3, 5, 6, 7, device=device, dtype=dtype)

        actual = op_script(img, (4, 3, 2))
        expected = kornia.center_crop3d(img, (4, 3, 2))
        assert_allclose(actual, expected)


class TestCropByBoxes3D:
    def test_crop_by_boxes_no_resizing(self, device, dtype):
        inp = torch.arange(0., 343., device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        src_box = torch.tensor([[
            [1., 1., 1.],
            [3., 1., 1.],
            [3., 3., 1.],
            [1., 3., 1.],
            [1., 1., 2.],
            [3., 1., 2.],
            [3., 3., 2.],
            [1., 3., 2.],
        ]], device=device, dtype=dtype)  # 1x8x3
        dst_box = torch.tensor([[
            [0., 0., 0.],
            [2., 0., 0.],
            [2., 2., 0.],
            [0., 2., 0.],
            [0., 0., 1.],
            [2., 0., 1.],
            [2., 2., 1.],
            [0., 2., 1.],
        ]], device=device, dtype=dtype)  # 1x8x3

        expected = inp[:, :, 1:3, 1:4, 1:4]

        patches = kornia.geometry.transform.crop.crop_by_boxes3d(inp, src_box, dst_box, align_corners=True)
        assert_allclose(patches, expected)

    def test_crop_by_boxes_resizing(self, device, dtype):
        inp = torch.arange(0., 343., device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        src_box = torch.tensor([[
            [1., 1., 1.],
            [3., 1., 1.],
            [3., 3., 1.],
            [1., 3., 1.],
            [1., 1., 2.],
            [3., 1., 2.],
            [3., 3., 2.],
            [1., 3., 2.],
        ]], device=device, dtype=dtype)  # 1x8x3
        dst_box = torch.tensor([[
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            [0., 1., 1.],
        ]], device=device, dtype=dtype)  # 1x8x3

        expected = torch.tensor([[[
            [[57.0000, 59.0000],
             [71.0000, 73.0000]],
            [[106.0000, 108.0000],
             [120.0000, 122.0000]]
        ]]], device=device, dtype=dtype)

        patches = kornia.geometry.transform.crop.crop_by_boxes3d(inp, src_box, dst_box, align_corners=True)
        assert_allclose(patches, expected)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.geometry.transform.crop.crop_by_boxes3d
        op_script = torch.jit.script(op)
        # Define input
        inp = torch.randn((1, 1, 7, 7, 7), device=device, dtype=dtype)
        src_box = torch.tensor([[
            [1., 1., 1.],
            [3., 1., 1.],
            [3., 3., 1.],
            [1., 3., 1.],
            [1., 1., 2.],
            [3., 1., 2.],
            [3., 3., 2.],
            [1., 3., 2.],
        ]], device=device, dtype=dtype)  # 1x8x3
        dst_box = torch.tensor([[
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            [0., 1., 1.],
        ]], device=device, dtype=dtype)  # 1x8x3

        actual = op_script(inp, src_box, dst_box, align_corners=True)
        expected = op(inp, src_box, dst_box, align_corners=True)
        assert_allclose(actual, expected)

    def test_gradcheck(self, device, dtype):
        inp = torch.randn((1, 1, 7, 7, 7), device=device, dtype=dtype)
        src_box = torch.tensor([[
            [1., 1., 1.],
            [3., 1., 1.],
            [3., 3., 1.],
            [1., 3., 1.],
            [1., 1., 2.],
            [3., 1., 2.],
            [3., 3., 2.],
            [1., 3., 2.],
        ]], device=device, dtype=dtype)  # 1x8x3
        dst_box = torch.tensor([[
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            [0., 1., 1.],
        ]], device=device, dtype=dtype)  # 1x8x3

        inp = utils.tensor_to_gradcheck_var(inp, requires_grad=True)  # to var

        assert gradcheck(kornia.geometry.transform.crop.crop_by_boxes3d,
                         (inp, src_box, dst_box,), raise_exception=True)

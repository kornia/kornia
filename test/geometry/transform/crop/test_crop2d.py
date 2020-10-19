from typing import Tuple

import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestBoundingBoxInferring:
    def test_bounding_boxes_dim_inferring(self, device, dtype):
        boxes = torch.tensor([[
            [1., 1.],
            [3., 1.],
            [3., 2.],
            [1., 2.],
        ]], device=device, dtype=dtype)

        h, w = kornia.geometry.transform.crop.infer_box_shape(boxes)
        assert (h, w) == (2, 3)

    def test_bounding_boxes_dim_inferring_batch(self, device, dtype):
        boxes = torch.tensor([[
            [1., 1.],
            [3., 1.],
            [3., 2.],
            [1., 2.],
        ], [
            [2., 2.],
            [4., 2.],
            [4., 3.],
            [2., 3.],
        ]], device=device, dtype=dtype)
        h, w = kornia.geometry.transform.crop.infer_box_shape(boxes)
        assert (h.unique().item(), w.unique().item()) == (2, 3)

    def test_gradcheck(self, device, dtype):
        boxes = torch.tensor([[
            [1., 1.],
            [3., 1.],
            [3., 2.],
            [1., 2.],
        ]], device=device, dtype=dtype)
        boxes = utils.tensor_to_gradcheck_var(boxes)
        assert gradcheck(kornia.kornia.geometry.transform.crop.infer_box_shape,
                         (boxes,), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.geometry.transform.crop.infer_box_shape
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[
            [1., 1.],
            [3., 1.],
            [3., 2.],
            [1., 2.],
        ]], device=device, dtype=dtype)

        actual = op_script(boxes)
        expected = op(boxes)
        assert_allclose(actual, expected)


class TestCropAndResize:
    def test_crop(self, device, dtype):
        inp = torch.tensor([[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]]], device=device, dtype=dtype)

        height, width = 2, 3
        expected = torch.tensor(
            [[[[6.7222, 7.1667, 7.6111],
               [9.3889, 9.8333, 10.2778]]]], device=device, dtype=dtype)

        boxes = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]], device=device, dtype=dtype)  # 1x4x2

        patches = kornia.crop_and_resize(inp, boxes, (height, width))
        assert_allclose(patches, expected)

    def test_crop_batch(self, device, dtype):
        inp = torch.tensor([[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]], [[
            [1., 5., 9., 13.],
            [2., 6., 10., 14.],
            [3., 7., 11., 15.],
            [4., 8., 12., 16.],
        ]]], device=device, dtype=dtype)

        expected = torch.tensor([[[
            [6., 7.],
            [10., 11.],
        ]], [[
            [7., 15.],
            [8., 16.],
        ]]], device=device, dtype=dtype)

        boxes = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ], [
            [1., 2.],
            [3., 2.],
            [3., 3.],
            [1., 3.],
        ]], device=device, dtype=dtype)  # 2x4x2

        patches = kornia.crop_and_resize(inp, boxes, (2, 2), align_corners=True)
        assert_allclose(patches, expected)

    def test_crop_batch_broadcast(self, device, dtype):
        inp = torch.tensor([[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]], [[
            [1., 5., 9., 13.],
            [2., 6., 10., 14.],
            [3., 7., 11., 15.],
            [4., 8., 12., 16.],
        ]]], device=device, dtype=dtype)

        expected = torch.tensor([[[
            [6., 7.],
            [10., 11.],
        ]], [[
            [6., 10.],
            [7., 11.],
        ]]], device=device, dtype=dtype)

        boxes = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]], device=device, dtype=dtype)  # 1x4x2

        patches = kornia.crop_and_resize(inp, boxes, (2, 2), align_corners=True)
        assert_allclose(patches, expected)

    def test_gradcheck(self, device, dtype):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        boxes = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]], device=device, dtype=dtype)  # 1x4x2
        boxes = utils.tensor_to_gradcheck_var(boxes, requires_grad=False)  # to var

        assert gradcheck(kornia.crop_and_resize,
                         (img, boxes, (4, 2),),
                         raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.crop_and_resize
        op_script = torch.jit.script(op)
        # Define input
        img = torch.tensor([[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]]], device=device, dtype=dtype)
        boxes = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]], device=device, dtype=dtype)  # 1x4x2

        crop_height, crop_width = 4, 2
        actual = op_script(img, boxes, (crop_height, crop_width))
        expected = op(img, boxes, (crop_height, crop_width))
        assert_allclose(actual, expected)


class TestCenterCrop:
    def test_center_crop_h2_w4(self, device, dtype):
        inp = torch.tensor([[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]]], device=device, dtype=dtype)

        expected = torch.tensor([[[
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
        ]]], device=device, dtype=dtype)

        out_crop = kornia.center_crop(inp, (2, 4))
        assert_allclose(out_crop, expected)

    def test_center_crop_h4_w2(self, device, dtype):
        inp = torch.tensor([[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]]], device=device, dtype=dtype)

        height, width = 4, 2
        expected = torch.tensor([[[
            [2., 3.],
            [6., 7.],
            [10., 11.],
            [14., 15.],
        ]]], device=device, dtype=dtype)

        out_crop = kornia.center_crop(inp, (4, 2))
        assert_allclose(out_crop, expected)

    def test_center_crop_h4_w2_batch(self, device, dtype):
        inp = torch.tensor([
            [[[1., 2., 3., 4.],
              [5., 6., 7., 8.],
              [9., 10., 11., 12.],
              [13., 14., 15., 16.]]],
            [[[1., 5., 9., 13.],
              [2., 6., 10., 14.],
              [3., 7., 11., 15.],
              [4., 8., 12., 16.]]]
        ], device=device, dtype=dtype)

        expected = torch.tensor([[[
            [2., 3.],
            [6., 7.],
            [10., 11.],
            [14., 15.],
        ]], [[
            [5., 9.],
            [6., 10.],
            [7., 11.],
            [8., 12.],
        ]]], device=device, dtype=dtype)

        out_crop = kornia.center_crop(inp, (4, 2))
        assert_allclose(out_crop, expected)

    def test_gradcheck(self, device, dtype):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        assert gradcheck(kornia.center_crop, (img, (4, 2),), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.center_crop
        op_script = torch.jit.script(op)
        # Define input
        img = torch.ones(1, 2, 5, 4, device=device, dtype=dtype)

        actual = op_script(img, (4, 2))
        expected = op(img, (4, 2))
        assert_allclose(actual, expected)

    def test_jit_trace(self, device, dtype):
        # Define script
        op = kornia.center_crop
        op_script = torch.jit.script(op)
        # Define input
        img = torch.ones(2, 1, 6, 3, device=device, dtype=dtype)
        op_trace = torch.jit.trace(op_script, (img, (torch.tensor(2), torch.tensor(3))))
        img = torch.ones(2, 1, 6, 3, device=device, dtype=dtype)
        # Run
        actual = op_trace(img, (torch.tensor(2), torch.tensor(3)))
        expected = op(img, (2, 3))
        assert_allclose(actual, expected)


class TestCropByBoxes:
    def test_crop_by_boxes_no_resizing(self, device, dtype):
        inp = torch.tensor([[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]]], device=device, dtype=dtype)

        src = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]], device=device, dtype=dtype)  # 1x4x2

        dst = torch.tensor([[
            [0., 0.],
            [1., 0.],
            [1., 1.],
            [0., 1.],
        ]], device=device, dtype=dtype)  # 1x4x2

        expected = torch.tensor([[[
            [6., 7.],
            [10., 11.],
        ]]], device=device, dtype=dtype)

        patches = kornia.geometry.transform.crop.crop_by_boxes(inp, src, dst, align_corners=True)
        assert_allclose(patches, expected)

    def test_crop_by_boxes_resizing(self, device, dtype):
        inp = torch.tensor([[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]]], device=device, dtype=dtype)

        src = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]], device=device, dtype=dtype)  # 1x4x2

        dst = torch.tensor([[
            [0., 0.],
            [2., 0.],
            [2., 1.],
            [0., 1.],
        ]], device=device, dtype=dtype)  # 1x4x2

        expected = torch.tensor([[[
            [6., 6.5, 7.],
            [10., 10.5, 11.],
        ]]], device=device, dtype=dtype)

        patches = kornia.geometry.transform.crop.crop_by_boxes(inp, src, dst, align_corners=True)
        assert_allclose(patches, expected)

    def test_gradcheck(self, device, dtype):
        inp = torch.randn((1, 1, 3, 3), device=device, dtype=dtype)
        src = torch.tensor([[
            [1., 0.],
            [2., 0.],
            [2., 1.],
            [1., 1.]]], device=device, dtype=dtype)
        dst = torch.tensor([[
            [0., 0.],
            [1., 0.],
            [1., 1.],
            [0., 1.]]], device=device, dtype=dtype)

        inp = utils.tensor_to_gradcheck_var(inp, requires_grad=True)  # to var

        assert gradcheck(kornia.geometry.transform.crop.crop_by_boxes,
                         (inp, src, dst,),
                         raise_exception=True)

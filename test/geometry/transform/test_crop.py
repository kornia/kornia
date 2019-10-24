from typing import Tuple

import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestCropAndResize:
    def test_crop(self):
        inp = torch.tensor([[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]])

        height, width = 2, 3
        expected = torch.tensor([[
            [6., 6.5, 7.],
            [10., 10.5, 11.],
        ]])

        boxes = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]])  # 1x4x2

        patches = kornia.crop_and_resize(inp, boxes, (height, width))
        assert_allclose(patches, expected)

    def test_crop_batch(self):
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
        ]]])

        height, width = 2, 2
        expected = torch.tensor([[[
            [6., 7.],
            [10., 11.],
        ]], [[
            [7., 15.],
            [8., 16.],
        ]]])

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
        ]])  # 2x4x2

        patches = kornia.crop_and_resize(inp, boxes, (height, width))
        assert_allclose(patches, expected)

    def test_crop_batch_broadcast(self):
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
        ]]])

        height, width = 2, 2
        expected = torch.tensor([[[
            [6., 7.],
            [10., 11.],
        ]], [[
            [6., 10.],
            [7., 11.],
        ]]])

        boxes = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]])  # 1x4x2

        patches = kornia.crop_and_resize(inp, boxes, (height, width))
        assert_allclose(patches, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        boxes = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]])  # 1x4x2
        boxes = utils.tensor_to_gradcheck_var(
            boxes, requires_grad=False)  # to var

        crop_height, crop_width = 4, 2
        assert gradcheck(kornia.crop_and_resize,
                         (img, boxes, (crop_height, crop_width),),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input: torch.Tensor,
                      boxes: torch.Tensor,
                      size: Tuple[int, int]) -> torch.Tensor:
            return kornia.crop_and_resize(input, boxes, size)
        img = torch.tensor([[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]])
        boxes = torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]])  # 1x4x2

        crop_height, crop_width = 4, 2
        actual = op_script(img, boxes, (crop_height, crop_width))
        expected = kornia.crop_and_resize(img, boxes, (crop_height, crop_width))
        assert_allclose(actual, expected)


class TestCenterCrop:
    def test_center_crop_h2_w4(self):
        inp = torch.tensor([[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]])

        height, width = 2, 4
        expected = torch.tensor([[
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
        ]])

        out_crop = kornia.center_crop(inp, (height, width))
        assert_allclose(out_crop, expected)

    def test_center_crop_h4_w2(self):
        inp = torch.tensor([[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]])

        height, width = 4, 2
        expected = torch.tensor([[
            [2., 3.],
            [6., 7.],
            [10., 11.],
            [14., 15.],
        ]])

        out_crop = kornia.center_crop(inp, (height, width))
        assert_allclose(out_crop, expected)

    def test_center_crop_h4_w2_batch(self):
        inp = torch.tensor([[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ], [
            [1., 5., 9., 13.],
            [2., 6., 10., 14.],
            [3., 7., 11., 15.],
            [4., 8., 12., 16.],
        ]])

        height, width = 4, 2
        expected = torch.tensor([[
            [2., 3.],
            [6., 7.],
            [10., 11.],
            [14., 15.],
        ], [
            [5., 9.],
            [6., 10.],
            [7., 11.],
            [8., 12.],
        ]])

        out_crop = kornia.center_crop(inp, (height, width))
        assert_allclose(out_crop, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        crop_height, crop_width = 4, 2
        assert gradcheck(kornia.center_crop, (img, (crop_height, crop_width),),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input: torch.Tensor,
                      size: Tuple[int, int]) -> torch.Tensor:
            return kornia.center_crop(input, size)
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.ones(batch_size, channels, height, width)

        crop_height, crop_width = 4, 2
        actual = op_script(img, (crop_height, crop_width))
        expected = kornia.center_crop(img, (crop_height, crop_width))
        assert_allclose(actual, expected)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit_trace(self):
        @torch.jit.script
        def op_script(input: torch.Tensor,
                      size: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            return kornia.center_crop(input, size)
        # 1. Trace op
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.ones(batch_size, channels, height, width)

        crop_height, crop_width = 4, 2
        op_trace = torch.jit.trace(
            op_script,
            (img, (torch.tensor(crop_height), torch.tensor(crop_width))))

        # 2. Generate new input
        batch_size, channels, height, width = 2, 1, 6, 3
        img = torch.ones(batch_size, channels, height, width)

        # 3. Evaluate
        crop_height, crop_width = 2, 3
        actual = op_trace(
            img, (torch.tensor(crop_height), torch.tensor(crop_width)))
        expected = kornia.center_crop(img, (crop_height, crop_width))
        assert_allclose(actual, expected)

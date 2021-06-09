import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia
import kornia.testing as utils  # test utils


class TestBoundingBoxInferring:
    def test_bounding_boxes_dim_inferring(self, device, dtype):
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)

        h, w = kornia.geometry.transform.crop.infer_box_shape(boxes)
        assert (h, w) == (2, 3)

    def test_bounding_boxes_dim_inferring_batch(self, device, dtype):
        boxes = torch.tensor(
            [[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]], [[2.0, 2.0], [4.0, 2.0], [4.0, 3.0], [2.0, 3.0]]],
            device=device,
            dtype=dtype,
        )
        h, w = kornia.geometry.transform.crop.infer_box_shape(boxes)
        assert (h.unique().item(), w.unique().item()) == (2, 3)

    def test_gradcheck(self, device, dtype):
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        boxes = utils.tensor_to_gradcheck_var(boxes)
        assert gradcheck(kornia.kornia.geometry.transform.crop.infer_box_shape, (boxes,), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.geometry.transform.crop.infer_box_shape
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)

        actual = op_script(boxes)
        expected = op(boxes)
        assert_allclose(actual, expected)


class TestCropAndResize:
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
        patches = kornia.crop_and_resize(inp, boxes, (height, width))
        assert_allclose(patches, expected, rtol=1e-4, atol=1e-4)

    def test_align_corners_false(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        height, width = 2, 3
        expected = torch.tensor([[[[6.7222, 7.1667, 7.6111], [9.3889, 9.8333, 10.2778]]]], device=device, dtype=dtype)

        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        patches = kornia.crop_and_resize(inp, boxes, (height, width), align_corners=False)
        assert_allclose(patches, expected, rtol=1e-4, atol=1e-4)

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

        patches = kornia.crop_and_resize(inp, boxes, (2, 2))
        assert_allclose(patches, expected, rtol=1e-4, atol=1e-4)

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

        patches = kornia.crop_and_resize(inp, boxes, (2, 2))
        assert_allclose(patches, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2
        boxes = utils.tensor_to_gradcheck_var(boxes, requires_grad=False)  # to var

        assert gradcheck(kornia.crop_and_resize, (img, boxes, (4, 2)), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.crop_and_resize
        op_script = torch.jit.script(op)
        # Define input
        img = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )
        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        crop_height, crop_width = 4, 2
        actual = op_script(img, boxes, (crop_height, crop_width))
        expected = op(img, boxes, (crop_height, crop_width))
        assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


class TestCenterCrop:
    def test_center_crop_h2_w4(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor([[[[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]], device=device, dtype=dtype)

        out_crop = kornia.center_crop(inp, (2, 4))
        assert_allclose(out_crop, expected, rtol=1e-4, atol=1e-4)

    def test_center_crop_h4_w2(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        height, width = 4, 2
        expected = torch.tensor([[[[2.0, 3.0], [6.0, 7.0], [10.0, 11.0], [14.0, 15.0]]]], device=device, dtype=dtype)

        out_crop = kornia.center_crop(inp, (height, width))
        assert_allclose(out_crop, expected, rtol=1e-4, atol=1e-4)

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

        out_crop = kornia.center_crop(inp, (4, 2))
        assert_allclose(out_crop, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        assert gradcheck(kornia.center_crop, (img, (4, 2)), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = kornia.center_crop
        op_script = torch.jit.script(op)
        # Define input
        img = torch.ones(1, 2, 5, 4, device=device, dtype=dtype)

        actual = op_script(img, (4, 2))
        expected = op(img, (4, 2))
        assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

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
        assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


class TestCropByBoxes:
    def test_crop_by_boxes_no_resizing(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        src = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        dst = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)  # 1x4x2

        expected = torch.tensor([[[[6.0, 7.0], [10.0, 11.0]]]], device=device, dtype=dtype)

        patches = kornia.geometry.transform.crop.crop_by_boxes(inp, src, dst)
        assert_allclose(patches, expected)

    def test_crop_by_boxes_resizing(self, device, dtype):
        inp = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            device=device,
            dtype=dtype,
        )

        src = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)  # 1x4x2

        dst = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)  # 1x4x2

        expected = torch.tensor([[[[6.0, 6.5, 7.0], [10.0, 10.5, 11.0]]]], device=device, dtype=dtype)

        patches = kornia.geometry.transform.crop.crop_by_boxes(inp, src, dst)
        assert_allclose(patches, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        inp = torch.randn((1, 1, 3, 3), device=device, dtype=dtype)
        src = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]], device=device, dtype=dtype)
        dst = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)

        inp = utils.tensor_to_gradcheck_var(inp, requires_grad=True)  # to var

        assert gradcheck(kornia.geometry.transform.crop.crop_by_boxes, (inp, src, dst), raise_exception=True)


class TestCropByTransform:
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

        patches = kornia.geometry.transform.crop.crop_by_transform_mat(inp, transform, (2, 2))
        assert_allclose(patches, expected)

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

        patches = kornia.geometry.transform.crop.crop_by_transform_mat(inp, transform, (2, 3))
        assert_allclose(patches, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        inp = torch.randn((1, 1, 3, 3), device=device, dtype=dtype)
        transform = torch.tensor(
            [[[2.0, 0.0, -2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 1x3x3

        inp = utils.tensor_to_gradcheck_var(inp, requires_grad=True)  # to var

        assert gradcheck(
            kornia.geometry.transform.crop.crop_by_transform_mat, (inp, transform, (2, 2)), raise_exception=True
        )

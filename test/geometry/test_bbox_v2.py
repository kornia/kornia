from typing import Callable, Tuple

import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia
import kornia.testing as utils
from kornia.geometry.bbox_v2 import (
    bbox_to_kornia_bbox,
    bbox_to_mask,
    infer_bbox_shape,
    kornia_bbox_to_bbox,
    transform_bbox,
    validate_bbox,
    validate_bbox3d,
)


class TestBbox2D:
    def test_smoke(self, device, dtype):
        def _create_bbox():
            # Sample two points of the rectangle
            points = torch.rand(1, 4, device=device, dtype=dtype)

            # Fill according missing points
            bbox = torch.zeros(1, 4, 2, device=device, dtype=dtype)
            bbox[0, 0] = points[0][:2]
            bbox[0, 1, 0] = points[0][2]
            bbox[0, 1, 1] = points[0][1]
            bbox[0, 2] = points[0][2:]
            bbox[0, 3, 0] = points[0][0]
            bbox[0, 3, 1] = points[0][3]
            return bbox

        bbox = _create_bbox()
        # Validate
        assert validate_bbox(bbox)

        # Batch od 2 samples
        batched_bbox = torch.stack([_create_bbox(), _create_bbox()])
        assert validate_bbox(batched_bbox)

    def test_bounding_boxes_dim_inferring(self, device, dtype):
        box = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        boxes = torch.tensor(
            [[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]], [[2.0, 2.0], [5.0, 2.0], [5.0, 4.0], [2.0, 4.0]]],
            device=device,
            dtype=dtype,
        )

        h, w = infer_bbox_shape(box)
        assert (h.item(), w.item()) == (1, 2)

        h, w = infer_bbox_shape(boxes)
        assert h.ndim == 1 and w.ndim == 1
        assert len(h) == 2 and len(w) == 2
        assert (h == torch.as_tensor([1, 2])).all() and (w == torch.as_tensor([2, 3])).all()

    def test_bounding_boxes_dim_inferring_batch(self, device, dtype):
        box1 = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        box2 = torch.tensor([[[2.0, 2.0], [5.0, 2.0], [5.0, 4.0], [2.0, 4.0]]], device=device, dtype=dtype)
        batched_boxes = torch.stack([box1, box2])

        h, w = infer_bbox_shape(batched_boxes)
        assert h.ndim == 2 and w.ndim == 2
        assert h.shape == (2, 1) and w.shape == (2, 1)
        assert (h == torch.as_tensor([[1], [2]])).all() and (w == torch.as_tensor([[2], [3]])).all()

    @pytest.mark.parametrize('shape', [(1, 4), (1, 1, 4)])
    def test_bounding_boxes_convert_to_kornia(self, shape: Tuple[int], device, dtype):
        box_xyxy = torch.as_tensor([[1, 2, 3, 4]], device=device, dtype=dtype).view(*shape)
        box_xyxy_plus_1 = torch.as_tensor([[1, 2, 4, 5]], device=device, dtype=dtype).view(*shape)
        box_xywh = torch.as_tensor([[1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        box_xywh_plus_1 = torch.as_tensor([[1, 2, 3, 3]], device=device, dtype=dtype).view(*shape)

        expected_box = torch.as_tensor([[[1, 2], [3, 2], [3, 4], [1, 4]]], device=device, dtype=dtype).view(*shape, 2)

        kornia_xyxy = bbox_to_kornia_bbox(box_xyxy, mode='xyxy')
        kornia_xyxy_plus_1 = bbox_to_kornia_bbox(box_xyxy_plus_1, mode='xyxy_plus_1')
        kornia_xywh = bbox_to_kornia_bbox(box_xywh, mode='xywh')
        kornia_xywh_plus_1 = bbox_to_kornia_bbox(box_xywh_plus_1, mode='xywh_plus_1')

        assert kornia_xyxy.shape == expected_box.shape
        assert_allclose(kornia_xyxy, expected_box)

        assert kornia_xyxy_plus_1.shape == expected_box.shape
        assert_allclose(kornia_xyxy_plus_1, expected_box)

        assert kornia_xywh.shape == expected_box.shape
        assert_allclose(kornia_xywh, expected_box)

        assert kornia_xywh_plus_1.shape == expected_box.shape
        assert_allclose(kornia_xywh_plus_1, expected_box)

    @pytest.mark.parametrize('shape', [(1, 4), (1, 1, 4)])
    def test_bounding_boxes_convert_from_kornia(self, shape: Tuple[int], device, dtype):
        box = torch.as_tensor([[[1, 2], [3, 2], [3, 4], [1, 4]]], device=device, dtype=dtype).view(*shape, 2)

        expected_box_xyxy = torch.as_tensor([[1, 2, 3, 4]], device=device, dtype=dtype).view(*shape)
        expected_box_xyxy_plus_1 = torch.as_tensor([[1, 2, 4, 5]], device=device, dtype=dtype).view(*shape)
        expected_box_xywh = torch.as_tensor([[1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        expected_box_xywh_plus_1 = torch.as_tensor([[1, 2, 3, 3]], device=device, dtype=dtype).view(*shape)

        kornia_xyxy = kornia_bbox_to_bbox(box, mode='xyxy')
        kornia_xyxy_plus_1 = kornia_bbox_to_bbox(box, mode='xyxy_plus_1')
        kornia_xywh = kornia_bbox_to_bbox(box, mode='xywh')
        kornia_xywh_plus_1 = kornia_bbox_to_bbox(box, mode='xywh_plus_1')

        assert kornia_xyxy.shape == expected_box_xyxy.shape
        assert_allclose(kornia_xyxy, expected_box_xyxy)

        assert kornia_xyxy_plus_1.shape == expected_box_xyxy_plus_1.shape
        assert_allclose(kornia_xyxy_plus_1, expected_box_xyxy_plus_1)

        assert kornia_xywh.shape == expected_box_xywh.shape
        assert_allclose(kornia_xywh, expected_box_xywh)

        assert kornia_xywh_plus_1.shape == expected_box_xywh_plus_1.shape
        assert_allclose(kornia_xywh_plus_1, expected_box_xywh_plus_1)

    def test_gradcheck(self, device, dtype):
        boxes1 = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        boxes1 = utils.tensor_to_gradcheck_var(boxes1)
        boxes2 = utils.tensor_to_gradcheck_var(boxes1.detach().clone())

        boxes_xyxy = torch.tensor([[1.0, 3.0, 5.0, 6.0]])

        assert gradcheck(infer_bbox_shape, (boxes1,), raise_exception=True)
        assert gradcheck(kornia_bbox_to_bbox, (boxes2,), raise_exception=True)
        assert gradcheck(bbox_to_kornia_bbox, (boxes_xyxy,), raise_exception=True)

    @pytest.mark.parametrize(
        'op', [validate_bbox, infer_bbox_shape, bbox_to_mask, bbox_to_kornia_bbox, kornia_bbox_to_bbox]
    )
    def test_jit(self, op: Callable, device, dtype):
        # Define script
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        # Run
        expected = op(boxes)
        actual = op_script(boxes)
        # Compare
        assert_allclose(actual, expected)

    def test_jit_convert_to_kornia_format(self, device, dtype):
        # Define script
        op = bbox_to_kornia_bbox
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[1, 2, 3, 4]], device=device, dtype=dtype)
        # Run
        expected = op(boxes)
        actual = op_script(boxes)
        # Compare
        assert_allclose(actual, expected)


class TestTransformBoxes2D:
    def test_transform_boxes(self, device, dtype):

        boxes = bbox_to_kornia_bbox(
            torch.tensor([[139.2640, 103.0150, 397.3120, 410.5225]], device=device, dtype=dtype)
        )

        expected = bbox_to_kornia_bbox(
            torch.tensor([[372.7360, 103.0150, 114.6880, 410.5225]], device=device, dtype=dtype)
        )

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        out = transform_bbox(boxes, trans_mat)
        assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_transform_multiple_boxes(self, device, dtype):

        boxes = torch.tensor(
            [
                [139.2640, 103.0150, 397.3120, 410.5225],
                [1.0240, 80.5547, 512.0000, 512.0000],
                [165.2053, 262.1440, 510.6347, 508.9280],
                [119.8080, 144.2067, 257.0240, 410.1292],
            ],
            device=device,
            dtype=dtype,
        )

        boxes = boxes.repeat(2, 1, 1)  # 2 x 4 x 4 two images 4 boxes each

        expected = torch.tensor(
            [
                [
                    [372.7360, 103.0150, 114.6880, 410.5225],
                    [510.9760, 80.5547, 0.0000, 512.0000],
                    [346.7947, 262.1440, 1.3653, 508.9280],
                    [392.1920, 144.2067, 254.9760, 410.1292],
                ],
                [
                    [139.2640, 103.0150, 397.3120, 410.5225],
                    [1.0240, 80.5547, 512.0000, 512.0000],
                    [165.2053, 262.1440, 510.6347, 508.9280],
                    [119.8080, 144.2067, 257.0240, 410.1292],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        trans_mat = torch.tensor(
            [
                [[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            device=device,
            dtype=dtype,
        )

        kornia_bboxes = bbox_to_kornia_bbox(boxes)
        expected_kornia_bboxes = bbox_to_kornia_bbox(expected)

        out = transform_bbox(kornia_bboxes, trans_mat)
        assert_allclose(out, expected_kornia_bboxes, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):

        boxes = torch.tensor(
            [
                [139.2640, 103.0150, 258.0480, 307.5075],
                [1.0240, 80.5547, 510.9760, 431.4453],
                [165.2053, 262.1440, 345.4293, 246.7840],
                [119.8080, 144.2067, 137.2160, 265.9225],
            ],
            device=device,
            dtype=dtype,
        )

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        trans_mat = utils.tensor_to_gradcheck_var(trans_mat)
        boxes = utils.tensor_to_gradcheck_var(bbox_to_kornia_bbox(boxes))

        assert gradcheck(transform_bbox, (boxes, trans_mat), raise_exception=True)

    def test_jit(self, device, dtype):
        boxes = torch.tensor([[139.2640, 103.0150, 258.0480, 307.5075]], device=device, dtype=dtype)
        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        args = (trans_mat, bbox_to_kornia_bbox(boxes))
        op = kornia.geometry.transform_points
        op_jit = torch.jit.script(op)
        assert_allclose(op(*args), op_jit(*args))


class TestBbox3D:
    def test_smoke(self, device, dtype):
        # Sample two points of the 3d rect
        points = torch.rand(1, 6, device=device, dtype=dtype)

        # Fill acording missing points
        bbox = torch.zeros(1, 8, 3, device=device, dtype=dtype)
        bbox[0, 0] = points[0][:3]
        bbox[0, 1, 0] = points[0][3]
        bbox[0, 1, 1] = points[0][1]
        bbox[0, 1, 2] = points[0][2]
        bbox[0, 2, 0] = points[0][3]
        bbox[0, 2, 1] = points[0][4]
        bbox[0, 2, 2] = points[0][2]
        bbox[0, 3, 0] = points[0][0]
        bbox[0, 3, 1] = points[0][4]
        bbox[0, 3, 2] = points[0][2]
        bbox[0, 4, 0] = points[0][0]
        bbox[0, 4, 1] = points[0][1]
        bbox[0, 4, 2] = points[0][5]
        bbox[0, 5, 0] = points[0][3]
        bbox[0, 5, 1] = points[0][1]
        bbox[0, 5, 2] = points[0][5]
        bbox[0, 6] = points[0][3:]
        bbox[0, 7, 0] = points[0][0]
        bbox[0, 7, 1] = points[0][4]
        bbox[0, 7, 2] = points[0][5]

        # Validate
        assert validate_bbox3d(bbox)

    def test_bounding_boxes_dim_inferring(self, device, dtype):
        boxes = torch.tensor(
            [
                [[0, 1, 2], [10, 1, 2], [10, 21, 2], [0, 21, 2], [0, 1, 32], [10, 1, 32], [10, 21, 32], [0, 21, 32]],
                [[3, 4, 5], [43, 4, 5], [43, 54, 5], [3, 54, 5], [3, 4, 65], [43, 4, 65], [43, 54, 65], [3, 54, 65]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x8x3
        d, h, w = infer_bbox_shape3d(boxes)

        assert_allclose(d, torch.tensor([31.0, 61.0], device=device, dtype=dtype))
        assert_allclose(h, torch.tensor([21.0, 51.0], device=device, dtype=dtype))
        assert_allclose(w, torch.tensor([11.0, 41.0], device=device, dtype=dtype))

    def test_gradcheck(self, device, dtype):
        boxes = torch.tensor(
            [[[0, 1, 2], [10, 1, 2], [10, 21, 2], [0, 21, 2], [0, 1, 32], [10, 1, 32], [10, 21, 32], [0, 21, 32]]],
            device=device,
            dtype=dtype,
        )
        boxes = utils.tensor_to_gradcheck_var(boxes)
        assert gradcheck(infer_bbox_shape3d, (boxes,), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = infer_bbox_shape3d
        op_script = torch.jit.script(op)

        boxes = torch.tensor(
            [[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        actual = op_script(boxes)
        expected = op(boxes)
        assert_allclose(actual, expected)

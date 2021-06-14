import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia.testing as utils
from kornia.geometry.bbox import infer_bbox_shape_2d, infer_bbox_shape_3d, validate_bbox_2d, validate_bbox_3d


class TestBbox2D:
    def test_smoke(self, device, dtype):

        # Sample two points of the rectangle
        points = torch.rand(1, 4, device=device, dtype=dtype)

        # Fill acording missing points
        bbox = torch.zeros(1, 4, 2, device=device, dtype=dtype)
        bbox[0, 0] = points[0][:2]
        bbox[0, 1, 0] = points[0][2]
        bbox[0, 1, 1] = points[0][1]
        bbox[0, 2] = points[0][2:]
        bbox[0, 3, 0] = points[0][0]
        bbox[0, 3, 1] = points[0][3]

        # Validate
        assert validate_bbox_2d(bbox)

    def test_bounding_boxes_dim_inferring(self, device, dtype):
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        h, w = infer_bbox_shape_2d(boxes)
        assert (h, w) == (2, 3)

    def test_bounding_boxes_dim_inferring_batch(self, device, dtype):
        boxes = torch.tensor(
            [[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]], [[2.0, 2.0], [4.0, 2.0], [4.0, 3.0], [2.0, 3.0]]],
            device=device,
            dtype=dtype,
        )
        h, w = infer_bbox_shape_2d(boxes)
        assert (h.unique().item(), w.unique().item()) == (2, 3)

    def test_gradcheck(self, device, dtype):
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        boxes = utils.tensor_to_gradcheck_var(boxes)
        assert gradcheck(infer_bbox_shape_2d, (boxes,), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = infer_bbox_shape_2d
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        # Run
        expected = op(boxes)
        actual = op_script(boxes)
        # Compare
        assert_allclose(actual, expected)


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
        assert validate_bbox_3d(bbox)

    def test_bounding_boxes_dim_inferring(self, device, dtype):
        boxes = torch.tensor(
            [
                [[0, 1, 2], [10, 1, 2], [10, 21, 2], [0, 21, 2], [0, 1, 32], [10, 1, 32], [10, 21, 32], [0, 21, 32]],
                [[3, 4, 5], [43, 4, 5], [43, 54, 5], [3, 54, 5], [3, 4, 65], [43, 4, 65], [43, 54, 65], [3, 54, 65]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x8x3
        d, h, w = infer_bbox_shape_3d(boxes)

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
        assert gradcheck(infer_bbox_shape_3d, (boxes,), raise_exception=True)

    def test_jit(self, device, dtype):
        # Define script
        op = infer_bbox_shape_3d
        op_script = torch.jit.script(op)

        boxes = torch.tensor(
            [[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        actual = op_script(boxes)
        expected = op(boxes)
        assert_allclose(actual, expected)

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

import torch

import kornia
from kornia.geometry.bbox import (
    infer_bbox_shape,
    infer_bbox_shape3d,
    nms,
    transform_bbox,
    validate_bbox,
    validate_bbox3d,
)

from testing.base import BaseTester


class TestBbox2D(BaseTester):
    def test_smoke(self, device, dtype):
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

        # Validate
        assert validate_bbox(bbox)

    def test_bounding_boxes_dim_inferring(self, device, dtype):
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        h, w = infer_bbox_shape(boxes)
        assert (h, w) == (2, 3)

    def test_bounding_boxes_dim_inferring_batch(self, device, dtype):
        boxes = torch.tensor(
            [[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]], [[2.0, 2.0], [4.0, 2.0], [4.0, 3.0], [2.0, 3.0]]],
            device=device,
            dtype=dtype,
        )
        h, w = infer_bbox_shape(boxes)
        assert (h.unique().item(), w.unique().item()) == (2, 3)

    def test_gradcheck(self, device):
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=torch.float64)
        self.gradcheck(infer_bbox_shape, (boxes,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        # Define script
        op = infer_bbox_shape
        op_optimized = torch_optimizer(op)
        # Define input
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        # Run
        expected = op(boxes)
        actual = op_optimized(boxes)
        # Compare
        self.assert_close(actual, expected)

    def test_jit(self, device, dtype):
        # Test with valid rectangular box
        boxes = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)

        # JIT compile the validate_bbox function
        scripted_fn = torch.jit.script(validate_bbox)

        # Test with valid box
        self.assert_close(scripted_fn(boxes), validate_bbox(boxes))

        # Test with non-rectangular box
        boxes_invalid = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]]], device=device, dtype=dtype)
        self.assert_close(scripted_fn(boxes_invalid), validate_bbox(boxes_invalid))

        # Test with invalid shape
        boxes_wrong_shape = torch.rand(1, 3, 2, device=device, dtype=dtype)
        self.assert_close(scripted_fn(boxes_wrong_shape), validate_bbox(boxes_wrong_shape))


class TestTransformBoxes2D(BaseTester):
    def test_transform_boxes(self, device, dtype):
        boxes = torch.tensor([[139.2640, 103.0150, 397.3120, 410.5225]], device=device, dtype=dtype)

        expected = torch.tensor([[114.6880, 103.0150, 372.7360, 410.5225]], device=device, dtype=dtype)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        out = transform_bbox(trans_mat, boxes, restore_coordinates=True)
        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

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
                    [114.6880, 103.0150, 372.7360, 410.5225],
                    [0.0000, 80.5547, 510.9760, 512.0000],
                    [1.3652, 262.1440, 346.7947, 508.9280],
                    [254.9760, 144.2067, 392.1920, 410.1292],
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

        out = transform_bbox(trans_mat, boxes, restore_coordinates=True)
        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_transform_boxes_wh(self, device, dtype):
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

        expected = torch.tensor(
            [
                [114.6880, 103.0150, 258.0480, 307.5075],
                [0.0000, 80.5547, 510.9760, 431.4453],
                [1.3654, 262.1440, 345.4293, 246.7840],
                [254.9760, 144.2067, 137.2160, 265.9225],
            ],
            device=device,
            dtype=dtype,
        )

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        out = transform_bbox(trans_mat, boxes, mode="xywh", restore_coordinates=True)
        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        boxes = torch.tensor(
            [
                [139.2640, 103.0150, 258.0480, 307.5075],
                [1.0240, 80.5547, 510.9760, 431.4453],
                [165.2053, 262.1440, 345.4293, 246.7840],
                [119.8080, 144.2067, 137.2160, 265.9225],
            ],
            device=device,
            dtype=torch.float64,
        )

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=torch.float64
        )

        self.gradcheck(transform_bbox, (trans_mat, boxes, "xyxy", True))

    def test_dynamo(self, device, dtype, torch_optimizer):
        boxes = torch.tensor([[139.2640, 103.0150, 258.0480, 307.5075]], device=device, dtype=dtype)
        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        args = (boxes, trans_mat)
        op = kornia.geometry.transform_points
        op_optimized = torch_optimizer(op)
        self.assert_close(op(*args), op_optimized(*args))


class TestBbox3D(BaseTester):
    def test_smoke(self, device, dtype):
        # Sample two points of the 3d rect
        points = torch.rand(1, 6, device=device, dtype=dtype)

        # Fill according missing points
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

        self.assert_close(d, torch.tensor([31.0, 61.0], device=device, dtype=dtype))
        self.assert_close(h, torch.tensor([21.0, 51.0], device=device, dtype=dtype))
        self.assert_close(w, torch.tensor([11.0, 41.0], device=device, dtype=dtype))

    def test_gradcheck(self, device):
        boxes = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0],
                    [10, 1, 2],
                    [10, 21, 2],
                    [0, 21, 2],
                    [0, 1, 32],
                    [10, 1, 32],
                    [10, 21, 32],
                    [0, 21, 32],
                ]
            ],
            device=device,
            dtype=torch.float64,
        )
        self.gradcheck(infer_bbox_shape3d, (boxes,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        # Define script
        op = infer_bbox_shape3d
        op_script = torch_optimizer(op)

        boxes = torch.tensor(
            [[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        actual = op_script(boxes)
        expected = op(boxes)
        self.assert_close(actual, expected)


class TestNMS(BaseTester):
    def test_smoke(self, device, dtype):
        boxes = torch.tensor(
            [
                [10.0, 10.0, 20.0, 20.0],
                [15.0, 5.0, 15.0, 25.0],
                [100.0, 100.0, 200.0, 200.0],
                [100.0, 100.0, 200.0, 200.0],
            ],
            device=device,
            dtype=dtype,
        )
        scores = torch.tensor([0.9, 0.8, 0.7, 0.9], device=device, dtype=dtype)
        expected = torch.tensor([0, 3, 1], device=device, dtype=torch.long)
        actual = nms(boxes, scores, iou_threshold=0.8)
        self.assert_close(actual, expected)

    def test_no_overlap(self, device, dtype):
        boxes = torch.tensor(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ],
            device=device, dtype=dtype
        )
        scores = torch.tensor([0.9, 0.8, 0.7], device=device, dtype=dtype)
        expected = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
        actual = nms(boxes, scores, iou_threshold=0.5)
        self.assert_close(actual, expected)

    def test_complete_overlap(self, device, dtype):
        boxes = torch.tensor(
            [
                [0, 0, 10, 10],
                [0, 0, 10, 10],
                [0, 0, 10, 10],
            ],
            device=device, dtype=dtype
        )
        scores = torch.tensor([0.5, 0.9, 0.7], device=device, dtype=dtype)
        # Highest score (idx=1) kept, others removed
        expected = torch.tensor([1], device=device, dtype=torch.long)
        actual = nms(boxes, scores, iou_threshold=0.5)
        self.assert_close(actual, expected)

    def test_empty_input(self, device, dtype):
        boxes = torch.empty((0, 4), device=device, dtype=dtype)
        scores = torch.empty((0,), device=device, dtype=dtype)
        expected = torch.empty((0,), device=device, dtype=torch.long)
        actual = nms(boxes, scores, iou_threshold=0.5)
        self.assert_close(actual, expected)

    def test_single_box(self, device, dtype):
        boxes = torch.tensor([[0, 0, 10, 10]], device=device, dtype=dtype)
        scores = torch.tensor([0.8], device=device, dtype=dtype)
        expected = torch.tensor([0], device=device, dtype=torch.long)
        actual = nms(boxes, scores, iou_threshold=0.5)
        self.assert_close(actual, expected)

    def test_threshold_effect(self, device, dtype):
        boxes = torch.tensor(
            [
                [0, 0, 10, 10],
                [1, 1, 11, 11],  # High overlap
            ],
            device=device, dtype=dtype
        )
        scores = torch.tensor([0.9, 0.8], device=device, dtype=dtype)
        # Low threshold removes second box
        out_low = nms(boxes, scores, iou_threshold=0.3)
        self.assert_close(out_low, torch.tensor([0], device=device, dtype=torch.long))
        # High threshold keeps both
        out_high = nms(boxes, scores, iou_threshold=0.95)
        self.assert_close(out_high, torch.tensor([0, 1], device=device, dtype=torch.long))

    def test_identical_scores(self, device, dtype):
        boxes = torch.tensor(
            [
                [0, 0, 10, 10],
                [5, 5, 15, 15],
            ],
            device=device, dtype=dtype
        )
        scores = torch.tensor([0.8, 0.8], device=device, dtype=dtype)
        actual = nms(boxes, scores, iou_threshold=0.5)
        # Ensure deterministic ordering (PyTorch NMS is stable)
        expected = torch.tensor([0, 1], device=device, dtype=torch.long)
        self.assert_close(actual, expected)

    def test_large_random(self, device, dtype):
        torch.manual_seed(42)
        boxes = torch.rand((1000, 4), device=device, dtype=dtype) * 100
        boxes[:, 2:] += boxes[:, :2]  # ensure x2 >= x1, y2 >= y1
        scores = torch.rand((1000,), device=device, dtype=dtype)
        out = nms(boxes, scores, iou_threshold=0.5)
        assert out.ndim == 1
        assert out.dtype == torch.long

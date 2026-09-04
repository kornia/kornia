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
from kornia.geometry.bbox import (
    bbox_generator,
    bbox_generator3d,
    infer_bbox_shape,
    infer_bbox_shape3d,
    nms,
    transform_bbox,
    validate_bbox,
    validate_bbox3d,
)

from testing.base import BaseTester


class TestBbox2D(BaseTester):
    def test_wart_infer_bbox_shape_is_inclusive_3934(self, device, dtype):
        # Wart pin for kornia#3934: a raw box spanning x=1..4 and y=1..2 reports
        # (height, width) = (2, 4) because infer_bbox_shape adds one on both axes.
        # The non-square extents also pin the (heights, widths) return order.
        boxes = torch.tensor([[[1.0, 1.0], [4.0, 1.0], [4.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        height, width = infer_bbox_shape(boxes)
        self.assert_close(height, torch.tensor([2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(width, torch.tensor([4.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_convention_infer_bbox_shape_reads_fixed_vertex_indices(self, device, dtype):
        # Convention pin: the extents come from fixed vertex indices, not from a
        # maximum - minimum reduction. Swapping the top-left and top-right vertices in
        # x therefore yields a negative width, where max - min + 1 would give 5.
        # The -3 is 1 - 5 + 1; the inclusive offset is tracked in kornia#3934.
        boxes = torch.tensor([[[5.0, 2.0], [1.0, 2.0], [1.0, 3.0], [5.0, 3.0]]], device=device, dtype=dtype)
        height, width = infer_bbox_shape(boxes)
        self.assert_close(height, torch.tensor([2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(width, torch.tensor([-3.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_convention_infer_bbox_shape_preserves_generated_zero_extent(self, device, dtype):
        # bbox_generator represents a zero width by placing the top-right vertex one
        # unit left of the top-left vertex. RandomCutMixV2 relies on the fixed-index
        # reading recovering width 0; a maximum - minimum reduction would return 2.
        # The recovered zero depends on the inclusive +1 tracked in kornia#3934.
        x_start = torch.tensor([3.0], device=device, dtype=dtype)
        y_start = torch.tensor([4.0], device=device, dtype=dtype)
        boxes = bbox_generator(x_start, y_start, torch.zeros_like(x_start), torch.full_like(y_start, 5.0))
        height, width = infer_bbox_shape(boxes)
        self.assert_close(height, torch.tensor([5.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(width, torch.tensor([0.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_wart_infer_bbox_shape_rank4_is_not_validated_4180(self, device, dtype):
        # Wart pin for kornia#4180: rank-4 input is read at fixed box indices instead
        # of vertex indices. Fewer than three boxes raise, while three or more return
        # extents of shape (B, 2) built from the wrong vertices.
        with pytest.raises(IndexError):
            infer_bbox_shape(torch.zeros(1, 2, 4, 2, device=device, dtype=dtype))

        boxes = torch.arange(24, device=device, dtype=dtype).reshape(1, 3, 4, 2)
        height, width = infer_bbox_shape(boxes)
        self.assert_close(height, torch.tensor([[17.0, 17.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(width, torch.tensor([[9.0, 9.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_convention_validate_bbox_checks_parallelograms_and_accepts_contiguous_batched_boxes(self, device, dtype):
        # The validator only compares the top and bottom edge vectors at fixed vertex
        # indices. A cyclically ordered sheared parallelogram and rotated rectangle
        # both return True, while the trapezoid fails. The leading dimensions also
        # pin contiguous rank-4 input.
        sheared_parallelogram = torch.tensor(
            [[[[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]]]], device=device, dtype=dtype
        )
        assert validate_bbox(sheared_parallelogram) is True

        rotated_rectangle = torch.tensor(
            [[[[0.0, 0.0], [3.0, 4.0], [-1.0, 7.0], [-4.0, 3.0]]]], device=device, dtype=dtype
        )
        assert validate_bbox(rotated_rectangle) is True

        relabeled_rectangle = torch.tensor(
            [[[[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [3.0, 2.0]]]], device=device, dtype=dtype
        )
        assert validate_bbox(relabeled_rectangle) is False

        trapezoid = torch.tensor([[[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [3.0, 5.0]]]], device=device, dtype=dtype)
        assert validate_bbox(trapezoid) is False

    def test_convention_validate_bbox_accepts_noncontiguous_rank4_layout_4174(self, device, dtype):
        # Fix pin for kornia#4174: rank-4 input is flattened with reshape, so a stride
        # layout whose leading dimensions cannot be merged by view now returns a
        # boolean instead of raising RuntimeError.
        boxes = torch.zeros(2, 3, 4, 2, device=device, dtype=dtype).transpose(0, 1)
        assert not boxes.is_contiguous()
        assert validate_bbox(boxes) is True

        sliced = torch.zeros(2, 4, 4, 2, device=device, dtype=dtype)[:, 1:]
        assert not sliced.is_contiguous()
        assert validate_bbox(sliced) is True

    def test_convention_validate_bbox_invariance_is_exact_arithmetic_only(self, device):
        # In float16 the inclusive +1 rounds distinct sub-unit spans to the same
        # value, although the exclusive span difference exceeds the 1e-4 threshold.
        # The True below holds only with the +1 tracked in kornia#3934.
        boxes = torch.tensor(
            [[[0.0, 0.0], [0.0005, 0.0], [0.001, 0.001], [0.0, 0.001]]], device=device, dtype=torch.float16
        )
        assert validate_bbox(boxes) is True
        top_span = boxes[..., 1, 0] - boxes[..., 0, 0]
        bottom_span = boxes[..., 2, 0] - boxes[..., 3, 0]
        assert torch.all(torch.abs(top_span - bottom_span) > 1e-4)

    def test_wart_validate_bbox_returns_false_where_validate_bbox3d_raises_4013(self, device, dtype):
        # Wart pin for kornia#4013: for the same invalid shape, the 2D validator
        # returns False while the 3D validator raises.
        invalid_shape = torch.rand(1, 3, 3, device=device, dtype=dtype)
        assert validate_bbox(invalid_shape) is False
        with pytest.raises(AssertionError):
            validate_bbox3d(invalid_shape)

        non_cube = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        with pytest.raises(AssertionError):
            validate_bbox3d(non_cube)

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

        # Test with a valid sheared parallelogram
        boxes_sheared = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]]], device=device, dtype=dtype)
        self.assert_close(scripted_fn(boxes_sheared), validate_bbox(boxes_sheared))

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
        boxes_before = boxes.clone()

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
        assert torch.equal(boxes, boxes_before)

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
    def test_generator_scalar_inputs(self, device, dtype):
        args = [torch.tensor(value, device=device, dtype=dtype) for value in (1, 2, 3, 4, 5, 6)]
        boxes = bbox_generator3d(*args)
        assert boxes.shape == (1, 8, 3)
        self.assert_close(boxes[0, 0], torch.tensor([1, 2, 3], device=device, dtype=dtype))
        self.assert_close(boxes[0, 6], torch.tensor([5, 7, 9], device=device, dtype=dtype))

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

    def test_convention_validate_bbox3d_accepts_noncontiguous_rank4_layout_4174(self, device, dtype):
        # Fix pin for kornia#4174: rank-4 input is flattened with reshape, so a stride
        # layout whose leading dimensions cannot be merged by view now returns a
        # boolean instead of raising RuntimeError.
        boxes = torch.zeros(2, 3, 8, 3, device=device, dtype=dtype).transpose(0, 1)
        assert not boxes.is_contiguous()
        assert validate_bbox3d(boxes) is True

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
    def test_empty(self, device):
        boxes = torch.empty((0, 4), device=device)
        scores = torch.empty((0,), device=device)
        actual = nms(boxes, scores, iou_threshold=0.8)
        assert actual.shape == (0,)
        assert actual.dtype == torch.long
        assert actual.device == boxes.device

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

    @pytest.mark.parametrize(
        ("boxes_shape", "scores_shape"),
        [
            ((3, 5), (3,)),
            ((2, 3, 4), (2,)),
        ],
    )
    def test_invalid_boxes_shape(self, boxes_shape, scores_shape, device, dtype):
        boxes = torch.zeros(boxes_shape, device=device, dtype=dtype)
        scores = torch.zeros(scores_shape, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="boxes expected as Nx4"):
            nms(boxes, scores, iou_threshold=0.8)

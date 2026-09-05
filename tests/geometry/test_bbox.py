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
from kornia.core.exceptions import ShapeError
from kornia.geometry import bbox as bbox_module
from kornia.geometry.bbox import (
    bbox_generator,
    bbox_generator3d,
    bbox_to_mask,
    bbox_to_mask3d,
    infer_bbox_shape,
    infer_bbox_shape3d,
    nms,
    transform_bbox,
    validate_bbox,
    validate_bbox3d,
)
from kornia.geometry.boxes import Boxes3D

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

    @pytest.mark.parametrize("num_boxes", [2, 3])
    def test_infer_bbox_shape_rejects_rank4_4180(self, device, dtype, num_boxes):
        boxes = torch.zeros(1, num_boxes, 4, 2, device=device, dtype=dtype)

        with pytest.raises(ShapeError, match="expected 3 dimensions, got 4"):
            infer_bbox_shape(boxes)

    @pytest.mark.parametrize("num_boxes", [2, 3])
    def test_bbox_to_mask_rejects_rank4_4180(self, device, dtype, num_boxes):
        boxes = torch.zeros(1, num_boxes, 4, 2, device=device, dtype=dtype)

        with pytest.raises(ShapeError, match="expected 3 dimensions, got 4"):
            bbox_to_mask(boxes, width=5, height=5)

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

        expanded = torch.zeros(1, 3, 4, 2, device=device, dtype=dtype).expand(2, -1, -1, -1)
        assert expanded.stride(0) == 0
        assert validate_bbox(expanded) is True

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

    def test_convention_bbox_to_mask_takes_width_then_height_and_reads_two_vertices_4014(self, device, dtype):
        # Convention pin (kornia#4014 records the argument-order split with Boxes.to_mask):
        # bbox_to_mask(boxes, width, height) returns (B, height, width), fills through the
        # top-left (index 0) and bottom-right (index 2) vertices inclusively, ignores the two
        # other vertices, keeps the input dtype, and carries no gradient path.
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        mask = bbox_to_mask(boxes, 5, 3)
        assert mask.shape == (1, 3, 5)
        assert mask.dtype == dtype
        expected = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(mask, expected, atol=0.0, rtol=0.0)
        assert bbox_to_mask(boxes, 3, 5).shape == (1, 5, 3)

        corrupted = boxes.clone()
        corrupted[0, 1] = 9.0
        corrupted[0, 3] = -9.0
        self.assert_close(bbox_to_mask(corrupted, 5, 3), expected, atol=0.0, rtol=0.0)

        assert bbox_to_mask(boxes.to(torch.int64), 5, 3).dtype == torch.int64
        assert not bbox_to_mask(boxes.clone().requires_grad_(), 5, 3).requires_grad

    def test_wart_bbox_to_mask_compares_raw_floats_inclusively_4015(self, device, dtype):
        # Wart pin for kornia#4015: the fractional box [1.4, 3.0] x [2.0, 2.6] covers the integer
        # columns 2..3 of row 2 under inclusive raw-float comparison, so the integer edges x=3 and
        # y=2 are inside. The [1.4, 3.6] x [1.4, 2.6] box in test_boxes.py covers two pixels here
        # and twelve under Boxes.to_mask's rounding of the exclusive export.
        boxes = torch.tensor([[[1.4, 2.0], [3.0, 2.0], [3.0, 2.6], [1.4, 2.6]]], device=device, dtype=dtype)
        expected = torch.zeros(1, 5, 6, device=device, dtype=dtype)
        expected[0, 2, 2:4] = 1.0
        self.assert_close(bbox_to_mask(boxes, 6, 5), expected, atol=0.0, rtol=0.0)
        fractional = torch.tensor([[[1.4, 1.4], [3.6, 1.4], [3.6, 2.6], [1.4, 2.6]]], device=device, dtype=dtype)
        assert bbox_to_mask(fractional, 6, 5).sum().item() == 2.0

    def test_convention_bbox_generator_far_corner_is_start_plus_size_minus_one_3934(self, device, dtype):
        # Convention pin (kornia#3934 tracks the inclusive arithmetic): width 3 from x=1 places
        # the right edge at x=3, which infer_bbox_shape reads back as width 3. A scalar input is a
        # batch of one, dtype and device are preserved, mixed dtypes raise, and the output keeps a
        # gradient path to its inputs.
        start = torch.tensor([1.0], device=device, dtype=dtype)
        size = torch.tensor([3.0], device=device, dtype=dtype)
        boxes = bbox_generator(start, start, size, size)
        expected = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype)
        self.assert_close(boxes, expected, atol=0.0, rtol=0.0)
        heights, widths = infer_bbox_shape(boxes)
        self.assert_close(heights, size, atol=0.0, rtol=0.0)
        self.assert_close(widths, size, atol=0.0, rtol=0.0)
        assert boxes.device == start.device

        self.assert_close(bbox_generator(start[0], start[0], size[0], size[0]), expected, atol=0.0, rtol=0.0)
        assert bbox_generator(*(t.clone().requires_grad_() for t in (start, start, size, size))).requires_grad
        other = torch.float32 if dtype == torch.float16 else torch.float16
        with pytest.raises(AssertionError, match="same dtype"):
            bbox_generator(start, start, size.to(other), size)


class TestTransformBoxes2D(BaseTester):
    def test_convention_transform_bbox_restores_vector_endpoints(self, device, dtype):
        # Under a horizontal flip, restoration sorts xyxy coordinates while
        # polygon vertices retain their transformed cyclic order.
        matrix = torch.tensor([[[-1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        vector = torch.tensor([[1.0, 1.0, 3.0, 2.0]], device=device, dtype=dtype)
        polygon = torch.tensor([[[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]]], device=device, dtype=dtype)

        self.assert_close(
            transform_bbox(matrix, vector), torch.tensor([[7.0, 1.0, 9.0, 2.0]], device=device, dtype=dtype)
        )
        self.assert_close(
            transform_bbox(matrix, polygon),
            torch.tensor([[[[9.0, 1.0], [7.0, 1.0], [7.0, 2.0], [9.0, 2.0]]]], device=device, dtype=dtype),
        )

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

    @staticmethod
    def _unit_cuboid(device, dtype) -> torch.Tensor:
        # Inclusive vertices for x in 1..3, y in 1..3, z in 1..2: width 3, height 3, depth 2.
        return torch.tensor(
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
        )

    def test_wart_validate_bbox3d_checks_equal_edge_extents_not_cuboids_4013(self, device, dtype, monkeypatch):
        # Wart pin for kornia#4013 (the 3D validator raises where the 2D one returns False): the
        # check compares the inclusive extents of the four edges parallel to each axis, so a
        # sheared parallelepiped and a zero-extent box pass, and only unequal edges raise. Under
        # graph capture the extent check is skipped and the shape check alone returns True.
        cuboid = self._unit_cuboid(device, dtype)
        assert validate_bbox3d(cuboid) is True
        assert validate_bbox3d(cuboid[None]) is True
        assert validate_bbox3d(torch.zeros(1, 8, 3, device=device, dtype=dtype)) is True

        sheared = cuboid.clone()
        sheared[0, 4:, 0] += 1.0
        assert validate_bbox3d(sheared) is True
        for extent, expected in zip(infer_bbox_shape3d(sheared), (2.0, 3.0, 3.0)):
            self.assert_close(extent, torch.tensor([expected], device=device, dtype=dtype), atol=0.0, rtol=0.0)

        unequal = cuboid.clone()
        unequal[0, 1, 0] = 5.0
        with pytest.raises(AssertionError, match="widths"):
            validate_bbox3d(unequal)
        with pytest.raises(AssertionError, match="shape"):
            validate_bbox3d(cuboid[0])
        monkeypatch.setattr(bbox_module, "is_exporting", lambda: True)
        assert validate_bbox3d(unequal) is True
        with pytest.raises(AssertionError, match="shape"):
            validate_bbox3d(cuboid[0])

    def test_convention_infer_bbox_shape3d_returns_depth_height_width_inclusive_3934(self, device, dtype):
        # Convention pin (kornia#3934 tracks the inclusive arithmetic): the tuple order is
        # (depths, heights, widths) and each extent is one larger than its coordinate span.
        boxes = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 2.0, 3.0],
                    [4.0, 4.0, 3.0],
                    [1.0, 4.0, 3.0],
                    [1.0, 2.0, 7.0],
                    [4.0, 2.0, 7.0],
                    [4.0, 4.0, 7.0],
                    [1.0, 4.0, 7.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        depths, heights, widths = infer_bbox_shape3d(boxes)
        self.assert_close(depths, torch.tensor([5.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(heights, torch.tensor([3.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(widths, torch.tensor([4.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("num_boxes", [1, 8])
    def test_wart_rank4_input_passes_validate_bbox3d_then_breaks_the_3d_helpers_4248(self, device, dtype, num_boxes):
        # Wart pin for kornia#4248: validate_bbox3d accepts (B, N, 8, 3), but infer_bbox_shape3d and bbox_to_mask3d
        # then index the box axis as the vertex axis. With one box that raises out-of-bounds errors;
        # with eight boxes infer_bbox_shape3d instead returns three (1, 3) tensors, one value per
        # coordinate rather than per box. The 2D helpers had the same defect until kornia#4180.
        boxes = self._unit_cuboid(device, dtype).expand(num_boxes, 8, 3)[None].contiguous()
        assert validate_bbox3d(boxes) is True
        if num_boxes == 8:
            depths, heights, widths = infer_bbox_shape3d(boxes)
            assert depths.shape == heights.shape == widths.shape == (1, 3)
            with pytest.raises(RuntimeError):
                bbox_to_mask3d(boxes, (4, 5, 5))
        else:
            with pytest.raises((RuntimeError, IndexError)):
                infer_bbox_shape3d(boxes)
            with pytest.raises((RuntimeError, IndexError)):
                bbox_to_mask3d(boxes, (4, 5, 5))

    @pytest.mark.parametrize("num_boxes", [1, 8])
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="kornia#4248: rank-4 3D input is not rejected with ShapeError as the 2D helpers do since kornia#4218",
    )
    def test_convention_rank4_input_is_rejected_by_the_3d_helpers_4248(self, device, dtype, num_boxes):
        boxes = self._unit_cuboid(device, dtype).expand(num_boxes, 8, 3)[None].contiguous()
        for call in (lambda: infer_bbox_shape3d(boxes), lambda: bbox_to_mask3d(boxes, (4, 5, 5))):
            try:
                call()
            except ShapeError:
                continue
            except Exception as error:
                raise AssertionError(f"expected ShapeError, got {type(error).__name__}") from error
            raise AssertionError("expected ShapeError, the call returned")

    def test_wart_bbox_generator3d_extent_is_one_larger_than_requested_4018(self, device, dtype):
        # Wart pin for kornia#4018: the 3D generator places the far corner at start + size, so
        # width 3 from x=1 spans x=1..4 and infer_bbox_shape3d reads 4 on every axis, where the
        # 2D generator with the same arguments spans x=1..3 and reads 3.
        start = torch.tensor([1.0], device=device, dtype=dtype)
        size = torch.tensor([3.0], device=device, dtype=dtype)
        boxes = bbox_generator3d(start, start, start, size, size, size)
        expected_x = torch.tensor([[1.0, 4.0, 4.0, 1.0, 1.0, 4.0, 4.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(boxes[..., 0], expected_x, atol=0.0, rtol=0.0)
        for extent in infer_bbox_shape3d(boxes):
            self.assert_close(extent, size + 1, atol=0.0, rtol=0.0)
        for extent in infer_bbox_shape(bbox_generator(start, start, size, size)):
            self.assert_close(extent, size, atol=0.0, rtol=0.0)

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason="kornia#4018: bbox_generator3d extent is size + 1")
    def test_convention_bbox_generator3d_matches_bbox_generator_extent_4018(self, device, dtype):
        start = torch.tensor([1.0], device=device, dtype=dtype)
        size = torch.tensor([3.0], device=device, dtype=dtype)
        for extent in infer_bbox_shape3d(bbox_generator3d(start, start, start, size, size, size)):
            self.assert_close(extent, size, atol=0.0, rtol=0.0)

    def test_wart_bbox_to_mask3d_truncates_fractional_coordinates_4015(self, device, dtype):
        # Wart pin for kornia#4015: x in [1.6, 3.4] truncates to 1..3 and y, z in [1.6, 2.4] to
        # 1..2, filled inclusively: twelve voxels, where Boxes3D.to_mask fills two (test_boxes.py).
        boxes = torch.tensor(
            [
                [
                    [1.6, 1.6, 1.6],
                    [3.4, 1.6, 1.6],
                    [3.4, 2.4, 1.6],
                    [1.6, 2.4, 1.6],
                    [1.6, 1.6, 2.4],
                    [3.4, 1.6, 2.4],
                    [3.4, 2.4, 2.4],
                    [1.6, 2.4, 2.4],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        expected = torch.zeros(1, 1, 5, 5, 6, device=device, dtype=torch.float32)
        expected[0, 0, 1:3, 1:3, 1:4] = 1.0
        self.assert_close(bbox_to_mask3d(boxes, (5, 5, 6)), expected, atol=0.0, rtol=0.0)

    def test_wart_bbox_to_mask3d_returns_float32_with_a_channel_axis_4250(self, device, dtype):
        # Wart pin for kornia#4250: the 3D free function returns float32 (B, 1, D, H, W) whatever the input dtype,
        # while bbox_to_mask keeps the input dtype and returns (B, H, W). Neither carries a gradient.
        cuboid = self._unit_cuboid(device, dtype)
        mask = bbox_to_mask3d(cuboid, (4, 5, 6))
        assert mask.shape == (1, 1, 4, 5, 6)
        assert mask.dtype == torch.float32
        assert bbox_to_mask3d(cuboid.repeat(2, 1, 1), (4, 5, 6)).shape == (2, 1, 4, 5, 6)
        assert bbox_to_mask3d(cuboid.to(torch.int64), (4, 5, 6)).dtype == torch.float32
        assert not bbox_to_mask3d(cuboid.clone().requires_grad_(), (4, 5, 6)).requires_grad
        square = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        assert bbox_to_mask(square, 6, 5).dtype == dtype

    @pytest.mark.parametrize("case", ["full_width", "full_height", "full_depth", "overhang"])
    def test_wart_bbox_to_mask3d_fills_the_whole_volume_when_a_box_spans_an_axis_4255(self, device, dtype, case):
        # Wart pin for kornia#4255: once one axis slab covers every index of the (4, 4, 5) volume, the
        # union-of-planes intermediate is all true and its reductions lose the other two bounds, so all
        # 80 voxels are filled where Boxes3D.to_mask fills the intersection. The interior box is unaffected.
        xyzxyz_plus, intersection = {
            "full_width": ([0.0, 1.0, 1.0, 4.0, 2.0, 2.0], 20.0),
            "full_height": ([1.0, 0.0, 1.0, 2.0, 3.0, 2.0], 16.0),
            "full_depth": ([1.0, 1.0, 0.0, 2.0, 2.0, 3.0], 16.0),
            "overhang": ([-1.0, 1.0, 1.0, 5.0, 2.0, 2.0], 20.0),
        }[case]
        boxes = Boxes3D.from_tensor(torch.tensor([xyzxyz_plus], device=device, dtype=dtype), mode="xyzxyz_plus")
        assert bbox_to_mask3d(boxes.data, (4, 4, 5)).sum().item() == 80.0
        assert boxes.to_mask(4, 4, 5).sum().item() == intersection
        interior = Boxes3D.from_tensor(
            torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]], device=device, dtype=dtype), mode="xyzxyz_plus"
        )
        assert bbox_to_mask3d(interior.data, (4, 4, 5)).sum().item() == interior.to_mask(4, 4, 5).sum().item() == 8.0

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="kornia#4255: bbox_to_mask3d fills the whole volume for a full-axis box",
    )
    def test_convention_bbox_to_mask3d_intersects_the_axis_ranges_for_a_full_axis_box_4255(self, device, dtype):
        # x spans the whole width 0..4, y and z cover 1..2: the intersection is 2 * 2 * 5 = 20 voxels.
        boxes = Boxes3D.from_tensor(
            torch.tensor([[0.0, 1.0, 1.0, 4.0, 2.0, 2.0]], device=device, dtype=dtype), mode="xyzxyz_plus"
        )
        expected = torch.zeros(1, 1, 4, 4, 5, device=device, dtype=torch.float32)
        expected[0, 0, 1:3, 1:3, :] = 1.0
        self.assert_close(bbox_to_mask3d(boxes.data, (4, 4, 5)), expected, atol=0.0, rtol=0.0)


class TestNMS(BaseTester):
    def test_convention_nms_uses_exclusive_xyxy_iou_4008(self, device, dtype):
        # The boxes overlap by 1 / 4 under exclusive xyxy arithmetic. At 0.3, NMS
        # keeps both; inclusive corner arithmetic would compute 4 / 9 and suppress
        # the lower-scored box. The literal also pins descending-score index order.
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]], device=device, dtype=dtype)
        scores = torch.tensor([0.8, 0.9], device=device, dtype=dtype)

        actual = nms(boxes, scores, iou_threshold=0.3)
        self.assert_close(actual, torch.tensor([1, 0], device=device, dtype=torch.long), atol=0.0, rtol=0.0)

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

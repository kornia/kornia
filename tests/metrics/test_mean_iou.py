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


class TestMeanIoU(BaseTester):
    def test_two_classes_perfect(self, device, dtype):
        batch_size = 1
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long)

        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor([[1.0, 1.0]], device=device, dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        self.assert_close(mean_iou, mean_iou_real)

    def test_two_classes_perfect_batch2(self, device, dtype):
        batch_size = 2
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long).repeat(batch_size, 1)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long).repeat(batch_size, 1)

        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device, dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        self.assert_close(mean_iou, mean_iou_real)

    def test_two_classes(self, device, dtype):
        batch_size = 1
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1]], device=device, dtype=torch.long)

        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor([[0.75, 0.80]], device=device, dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        self.assert_close(mean_iou, mean_iou_real)

    def test_four_classes_2d_perfect(self, device, dtype):
        batch_size = 1
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=device, dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        self.assert_close(mean_iou, mean_iou_real)

    def test_four_classes_one_missing(self, device, dtype):
        batch_size = 1
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[3, 3, 2, 2], [3, 3, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor([[0.0, 1.0, 0.5, 0.5]], device=device, dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        self.assert_close(mean_iou, mean_iou_real)

    def test_exception_shape_mismatch(self, device, dtype):
        pred = torch.zeros(1, 4, dtype=torch.long, device=device)
        target = torch.zeros(1, 5, dtype=torch.long, device=device)
        with pytest.raises(ValueError, match="same shape"):
            kornia.metrics.mean_iou(pred, target, num_classes=2)

    def test_exception_num_classes_too_small(self, device, dtype):
        pred = torch.zeros(1, 4, dtype=torch.long, device=device)
        with pytest.raises(ValueError, match="bigger than two"):
            kornia.metrics.mean_iou(pred, pred, num_classes=1)

    def test_empty_batch(self, device, dtype):
        num_classes = 3
        pred = torch.zeros(0, 4, 4, device=device, dtype=torch.long)
        target = torch.zeros(0, 4, 4, device=device, dtype=torch.long)

        mean_iou = kornia.metrics.mean_iou(pred, target, num_classes)
        assert mean_iou.shape == (0, num_classes)
        assert mean_iou.dtype == torch.float32
        assert mean_iou.device == pred.device


class TestMeanIoUBBox(BaseTester):
    """Tests for mean_iou_bbox with different box formats."""

    @pytest.mark.parametrize(
        "box_format,coordinates",
        [
            ("xyxy", [[0, 0, 512, 512], [256, 0, 768, 512]]),
            ("xywh", [[0, 0, 512, 512], [256, 0, 512, 512]]),
            ("cxcywh", [[256, 256, 512, 512], [512, 256, 512, 512]]),
        ],
    )
    def test_image_sized_boxes(self, box_format, coordinates, device, dtype):
        boxes = torch.tensor(coordinates, device=device, dtype=dtype)
        original = boxes.clone()
        actual = kornia.metrics.mean_iou_bbox(boxes, boxes, box_format)
        # Each box has area 512**2; their intersection is 256*512 and their union is 768*512.
        expected = torch.tensor([[1.0, 1.0 / 3.0], [1.0 / 3.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(actual, expected, atol=0.0, rtol=0.0)
        self.assert_close(boxes, original, atol=0.0, rtol=0.0)
        assert actual.dtype == dtype
        assert actual.device == device

    @pytest.mark.parametrize("empty_shape", [(0, 4), (2, 4)])
    def test_empty_box_set(self, empty_shape, device, dtype):
        empty = torch.empty(0, 4, device=device, dtype=dtype)
        boxes = torch.tensor([[0, 0, 512, 512], [256, 0, 768, 512]], device=device, dtype=dtype)
        other = empty if empty_shape[0] == 0 else boxes
        actual = kornia.metrics.mean_iou_bbox(empty, other)
        assert actual.shape == (0, empty_shape[0])
        assert actual.dtype == dtype

    def test_image_sized_boxes_backward(self, device, dtype):
        first_coordinates = [[0, 0, 512, 512]]
        second_coordinates = [[256, 128, 768, 640]]
        first = torch.tensor(first_coordinates, device=device, dtype=dtype, requires_grad=True)
        second = torch.tensor(second_coordinates, device=device, dtype=dtype)
        actual = kornia.metrics.mean_iou_bbox(first, second)
        reference_input = torch.tensor(first_coordinates, dtype=torch.float64, requires_grad=True)
        reference = kornia.metrics.mean_iou_bbox(reference_input, torch.tensor(second_coordinates, dtype=torch.float64))
        actual.sum().backward()
        reference.sum().backward()
        assert first.grad is not None
        assert reference_input.grad is not None
        self.assert_close(first.grad, reference_input.grad.to(dtype=dtype).to(device=device), atol=1e-6, rtol=1e-3)

    @pytest.mark.parametrize("swap", [False, True])
    def test_areas_are_not_rounded_to_the_box_dtype(self, swap, device, dtype):
        # In float16 the areas 40000 and 40800 sum past the maximum, which gave an IoU of 0. Computed in bfloat16 the
        # IoU was 0.9765625, and rounding only the taller box's area gives 0.984375, instead of 0.98046875. Every
        # coordinate is exact in both dtypes.
        square = torch.tensor([[0, 0, 200, 200]], device=device, dtype=dtype)
        taller = torch.tensor([[0, 0, 200, 204]], device=device, dtype=dtype)
        boxes_1, boxes_2 = (taller, square) if swap else (square, taller)
        actual = kornia.metrics.mean_iou_bbox(boxes_1, boxes_2)
        expected = torch.tensor([[40000 / 40800]], device=device, dtype=dtype)
        self.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_integer_boxes_return_float32(self, device):
        boxes = torch.tensor([[0, 0, 512, 512], [256, 0, 768, 512]], device=device)
        actual = kornia.metrics.mean_iou_bbox(boxes, boxes)
        assert actual.dtype == torch.float32
        expected = torch.tensor([[1.0, 1.0 / 3.0], [1.0 / 3.0, 1.0]], device=device)
        self.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_bbox_xyxy_format(self, device, dtype):
        """Test XYXY format (original behavior)."""
        boxes_1 = torch.tensor([[40, 40, 60, 60], [30, 40, 50, 60]], device=device, dtype=dtype)
        boxes_2 = torch.tensor([[40, 50, 60, 70], [30, 40, 40, 50]], device=device, dtype=dtype)

        iou = kornia.metrics.mean_iou_bbox(boxes_1, boxes_2, box_format="xyxy")
        expected = torch.tensor([[0.3333, 0.0000], [0.1429, 0.2500]], device=device, dtype=dtype)

        self.assert_close(iou, expected, rtol=1e-3, atol=1e-4)

    def test_bbox_xywh_format(self, device, dtype):
        """Test XYWH format."""
        # Same boxes as xyxy test, but in xywh format
        boxes_1_xywh = torch.tensor([[40, 40, 20, 20], [30, 40, 20, 20]], device=device, dtype=dtype)
        boxes_2_xywh = torch.tensor([[40, 50, 20, 20], [30, 40, 10, 10]], device=device, dtype=dtype)

        iou = kornia.metrics.mean_iou_bbox(boxes_1_xywh, boxes_2_xywh, box_format="xywh")
        expected = torch.tensor([[0.3333, 0.0000], [0.1429, 0.2500]], device=device, dtype=dtype)

        self.assert_close(iou, expected, rtol=1e-3, atol=1e-4)

    def test_bbox_cxcywh_format(self, device, dtype):
        """Test CXCYWH format."""
        # Same boxes as xyxy test, but in cxcywh format
        boxes_1_cxcywh = torch.tensor([[50, 50, 20, 20], [40, 50, 20, 20]], device=device, dtype=dtype)
        boxes_2_cxcywh = torch.tensor([[50, 60, 20, 20], [35, 45, 10, 10]], device=device, dtype=dtype)

        iou = kornia.metrics.mean_iou_bbox(boxes_1_cxcywh, boxes_2_cxcywh, box_format="cxcywh")
        expected = torch.tensor([[0.3333, 0.0000], [0.1429, 0.2500]], device=device, dtype=dtype)

        self.assert_close(iou, expected, rtol=1e-3, atol=1e-4)

    def test_bbox_format_consistency(self, device, dtype):
        """Test that all formats produce same results for equivalent boxes."""
        # Define same boxes in three formats
        boxes_xyxy = torch.tensor([[10, 10, 20, 20]], device=device, dtype=dtype)
        boxes_xywh = torch.tensor([[10, 10, 10, 10]], device=device, dtype=dtype)
        boxes_cxcywh = torch.tensor([[15, 15, 10, 10]], device=device, dtype=dtype)

        iou_xyxy = kornia.metrics.mean_iou_bbox(boxes_xyxy, boxes_xyxy, box_format="xyxy")
        iou_xywh = kornia.metrics.mean_iou_bbox(boxes_xywh, boxes_xywh, box_format="xywh")
        iou_cxcywh = kornia.metrics.mean_iou_bbox(boxes_cxcywh, boxes_cxcywh, box_format="cxcywh")

        # All should give perfect IoU (1.0)
        expected = torch.tensor([[1.0]], device=device, dtype=dtype)
        self.assert_close(iou_xyxy, expected)
        self.assert_close(iou_xywh, expected)
        self.assert_close(iou_cxcywh, expected)

    def test_bbox_invalid_format(self, device, dtype):
        """Test that invalid format raises ValueError."""
        boxes = torch.tensor([[10, 10, 20, 20]], device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Unsupported box format"):
            kornia.metrics.mean_iou_bbox(boxes, boxes, box_format="invalid")

    def test_bbox_default_format(self, device, dtype):
        """Test that default format is xyxy."""
        boxes_1 = torch.tensor([[40, 40, 60, 60]], device=device, dtype=dtype)
        boxes_2 = torch.tensor([[40, 50, 60, 70]], device=device, dtype=dtype)

        iou_default = kornia.metrics.mean_iou_bbox(boxes_1, boxes_2)
        iou_explicit = kornia.metrics.mean_iou_bbox(boxes_1, boxes_2, box_format="xyxy")

        self.assert_close(iou_default, iou_explicit)

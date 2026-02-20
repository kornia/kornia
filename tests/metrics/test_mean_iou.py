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


class TestMeanIoUBBox(BaseTester):
    """Tests for mean_iou_bbox with different box formats."""

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

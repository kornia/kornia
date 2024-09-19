import pytest
import torch
from numpy.testing import assert_almost_equal

from kornia.core import tensor
from kornia.models.detection.utils import BoxFiltering  # Replace with the actual module path


class TestBoxFiltering:
    @pytest.fixture
    def sample_boxes(self):
        # Setup some sample boxes with the format [class_id, confidence_score, x, y, w, h]
        return tensor(
            [
                [
                    [1, 0.9, 10, 10, 20, 20],  # High confidence, class 1
                    [2, 0.7, 15, 15, 25, 25],  # Medium confidence, class 2
                    [3, 0.7, 15, 15, 25, 25],  # Medium confidence, class 3
                    [4, 0.3, 5, 5, 10, 10],
                ],  # Low confidence, class 4
                [
                    [1, 0.95, 12, 12, 18, 18],  # High confidence, class 1
                    [2, 0.5, 13, 13, 20, 20],  # Low confidence, class 2
                    [3, 0.5, 13, 13, 20, 20],  # Low confidence, class 3
                    [4, 0.2, 7, 7, 14, 14],
                ],  # Very low confidence, class 4
                [
                    [1, 0.1, 12, 12, 18, 18],  # Very Low confidence, class 1
                    [2, 0.1, 13, 13, 20, 20],  # Very Low confidence, class 2
                    [3, 0.1, 13, 13, 20, 20],  # Very Low confidence, class 3
                    [4, 0.1, 7, 7, 14, 14],
                ],  # Very Low confidence, class 4
            ]
        )  # Shape: [3, 4, 6], i.e., [B, D, 6]

    def test_confidence_filtering(self, sample_boxes):
        """Test filtering based on confidence threshold."""
        # Set a confidence threshold of 0.7
        filter = BoxFiltering(confidence_threshold=0.7)
        filtered_boxes = filter(sample_boxes)

        # Expected output: only boxes with confidence > 0.7 should be kept
        assert len(filtered_boxes[0]) == 1  # Only one box in the first batch
        assert_almost_equal(filtered_boxes[0][0][1].item(), 0.9)  # Box with confidence 0.9
        assert len(filtered_boxes[1]) == 1  # Only one box in the second batch
        assert_almost_equal(filtered_boxes[1][0][1].item(), 0.95)  # Box with confidence 0.95
        assert len(filtered_boxes[2]) == 0  # No boxes in the third batch

    def test_class_filtering(self, sample_boxes):
        """Test filtering based on class IDs."""
        # Set classes_to_keep to [1, 2]
        filter = BoxFiltering(classes_to_keep=tensor([1, 2]))
        filtered_boxes = filter(sample_boxes)

        # Expected output: only boxes with class_id 1 and 2 should be kept
        assert len(filtered_boxes[0]) == 2  # Two boxes in the first batch
        assert filtered_boxes[0][0][0].item() == 1  # Box with class_id 1
        assert filtered_boxes[0][1][0].item() == 2  # Box with class_id 2
        assert len(filtered_boxes[1]) == 2  # Two boxes in the second batch
        assert filtered_boxes[1][0][0].item() == 1  # Box with class_id 1
        assert filtered_boxes[1][1][0].item() == 2  # Box with class_id 2
        assert len(filtered_boxes[2]) == 2  # Two boxes in the third batch
        assert filtered_boxes[2][0][0].item() == 1  # Box with class_id 1
        assert filtered_boxes[2][1][0].item() == 2  # Box with class_id 2

    def test_combined_confidence_and_class_filtering(self, sample_boxes):
        """Test filtering based on both confidence and class IDs."""
        # Set confidence threshold to 0.6 and classes_to_keep to [1, 3]
        filter = BoxFiltering(confidence_threshold=0.6, classes_to_keep=tensor([1, 3]))
        filtered_boxes = filter(sample_boxes)

        # Expected output: only boxes with confidence > 0.6 and class_id in [1, 3] should be kept
        assert len(filtered_boxes[0]) == 2  # Two boxes in the first batch
        assert filtered_boxes[0][0][0].item() == 1  # Class_id 1
        assert filtered_boxes[0][1][0].item() == 3  # Class_id 3
        assert filtered_boxes[1][0][0].item() == 1  # Class_id 1
        assert len(filtered_boxes[1]) == 1  # No boxes in the second batch
        assert len(filtered_boxes[2]) == 0  # No boxes in the third batch

    def test_filter_as_zero(self, sample_boxes):
        """Test filtering boxes as zero when filter_as_zero is True."""
        filter = BoxFiltering(confidence_threshold=0.8, filter_as_zero=True)
        filtered_boxes = filter(sample_boxes)

        # Expected output: boxes with confidence <= 0.8 should be zeroed out
        assert torch.all(filtered_boxes[0][0] != 0)  # Box with confidence 0.9 should remain
        assert torch.all(filtered_boxes[0][1:] == 0)  # Remaining boxes should be zeroed
        assert torch.all(filtered_boxes[1][0] != 0)  # Box with confidence 0.95 should remain
        assert torch.all(filtered_boxes[1][1:] == 0)  # Remaining boxes should be zeroed
        assert torch.all(filtered_boxes[2] == 0)  # All boxes in the third batch should be zeroed

    def test_no_class_or_confidence_filtering(self, sample_boxes):
        """Test when no class or confidence filtering is applied."""
        filter = BoxFiltering()  # No thresholds set
        filtered_boxes = filter(sample_boxes)

        # Expected output: all boxes should be returned as-is
        assert len(filtered_boxes[0]) == 4  # All boxes in the first batch should be kept
        assert len(filtered_boxes[1]) == 4  # All boxes in the second batch should be kept
        assert len(filtered_boxes[2]) == 4  # All boxes in the third batch should be kept

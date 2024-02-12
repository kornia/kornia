import pytest
import torch

import kornia

from testing.base import BaseTester


class TestMeanAveragePrecision(BaseTester):
    def test_smoke(self, device, dtype):
        boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        scores = torch.tensor([0.7], device=device, dtype=dtype)

        gt_boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        gt_labels = torch.tensor([1], device=device, dtype=torch.long)

        mean_ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [gt_boxes], [gt_labels], 2)

        self.assert_close(mean_ap[0], torch.tensor(1.0, device=device, dtype=dtype))
        self.assert_close(mean_ap[1][1], 1.0)

    def test_raise(self, device, dtype):
        boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        scores = torch.tensor([0.7], device=device, dtype=dtype)

        gt_boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        gt_labels = torch.tensor([1], device=device, dtype=torch.long)

        with pytest.raises(AssertionError):
            _ = kornia.metrics.mean_average_precision(boxes[0], [labels], [scores], [gt_boxes], [gt_labels], 2)

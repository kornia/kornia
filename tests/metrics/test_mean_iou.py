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

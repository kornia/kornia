from __future__ import annotations

import pytest
import torch

import kornia
from kornia.testing import assert_close


class TestMeanIoU:
    def test_two_classes_perfect(self, device, dtype):
        batch_size = 1
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long)

        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor([[1.0, 1.0]], device=device, dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        assert_close(mean_iou, mean_iou_real)

    def test_two_classes_perfect_batch2(self, device, dtype):
        batch_size = 2
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long).repeat(batch_size, 1)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long).repeat(batch_size, 1)

        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device, dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        assert_close(mean_iou, mean_iou_real)

    def test_two_classes(self, device, dtype):
        batch_size = 1
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1]], device=device, dtype=torch.long)

        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou = kornia.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor([[0.75, 0.80]], device=device, dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        assert_close(mean_iou, mean_iou_real)

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
        assert_close(mean_iou, mean_iou_real)

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
        assert_close(mean_iou, mean_iou_real)


class TestConfusionMatrix:
    def test_two_classes(self, device, dtype):
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1]], device=device, dtype=torch.long)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor([[[3, 1], [0, 4]]], device=device, dtype=torch.float32)
        assert_close(conf_mat, conf_mat_real)

    def test_two_classes_batch2(self, device, dtype):
        batch_size = 2
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long).repeat(batch_size, 1)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1]], device=device, dtype=torch.long).repeat(batch_size, 1)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor([[[3, 1], [0, 4]], [[3, 1], [0, 4]]], device=device, dtype=torch.float32)
        assert_close(conf_mat, conf_mat_real)

    def test_three_classes(self, device, dtype):
        num_classes = 3
        actual = torch.tensor([[2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 1, 2, 1, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2]], device=device, dtype=torch.long)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor([[[4, 1, 2], [3, 0, 2], [1, 2, 1]]], device=device, dtype=torch.float32)
        assert_close(conf_mat, conf_mat_real)

    def test_four_classes_one_missing(self, device, dtype):
        num_classes = 4
        actual = torch.tensor([[3, 3, 1, 1, 2, 1, 1, 3, 2, 2, 1, 1, 2, 3, 2, 1]], device=device, dtype=torch.long)
        predicted = torch.tensor([[3, 2, 1, 1, 1, 1, 1, 2, 1, 3, 3, 2, 1, 1, 3, 3]], device=device, dtype=torch.long)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 0, 0], [0, 4, 1, 2], [0, 3, 0, 2], [0, 1, 2, 1]]], device=device, dtype=torch.float32
        )
        assert_close(conf_mat, conf_mat_real)

    def test_three_classes_normalized(self, device, dtype):
        num_classes = 3
        normalized = True
        actual = torch.tensor([[2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 1, 2, 1, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2]], device=device, dtype=torch.long)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes, normalized)

        conf_mat_real = torch.tensor(
            [[[0.5000, 0.3333, 0.4000], [0.3750, 0.0000, 0.4000], [0.1250, 0.6667, 0.2000]]],
            device=device,
            dtype=torch.float32,
        )

        assert_close(conf_mat, conf_mat_real)

    def test_four_classes_2d_perfect(self, device, dtype):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]], device=device, dtype=torch.float32
        )
        assert_close(conf_mat, conf_mat_real)

    def test_four_classes_2d_one_class_nonperfect(self, device, dtype):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[0, 0, 1, 1], [0, 3, 0, 1], [2, 2, 1, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[3, 0, 0, 1], [1, 3, 0, 0], [0, 0, 4, 0], [0, 1, 0, 3]]], device=device, dtype=torch.float32
        )
        assert_close(conf_mat, conf_mat_real)

    def test_four_classes_2d_one_class_missing(self, device, dtype):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[3, 3, 1, 1], [3, 3, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 0, 4], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]], device=device, dtype=torch.float32
        )
        assert_close(conf_mat, conf_mat_real)

    def test_four_classes_2d_one_class_no_predicted(self, device, dtype):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[3, 3, 2, 2], [3, 3, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 4, 4], [0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]], device=device, dtype=torch.float32
        )
        assert_close(conf_mat, conf_mat_real)


class TestPsnr:
    def test_metric(self, device, dtype):
        sample = torch.ones(1, device=device, dtype=dtype)
        expected = torch.tensor(20.0, device=device, dtype=dtype)
        actual = kornia.metrics.psnr(sample, 1.2 * sample, 2.0)
        assert_close(actual, expected)


class TestMeanAveragePrecision:
    def test_smoke(self, device, dtype):
        boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        scores = torch.tensor([0.7], device=device, dtype=dtype)

        gt_boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        gt_labels = torch.tensor([1], device=device, dtype=torch.long)

        mean_ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [gt_boxes], [gt_labels], 2)

        assert_close(mean_ap[0], torch.tensor(1.0, device=device, dtype=dtype))
        assert_close(mean_ap[1][1], 1.0)

    def test_raise(self, device, dtype):
        boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        scores = torch.tensor([0.7], device=device, dtype=dtype)

        gt_boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        gt_labels = torch.tensor([1], device=device, dtype=torch.long)

        with pytest.raises(AssertionError):
            _ = kornia.metrics.mean_average_precision(boxes[0], [labels], [scores], [gt_boxes], [gt_labels], 2)

import pytest

import torch
import kornia as kornia
import kornia.testing as utils

from torch.testing import assert_allclose


class TestMeanIoU:
    def test_two_classes_perfect(self):
        batch_size = 1
        num_classes = 2
        actual = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]])
        predicted = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]])

        mean_iou = kornia.utils.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor(
            [[1.0, 1.0]], dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        assert_allclose(mean_iou, mean_iou_real)

    def test_two_classes_perfect_batch2(self):
        batch_size = 2
        num_classes = 2
        actual = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]]).repeat(batch_size, 1)
        predicted = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]]).repeat(batch_size, 1)

        mean_iou = kornia.utils.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor(
            [[1.0, 1.0]], dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        assert_allclose(mean_iou, mean_iou_real)

    def test_two_classes(self):
        batch_size = 1
        num_classes = 2
        actual = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]])
        predicted = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 1]])

        mean_iou = kornia.utils.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou = kornia.utils.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor(
            [[0.75, 0.80]], dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        assert_allclose(mean_iou, mean_iou_real)

    def test_four_classes_2d_perfect(self):
        batch_size = 1
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1],
              [0, 0, 1, 1],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])
        predicted = torch.tensor(
            [[[0, 0, 1, 1],
              [0, 0, 1, 1],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])

        mean_iou = kornia.utils.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        assert_allclose(mean_iou, mean_iou_real)

    def test_four_classes_one_missing(self):
        batch_size = 1
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 0, 0],
              [0, 0, 0, 0],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])
        predicted = torch.tensor(
            [[[3, 3, 2, 2],
              [3, 3, 2, 2],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])

        mean_iou = kornia.utils.metrics.mean_iou(predicted, actual, num_classes)
        mean_iou_real = torch.tensor(
            [[0.0, 1.0, 0.5, 0.5]], dtype=torch.float32)
        assert mean_iou.shape == (batch_size, num_classes)
        assert_allclose(mean_iou, mean_iou_real)


class TestConfusionMatrix:
    def test_two_classes(self):
        num_classes = 2
        actual = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]])
        predicted = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 1]])

        conf_mat = kornia.utils.metrics.confusion_matrix(
            predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[3, 1],
              [0, 4]]], dtype=torch.float32)
        assert_allclose(conf_mat, conf_mat_real)

    def test_two_classes_batch2(self):
        batch_size = 2
        num_classes = 2
        actual = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]]).repeat(batch_size, 1)
        predicted = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 1]]).repeat(batch_size, 1)

        conf_mat = kornia.utils.metrics.confusion_matrix(
            predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[3, 1],
              [0, 4]]], dtype=torch.float32)
        assert_allclose(conf_mat, conf_mat_real)

    def test_three_classes(self):
        num_classes = 3
        actual = torch.tensor(
            [[2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 1, 2, 1, 0]])
        predicted = torch.tensor(
            [[2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2]])

        conf_mat = kornia.utils.metrics.confusion_matrix(
            predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[4, 1, 2],
              [3, 0, 2],
              [1, 2, 1]]], dtype=torch.float32)
        assert_allclose(conf_mat, conf_mat_real)

    def test_four_classes_one_missing(self):
        num_classes = 4
        actual = torch.tensor(
            [[3, 3, 1, 1, 2, 1, 1, 3, 2, 2, 1, 1, 2, 3, 2, 1]])
        predicted = torch.tensor(
            [[3, 2, 1, 1, 1, 1, 1, 2, 1, 3, 3, 2, 1, 1, 3, 3]])

        conf_mat = kornia.utils.metrics.confusion_matrix(
            predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 0, 0],
              [0, 4, 1, 2],
              [0, 3, 0, 2],
              [0, 1, 2, 1]]], dtype=torch.float32)
        assert_allclose(conf_mat, conf_mat_real)

    def test_three_classes_normalized(self):
        num_classes = 3
        normalized = True
        actual = torch.tensor(
            [[2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 1, 2, 1, 0]])
        predicted = torch.tensor(
            [[2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2]])

        conf_mat = kornia.utils.metrics.confusion_matrix(
            predicted, actual, num_classes, normalized)

        conf_mat_real = torch.tensor(
            [[[0.5000, 0.3333, 0.4000],
              [0.3750, 0.0000, 0.4000],
              [0.1250, 0.6667, 0.2000]]], dtype=torch.float32)
        assert_allclose(conf_mat, conf_mat_real)

    def test_four_classes_2d_perfect(self):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1],
              [0, 0, 1, 1],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])
        predicted = torch.tensor(
            [[[0, 0, 1, 1],
              [0, 0, 1, 1],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])

        conf_mat = kornia.utils.metrics.confusion_matrix(
            predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[4, 0, 0, 0],
              [0, 4, 0, 0],
              [0, 0, 4, 0],
              [0, 0, 0, 4]]], dtype=torch.float32)
        assert_allclose(conf_mat, conf_mat_real)

    def test_four_classes_2d_one_class_nonperfect(self):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1],
              [0, 0, 1, 1],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])
        predicted = torch.tensor(
            [[[0, 0, 1, 1],
              [0, 3, 0, 1],
              [2, 2, 1, 3],
              [2, 2, 3, 3]]])

        conf_mat = kornia.utils.metrics.confusion_matrix(
            predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[3, 0, 0, 1],
              [1, 3, 0, 0],
              [0, 0, 4, 0],
              [0, 1, 0, 3]]], dtype=torch.float32)
        assert_allclose(conf_mat, conf_mat_real)

    def test_four_classes_2d_one_class_missing(self):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1],
              [0, 0, 1, 1],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])
        predicted = torch.tensor(
            [[[3, 3, 1, 1],
              [3, 3, 1, 1],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])

        conf_mat = kornia.utils.metrics.confusion_matrix(
            predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 0, 4],
              [0, 4, 0, 0],
              [0, 0, 4, 0],
              [0, 0, 0, 4]]], dtype=torch.float32)
        assert_allclose(conf_mat, conf_mat_real)

    def test_four_classes_2d_one_class_no_predicted(self):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 0, 0],
              [0, 0, 0, 0],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])
        predicted = torch.tensor(
            [[[3, 3, 2, 2],
              [3, 3, 2, 2],
              [2, 2, 3, 3],
              [2, 2, 3, 3]]])

        conf_mat = kornia.utils.metrics.confusion_matrix(
            predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 4, 4],
              [0, 0, 0, 0],
              [0, 0, 4, 0],
              [0, 0, 0, 4]]], dtype=torch.float32)
        assert_allclose(conf_mat, conf_mat_real)

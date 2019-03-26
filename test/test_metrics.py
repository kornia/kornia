import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils
from common import device_type


class TestMeanIoU:
    def test_two_classes_perfect(self):
        num_classes = 2
        actual = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]])
        predicted = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]])

        mean_iou = tgm.metrics.mean_iou(predicted, actual, num_classes)
        assert mean_iou.shape == (1, num_classes)
        assert pytest.approx(mean_iou[..., 0].item(), 1.00)
        assert pytest.approx(mean_iou[..., 1].item(), 1.00)

    def test_two_classes(self):
        num_classes = 2
        actual = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]])
        predicted = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 1]])

        mean_iou = tgm.metrics.mean_iou(predicted, actual, num_classes)
        assert mean_iou.shape == (1, num_classes)
        assert pytest.approx(mean_iou[..., 0].item(), 0.75)
        assert pytest.approx(mean_iou[..., 1].item(), 0.80)

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

        mean_iou = tgm.metrics.mean_iou(predicted, actual, num_classes)
        assert mean_iou.shape == (1, num_classes)
        assert pytest.approx(mean_iou[..., 0].item(), 1.00)
        assert pytest.approx(mean_iou[..., 1].item(), 1.00)
        assert pytest.approx(mean_iou[..., 2].item(), 1.00)
        assert pytest.approx(mean_iou[..., 3].item(), 1.00)

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

        mean_iou = tgm.metrics.mean_iou(predicted, actual, num_classes)
        assert mean_iou.shape == (1, num_classes)
        assert pytest.approx(mean_iou[..., 0].item(), 0.00)
        assert pytest.approx(mean_iou[..., 1].item(), 0.00)
        assert pytest.approx(mean_iou[..., 2].item(), 0.50)
        assert pytest.approx(mean_iou[..., 3].item(), 0.50)


class TestConfusionMatrix:
    def test_two_classes(self):
        num_classes = 2
        actual = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]])
        predicted = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 1]])

        conf_mat = tgm.metrics.confusion_matrix(predicted, actual, num_classes)
        assert conf_mat[..., 0, 0].item() == 3
        assert conf_mat[..., 0, 1].item() == 1
        assert conf_mat[..., 1, 0].item() == 0
        assert conf_mat[..., 1, 1].item() == 4

    def test_three_classes(self):
        num_classes = 3
        actual = torch.tensor(
            [[2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 1, 2, 1, 0]])
        predicted = torch.tensor(
            [[2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2]])

        conf_mat = tgm.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[4, 1, 2],
              [3, 0, 2],
              [1, 2, 1]]])
        assert torch.equal(conf_mat, conf_mat_real)

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

        conf_mat = tgm.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[4, 0, 0, 0],
              [0, 4, 0, 0],
              [0, 0, 4, 0],
              [0, 0, 0, 4]]])
        assert torch.equal(conf_mat, conf_mat_real)

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

        conf_mat = tgm.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[3, 0, 0, 1],
              [1, 3, 0, 0],
              [0, 0, 4, 0],
              [0, 1, 0, 3]]])
        assert torch.equal(conf_mat, conf_mat_real)

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

        conf_mat = tgm.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 0, 4],
              [0, 4, 0, 0],
              [0, 0, 4, 0],
              [0, 0, 0, 4]]])
        assert torch.equal(conf_mat, conf_mat_real)

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

        conf_mat = tgm.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 4, 4],
              [0, 0, 0, 0],
              [0, 0, 4, 0],
              [0, 0, 0, 4]]])
        assert torch.equal(conf_mat, conf_mat_real)

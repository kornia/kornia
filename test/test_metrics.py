import pytest
import torch

import kornia
from kornia.testing import BaseTester, assert_close, tensor_to_gradcheck_var


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

class TestEpe:
    def test_metric(self, device, dtype):
        sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
        expected = torch.tensor(0.32, device=device, dtype=dtype)
        actual = kornia.metrics.epe(sample, 1.4 * sample)
        assert_close(actual, expected)

    def test_smoke(self, device, dtype):
        input = torch.rand(3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(3, 3, 2, device=device, dtype=dtype)

        criterion = kornia.metrics.EPE()
        assert criterion(input, target) is not None


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


class TestSSIM3d(BaseTester):
    @pytest.mark.parametrize(
        "shape,padding,window_size,max_value",
        [
            ((1, 1, 3, 3, 3), 'same', 5, 1.0),
            ((1, 1, 3, 3, 3), 'same', 3, 2.0),
            ((1, 1, 3, 3, 3), 'same', 3, 0.5),
            ((1, 1, 3, 3, 3), 'valid', 3, 1.0),
            ((2, 4, 3, 3, 3), 'same', 3, 1.0),
        ],
    )
    def test_smoke(self, shape, padding, window_size, max_value, device, dtype):
        img_a = (torch.ones(shape, device=device, dtype=dtype) * max_value).clamp(0.0, max_value)
        img_b = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.metrics.ssim3d(img_a, img_b, window_size, max_value, padding=padding)
        expected = torch.ones_like(actual, device=device, dtype=dtype)

        self.assert_close(actual, expected * 0.0001)

        actual = kornia.metrics.ssim3d(img_a, img_a, window_size, max_value, padding=padding)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "shape,padding,window_size,expected",
        [
            ((1, 1, 2, 2, 3), 'same', 3, (1, 1, 2, 2, 3)),
            ((1, 1, 3, 3, 3), 'same', 5, (1, 1, 3, 3, 3)),
            ((1, 1, 3, 3, 3), 'valid', 3, (1, 1, 1, 1, 1)),
            ((2, 4, 3, 3, 3), 'same', 3, (2, 4, 3, 3, 3)),
        ],
    )
    def test_cardinality(self, shape, padding, window_size, expected, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.metrics.ssim3d(img, img, window_size, padding=padding)

        assert actual.shape == expected

    def test_exception(self, device, dtype):
        img = torch.rand(1, 1, 3, 3, 3, device=device, dtype=dtype)

        # Check if both are tensors
        with pytest.raises(TypeError) as errinfo:
            kornia.metrics.ssim3d(1.0, img, 3)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            kornia.metrics.ssim3d(img, 1.0, 3)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        # Check both shapes
        img_wrong_shape = torch.rand(3, 3, device=device, dtype=dtype)
        with pytest.raises(TypeError) as errinfo:
            kornia.metrics.ssim3d(img, img_wrong_shape, 3)
        assert 'shape must be [' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            kornia.metrics.ssim3d(img_wrong_shape, img, 3)
        assert 'shape must be [' in str(errinfo)

        # Check if same shape
        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(Exception) as errinfo:
            kornia.metrics.ssim3d(img, img_b, 3)
        assert 'img1 and img2 shapes must be the same. Got:' in str(errinfo)

    def test_unit(self, device, dtype):
        img_a = torch.tensor(
            [
                [
                    [
                        [[0.7, 1.0, 0.5], [1.0, 0.3, 1.0], [0.2, 1.0, 0.1]],
                        [[0.2, 1.0, 0.1], [1.0, 0.3, 1.0], [0.7, 1.0, 0.5]],
                        [[1.0, 0.3, 1.0], [0.7, 1.0, 0.5], [0.2, 1.0, 0.1]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        img_b = torch.ones(1, 1, 3, 3, 3, device=device, dtype=dtype) * 0.5

        actual = kornia.metrics.ssim3d(img_a, img_b, 3, padding='same')

        expected = torch.tensor(
            [
                [
                    [
                        [[0.0093, 0.0080, 0.0075], [0.0075, 0.0068, 0.0063], [0.0067, 0.0060, 0.0056]],
                        [[0.0077, 0.0070, 0.0065], [0.0077, 0.0069, 0.0064], [0.0075, 0.0066, 0.0062]],
                        [[0.0075, 0.0069, 0.0064], [0.0078, 0.0070, 0.0065], [0.0077, 0.0067, 0.0064]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "shape,padding,window_size,max_value",
        [
            ((1, 1, 3, 3, 3), 'same', 5, 1.0),
            ((1, 1, 3, 3, 3), 'same', 3, 2.0),
            ((1, 1, 3, 3, 3), 'same', 3, 0.5),
            ((1, 1, 3, 3, 3), 'valid', 3, 1.0),
        ],
    )
    def test_module(self, shape, padding, window_size, max_value, device, dtype):
        img_a = torch.rand(shape, device=device, dtype=dtype).clamp(0.0, max_value)
        img_b = torch.rand(shape, device=device, dtype=dtype).clamp(0.0, max_value)

        ops = kornia.metrics.ssim3d
        mod = kornia.metrics.SSIM3D(window_size, max_value, padding=padding)

        ops_out = ops(img_a, img_b, window_size, max_value, padding=padding)
        mod_out = mod(img_a, img_b)

        self.assert_close(ops_out, mod_out)

    def test_gradcheck(self, device):
        img = torch.rand(1, 1, 3, 3, 3, device=device)

        op = kornia.metrics.ssim3d
        img = tensor_to_gradcheck_var(img)

        assert self.gradcheck(op, (img, img, 3), nondet_tol=1e-8)

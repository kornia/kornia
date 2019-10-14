import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import math
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import numpy as np


class TestFocalLoss:
    def _test_smoke_none(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        assert kornia.losses.focal_loss(
            logits, labels, alpha=0.5, gamma=2.0, reduction="none"
        ).shape == (2, 3, 2)

    def _test_smoke_sum(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        assert (
            kornia.losses.focal_loss(
                logits, labels, alpha=0.5, gamma=2.0, reduction="sum"
            ).shape == ()
        )

    def _test_smoke_mean(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        assert (
            kornia.losses.focal_loss(
                logits, labels, alpha=0.5, gamma=2.0, reduction="mean"
            ).shape == ()
        )

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        num_classes = 3
        alpha, gamma = 0.5, 2.0  # for focal loss
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.focal_loss,
            (logits, labels, alpha, gamma),
            raise_exception=True,
        )

    def test_run_all(self):
        self._test_smoke_none()
        self._test_smoke_sum()
        self._test_smoke_mean()
        self._test_gradcheck()


class TestTverskyLoss:
    def _test_smoke(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        loss = criterion(logits, labels)

    def _test_all_zeros(self):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, dtype=torch.int64)

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        loss = criterion(logits, labels)
        assert pytest.approx(loss.item(), 0.0)

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        num_classes = 3
        alpha, beta = 0.5, 0.5  # for tversky loss
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.tversky_loss,
            (logits, labels, alpha, beta),
            raise_exception=True,
        )

    def test_run_all(self):
        self._test_smoke()
        self._test_all_zeros()
        self._test_gradcheck()


class TestDiceLoss:
    def _test_smoke(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        criterion = kornia.losses.DiceLoss()
        loss = criterion(logits, labels)

    def _test_all_zeros(self):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, dtype=torch.int64)

        criterion = kornia.losses.DiceLoss()
        loss = criterion(logits, labels)
        assert pytest.approx(loss.item(), 0.0)

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.dice_loss, (logits, labels), raise_exception=True
        )

    def test_run_all(self):
        self._test_smoke()
        self._test_all_zeros()
        self._test_gradcheck()


class TestDepthSmoothnessLoss:
    def _test_smoke(self):
        image = self.image.clone()
        depth = self.depth.clone()

        criterion = kornia.losses.InverseDepthSmoothnessLoss()
        loss = criterion(depth, image)

    # TODO: implement me
    def _test_1(self):
        pass

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        image = self.image.clone()
        depth = self.depth.clone()
        depth = utils.tensor_to_gradcheck_var(depth)  # to var
        image = utils.tensor_to_gradcheck_var(image)  # to var
        assert gradcheck(
            kornia.losses.inverse_depth_smoothness_loss,
            (depth, image),
            raise_exception=True,
        )

    @pytest.mark.parametrize("batch_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_run_all(self, batch_shape, device_type):
        self.image = torch.rand(batch_shape).to(torch.device(device_type))
        self.depth = torch.rand(batch_shape).to(torch.device(device_type))

        self._test_smoke()
        self._test_gradcheck()


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("reduction_type", ["none", "mean", "sum"])
@pytest.mark.parametrize("batch_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
def test_ssim(batch_shape, device_type, window_size, reduction_type):
    # input data
    device = torch.device(device_type)
    img1 = torch.rand(batch_shape).to(device)
    img2 = torch.rand(batch_shape).to(device)

    ssim = kornia.losses.SSIM(window_size, reduction_type)
    ssim_loss_val = ssim(img1, img2)

    if reduction_type == "none":
        assert ssim_loss_val.shape == batch_shape
    else:
        assert ssim_loss_val.dim() == 0

    assert pytest.approx(ssim(img1, img1).sum().item(), 0.0)
    assert pytest.approx(ssim(img2, img2).sum().item(), 0.0)

    # functional
    assert_allclose(
        ssim_loss_val,
        kornia.losses.ssim(img1, img2, window_size, reduction_type),
    )

    # evaluate function gradient
    img1 = utils.tensor_to_gradcheck_var(img1)  # to var
    img2 = utils.tensor_to_gradcheck_var(img2, requires_grad=False)  # to var
    assert gradcheck(ssim, (img1, img2), raise_exception=True)


class TestDivergenceLoss:
    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.346574),
        (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.346574),
    ])
    def test_js_div_loss_2d(self, input, target, expected):
        actual = kornia.losses.js_div_loss_2d(input, target).item()
        assert actual == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
        (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
    ])
    def test_kl_div_loss_2d(self, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input, target).item()
        assert actual == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125),
         torch.full((1, 1, 2, 4), 0.125),
         torch.full((1, 1), 0.0)),
        (torch.full((1, 7, 2, 4), 0.125),
         torch.full((1, 7, 2, 4), 0.125),
         torch.full((1, 7), 0.0)),
        (torch.full((1, 7, 2, 4), 0.125),
         torch.zeros((1, 7, 2, 4)),
         torch.full((1, 7), 0.0)),
        (torch.zeros((1, 7, 2, 4)),
         torch.full((1, 7, 2, 4), 0.125),
         torch.full((1, 7), math.inf)),
    ])
    def test_kl_div_loss_2d_without_reduction(self, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input, target, reduction='none')
        assert_allclose(actual, expected)


class TestTotalVariation:
    # Total variation of constant vectors is 0
    @pytest.mark.parametrize('input,expected', [
        (torch.ones(3, 4, 5), torch.zeros(())),
        (2 * torch.ones(2, 3, 4, 5), torch.zeros(2)),
    ])
    def test_tv_on_constant(self, input, expected):
        actual = kornia.losses.total_variation(input)
        assert_allclose(actual, expected)

    # Total variation for 3D tensors
    @pytest.mark.parametrize('input', [
        torch.rand(3, 4, 5),
        torch.rand(1, 2, 3),
    ])
    def test_tv_on_3d(self, input):
        actual = kornia.losses.total_variation(input)
        input_np = input.numpy()
        pixel_dif1 = input_np[:, 1:, :] - input_np[:, :-1, :]
        pixel_dif2 = input_np[:, :, 1:] - input_np[:, :, :-1]
        expected = torch.tensor(np.sum(np.abs(pixel_dif1)) + np.sum(np.abs(pixel_dif2)))
        assert_allclose(actual, expected)

    # Total variation for 3D tensors
    @pytest.mark.parametrize('input', [
        torch.rand(2, 3, 4, 5),
        torch.rand(3, 1, 2, 3),
    ])
    def test_tv_on_4d(self, input):
        actual = kornia.losses.total_variation(input)
        input_np = input.numpy()
        pixel_dif1 = input_np[:, :, 1:, :] - input_np[:, :, :-1, :]
        pixel_dif2 = input_np[:, :, :, 1:] - input_np[:, :, :, :-1]
        expected = torch.tensor(np.sum(np.abs(pixel_dif1), axis=(1, 2, 3)) + np.sum(np.abs(pixel_dif2), axis=(1, 2, 3)))
        assert_allclose(actual, expected)

    # Expect ValueError to be raised when tensors of rank != 3 or 4 are passed
    @pytest.mark.parametrize('input', [
        torch.rand(2, 3, 4, 5, 3),
        torch.rand(3, 1),
    ])
    def test_tv_on_invalid_dims(self, input):
        with pytest.raises(ValueError) as ex_info:
            kornia.losses.total_variation(input)

    # Expect TypeError to be raised when non-torch tensors are passed
    @pytest.mark.parametrize('input', [
        1,
        [1, 2],
    ])
    def test_tv_on_invalid_types(self, input):
        with pytest.raises(TypeError) as ex_info:
            kornia.losses.total_variation(input)

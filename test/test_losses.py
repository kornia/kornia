import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils
from common import TEST_DEVICES


class TestDiceLoss:
    def test_make_one_hot(self):
        num_classes = 4
        labels = torch.zeros(2, 2, 1, dtype=torch.int64)
        labels[0, 0, 0] = 0
        labels[0, 1, 0] = 1
        labels[1, 0, 0] = 2
        labels[1, 1, 0] = 3

        # convert labels to one hot tensor
        one_hot = tgm.losses.DiceLoss.make_one_hot(labels, num_classes)

        assert pytest.approx(one_hot[0, labels[0, 0, 0], 0, 0].item(), 1.0)
        assert pytest.approx(one_hot[0, labels[0, 1, 0], 1, 0].item(), 1.0)
        assert pytest.approx(one_hot[1, labels[1, 0, 0], 0, 0].item(), 1.0)
        assert pytest.approx(one_hot[1, labels[1, 1, 0], 1, 0].item(), 1.0)

    def _test_smoke(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        criterion = tgm.losses.DiceLoss(reduction='none')
        loss = criterion(logits, labels)

    # TODO: implement me
    def _test_all_zeros(self):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, dtype=torch.int64)

        criterion = tgm.losses.DiceLoss(reduction='mean')
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
        assert gradcheck(tgm.losses.dice_loss,
                         (logits, labels,), raise_exception=True)

    def test_run_all(self):
        self._test_smoke()
        self._test_all_zeros()
        self._test_gradcheck()


class TestDepthSmoothnessLoss:
    def _test_smoke(self):
        image = self.image.clone()
        depth = self.depth.clone()

        criterion = tgm.losses.DepthSmoothnessLoss()
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
        assert gradcheck(tgm.losses.depth_smoothness_loss,
                         (depth, image,), raise_exception=True)

    @pytest.mark.parametrize("device_type", TEST_DEVICES)
    @pytest.mark.parametrize("batch_shape",
                             [(1, 1, 10, 16), (2, 4, 8, 15), ])
    def test_run_all(self, batch_shape, device_type):
        self.image = torch.rand(batch_shape).to(torch.device(device_type))
        self.depth = torch.rand(batch_shape).to(torch.device(device_type))

        self._test_smoke()
        self._test_gradcheck()


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("reduction_type", ['none', 'mean', 'sum'])
@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_shape",
                         [(1, 1, 10, 16), (2, 4, 8, 15), ])
def test_ssim(batch_shape, device_type, window_size, reduction_type):
    # input data
    device = torch.device(device_type)
    img1 = torch.rand(batch_shape).to(device)
    img2 = torch.rand(batch_shape).to(device)

    ssim = tgm.losses.SSIM(window_size, reduction_type)
    ssim_loss_val = ssim(img1, img2)

    if reduction_type == 'none':
        assert ssim_loss_val.shape == batch_shape
    else:
        assert ssim_loss_val.dim() == 0

    assert pytest.approx(ssim(img1, img1).sum().item(), 0.0)
    assert pytest.approx(ssim(img2, img2).sum().item(), 0.0)

    # functional
    assert utils.check_equal_torch(
        ssim_loss_val, tgm.losses.ssim(
            img1, img2, window_size, reduction_type))

    # evaluate function gradient
    img1 = utils.tensor_to_gradcheck_var(img1)  # to var
    img2 = utils.tensor_to_gradcheck_var(img2, requires_grad=False)  # to var
    assert gradcheck(ssim, (img1, img2,), raise_exception=True)

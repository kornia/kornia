import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils
from common import TEST_DEVICES


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

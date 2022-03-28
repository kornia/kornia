import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestSSIMLoss:
    def test_ssim_equal_none(self, device, dtype):
        # input data
        img1 = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)

        ssim1 = kornia.losses.ssim_loss(img1, img1, window_size=5, reduction="none")
        ssim2 = kornia.losses.ssim_loss(img2, img2, window_size=5, reduction="none")

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(ssim1, torch.zeros_like(img1), rtol=tol_val, atol=tol_val)
        assert_close(ssim2, torch.zeros_like(img2), rtol=tol_val, atol=tol_val)

    @pytest.mark.parametrize("window_size", [5, 11])
    @pytest.mark.parametrize("reduction_type", ["mean", "sum"])
    @pytest.mark.parametrize("batch_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_ssim(self, device, dtype, batch_shape, window_size, reduction_type):
        if device.type == 'xla':
            pytest.skip("test highly unstable with tpu")

        # input data
        img = torch.rand(batch_shape, device=device, dtype=dtype)

        loss = kornia.losses.ssim_loss(img, img, window_size, reduction=reduction_type)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(loss.item(), 0.0, rtol=tol_val, atol=tol_val)

    def test_jit(self, device, dtype):
        img1 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        img2 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        args = (img1, img2, 5, 1.0, 1e-6, 'mean')

        op = kornia.losses.ssim_loss
        op_script = torch.jit.script(op)

        assert_close(op(*args), op_script(*args))

    def test_module(self, device, dtype):
        img1 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        img2 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        args = (img1, img2, 5, 1.0, 1e-12, 'mean')

        op = kornia.losses.ssim_loss
        op_module = kornia.losses.SSIMLoss(*args[2:])

        assert_close(op(*args), op_module(*args[:2]))

    def test_gradcheck(self, device, dtype):
        # input data
        window_size = 3
        img1 = torch.rand(1, 1, 5, 4, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 5, 4, device=device, dtype=dtype)

        # evaluate function gradient
        img1 = utils.tensor_to_gradcheck_var(img1)  # to var
        img2 = utils.tensor_to_gradcheck_var(img2)  # to var

        # TODO: review method since it needs `nondet_tol` in cuda sometimes.
        assert gradcheck(kornia.losses.ssim_loss, (img1, img2, window_size), raise_exception=True, nondet_tol=1e-8)


class TestMS_SSIMLoss:
    def test_msssim_equal_none(self, device, dtype):
        # input data
        img1 = torch.rand(1, 3, 10, 16, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 10, 16, device=device, dtype=dtype)

        msssim = kornia.losses.MS_SSIMLoss().to(device, dtype)
        msssim1 = msssim(img1, img1)
        msssim2 = msssim(img2, img2)

        assert_close(msssim1.item(), 0.0)
        assert_close(msssim2.item(), 0.0)

    # TODO: implement for single channel image
    @pytest.mark.parametrize("reduction_type", ["mean", "sum"])
    @pytest.mark.parametrize("batch_shape", [(2, 1, 2, 3), (1, 3, 10, 16)])
    def test_msssim(self, device, dtype, batch_shape, reduction_type):
        img = torch.rand(batch_shape, device=device, dtype=dtype)

        msssiml1 = kornia.losses.MS_SSIMLoss(reduction=reduction_type).to(device, dtype)
        loss = msssiml1(img, img)

        assert_close(loss.item(), 0.0)

    def test_gradcheck(self, device):
        # input data
        dtype = torch.float64
        img1 = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)

        # evaluate function gradient
        img1 = utils.tensor_to_gradcheck_var(img1)  # to var
        img2 = utils.tensor_to_gradcheck_var(img2)  # to var

        loss = kornia.losses.MS_SSIMLoss().to(device, dtype)

        assert gradcheck(loss, (img1, img2), raise_exception=True, nondet_tol=1e-8)

    def test_jit(self, device, dtype):
        img1 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)

        args = (img1, img2)

        op = kornia.losses.MS_SSIMLoss().to(device, dtype)
        op_script = torch.jit.script(op)

        assert_close(op(*args), op_script(*args))

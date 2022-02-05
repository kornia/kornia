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


class TestMSSSIML1Loss:
    def test_msssiml1_equal_none(self, device, dtype):
        # input data
        img1 = torch.rand(1, 3, 10, 16, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 10, 16, device=device, dtype=dtype)

        msssiml1 = kornia.losses.MS_SSIM_L1Loss()
        msssiml11 = msssiml1(img1, img1)
        msssiml12 = msssiml1(img2, img2)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(msssiml11.item(), 0.0, rtol=tol_val, atol=tol_val)
        assert_close(msssiml12.item(), 0.0, rtol=tol_val, atol=tol_val)

    def test_msssiml1(self, device, dtype, batch_shape):
        if device.type == 'xla':
            pytest.skip("test highly unstable with tpu")

        # input data
        img = torch.rand(batch_shape, device=device, dtype=dtype)

        msssiml1 = kornia.losses.MS_SSIM_L1Loss()
        loss = msssiml1(img, img)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(loss.item(), 0.0, rtol=tol_val, atol=tol_val)

    @pytest.mark.parametrize(
        "msl1", [kornia.losses.MS_SSIM_L1Loss]
    )
    def test_gradcheck(self, msl1, device, dtype):
        # input data
        img1 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)

        # evaluate function gradient
        img1 = utils.tensor_to_gradcheck_var(img1)  # to var
        img2 = utils.tensor_to_gradcheck_var(img2)  # to var

        loss = msl1()

        assert gradcheck(loss, (img1, img2), raise_exception=True, nondet_tol=1e-8)

    def test_jit(self, device, dtype):
        img1 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)

        args = (img1, img2)

        op = kornia.losses.MS_SSIM_L1Loss
        op_script = torch.jit.script(op)

        assert_close(op(*args), op_script(*args))

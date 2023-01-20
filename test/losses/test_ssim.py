import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import BaseTester, assert_close


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
        assert gradcheck(
            kornia.losses.ssim_loss, (img1, img2, window_size), raise_exception=True, nondet_tol=1e-8, fast_mode=True
        )


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

        assert gradcheck(loss, (img1, img2), raise_exception=True, nondet_tol=1e-8, fast_mode=True)

    def test_jit(self, device, dtype):
        img1 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)

        args = (img1, img2)

        op = kornia.losses.MS_SSIMLoss().to(device, dtype)
        op_script = torch.jit.script(op)

        assert_close(op(*args), op_script(*args))


class TestSSIM3DLoss(BaseTester):
    def test_smoke(self, device, dtype):
        # input data
        img1 = torch.rand(1, 1, 2, 4, 3, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 2, 4, 4, device=device, dtype=dtype)

        ssim1 = kornia.losses.ssim3d_loss(img1, img1, window_size=3, reduction="none")
        ssim2 = kornia.losses.ssim3d_loss(img2, img2, window_size=3, reduction="none")

        self.assert_close(ssim1, torch.zeros_like(img1))
        self.assert_close(ssim2, torch.zeros_like(img2))

    @pytest.mark.parametrize("window_size", [5, 11])
    @pytest.mark.parametrize("reduction_type", ["mean", "sum"])
    @pytest.mark.parametrize("shape", [(1, 1, 2, 16, 16), (2, 4, 2, 15, 20)])
    def test_ssim(self, device, dtype, shape, window_size, reduction_type):
        if device.type == 'xla':
            pytest.skip("test highly unstable with tpu")

        # Sanity test
        img = torch.rand(shape, device=device, dtype=dtype)
        actual = kornia.losses.ssim3d_loss(img, img, window_size, reduction=reduction_type)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(actual, expected)

        # Check loss computation
        img1 = torch.ones(shape, device=device, dtype=dtype)
        img2 = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.losses.ssim3d_loss(img1, img2, window_size, reduction=reduction_type)

        if reduction_type == 'mean':
            expected = torch.tensor(0.9999, device=device, dtype=dtype)
        elif reduction_type == 'sum':
            expected = (torch.ones_like(img1, device=device, dtype=dtype) * 0.9999).sum()

        self.assert_close(actual, expected)

    def test_jit(self, device, dtype):
        img1 = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)
        img2 = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)

        args = (img1, img2, 5, 1.0, 1e-6, 'mean')

        op = kornia.losses.ssim3d_loss
        op_script = torch.jit.script(op)

        self.assert_close(op(*args), op_script(*args))

    def test_module(self, device, dtype):
        img1 = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)
        img2 = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)

        args = (img1, img2, 5, 1.0, 1e-12, 'mean')

        op = kornia.losses.ssim3d_loss
        op_module = kornia.losses.SSIM3DLoss(*args[2:])

        self.assert_close(op(*args), op_module(*args[:2]))

    def test_gradcheck(self, device):
        # input data
        img = torch.rand(1, 1, 5, 4, 3, device=device)

        # evaluate function gradient
        img = utils.tensor_to_gradcheck_var(img)  # to var

        # TODO: review method since it needs `nondet_tol` in cuda sometimes.
        assert gradcheck(
            kornia.losses.ssim3d_loss, (img, img, 3), raise_exception=True, nondet_tol=1e-8, fast_mode=True
        )

    @pytest.mark.parametrize("shape", [(1, 2, 3, 5, 5), (2, 4, 2, 5, 5)])
    def test_cardinality(self, shape, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.SSIM3DLoss(5, reduction='none')(img, img)
        assert actual.shape == shape

        actual = kornia.losses.SSIM3DLoss(5)(img, img)
        assert actual.shape == ()

    @pytest.mark.skip('loss have no exception case')
    def test_exception(self):
        pass

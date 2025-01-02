# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

import kornia

from testing.base import BaseTester


class TestSSIMLoss(BaseTester):
    def test_ssim_equal_none(self, device, dtype):
        # input data
        img1 = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)

        ssim1 = kornia.losses.ssim_loss(img1, img1, window_size=5, reduction="none")
        ssim2 = kornia.losses.ssim_loss(img2, img2, window_size=5, reduction="none")

        self.assert_close(ssim1, torch.zeros_like(img1), low_tolerance=True)  # rtol=tol_val, atol=tol_val)
        self.assert_close(ssim2, torch.zeros_like(img2), low_tolerance=True)  # rtol=tol_val, atol=tol_val)

    @pytest.mark.parametrize("window_size", [5, 11])
    @pytest.mark.parametrize("reduction_type", ["mean", "sum", "none"])
    @pytest.mark.parametrize("batch_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_ssim(self, device, dtype, batch_shape, window_size, reduction_type):
        if device.type == "xla":
            pytest.skip("test highly unstable with tpu")

        # input data
        img = torch.rand(batch_shape, device=device, dtype=dtype)

        loss = kornia.losses.ssim_loss(img, img, window_size, reduction=reduction_type)

        if reduction_type == "none":
            expected = torch.zeros_like(img)
        else:
            expected = torch.tensor(0.0, device=device, dtype=dtype)

        self.assert_close(loss, expected)

    def test_module(self, device, dtype):
        img1 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        img2 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        args = (img1, img2, 5, 1.0, 1e-12, "mean")

        op = kornia.losses.ssim_loss
        op_module = kornia.losses.SSIMLoss(*args[2:])

        self.assert_close(op(*args), op_module(*args[:2]))

    def test_gradcheck(self, device):
        # input data
        window_size = 3
        img1 = torch.rand(1, 1, 5, 4, device=device, dtype=torch.float64)
        img2 = torch.rand(1, 1, 5, 4, device=device, dtype=torch.float64)

        # evaluate function gradient

        # TODO: review method since it needs `nondet_tol` in cuda sometimes.
        self.gradcheck(kornia.losses.ssim_loss, (img1, img2, window_size), nondet_tol=1e-8)


class TestMS_SSIMLoss(BaseTester):
    def test_msssim_equal_none(self, device, dtype):
        # input data
        img1 = torch.rand(1, 3, 10, 16, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 10, 16, device=device, dtype=dtype)

        msssim = kornia.losses.MS_SSIMLoss().to(device, dtype)
        msssim1 = msssim(img1, img1)
        msssim2 = msssim(img2, img2)

        self.assert_close(msssim1.item(), 0.0)
        self.assert_close(msssim2.item(), 0.0)

    def test_exception(self):
        criterion = kornia.losses.MS_SSIMLoss()

        with pytest.raises(TypeError) as errinfo:
            criterion(1, 2)
        assert "Input type is not a torch.Tensor. Got" in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            criterion(torch.rand(1), 2)
        assert "Output type is not a torch.Tensor. Got" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1), torch.rand(1, 2))
        assert "Input shapes should be same. Got" in str(errinfo)

    # TODO: implement for single channel image
    @pytest.mark.parametrize("reduction_type", ["mean", "sum", "none"])
    @pytest.mark.parametrize("batch_shape", [(2, 1, 2, 3), (1, 3, 10, 16)])
    def test_msssim(self, device, dtype, batch_shape, reduction_type):
        img = torch.rand(batch_shape, device=device, dtype=dtype)

        msssiml1 = kornia.losses.MS_SSIMLoss(reduction=reduction_type).to(device, dtype)
        loss = msssiml1(img, img)

        self.assert_close(loss.sum().item(), 0.0)

    def test_gradcheck(self, device):
        # input data
        dtype = torch.float64
        img1 = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)

        # evaluate function gradient
        loss = kornia.losses.MS_SSIMLoss().to(device, dtype)

        self.gradcheck(loss, (img1, img2), nondet_tol=1e-8)

    def test_jit(self, device, dtype):
        img1 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 10, 10, device=device, dtype=dtype)

        args = (img1, img2)

        op = kornia.losses.MS_SSIMLoss().to(device, dtype)
        op_script = torch.jit.script(op)

        self.assert_close(op(*args), op_script(*args))


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
    @pytest.mark.parametrize("reduction_type", ["mean", "sum", "none"])
    @pytest.mark.parametrize("shape", [(1, 1, 2, 16, 16), (2, 4, 2, 15, 20)])
    def test_ssim(self, device, dtype, shape, window_size, reduction_type):
        if device.type == "xla":
            pytest.skip("test highly unstable with tpu")

        # Sanity test
        img = torch.rand(shape, device=device, dtype=dtype)
        actual = kornia.losses.ssim3d_loss(img, img, window_size, reduction=reduction_type)
        if reduction_type == "none":
            expected = torch.zeros_like(img)
        else:
            expected = torch.tensor(0.0, device=device, dtype=dtype)

        self.assert_close(actual, expected)

        # Check loss computation
        img1 = torch.ones(shape, device=device, dtype=dtype)
        img2 = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.losses.ssim3d_loss(img1, img2, window_size, reduction=reduction_type)

        if reduction_type == "mean":
            expected = torch.tensor(0.9999, device=device, dtype=dtype)
        elif reduction_type == "sum":
            expected = (torch.ones_like(img1, device=device, dtype=dtype) * 0.9999).sum()
        elif reduction_type == "none":
            expected = torch.ones_like(img1, device=device, dtype=dtype) * 0.9999

        self.assert_close(actual, expected)

    def test_module(self, device, dtype):
        img1 = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)
        img2 = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)

        args = (img1, img2, 5, 1.0, 1e-12, "mean")

        op = kornia.losses.ssim3d_loss
        op_module = kornia.losses.SSIM3DLoss(*args[2:])

        self.assert_close(op(*args), op_module(*args[:2]))

    def test_gradcheck(self, device):
        # input data
        img = torch.rand(1, 1, 5, 4, 3, device=device)

        # evaluate function gradient

        # TODO: review method since it needs `nondet_tol` in cuda sometimes.
        self.gradcheck(kornia.losses.ssim3d_loss, (img, img, 3), nondet_tol=1e-8)

    @pytest.mark.parametrize("shape", [(1, 2, 3, 5, 5), (2, 4, 2, 5, 5)])
    def test_cardinality(self, shape, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.SSIM3DLoss(5, reduction="none")(img, img)
        assert actual.shape == shape

        actual = kornia.losses.SSIM3DLoss(5)(img, img)
        assert actual.shape == ()

    @pytest.mark.skip("loss have no exception case")
    def test_exception(self):
        pass

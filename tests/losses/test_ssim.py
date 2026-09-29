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

    def test_gradcheck(self, device, dtype):
        # input data
        window_size = 3
        img1 = torch.rand(1, 1, 5, 4, device=device, dtype=torch.float64)
        img2 = torch.rand(1, 1, 5, 4, device=device, dtype=torch.float64)

        # evaluate function gradient

        # TODO: review method since it needs `nondet_tol` in cuda sometimes.
        self.gradcheck(kornia.losses.ssim_loss, (img1, img2, window_size), nondet_tol=1e-8)


# The variances are differences of Gaussian-filtered moments, which cancel badly in half precision.
_MS_SSIM_TOL = {torch.float16: 1e-2, torch.bfloat16: 5e-2}


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

    @pytest.mark.parametrize("reduction_type", ["mean", "sum", "none"])
    @pytest.mark.parametrize("batch_shape", [(2, 1, 2, 3), (1, 3, 10, 16)])
    def test_msssim(self, device, dtype, batch_shape, reduction_type):
        img = torch.rand(batch_shape, device=device, dtype=dtype)

        msssiml1 = kornia.losses.MS_SSIMLoss(reduction=reduction_type).to(device, dtype)
        loss = msssiml1(img, img)

        self.assert_close(loss.sum().item(), 0.0)

    @pytest.mark.parametrize("channels", [1, 2, 3, 4])
    def test_cardinality(self, device, dtype, channels):
        img1 = torch.rand(2, channels, 10, 16, device=device, dtype=dtype)
        img2 = torch.rand(2, channels, 10, 16, device=device, dtype=dtype)

        loss = kornia.losses.MS_SSIMLoss(reduction="none").to(device, dtype)

        assert loss(img1, img2).shape == (2, 10, 16)

    def test_channel_order(self, device, dtype):
        img1 = torch.rand(2, 3, 12, 12, device=device, dtype=dtype)
        img2 = (img1 + 0.2 * torch.rand_like(img1)).clamp(0, 1)
        bgr = [2, 1, 0]

        loss = kornia.losses.MS_SSIMLoss(alpha=1.0, compensation=1.0, reduction="none").to(device, dtype)

        tol = _MS_SSIM_TOL.get(dtype, 1e-4)
        self.assert_close(loss(img1[:, bgr], img2[:, bgr]), loss(img1, img2), atol=tol, rtol=tol)

    def test_brightness_shift_in_any_channel(self, device, dtype):
        # Three copies of one plane: brightening any single copy must cost the same, and the luminance term must see it.
        img = torch.rand(1, 1, 12, 12, device=device, dtype=dtype).repeat(1, 3, 1, 1) * 0.5
        loss = kornia.losses.MS_SSIMLoss(alpha=1.0, compensation=1.0, reduction="none").to(device, dtype)

        shifted = []
        for channel in range(3):
            img_shifted = img.clone()
            img_shifted[:, channel] += 0.3
            shifted.append(loss(img, img_shifted))

        tol = _MS_SSIM_TOL.get(dtype, 1e-4)
        self.assert_close(shifted[1], shifted[0], atol=tol, rtol=tol)
        self.assert_close(shifted[2], shifted[0], atol=tol, rtol=tol)
        assert shifted[0].mean() > 0.05

    @pytest.mark.parametrize("channels", [1, 3])
    def test_reference(self, device, dtype, channels):
        # Snippet used to generate expected (requires numpy only): the per-pixel, Gaussian-window MS-SSIM of
        # Zhao et al. (2017) for every channel, multiplied over the channels.
        # import numpy as np
        # def gauss(s, k):
        #     g = np.exp(-((np.arange(k) - k // 2) ** 2) / (2 * s * s)); g /= g.sum(); return np.outer(g, g)
        # def filt(a, w):  # zero-padded "same" correlation
        #     k = w.shape[0]; a = np.pad(a, k // 2); h, v = a.shape[0] - k + 1, a.shape[1] - k + 1
        #     return np.array([[(a[i : i + k, j : j + k] * w).sum() for j in range(v)] for i in range(h)])
        # def ms_ssim_loss(x, y, sigmas=(0.5, 1.0, 2.0), c1=0.01**2, c2=0.03**2):  # x, y: (C, H, W)
        #     k = int(4 * sigmas[-1] + 1); prod = np.ones(x.shape[1:])
        #     for xc, yc in zip(x, y):
        #         for i, s in enumerate(sigmas):
        #             w = gauss(s, k); mx, my = filt(xc, w), filt(yc, w)
        #             vx, vy, cxy = filt(xc * xc, w) - mx * mx, filt(yc * yc, w) - my * my, filt(xc * yc, w) - mx * my
        #             prod *= (2 * cxy + c2) / (vx + vy + c2)
        #             if i == len(sigmas) - 1:
        #                 prod *= (2 * mx * my + c1) / (mx * mx + my * my + c1)
        #     return 1 - prod
        # x = (np.arange(60).reshape(3, 4, 5) * np.array([1, 3, 7]).reshape(3, 1, 1) % 11) / 10
        # expected = ms_ssim_loss(x[:channels], x[:channels] ** 2)
        img1 = torch.arange(60, dtype=torch.float64).reshape(1, 3, 4, 5)
        img1 = (img1 * torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64).view(1, 3, 1, 1) % 11) / 10
        img1 = img1[:, :channels].to(device, dtype)
        img2 = img1**2
        expected = {
            1: [
                [0.37865300, 0.31442640, 0.26742503, 0.25432102, 0.27785429],
                [0.27414722, 0.26719022, 0.20145665, 0.17667983, 0.18669405],
                [0.20643671, 0.23914177, 0.22795677, 0.17534234, 0.19646084],
                [0.27640491, 0.33963748, 0.28366937, 0.22100261, 0.19366287],
            ],
            3: [
                [0.66712644, 0.55020783, 0.54474428, 0.59134973, 0.69012946],
                [0.51310637, 0.49969304, 0.48765724, 0.47662429, 0.53276831],
                [0.46020250, 0.49761160, 0.46484452, 0.46814626, 0.51647284],
                [0.53283039, 0.56160072, 0.51846598, 0.54783399, 0.52460798],
            ],
        }[channels]
        expected = torch.tensor([expected], device=device, dtype=dtype)

        loss = kornia.losses.MS_SSIMLoss(sigmas=(0.5, 1.0, 2.0), alpha=1.0, compensation=1.0, reduction="none")
        loss = loss.to(device, dtype)

        tol = _MS_SSIM_TOL.get(dtype, 1e-4)
        self.assert_close(loss(img1, img2), expected, atol=tol, rtol=tol)

    def test_load_legacy_state_dict(self, device, dtype):
        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.criterion = kornia.losses.MS_SSIMLoss()

        img1 = torch.rand(1, 3, 10, 16, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 10, 16, device=device, dtype=dtype)
        model = Wrapper().to(device, dtype)
        expected = model.criterion(img1, img2)

        assert "criterion._g_masks" not in model.state_dict()
        # Older releases persisted three scale-major copies of each of the five default masks.
        legacy_masks = model.criterion._g_masks.repeat_interleave(3, dim=0)
        model.load_state_dict({"criterion._g_masks": legacy_masks}, strict=True)

        self.assert_close(model.criterion(img1, img2), expected)

    def test_gradcheck(self, device, dtype):
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

    def test_gradcheck(self, device, dtype):
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

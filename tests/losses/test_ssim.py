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
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_high_pixel_values_use_float32_moments(self, device, dtype):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")

        generator = torch.Generator().manual_seed(0)
        img1 = (180 + torch.rand(1, 3, 64, 64, generator=generator) * 75).round()
        img2 = (img1 + torch.randn(1, 3, 64, 64, generator=generator) * 20).clamp(0, 255).round()

        reference = kornia.losses.MS_SSIMLoss(data_range=255.0).double()(img1.double(), img2.double())
        criterion = kornia.losses.MS_SSIMLoss(data_range=255.0).to(device, dtype)
        loss = criterion(img1.to(device, dtype), img2.to(device, dtype))

        assert loss.dtype == dtype
        assert torch.isfinite(loss)
        self.assert_close(loss.float().cpu(), reference.float(), atol=0.02, rtol=0.02)

    def test_int16_images_are_not_rounded_through_float16(self, device):
        generator = torch.Generator().manual_seed(0)
        img1 = (2048 + torch.rand(1, 3, 64, 64, generator=generator) * 8000).round().to(torch.int16)
        img2 = (
            (img1.float() + torch.randn(1, 3, 64, 64, generator=generator) * 200)
            .clamp(0, 32767)
            .round()
            .to(torch.int16)
        )

        criterion = kornia.losses.MS_SSIMLoss(data_range=32767.0).to(device, torch.float16)
        reference = kornia.losses.MS_SSIMLoss(data_range=32767.0).to(device, torch.float32)
        loss = criterion(img1.to(device), img2.to(device))

        assert loss.dtype == torch.float16
        assert torch.isfinite(loss)
        self.assert_close(
            loss.float(),
            reference(img1.to(device, torch.float32), img2.to(device, torch.float32)),
            atol=0.01,
            rtol=0.01,
        )

    def test_autocast_keeps_msssim_convolutions_in_float32(self):
        generator = torch.Generator().manual_seed(0)
        img1 = (180 + torch.rand(1, 3, 64, 64, generator=generator) * 75).round()
        img2 = (img1 + torch.randn(1, 3, 64, 64, generator=generator) * 20).clamp(0, 255).round()
        criterion = kornia.losses.MS_SSIMLoss(data_range=255.0)

        expected = criterion(img1, img2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = criterion(img1, img2)

        self.assert_close(actual, expected)

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
        # Zhao et al. (2017) for every channel, averaged over the channels as in their reference implementation
        # (NVlabs/PL4NN, ``MSSSIML1`` in src/loss.py).
        # import numpy as np
        # def gauss(s, k):
        #     g = np.exp(-((np.arange(k) - k // 2) ** 2) / (2 * s * s)); g /= g.sum(); return np.outer(g, g)
        # def filt(a, w):  # zero-padded "same" correlation
        #     k = w.shape[0]; a = np.pad(a, k // 2); h, v = a.shape[0] - k + 1, a.shape[1] - k + 1
        #     return np.array([[(a[i : i + k, j : j + k] * w).sum() for j in range(v)] for i in range(h)])
        # def ms_ssim_loss(x, y, sigmas=(0.5, 1.0, 2.0), c1=0.01**2, c2=0.03**2):  # x, y: (C, H, W)
        #     k = int(4 * sigmas[-1] + 1); per_channel = []
        #     for xc, yc in zip(x, y):
        #         ms = np.ones(x.shape[1:])
        #         for i, s in enumerate(sigmas):
        #             w = gauss(s, k); mx, my = filt(xc, w), filt(yc, w)
        #             vx, vy, cxy = filt(xc * xc, w) - mx * mx, filt(yc * yc, w) - my * my, filt(xc * yc, w) - mx * my
        #             ms *= (2 * cxy + c2) / (vx + vy + c2)
        #             if i == len(sigmas) - 1:
        #                 ms *= (2 * mx * my + c1) / (mx * mx + my * my + c1)
        #         per_channel.append(ms)
        #     return 1 - np.mean(per_channel, axis=0)
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
                [0.30440426, 0.23147867, 0.22960489, 0.25790863, 0.31691969],
                [0.21193412, 0.20488162, 0.19946196, 0.19401123, 0.22219212],
                [0.18556347, 0.20465673, 0.18753592, 0.18937725, 0.21497141],
                [0.22266887, 0.23662971, 0.21462762, 0.23004896, 0.21928525],
            ],
        }[channels]
        expected = torch.tensor([expected], device=device, dtype=dtype)

        loss = kornia.losses.MS_SSIMLoss(sigmas=(0.5, 1.0, 2.0), alpha=1.0, compensation=1.0, reduction="none")
        loss = loss.to(device, dtype)

        tol = _MS_SSIM_TOL.get(dtype, 1e-4)
        self.assert_close(loss(img1, img2), expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("channels", [1, 3])
    def test_reference_gaussian_l1(self, device, dtype, channels):
        # With alpha=0 the loss is the l1 map filtered by the coarsest Gaussian and averaged over the channels.
        # Generated with ``gauss`` and ``filt`` from the snippet in ``test_reference``:
        # w = gauss(2.0, 9)
        # expected = np.mean([filt(np.abs(c - c**2), w) for c in x[:channels]], axis=0)
        img1 = torch.arange(60, dtype=torch.float64).reshape(1, 3, 4, 5)
        img1 = (img1 * torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64).view(1, 3, 1, 1) % 11) / 10
        img1 = img1[:, :channels].to(device, dtype)
        img2 = img1**2
        expected = {
            1: [
                [0.04841360, 0.06432695, 0.07315782, 0.07117095, 0.05873239],
                [0.05998628, 0.07880676, 0.08869681, 0.08553489, 0.07010512],
                [0.06226608, 0.08101300, 0.09034629, 0.08643698, 0.07039810],
                [0.05424314, 0.06998091, 0.07739282, 0.07348809, 0.05948054],
            ],
            3: [
                [0.05252443, 0.06618850, 0.07189183, 0.06754884, 0.05445936],
                [0.06308681, 0.07960642, 0.08649714, 0.08128649, 0.06559485],
                [0.06331894, 0.08003596, 0.08705040, 0.08188926, 0.06620053],
                [0.05316348, 0.06731125, 0.07329963, 0.06904758, 0.05593615],
            ],
        }[channels]
        expected = torch.tensor([expected], device=device, dtype=dtype)

        loss = kornia.losses.MS_SSIMLoss(sigmas=(0.5, 1.0, 2.0), alpha=0.0, compensation=1.0, reduction="none")
        loss = loss.to(device, dtype)

        tol = {torch.float16: 1e-3, torch.bfloat16: 1e-2}.get(dtype, 1e-4)
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

    @pytest.mark.parametrize("sigmas", [(0.5, 1.3), (1.0, 3.3), (0.5, 1.0, 2.0)])
    def test_none_reduction_keeps_input_shape_5126(self, device, dtype, sigmas):
        # int(4 * sigma + 1) is even for sigma 1.3 (6) and 3.3 (14): the window sat half a pixel off centre and the
        # reduction="none" map came out (N, H - 1, W - 1) instead of (N, H, W). With an odd window the map keeps the
        # input shape and is mirror-symmetric: flipping both inputs flips the map, with no one-pixel shift.
        img1 = torch.rand(1, 3, 16, 20, device=device, dtype=dtype)
        img2 = torch.rand(1, 3, 16, 20, device=device, dtype=dtype)
        loss = kornia.losses.MS_SSIMLoss(sigmas=sigmas, reduction="none").to(device, dtype)
        assert loss._g_masks.shape[-1] % 2 == 1
        out = loss(img1, img2)
        assert out.shape == (1, 16, 20)
        self.assert_close(out, loss(img1.flip(-1), img2.flip(-1)).flip(-1))

    @pytest.mark.parametrize(
        "input_dtype, data_range", [(torch.uint8, 255.0), (torch.int16, 255.0), (torch.int64, 255.0), (torch.bool, 1.0)]
    )
    def test_integer_images_are_computed_in_the_mask_dtype_5351(self, device, dtype, input_dtype, data_range):
        # An integer or bool image is converted to the dtype of the Gaussian masks before filtering, so it gives the
        # loss of the same values held in that dtype, for two integer images and for one integer and one float image.
        g = torch.Generator().manual_seed(0)
        values1 = torch.randint(0, 256, (1, 3, 12, 16), generator=g)
        values2 = (values1 + torch.randn(1, 3, 12, 16, generator=g) * 40).round().clamp(0, 255).long()
        if input_dtype is torch.bool:
            values1, values2 = values1 > 127, values2 > 127
        img1 = values1.to(device, input_dtype)
        img2 = values2.to(device, input_dtype)
        criterion = kornia.losses.MS_SSIMLoss(data_range=data_range).to(device, dtype)

        expected = criterion(img1.to(dtype), img2.to(dtype))
        assert expected.dtype == dtype
        assert expected.item() > 0

        loss = criterion(img1, img2)
        assert loss.dtype == dtype
        self.assert_close(loss, expected)
        self.assert_close(criterion(img1, img2.to(dtype)), expected)
        self.assert_close(criterion(img1.to(dtype), img2), expected)

    def test_integer_images_follow_the_module_dtype_5351(self, device, dtype):
        # The masks follow the module dtype, so a float64 module computes uint8 images in float64.
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        img1 = torch.randint(0, 256, (1, 1, 12, 16), device=device, dtype=torch.uint8)
        img2 = torch.randint(0, 256, (1, 1, 12, 16), device=device, dtype=torch.uint8)
        criterion = kornia.losses.MS_SSIMLoss(data_range=255.0).to(device, torch.float64)

        loss = criterion(img1, img2)
        assert loss.dtype == torch.float64
        self.assert_close(loss, criterion(img1.double(), img2.double()))

    def test_complex_images_are_not_cast_5351(self, device, dtype):
        # Only integer and bool images are converted: casting a complex image to the real mask dtype would drop its
        # imaginary part, so it still raises.
        g = torch.Generator().manual_seed(0)
        img1 = torch.rand(1, 1, 12, 16, generator=g).to(device, dtype)
        img2 = torch.rand(1, 1, 12, 16, generator=g).to(device, dtype)
        criterion = kornia.losses.MS_SSIMLoss().to(device, dtype)
        with pytest.raises(RuntimeError):
            criterion(img1.to(torch.complex64), img2)
        with pytest.raises(RuntimeError):
            criterion(img1, img2.to(torch.complex64))

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

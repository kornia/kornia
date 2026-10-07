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

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kornia.core.exceptions import BaseError
from kornia.metrics.ssim import SSIM, ssim

from testing.base import BaseTester


class TestSsim(BaseTester):
    @pytest.mark.parametrize("image_dtype", [torch.uint8, torch.int16])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_integer_images(self, device, image_dtype, padding):
        # Integer Gaussian weights must not truncate to zero (#5297).
        black = torch.zeros((1, 1, 5, 5), device=device, dtype=image_dtype)
        white = torch.full_like(black, 255)
        actual = ssim(black, white, 3, max_val=255.0, padding=padding)
        assert actual.dtype == torch.float32
        self.assert_close(actual, torch.full_like(actual, 0.0001), atol=1e-6, rtol=1e-4)

    @pytest.mark.parametrize("image_dtype", [torch.uint8, torch.int16, torch.int32, torch.int64])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_integer_images_match_the_float32_result(self, device, image_dtype, padding):
        # A textured pair exercises the variances and the covariance, which a pair of constant images cancels.
        # The 12-bit values of the wider types are not all exact in float16.
        max_val = 255.0 if image_dtype == torch.uint8 else 4095.0
        generator = torch.Generator().manual_seed(0)
        img1 = (torch.rand((1, 2, 16, 16), generator=generator) * max_val).round()
        img2 = (img1 + torch.randn((1, 2, 16, 16), generator=generator) * max_val / 6).clamp(0, max_val).round()
        img1, img2 = img1.to(device), img2.to(device)
        expected = ssim(img1, img2, 5, max_val=max_val, padding=padding)
        actual = ssim(img1.to(image_dtype), img2.to(image_dtype), 5, max_val=max_val, padding=padding)
        assert actual.dtype == torch.float32
        assert expected.mean() < 0.95
        self.assert_close(actual, expected, rtol=0, atol=0)

    def test_dynamo_dynamic_range(self, device, dtype, torch_optimizer):
        img = torch.full((1, 1) + (8,) * 2, 128.0, device=device, dtype=dtype)
        optimized = torch_optimizer(ssim)
        actual = optimized(img, img, 3, max_val=255.0)
        assert actual.dtype == dtype
        self.assert_close(actual, torch.ones_like(actual))

    def test_mixed_input_dtypes(self, device, dtype):
        first = torch.full((1, 1) + (8,) * 2, 128.0, device=device, dtype=dtype)
        second = torch.full_like(first, 64.0, dtype=torch.float32)
        actual = ssim(first, second, 3, max_val=255.0)
        output_dtype = torch.promote_types(dtype, torch.float32)
        assert actual.dtype == output_dtype
        c1 = (0.01 * 255.0) ** 2
        expected = torch.full_like(actual, (2 * 128 * 64 + c1) / (128**2 + 64**2 + c1))
        self.assert_close(actual, expected)

    def test_pixel_range_gradients(self, device, dtype):
        if device.type == "mps":
            pytest.skip("Float64 reference is unsupported on MPS")
        img1 = torch.arange(5**2, device=device, dtype=dtype).reshape((1, 1) + (5,) * 2) % 17 * 15
        img2 = (img1.flip(-1) * 0.8).detach().requires_grad_()
        img1 = img1.detach().requires_grad_()
        ref1, ref2 = img1.detach().double().requires_grad_(), img2.detach().double().requires_grad_()
        actual = ssim(img1, img2, 3, max_val=255.0)
        reference = ssim(ref1, ref2, 3, max_val=255.0)
        self.assert_close(actual, reference.to(dtype))
        actual_grads = torch.autograd.grad(actual.sum(), (img1, img2))
        reference_grads = torch.autograd.grad(reference.sum(), (ref1, ref2))
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            assert torch.isfinite(actual_grad).all()
            self.assert_close(actual_grad, reference_grad.to(dtype), rtol=2e-2, atol=1e-5)

    def test_autocast_dynamic_range(self, device, dtype):
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Autocast regression covers CPU and CUDA backends")
        img1 = torch.arange(8**2, device=device, dtype=dtype).reshape((1, 1) + (8,) * 2) % 17 * 15
        img2 = img1.flip(-1) * 0.8
        expected = ssim(img1.double(), img2.double(), 3, max_val=255.0).to(dtype)
        autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            actual = ssim(img1, img2, 3, max_val=255.0)
        assert actual.dtype == dtype
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("values", [(128.0, 128.0, 255.0), (128.0, 64.0, 255.0), (0.0, 0.0, 0.5)])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_constant_image_dynamic_range(self, device, dtype, values, padding):
        first, second, max_val = values
        shape = (1, 2) + (8,) * 2
        img1 = torch.full(shape, first, device=device, dtype=dtype)
        img2 = torch.full(shape, second, device=device, dtype=dtype)
        actual = ssim(img1, img2, 3, max_val=max_val, padding=padding)
        # Constant images have zero local variance/covariance in the SSIM formula.
        c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
        expected_value = (2 * first * second + c1) * c2 / ((first**2 + second**2 + c1) * c2 + 1e-12)
        expected = torch.full_like(actual, expected_value)
        assert actual.dtype == dtype
        self.assert_close(actual, expected)

    def test_same_image_returns_ones(self, device, dtype):
        img = torch.rand(1, 3, 16, 16, device=device, dtype=dtype)
        out = ssim(img, img, window_size=5)
        assert out.shape == img.shape
        assert (out > 0.99).all()

    def test_padding_valid(self, device, dtype):
        img1 = torch.rand(1, 1, 16, 16, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 16, 16, device=device, dtype=dtype)
        out_same = ssim(img1, img2, window_size=5, padding="same")
        out_valid = ssim(img1, img2, window_size=5, padding="valid")
        # valid crops the border — output is smaller than 'same'
        assert out_valid.shape[2] < out_same.shape[2]
        assert out_valid.shape[3] < out_same.shape[3]

    @pytest.mark.parametrize("padding", ["VALID", "Valid"])
    def test_padding_case_insensitive(self, device, dtype, padding):
        # Case variants select their branch as the filters do since #5156 (#5537).
        img1 = torch.rand(1, 1, 13, 17, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 13, 17, device=device, dtype=dtype)
        expected = ssim(img1, img2, window_size=5, padding="valid")
        actual = ssim(img1, img2, window_size=5, padding=padding)
        assert actual.shape == (1, 1, 9, 13)
        self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(SSIM(5, padding=padding)(img1, img2), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("padding", ["full", "bogus"])
    def test_exception_invalid_padding(self, device, dtype, padding):
        # Any other value raises instead of silently returning the 'same' map (#5537).
        img = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="Invalid padding mode"):
            ssim(img, img, window_size=3, padding=padding)
        with pytest.raises(BaseError, match="Invalid padding mode"):
            SSIM(3, padding=padding)(img, img)

    def test_exception_non_tensor_img1(self, device, dtype):
        img2 = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        with pytest.raises(TypeError, match=r"Input img1 type is not a torch\.Tensor"):
            ssim([1, 2, 3], img2, window_size=3)

    def test_exception_non_tensor_img2(self, device, dtype):
        img1 = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        with pytest.raises(TypeError, match=r"Input img2 type is not a torch\.Tensor"):
            ssim(img1, [1, 2, 3], window_size=3)

    def test_exception_non_float_max_val(self, device, dtype):
        img = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="Input max_val type is not a float"):
            ssim(img, img, window_size=3, max_val=1)

    def test_exception_wrong_ndim_img1(self, device, dtype):
        img1 = torch.rand(1, 8, 8, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Invalid img1 shape"):
            ssim(img1, img2, window_size=3)

    def test_exception_wrong_ndim_img2(self, device, dtype):
        img1 = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        img2 = torch.rand(1, 8, 8, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Invalid img2 shape"):
            ssim(img1, img2, window_size=3)

    def test_exception_shape_mismatch(self, device, dtype):
        img1 = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 8, 16, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="img1 and img2 shapes must be the same"):
            ssim(img1, img2, window_size=3)

    def test_ssim_module(self, device, dtype):
        img1 = torch.rand(2, 3, 16, 16, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 16, 16, device=device, dtype=dtype)
        module = SSIM(window_size=5)
        out_module = module(img1, img2)
        out_fn = ssim(img1, img2, window_size=5)
        assert out_module.shape == out_fn.shape


class TestConventionsSSIM(BaseTester):
    """Pins for the map shape, the border, the window, ``max_val`` and ``padding`` validation of :func:`ssim`."""

    @staticmethod
    def _pair(device, dtype):
        # a two-sample, three-channel batch with H != W; y is a noisy copy of x
        g = torch.Generator().manual_seed(0)
        x = torch.rand(2, 3, 13, 17, generator=g)
        y = (x + 0.3 * torch.randn(2, 3, 13, 17, generator=g)).clamp(0, 1)
        return x.to(device, dtype), y.to(device, dtype)

    def test_convention_ssim_map_shape_and_valid_crop(self, device, dtype):
        # 'same' returns the per-pixel map (B, C, H, W). 'valid' crops (window_size - 1) / 2 pixels per side, giving
        # (B, C, H - window_size + 1, W - window_size + 1), and equals the 'same' map with that border cut off.
        x, y = self._pair(device, dtype)
        same = ssim(x, y, 5)
        valid = ssim(x, y, 5, padding="valid")
        assert same.shape == (2, 3, 13, 17)
        assert valid.shape == (2, 3, 9, 13)
        self.assert_close(valid, same[..., 2:-2, 2:-2])
        # Relabelling check: transposing both images transposes the map.
        self.assert_close(ssim(x.transpose(-2, -1), y.transpose(-2, -1), 5), same.transpose(-2, -1))
        # window_size is the odd tap count of the Gaussian window
        with pytest.raises((ValueError, BaseError)):
            ssim(x, y, 4)

    def test_convention_ssim_same_border_is_reflect(self, device, dtype):
        # 'same' pads with torch's 'reflect', which does not repeat the edge pixel: the 'same' map equals the 'valid'
        # map of the reflect-padded images. Padding by 'replicate' instead changes the border.
        x, y = self._pair(device, dtype)
        work = torch.promote_types(dtype, torch.float32)  # ssim computes half-precision images in float32

        def padded(mode):
            xp = F.pad(x.to(work), (2, 2, 2, 2), mode=mode)
            yp = F.pad(y.to(work), (2, 2, 2, 2), mode=mode)
            return ssim(xp, yp, 5, padding="valid").to(dtype)

        same = ssim(x, y, 5)
        self.assert_close(same, padded("reflect"))
        assert (same - padded("replicate")).abs().max() > 0.1

    def test_convention_ssim_matches_scikit_image_gaussian_ssim(self, device, dtype):
        # The window is a sampled Gaussian with sigma = 1.5 whatever window_size is; window_size=11 with 'valid' and
        # eps=0 matches scikit-image's Gaussian SSIM (Wang et al. 2004), averaged over the channels.
        # Snippet used to generate expected (scikit-image 0.26.0, numpy 2.0.0):
        #   x = (np.arange(2 * 13 * 17).reshape(2, 13, 17) * 7 % 11) / 16
        #   structural_similarity(x, x**2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        #                         data_range=1.0, channel_axis=0)
        x = (torch.arange(2 * 13 * 17, dtype=torch.float32).reshape(1, 2, 13, 17) * 7 % 11) / 16
        x = x.to(device, dtype)
        actual = ssim(x, x**2, 11, max_val=1.0, eps=0.0, padding="valid").mean()
        self.assert_close(actual, torch.tensor(0.649510247445418, device=device, dtype=dtype))
        # The same sigma at another size: window_size=7 equals SSIM with a sampled 7-tap Gaussian of sigma 1.5 and
        # reflect padding, written out with the default constants C1 = 0.01**2, C2 = 0.03**2 and eps.
        x, y = self._pair(device, dtype)
        work = torch.promote_types(dtype, torch.float32)  # ssim computes half-precision images in float32
        k = torch.exp(-((torch.arange(7, dtype=torch.float64) - 3) ** 2) / (2 * 1.5**2))
        k = (k / k.sum()).to(device, work)
        kernel = (k[:, None] * k[None, :]).expand(3, 1, 7, 7)

        def blur(t):
            return F.conv2d(F.pad(t, (3, 3, 3, 3), mode="reflect"), kernel, groups=3)

        xw, yw = x.to(work), y.to(work)
        mx, my = blur(xw), blur(yw)
        sx, sy, sxy = blur(xw * xw) - mx**2, blur(yw * yw) - my**2, blur(xw * yw) - mx * my
        expected = (2 * mx * my + 1e-4) * (2 * sxy + 9e-4) / ((mx**2 + my**2 + 1e-4) * (sx + sy + 9e-4) + 1e-12)
        self.assert_close(ssim(x, y, 7), expected.to(dtype))

    @pytest.mark.parametrize(
        "max_val,expected",
        [(1.0, 0.9999888890123443), (0.1, 0.9), (0.01, 0.0008991907283444898)],
    )
    def test_convention_ssim_epsilon_biases_identical_black_images(self, device, dtype, max_val, expected):
        # Identical black images have SSIM 1 in the reference implementations. Kornia's denominator epsilon changes
        # the score to C1*C2/(C1*C2 + eps), even at max_val=1; at small ranges the difference is substantial.
        # The expected values use C1=(0.01*max_val)**2, C2=(0.03*max_val)**2, eps=1e-12 in float64 arithmetic.
        x = torch.zeros(1, 1, 13, 17, device=device, dtype=dtype)
        score = ssim(x, x, 11, max_val=max_val, padding="valid").mean()
        self.assert_close(score, x.new_tensor(expected))
        reference = ssim(x, x, 11, max_val=max_val, eps=0.0, padding="valid").mean()
        self.assert_close(reference, x.new_tensor(1.0), rtol=0.0, atol=0.0)
        if max_val < 1.0:
            assert (reference - score) > 0.05

    def test_convention_ssim_max_val_is_the_data_range(self, device, dtype):
        # max_val is the data range L in C1 = (0.01 L)**2 and C2 = (0.03 L)**2: images and max_val scaled together give
        # the same map, and the default max_val=1.0 on the scaled images gives another one (the range is not inferred).
        x, y = self._pair(device, dtype)
        expected = ssim(x, y, 5)
        self.assert_close(ssim(256 * x, 256 * y, 5, max_val=256.0), expected)
        assert (ssim(256 * x, 256 * y, 5) - expected).abs().max() > 0.01

    def test_convention_ssim_is_per_sample_and_per_channel(self, device, dtype):
        # Every (sample, channel) plane is scored on its own, and the map is symmetric in (img1, img2).
        x, y = self._pair(device, dtype)
        base = ssim(x, y, 5)
        y_moved = y.clone()
        y_moved[1, 1] = 1.0 - y_moved[1, 1]
        moved = ssim(x, y_moved, 5)
        self.assert_close(moved[0], base[0])
        self.assert_close(moved[1, 0], base[1, 0])
        self.assert_close(moved[1, 2], base[1, 2])
        assert (moved[1, 1] - base[1, 1]).abs().max() > 0.1
        self.assert_close(ssim(y, x, 5), base)

    def test_convention_ssim_padding_is_case_insensitive_and_validated_5537(self, device, dtype):
        """'VALID' selects the valid map, while an unknown padding mode raises (#5537)."""
        x, y = self._pair(device, dtype)
        valid = ssim(x, y, 5, padding="valid")
        self.assert_close(ssim(x, y, 5, padding="VALID"), valid, rtol=0.0, atol=0.0)
        with pytest.raises(BaseError, match="Invalid padding mode"):
            ssim(x, y, 5, padding="bogus")

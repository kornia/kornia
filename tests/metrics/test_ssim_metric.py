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

    def test_mixed_float32_float64_pair_is_computed_in_float64(self, device, dtype):
        # A float32/float64 pair is filtered in float64, the result dtype, with a float64 window, whichever argument
        # is the float32 one, so it equals the all-float64 result to roundoff (#5574).
        if dtype != torch.float64:
            pytest.skip("the pair needs a float64 argument")
        generator = torch.Generator().manual_seed(0)
        img1 = (0.9 + 0.1 * torch.rand(1, 1, 16, 16, generator=generator, dtype=torch.float64)).float().to(device)
        img2 = (0.9 + 0.1 * torch.rand(1, 1, 16, 16, generator=generator, dtype=torch.float64)).to(device)
        expected = ssim(img1.double(), img2, 5)
        self.assert_close(ssim(img1, img2, 5), expected, rtol=1e-12, atol=1e-12)
        self.assert_close(ssim(img2, img1, 5), expected, rtol=1e-12, atol=1e-12)

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

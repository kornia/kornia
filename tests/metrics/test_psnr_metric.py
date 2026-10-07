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

import math

import pytest
import torch

import kornia
from kornia.core.exceptions import BaseError

from testing.base import BaseTester


class TestPsnr(BaseTester):
    def test_metric(self, device, dtype):
        sample = torch.ones(1, device=device, dtype=dtype)
        expected = torch.tensor(20.0, device=device, dtype=dtype)
        actual = kornia.metrics.psnr(sample, 1.2 * sample, 2.0)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("image_dtype", [torch.uint8, torch.int64, torch.bool])
    def test_integer_images_match_the_float32_result(self, device, image_dtype):
        # Integer images are computed in float32, as ssim computes them (#5536).
        generator = torch.Generator().manual_seed(0)
        image = (torch.rand(2, 3, 9, 13, generator=generator) * 255).to(device)
        target = (torch.rand(2, 3, 9, 13, generator=generator) * 255).to(device)
        if image_dtype == torch.bool:
            # threshold: a plain cast to bool makes both images all True, and identical images give inf
            image, target, max_val = image > 127, target > 127, 1.0
        else:
            image, target, max_val = image.to(image_dtype), target.to(image_dtype), 255.0
        expected = kornia.metrics.psnr(image.float(), target.float(), max_val)
        actual = kornia.metrics.psnr(image, target, max_val)
        assert actual.dtype == torch.float32
        assert torch.isfinite(expected)
        self.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("partner_dtype", [torch.float16, torch.bfloat16, torch.float64])
    def test_integer_image_with_a_floating_partner(self, device, partner_dtype):
        # The integer image counts as float32 and the pair is computed in the promoted dtype, as in ssim, so a half
        # partner gives float32, not the half dtype. Given two dtypes, mse_loss aborts the process on MPS and its
        # backward raises on torch 2.5.1 (#5536).
        if device.type == "mps" and partner_dtype == torch.float64:
            pytest.skip("MPS has no float64")
        generator = torch.Generator().manual_seed(0)
        image = (torch.rand(2, 3, 9, 13, generator=generator) * 255).to(torch.uint8).to(device)
        partner = (torch.rand(2, 3, 9, 13, generator=generator) * 255).round().to(device, partner_dtype)
        compute_dtype = torch.promote_types(torch.float32, partner_dtype)
        expected = kornia.metrics.psnr(image.to(compute_dtype), partner.to(compute_dtype), 255.0)
        for actual in (kornia.metrics.psnr(image, partner, 255.0), kornia.metrics.psnr(partner, image, 255.0)):
            assert actual.dtype == compute_dtype
            self.assert_close(actual, expected, rtol=0, atol=0)
        partner.requires_grad_()
        kornia.metrics.psnr(image, partner, 255.0).backward()
        assert partner.grad is not None
        assert partner.grad.dtype == partner_dtype

    @pytest.mark.parametrize(
        "dtypes",
        [
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.bfloat16),
            (torch.float32, torch.float64),
        ],
        ids=["f16-f32", "bf16-f32", "f16-bf16", "f32-f64"],
    )
    def test_two_floating_dtypes(self, device, dtypes):
        # Two floating images of different dtypes are compared in their promoted dtype, in either order, and each
        # gradient keeps its image's dtype. Given two dtypes, mse_loss aborts the process on MPS and its backward
        # raises on torch 2.5.1 (#5536).
        if device.type == "mps" and torch.float64 in dtypes:
            pytest.skip("MPS has no float64")
        generator = torch.Generator().manual_seed(0)
        image = torch.rand(2, 3, 9, 13, generator=generator).to(device, dtypes[0])
        target = torch.rand(2, 3, 9, 13, generator=generator).to(device, dtypes[1])
        compute_dtype = torch.promote_types(*dtypes)
        expected = kornia.metrics.psnr(image.to(compute_dtype), target.to(compute_dtype), 1.0)
        for actual in (kornia.metrics.psnr(image, target, 1.0), kornia.metrics.psnr(target, image, 1.0)):
            assert actual.dtype == compute_dtype
            self.assert_close(actual, expected, rtol=0, atol=0)
        image.requires_grad_()
        target.requires_grad_()
        kornia.metrics.psnr(image, target, 1.0).backward()
        assert image.grad.dtype == dtypes[0]
        assert target.grad.dtype == dtypes[1]

    def test_exception_shape_mismatch(self, device, dtype):
        a = torch.ones(4, device=device, dtype=dtype)
        b = torch.ones(8, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="Expected tensors of equal shapes"):
            kornia.metrics.psnr(a, b, max_val=1.0)


class TestConventionsPsnr(BaseTester):
    """Pins for the batch pooling, ``max_val``, identical images and integer images of :func:`psnr`."""

    def test_convention_psnr_pools_mse_over_the_batch(self, device, dtype):
        a = torch.zeros(2, 1, 4, 6, device=device, dtype=dtype)
        b = a.clone()
        b[0] += 0.125
        b[1] += 0.375
        # one MSE over every element of the batch: (0.125**2 + 0.375**2) / 2 -> 10 log10(1 / mse) = 11.0721,
        # not the mean of the per-image PSNRs (18.0618 and 8.5194 -> 13.2906)
        expected = torch.tensor(10.0 * math.log10(2.0 / (0.125**2 + 0.375**2)), device=device, dtype=dtype)
        self.assert_close(kornia.metrics.psnr(a, b, 1.0), expected)

    def test_convention_psnr_max_val_is_the_data_range(self, device, dtype):
        # max_val is MAX_I in 10 log10(MAX_I**2 / MSE), the data range rather than the largest value of the images:
        # images and max_val scaled together keep the value, and max_val=2 gives 10 log10(4 / mse) = 17.0927 on images
        # whose largest value is 0.375. An int max_val is accepted.
        a = torch.zeros(2, 1, 4, 6, device=device, dtype=dtype)
        b = a.clone()
        b[0] += 0.125
        b[1] += 0.375
        expected = torch.tensor(10.0 * math.log10(4.0 * 2.0 / (0.125**2 + 0.375**2)), device=device, dtype=dtype)
        self.assert_close(kornia.metrics.psnr(a, b, 2.0), expected)
        self.assert_close(kornia.metrics.psnr(a, b, 2), expected)
        self.assert_close(kornia.metrics.psnr(8 * a, 8 * b, 16.0), expected)

    def test_convention_psnr_identical_images_give_inf(self, device, dtype):
        # identical images give MSE 0 and +inf; a batch in which only one pair is identical stays finite, because the
        # MSE is pooled: 10 log10(2 / 0.375**2) = 11.5297
        a = torch.zeros(2, 1, 4, 6, device=device, dtype=dtype)
        a[:, :, 1:, 2:] = 0.5
        assert kornia.metrics.psnr(a, a, 1.0).item() == math.inf
        b = a.clone()
        b[1] += 0.375
        expected = torch.tensor(10.0 * math.log10(2.0 / 0.375**2), device=device, dtype=dtype)
        self.assert_close(kornia.metrics.psnr(a, b, 1.0), expected)

    def test_convention_psnr_needs_equal_shapes(self, device, dtype):
        # image and target must have the same shape, with no broadcasting, and the metric is symmetric in them
        a = torch.zeros(2, 1, 4, 6, device=device, dtype=dtype)
        b = torch.full((1, 1, 4, 6), 0.25, device=device, dtype=dtype)
        with pytest.raises((TypeError, ValueError, BaseError)):
            kornia.metrics.psnr(a, b, 1.0)
        b = b.expand(2, 1, 4, 6).clone()
        b[1, :, :, :3] = 0.75
        self.assert_close(kornia.metrics.psnr(b, a, 1.0), kornia.metrics.psnr(a, b, 1.0))

    def test_convention_psnr_computes_integer_images_in_float32_5536(self, device, dtype):
        """psnr computes integer images in float32, matching its float32 inputs (#5536)."""
        # These uint8 differences wrap when left unconverted, so this pair detects either an error or a wrong result.
        a = torch.tensor([[0, 64, 128], [192, 255, 32]], dtype=torch.uint8).view(1, 1, 2, 3).to(device)
        b = torch.tensor([[16, 64, 100], [200, 250, 0]], dtype=torch.uint8).view(1, 1, 2, 3).to(device)
        expected = kornia.metrics.psnr(a.float(), b.float(), 255.0)
        actual = kornia.metrics.psnr(a, b, 255.0)
        assert actual.dtype == torch.float32
        self.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert torch.isfinite(kornia.metrics.psnr(a.to(dtype), b.to(dtype), 255.0))

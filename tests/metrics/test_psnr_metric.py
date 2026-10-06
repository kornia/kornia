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

    def test_exception_shape_mismatch(self, device, dtype):
        a = torch.ones(4, device=device, dtype=dtype)
        b = torch.ones(8, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="Expected tensors of equal shapes"):
            kornia.metrics.psnr(a, b, max_val=1.0)


class TestConventionsPsnr(BaseTester):
    """Pins for the batch pooling, ``max_val``, identical images and the integer-image wart of :func:`psnr`."""

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

    def test_wart_psnr_does_not_compute_integer_images_in_float32_5536(self, device, dtype):
        """psnr does not compute integer images in float32, as ssim does: torch gets them unconverted (#5536)."""
        # The uint8 differences of this pair wrap, so the unconverted pair either raises or misses the value of its
        # float32 copy. The same values in a floating dtype give a finite value.
        a = torch.tensor([[0, 64, 128], [192, 255, 32]], dtype=torch.uint8).view(1, 1, 2, 3).to(device)
        b = torch.tensor([[16, 64, 100], [200, 250, 0]], dtype=torch.uint8).view(1, 1, 2, 3).to(device)
        expected = kornia.metrics.psnr(a.float(), b.float(), 255.0)
        try:
            actual = kornia.metrics.psnr(a, b, 255.0)
        except (NotImplementedError, RuntimeError):
            actual = None
        assert actual is None or (actual.float() - expected).abs() > 0.1
        assert torch.isfinite(kornia.metrics.psnr(a.to(dtype), b.to(dtype), 255.0))

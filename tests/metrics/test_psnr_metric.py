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

    def test_exception_shape_mismatch(self, device, dtype):
        a = torch.ones(4, device=device, dtype=dtype)
        b = torch.ones(8, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="Expected tensors of equal shapes"):
            kornia.metrics.psnr(a, b, max_val=1.0)

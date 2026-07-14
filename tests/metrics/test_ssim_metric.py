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

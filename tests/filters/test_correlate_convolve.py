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

import torch

import kornia

from testing.base import BaseTester


class TestCorrelate2d(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 3, 4, 5, device=device, dtype=dtype)
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        output = kornia.filters.correlate2d(img, kernel)
        assert output.shape == img.shape

    def test_consistency_with_filter2d(self, device, dtype):
        img = torch.rand(1, 2, 8, 8, device=device, dtype=dtype)
        kernel = torch.rand(1, 5, 5, device=device, dtype=dtype)

        result_correlate = kornia.filters.correlate2d(img, kernel)
        result_filter2d = kornia.filters.filter2d(img, kernel, behaviour="corr")

        self.assert_close(result_correlate, result_filter2d)

    def test_with_padding_valid(self, device, dtype):
        img = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        kernel = torch.ones(1, 3, 3, device=device, dtype=dtype)
        output = kornia.filters.correlate2d(img, kernel, padding="valid")
        assert output.shape == (1, 1, 3, 3)

    def test_with_border_constant(self, device, dtype):
        img = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        kernel = torch.ones(1, 3, 3, device=device, dtype=dtype)
        output = kornia.filters.correlate2d(img, kernel, border_type="constant")
        assert output.shape == img.shape

    def test_gradcheck(self, device):
        img = torch.rand(1, 1, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        kernel = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        self.gradcheck(kornia.filters.correlate2d, (img, kernel), nondet_tol=1e-8)


class TestConvolve2d(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 3, 4, 5, device=device, dtype=dtype)
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        output = kornia.filters.convolve2d(img, kernel)
        assert output.shape == img.shape

    def test_consistency_with_filter2d(self, device, dtype):
        img = torch.rand(1, 2, 8, 8, device=device, dtype=dtype)
        kernel = torch.rand(1, 5, 5, device=device, dtype=dtype)

        result_convolve = kornia.filters.convolve2d(img, kernel)
        result_filter2d = kornia.filters.filter2d(img, kernel, behaviour="conv")

        self.assert_close(result_convolve, result_filter2d)

    def test_kernel_flipping(self, device, dtype):
        img = torch.rand(2, 1, 5, 5, device=device, dtype=dtype)
        kernel = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=device, dtype=dtype).unsqueeze(0)

        result_convolve = kornia.filters.convolve2d(img, kernel)
        result_correlate = kornia.filters.correlate2d(img, kernel.flip(-2, -1))

        self.assert_close(result_convolve, result_correlate)

    def test_with_padding_valid(self, device, dtype):
        img = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        kernel = torch.ones(1, 3, 3, device=device, dtype=dtype)
        output = kornia.filters.convolve2d(img, kernel, padding="valid")
        assert output.shape == (1, 1, 3, 3)

    def test_with_border_circular(self, device, dtype):
        img = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        kernel = torch.ones(1, 3, 3, device=device, dtype=dtype)
        output = kornia.filters.convolve2d(img, kernel, border_type="circular")
        assert output.shape == img.shape

    def test_gradcheck(self, device):
        img = torch.rand(1, 1, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        kernel = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        self.gradcheck(kornia.filters.convolve2d, (img, kernel), nondet_tol=1e-8)


class TestCorrelateConvolveEquivalence(BaseTester):
    def test_kernel_relationship(self, device, dtype):
        img = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device=device, dtype=dtype)
        kernel = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device=device, dtype=dtype)

        conv_result = kornia.filters.convolve2d(img, kernel, padding="valid")

        flipped_kernel = kernel.flip(-2, -1)
        conv_from_flipped_corr = kornia.filters.correlate2d(img, flipped_kernel, padding="valid")

        self.assert_close(conv_result, conv_from_flipped_corr)

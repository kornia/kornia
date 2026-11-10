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

from kornia.filters.kernels import get_pascal_kernel_1d, get_pascal_kernel_2d

from testing.base import BaseTester


class TestPascalKernel2d(BaseTester):
    @pytest.mark.parametrize("kernel_size", [5, 7, 9, 11, 17, 33, (9, 11), (3, 17)])
    def test_normalized_binomial_coefficients(self, kernel_size, device, dtype):
        ky, kx = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        # Pascal rows sum to 2**(size - 1), so the reference needs no tensor reduction.
        expected = torch.tensor(
            [[math.comb(ky - 1, y) * math.comb(kx - 1, x) / 2 ** (ky + kx - 2) for x in range(kx)] for y in range(ky)],
            device=device,
            dtype=dtype,
        )
        actual = get_pascal_kernel_2d(kernel_size, device=device, dtype=dtype)

        self.assert_close(actual, expected)
        self.assert_close(actual.sum(), actual.new_tensor(1.0))

    def test_unnormalized(self, device, dtype):
        actual = get_pascal_kernel_2d((3, 4), norm=False, device=device, dtype=dtype)
        expected = torch.tensor([[1, 3, 3, 1], [2, 6, 6, 2], [1, 3, 3, 1]], device=device, dtype=dtype)
        self.assert_close(actual, expected, atol=0, rtol=0)


class TestPascalKernel1d(BaseTester):
    @pytest.mark.parametrize("kernel_size", [9, 17, 33])
    def test_normalized_binomial_coefficients(self, kernel_size, device, dtype):
        expected = torch.tensor(
            [math.comb(kernel_size - 1, x) / 2 ** (kernel_size - 1) for x in range(kernel_size)],
            device=device,
            dtype=dtype,
        )
        actual = get_pascal_kernel_1d(kernel_size, norm=True, device=device, dtype=dtype)
        self.assert_close(actual, expected)
        self.assert_close(actual.sum(), actual.new_tensor(1.0))

    @pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
    def test_normalized_nonfloating_dtype(self, device, dtype):
        row = get_pascal_kernel_1d(5, device=device, dtype=dtype)
        expected = row / row.sum()
        self.assert_close(get_pascal_kernel_1d(5, norm=True, device=device, dtype=dtype), expected)
        raw = row[:, None] * row
        self.assert_close(get_pascal_kernel_2d(5, device=device, dtype=dtype), raw / raw.sum())

    def test_default_float16(self, device):
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            actual = get_pascal_kernel_2d(17, device=device)
            expected = get_pascal_kernel_2d(17, device=device, dtype=torch.float32).half()
            self.assert_close(actual, expected)
            self.assert_close(actual.sum(), actual.new_tensor(1.0))
        finally:
            torch.set_default_dtype(previous_dtype)

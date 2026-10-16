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

from kornia.filters import get_hanning_kernel1d, get_hanning_kernel2d

from testing.base import BaseTester, assert_close


@pytest.mark.parametrize("window_size", [5, 11])
def test_get_hanning_kernel(window_size, device, dtype):
    kernel = get_hanning_kernel1d(window_size, dtype=dtype, device=device)
    assert kernel.shape == (window_size,)
    assert kernel.max().item() == pytest.approx(1.0)


@pytest.mark.parametrize("ksize_x", [5, 11])
@pytest.mark.parametrize("ksize_y", [3, 7])
def test_get_hanning_kernel2d(ksize_x, ksize_y, device, dtype):
    kernel = get_hanning_kernel2d((ksize_x, ksize_y), dtype=dtype, device=device)
    assert kernel.shape == (ksize_x, ksize_y)
    assert kernel.max().item() == pytest.approx(1.0)


def test_get_hanning_kernel1d_5(device, dtype):
    kernel = get_hanning_kernel1d(5, dtype=dtype, device=device)
    expected = torch.tensor([0, 0.5, 1.0, 0.5, 0], dtype=dtype, device=device)
    assert kernel.shape == (5,)
    assert_close(kernel, expected)


def test_get_hanning_kernel2d_3x4(device, dtype):
    kernel = get_hanning_kernel2d((3, 4), dtype=dtype, device=device)
    expected = torch.tensor(
        [[0.0, 0.00, 0.00, 0.0], [0.0, 0.75, 0.75, 0.0], [0.0, 0.00, 0.00, 0.0]], dtype=dtype, device=device
    )
    assert kernel.shape == (3, 4)
    assert_close(kernel, expected)


class TestConventionsHanningKernels(BaseTester):
    @pytest.mark.parametrize("window_size", [4, 5, 6])
    def test_convention_hanning_kernel1d_is_the_unnormalised_symmetric_window(self, window_size, device, dtype):
        kernel = get_hanning_kernel1d(window_size, device=device, dtype=dtype)
        # numpy.hanning, torch.hann_window(periodic=False): 0.5 - 0.5 cos(2 pi n / (k - 1)), zero at both ends
        n = torch.arange(window_size, dtype=torch.float64)
        expected = (0.5 - 0.5 * torch.cos(2 * math.pi * n / (window_size - 1))).to(device=device, dtype=dtype)
        half = dtype in (torch.float16, torch.bfloat16)  # the cosine is evaluated in the kernel's dtype
        self.assert_close(kernel, expected, low_tolerance=half)
        # not normalised: the taps sum to (k - 1) / 2
        total = torch.tensor((window_size - 1) / 2, device=device, dtype=dtype)
        self.assert_close(kernel.sum(), total, low_tolerance=half)

    def test_convention_hanning_kernel2d_is_the_outer_product_in_y_x_order(self, device, dtype):
        kernel = get_hanning_kernel2d((5, 4), device=device, dtype=dtype)
        assert kernel.shape == (5, 4)
        along_y = get_hanning_kernel1d(5, device=device, dtype=dtype)
        along_x = get_hanning_kernel1d(4, device=device, dtype=dtype)
        self.assert_close(kernel, along_y[:, None] * along_x[None, :])

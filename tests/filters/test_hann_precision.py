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

from kornia.filters import get_hanning_kernel1d, get_hanning_kernel2d

from testing.base import BaseTester


class TestHannPrecision(BaseTester):
    @pytest.mark.parametrize(
        "size,indices,values",
        [
            (
                64,
                [1, 8, 16, 31, 32, 47, 55, 62],
                [
                    0.002484612317299295,
                    0.15088159095696357,
                    0.5124653458690365,
                    0.9993784606094611,
                    0.9993784606094611,
                    0.5124653458690366,
                    0.15088159095696352,
                    0.0024846123172993506,
                ],
            ),
            (
                256,
                [1, 8, 32, 63, 64, 223, 247, 254],
                [
                    0.00015177401106419852,
                    0.009682614765511133,
                    0.14753722654692641,
                    0.490760547520435,
                    0.5030799733190693,
                    0.14753722654692636,
                    0.009682614765511188,
                    0.00015177401106419852,
                ],
            ),
        ],
    )
    def test_convention_sample_values(self, size, indices, values, device, dtype):
        # Offline literals: 0.5 - 0.5 * cos(2*pi*n/(M-1)), at the listed integer n.
        actual = get_hanning_kernel1d(size, device=device, dtype=dtype)
        expected = torch.tensor(values, device=device, dtype=dtype)
        self.assert_close(actual[indices], expected, atol=1e-7, rtol=2 * torch.finfo(dtype).eps + 1e-6)
        self.assert_close(actual, actual.flip(0), atol=1e-6, rtol=2 * torch.finfo(dtype).eps)
        self.assert_close(
            get_hanning_kernel2d((size, size), device=device, dtype=dtype), actual[:, None] * actual[None, :]
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        actual = torch_optimizer(get_hanning_kernel1d)(256, device=device, dtype=dtype)
        expected = torch.tensor(0.00015177401106419852, device=device, dtype=dtype)
        self.assert_close(actual[1], expected, atol=1e-7, rtol=2 * torch.finfo(dtype).eps + 1e-6)

    def test_default_dtype(self, device, dtype):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            actual = get_hanning_kernel1d(256, device=device)
            expected = torch.tensor(0.00015177401106419852, device=device, dtype=dtype)
            assert actual.dtype == dtype
            self.assert_close(actual[1], expected, atol=1e-7, rtol=2 * torch.finfo(dtype).eps + 1e-6)
        finally:
            torch.set_default_dtype(previous)

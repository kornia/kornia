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

from testing import unrepresentable_sizes


class TestUnrepresentableSizes:
    def test_bfloat16_includes_the_known_traps(self):
        sizes = unrepresentable_sizes(torch.bfloat16)
        # 258/300/1000/2050/3000: ``n - 1`` rounds (wave-5 and wave-8 divisor bugs in kornia#4006).
        # 257: ``n`` itself rounds to 256 (wave-9 size bug). Both operands matter.
        for n in (257, 258, 300, 1000, 2050, 3000):
            assert n in sizes

    def test_bfloat16_excludes_exact_neighbourhoods(self):
        sizes = unrepresentable_sizes(torch.bfloat16)
        # Every integer up to 256 is exact in bfloat16, so ``n`` and ``n - 1`` are both exact.
        assert all(n not in sizes for n in range(2, 257))

    def test_float16_boundary(self):
        sizes = unrepresentable_sizes(torch.float16)
        assert all(n not in sizes for n in range(2, 2049))
        for n in (2049, 2050, 3000, 3001):
            assert n in sizes

    def test_float32_is_empty_in_range(self):
        assert unrepresentable_sizes(torch.float32) == []
        assert unrepresentable_sizes(torch.float64) == []

    def test_is_sorted_and_bounded(self):
        sizes = unrepresentable_sizes(torch.bfloat16, lo=250, hi=270)
        assert sizes == sorted(sizes)
        assert sizes[0] >= 250 and sizes[-1] <= 270
        assert sizes == list(range(257, 271))  # every odd n is inexact; every even n has an odd n - 1

    def test_rejects_non_floating_dtype(self):
        with pytest.raises(TypeError, match="floating"):
            unrepresentable_sizes(torch.int64)

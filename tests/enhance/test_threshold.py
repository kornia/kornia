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

from kornia.enhance.threshold import ThresholdType, threshold


class TestThreshold:
    @pytest.mark.parametrize(
        "ttype",
        [
            ThresholdType.THRESH_BINARY,
            ThresholdType.THRESH_BINARY_INV,
            ThresholdType.THRESH_TRUNC,
            ThresholdType.THRESH_TOZERO,
            ThresholdType.THRESH_TOZERO_INV,
        ],
    )
    @pytest.mark.parametrize("shape", [(1, 1, 5, 7), (2, 3, 11, 9)])
    def test_output_properties(self, ttype, shape, device, dtype):
        x = torch.rand(shape, device=device, dtype=dtype)
        out = threshold(x, thresh=0.5, maxval=1.0, type=ttype)

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

    def test_binary_rule_strict_greater(self, device, dtype):
        x = torch.tensor([0.2, 0.5, 0.7], device=device, dtype=dtype)
        out = threshold(x, thresh=0.5, maxval=9.0, type=ThresholdType.THRESH_BINARY)
        expected = torch.tensor([0.0, 0.0, 9.0], device=device, dtype=dtype)
        assert torch.allclose(out, expected)

    def test_binary_inv_rule_strict_greater(self, device, dtype):
        x = torch.tensor([0.2, 0.5, 0.7], device=device, dtype=dtype)
        out = threshold(x, thresh=0.5, maxval=9.0, type=ThresholdType.THRESH_BINARY_INV)
        expected = torch.tensor([9.0, 9.0, 0.0], device=device, dtype=dtype)
        assert torch.allclose(out, expected)

    def test_trunc(self, device, dtype):
        x = torch.tensor([0.2, 0.5, 0.7], device=device, dtype=dtype)
        out = threshold(x, thresh=0.5, maxval=9.0, type=ThresholdType.THRESH_TRUNC)
        expected = torch.tensor([0.2, 0.5, 0.5], device=device, dtype=dtype)
        assert torch.allclose(out, expected)

    def test_tozero(self, device, dtype):
        x = torch.tensor([0.2, 0.5, 0.7], device=device, dtype=dtype)
        out = threshold(x, thresh=0.5, maxval=9.0, type=ThresholdType.THRESH_TOZERO)
        expected = torch.tensor([0.0, 0.0, 0.7], device=device, dtype=dtype)
        assert torch.allclose(out, expected)

    def test_tozero_inv(self, device, dtype):
        x = torch.tensor([0.2, 0.5, 0.7], device=device, dtype=dtype)
        out = threshold(x, thresh=0.5, maxval=9.0, type=ThresholdType.THRESH_TOZERO_INV)
        expected = torch.tensor([0.2, 0.5, 0.0], device=device, dtype=dtype)
        assert torch.allclose(out, expected)

    def test_otsu_raises(self, device, dtype):
        x = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        with pytest.raises(NotImplementedError):
            threshold(x, thresh=0.0, maxval=1.0, type=ThresholdType.THRESH_OTSU)

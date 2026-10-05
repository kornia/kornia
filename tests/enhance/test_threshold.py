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

    @pytest.mark.parametrize(
        "image_dtype, values, thresh, ttype, expected",
        [
            # A threshold below the range passes every element instead of wrapping to 255 or raising.
            (torch.uint8, [0, 100, 255], -1, ThresholdType.THRESH_BINARY, [9, 9, 9]),
            (torch.uint8, [0, 100, 255], -0.5, ThresholdType.THRESH_BINARY, [9, 9, 9]),
            (torch.uint8, [0, 100, 255], -1, ThresholdType.THRESH_TRUNC, [0, 0, 0]),
            # A threshold above the range passes no element instead of raising.
            (torch.uint8, [0, 100, 255], 300, ThresholdType.THRESH_BINARY_INV, [9, 9, 9]),
            (torch.uint8, [0, 100, 255], 255.5, ThresholdType.THRESH_TOZERO, [0, 0, 0]),
            # A negative fraction is rounded down, not truncated toward zero.
            (torch.int16, [-2, -1, 0, 1], -0.5, ThresholdType.THRESH_BINARY, [0, 0, 9, 9]),
            (torch.int16, [-2, -1, 0, 1], -1.5, ThresholdType.THRESH_TRUNC, [-2, -2, -2, -2]),
            (torch.int16, [-2, -1, 0, 1], -1.5, ThresholdType.THRESH_TOZERO_INV, [-2, 0, 0, 0]),
            (torch.uint8, [100, 101], 100.7, ThresholdType.THRESH_BINARY, [0, 9]),
            (torch.int16, [-1, 0, 1], float("nan"), ThresholdType.THRESH_TOZERO, [0, 0, 0]),
            (torch.int64, [2**60, 2**60 + 1, 2**60 + 2], 2**60 + 1, ThresholdType.THRESH_BINARY, [0, 0, 9]),
        ],
    )
    @pytest.mark.parametrize("tensor_thresh", [False, True])
    def test_integer_input_compares_against_the_threshold_as_given(
        self, image_dtype, values, thresh, ttype, expected, tensor_thresh, device
    ):
        x = torch.tensor(values, device=device, dtype=image_dtype)
        if tensor_thresh:
            thresh = torch.tensor(thresh, device=device)
        out = threshold(x, thresh=thresh, maxval=9, type=ttype)
        assert out.dtype == image_dtype
        assert out.tolist() == expected

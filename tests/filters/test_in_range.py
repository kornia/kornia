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

import re

import pytest
import torch

from kornia.filters import InRange, in_range
from kornia.utils._compat import torch_version

from testing.base import BaseTester, assert_close


def test_in_range(device, dtype):
    torch.manual_seed(1)
    input_tensor = torch.rand(1, 3, 3, 3, device=device)
    input_tensor = input_tensor.to(dtype=dtype)
    expected = torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]], device=device, dtype=dtype)
    lower = (0.2, 0.3, 0.4)
    upper = (0.8, 0.9, 1.0)
    result = in_range(input_tensor, lower, upper, return_mask=True)

    assert_close(result, expected, atol=1e-4, rtol=1e-4)


class TestInRange(BaseTester):
    def _get_expected(self, device, dtype):
        return torch.tensor(
            [[[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]],
            device=device,
            dtype=dtype,
        )

    def test_smoke(self, device, dtype):
        torch.manual_seed(1)
        input_tensor = torch.rand(1, 3, 3, 3, device=device)
        input_tensor = input_tensor.to(dtype=dtype)
        expected = self._get_expected(device=device, dtype=dtype)
        res = InRange(lower=(0.2, 0.3, 0.4), upper=(0.8, 0.9, 1.0), return_mask=True)(input_tensor)
        assert expected.shape == res.shape
        self.assert_close(res, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "input_shape, lower, upper",
        [
            ((1, 3, 3, 3), (0.2, 0.2, 0.2), (0.6, 0.6, 0.6)),
            ((2, 3, 3, 3), (0.2, 0.2, 0.2), (0.6, 0.6, 0.6)),
            ((5, 5, 3, 3), (0.2, 0.2, 0.2, 0.2, 0.2), (0.6, 0.6, 0.6, 0.6, 0.6)),
            ((3, 3), (0.2,), (0.6,)),
            ((2, 3, 3), (0.2, 0.2), (0.6, 0.6)),
        ],
    )
    def test_cardinality(self, input_shape, lower, upper, device, dtype):
        input_tensor = torch.rand(input_shape, device=device, dtype=dtype)
        res = InRange(lower=lower, upper=upper, return_mask=True)(input_tensor)

        if len(input_tensor.shape) == 2:
            assert res.shape == (res.shape[-2], res.shape[-1])
        elif len(input_tensor.shape) == 3:
            assert res.shape == (1, res.shape[-2], res.shape[-1])
        else:
            assert res.shape == (res.shape[0], 1, res.shape[-2], res.shape[-1])

    def test_exception(self, device, dtype):
        input_tensor = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        with pytest.raises(Exception, match="Invalid `lower` and `upper` format. Should be tuple or Tensor."):
            InRange(lower=3, upper=3)(input_tensor)

        with pytest.raises(Exception, match="Invalid `lower` and `upper` format. Should be tuple or Tensor."):
            InRange(lower=[0.2, 0.2], upper=[0.2, 0.2])(input_tensor)

        with pytest.raises(Exception, match="Invalid `lower` and `upper` format. Should be tuple or Tensor."):
            InRange(lower=(0.2), upper=(0.2))(input_tensor)

        with pytest.raises(
            ValueError, match="Shape of `lower`, `upper` and `input` image channels must have same shape."
        ):
            InRange(lower=(0.2,), upper=(0.2,))(input_tensor)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "`lower` and `upper` bounds as Tensors must have compatible shapes with the input (B, C, 1, 1)."
            ),
        ):
            lower = torch.tensor([0.2, 0.2, 0.2])
            upper = torch.tensor([0.6, 0.6, 0.6])
            InRange(lower=lower, upper=upper)(input_tensor)

        with pytest.raises(Exception, match="Invalid `return_mask` format. Should be boolean."):
            lower = torch.tensor([0.2, 0.2, 0.2])
            upper = torch.tensor([0.6, 0.6, 0.6])
            InRange(lower=lower, upper=upper, return_mask=2)(input_tensor)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(1, 3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        actual = InRange((0.2, 0.2, 0.2), (0.6, 0.6, 0.6), return_mask=True)(inp)
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 3, 5, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(in_range, (img, (0.2, 0.2, 0.2), (0.6, 0.6, 0.6), True))

    @pytest.mark.parametrize(
        "input_shape, lower, upper",
        [
            ((1, 3, 3, 3), (0.2, 0.2, 0.2), (0.6, 0.6, 0.6)),
            ((2, 3, 3, 3), (0.2, 0.2, 0.2), (0.6, 0.6, 0.6)),
            ((3, 3), (0.2,), (0.6,)),
        ],
    )
    def test_module(self, input_shape, lower, upper, device, dtype):
        img = torch.rand(input_shape, device=device, dtype=dtype)
        op = in_range
        op_module = InRange(lower=lower, upper=upper, return_mask=True)
        actual = op_module(img)
        expected = op(img, lower, upper, True)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, device, dtype, torch_optimizer):
        if device == torch.device("cpu") and torch_version() in {"2.3.0", "2.3.1"}:
            pytest.skip("Failing to compile on CPU see pytorch/pytorch#126619")
        data = torch.rand(batch_size, 3, 5, 5, device=device, dtype=dtype)
        op = InRange(lower=(0.2, 0.2, 0.2), upper=(0.6, 0.6, 0.6), return_mask=True)
        op_optimized = torch_optimizer(op)
        self.assert_close(op(data), op_optimized(data))

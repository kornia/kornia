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

from kornia.enhance.integral import IntegralImage, IntegralTensor, integral_image, integral_tensor

from testing.base import BaseTester


class TestIntegralTensor(BaseTester):
    @pytest.mark.applies_to(dtypes=("float16", "bfloat16"), reason="Half-precision accumulation regression")
    @pytest.mark.parametrize("dim", [(0, 1), (1, 0)])
    def test_half_cancellation_across_axes(self, device, dtype, dim):
        eps = torch.finfo(dtype).eps
        rows = int(2 / eps) + 1
        tensor = torch.tensor([1.0, -(1.0 - eps)], device=device, dtype=dtype).expand(rows, 2)
        actual = integral_tensor(tensor, dim)
        # Each complete row sums to eps. The last sum is 2 + eps, which rounds
        # to exactly 2 in either half dtype; rounding each axis first gives 1.
        expected = torch.tensor([rows, 2.0], device=device, dtype=dtype)
        self.assert_close(actual[-1], expected)

    @pytest.mark.applies_to(dtypes=("float16",), reason="Float16 intermediate overflow")
    def test_half_representable_sum_after_overflowing_prefix(self, device, dtype):
        tensor = torch.tensor([1.0, -1.0], device=device, dtype=dtype).expand(256, 256, 2)
        actual = integral_tensor(tensor, (0, 1, 2))
        # All inputs are in [-1, 1]. The positive-only prefix truly exceeds the
        # output range, but adding its negative partner has the exact sum zero.
        expected = torch.tensor([float("inf"), 0.0], device=device, dtype=dtype)
        self.assert_close(actual[-1, -1], expected)

    def test_smoke(self, device, dtype):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        output = integral_tensor(tensor)
        assert output.shape == (1, 1, 4, 4)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1)])
    def test_cardinality(self, device, dtype, shape):
        tensor = torch.rand(*shape, device=device, dtype=dtype)
        output = integral_tensor(tensor)
        assert output.shape == shape

    def test_exception(self, device, dtype):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        with pytest.raises(Exception):
            integral_tensor(tensor, (0, 1, 2, 3, 4))
        with pytest.raises(Exception):
            integral_tensor(tensor, (4, 5))
        with pytest.raises(Exception) as errinfo:
            integral_tensor(tensor, ())
        assert "dim must be a non-empty tuple." in str(errinfo)

    @pytest.mark.parametrize("dim", [(1, 1), (-1, 1), (1, -1)])
    def test_duplicate_dimensions(self, device, dtype, dim):
        tensor = torch.ones(2, 2, device=device, dtype=dtype)
        with pytest.raises(Exception, match="unique dimensions"):
            integral_tensor(tensor, dim)

    @pytest.mark.parametrize("dim", [(2,), (-3,), (0, -3)])
    def test_out_of_range_dimensions(self, device, dtype, dim):
        # -3 + 2 = -1 is a valid cumsum axis, so a missing bounds check would integrate the last axis silently.
        tensor = torch.ones(2, 2, device=device, dtype=dtype)
        with pytest.raises(Exception, match="valid dimensions"):
            integral_tensor(tensor, dim)

    @pytest.mark.parametrize("dim", [(0, 1), (1, 0), (-2, -1)])
    def test_distinct_dimensions(self, device, dtype, dim):
        tensor = torch.ones(2, 2, device=device, dtype=dtype)
        output = integral_tensor(tensor, dim)
        expected = torch.tensor([[1, 2], [2, 4]], device=device, dtype=dtype)
        self.assert_close(output, expected)

    def test_module(self, device, dtype):
        mod = IntegralTensor()
        op = integral_tensor
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        self.assert_close(mod(tensor), op(tensor))

    def test_gradcheck(self, device):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(integral_tensor, (tensor,))

    def test_value(self, device, dtype):
        tensor = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device=device, dtype=dtype)
        output = integral_tensor(tensor)
        expected = torch.tensor([[[[1.0, 3.0, 6.0], [4.0, 9.0, 15.0], [7.0, 15.0, 24.0]]]], device=device, dtype=dtype)
        self.assert_close(output, expected)


class TestIntegralImage(BaseTester):
    @pytest.mark.applies_to(dtypes=("float16", "bfloat16"), reason="Half-precision accumulation regression")
    def test_half_gradient_cancellation_across_axes(self, device, dtype):
        eps = torch.finfo(dtype).eps
        columns = int(2 / eps) + 1
        tensor = torch.zeros(2, columns, device=device, dtype=dtype, requires_grad=True)
        weights = torch.tensor([[1.0], [-(1.0 - eps)]], device=device, dtype=dtype).expand(2, columns)
        actual = IntegralImage()(tensor)
        (gradient,) = torch.autograd.grad(actual, tensor, weights)
        # Every output contains the top-left input. Its derivative is the sum
        # of all weights: columns * eps = 2 + eps, rounded to 2, not 1.
        self.assert_close(gradient[0, 0], torch.tensor(2.0, device=device, dtype=dtype))

    @pytest.mark.applies_to(dtypes=("float16", "bfloat16"), reason="Half-precision accumulation regression")
    def test_dynamo_half_cancellation(self, device, dtype, torch_optimizer):
        eps = torch.finfo(dtype).eps
        length = int(2 / eps) + 1
        op = torch_optimizer(IntegralImage())
        tensor = torch.tensor([1.0, -(1.0 - eps)], device=device, dtype=dtype).expand(length, 2)
        self.assert_close(op(tensor)[-1], torch.tensor([length, 2.0], device=device, dtype=dtype))

        tensor = torch.zeros(2, length, device=device, dtype=dtype, requires_grad=True)
        weights = torch.tensor([[1.0], [-(1.0 - eps)]], device=device, dtype=dtype).expand(2, length)
        (gradient,) = torch.autograd.grad(op(tensor), tensor, weights)
        self.assert_close(gradient[0, 0], torch.tensor(2.0, device=device, dtype=dtype))

    def test_smoke(self, device, dtype):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        output = integral_image(tensor)
        assert output.shape == (1, 1, 4, 4)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1)])
    def test_cardinality(self, device, dtype, shape):
        tensor = torch.rand(*shape, device=device, dtype=dtype)
        output = integral_image(tensor)
        assert output.shape == shape

    def test_exception(self, device, dtype):
        tensor = torch.rand(4, device=device, dtype=dtype)
        with pytest.raises(Exception):
            integral_image(tensor)

    def test_module(self, device, dtype):
        mod = IntegralImage()
        op = integral_image
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        self.assert_close(mod(tensor), op(tensor))

    def test_gradcheck(self, device):
        tensor = torch.rand(1, 1, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(integral_image, (tensor,))

    def test_values(self, device, dtype):
        tensor = torch.tensor(
            [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], device=device, dtype=dtype
        )
        output = integral_image(tensor)
        expected = torch.tensor(
            [[[[1, 3, 6, 10], [6.0, 14.0, 24.0, 36.0], [15.0, 33.0, 54.0, 78.0], [28.0, 60.0, 96.0, 136.0]]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(output, expected)

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

import numpy as np
import pytest
import torch

from kornia.core.exceptions import BaseError
from kornia.filters import (
    BlurPool2D,
    EdgeAwareBlurPool2D,
    MaxBlurPool2D,
    blur_pool2d,
    edge_aware_blur_pool2d,
    max_blur_pool2d,
)

from testing.base import BaseTester


def _zero_padded_reference(x: torch.Tensor, k: int, s: int) -> torch.Tensor:
    """Blur pool as adobe/antialiased-cnns BlurPool with pad_type='zero', in float64 on CPU.

    Pads ``(k - 1) // 2`` zeros before and ``k // 2`` after along each spatial axis, then correlates with the normalised
    Pascal kernel at stride ``s``, one output pixel at a time, without F.pad or F.conv2d.
    """
    x = x.cpu().double()
    *lead, h, w = x.shape
    taps = torch.tensor([math.comb(k - 1, i) for i in range(k)], dtype=torch.float64)
    kernel = torch.outer(taps, taps) / (2.0 ** (2 * (k - 1)))
    before = (k - 1) // 2
    padded = torch.zeros(*lead, h + k - 1, w + k - 1, dtype=torch.float64)
    padded[..., before : before + h, before : before + w] = x
    h_out, w_out = math.ceil(h / s), math.ceil(w / s)
    out = torch.zeros(*lead, h_out, w_out, dtype=torch.float64)
    for i in range(h_out):
        for j in range(w_out):
            out[..., i, j] = (padded[..., i * s : i * s + k, j * s : j * s + k] * kernel).sum(dim=(-2, -1))
    return out


class TestMaxBlurPool(BaseTester):
    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_smoke(self, kernel_size, ceil_mode, device, dtype):
        data = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)
        actual = MaxBlurPool2D(kernel_size, ceil_mode=ceil_mode)(data)

        assert actual.shape == (1, 1, 5, 5)

    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, batch_size, kernel_size, ceil_mode, device, dtype):
        data = torch.zeros(batch_size, 4, 4, 8, device=device, dtype=dtype)
        blur = MaxBlurPool2D(kernel_size, ceil_mode=ceil_mode)
        assert blur(data).shape == (batch_size, 4, 2, 4)

    def test_exception(self):
        data = torch.rand(1, 1, 3, 3)
        with pytest.raises(Exception) as errinfo:
            MaxBlurPool2D((3, 5))(data)
        assert "Invalid kernel shape. Expect CxC_outxNxN" in str(errinfo)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_noncontiguous(self, batch_size, device, dtype):
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = max_blur_pool2d(inp, 3)

        assert actual.is_contiguous()

    @pytest.mark.parametrize("kernel_size", [3, 4])
    def test_gradcheck(self, kernel_size, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(max_blur_pool2d, (img, kernel_size))

    @pytest.mark.parametrize("kernel_size", [(3, 3), 5])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_module(self, kernel_size, batch_size, device, dtype):
        op = max_blur_pool2d
        op_module = MaxBlurPool2D

        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("kernel_size", [3, 4, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_dynamo(self, batch_size, kernel_size, ceil_mode, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = MaxBlurPool2D(kernel_size, ceil_mode=ceil_mode)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

    @pytest.mark.parametrize("kernel_size", [2, 3, 4])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("max_pool_size", [1, 2, 3])
    def test_even_kernel_keeps_ceil_size_5166(self, kernel_size, stride, max_pool_size, device, dtype):
        # An even kernel_size used to lose a row and a column: (7, 10) with kernel_size 2, stride 1, max_pool_size 1
        # returned (6, 9). The size is ceil((H - max_pool_size + 1) / stride) for every kernel size.
        data = torch.rand(1, 2, 7, 10, device=device, dtype=dtype)
        actual = max_blur_pool2d(data, kernel_size, stride=stride, max_pool_size=max_pool_size)
        expected = tuple(math.ceil((n - max_pool_size + 1) / stride) for n in (7, 10))
        assert actual.shape == (1, 2, *expected)

    @pytest.mark.parametrize("kernel_size", [2, 3, 4])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("max_pool_size", [1, 2, 3])
    def test_even_kernel_is_max_pool_then_blur_pool_5166(self, kernel_size, stride, max_pool_size, device, dtype):
        # Max pooling at stride 1, then the float64 zero-padded reference blur pool instead of kornia's own helper.
        data = torch.rand(1, 2, 7, 10, device=device, dtype=dtype)
        pooled = torch.nn.functional.max_pool2d(data.cpu().double(), max_pool_size, stride=1)
        expected = _zero_padded_reference(pooled, kernel_size, stride)
        actual = max_blur_pool2d(data, kernel_size, stride=stride, max_pool_size=max_pool_size)
        self.assert_close(actual, expected.to(device=device, dtype=dtype))


class TestBlurPool(BaseTester):
    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_smoke(self, kernel_size, stride, device, dtype):
        data = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)
        actual = BlurPool2D(kernel_size, stride=stride)(data)
        expected = (1, 1, int(10 / stride), int(10 / stride))
        assert actual.shape == expected

    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_cardinality(self, batch_size, kernel_size, stride, device, dtype):
        data = torch.zeros(batch_size, 4, 4, 8, device=device, dtype=dtype)
        actual = BlurPool2D(kernel_size, stride=stride)(data)
        expected = (batch_size, 4, int(4 / stride), int(8 / stride))
        assert actual.shape == expected

    def test_exception(self):
        data = torch.rand(1, 1, 3, 3)
        with pytest.raises(Exception) as errinfo:
            BlurPool2D((3, 5))(data)
        assert "Invalid kernel shape. Expect CxC_(out, None)xNxN" in str(errinfo)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_noncontiguous(self, batch_size, device, dtype):
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = blur_pool2d(inp, 3)
        assert actual.is_contiguous()

    @pytest.mark.parametrize("kernel_size", [3, 4])
    def test_gradcheck(self, kernel_size, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(blur_pool2d, (img, kernel_size))

    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_module(self, batch_size, kernel_size, stride, device, dtype):
        op = blur_pool2d
        op_module = BlurPool2D

        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("kernel_size", [3, 4, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_dynamo(self, batch_size, kernel_size, stride, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = BlurPool2D(kernel_size, stride=stride)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

    @pytest.mark.parametrize("kernel_size", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_even_kernel_keeps_ceil_size_5166(self, kernel_size, stride, device, dtype):
        # An even kernel_size used to lose a row and a column: (7, 10) returned (6, 9) at stride 1 and (3, 5) at
        # stride 2. The size is ceil(H / stride) x ceil(W / stride) for every kernel size.
        data = torch.rand(1, 2, 7, 10, device=device, dtype=dtype)
        actual = blur_pool2d(data, kernel_size, stride=stride)
        assert actual.shape == (1, 2, math.ceil(7 / stride), math.ceil(10 / stride))

    @pytest.mark.parametrize("kernel_size", [2, 3, 4, 5])
    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_matches_zero_padded_reference_5166(self, kernel_size, stride, device, dtype):
        data = torch.rand(2, 3, 7, 10, device=device, dtype=dtype)
        expected = _zero_padded_reference(data, kernel_size, stride)
        self.assert_close(blur_pool2d(data, kernel_size, stride=stride), expected.to(device=device, dtype=dtype))


class TestEdgeAwareBlurPool(BaseTester):
    @pytest.mark.parametrize("edge_dilation_kernel_size", [0, -1, 2, 4])
    def test_exception_edge_dilation_kernel_size(self, edge_dilation_kernel_size, device, dtype):
        data = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="Kernel size must be an odd integer"):
            edge_aware_blur_pool2d(data, 3, edge_dilation_kernel_size=edge_dilation_kernel_size)
        with pytest.raises(BaseError, match="Kernel size must be an odd integer"):
            EdgeAwareBlurPool2D(3, edge_dilation_kernel_size=edge_dilation_kernel_size)

    @pytest.mark.parametrize("make_size", [np.int64, torch.tensor], ids=["numpy_int", "zero_dim_tensor"])
    def test_edge_dilation_kernel_size_accepts_integer_like(self, make_size, device, dtype):
        # integer-like sizes (numpy ints, 0-d integer tensors) are accepted like Python ints
        data = torch.rand(1, 1, 8, 8, device=device, dtype=dtype) + 0.1
        expected = edge_aware_blur_pool2d(data, 3, edge_dilation_kernel_size=3)
        self.assert_close(edge_aware_blur_pool2d(data, 3, edge_dilation_kernel_size=make_size(3)), expected)
        self.assert_close(EdgeAwareBlurPool2D(3, edge_dilation_kernel_size=make_size(3))(data), expected)

    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("edge_threshold", [1.25, 2.5])
    @pytest.mark.parametrize("edge_dilation_kernel_size", [3, 5])
    def test_smoke(self, kernel_size, batch_size, edge_threshold, edge_dilation_kernel_size, device, dtype):
        data = torch.zeros(batch_size, 3, 8, 8, device=device, dtype=dtype)
        actual = edge_aware_blur_pool2d(data, kernel_size, edge_threshold, edge_dilation_kernel_size)
        assert actual.shape == data.shape

    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, kernel_size, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 8, 8, device=device, dtype=dtype)
        blur = edge_aware_blur_pool2d(inp, kernel_size=kernel_size)
        assert blur.shape == inp.shape

    def test_exception(self):
        from kornia.core.exceptions import BaseError, ShapeError

        data = torch.rand(1, 3, 3)
        with pytest.raises(ShapeError) as errinfo:
            edge_aware_blur_pool2d(data, 3)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)
        data = torch.rand(1, 1, 3, 3)
        with pytest.raises(BaseError) as errinfo:
            edge_aware_blur_pool2d(data, 3, edge_threshold=-1)
        assert "edge threshold should be positive, but got" in str(errinfo.value)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_noncontiguous(self, batch_size, device, dtype):
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = edge_aware_blur_pool2d(inp, 3)
        assert actual.is_contiguous()

    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_module(self, kernel_size, batch_size, device, dtype):
        op = edge_aware_blur_pool2d
        op_module = EdgeAwareBlurPool2D

        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("kernel_size", [3, 4])
    def test_gradcheck(self, kernel_size, device):
        img = torch.rand((1, 2, 5, 4), device=device, dtype=torch.float64)
        self.gradcheck(edge_aware_blur_pool2d, (img, kernel_size))

    def test_smooth(self, device, dtype):
        img = torch.ones(1, 1, 5, 5, device=device, dtype=dtype)
        img[0, 0, :, :2] = 0
        blur = edge_aware_blur_pool2d(img, kernel_size=3, edge_threshold=32.0)
        self.assert_close(img, blur)

    @pytest.mark.parametrize("kernel_size", [3, 4, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        op = edge_aware_blur_pool2d
        data = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        op = EdgeAwareBlurPool2D(kernel_size)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

    @pytest.mark.parametrize("kernel_size", [2, 4])
    def test_even_kernel_keeps_shape_5166(self, kernel_size, device, dtype):
        # blur_pool2d(stride=1) used to return one row and one column less for an even kernel_size, and the fusion
        # with the input raised a shape error.
        data = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)
        assert edge_aware_blur_pool2d(data, kernel_size).shape == data.shape

    @pytest.mark.parametrize("kernel_size", [2, 4])
    @pytest.mark.parametrize("shape", [(8, 8), (8, 9)])
    def test_even_kernel_matches_blur_without_edges_5166(self, kernel_size, shape, device, dtype):
        # With edge_threshold=1e6 no pixel is an edge, so the output is the stride-1 blur of the input reflect-padded by
        # 2 pixels, cropped back to the input size.
        data = torch.rand(1, 3, *shape, device=device, dtype=dtype) + 0.1
        actual = edge_aware_blur_pool2d(data, kernel_size, edge_threshold=1e6)
        padded = torch.nn.functional.pad(data.cpu().double(), (2, 2, 2, 2), mode="reflect")
        expected = _zero_padded_reference(padded, kernel_size, 1)[..., 2:-2, 2:-2]
        self.assert_close(actual, expected.to(device=device, dtype=dtype))

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
import warnings

import pytest
import torch
import torch.nn.functional as F

from kornia.filters import (
    BlurPool2D,
    EdgeAwareBlurPool2D,
    MaxBlurPool2D,
    blur_pool2d,
    edge_aware_blur_pool2d,
    max_blur_pool2d,
)

from testing.base import BaseTester, supports_reflect_padding


def _blur_pool_reference(image: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """``BlurPool`` of adobe/antialiased-cnns with ``pad_type='zero'``, in float64 on the CPU.

    ``antialiased_cnns/blurpool.py`` pads ``(k - 1) // 2`` pixels before and ``ceil((k - 1) / 2)`` after on each axis,
    then correlates every channel at ``stride`` with the outer product of the binomial row ``C(k - 1, i)``, normalised
    to sum to one. Written here with a zero canvas and ``unfold``, without ``F.pad`` or ``F.conv2d``.
    """
    k = kernel_size
    x = image.detach().cpu().double()
    h, w = x.shape[-2:]
    before, after = (k - 1) // 2, math.ceil((k - 1) / 2)
    canvas = x.new_zeros(*x.shape[:-2], h + before + after, w + before + after)
    canvas[..., before : before + h, before : before + w] = x
    taps = torch.tensor([math.comb(k - 1, i) for i in range(k)], dtype=torch.float64)
    weights = torch.outer(taps, taps) / taps.sum() ** 2
    windows = canvas.unfold(-2, k, stride).unfold(-2, k, stride)  # (..., H_out, W_out, k, k)
    return (windows * weights).sum((-2, -1))


class TestMaxBlurPool(BaseTester):
    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    def test_smoke(self, kernel_size, device, dtype):
        data = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)
        actual = MaxBlurPool2D(kernel_size)(data)

        assert actual.shape == (1, 1, 5, 5)

    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, batch_size, kernel_size, device, dtype):
        data = torch.zeros(batch_size, 4, 4, 8, device=device, dtype=dtype)
        blur = MaxBlurPool2D(kernel_size)
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

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(max_blur_pool2d, (img, 3))

    @pytest.mark.parametrize("kernel_size", [(3, 3), 5])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_module(self, kernel_size, batch_size, device, dtype):
        op = max_blur_pool2d
        op_module = MaxBlurPool2D

        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("kernel_size", [3, (5, 5), 4])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = MaxBlurPool2D(kernel_size)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

    def test_even_kernel_keeps_the_last_row_and_column(self, device, dtype):
        # An even kernel_size used to lose a row and a column: 8x8 at stride 1 with max_pool_size 1 returned 7x7.
        data = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        assert max_blur_pool2d(data, 4, stride=1, max_pool_size=1).shape == (1, 1, 8, 8)
        assert MaxBlurPool2D(4, stride=1, max_pool_size=1)(data).shape == (1, 1, 8, 8)

    @pytest.mark.parametrize("kernel_size", [2, 3, 4, 5])
    @pytest.mark.parametrize("stride", [1, 2, 3])
    @pytest.mark.parametrize("max_pool_size", [1, 2, 3])
    def test_matches_max_pool_then_reference(self, kernel_size, stride, max_pool_size, device, dtype):
        # antialiased-cnns' MaxBlurPool is a stride-1 MaxPool2d followed by BlurPool, so the output is
        # ceil((H - max_pool_size + 1) / stride) x ceil((W - max_pool_size + 1) / stride) for every kernel size.
        data = torch.rand(2, 3, 7, 10, device=device, dtype=dtype)
        actual = max_blur_pool2d(data, kernel_size, stride=stride, max_pool_size=max_pool_size)
        size = tuple(math.ceil((n - max_pool_size + 1) / stride) for n in (7, 10))
        assert actual.shape == (2, 3, *size)
        expected = _blur_pool_reference(F.max_pool2d(data, max_pool_size, stride=1), kernel_size, stride)
        self.assert_close(actual, expected.to(device=device, dtype=dtype))
        self.assert_close(MaxBlurPool2D(kernel_size, stride, max_pool_size)(data), actual)

    def test_ceil_mode_is_deprecated(self, device, dtype):
        # ceil_mode reached a stride-1 max pool, where floor and ceil rounding agree, so it never changed the output.
        data = torch.rand(1, 2, 7, 10, device=device, dtype=dtype)
        expected = max_blur_pool2d(data, 3, stride=2, max_pool_size=2)
        for ceil_mode in (True, False):
            with pytest.warns(DeprecationWarning, match="`ceil_mode` is deprecated") as record:
                actual = max_blur_pool2d(data, 3, stride=2, max_pool_size=2, ceil_mode=ceil_mode)
            assert record[0].filename == __file__
            assert torch.equal(actual, expected)
            with pytest.warns(DeprecationWarning, match="`ceil_mode` is deprecated"):
                assert torch.equal(max_blur_pool2d(data, 3, 2, 2, ceil_mode), expected)
            with pytest.warns(DeprecationWarning, match="`ceil_mode` is deprecated") as record:
                module = MaxBlurPool2D(3, stride=2, max_pool_size=2, ceil_mode=ceil_mode)
            assert record[0].filename == __file__
            with pytest.warns(DeprecationWarning, match="`ceil_mode` is deprecated"):
                MaxBlurPool2D(3, 2, 2, ceil_mode)
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                assert torch.equal(module(data), expected)

    def test_no_warning_without_ceil_mode(self, device, dtype):
        data = torch.rand(1, 2, 7, 10, device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            max_blur_pool2d(data, 3)
            MaxBlurPool2D(3)(data)


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

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(blur_pool2d, (img, 3))

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

    @pytest.mark.parametrize("kernel_size", [3, (5, 5), 4])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_dynamo(self, batch_size, kernel_size, stride, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = BlurPool2D(kernel_size, stride=stride)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

    @pytest.mark.parametrize("kernel_size", [2, 4, 6, (4, 4)])
    def test_even_kernel_keeps_the_last_row_and_column(self, kernel_size, device, dtype):
        # An even kernel_size used to lose a row and a column: 7x10 returned 6x9 at stride 1 and 3x5 at stride 2.
        data = torch.rand(1, 2, 7, 10, device=device, dtype=dtype)
        for stride, size in ((1, (7, 10)), (2, (4, 5)), (3, (3, 4))):
            assert blur_pool2d(data, kernel_size, stride=stride).shape == (1, 2, *size)
            assert BlurPool2D(kernel_size, stride=stride)(data).shape == (1, 2, *size)

    @pytest.mark.parametrize("kernel_size", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_matches_antialiased_cnns_reference(self, kernel_size, stride, device, dtype):
        data = torch.rand(2, 3, 7, 10, device=device, dtype=dtype)
        actual = blur_pool2d(data, kernel_size, stride=stride)
        assert actual.shape == (2, 3, math.ceil(7 / stride), math.ceil(10 / stride))
        self.assert_close(actual, _blur_pool_reference(data, kernel_size, stride).to(device=device, dtype=dtype))
        self.assert_close(BlurPool2D(kernel_size, stride=stride)(data), actual)

    @pytest.mark.parametrize("kernel_size", [2, 3, 4])
    def test_gradcheck_kernel_sizes(self, kernel_size, device):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        self.gradcheck(blur_pool2d, (img, kernel_size, 2))
        self.gradcheck(max_blur_pool2d, (img, kernel_size, 1))


class TestEdgeAwareBlurPool(BaseTester):
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

    def test_gradcheck(self, device):
        img = torch.rand((1, 2, 5, 4), device=device, dtype=torch.float64)
        self.gradcheck(edge_aware_blur_pool2d, (img, 3))

    def test_smooth(self, device, dtype):
        img = torch.ones(1, 1, 5, 5, device=device, dtype=dtype)
        img[0, 0, :, :2] = 0
        blur = edge_aware_blur_pool2d(img, kernel_size=3, edge_threshold=32.0)
        self.assert_close(img, blur)

    @pytest.mark.parametrize("kernel_size", [3, (5, 5), 4])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        op = edge_aware_blur_pool2d
        data = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        op = EdgeAwareBlurPool2D(kernel_size)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

    @pytest.mark.parametrize("kernel_size", [2, 4, (4, 4)])
    @pytest.mark.parametrize("shape", [(8, 8), (8, 9)])
    def test_even_kernel_size(self, kernel_size, shape, device, dtype):
        # An even kernel_size used to fail with a raw shape error in the final blend: the stride-1 blur came back one
        # row and one column short. Away from edges the output is the stride-1 blur pool of the reflect-padded input.
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        data = torch.rand(2, 3, *shape, device=device, dtype=dtype) + 0.1
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        actual = edge_aware_blur_pool2d(data, kernel_size, edge_threshold=1e6)
        assert actual.shape == data.shape
        padded = F.pad(data, (2, 2, 2, 2), mode="reflect")
        expected = _blur_pool_reference(padded, k, 1)[..., 2:-2, 2:-2]
        self.assert_close(actual, expected.to(device=device, dtype=dtype))
        self.assert_close(EdgeAwareBlurPool2D(kernel_size, edge_threshold=1e6)(data), actual)
        assert EdgeAwareBlurPool2D(kernel_size)(data).shape == data.shape

    def test_gradcheck_even_kernel_size(self, device):
        img = torch.rand((1, 2, 5, 4), device=device, dtype=torch.float64) + 0.1
        self.gradcheck(edge_aware_blur_pool2d, (img, 4))

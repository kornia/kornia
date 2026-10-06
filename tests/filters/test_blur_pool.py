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

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kornia.core.exceptions import BaseError
from kornia.filters import (
    BlurPool2D,
    EdgeAwareBlurPool2D,
    MaxBlurPool2D,
    blur_pool2d,
    edge_aware_blur_pool2d,
    max_blur_pool2d,
)

from testing.base import BaseTester, supports_reflect_padding


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

    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize(("height", "width"), [(7, 10), (8, 11), (5, 5)])
    @pytest.mark.parametrize(("kernel_size", "max_pool_size"), [(3, 2), (3, 3), (4, 3)])
    def test_ceil_mode_is_deprecated_and_changes_nothing(
        self, ceil_mode, height, width, kernel_size, max_pool_size, device, dtype
    ):
        """At the stride-1 max pool floor and ceil agree, so the flag never did anything (#5165)."""
        data = torch.rand(1, 2, height, width, device=device, dtype=dtype)
        expected = max_blur_pool2d(data, kernel_size, stride=2, max_pool_size=max_pool_size)

        with pytest.warns(DeprecationWarning, match="ceil_mode") as record:
            actual = max_blur_pool2d(data, kernel_size, stride=2, max_pool_size=max_pool_size, ceil_mode=ceil_mode)
        self.assert_close(actual, expected)
        # The warning points at the caller's line, not at kornia.
        assert [w.filename for w in record if "ceil_mode" in str(w.message)] == [__file__]

        with pytest.warns(DeprecationWarning, match="ceil_mode") as record:
            module = MaxBlurPool2D(kernel_size, stride=2, max_pool_size=max_pool_size, ceil_mode=ceil_mode)
        self.assert_close(module(data), expected)
        assert [w.filename for w in record if "ceil_mode" in str(w.message)] == [__file__]
        assert module.ceil_mode is ceil_mode

    def test_no_warning_without_ceil_mode(self, device, dtype):
        data = torch.rand(1, 2, 8, 8, device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            module = MaxBlurPool2D(3)
            module(data)
            max_blur_pool2d(data, 3)
        # Code that reads the attribute still gets a bool.
        assert module.ceil_mode is False

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
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = MaxBlurPool2D(kernel_size)
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
    @pytest.mark.parametrize("kernel_size", [5, 7, 9, 11, 17])
    @pytest.mark.parametrize("op", [blur_pool2d, max_blur_pool2d, edge_aware_blur_pool2d])
    def test_large_kernel_matches_float32(self, kernel_size, op, device, dtype):
        data = torch.ones(1, 2, 32, 32, device=device, dtype=dtype)
        expected = op(data.float(), kernel_size).to(dtype)
        self.assert_close(op(data, kernel_size), expected)

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

    @pytest.mark.parametrize(("kernel_size", "image_size"), [(3, 16), (5, 16), (7, 16), (9, 16), (9, 4)])
    def test_constant_image_is_preserved_at_boundaries(self, kernel_size, image_size, device, dtype):
        inp = torch.ones(1, 1, image_size, image_size, device=device, dtype=dtype)
        actual = edge_aware_blur_pool2d(inp, kernel_size=kernel_size)
        self.assert_close(actual, inp)

    def test_exception(self):
        from kornia.core.exceptions import BaseError, ShapeError

        data = torch.rand(1, 3, 3)
        with pytest.raises(ShapeError) as errinfo:
            edge_aware_blur_pool2d(data, 3)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)
        data = torch.rand(1, 1, 3, 3)
        with pytest.raises(BaseError) as errinfo:
            edge_aware_blur_pool2d(data, 3, edge_threshold=-1)
        assert "edge_threshold must be greater than 1. Got -1" in str(errinfo.value)

    @pytest.mark.parametrize("edge_threshold", [0.5, 1, 1.0], ids=["half", "int_one", "float_one"])
    def test_convention_edge_threshold_must_be_greater_than_1_5169(self, edge_threshold, device, dtype):
        # The threshold is an intensity ratio compared through log2: below 1 every pixel is an edge and the blur never
        # runs, and at 1 any difference between pixels 4 apart is an edge. The function and the module reject both,
        # and the bound is strict.
        data = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)
        with pytest.raises(BaseError, match=f"edge_threshold must be greater than 1. Got {edge_threshold}"):
            edge_aware_blur_pool2d(data, 3, edge_threshold=edge_threshold)
        with pytest.raises(BaseError, match=f"edge_threshold must be greater than 1. Got {edge_threshold}"):
            EdgeAwareBlurPool2D(3, edge_threshold=edge_threshold)(data)
        # Only the accepted threshold reaches the reflect padding; the rejections above run everywhere.
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        assert edge_aware_blur_pool2d(data, 3, edge_threshold=1.0001).shape == data.shape

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

    @pytest.mark.parametrize("kernel_size", [1, 2, 3, 4, 5, 6, 7, 8, 9, 15])
    @pytest.mark.parametrize("shape", [(17, 19), (5, 6)])
    def test_matches_reflect_padded_blur_without_edges_5228(self, kernel_size, shape, device, dtype):
        # With edge_threshold=1e6 no pixel is an edge, so the output is the stride-1 blur of the input reflect-padded
        # far enough for the kernel, cropped back to the input size: no zero padding reaches the image (#5228). On a
        # 5 x 6 image the larger kernels reach past the reflected copy, which is reflected again, as numpy pads.
        rows = torch.arange(shape[0], dtype=torch.float64)[:, None]
        cols = torch.arange(shape[1], dtype=torch.float64)[None, :]
        data = (0.5 + 0.25 * torch.sin(0.7 * rows + 1.3 * cols))[None, None]
        pad = kernel_size
        padded = torch.from_numpy(np.pad(data.numpy(), ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect"))
        expected = _zero_padded_reference(padded, kernel_size, 1)[..., pad:-pad, pad:-pad]
        actual = edge_aware_blur_pool2d(data.to(device=device, dtype=dtype), kernel_size, edge_threshold=1e6)
        self.assert_close(actual, expected.to(device=device, dtype=dtype))


class TestConventionsBlurPool(BaseTester):
    """Pins for the sampling, padding and anchor of the blur pools, and for their filed defects."""

    def test_convention_blur_pool2d_odd_kernel_samples_every_stride_th_pixel_from_0(self, device, dtype):
        # For an odd kernel, blur_pool2d zero-pads (k - 1) // 2 per side, blurs, and keeps rows and columns 0, s, 2s,
        # ... of the stride-1 blur: the output is ceil(H / s) x ceil(W / s). pyrdown instead resamples between pixels
        # to floor(H / 2) x floor(W / 2).
        torch.manual_seed(0)
        image = torch.rand(1, 2, 7, 10).to(device=device, dtype=dtype)
        for kernel_size in (3, 5):
            dense = blur_pool2d(image, kernel_size, stride=1)
            assert dense.shape == image.shape
            for stride in (2, 3):
                out = blur_pool2d(image, kernel_size, stride=stride)
                assert out.shape[-2:] == (math.ceil(7 / stride), math.ceil(10 / stride))
                self.assert_close(out, dense[..., ::stride, ::stride])
                self.assert_close(BlurPool2D(kernel_size, stride=stride)(image), out)

    def test_convention_blur_pool2d_zero_pads_the_border(self, device, dtype):
        # blur_pool2d pads with zeros, so a constant map darkens along its border: each padded side drops the outer
        # quarter of the binomial [1, 2, 1] / 4 weights. antialiased-cnns' BlurPool reflect-pads by default and
        # matches kornia with pad_type='zero'.
        # Snippet used to generate expected:
        #   out = blur_pool2d(torch.ones(1, 1, 7, 10), 3); print(out[0, 0, 0], out[0, 0, :, 0])
        ones = torch.ones(1, 1, 7, 10, device=device, dtype=dtype)
        out = blur_pool2d(ones, 3)
        self.assert_close(out[0, 0, 0], torch.tensor([0.5625, 0.75, 0.75, 0.75, 0.75], device=device, dtype=dtype))
        self.assert_close(out[0, 0, :, 0], torch.tensor([0.5625, 0.75, 0.75, 0.5625], device=device, dtype=dtype))
        self.assert_close(out[0, 0, 1:-1, 1:], torch.ones(2, 4, device=device, dtype=dtype))

    def test_convention_blur_pool2d_even_kernel_is_anchored_at_k_minus_1_over_2(self, device, dtype):
        # An even kernel's window covers [i - (k - 1) // 2, i + k // 2] -- the filters convention (filter2d, box_blur)
        # and antialiased-cnns' anchor -- so a delta off the centre lights the rows and columns below.
        # Snippet used to generate expected:
        #   x = torch.zeros(1, 1, 7, 9); x[0, 0, 3, 4] = 1
        #   nz = blur_pool2d(x, k, stride=1)[0, 0].nonzero(); print(nz[:, 0].unique(), nz[:, 1].unique())
        image = torch.zeros(1, 1, 7, 9, device=device, dtype=dtype)
        image[0, 0, 3, 4] = 1.0
        for kernel_size, rows, cols in (
            (2, [2, 3], [3, 4]),
            (3, [2, 3, 4], [3, 4, 5]),
            (4, [1, 2, 3, 4], [2, 3, 4, 5]),
        ):
            lit = blur_pool2d(image, kernel_size, stride=1)[0, 0].nonzero()
            assert lit[:, 0].unique().tolist() == rows
            assert lit[:, 1].unique().tolist() == cols

    def test_convention_max_blur_pool2d_is_a_stride_one_max_pool_then_blur_pool2d(self, device, dtype):
        # max_blur_pool2d(x, k, stride, max_pool_size) = blur_pool2d(F.max_pool2d(x, max_pool_size, stride=1), k,
        # stride): the max pool never strides, so the output is ceil((H - max_pool_size + 1) / stride), not H / stride.
        torch.manual_seed(0)
        image = torch.rand(1, 2, 7, 10).to(device=device, dtype=dtype)
        for kernel_size, max_pool_size, size in ((3, 2, (3, 5)), (3, 3, (3, 4)), (5, 2, (3, 5)), (4, 2, (3, 5))):
            out = max_blur_pool2d(image, kernel_size, stride=2, max_pool_size=max_pool_size)
            assert out.shape[-2:] == size
            self.assert_close(out, blur_pool2d(F.max_pool2d(image, max_pool_size, stride=1), kernel_size, stride=2))
            self.assert_close(MaxBlurPool2D(kernel_size, 2, max_pool_size)(image), out)

    def test_convention_edge_aware_blur_pool2d_edge_is_an_intensity_ratio(self, device, dtype):
        # A pixel keeps its value where the channel mean of log2(I(x + 2) / I(x - 2)), along x or y, exceeds
        # log2(edge_threshold) in magnitude (dilated by edge_dilation_kernel_size), and is blurred elsewhere: the
        # threshold is a ratio of intensities 4 px apart, so the decision is scale-invariant for intensities well
        # above epsilon. The comparison is strict. Positive intensities are assumed: a negative pixel's log is NaN,
        # which never counts as an edge.
        # The default edge_threshold is 1.25, and the output keeps the input size. On a vertical step between
        # columns 5 and 6 the blur would move those two columns by a quarter of the step.
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")

        def moved(step, out):
            return (out[..., 5:7] - step[..., 5:7]).abs().max()

        for low, high, kept in ((1.0, 1.2, False), (1.0, 1.3, True), (10.0, 12.0, False), (10.0, 13.0, True)):
            step = torch.full((1, 1, 9, 12), low, device=device, dtype=dtype)
            step[..., 6:] = high
            out = edge_aware_blur_pool2d(step, 3)
            assert out.shape == step.shape
            if kept:
                self.assert_close(out[..., 5:7], step[..., 5:7])
            else:
                assert moved(step, out) > (high - low) / 8
        # a ratio of 1.3 in one channel of two averages below the threshold
        step = torch.ones(1, 2, 9, 12, device=device, dtype=dtype)
        step[:, 0, :, 6:] = 1.3
        assert moved(step, edge_aware_blur_pool2d(step, 3)) > 0.3 / 8
        # a negative step with the same ratio is blurred, and no NaN reaches the output
        step = torch.full((1, 1, 9, 12), -1.0, device=device, dtype=dtype)
        step[..., 6:] = -1.3
        out = edge_aware_blur_pool2d(step, 3)
        assert moved(step, out) > 0.3 / 8
        assert not out.isnan().any()
        # strict: with epsilon=0 a 1 -> 2 step has a log2 ratio of exactly 1 = log2(2), which is not an edge at
        # edge_threshold=2 and is one just below it
        step = torch.ones(1, 1, 9, 12, device=device, dtype=dtype)
        step[..., 6:] = 2.0
        assert moved(step, edge_aware_blur_pool2d(step, 3, edge_threshold=2.0, epsilon=0.0)) > 1 / 8
        self.assert_close(edge_aware_blur_pool2d(step, 3, edge_threshold=1.99, epsilon=0.0)[..., 5:7], step[..., 5:7])

    def test_convention_edge_aware_blur_pool2d_compares_pixels_two_apart_and_dilates_the_edges(self, device, dtype):
        # The edge test at x compares x - 2 with x + 2 (and y - 2 with y + 2), and edge_dilation_kernel_size widens
        # the kept band. Period-2 stripes, 1.0 | 1.15, scaled by 1.5 from column 8 on: pixels two apart match away
        # from the step, so only columns 6..9 see the 1.5 ratio, and the stripes make every blurred column differ
        # from the input. Transposing the image moves the band to rows 6..9.
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        step = (1.0 + 0.15 * (torch.arange(16) % 2)).expand(1, 1, 9, 16).to(device=device, dtype=dtype).contiguous()
        step[..., 8:] *= 1.5
        for dilation, kept in ((1, [6, 7, 8, 9]), (3, [5, 6, 7, 8, 9, 10])):
            out = edge_aware_blur_pool2d(step, 3, edge_dilation_kernel_size=dilation)
            assert [c for c in range(16) if torch.equal(out[..., c], step[..., c])] == kept
        transposed = step.transpose(-1, -2).contiguous()
        out = edge_aware_blur_pool2d(transposed, 3, edge_dilation_kernel_size=1)
        assert [r for r in range(16) if torch.equal(out[..., r, :], transposed[..., r, :])] == [6, 7, 8, 9]

    def test_convention_blur_pool2d_even_kernel_keeps_ceil_h_over_stride_5166(self, device, dtype):
        """An even kernel pads (k - 1) // 2 before and k // 2 after, so the blur pools keep ceil(H / stride) (#5166)."""
        torch.manual_seed(0)
        image = torch.rand(1, 2, 7, 10).to(device=device, dtype=dtype)
        for kernel_size in (2, 4):
            dense = blur_pool2d(image, kernel_size, stride=1)
            assert dense.shape == image.shape
            assert BlurPool2D(kernel_size, stride=1)(image).shape == image.shape
            assert max_blur_pool2d(image, kernel_size, stride=1, max_pool_size=1).shape == image.shape
            for stride in (2, 3):
                out = blur_pool2d(image, kernel_size, stride=stride)
                assert out.shape[-2:] == (math.ceil(7 / stride), math.ceil(10 / stride))
                self.assert_close(out, dense[..., ::stride, ::stride])

    @pytest.mark.parametrize("kernel_size", [2, 4])
    def test_convention_edge_aware_blur_pool2d_even_kernel_size_keeps_the_input_shape_5163(
        self, kernel_size, device, dtype
    ):
        """edge_aware_blur_pool2d blurs with an even kernel_size as blur_pool2d does and keeps the shape (#5163)."""
        # With edge_threshold=1e6 no pixel is an edge, so the output is the stride-1 blur_pool2d of the input
        # reflect-padded by 2, cropped back: an even kernel is anchored at (k - 1) // 2 there too.
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        image = torch.rand(1, 1, 8, 9).to(device=device, dtype=dtype) + 0.1
        out = edge_aware_blur_pool2d(image, kernel_size, edge_threshold=1e6)
        assert out.shape == image.shape
        self.assert_close(EdgeAwareBlurPool2D(kernel_size, edge_threshold=1e6)(image), out)
        padded = F.pad(image.cpu().double(), (2, 2, 2, 2), mode="reflect")
        expected = blur_pool2d(padded, kernel_size, stride=1)[..., 2:-2, 2:-2]
        self.assert_close(out, expected.to(device=device, dtype=dtype))

    def test_convention_edge_aware_blur_pool2d_keeps_a_constant_image_at_any_kernel_size_5228(self, device, dtype):
        """edge_aware_blur_pool2d reflects the border as far as its kernel reaches: a constant image stays (#5228)."""
        # A 7-tap binomial reaches 3 px past the border and a 15-tap one 7 px, past the 2 px the edge test reads.
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        ones = torch.ones(1, 1, 15, 18, device=device, dtype=dtype)
        for kernel_size in (3, 5, 7, 9, 15):
            self.assert_close(edge_aware_blur_pool2d(ones, kernel_size), ones)

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

from __future__ import annotations

import math
import re
from itertools import product

import pytest
import torch
import torch.nn.functional as F

from kornia.core.exceptions import ShapeError
from kornia.filters import (
    EdgeAwareBlurPool2D,
    MaxBlurPool2D,
    bilateral_blur,
    blur_pool2d,
    box_blur,
    edge_aware_blur_pool2d,
    fft_conv,
    filter2d,
    filter3d,
    gaussian_blur2d,
    joint_bilateral_blur,
    laplacian,
    max_blur_pool2d,
    median_blur,
)
from kornia.filters.filter import _pad
from kornia.geometry.transform import build_pyramid

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding, supports_replicate_padding_3d


class TestSpatialPadding(BaseTester):
    @pytest.mark.parametrize(
        "border_type, row",
        [
            ("reflect", [4, 1, 4, 9, 4, 1, 4, 9, 4, 1, 4, 9, 4, 1]),
            ("circular", [1, 4, 9, 1, 4, 9, 1, 4, 9, 1, 4, 9, 1, 4]),
        ],
    )
    @pytest.mark.parametrize("transpose", [False, True])
    def test_opencv_asymmetric_multi_period_border(self, device, dtype, border_type, row, transpose):
        # OpenCV 4.12: copyMakeBorder([[1., 4., 9.]], 2, 1, 9, 2, BORDER_REFLECT_101 / BORDER_WRAP).
        image = torch.tensor([[[[1, 4, 9]]]], device=device, dtype=dtype)
        expected = torch.tensor(row, device=device, dtype=dtype).repeat(4, 1)[None, None]
        padding = [9, 2, 2, 1]
        if transpose:
            image, expected = image.transpose(-2, -1), expected.transpose(-2, -1)
            padding = [2, 1, 9, 2]
        self.assert_close(_pad(image, padding, border_type), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("border_type", ["reflect", "circular"])
    def test_single_pixel_border(self, device, dtype, border_type):
        image = torch.full((1, 1, 1, 1), 0.5, device=device, dtype=dtype)
        expected = image.expand(1, 1, 6, 10)
        self.assert_close(_pad(image, [4, 5, 2, 3], border_type), expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "border_type, rows",
        [
            ("reflect", [[3, 4, 5, 4, 3, 4, 5, 4, 3], [0, 1, 2, 1, 0, 1, 2, 1, 0]]),
            ("circular", [[5, 3, 4, 5, 3, 4, 5, 3, 4], [2, 0, 1, 2, 0, 1, 2, 0, 1]]),
        ],
    )
    def test_three_dimensional_extension(self, device, dtype, border_type, rows):
        # Reflect101 repeats the endpoints every 2*(n-1) steps; wrap repeats every n steps.
        image = torch.arange(6, device=device, dtype=dtype).reshape(1, 1, 2, 1, 3)
        expected = torch.tensor(rows * 4, device=device, dtype=dtype)[:, None, :].expand(8, 4, 9)[None, None]
        self.assert_close(_pad(image, [4, 2, 2, 1, 5, 1], border_type), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("border_type", ["constant", "replicate", "reflect", "circular"])
    @pytest.mark.parametrize("layout", ["contiguous", "transposed", "channels_last", "volume", "cropped"])
    def test_accepted_native_padding_is_bitwise_unchanged(self, device, dtype, border_type, layout):
        image = torch.arange(240, device=device, dtype=dtype).reshape(2, 3, 4, 10) / 256
        padding = [2, 1, 1, 2]
        if layout == "transposed":
            image = image.transpose(-2, -1)
        elif layout == "channels_last":
            image = image.contiguous(memory_format=torch.channels_last)
        elif layout == "volume":
            image = image.reshape(2, 3, 2, 4, 5)
            padding = [2, 1, 1, 2, 1, 1]
        elif layout == "cropped":
            padding = [-1, 1, 0, 0]
        try:
            expected = F.pad(image, padding, mode=border_type)
        except RuntimeError as native:
            with pytest.raises(type(native), match=f"^{re.escape(str(native))}$"):
                _pad(image, padding, border_type)
        else:
            self.assert_close(_pad(image, padding, border_type), expected, rtol=0, atol=0)

    def test_circular_padding_equal_to_axis_keeps_native_path(self, device, dtype):
        image = torch.arange(6, device=device, dtype=dtype).reshape(1, 1, 2, 3)
        padding = [3, 3, 2, 2]
        self.assert_close(_pad(image, padding, "circular"), F.pad(image, padding, mode="circular"), rtol=0, atol=0)

    @pytest.mark.parametrize("border_type", ["reflect", "circular"])
    @pytest.mark.parametrize(
        "shape, padding",
        [((1, 1, 0, 3), [1, 1, 1, 1]), ((1, 1, 1, 2, 3), [4, 4, 0, 0]), ((1, 1, 2, 3), [1, 2, 3])],
    )
    def test_backend_errors_are_preserved(self, device, dtype, border_type, shape, padding):
        image = torch.empty(shape, device=device, dtype=dtype)
        with pytest.raises((RuntimeError, NotImplementedError)) as native:
            F.pad(image, padding, mode=border_type)
        with pytest.raises(type(native.value)) as actual:
            _pad(image, padding, border_type)
        assert str(actual.value) == str(native.value)

    @pytest.mark.parametrize("border_type", ["reflect", "circular"])
    def test_gradcheck(self, device, border_type):
        image = torch.tensor([[[[0.0, 0.25, 1.0]]]], device=device)
        self.gradcheck(lambda x: _pad(x, [5, 4, 2, 1], border_type), (image,))


class TestThinFilterBorders(BaseTester):
    @pytest.mark.parametrize(
        "image, expected",
        [
            ([[0.0, 0.25, 1.0]], [[0.23124580402050804, 0.3732346387756791, 0.5222849184281341]]),
            (
                [[0.0, 0.25, 1.0], [0.75, 0.5, 0.125]],
                [
                    [0.392935850731705, 0.4205353965804977, 0.45084531427081853],
                    [0.3975683362351059, 0.42189058250126177, 0.4487985405988523],
                ],
            ),
        ],
    )
    @pytest.mark.parametrize("separable", [False, True])
    def test_gaussian_matches_opencv(self, device, dtype, image, expected, separable):
        # cv2.GaussianBlur(np.array(image, np.float64), (7, 7), 1., sigmaY=1., borderType=cv2.BORDER_REFLECT_101).
        image = torch.tensor(image, device=device, dtype=dtype)[None, None]
        expected = torch.tensor(expected, device=device, dtype=dtype)[None, None]
        self.assert_close(gaussian_blur2d(image, 7, (1.0, 1.0), separable=separable), expected)

    @pytest.mark.parametrize("border_type, expected", [("reflect", [94, 77, 46]), ("circular", [99, 53, 58])])
    def test_even_kernel_keeps_its_anchor(self, device, dtype, border_type, expected):
        if border_type == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        # OpenCV filter2D on the bordered image, anchor=(1, 0), cropped to the original width.
        image = torch.tensor([[[[1, 4, 9]]]], device=device, dtype=dtype)
        kernel = torch.tensor([[[1, 2, 4, 8]]], device=device, dtype=dtype)
        expected = torch.tensor(expected, device=device, dtype=dtype).reshape_as(image)
        self.assert_close(filter2d(image, kernel, border_type), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("border_type", ["constant", "replicate", "reflect", "circular"])
    @pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 1), (1, 2, 3), (3, 1, 1)])
    @pytest.mark.parametrize("size", [3, 7])
    def test_normalized_filter_on_thin_constant_input(self, device, dtype, border_type, shape, size):
        if border_type == "replicate":
            supported = supports_replicate_padding if len(shape) == 2 else supports_replicate_padding_3d
            if not supported(device, dtype):
                pytest.skip("native replicate padding is unavailable for this device/dtype")
        if (
            border_type == "reflect"
            and len(shape) == 2
            and size // 2 < min(shape)
            and not supports_reflect_padding(device, dtype)
        ):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        image = torch.full((1, 1, *shape), 0.5, device=device, dtype=dtype)
        kernel = torch.ones((1, *((size,) * len(shape))), device=device, dtype=dtype)
        op = filter2d if len(shape) == 2 else filter3d
        output = op(image, kernel, border_type, normalized=True)
        expected = torch.full_like(image, 0.5)
        if border_type == "constant":
            radius = size // 2
            for index in product(*(range(side) for side in shape)):
                counts = [min(side, i + radius + 1) - max(0, i - radius) for i, side in zip(index, shape)]
                expected[(0, 0, *index)] = 0.5 * math.prod(counts) / size ** len(shape)
        self.assert_close(output, expected)

    @pytest.mark.parametrize("border_type", ["reflect", "circular"])
    @pytest.mark.parametrize(
        "op, kwargs, expected",
        [
            (fft_conv, {"kernel": torch.ones(1, 7, 7), "normalized": True}, 0.5),
            (box_blur, {"kernel_size": 7}, 0.5),
            (median_blur, {"kernel_size": 7}, 0.5),
            (laplacian, {"kernel_size": 7}, 0.0),
            (bilateral_blur, {"kernel_size": 7, "sigma_color": 0.2, "sigma_space": (1.0, 1.0)}, 0.5),
        ],
        ids=["fft", "box", "median", "laplacian", "bilateral"],
    )
    def test_public_filter_families(self, device, dtype, border_type, op, kwargs, expected):
        image = torch.full((1, 1, 1, 3), 0.5, device=device, dtype=dtype)
        self.assert_close(op(image, border_type=border_type, **kwargs), torch.full_like(image, expected))
        if op is bilateral_blur:
            self.assert_close(joint_bilateral_blur(image, image.clone(), border_type=border_type, **kwargs), image)

    @pytest.mark.parametrize("shape", [(1, 7), (7, 1), (1, 1)])
    @pytest.mark.parametrize("size", [3, 7])
    def test_edge_halo_and_blur_halo_accept_singleton_axes(self, device, dtype, shape, size):
        image = torch.ones((1, 1, *shape), device=device, dtype=dtype)
        self.assert_close(edge_aware_blur_pool2d(image, size), image, rtol=0, atol=0)
        self.assert_close(EdgeAwareBlurPool2D(size)(image), image, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", [(1, 7), (7, 1), (1, 1)])
    def test_max_blur_pool_names_minimum_size(self, device, dtype, shape):
        image = torch.ones((1, 1, *shape), device=device, dtype=dtype)
        with pytest.raises(ShapeError, match="at least 2 pixels"):
            max_blur_pool2d(image, 3)
        with pytest.raises(ShapeError, match="at least 2 pixels"):
            MaxBlurPool2D(3)(image)

    def test_unit_max_pool_still_accepts_singleton_axes(self, device, dtype):
        image = torch.ones((1, 1, 1, 3), device=device, dtype=dtype)
        expected = blur_pool2d(image, 3)
        self.assert_close(max_blur_pool2d(image, 3, max_pool_size=1), expected, rtol=0, atol=0)
        self.assert_close(MaxBlurPool2D(3, max_pool_size=1)(image), expected, rtol=0, atol=0)

    def test_empty_batch_retains_input_and_kernel_gradients(self, device, dtype):
        image = torch.empty((0, 2, 1, 3), device=device, dtype=dtype, requires_grad=True)
        kernel = torch.ones((1, 7, 7), device=device, dtype=dtype, requires_grad=True)
        output = filter2d(image, kernel)
        assert output.shape == image.shape
        output.sum().backward()
        assert image.grad is not None
        assert kernel.grad is not None
        self.assert_close(kernel.grad, torch.zeros_like(kernel), rtol=0, atol=0)

    def test_pyramid_reaches_one_pixel_level(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        image = torch.full((1, 1, 640, 640), 0.5, device=device, dtype=dtype)
        pyramid = build_pyramid(image, 10)
        assert [level.shape[-2:] for level in pyramid] == [(n, n) for n in (640, 320, 160, 80, 40, 20, 10, 5, 2, 1)]
        for level in pyramid:
            self.assert_close(level, torch.full_like(level, 0.5), rtol=0, atol=0)

    @pytest.mark.parametrize("border_type", ["reflect", "circular"])
    def test_dynamo_fullgraph_thin_filter(self, device, dtype, border_type, optimizer_backend, torch_optimizer):
        if optimizer_backend in (None, "", "jit"):
            pytest.skip("this check requires a torch.compile backend")
        image = torch.tensor([[[[0.0, 0.25, 1.0]]]], device=device, dtype=dtype)
        kernel = torch.ones((1, 7, 7), device=device, dtype=dtype)

        def op(x):
            return filter2d(x, kernel, border_type, normalized=True)

        self.assert_close(torch_optimizer(op, fullgraph=True)(image), op(image))

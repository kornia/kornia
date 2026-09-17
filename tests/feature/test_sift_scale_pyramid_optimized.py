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
# limitations under the License.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kornia.feature.sift.scale_space import _SIFTScalePyramid

from testing.base import (
    DYNAMO_UNAVAILABLE_REASON,
    BaseTester,
    dynamo_is_available,
    supports_reflect_padding,
    supports_replicate_padding,
)


def _convolution_blur(pyramid: _SIFTScalePyramid, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    radius = kernel.numel() // 2
    horizontal = F.conv2d(pyramid._reflect_pad(image, radius, True), kernel.view(1, 1, 1, -1))
    return F.conv2d(pyramid._reflect_pad(horizontal, radius, False), kernel.view(1, 1, -1, 1))


def _convolution_pyramid(pyramid: _SIFTScalePyramid, image: torch.Tensor) -> list[torch.Tensor]:
    first = _convolution_blur(pyramid, pyramid._double(image), pyramid.kernel_0.to(image))
    result = []
    while True:
        levels = [first]
        for index in range(1, 6):
            levels.append(_convolution_blur(pyramid, levels[-1], getattr(pyramid, f"kernel_{index}").to(image)))
        result.append(torch.stack(levels, dim=2))
        height, width = first.shape[-2:]
        if min(height // 2, width // 2) < 12:
            return result
        first = levels[3][..., : 2 * (height // 2) : 2, : 2 * (width // 2) : 2]


class TestSIFTScalePyramidCPUOptimized(BaseTester):
    @staticmethod
    def _cpu(device: torch.device, dtype: torch.dtype) -> None:
        if device.type != "cpu":
            pytest.skip("the optimized implementation is CPU-only")
        if not supports_replicate_padding(device, dtype) or not supports_reflect_padding(device, dtype):
            pytest.skip("direct pyramid helper requires native border padding; the detector promotes half inputs")

    @pytest.mark.parametrize("shape", [(1, 1), (2, 3)])
    def test_blur_matches_convolution_with_asymmetric_kernel(self, device, dtype, shape):
        self._cpu(device, dtype)
        pyramid = _SIFTScalePyramid().to(device, dtype)
        image = torch.rand(1, 1, *shape, device=device, dtype=dtype)
        kernel = torch.rand_like(pyramid.kernel_4)

        actual = (
            pyramid._blur_cpu(image, kernel, kernel.numel() // 2)
            if dtype in (torch.float32, torch.float64)
            else pyramid._blur(image, kernel)
        )
        expected = _convolution_blur(pyramid, image, kernel)

        self.assert_close(actual, expected)

    def test_pyramid_matches_convolution(self, device, dtype):
        self._cpu(device, dtype)
        pyramid = _SIFTScalePyramid().to(device, dtype)
        # The doubled first octave exceeds the CPU slice threshold; subsequent
        # octaves exercise the convolution fallback for smaller arrays.
        image = torch.rand(1, 1, 129, 131, device=device, dtype=dtype)

        actual = pyramid(image)
        expected = _convolution_pyramid(pyramid, image)

        for actual_octave, expected_octave in zip(actual, expected):
            self.assert_close(actual_octave, expected_octave)

    def test_blur_has_finite_gradient(self, device, dtype):
        self._cpu(device, dtype)
        pyramid = _SIFTScalePyramid().to(device, dtype)
        actual_input = torch.rand(1, 1, 17, 19, device=device, dtype=dtype, requires_grad=True)
        expected_input = actual_input.detach().clone().requires_grad_()
        kernel = pyramid.kernel_4

        actual = (
            pyramid._blur_cpu(actual_input, kernel, kernel.numel() // 2)
            if dtype in (torch.float32, torch.float64)
            else pyramid._blur(actual_input, kernel)
        )
        expected = _convolution_blur(pyramid, expected_input, kernel)
        actual.square().mean().backward()
        expected.square().mean().backward()

        assert actual_input.grad is not None and torch.isfinite(actual_input.grad).all()
        self.assert_close(actual_input.grad, expected_input.grad)

    def test_dynamo(self, device, dtype, torch_optimizer):
        self._cpu(device, dtype)
        pyramid = _SIFTScalePyramid().to(device, dtype)
        image = torch.rand(1, 1, 17, 19, device=device, dtype=dtype)
        expected = pyramid(image)

        for actual_octave, expected_octave in zip(torch_optimizer(pyramid)(image), expected):
            self.assert_close(actual_octave, expected_octave)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_dynamo_fullgraph(self, device, dtype):
        self._cpu(device, dtype)
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("the CPU slice implementation uses full precision")
        pyramid = _SIFTScalePyramid().to(device, dtype)
        image = torch.rand(1, 1, 17, 19, device=device, dtype=dtype)
        kernel = pyramid.kernel_4
        radius = kernel.numel() // 2
        expected = pyramid._blur_cpu(image, kernel, radius)
        torch._dynamo.reset()
        actual = torch.compile(pyramid._blur_cpu, fullgraph=True)(image, kernel, radius)
        self.assert_close(actual, expected)

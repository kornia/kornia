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

import importlib

import pytest
import torch

from kornia.filters import MedianBlur, median_blur
from kornia.filters.kernels import get_binary_kernel2d

from testing.base import BaseTester

median_module = importlib.import_module("kornia.filters.median")


class TestMedianBlur(BaseTester):
    def test_smoke(self, device, dtype):
        inp = torch.zeros(1, 3, 4, 4, device=device, dtype=dtype)
        actual = median_blur(inp, 3)
        assert isinstance(actual, torch.Tensor)

    @pytest.mark.parametrize("batch_size", [0, 1, 2])
    @pytest.mark.parametrize("channels", [0, 3])
    @pytest.mark.parametrize("kernel_size", [3, (5, 7)])
    def test_cardinality(self, batch_size, channels, kernel_size, device, dtype):
        inp = torch.zeros(batch_size, channels, 4, 4, device=device, dtype=dtype)
        actual = median_blur(inp, kernel_size)
        assert actual.shape == (batch_size, channels, 4, 4)

    def test_exception(self, device, dtype):
        from kornia.core.exceptions import ShapeError, TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            median_blur(1, 1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            median_blur(torch.ones(1, 1, device=device, dtype=dtype), 1)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

    def test_kernel_3x3(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 7.0, 5.0, 0.0],
                    [0.0, 3.0, 1.0, 1.0, 0.0],
                    [0.0, 6.0, 9.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [36.0, 7.0, 25.0, 0.0, 0.0],
                    [3.0, 14.0, 1.0, 0.0, 0.0],
                    [65.0, 59.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)

        kernel_size = (3, 3)
        actual = median_blur(inp, kernel_size)
        self.assert_close(actual[0, 0, 2, 2], torch.tensor(3.0, device=device, dtype=dtype))
        self.assert_close(actual[0, 1, 1, 1], torch.tensor(14.0, device=device, dtype=dtype))

    def test_kernel_3x1(self, device, dtype):
        inp = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 7.0, 5.0, 0.0],
                [0.0, 3.0, 1.0, 1.0, 0.0],
                [0.0, 6.0, 9.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        ).view(1, 1, 5, 5)

        ky, kx = 3, 1
        actual = median_blur(inp, (ky, kx))

        self.assert_close(actual[0, 0, 2, 2], torch.tensor(7.0, device=device, dtype=dtype))
        self.assert_close(actual[0, 0, 1, 1], torch.tensor(3.0, device=device, dtype=dtype))

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = (3, 3)
        actual = median_blur(inp, kernel_size)
        assert actual.is_contiguous()

    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("layout", ["contiguous", "transposed", "channels_last"])
    def test_selection_matches_convolution(self, kernel_size, layout, device, dtype):
        # Quantized signed values exercise ties and padding; the extra channel
        # has random values so both branches also see distinct order statistics.
        inp = torch.randint(-4, 5, (2, 3, 7, 9), device=device).to(dtype)
        inp[:, 0] = torch.rand(2, 7, 9, device=device, dtype=dtype) * 2 - 1
        if layout == "transposed":
            inp = inp.transpose(-1, -2)
        elif layout == "channels_last":
            inp = inp.contiguous(memory_format=torch.channels_last)
        b, c, h, w = inp.shape
        weights = get_binary_kernel2d(kernel_size, device=device, dtype=dtype)
        features = torch.nn.functional.conv2d(inp.reshape(b * c, 1, h, w), weights, padding=kernel_size // 2)
        expected = features.reshape(b, c, kernel_size**2, h, w).median(2).values
        actual = median_blur(inp, kernel_size)
        self.assert_close(actual, expected)
        assert actual.is_contiguous()

    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
    def test_selection_nonfinite(self, kernel_size, invalid, device, dtype):
        inp = torch.ones(1, 1, 7, 9, device=device, dtype=dtype)
        inp[..., 3, 4] = invalid
        weights = get_binary_kernel2d(kernel_size, device=device, dtype=dtype)
        expected = torch.nn.functional.conv2d(inp, weights, padding=kernel_size // 2).median(1).values[:, None]
        actual = median_blur(inp, kernel_size)
        self.assert_close(actual.isnan(), expected.isnan())
        self.assert_close(actual.nan_to_num(), expected.nan_to_num())

    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("shape", [(1, 2, 7, 9), (1, 4, 512, 512)])
    def test_selection_dispatch(self, monkeypatch, kernel_size, shape, device, dtype):
        # CPU always selects; eager CUDA only for 3x3 on inputs large enough to hide launches.
        network = median_module._median_blur_network
        calls = 0

        def counted_network(*args, **kwargs):
            nonlocal calls
            calls += 1
            return network(*args, **kwargs)

        monkeypatch.setattr(median_module, "_median_blur_network", counted_network)
        data = torch.rand(shape, device=device, dtype=dtype)
        median_blur(data, kernel_size)
        expected = device.type == "cpu" or (device.type == "cuda" and kernel_size == 3 and data.numel() >= 1 << 20)
        assert calls == int(expected)

    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_tied_gradients_unchanged(self, kernel_size, device, dtype):
        inp = torch.randint(-2, 3, (1, 1, 7, 9), device=device).to(dtype).requires_grad_()
        weights = get_binary_kernel2d(kernel_size, device=device, dtype=dtype)
        expected = torch.nn.functional.conv2d(inp, weights, padding=kernel_size // 2).median(1).values[:, None]
        expected_grad = torch.autograd.grad(expected.sum(), inp)[0]
        actual = median_blur(inp, kernel_size)
        self.assert_close(torch.autograd.grad(actual.sum(), inp)[0], expected_grad, atol=0, rtol=0)

    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_tied_forward_gradients_unchanged(self, kernel_size, device, dtype):
        inp = torch.randint(-2, 3, (1, 1, 7, 9), device=device).to(dtype)
        tangent = torch.randn_like(inp)
        weights = get_binary_kernel2d(kernel_size, device=device, dtype=dtype)

        def reference(value):
            return torch.nn.functional.conv2d(value, weights, padding=kernel_size // 2).median(1).values[:, None]

        expected, expected_jvp = torch.func.jvp(reference, (inp,), (tangent,))
        actual, actual_jvp = torch.func.jvp(lambda value: median_blur(value, kernel_size), (inp,), (tangent,))
        self.assert_close(actual, expected)
        self.assert_close(actual_jvp, expected_jvp)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
    def test_cpu_autocast(self, autocast_dtype):
        # Float32 inputs intentionally exercise CPU autocast's convolution cast.
        inp = torch.rand(1, 1, 7, 9, dtype=torch.float32)
        weights = get_binary_kernel2d(3, dtype=torch.float32)
        with torch.autocast("cpu", dtype=autocast_dtype):
            expected = torch.nn.functional.conv2d(inp, weights, padding=1).median(1).values[:, None]
            actual = median_blur(inp, 3)
        assert actual.dtype == expected.dtype == autocast_dtype
        self.assert_close(actual, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(median_blur, (img, (5, 3)))

    def test_module(self, device, dtype):
        kernel_size = (3, 5)
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = median_blur
        op_module = MedianBlur((3, 5))
        actual = op_module(img)
        expected = op(img, kernel_size)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("kernel_size", [3, 5, (5, 7)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        data = torch.rand(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = MedianBlur(kernel_size)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_dynamo_nonfinite(self, kernel_size, device, dtype, torch_optimizer):
        data = torch.ones(1, 1, 7, 9, device=device, dtype=dtype)
        data[..., 3, 4] = float("inf")
        op = MedianBlur(kernel_size)
        actual = torch_optimizer(op)(data)
        expected = op(data)
        self.assert_close(actual.isnan(), expected.isnan())
        self.assert_close(actual.nan_to_num(), expected.nan_to_num())

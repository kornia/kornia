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

from kornia.filters import Laplacian, filter2d, get_laplacian_kernel1d, get_laplacian_kernel2d, laplacian
from kornia.filters.kernels import normalize_kernel2d

from testing.base import BaseTester, assert_close

laplacian_module = importlib.import_module("kornia.filters.laplacian")


@pytest.mark.parametrize("window_size", [5, 11])
def test_get_laplacian_kernel1d(window_size, device, dtype):
    actual = get_laplacian_kernel1d(window_size, device=device, dtype=dtype)
    expected = torch.zeros(1, device=device, dtype=dtype)

    assert actual.shape == (window_size,)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("window_size", [5, 11, (3, 3)])
def test_get_laplacian_kernel2d(window_size, device, dtype):
    actual = get_laplacian_kernel2d(window_size, device=device, dtype=dtype)
    expected = torch.zeros(1, device=device, dtype=dtype)
    expected_shape = window_size if isinstance(window_size, tuple) else (window_size, window_size)

    assert actual.shape == expected_shape
    assert_close(actual.sum(), expected.sum())


def test_get_laplacian_kernel1d_exact(device, dtype):
    actual = get_laplacian_kernel1d(5, device=device, dtype=dtype)
    expected = torch.tensor([1.0, 1.0, -4.0, 1.0, 1.0], device=device, dtype=dtype)
    assert_close(expected, actual)


def test_get_laplacian_kernel2d_exact(device, dtype):
    actual = get_laplacian_kernel2d(7, device=device, dtype=dtype)
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, -48.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    assert_close(expected, actual)


class TestLaplacian(BaseTester):
    @pytest.mark.parametrize("kernel_size", [3, 5, (5, 7)])
    @pytest.mark.parametrize("border_type", ["constant", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_matches_dense_kernel(self, kernel_size, border_type, normalized, device, dtype):
        data = torch.rand(2, 3, 11, 13, device=device, dtype=dtype)
        kernel = get_laplacian_kernel2d(kernel_size, device=device, dtype=dtype)[None]
        if normalized:
            kernel = normalize_kernel2d(kernel)

        expected = filter2d(data, kernel, border_type)
        self.assert_close(laplacian(data, kernel_size, border_type, normalized), expected)

    @pytest.mark.parametrize("has_mkldnn", [False, True])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 10), (1, 4, 1024, 1024)])
    def test_slices_dispatch(self, monkeypatch, has_mkldnn, shape, device, dtype):
        # CPU uses slices except on large inputs with oneDNN; eager CUDA keeps cuDNN's convolution.
        monkeypatch.setattr(laplacian_module, "_HAS_MKLDNN", has_mkldnn)
        conv = laplacian_module.filter2d
        calls = 0

        def counted_conv(*args, **kwargs):
            nonlocal calls
            calls += 1
            return conv(*args, **kwargs)

        monkeypatch.setattr(laplacian_module, "filter2d", counted_conv)
        data = torch.rand(shape, device=device, dtype=dtype)
        laplacian(data, 5)
        small = not has_mkldnn or data.numel() < 1 << 22
        uses_slices = device.type == "cpu" and dtype in (torch.float32, torch.float64) and small
        assert calls == int(not uses_slices)

    @pytest.mark.parametrize("value", ["large", "inf", "nan"])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_extreme_cpu_values_match_convolution(self, monkeypatch, value, normalized, device, dtype):
        if device.type != "cpu" or dtype not in (torch.float32, torch.float64):
            pytest.skip("The CPU slice path supports float32/float64")
        data = torch.zeros(1, 1, 7, 9, device=device, dtype=dtype)
        data[..., 3, 4] = torch.finfo(dtype).max / 2 if value == "large" else float(value)
        kernel = get_laplacian_kernel2d(3, device=device, dtype=dtype)[None]
        if normalized:
            kernel = normalize_kernel2d(kernel)
        expected = filter2d(data, kernel, "constant")
        actual = laplacian(data, 3, "constant", normalized)
        self.assert_close(actual.isnan(), expected.isnan())
        self.assert_close(actual.isposinf(), expected.isposinf())
        self.assert_close(actual.isneginf(), expected.isneginf())
        self.assert_close(actual.nan_to_num(), expected.nan_to_num())

    def test_vmap(self, device, dtype):
        data = torch.rand(2, 1, 3, 7, 9, device=device, dtype=dtype)
        expected = torch.stack([laplacian(image, 3) for image in data])
        actual = torch.vmap(lambda image: laplacian(image, 3))(data)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("normalized", [True, False])
    def test_kernel_size_one(self, normalized, device, dtype):
        data = torch.rand(1, 1, 3, 5, device=device, dtype=dtype)
        actual = laplacian(data, 1, normalized=normalized)

        if normalized:
            assert torch.isnan(actual).all()
        else:
            self.assert_close(actual, torch.zeros_like(actual))

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (11, 7), (3, 3)])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_smoke(self, shape, kernel_size, normalized, device, dtype):
        data = torch.rand(shape, device=device, dtype=dtype)
        actual = laplacian(data, kernel_size, "reflect", normalized)
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (11, 7), 3])
    def test_cardinality(self, shape, kernel_size, device, dtype):
        sample = torch.rand(shape, device=device, dtype=dtype)
        actual = laplacian(sample, kernel_size)
        assert actual.shape == shape

    @pytest.mark.skip(reason="Nothing to test.")
    def test_exception(self): ...

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        sample = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = 3
        actual = laplacian(sample, kernel_size)
        assert actual.is_contiguous()

    def test_export(self, device, dtype):
        inp = torch.rand(1, 2, 7, 9, device=device, dtype=dtype)
        op = Laplacian((5, 7))
        exported = torch.export.export(op, (inp,), strict=True)
        self.assert_close(exported.module()(inp), op(inp))

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
    def test_cpu_autocast(self, autocast_dtype):
        # Float32 inputs intentionally exercise CPU autocast's convolution cast.
        inp = torch.rand(1, 1, 7, 9, dtype=torch.float32)
        weights = get_laplacian_kernel2d(3, dtype=torch.float32)[None]
        weights = normalize_kernel2d(weights)
        with torch.autocast("cpu", dtype=autocast_dtype):
            expected = filter2d(inp, weights)
            actual = laplacian(inp, 3)
        assert actual.dtype == expected.dtype == autocast_dtype
        self.assert_close(actual, expected)

    def test_gradcheck(self, device):
        # test parameters
        batch_shape = (1, 2, 5, 7)
        kernel_size = 3

        # evaluate function gradient
        sample = torch.rand(batch_shape, device=device, dtype=torch.float64)
        self.gradcheck(laplacian, (sample, kernel_size))

    def test_module(self, device, dtype):
        params = [3]
        op = laplacian
        op_module = Laplacian(*params)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.parametrize("kernel_size", [5, (5, 7)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = Laplacian(kernel_size)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

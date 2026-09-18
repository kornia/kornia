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

from kornia.filters import BoxBlur, box_blur, filter2d, filter2d_separable
from kornia.filters.kernels import get_box_kernel1d, get_box_kernel2d

from testing.base import BaseTester

blur_module = importlib.import_module("kornia.filters.blur")


class TestBoxBlur(BaseTester):
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_smoke(self, kernel_size, device, dtype):
        data = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)

        bb = BoxBlur(kernel_size, "reflect")
        actual = bb(data)
        assert actual.shape == (1, 1, 10, 10)

    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_separable(self, batch_size, kernel_size, device, dtype):
        data = torch.randn(batch_size, 3, 10, 10, device=device, dtype=dtype)
        out1 = box_blur(data, kernel_size, separable=False)
        out2 = box_blur(data, kernel_size, separable=True)
        self.assert_close(out1, out2)

    @pytest.mark.parametrize("border_type", ["constant", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("kernel_size", [(2, 4), (4, 3), (3, 5), (1, 5), (5, 1), (1, 1)])
    @pytest.mark.parametrize("separable", [False, True])
    def test_pooling_matches_convolution(self, border_type, kernel_size, separable, device, dtype):
        data = torch.randn(2, 3, 8, 9, device=device, dtype=dtype).transpose(-1, -2)

        actual = box_blur(data, kernel_size, border_type, separable=separable)
        if separable:
            ky, kx = kernel_size
            kernel_x = get_box_kernel1d(kx, device=device, dtype=dtype)
            kernel_y = get_box_kernel1d(ky, device=device, dtype=dtype)
            expected = filter2d_separable(data, kernel_x, kernel_y, border_type)
        else:
            kernel = get_box_kernel2d(kernel_size, device=device, dtype=dtype)
            expected = filter2d(data, kernel, border_type)

        self.assert_close(actual, expected)

    @pytest.mark.parametrize("separable", [False, True])
    @pytest.mark.parametrize("border_type", ["reflect", "constant"])
    def test_pooling_nonfinite(self, separable, border_type, device):
        data = torch.zeros(1, 1, 7, 8, device=device)
        data[0, 0, 1, 1] = float("inf")
        data[0, 0, 3, 4] = float("nan")
        data[0, 0, 5, 6] = -float("inf")

        actual = box_blur(data, (3, 5), border_type, separable=separable)
        expected = filter2d(data, get_box_kernel2d((3, 5), device=device), border_type)

        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
        assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))

    @pytest.mark.parametrize("separable", [False, True])
    @pytest.mark.parametrize("value", ["large", "inf", "nan"])
    def test_extreme_cpu_values_match_convolution(self, separable, value, device, dtype):
        if device.type != "cpu":
            pytest.skip("The data-dependent fallback is specific to eager CPU execution")
        fill = torch.finfo(dtype).max / 2 if value == "large" else float(value)
        data = torch.full((1, 1, 7, 9), fill, device=device, dtype=dtype)
        if separable:
            expected = filter2d_separable(
                data,
                get_box_kernel1d(5, device=device, dtype=dtype),
                get_box_kernel1d(3, device=device, dtype=dtype),
                "replicate",
            )
        else:
            expected = filter2d(data, get_box_kernel2d((3, 5), device=device, dtype=dtype), "replicate")
        actual = box_blur(data, (3, 5), "replicate", separable=separable)
        self.assert_close(actual.isnan(), expected.isnan())
        self.assert_close(actual.isposinf(), expected.isposinf())
        self.assert_close(actual.isneginf(), expected.isneginf())
        self.assert_close(actual.nan_to_num(), expected.nan_to_num())
        if value == "large":
            assert actual.isfinite().all()

    def test_vmap(self, device, dtype):
        data = torch.rand(2, 1, 3, 7, 9, device=device, dtype=dtype)
        expected = torch.stack([box_blur(image, 3) for image in data])
        actual = torch.vmap(lambda image: box_blur(image, 3))(data)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("separable", [False, True])
    def test_pooling_cpu_autocast(self, separable):
        data = torch.rand(1, 3, 8, 9)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = box_blur(data, (3, 5), separable=separable)
            module_actual = BoxBlur((3, 5), separable=separable)(data)
            if separable:
                expected = filter2d_separable(data, get_box_kernel1d(5), get_box_kernel1d(3))
            else:
                expected = filter2d(data, get_box_kernel2d((3, 5)))

        assert actual.dtype == torch.bfloat16
        self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(module_actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("separable", [False, True])
    def test_complex_convolution_fallback(self, separable):
        data = torch.rand(1, 3, 8, 9, dtype=torch.complex64)

        actual = box_blur(data, (3, 5), separable=separable)
        kernel = get_box_kernel2d((3, 5), dtype=data.dtype)
        expected = filter2d(data, kernel)

        self.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assert_close(BoxBlur((3, 5), separable=separable)(data), expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("has_mkldnn", [False, True])
    @pytest.mark.parametrize("kernel_size", [5, (3, 7), 9])
    @pytest.mark.parametrize("separable", [False, True])
    @pytest.mark.parametrize("shape", [(1, 2, 12, 13), (1, 4, 1024, 1024)])
    def test_pooling_dispatch(self, monkeypatch, has_mkldnn, kernel_size, separable, shape, device, dtype):
        # Eager oneDNN convolution beats CPU pooling except on large inputs with moderate windows.
        monkeypatch.setattr(blur_module, "_HAS_MKLDNN", has_mkldnn)
        pool = blur_module._box_blur_pool
        calls = 0

        def counted_pool(*args, **kwargs):
            nonlocal calls
            calls += 1
            return pool(*args, **kwargs)

        monkeypatch.setattr(blur_module, "_box_blur_pool", counted_pool)
        data = torch.rand(shape, device=device, dtype=dtype)
        actual = box_blur(data, kernel_size, separable=separable)
        expected = filter2d(data, get_box_kernel2d(kernel_size, device=device, dtype=dtype))
        self.assert_close(actual, expected)
        size = max(kernel_size) if isinstance(kernel_size, tuple) else kernel_size
        onednn_pools = data.numel() >= 1 << 22 and size <= (15 if separable else 5)
        assert calls == int(not (device.type == "cpu" and has_mkldnn) or onednn_pools)

    @pytest.mark.parametrize("has_mkldnn", [False, True])
    @pytest.mark.parametrize("kernel_size,separable", [(3, False), (5, False), (5, True), (7, True)])
    @pytest.mark.parametrize("shape", [(1, 2, 12, 13), (1, 4, 1024, 1024)])
    def test_dynamo_pooling_dispatch(
        self, monkeypatch, has_mkldnn, kernel_size, separable, shape, device, dtype, torch_optimizer
    ):
        monkeypatch.setattr(blur_module, "_HAS_MKLDNN", has_mkldnn)
        pool = blur_module._box_blur_pool
        calls = 0

        def counted_pool(*args, **kwargs):
            nonlocal calls
            calls += 1
            return pool(*args, **kwargs)

        monkeypatch.setattr(blur_module, "_box_blur_pool", counted_pool)
        data = torch.rand(shape, device=device, dtype=dtype)
        compiled = torch_optimizer(lambda x: box_blur(x, kernel_size, separable=separable), fullgraph=True)
        actual = compiled(data)
        expected = filter2d(data, get_box_kernel2d(kernel_size, device=device, dtype=dtype))
        self.assert_close(actual, expected)
        onednn_pools = kernel_size <= (5 if separable else 3)
        assert calls == int(not (device.type == "cpu" and has_mkldnn) or onednn_pools)

    def test_separable_empty_batch(self, device, dtype):
        data = torch.empty(0, 3, 8, 9, device=device, dtype=dtype)

        actual = box_blur(data, (3, 5), separable=True)

        assert actual.shape == data.shape

    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_separable_module(self, kernel_size, device, dtype):
        data = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)
        blur_sep = BoxBlur(kernel_size, "reflect", separable=True)
        blur_ref = BoxBlur(kernel_size, "reflect", separable=False)
        self.assert_close(blur_sep(data), blur_ref(data))

    def test_default_is_separable(self, device, dtype):
        data = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)

        actual = box_blur(data, 5)
        expected = box_blur(data, 5, separable=True)

        assert torch.equal(actual, expected)

    @pytest.mark.parametrize("separable", [False, True])
    @pytest.mark.parametrize("legacy_separable", [False, True])
    @pytest.mark.parametrize("nested", [False, True])
    def test_legacy_state_dict(self, separable, legacy_separable, nested, device, dtype):
        module = BoxBlur(3, separable=separable)
        module = torch.nn.Sequential(module) if nested else module
        prefix = "0." if nested else ""
        # These are no longer parameters of the operation. Even modified
        # legacy weights are accepted and ignored, as documented in the migration.
        keys = ("kernel_x", "kernel_y") if legacy_separable else ("kernel",)
        state = {prefix + name: torch.zeros(1, 3, 3) for name in keys}
        module.load_state_dict(state, strict=True)
        data = torch.ones(1, 1, 5, 5, device=device, dtype=dtype)
        self.assert_close(module(data), data)
        assert module.state_dict() == {}
        with pytest.raises(RuntimeError, match="Unexpected key"):
            module.load_state_dict({prefix + "unrelated": torch.zeros(1)}, strict=True)

    def test_exception(self):
        data = torch.rand(1, 1, 3, 3)

        with pytest.raises(Exception) as errinfo:
            box_blur(data, (1,))
        assert "2D Kernel size should have a length of 2." in str(errinfo)

    @pytest.mark.parametrize("kernel_size", [(3, 3), 5, (5, 7)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, batch_size, kernel_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)
        blur = BoxBlur(kernel_size)
        actual = blur(inp)
        expected = (batch_size, 3, 4, 4)
        assert actual.shape == expected

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_kernel_3x3(self, batch_size, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1, 1)

        kernel_size = (3, 3)
        actual = box_blur(inp, kernel_size)
        expected = torch.tensor(35.0 * batch_size, device=device, dtype=dtype)

        self.assert_close(actual.sum(), expected)

    @pytest.mark.parametrize("batch_size", [None, 1, 3])
    def test_kernel_5x5(self, batch_size, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        if batch_size:
            inp = inp.repeat(batch_size, 1, 1, 1)

        kernel_size = (5, 5)

        actual = box_blur(inp, kernel_size)
        expected = inp.sum((1, 2, 3)) / torch.mul(*kernel_size)

        self.assert_close(actual[:, 0, 2, 2], expected)

    def test_kernel_3x1(self, device, dtype):
        inp = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)

        ky, kx = 3, 1
        actual = box_blur(inp, (ky, kx))

        self.assert_close(actual[0, 0, 0, 0], torch.tensor((4 + 0 + 4) / 3, device=device, dtype=dtype))
        self.assert_close(actual[0, 0, 1, 0], torch.tensor((0 + 4 + 8) / 3, device=device, dtype=dtype))

    @pytest.mark.parametrize("separable", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_noncontiguous(self, batch_size, separable, device, dtype):
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = box_blur(inp, 3, separable=separable)

        assert actual.is_contiguous()

    @pytest.mark.parametrize("kernel_size", [(3, 3), 5, (5, 7)])
    @pytest.mark.parametrize("separable", [False, True])
    @pytest.mark.parametrize("border_type", ["reflect", "constant"])
    def test_gradcheck(self, kernel_size, separable, border_type, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        fast_mode = "cpu" in str(device)  # Disable fast mode for GPU
        self.gradcheck(
            lambda x: box_blur(x, kernel_size, border_type, separable=separable), (img,), fast_mode=fast_mode
        )

    @pytest.mark.parametrize("kernel_size", [(3, 3), 5, (5, 7)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_module(self, kernel_size, batch_size, device, dtype):
        op = box_blur
        op_module = BoxBlur

        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)

        self.assert_close(actual, expected)

    @pytest.mark.parametrize("separable", [False, True])
    @pytest.mark.parametrize("kernel_size", [5, (5, 7), (1, 5), (5, 1)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("border_type", ["reflect", "constant"])
    def test_dynamo(self, batch_size, kernel_size, separable, border_type, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = BoxBlur(kernel_size, border_type, separable=separable)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))
        functional_optimized = torch_optimizer(box_blur)
        self.assert_close(
            box_blur(data, kernel_size, border_type, separable=separable),
            functional_optimized(data, kernel_size, border_type, separable=separable),
        )

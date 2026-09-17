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
import io
import warnings

import pytest
import torch

from kornia.core._compat import torch_version_lt
from kornia.filters import GaussianBlur2d, gaussian_blur2d, get_gaussian_kernel1d
from kornia.filters.filter import filter2d_separable
from kornia.filters.gaussian import _gaussian_blur2d_cpu

from testing.base import BaseTester

gaussian_module = importlib.import_module("kornia.filters.gaussian")


def _require_native_cpu(device: torch.device, dtype: torch.dtype) -> None:
    if device.type != "cpu" or dtype not in (torch.float32, torch.float64):
        pytest.skip("the CPU slice accumulation path supports native CPU float32 and float64 only")


def _use_native_cpu_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The production guard deliberately leaves oneDNN convolutions alone. Force
    # its unavailable branch so this suite exercises the native fallback on CI.
    monkeypatch.setattr(gaussian_module, "_HAS_MKLDNN", False)


def _cpu_tolerance(dtype: torch.dtype) -> float:
    return 5e-7 if dtype == torch.float32 else 1e-12


class TestGaussianBlurCpu(BaseTester):
    @pytest.mark.parametrize("border_type", ("constant", "reflect", "replicate", "circular"))
    def test_helper_matches_separable_convolution_per_sample_kernels(self, border_type, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(2, 2, 257, 256, device=device, dtype=dtype)
        sigma = torch.tensor([[0.6, 1.7], [1.4, 0.8]], device=device, dtype=dtype)
        kernel_x = get_gaussian_kernel1d(7, sigma[:, 1:])
        kernel_y = get_gaussian_kernel1d(5, sigma[:, :1])

        actual = _gaussian_blur2d_cpu(image, kernel_x, kernel_y, border_type)
        expected = filter2d_separable(image, kernel_x, kernel_y, border_type)

        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_helper_vmap_has_no_user_warning(self, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(2, 1, 7, 9, device=device, dtype=dtype)
        kernel_x = torch.rand(2, 5, device=device, dtype=dtype)
        kernel_y = torch.rand(2, 3, device=device, dtype=dtype)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            actual = torch.vmap(lambda x, kx, ky: _gaussian_blur2d_cpu(x[None], kx[None], ky[None], "reflect")[0])(
                image, kernel_x, kernel_y
            )
        assert not any(issubclass(warning.category, UserWarning) for warning in recorded)
        expected = filter2d_separable(image, kernel_x, kernel_y, "reflect")
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_helper_gradcheck_small_input_and_kernels(self, device):
        if device.type != "cpu":
            pytest.skip("the helper is CPU-only")
        image = torch.rand(1, 1, 7, 9, device=device, dtype=torch.float64, requires_grad=True)
        kernel_x = torch.rand(1, 5, device=device, dtype=torch.float64, requires_grad=True)
        kernel_y = torch.rand(1, 3, device=device, dtype=torch.float64, requires_grad=True)

        self.gradcheck(_gaussian_blur2d_cpu, (image, kernel_x, kernel_y, "reflect"))

    def test_public_large_path_dispatches_without_reverse_mode_gradients(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        _use_native_cpu_path(monkeypatch)
        image = torch.rand(2, 1, 256, 256, device=device, dtype=dtype)
        sigma = torch.tensor([[0.7, 1.8], [1.6, 0.9]], device=device, dtype=dtype)
        helper = gaussian_module._gaussian_blur2d_cpu
        calls = 0

        def counted_helper(*args, **kwargs):
            nonlocal calls
            calls += 1
            return helper(*args, **kwargs)

        monkeypatch.setattr(gaussian_module, "_gaussian_blur2d_cpu", counted_helper)

        actual = gaussian_blur2d(image, (5, 7), sigma, "reflect")
        kernel_x = get_gaussian_kernel1d(7, sigma[:, 1:])
        kernel_y = get_gaussian_kernel1d(5, sigma[:, :1])
        expected = filter2d_separable(image, kernel_x, kernel_y, "reflect")
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        assert calls == 1

    def test_public_reverse_mode_gradients_match_convolution(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        _use_native_cpu_path(monkeypatch)

        def should_not_run(*args, **kwargs):
            raise AssertionError("reverse-mode gradients must use the convolution path")

        monkeypatch.setattr(gaussian_module, "_gaussian_blur2d_cpu", should_not_run)
        image = torch.rand(2, 1, 256, 256, device=device, dtype=dtype, requires_grad=True)
        sigma = torch.tensor([[0.7, 1.8], [1.6, 0.9]], device=device, dtype=dtype, requires_grad=True)
        actual = gaussian_blur2d(image, (5, 7), sigma, "reflect")
        kernel_x = get_gaussian_kernel1d(7, sigma[:, 1:])
        kernel_y = get_gaussian_kernel1d(5, sigma[:, :1])
        expected = filter2d_separable(image, kernel_x, kernel_y, "reflect")
        actual_grads = torch.autograd.grad(actual.square().mean(), (image, sigma), create_graph=True)
        expected_grads = torch.autograd.grad(expected.square().mean(), (image, sigma), create_graph=True)
        self.assert_close(actual_grads[0], expected_grads[0], rtol=3e-5, atol=3e-5)
        self.assert_close(actual_grads[1], expected_grads[1], rtol=3e-5, atol=3e-5)
        self.assert_close(
            torch.autograd.grad(actual_grads[1].sum(), sigma)[0],
            torch.autograd.grad(expected_grads[1].sum(), sigma)[0],
            rtol=4e-5,
            atol=4e-5,
        )

    def test_public_large_path_gradcheck_with_scalar_image_parameter(self, monkeypatch, device):
        if device.type != "cpu":
            pytest.skip("the optimized path is CPU-only")
        _use_native_cpu_path(monkeypatch)
        base = torch.rand(1, 1, 256, 512, device=device, dtype=torch.float64)
        scale = torch.tensor(0.8, device=device, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor([[0.9, 1.3]], device=device, dtype=torch.float64, requires_grad=True)

        def op(image_scale, blur_sigma):
            return gaussian_blur2d(base * image_scale, (5, 7), blur_sigma, "reflect")[..., :2, :2]

        self.gradcheck(op, (scale, sigma))
        assert torch.autograd.gradgradcheck(op, (scale, sigma))

    def test_public_large_path_jvp_through_sigma(self, monkeypatch, device):
        if device.type != "cpu":
            pytest.skip("the optimized path is CPU-only")
        _use_native_cpu_path(monkeypatch)
        image = torch.rand(1, 1, 256, 512, device=device, dtype=torch.float64)
        sigma = torch.tensor([[0.9, 1.3]], device=device, dtype=torch.float64)
        tangent = torch.tensor([[0.2, -0.3]], device=device, dtype=torch.float64)
        helper = gaussian_module._gaussian_blur2d_cpu
        calls = 0

        def counted_helper(*args, **kwargs):
            nonlocal calls
            calls += 1
            return helper(*args, **kwargs)

        monkeypatch.setattr(gaussian_module, "_gaussian_blur2d_cpu", counted_helper)

        def fast(blur_sigma):
            return gaussian_blur2d(image, (5, 7), blur_sigma, "replicate")

        def reference(blur_sigma):
            return filter2d_separable(
                image,
                get_gaussian_kernel1d(7, blur_sigma[:, 1:]),
                get_gaussian_kernel1d(5, blur_sigma[:, :1]),
                "replicate",
            )

        actual, actual_tangent = torch.func.jvp(fast, (sigma,), (tangent,))
        expected, expected_tangent = torch.func.jvp(reference, (sigma,), (tangent,))
        self.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
        self.assert_close(actual_tangent, expected_tangent, rtol=3e-10, atol=3e-10)
        assert calls == 1

    def test_noncontiguous_and_small_inputs_fall_back_to_convolution(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        _use_native_cpu_path(monkeypatch)

        def should_not_run(*args, **kwargs):
            raise AssertionError("the optimized CPU helper should not run")

        monkeypatch.setattr(gaussian_module, "_gaussian_blur2d_cpu", should_not_run)
        noncontiguous = torch.rand(1, 1, 512, 256, device=device, dtype=dtype).transpose(-1, -2)
        small = torch.rand(1, 1, 256, 256, device=device, dtype=dtype)
        for image in (noncontiguous, small):
            actual = gaussian_blur2d(image, (5, 7), (0.9, 1.3), "reflect")
            expected = filter2d_separable(
                image,
                get_gaussian_kernel1d(7, 1.3, device=device, dtype=dtype),
                get_gaussian_kernel1d(5, 0.9, device=device, dtype=dtype),
                "reflect",
            )
            self.assert_close(actual, expected)

    def test_onednn_available_falls_back_to_convolution(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        monkeypatch.setattr(gaussian_module, "_HAS_MKLDNN", True)

        def should_not_run(*args, **kwargs):
            raise AssertionError("oneDNN hosts must keep their convolution implementation")

        monkeypatch.setattr(gaussian_module, "_gaussian_blur2d_cpu", should_not_run)
        image = torch.rand(1, 1, 256, 512, device=device, dtype=dtype)
        actual = gaussian_blur2d(image, (5, 7), (0.9, 1.3), "reflect")
        expected = filter2d_separable(
            image,
            get_gaussian_kernel1d(7, 1.3, device=device, dtype=dtype),
            get_gaussian_kernel1d(5, 0.9, device=device, dtype=dtype),
            "reflect",
        )
        self.assert_close(actual, expected)

    def test_autocast_falls_back_to_convolution(self, monkeypatch, device, dtype):
        if device.type != "cpu" or dtype != torch.float32:
            pytest.skip("CPU autocast is required")
        _use_native_cpu_path(monkeypatch)

        def should_not_run(*args, **kwargs):
            raise AssertionError("the optimized CPU helper should not run")

        monkeypatch.setattr(gaussian_module, "_gaussian_blur2d_cpu", should_not_run)
        image = torch.rand(1, 1, 256, 512, device=device)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            autocast_output = gaussian_blur2d(image, (5, 7), (0.9, 1.3), "reflect")
            autocast_reference = filter2d_separable(
                image,
                get_gaussian_kernel1d(7, 1.3, device=device),
                get_gaussian_kernel1d(5, 0.9, device=device),
                "reflect",
            )
        assert autocast_output.dtype == autocast_reference.dtype
        self.assert_close(autocast_output, autocast_reference)

    @pytest.mark.parametrize("border_type", ("constant", "reflect", "replicate", "circular"))
    @pytest.mark.parametrize("has_mkldnn", (False, True))
    def test_dynamo_large_input_dispatch(self, monkeypatch, device, dtype, torch_optimizer, border_type, has_mkldnn):
        _require_native_cpu(device, dtype)
        monkeypatch.setattr(gaussian_module, "_HAS_MKLDNN", has_mkldnn)
        image = torch.rand(2, 1, 256, 256, device=device, dtype=dtype)
        sigma = torch.tensor([[0.7, 1.8], [1.6, 0.9]], device=device, dtype=dtype)
        reference = filter2d_separable(
            image,
            get_gaussian_kernel1d(7, sigma[:, 1:]),
            get_gaussian_kernel1d(5, sigma[:, :1]),
            border_type,
        )

        def should_not_run(*args, **kwargs):
            raise AssertionError("compilation must preserve the CPU backend eligibility guard")

        excluded = "_gaussian_blur2d_cpu" if has_mkldnn else "filter2d_separable"
        monkeypatch.setattr(gaussian_module, excluded, should_not_run)
        compiled = torch_optimizer(lambda x, s: gaussian_blur2d(x, (5, 7), s, border_type), fullgraph=True)
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(compiled(image, sigma), reference, rtol=tolerance, atol=tolerance)

    def test_export_large_input(self, monkeypatch, device, dtype):
        if device.type != "cpu" or dtype != torch.float32:
            pytest.skip("strict export is checked once on CPU float32")
        _use_native_cpu_path(monkeypatch)
        image = torch.rand(1, 2, 256, 256, device=device, dtype=dtype)
        module = GaussianBlur2d((5, 7), (0.9, 1.3))
        expected = module(image)
        exported = torch.export.export(module, (image,), strict=True)
        self.assert_close(exported.module()(image), expected)

    @pytest.mark.skipif(
        torch_version_lt(2, 6, 0), reason="the dynamo ONNX exporter is only non-experimental from PyTorch 2.6"
    )
    def test_onnx_modern_large_input(self, monkeypatch, device, dtype, tmp_path):
        if device.type != "cpu" or dtype != torch.float32:
            pytest.skip("modern ONNX runtime is checked once on CPU float32")
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        ort = pytest.importorskip("onnxruntime")
        _use_native_cpu_path(monkeypatch)
        image = torch.rand(1, 2, 256, 256, device=device, dtype=dtype)
        module = GaussianBlur2d((5, 7), (0.9, 1.3))
        expected = module(image)

        def should_not_run(*args, **kwargs):
            raise AssertionError("modern export should capture the eligible CPU slice path")

        monkeypatch.setattr(gaussian_module, "filter2d_separable", should_not_run)
        path = tmp_path / "gaussian.onnx"
        torch.onnx.export(module, (image,), str(path), dynamo=True, opset_version=18, input_names=["input"])
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        actual = torch.from_numpy(session.run(None, {"input": image.numpy()})[0])
        self.assert_close(actual, expected, rtol=5e-7, atol=5e-7)

    def test_onnx_trace_large_input_falls_back_to_convolution(self, monkeypatch, device, dtype):
        if device.type != "cpu" or dtype != torch.float32:
            pytest.skip("the trace fallback is checked once on CPU float32")
        pytest.importorskip("onnx")
        _use_native_cpu_path(monkeypatch)

        def should_not_run(*args, **kwargs):
            raise AssertionError("the optimized CPU helper should not run while tracing")

        monkeypatch.setattr(gaussian_module, "_gaussian_blur2d_cpu", should_not_run)
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(GaussianBlur2d((5, 7), (0.9, 1.3)), torch.rand(1, 1, 256, 512), buffer, dynamo=False)
        assert buffer.getbuffer().nbytes > 0

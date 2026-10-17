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

import pytest
import torch

import kornia.augmentation.utils.helpers as augmentation_helpers
from kornia.core import ops as core_ops
from kornia.core.utils import _to_device_nonblocking, _torch_inverse_cast

from testing.base import BaseTester

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# About 3 ms of queued GPU work on a current GPU, so that a copy which waited for the queue would be visible.
_SLEEP_CYCLES = 5_000_000


def _source(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    # float64 values that round differently in every narrower dtype, including exact ties of float16 and bfloat16.
    values = torch.randn(shape, dtype=torch.float64) * 1000.0
    if values.numel():
        values.view(-1)[0] = 1.0 + 2.0**-11  # a float16 tie
        values.view(-1)[-1] = 1.0 + 2.0**-8  # a bfloat16 tie
    return values.to(device)


class TestToDeviceNonblocking(BaseTester):
    @pytest.mark.parametrize("direction", ["host_to_device", "device_to_host", "same_device"])
    @pytest.mark.parametrize("shape", [(4, 5), (0, 3)])
    def test_values_equal_blocking_copy(self, direction, shape, device, dtype) -> None:
        cpu = torch.device("cpu")
        src_device, dst_device = {
            "host_to_device": (cpu, device),
            "device_to_host": (device, cpu),
            "same_device": (device, device),
        }[direction]
        src = _source(shape, src_device)
        for target_dtype in (dtype, None):
            out = _to_device_nonblocking(src, dst_device, target_dtype)
            expected = src.to(dst_device, target_dtype) if target_dtype is not None else src.to(dst_device)
            assert out.device.type == dst_device.type
            assert out.dtype == expected.dtype
            assert torch.equal(out.cpu(), expected.cpu())

    @pytest.mark.parametrize("shape", [(4, 5), (0, 3)])
    def test_integer_values_equal_blocking_copy(self, shape, device) -> None:
        src = torch.randint(-(2**40), 2**40, shape, dtype=torch.int64)
        out = _to_device_nonblocking(src, device, torch.int64)
        assert torch.equal(out.cpu(), src.to(device, torch.int64).cpu())

    def test_same_device_and_dtype_returns_input(self, device, dtype) -> None:
        src = torch.rand(3, device=device, dtype=dtype)
        assert _to_device_nonblocking(src, device, dtype) is src
        assert _to_device_nonblocking(src, device) is src

    def test_scriptable(self, device, dtype) -> None:
        scripted = torch.jit.script(_to_device_nonblocking)
        src = torch.rand(3, dtype=torch.float64)
        assert torch.equal(scripted(src, device, dtype).cpu(), src.to(device, dtype).cpu())

    @requires_cuda
    def test_cpu_to_cuda_does_not_sync(self) -> None:
        src = torch.rand(8, 3)
        torch.cuda.synchronize()
        try:
            torch.cuda.set_sync_debug_mode("error")
            for dtype in (None, torch.float32, torch.float16, torch.float64):
                _to_device_nonblocking(src, torch.device("cuda"), dtype)
        finally:
            torch.cuda.set_sync_debug_mode(0)

    @requires_cuda
    def test_source_mutation_after_copy(self) -> None:
        cuda = torch.device("cuda")
        for dtype in (None, torch.float16):
            for _ in range(50):
                src = torch.rand(64, 3)
                expected = src.to(dtype) if dtype is not None else src.clone()
                torch.cuda._sleep(_SLEEP_CYCLES)
                out = _to_device_nonblocking(src, cuda, dtype)
                src.fill_(-1.0)
                torch.cuda.synchronize()
                assert torch.equal(out.cpu(), expected)

    @requires_cuda
    def test_cuda_to_cpu_is_blocking(self) -> None:
        x = torch.rand(256, 256)
        x_cuda = x.cuda()
        torch.cuda.synchronize()
        torch.cuda._sleep(_SLEEP_CYCLES)
        y = x_cuda * 2.0  # queued behind the sleep
        out = _to_device_nonblocking(y, torch.device("cpu"))
        # No synchronize: a non-blocking device-to-host copy could be read before it lands.
        assert torch.equal(out, x * 2.0)
        # A non-blocking device-to-host copy does not synchronize, and the host could read it too early.
        try:
            torch.cuda.set_sync_debug_mode("error")
            with pytest.raises(RuntimeError, match="synchronizing"):
                _to_device_nonblocking(y, torch.device("cpu"))
        finally:
            torch.cuda.set_sync_debug_mode(0)


class TestInverseWithoutErrorCheck(BaseTester):
    @pytest.mark.parametrize("n", [3, 4])
    def test_inverse_without_error_check_is_identical(self, n, device, dtype) -> None:
        # Well conditioned: the identity plus a small random perturbation.
        a = torch.eye(n, device=device, dtype=dtype) + 0.1 * torch.randn(8, n, n, device=device, dtype=dtype)
        expected = _torch_inverse_cast(a)
        if device.type == "cuda":
            torch.cuda.synchronize()
            try:
                torch.cuda.set_sync_debug_mode("error")
                actual = _torch_inverse_cast(a, check_errors=False)
            finally:
                torch.cuda.set_sync_debug_mode(0)
        else:
            actual = _torch_inverse_cast(a, check_errors=False)
        assert torch.equal(actual, expected)

    def test_singular_still_raises_with_error_check(self, device, dtype) -> None:
        with pytest.raises(RuntimeError):
            _torch_inverse_cast(torch.zeros(1, 3, 3, device=device, dtype=dtype))


class TestConstantTensorUnderAutocast(BaseTester):
    @pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
    def test_dtype_and_values_ignore_autocast(self, autocast_dtype, device, dtype) -> None:
        # A literal keeps the requested dtype in an autocast region of another precision, as torch.tensor does.
        data = [[0.299, 0.0, -1.0], [0.0, 0.587, 1.0 / 3.0]]
        expected = torch.tensor(data, device=device, dtype=dtype)
        with torch.autocast(device.type, dtype=autocast_dtype):
            actual = core_ops._constant_tensor(data, device=device, dtype=dtype)
        assert actual.dtype == dtype
        assert torch.equal(actual, expected)


def test_constant_tensor_import_paths() -> None:
    assert augmentation_helpers._constant_tensor is core_ops._constant_tensor
    assert augmentation_helpers._flatten_constant is core_ops._flatten_constant

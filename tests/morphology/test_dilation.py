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

import pytest
import torch

from kornia.morphology import dilation
from kornia.morphology.morphology import _records_grad, _resolve_engine

from testing.base import BaseTester, assert_close
from testing.parametrized_tester import parametrized_test


@parametrized_test(
    smoke_inputs=lambda device, dtype: (
        torch.rand(1, 3, 4, 4, device=device, dtype=dtype),
        torch.ones((3, 3), device=device, dtype=dtype),
    ),
    cardinality_tests=[
        {
            "inputs": lambda device, dtype: (
                torch.ones((1, 3, 4, 4), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([1, 3, 4, 4]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.ones((2, 3, 2, 4), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([2, 3, 2, 4]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.ones((3, 3, 4, 1), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([3, 3, 4, 1]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.ones((3, 2, 5, 5), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([3, 2, 5, 5]),
        },
    ],
    gradcheck_inputs=lambda device: (
        torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=torch.float64),
        torch.rand(3, 3, requires_grad=True, device=device, dtype=torch.float64),
    ),
)
class TestDilate(BaseTester):
    def setup_method(self) -> None:
        self.func = dilation

    def test_kernel(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 1.0, 1.0], [0.7, 1.0, 0.8], [0.9, 0.9, 0.9]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(dilation(tensor, kernel, engine="unfold"), expected, atol=1e-4, rtol=1e-4)
        # The convolution engine measures ~3.9e-4 absolute / ~1.5e-3 relative error on macOS's
        # Apple-backend float32 conv path, above the harness's generic float32 default
        # (atol=1e-5, rtol=1e-4). This explicit tolerance is scoped to this backend-specific case.
        assert_close(dilation(tensor, kernel, engine="convolution"), expected, atol=1e-3, rtol=1e-3)

    def test_structural_element(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        structural_element = torch.tensor(
            [[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, -1.0]], device=device, dtype=dtype
        )
        expected = torch.tensor([[1.0, 1.0, 1.0], [0.7, 1.0, 0.8], [0.9, 0.9, 0.9]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(
            dilation(
                tensor, torch.ones_like(structural_element), structuring_element=structural_element, engine="unfold"
            ),
            expected,
        )
        # See test_kernel: convolution engine needs an explicit tolerance for macOS's
        # Apple-backend float32 numerical error, above the harness's generic float32 default.
        assert_close(
            dilation(
                tensor,
                torch.ones_like(structural_element),
                structuring_element=structural_element,
                engine="convolution",
            ),
            expected,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_flip(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        kernel = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.7, 1.0, 1.0], [0.7, 1.0, 1.0], [0.7, 0.9, 0.9]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(dilation(tensor, kernel), expected)

    def test_exception(self, device, dtype):
        tensor = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert dilation([0.0], kernel)

        with pytest.raises(TypeError):
            assert dilation(tensor, [0.0])

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert dilation(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert dilation(tensor, test)

        with pytest.raises(NotImplementedError, match="unknown"):
            dilation(tensor, kernel, engine="invalid_engine")

    def test_custom_origin(self, device, dtype):
        # Custom origin shifts the structuring element anchor point
        tensor = torch.zeros(1, 1, 5, 5, device=device, dtype=dtype)
        tensor[0, 0, 2, 2] = 1.0  # single hot pixel in center
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        # Default origin (center): dilation spreads symmetrically
        out_default = dilation(tensor, kernel)
        # Custom origin (top-left corner): effect shifts
        out_custom = dilation(tensor, kernel, origin=[0, 0])
        # Results should differ when the anchor point changes
        assert not torch.equal(out_default, out_custom)

    def test_jit(self, device, dtype):
        op = dilation
        op_script = torch.jit.script(op)

        tensor = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(tensor, kernel)
        expected = op(tensor, kernel)

        assert_close(actual, expected)

    def test_convolution_engine_dtype_mismatch(self, device, dtype):
        # engine="convolution" used to crash when tensor.dtype != kernel.dtype, because the
        # conv weight/bias were built from kernel.dtype instead of the input's dtype. See #4541.
        # Passing a mismatched kernel must match casting the kernel to the input dtype up front;
        # that is the same computation, so the results are bitwise equal (no tolerance needed).
        other_dtype = torch.float16 if dtype == torch.float32 else torch.float32

        tensor = torch.rand(1, 2, 5, 5, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=other_dtype)

        result = dilation(tensor, kernel, engine="convolution")

        assert result.dtype == dtype
        self.assert_close(result, dilation(tensor, kernel.to(dtype), engine="convolution"))

    def test_auto_engine(self, device, dtype):
        # engine="auto", the default, runs "unfold" on CUDA and the exact "shift" engine everywhere
        # else (#4525). An explicit engine is passed through unchanged.
        tensor = torch.rand(2, 3, 9, 9, device=device, dtype=dtype)
        kernel = torch.ones(3, 5, device=device, dtype=dtype)
        kernel[0, 0] = 0.0
        expected_engine = "unfold" if device.type == "cuda" else "shift"

        assert _resolve_engine("auto", tensor) == expected_engine
        assert _resolve_engine("unfold", tensor) == "unfold"
        assert _resolve_engine("convolution", tensor) == "convolution"
        assert _resolve_engine("shift", tensor) == "shift"
        expected = dilation(tensor, kernel, engine=expected_engine)
        assert torch.equal(dilation(tensor, kernel), expected)
        assert torch.equal(dilation(tensor, kernel, engine="auto"), expected)

    def test_auto_engine_is_grad_aware(self, device, dtype):
        # A CPU call that records a backward graph takes "unfold" in float32/float64, where "shift"
        # is up to 3.4x slower; half precision and the forward-only path keep "shift". CUDA is
        # "unfold" either way, and MPS is "shift" either way -- the grad branch is CPU-only, so the
        # dtype does not enter off CPU. Bitwise equal forward output makes the switch value-preserving.
        tensor = torch.rand(2, 3, 9, 9, device=device, dtype=dtype)
        kernel = torch.ones(3, 5, device=device, dtype=dtype)
        wide = dtype in (torch.float32, torch.float64)
        grad_engine = "unfold" if device.type == "cuda" or (device.type == "cpu" and wide) else "shift"
        plain_engine = "unfold" if device.type == "cuda" else "shift"

        assert _resolve_engine("auto", tensor, True) == grad_engine
        assert _resolve_engine("auto", tensor, False) == plain_engine
        assert _resolve_engine("shift", tensor, True) == "shift"
        assert torch.equal(dilation(tensor, kernel), dilation(tensor, kernel, engine=plain_engine))

        grad_tensor = tensor.clone().requires_grad_(True)
        assert torch.equal(dilation(grad_tensor, kernel), dilation(grad_tensor, kernel, engine=grad_engine))
        # torch.no_grad() leaves requires_grad set but records nothing, so the forward-only engine wins.
        with torch.no_grad():
            assert torch.equal(dilation(grad_tensor, kernel), dilation(grad_tensor, kernel, engine=plain_engine))

    def test_auto_engine_follows_kernel_grad(self, device, dtype):
        # The neighborhood is built from the kernel and the structuring element, so grad on either
        # one records a backward graph even when the image does not require grad (#4525).
        tensor = torch.rand(2, 3, 9, 9, device=device, dtype=dtype)
        kernel = torch.ones(3, 5, device=device, dtype=dtype)
        wide = dtype in (torch.float32, torch.float64)
        grad_engine = "unfold" if device.type == "cuda" or (device.type == "cpu" and wide) else "shift"

        assert _records_grad(tensor, kernel, None) is False
        assert _records_grad(tensor, kernel.clone().requires_grad_(True), None) is True
        assert _records_grad(tensor, kernel, torch.rand(3, 5, device=device, dtype=dtype)) is False
        se = torch.rand(3, 5, device=device, dtype=dtype).requires_grad_(True)
        assert _records_grad(tensor, kernel, se) is True

        grad_kernel = kernel.clone().requires_grad_(True)
        assert torch.equal(dilation(tensor, grad_kernel), dilation(tensor, grad_kernel, engine=grad_engine))

    @pytest.mark.parametrize("kernel_shape", [(1, 1), (3, 3), (3, 5), (4, 2)])
    @pytest.mark.parametrize("origin", ["center", "first", "last"])
    @pytest.mark.parametrize("border_type", ["geodesic", "constant", "reflect", "replicate"])
    @pytest.mark.parametrize("non_flat", [False, True])
    def test_shift_engine_matches_unfold(self, device, dtype, kernel_shape, origin, border_type, non_flat):
        # engine="shift" reduces the same max-plus terms as engine="unfold" in a different order;
        # max and min are exact in any order, so the outputs are bitwise equal (#4729).
        kh, kw = kernel_shape
        origin_yx = {"center": None, "first": [0, 0], "last": [kh - 1, kw - 1]}[origin]
        anchor = origin_yx if origin_yx is not None else [kh // 2, kw // 2]
        tensor = torch.rand(2, 3, 6, 7, device=device, dtype=dtype) * 10 - 5
        kernel = (torch.rand(kh, kw, device=device) > 0.3).to(dtype)
        kernel[anchor[0], anchor[1]] = 1.0
        structuring_element = (torch.rand(kh, kw, device=device) * 2 - 1).to(dtype) if non_flat else None
        border_value = 0.5 if border_type == "constant" else 0.0
        kwargs = {
            "structuring_element": structuring_element,
            "origin": origin_yx,
            "border_type": border_type,
            "border_value": border_value,
        }

        expected = dilation(tensor, kernel, engine="unfold", **kwargs)
        actual = dilation(tensor, kernel, engine="shift", **kwargs)

        assert actual.dtype == tensor.dtype
        assert torch.equal(actual, expected)

    def test_shift_engine_gradcheck(self, device):
        tensor = torch.rand(2, 3, 5, 5, device=device, dtype=torch.float64)
        kernel = torch.ones(3, 3, device=device, dtype=torch.float64)
        kernel[0, 2] = 0.0
        self.gradcheck(lambda t: dilation(t, kernel, engine="shift"), (tensor,))

    def test_shift_engine_jit(self, device, dtype):
        op_script = torch.jit.script(dilation)
        tensor = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        assert torch.equal(op_script(tensor, kernel, engine="shift"), dilation(tensor, kernel, engine="shift"))

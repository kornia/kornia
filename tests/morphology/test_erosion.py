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

from kornia.morphology import erosion
from kornia.morphology import morphology as morphology_module
from kornia.morphology.morphology import _records_grad, _resolve_engine

from testing.base import BaseTester, assert_close, supports_reflect_padding, supports_replicate_padding
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
class TestErode(BaseTester):
    def setup_method(self) -> None:
        self.func = erosion

    def test_kernel(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.5, 0.3, 0.3], [0.3, 0.3, 0.2], [0.4, 0.2, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(erosion(tensor, kernel), expected, atol=1e-4, rtol=1e-4)
        # The convolution engine measures ~3.9e-4 absolute / ~1.5e-3 relative error on macOS's
        # Apple-backend float32 conv path, above the harness's generic float32 default
        # (atol=1e-5, rtol=1e-4). This explicit tolerance is scoped to this backend-specific case.
        assert_close(erosion(tensor, kernel, engine="convolution"), expected, atol=1e-3, rtol=1e-3)

    def test_structural_element(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        structural_element = torch.tensor(
            [[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, -1.0]], device=device, dtype=dtype
        )
        expected = torch.tensor([[0.5, 0.3, 0.3], [0.3, 0.3, 0.2], [0.4, 0.2, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(
            erosion(tensor, torch.ones_like(structural_element), structuring_element=structural_element),
            expected,
            atol=1e-4,
            rtol=1e-4,
        )
        # See test_kernel: convolution engine needs an explicit tolerance for macOS's
        # Apple-backend float32 numerical error, above the harness's generic float32 default.
        assert_close(
            erosion(
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
        expected = torch.tensor([[0.3, 0.3, 0.3], [0.3, 0.2, 0.2], [0.3, 0.2, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(erosion(tensor, kernel, engine="unfold"), expected, atol=1e-4, rtol=1e-4)
        # See test_kernel: convolution engine needs an explicit tolerance for macOS's
        # Apple-backend float32 numerical error, above the harness's generic float32 default.
        assert_close(erosion(tensor, kernel, engine="convolution"), expected, atol=1e-3, rtol=1e-3)

    def test_exception(self, device, dtype):
        tensor = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert erosion([0.0], kernel)

        with pytest.raises(TypeError):
            assert erosion(tensor, [0.0])

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert erosion(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert erosion(tensor, test)

    def test_jit(self, device, dtype):
        op = erosion
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

        result = erosion(tensor, kernel, engine="convolution")

        assert result.dtype == dtype
        self.assert_close(result, erosion(tensor, kernel.to(dtype), engine="convolution"))

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
        expected = erosion(tensor, kernel, engine=expected_engine)
        assert torch.equal(erosion(tensor, kernel), expected)
        assert torch.equal(erosion(tensor, kernel, engine="auto"), expected)

    def test_auto_engine_is_grad_aware(self, device, dtype):
        # A CPU call that records a backward graph takes "unfold" in float32/float64, where "shift"
        # is up to 3.4x slower; half precision and the forward-only path keep "shift". CUDA is
        # "unfold" either way, and MPS is "shift" either way -- the grad branch is CPU-only, so the
        # dtype does not enter off CPU. Equal forward output makes the switch value-preserving.
        tensor = torch.rand(2, 3, 9, 9, device=device, dtype=dtype)
        kernel = torch.ones(3, 5, device=device, dtype=dtype)
        wide = dtype in (torch.float32, torch.float64)
        grad_engine = "unfold" if device.type == "cuda" or (device.type == "cpu" and wide) else "shift"
        plain_engine = "unfold" if device.type == "cuda" else "shift"

        assert _resolve_engine("auto", tensor, True) == grad_engine
        assert _resolve_engine("auto", tensor, False) == plain_engine
        assert _resolve_engine("shift", tensor, True) == "shift"
        assert torch.equal(erosion(tensor, kernel), erosion(tensor, kernel, engine=plain_engine))

        grad_tensor = tensor.clone().requires_grad_(True)
        assert torch.equal(erosion(grad_tensor, kernel), erosion(grad_tensor, kernel, engine=grad_engine))
        # torch.no_grad() leaves requires_grad set but records nothing, so the forward-only engine wins.
        with torch.no_grad():
            assert torch.equal(erosion(grad_tensor, kernel), erosion(grad_tensor, kernel, engine=plain_engine))

    def test_auto_engine_follows_structuring_element_grad(self, device, dtype, monkeypatch):
        # The neighborhood is built from the structuring element, so grad on it records a backward
        # graph even when the image does not require grad. The kernel enters only through the
        # ``kernel == 0`` mask, so a kernel that requires grad records nothing and must not push a
        # forward-only call onto the slower grad-mode engine (#4525).
        tensor = torch.rand(2, 3, 9, 9, device=device, dtype=dtype)
        kernel = torch.ones(3, 5, device=device, dtype=dtype)
        se = torch.rand(3, 5, device=device, dtype=dtype)
        seen = []
        resolve = morphology_module._resolve_engine
        monkeypatch.setattr(
            morphology_module,
            "_resolve_engine",
            lambda engine, t, recording_grad=False, dtype=None: (
                seen.append(recording_grad) or resolve(engine, t, recording_grad, dtype)
            ),
        )

        assert _records_grad(tensor, None) is False
        assert _records_grad(tensor, se) is False
        assert _records_grad(tensor, se.clone().requires_grad_(True)) is True

        erosion(tensor, kernel.clone().requires_grad_(True), engine="shift")
        erosion(tensor, kernel, se.clone().requires_grad_(True), engine="shift")
        erosion(tensor.clone().requires_grad_(True), kernel, engine="shift")
        with torch.no_grad():
            erosion(tensor.clone().requires_grad_(True), kernel, engine="shift")
        assert seen == [False, True, True, False]
        assert not erosion(tensor, kernel.clone().requires_grad_(True)).requires_grad

    def test_auto_engine_resolves_from_promoted_dtype(self, device, monkeypatch):
        # A float16 image with a float32 kernel computes, and differentiates, in float32, so the
        # float32/float64 grad rule of "auto" must see the promoted dtype rather than the image's.
        # On CPU that is the difference between "unfold" and a "shift" backward that is ~5x slower.
        tensor = torch.rand(2, 3, 9, 9, device=device, dtype=torch.float16).requires_grad_(True)
        kernel = torch.ones(3, 5, device=device, dtype=torch.float32)
        grad_engine = "unfold" if device.type in ("cuda", "cpu") else "shift"
        seen = []
        resolve = morphology_module._resolve_engine

        def recording_resolve(engine, t, recording_grad=False, dtype=None):
            seen.append((dtype, resolve(engine, t, recording_grad, dtype)))
            return seen[-1][1]

        monkeypatch.setattr(morphology_module, "_resolve_engine", recording_resolve)
        assert _resolve_engine("auto", tensor, True, torch.float32) == grad_engine
        assert _resolve_engine("auto", tensor, True) == ("unfold" if device.type == "cuda" else "shift")

        actual = erosion(tensor, kernel)
        expected = erosion(tensor, kernel, engine=grad_engine)
        assert seen[0] == (torch.float32, grad_engine)
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
        # The same call without a backward graph keeps the forward-only engine off CUDA.
        with torch.no_grad():
            erosion(tensor, kernel)
        assert seen[-1] == (torch.float32, "unfold" if device.type == "cuda" else "shift")

    def test_auto_engine_forward_ad_takes_forward_engine(self, device):
        # Forward-mode AD records no backward graph, so off CUDA "auto" takes "shift" and the tangent
        # at tied maxima is the "shift" one: torch.maximum averages the tied tangents where
        # torch.max(dim=) (the "unfold" engine) forwards one argmax's tangent. The all-ones input makes
        # every window a tie, so the two engines' tangents differ and the pin can fail.
        tensor = torch.ones(1, 1, 2, 2, device=device)
        tangent = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]], device=device)
        kernel = torch.ones(2, 2, device=device)
        expected_engine = "unfold" if device.type == "cuda" else "shift"

        def run(engine):
            return torch.func.jvp(lambda t: erosion(t, kernel, origin=[0, 0], engine=engine), (tensor,), (tangent,))[1]

        assert not torch.equal(run("unfold"), run("shift"))
        assert torch.equal(run("auto"), run(expected_engine))

    @pytest.mark.parametrize("kernel_shape", [(1, 1), (3, 3), (3, 5), (4, 2)])
    @pytest.mark.parametrize("origin", ["center", "first", "last"])
    @pytest.mark.parametrize("border_type", ["geodesic", "constant", "reflect", "replicate"])
    @pytest.mark.parametrize("non_flat", [False, True])
    def test_shift_engine_matches_unfold(self, device, dtype, kernel_shape, origin, border_type, non_flat):
        # engine="shift" reduces the same finite max-plus terms as engine="unfold" in the same order,
        # so the outputs are equal (#4729).
        if border_type == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        if border_type == "replicate" and not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
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

        expected = erosion(tensor, kernel, engine="unfold", **kwargs)
        actual = erosion(tensor, kernel, engine="shift", **kwargs)

        assert actual.dtype == tensor.dtype
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize(
        ("border_type", "row"),
        [
            ("reflect", [3.0, 2.0, 1.0, 2.0, 3.0]),
            ("replicate", [1.0, 1.0, 1.0, 2.0, 3.0]),
            ("circular", [4.0, 5.0, 1.0, 2.0, 3.0]),
        ],
    )
    def test_border_value_ignored_for_non_constant_border(self, device, dtype, border_type, row):
        # border_value only applies to border_type="constant"; the other modes ignore it
        # instead of forwarding it to F.pad, which rejects a value for them (#4748).
        if border_type == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        if border_type == "replicate" and not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        # The single-cell kernel at the default origin [0, 2] makes erosion read x(p - 2), so the two
        # leftmost outputs are pure padding. Were border_value used as a fill they would be -7; were the
        # mode dropped they would be 0. Integral values are exact in every dtype.
        ramp = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device, dtype=dtype)[None, None]
        kernel = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([row], device=device, dtype=dtype)[None, None]

        assert torch.equal(erosion(ramp, kernel, border_type=border_type, border_value=-7.0), expected)
        assert torch.equal(erosion(ramp, kernel, border_type=border_type), expected)

    def test_shift_engine_mixed_dtype_non_flat(self, device):
        tensor = torch.zeros(1, 1, 2, 2, device=device, dtype=torch.float16)
        kernel = torch.ones(2, 2, device=device, dtype=torch.float32)
        structuring_element = torch.tensor(
            [[0.1234567, 0.2345678], [0.3456789, 0.4567890]], device=device, dtype=torch.float32
        )
        kwargs = {"structuring_element": structuring_element, "origin": [0, 0]}

        expected = erosion(tensor, kernel, engine="unfold", **kwargs)
        actual = erosion(tensor, kernel, engine="shift", **kwargs)

        assert expected.dtype == torch.float32
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        assert torch.equal(erosion(tensor, kernel, **kwargs), expected)

    def test_shift_engine_signed_zero_is_equal_value(self, device):
        # Windows that tie between +0 and -0 give an equal value on every engine; which zero comes out
        # is the backend's tie choice for max/min (CPU keeps the first operand, MPS the second) and is
        # deliberately not pinned, on either engine.
        tensor = torch.tensor([[[[1.0, -0.0], [0.0, -0.0]]]], device=device)
        kernel = torch.ones(2, 2, device=device)

        expected = erosion(tensor, kernel, origin=[0, 0], engine="unfold")
        actual = erosion(tensor, kernel, origin=[0, 0], engine="shift")

        assert torch.equal(actual, expected)
        assert torch.equal(actual, torch.zeros_like(actual))

    def test_shift_engine_gradcheck(self, device):
        tensor = torch.rand(2, 3, 5, 5, device=device, dtype=torch.float64)
        kernel = torch.ones(3, 3, device=device, dtype=torch.float64)
        kernel[0, 2] = 0.0
        self.gradcheck(lambda t: erosion(t, kernel, engine="shift"), (tensor,))

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_dynamo(self, device, dtype, torch_optimizer, requires_grad):
        # engine="auto" reads grad mode, ``requires_grad`` and the promoted dtype when the graph is traced,
        # so both engines it can select on this device (``shift`` forward-only, ``unfold`` for a CPU
        # float32/float64 backward graph) have to compile to the eager result.
        tensor = torch.rand(2, 3, 9, 9, device=device, dtype=dtype).requires_grad_(requires_grad)
        kernel = torch.ones(3, 5, device=device, dtype=dtype)
        kernel[0, 0] = 0.0
        op_optimized = torch_optimizer(erosion)

        self.assert_close(erosion(tensor, kernel), op_optimized(tensor, kernel))

    def test_shift_engine_jit(self, device, dtype):
        op_script = torch.jit.script(erosion)
        tensor = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        assert torch.equal(op_script(tensor, kernel, engine="shift"), erosion(tensor, kernel, engine="shift"))

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

from kornia.morphology import dilation, erosion
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
        # The convolution and unfold engines agree within the test tolerance.
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
        # The convolution and unfold engines agree within the test tolerance.
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

        with pytest.raises(ValueError, match="Unknown `border_type`"):
            dilation(tensor, kernel, border_type="banana")

        with pytest.raises(ValueError, match="`structuring_element` shape must match `kernel` shape"):
            dilation(tensor, kernel, structuring_element=torch.ones(3, 2, device=device, dtype=dtype))

    @pytest.mark.parametrize("kernel_dtype", [torch.bool, torch.uint8, torch.int8, torch.int64])
    @pytest.mark.parametrize("engine", ["unfold", "shift", "auto"])
    def test_non_float_kernel_matches_float_kernel(self, device, dtype, kernel_dtype, engine):
        # The kernel is only a membership mask (#4736): a bool or integer kernel must give exactly the
        # float kernel's result and dtype. The cross has zeros, so an excluded neighbor that leaks in shows.
        tensor = torch.rand(2, 3, 6, 7, device=device, dtype=dtype)
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected = dilation(tensor, kernel, engine=engine)
        actual = dilation(tensor, kernel.to(kernel_dtype), engine=engine)
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)

    def test_integer_image_keeps_the_kernel_dtype(self, device):
        # Only a floating-point image lends its dtype to a non-float kernel (#4736). An integer image
        # keeps the kernel's own dtype, as before: int32 with int64 computes and returns int64.
        tensor = torch.tensor([[0, 3, 0, 0, 7]], dtype=torch.int32, device=device)[None, None]
        kernel = torch.tensor([[1, 0, 1]], dtype=torch.int64, device=device)
        actual = dilation(tensor, kernel)
        assert actual.dtype == torch.int64
        assert actual.flatten().tolist() == [3, 0, 3, 7, 0]

    @pytest.mark.parametrize("border_type", ["geodesic", "constant", "reflect", "replicate", "circular"])
    def test_accepted_border_types(self, device, dtype, border_type):
        # Every documented border_type must pass the validation (#4736).
        if border_type == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        if border_type == "replicate" and not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        tensor = torch.rand(1, 2, 5, 6, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)
        assert dilation(tensor, kernel, border_type=border_type).shape == tensor.shape

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
        # dtype does not enter off CPU. Equal forward output makes the switch value-preserving.
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

        dilation(tensor, kernel.clone().requires_grad_(True), engine="shift")
        dilation(tensor, kernel, se.clone().requires_grad_(True), engine="shift")
        dilation(tensor.clone().requires_grad_(True), kernel, engine="shift")
        with torch.no_grad():
            dilation(tensor.clone().requires_grad_(True), kernel, engine="shift")
        assert seen == [False, True, True, False]
        assert not dilation(tensor, kernel.clone().requires_grad_(True)).requires_grad

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

        actual = dilation(tensor, kernel)
        expected = dilation(tensor, kernel, engine=grad_engine)
        assert seen[0] == (torch.float32, grad_engine)
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
        # The same call without a backward graph keeps the forward-only engine off CUDA.
        with torch.no_grad():
            dilation(tensor, kernel)
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
            return torch.func.jvp(lambda t: dilation(t, kernel, origin=[0, 0], engine=engine), (tensor,), (tangent,))[1]

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

        expected = dilation(tensor, kernel, engine="unfold", **kwargs)
        actual = dilation(tensor, kernel, engine="shift", **kwargs)

        assert actual.dtype == tensor.dtype
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize(
        ("border_type", "row"),
        [
            ("reflect", [3.0, 4.0, 5.0, 4.0, 3.0]),
            ("replicate", [3.0, 4.0, 5.0, 5.0, 5.0]),
            ("circular", [3.0, 4.0, 5.0, 1.0, 2.0]),
        ],
    )
    def test_border_value_ignored_for_non_constant_border(self, device, dtype, border_type, row):
        # border_value only applies to border_type="constant"; the other modes ignore it
        # instead of forwarding it to F.pad, which rejects a value for them (#4748).
        if border_type == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        if border_type == "replicate" and not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        # The single-cell kernel at the default origin [0, 2] makes dilation read x(p + 2), so the two
        # rightmost outputs are pure padding. Were border_value used as a fill they would be -7; were the
        # mode dropped they would be 0. Integral values are exact in every dtype.
        ramp = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device, dtype=dtype)[None, None]
        kernel = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([row], device=device, dtype=dtype)[None, None]

        assert torch.equal(dilation(ramp, kernel, border_type=border_type, border_value=-7.0), expected)
        assert torch.equal(dilation(ramp, kernel, border_type=border_type), expected)

    def test_shift_engine_mixed_dtype_non_flat(self, device):
        tensor = torch.zeros(1, 1, 2, 2, device=device, dtype=torch.float16)
        kernel = torch.ones(2, 2, device=device, dtype=torch.float32)
        structuring_element = torch.tensor(
            [[0.1234567, 0.2345678], [0.3456789, 0.4567890]], device=device, dtype=torch.float32
        )
        kwargs = {"structuring_element": structuring_element, "origin": [0, 0]}

        expected = dilation(tensor, kernel, engine="unfold", **kwargs)
        actual = dilation(tensor, kernel, engine="shift", **kwargs)

        assert expected.dtype == torch.float32
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        assert torch.equal(dilation(tensor, kernel, **kwargs), expected)

    def test_shift_engine_signed_zero_is_equal_value(self, device):
        # Windows that tie between +0 and -0 give an equal value on every engine; which zero comes out
        # is the backend's tie choice for max/min (CPU keeps the first operand, MPS the second) and is
        # deliberately not pinned, on either engine.
        tensor = torch.tensor([[[[-1.0, -0.0], [0.0, -0.0]]]], device=device)
        kernel = torch.ones(2, 2, device=device)

        expected = dilation(tensor, kernel, origin=[1, 1], engine="unfold")
        actual = dilation(tensor, kernel, origin=[1, 1], engine="shift")

        assert torch.equal(actual, expected)
        assert torch.equal(actual, torch.zeros_like(actual))

    def test_shift_engine_gradcheck(self, device):
        tensor = torch.rand(2, 3, 5, 5, device=device, dtype=torch.float64)
        kernel = torch.ones(3, 3, device=device, dtype=torch.float64)
        kernel[0, 2] = 0.0
        self.gradcheck(lambda t: dilation(t, kernel, engine="shift"), (tensor,))

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_dynamo(self, device, dtype, torch_optimizer, requires_grad):
        # engine="auto" reads grad mode, ``requires_grad`` and the promoted dtype when the graph is traced,
        # so both engines it can select on this device (``shift`` forward-only, ``unfold`` for a CPU
        # float32/float64 backward graph) have to compile to the eager result.
        tensor = torch.rand(2, 3, 9, 9, device=device, dtype=dtype).requires_grad_(requires_grad)
        kernel = torch.ones(3, 5, device=device, dtype=dtype)
        kernel[0, 0] = 0.0
        op_optimized = torch_optimizer(dilation)

        self.assert_close(dilation(tensor, kernel), op_optimized(tensor, kernel))

    def test_shift_engine_forward_only_matches_autograd_safe(self, device, dtype):
        tensor = torch.rand(2, 3, 9, 9, device=device, dtype=dtype)
        kernel = torch.randn(3, 3, device=device, dtype=dtype)

        forward = dilation(tensor, kernel, engine="shift")
        safe = dilation(tensor.clone().requires_grad_(True), kernel, engine="shift").detach()

        assert torch.equal(forward, safe)

    @pytest.mark.parametrize("operand", ["image", "structuring_element"])
    def test_shift_engine_forward_ad(self, device, dtype, operand):
        # ``out=`` ops have no forward-mode AD formula, so a tangent on either operand must keep the
        # out-of-place reduction. ``make_dual`` is used because ``torch.func.jvp`` is a functorch transform,
        # which the in-place gate declines before it looks at the tangents.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("half-precision sums tie, and tied tangents differ between engines")

        tensor = torch.rand(1, 1, 9, 11, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)
        structuring_element = torch.randn(3, 3, device=device, dtype=dtype)
        primal = tensor if operand == "image" else structuring_element
        tangent = torch.randn_like(primal)

        def run(engine):
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(primal, tangent)
                if operand == "image":
                    output = dilation(dual, kernel, structuring_element, engine=engine)
                else:
                    output = dilation(tensor, kernel, dual, engine=engine)
                return torch.autograd.forward_ad.unpack_dual(output)

        actual, expected = run("shift"), run("unfold")

        self.assert_close(actual.primal, expected.primal)
        self.assert_close(actual.tangent, expected.tangent)

    def test_shift_engine_vmap(self, device, dtype):
        tensor = torch.rand(2, 3, 1, 9, 11, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = torch.func.vmap(lambda x: dilation(x, kernel, engine="shift"))(tensor)
        expected = torch.func.vmap(lambda x: dilation(x, kernel, engine="unfold"))(tensor)

        assert torch.equal(actual, expected)

    def test_shift_engine_jit_save(self, device, dtype):
        import io

        scripted = torch.jit.script(dilation)
        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)
        assert buffer.getbuffer().nbytes > 0

    def test_shift_engine_onnx_trace(self, device, dtype):
        import io

        if device.type != "cpu":
            pytest.skip("the TorchScript-based ONNX export is checked on CPU")
        pytest.importorskip("onnx")

        class Morphology(torch.nn.Module):
            def forward(self, x):
                kernel = torch.ones(3, 3, device=x.device, dtype=x.dtype)
                return dilation(x, kernel, engine="shift")

        tensor = torch.rand(1, 1, 9, 11, device=device, dtype=dtype)
        buffer = io.BytesIO()

        torch.onnx.export(
            Morphology(),
            (tensor,),
            buffer,
            dynamo=False,
        )

        assert buffer.getbuffer().nbytes > 0

    def test_shift_engine_reduces_in_place_only_without_grad(self, device, dtype, monkeypatch):
        calls = []
        original = morphology_module._shift_reduce

        def wrapped(*args, **kwargs):
            calls.append(args[-1])
            return original(*args, **kwargs)

        monkeypatch.setattr(morphology_module, "_shift_reduce", wrapped)

        tensor = torch.rand(1, 1, 9, 11, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        dilation(tensor, kernel, engine="shift")

        with torch.enable_grad():
            dilation(tensor.requires_grad_(True), kernel, engine="shift")

        with torch.no_grad():
            dilation(tensor, kernel, engine="shift")

        assert calls == [True, False, True]

    def test_shift_engine_jit(self, device, dtype):
        op_script = torch.jit.script(dilation)
        tensor = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        assert torch.equal(op_script(tensor, kernel, engine="shift"), dilation(tensor, kernel, engine="shift"))

    def test_even_kernel_centre(self, device, dtype):
        # `dilation` reflects the kernel contents but padded with the unreflected origin, so an
        # even-sized kernel's default-origin window was anchored one pixel toward the top-left of
        # where the Minkowski dilation ``out(p) = max_{q: kernel[q] != 0} x(p - (q - origin))``
        # anchors it. 0/1 fixtures are exact in every dtype (no arithmetic rounding), so this
        # compares with `torch.equal`.
        # Generated with:
        #   x = torch.zeros(1, 1, 6, 7); x[..., 2, 3] = 1
        #   dilation(x, torch.ones(2, 2))
        # scipy 1.17.1 `ndi.grey_dilation(x, footprint=np.ones((2, 2), bool))` and skimage 0.26.0
        # `sm.dilation(x, np.ones((2, 2)))` both give this (fixed) set; OpenCV 5.0.0 `cv2.dilate`
        # does not reflect the kernel and gives {(2, 3), (2, 4), (3, 3), (3, 4)} instead.
        tensor = torch.zeros(1, 1, 6, 7, device=device, dtype=dtype)
        tensor[..., 2, 3] = 1.0
        kernel = torch.ones(2, 2, device=device, dtype=dtype)

        expected = torch.zeros(1, 1, 6, 7, device=device, dtype=dtype)
        expected[..., 1, 2] = 1.0
        expected[..., 1, 3] = 1.0
        expected[..., 2, 2] = 1.0
        expected[..., 2, 3] = 1.0

        assert torch.equal(dilation(tensor, kernel), expected)

    def test_custom_origin_direction(self, device, dtype):
        # With an origin index `o`, dilation collects `x(p - (q - origin))`, so `origin=[0, 0]`
        # must spread the window toward +rows/+cols -- the same direction `erosion(...,
        # origin=[0, 0])` already uses (erosion at `p` reads rows/cols `p..p+k-1`). Pre-fix,
        # `dilation` spread it toward -rows/-cols instead. 0/1 fixtures are exact in every dtype,
        # so this compares with `torch.equal`.
        # Generated with:
        #   x = torch.zeros(1, 1, 6, 7); x[..., 2, 3] = 1
        #   dilation(x, torch.ones(3, 3), origin=[0, 0])
        tensor = torch.zeros(1, 1, 6, 7, device=device, dtype=dtype)
        tensor[..., 2, 3] = 1.0
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        expected = torch.zeros(1, 1, 6, 7, device=device, dtype=dtype)
        expected[..., 2:5, 3:6] = 1.0

        assert torch.equal(dilation(tensor, kernel, origin=[0, 0]), expected)

    def test_erosion_duality_custom_origin(self, device, dtype):
        # erosion(x, B, origin=o) == -dilation(-x, B.flip((0, 1)), origin=[k_h-1-o0, k_w-1-o1])
        # must hold exactly for every origin, not just an odd kernel's centre. Both sides only
        # ever select an already-present value of `x` (max/min, never interpolation), and
        # negation is exact in every dtype, so `torch.equal` is fine here too. Pre-fix, this held
        # only for an odd kernel's centre origin and failed for every other origin.
        # Generated with:
        #   torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0))
        #   L = torch.tensor([[0., 0., 0.], [0., 1., 1.], [0., 1., 0.]])
        # A local `torch.Generator` avoids touching the process-global (and any device) RNG state.
        tensor = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        l_kernel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        square_kernel = torch.ones(2, 2, device=device, dtype=dtype)

        for kernel in (l_kernel, square_kernel):
            k_h, k_w = kernel.shape
            flipped = kernel.flip((0, 1))
            for o0 in range(k_h):
                for o1 in range(k_w):
                    origin = [o0, o1]
                    dual_origin = [k_h - 1 - o0, k_w - 1 - o1]
                    lhs = erosion(tensor, kernel, origin=origin)
                    rhs = -dilation(-tensor, flipped, origin=dual_origin)
                    assert torch.equal(lhs, rhs), (tuple(kernel.shape), origin)

    def test_odd_kernel_default_origin_unchanged(self, device, dtype):
        # An odd kernel with the default (centred) origin pads symmetrically either way
        # (`se_w - origin[1] - 1 == origin[1]` when `origin[1] == (se_w - 1) // 2`), so the fix
        # must leave this case bitwise unchanged. This literal was captured from `dilation` on
        # unmodified `main` (commit 0e8cf3b57), before the pad-line fix was applied, so it also
        # pins the pre-fix value. 0/1 fixtures are exact in every dtype, so this compares with
        # `torch.equal`.
        # Generated with:
        #   L = torch.tensor([[0., 0., 0.], [0., 1., 1.], [0., 1., 0.]])
        #   x = torch.zeros(1, 1, 6, 7); x[..., 2, 3] = 1
        #   dilation(x, L)
        tensor = torch.zeros(1, 1, 6, 7, device=device, dtype=dtype)
        tensor[..., 2, 3] = 1.0
        l_kernel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)

        expected = torch.zeros(1, 1, 6, 7, device=device, dtype=dtype)
        expected[..., 2, 3] = 1.0
        expected[..., 2, 4] = 1.0
        expected[..., 3, 3] = 1.0

        assert torch.equal(dilation(tensor, l_kernel), expected)

    @pytest.mark.parametrize("engine", ["unfold", "convolution"])
    def test_even_kernel_centre_non_square(self, device, dtype, engine):
        # A rectangular kernel's rows and columns must each be padded with their OWN axis's
        # reflected origin: `se_h`/`origin[0]` (rows) must never leak into the width pad slot, or
        # `se_w`/`origin[1]` (cols) into the height pad slot. `ones(2, 2)`, `ones(3, 3)` and the L
        # kernel used elsewhere in this file are square (or symmetric enough) that a pad list with
        # rows and columns swapped still reproduces every one of their pins; only a genuinely
        # rectangular kernel with an off-centre hot pixel catches a row/col swap.
        # Derived from the Minkowski rule with the default origin [k_h // 2, k_w // 2]:
        #   out(p) = max_{q in kernel} x(p - (q - origin))
        # so the nonzero set is {p0 + q - origin : q in kernel}, clipped to the image bounds.
        # Confirmed against the fixed library. Generated with:
        #   x = torch.zeros(1, 1, 6, 7); x[..., 1, 4] = 1
        #   dilation(x, torch.ones(1, 4))  -> row 1, cols 2..5 (origin=[0, 2])
        #   dilation(x, torch.ones(4, 1))  -> rows 0..2, col 4 (origin=[2, 0]; row -1 clipped)
        tensor = torch.zeros(1, 1, 6, 7, device=device, dtype=dtype)
        tensor[..., 1, 4] = 1.0

        wide_kernel = torch.ones(1, 4, device=device, dtype=dtype)
        wide_nonzero = [[1, 2], [1, 3], [1, 4], [1, 5]]
        tall_kernel = torch.ones(4, 1, device=device, dtype=dtype)
        tall_nonzero = [[0, 4], [1, 4], [2, 4]]

        for kernel, expected_nonzero in ((wide_kernel, wide_nonzero), (tall_kernel, tall_nonzero)):
            actual = dilation(tensor, kernel, engine=engine)
            expected = torch.zeros(1, 1, 6, 7, device=device, dtype=dtype)
            for row, col in expected_nonzero:
                expected[..., row, col] = 1.0

            if engine == "convolution":
                # Compare at the harness's per-dtype tolerance rather than a strict nonzero() index list.
                self.assert_close(actual, expected)
            else:
                assert sorted(actual.nonzero()[:, 2:].tolist()) == expected_nonzero

    def test_convention_dilation_reflects_kernel(self, device, dtype):
        # `dilation` reflects the kernel: `out(p) = max_{q: kernel[q] != 0} x(p - (q - origin))`. The kernel
        # is asymmetric because `ones(3, 3)` cannot tell the two conventions apart.
        # Generated with scipy 1.17.1 / numpy 2.0.0 (scikit-image and OpenCV do not reflect: cols {2, 3}):
        #   x = np.zeros((1, 7), np.float32); x[0, 3] = 1.0; A = np.array([[0, 1, 1]], bool)
        #   ndi.grey_dilation(x, footprint=A, mode="constant", cval=-np.inf) -> [0, 0, 0, 1, 1, 0, 0]
        tensor = torch.zeros(1, 1, 1, 7, device=device, dtype=dtype)
        tensor[..., 3] = 1.0
        kernel = torch.tensor([[0.0, 1.0, 1.0]], device=device, dtype=dtype)

        expected = torch.zeros(1, 1, 1, 7, device=device, dtype=dtype)
        expected[..., 3] = 1.0
        expected[..., 4] = 1.0

        # The default engine (`shift` on CPU and MPS without autograd) reflects too.
        self.assert_close(dilation(tensor, kernel), expected)
        self.assert_close(dilation(tensor, kernel, engine="unfold"), expected)
        self.assert_close(dilation(tensor, kernel, engine="convolution"), expected)

    def test_convention_dilation_non_flat_se_is_additive_and_reflected(self, device, dtype):
        # `structuring_element` is the additive (grey-level) part: the neighbourhood value `SE[q]` is
        # ADDED to `x` before the max (and SUBTRACTED before the min in `erosion`), while `kernel` only
        # selects the members. It is reflected in `dilation` together with the kernel, so the peak of
        # `SE = [0, 0.5, 1]` lands to the RIGHT of the hot pixel.
        # Every literal below is a multiple of 0.5 and therefore exact in float16/bfloat16 too.
        # Generated with scipy 1.17.1 / numpy 2.0.0:
        #   x = np.zeros((1, 7), np.float32); x[0, 3] = 1.0; SE = np.array([[0, .5, 1.]])
        #   ndi.grey_dilation(x, structure=SE, footprint=np.ones((1, 3), bool),
        #                     mode="constant", cval=-np.inf)   -> [0.5, 1, 1, 1.5, 2, 1, 1]
        #   ndi.grey_erosion(x + 5, structure=SE, footprint=np.ones((1, 3), bool),
        #                    mode="constant", cval=np.inf)     -> [4, 4, 4.5, 4, 4, 4, 4.5]
        # scikit-image and OpenCV have no non-flat structuring element at all.
        tensor = torch.zeros(1, 1, 1, 7, device=device, dtype=dtype)
        tensor[..., 3] = 1.0
        kernel = torch.ones(1, 3, device=device, dtype=dtype)
        structuring_element = torch.tensor([[0.0, 0.5, 1.0]], device=device, dtype=dtype)

        dilated = dilation(tensor, kernel, structuring_element=structuring_element)
        self.assert_close(
            dilated,
            torch.tensor([[[[0.5, 1.0, 1.0, 1.5, 2.0, 1.0, 1.0]]]], device=device, dtype=dtype),
        )

        eroded = erosion(tensor + 5.0, kernel, structuring_element=structuring_element)
        self.assert_close(
            eroded,
            torch.tensor([[[[4.0, 4.0, 4.5, 4.0, 4.0, 4.0, 4.5]]]], device=device, dtype=dtype),
        )

    def test_convention_kernel_is_a_membership_mask(self, device, dtype):
        # `kernel` is a flat membership mask tested with `!= 0`: negative and fractional entries are
        # members just like a 1, and their magnitude is ignored (use `structuring_element` for weights).
        # Entries of `structuring_element` under a zero `kernel` cell never reach the output.
        tensor = torch.rand(1, 2, 5, 6, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        reference_kernel = torch.tensor([[1.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected_dilation = dilation(tensor, reference_kernel)
        expected_erosion = erosion(tensor, reference_kernel)

        for entry in (-1.0, 0.5, 2.0):
            kernel = torch.tensor([[entry, 1.0, 0.0]], device=device, dtype=dtype)
            assert torch.equal(dilation(tensor, kernel), expected_dilation), entry
            assert torch.equal(erosion(tensor, kernel), expected_erosion), entry

        masked_kernel = torch.tensor([[0.0, 1.0, 1.0]], device=device, dtype=dtype)
        loud = torch.tensor([[99.0, 0.0, 0.0]], device=device, dtype=dtype)
        assert torch.equal(dilation(tensor, masked_kernel, structuring_element=loud), dilation(tensor, masked_kernel))
        assert torch.equal(erosion(tensor, masked_kernel, structuring_element=loud), erosion(tensor, masked_kernel))

    def test_convention_engines_agree_independent_of_max_val(
        self, device, dtype, cudnn_tf32_follows_option
    ):
        # The convolution and unfold engines should agree independently of the finite `max_val`.
        # `cudnn_tf32_follows_option` keeps CUDA's float32 `conv2d` out of TF32 unless `--tf32` is passed.
        tensor = torch.rand(1, 1, 9, 11, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        for max_val in (1.0, 1e2, 1e4):
            unfolded = dilation(tensor, kernel, max_val=max_val, engine="unfold")
            convolved = dilation(tensor, kernel, max_val=max_val, engine="convolution")
            assert torch.equal(convolved, unfolded)

    def test_dilation_ignores_finite_max_val_sentinel_4734(self, device):
        # Geodesic padding and masked kernel cells must be excluded rather than represented by a finite
        # sentinel. This keeps the result independent of the image range and avoids the convolution precision
        # loss caused by passing a finite sentinel through `conv2d`. float32 only: the literals
        # below are about the float32 default `max_val=1e4` (40000 is not representable in bfloat16).
        dtype = torch.float32
        single = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        side_kernel = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)

        # The window is empty at this origin, so dilation/erosion return their reduction identities.
        assert dilation(single, side_kernel, origin=[0, 2]).item() == float("-inf")
        assert erosion(single, side_kernel, origin=[0, 2]).item() == float("inf")

        big = torch.tensor([[0.0, 5e4, 0.0]], device=device, dtype=dtype)[None, None]
        gapped_kernel = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=dtype)
        assert dilation(big, gapped_kernel).flatten().tolist() == [50000.0, 0.0, 50000.0]
        # The result is independent of the finite `max_val` because excluded cells are masked out.
        assert dilation(big, gapped_kernel, max_val=1e6).flatten().tolist() == [50000.0, 0.0, 50000.0]

        negative = torch.tensor([[-5e4, -6e4]], device=device, dtype=dtype)[None, None]
        assert dilation(negative, torch.ones(1, 3, device=device, dtype=dtype)).flatten().tolist() == [
            -50000.0,
            -50000.0,
        ]

        # A masked-out in-image cell must be excluded just like an out-of-image cell.
        corner_kernel = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
        minus_two = torch.full((1, 1, 1, 1), -2.0, device=device, dtype=dtype)
        assert erosion(minus_two, corner_kernel, origin=[0, 0]).item() == float("inf")
        five = torch.full((1, 1, 1, 1), 5.0, device=device, dtype=dtype)
        assert dilation(five, corner_kernel, origin=[0, 0]).item() == float("-inf")
        # The same masking applies when a structuring element is given.
        flat_corner = torch.zeros_like(corner_kernel)
        assert erosion(minus_two, corner_kernel, structuring_element=flat_corner, origin=[0, 0]).item() == float("inf")
        assert dilation(five, corner_kernel, structuring_element=flat_corner, origin=[0, 0]).item() == float("-inf")

        # Empty windows now use the reduction identities rather than a finite `max_val` sentinel.
        dilated = dilation(minus_two, corner_kernel, origin=[0, 0], max_val=1.0)
        eroded = erosion(minus_two, corner_kernel, origin=[0, 0], max_val=1.0)
        assert dilated.item() == float("-inf")
        assert eroded.item() == float("inf")

        # Geodesic padding ignores pixels outside the image; constant padding still uses `border_value`.
        # A non-flat structuring element can therefore still make an outside position contribute through its
        # actual structuring-element value, while an all-zero kernel has an empty window.
        ramp = torch.tensor([[3.0, 4.0, 5.0]], device=device, dtype=dtype)[None, None]
        assert dilation(ramp, corner_kernel, origin=[0, 0]).flatten().tolist() == [float("-inf"), 3.0, 4.0]
        constant_ramp = dilation(ramp, corner_kernel, origin=[0, 0], border_type="constant")
        assert constant_ramp.flatten().tolist() == [0.0, 3.0, 4.0]
        valued_ramp = dilation(ramp, corner_kernel, origin=[0, 0], border_type="constant", border_value=7.0)
        assert valued_ramp.flatten().tolist() == [7.0, 3.0, 4.0]
        tall_se = torch.tensor([[0.0, 0.0, 15000.0]], device=device, dtype=dtype)
        leaked = dilation(
            torch.zeros(1, 1, 1, 4, device=device, dtype=dtype),
            torch.ones(1, 3, device=device, dtype=dtype),
            structuring_element=tall_se,
        )
        assert leaked.flatten().tolist() == [0.0, 15000.0, 15000.0, 15000.0]
        assert erosion(five, torch.zeros(1, 1, device=device, dtype=dtype)).item() == float("inf")

        # `max_val` no longer affects geodesic padding or masked-out cells.
        half_image = torch.rand(1, 1, 4, 4, generator=torch.Generator().manual_seed(0)).to(device, torch.float16)
        half_cross = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float16
        )
        assert torch.equal(dilation(half_image, half_cross, max_val=65504.0), dilation(half_image, half_cross))

    @pytest.mark.xfail(strict=True, reason="a non-float image is not rejected (#4735)")
    def test_wart_non_float_image_is_not_rejected_4735(self, device):
        # Only floating-point images are supported; kornia should reject the others. A `uint8` or `bool`
        # image under `border_type="constant"` runs today on every backend, so this XPASSes when #4735 adds
        # the check. Dtypes are explicit, so this pin takes `device` only.
        float_kernel = torch.ones(1, 3, device=device)
        for image_dtype in (torch.uint8, torch.bool):
            image = torch.zeros(1, 1, 1, 5, dtype=image_dtype, device=device)
            for op in (dilation, erosion):
                with pytest.raises((TypeError, ValueError)):
                    op(image, float_kernel, border_type="constant")

    def test_convention_mask_kernel_with_structuring_element(self, device, dtype):
        # With a floating structuring element the sentinel is stored there, the kernel is only the
        # `kernel == 0` mask, and a `uint8` or `bool` kernel returns what the floating kernel does.
        frame = torch.rand(1, 1, 4, 5, generator=torch.Generator().manual_seed(0)).to(device, dtype)
        flat_se = torch.zeros(1, 3, device=device, dtype=dtype)
        float_kernel = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=dtype)
        for mask_dtype in (torch.uint8, torch.bool):
            for op in (dilation, erosion):
                assert torch.equal(
                    op(frame, float_kernel.to(mask_dtype), structuring_element=flat_se),
                    op(frame, float_kernel, structuring_element=flat_se),
                ), (mask_dtype, op.__name__)

    def test_wart_convolution_engine_keeps_image_dtype_4762(self, device):
        # `unfold` and `shift` return the promoted dtype of the image and the kernel (or the structuring
        # element); `engine="convolution"` casts the kernel to the image's dtype instead, so the result dtype
        # depends on the engine. Tracked in #4762. Dtypes are explicit, so this pin takes `device` only.
        tensor = torch.rand(1, 1, 4, 5, generator=torch.Generator().manual_seed(0)).to(device, torch.float16)
        kernel = torch.ones(3, 3, device=device)
        for op in (dilation, erosion):
            assert op(tensor, kernel, engine="unfold").dtype == torch.float32, op.__name__
            assert op(tensor, kernel, engine="shift").dtype == torch.float32, op.__name__
            assert op(tensor, kernel, engine="convolution").dtype == torch.float16, op.__name__

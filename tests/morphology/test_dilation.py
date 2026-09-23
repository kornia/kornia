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

from kornia.morphology import dilation, erosion, gradient
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
        # The convolution engine measures 3.9065e-4 absolute / 4.3405e-4 relative error on THIS
        # fixture (re-measured for `dilation`; `erosion`'s figures differ, see test_erosion.py),
        # above the harness's generic float32 default (atol=1e-5, rtol=1e-4). The cause is the geodesic
        # `-max_val` pad, which `engine="convolution"` feeds through `F.conv2d` together with the image,
        # so the error scales with `max_val` rather than with the image range. The conv bias, which holds
        # `-max_val` for the masked cells, is not the cause: under `border_type="replicate"` the same
        # fixture's gap is 5.96e-8. The backend matters too: CPU float32 shows the gap and MPS measures
        # exactly 0 on this fixture (torch 2.14.0). One float32 ULP at the default `max_val=1e4` is
        # 9.7656e-4, and the largest engine gap measured (rand(1, 1, 9, 11) seed 0, ones(3, 3)) is
        # 1.2736e-3, i.e. ~1.3 ULP. `engine="unfold"` is exact. Tracked in #4734; this tolerance is scoped
        # to the convolution engine.
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
        # See test_kernel: the convolution engine needs an explicit tolerance because the geodesic
        # `-max_val` pad it feeds through `F.conv2d` costs on the order of one ULP of `max_val`
        # (9.7656e-4 in float32 at the default `max_val=1e4`) on CPU. Tracked in #4734.
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
                # See test_kernel: `engine="convolution"` feeds the geodesic `-max_val` pad through
                # `F.conv2d`, so the gap between the engines scales with `max_val` rather than with the
                # image range (#4734), on the backends whose conv shows it at all (CPU float32, not MPS).
                # On this 0/1 fixture the gap measures exactly 0 on CPU in float32, float64, float16
                # and bfloat16 and on MPS in float32, float16 and bfloat16 (torch 2.14.0); CUDA is
                # unmeasured, so this compares the full tensor at the harness's default per-dtype
                # tolerance rather than a strict nonzero() index list.
                self.assert_close(actual, expected)
            else:
                assert sorted(actual.nonzero()[:, 2:].tolist()) == expected_nonzero

    def test_convention_dilation_reflects_kernel(self, device, dtype):
        # `dilation` is the Minkowski dilation `out(p) = max_{q: kernel[q] != 0} x(p - (q - origin))`,
        # so the kernel *contents* are reflected. The asymmetric kernel is the whole point: `ones(3, 3)`
        # is invariant under the flip and cannot tell the two conventions apart.
        # The fixture is 0/1 and the engine gap on it measures exactly 0 on CPU in all four dtypes
        # and on MPS in float32/float16/bfloat16 (torch 2.14.0), but CUDA is unmeasured and the gap is
        # a geodesic-pad effect that scales with `max_val` (#4734), so this compares at the harness's
        # default per-dtype tolerance, like the neighbouring convolution branch above.
        # Generated with (scipy 1.17.1, scikit-image 0.26.0, opencv-python-headless 5.0.0, numpy 2.0.0):
        #   x = np.zeros((1, 7), np.float32); x[0, 3] = 1.0; A = np.array([[0, 1, 1]], bool)
        #   ndi.grey_dilation(x, footprint=A, mode="constant", cval=-np.inf)
        #       -> [0, 0, 0, 1, 1, 0, 0]   (cols {3, 4}; same as kornia)
        #   sm.dilation(x, A, mode="ignore") -> [0, 0, 1, 1, 0, 0, 0]   (cols {2, 3})
        #   cv2.dilate(x, A.astype(np.uint8)) -> [0, 0, 1, 1, 0, 0, 0]  (cols {2, 3})
        # Neither reference reflects. scikit-image returns kornia's dilation by the flipped kernel at
        # every kernel size; OpenCV does so only for an odd-sized kernel, because it also anchors an
        # even-sized kernel one cell earlier. Measured on
        # `torch.rand(7, 9, generator=torch.Generator().manual_seed(0), dtype=torch.float64)`, comparing
        # each reference at its default anchor against `dilation(x, K.flip((0, 1)))`:
        #   K = ones(1, 3) / [[0, 1, 1]]  (odd): skimage 0, cv2 0 at the default origin
        #   K = [[1, 0]] / ones(2, 2) / [[1, 1, 0, 1]]  (even): skimage 0 at the default origin, while
        #       cv2 differs there (`ones(2, 2)` 0.825, `[[1, 1, 0, 1]]` 0.732, `[[1, 0]]` 0.956 plus a
        #       -DBL_MAX column 0) and is 0 at `origin=[(k_h - 1) // 2, (k_w - 1) // 2]` -- except that for
        #       `[[1, 0]]` column 0 is then an empty window, where cv2 keeps -DBL_MAX and kornia returns
        #       `-max_val + x[:, 0]` (-9999.71 to -9999.03).
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
        # Generated with scikit-image 0.26.0 / scipy 1.17.1 / numpy 2.0.0 on the hot-pixel row:
        #   ndi.grey_dilation(x, footprint=np.array([[-1, 1, 0]]), ...) -> [0, 0, 1, 1, 0, 0, 0]
        #       (2 members; scipy reflects too, so this is kornia's own output)
        #   sm.dilation(x, np.array([[-1, 1, 0]]), mode="ignore")       -> [0, 0, 0, 1, 1, 0, 0]
        # scikit-image KEEPS the -1 cell: its `[[-1, 1, 0]]` output is bit-equal to its `[[1, 1, 0]]`
        # output and differs from its `[[0, 1, 0]]` output ([0, 0, 0, 1, 0, 0, 0]), which is what
        # dropping the cell would give. The remaining difference from kornia is the reflection --
        # cols {3, 4} there against cols {2, 3} here -- not the membership rule.
        # Entries of `structuring_element` under a zero `kernel` cell are overwritten by the
        # `-max_val` sentinel and never reach the output: with a zero structuring element elsewhere the
        # result is the flat one, in `dilation` and in `erosion` alike.
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

    def test_convention_engines_agree_within_a_few_ulps_of_max_val(self, device, dtype, cudnn_tf32_follows_option):
        # Both engines implement the same operation. `engine="unfold"` is exact; `engine="convolution"`
        # feeds the geodesic `-max_val` pad through `F.conv2d` together with the image, so on CPU its error
        # scales with `max_val` rather than with the image range (`ones(3, 3)` has no masked cell, so the
        # conv bias is all zeros here and plays no part). The exact gap is platform-dependent --
        # only CPU and MPS on torch 2.14.0 were measured here, and CUDA's `conv2d` picks a different
        # algorithm -- so this pins a BOUND, not the value, and the bound carries slack (4 eps rather
        # than the 2 eps that would just fit the measurement) so that one extra ULP on an unmeasured
        # backend or on the torch 2.5.1 floor does not red the pin. `cudnn_tf32_follows_option` keeps
        # CUDA's float32 `conv2d` out of TF32 unless `--tf32` is passed; TF32's 10-bit mantissa would swamp
        # a bound in float32 ULPs.
        # Tracked in #4734.
        # Measured (CPU, torch 2.14.0, float32, x = rand(1, 1, 9, 11) seed 0, kernel ones(3, 3)):
        #   max_val=1     -> 1.1921e-07    max_val=1e2 -> 7.1526e-06    max_val=1e4 -> 1.2736e-03
        # i.e. 1.0, 0.94 and 1.3 float32 ULPs of `max_val` (ULP(1) = 1.1921e-07, ULP(1e2) = 7.6294e-06,
        # ULP(1e4) = 9.7656e-04). float64, float16 and bfloat16 measured 0 at all three values.
        tensor = torch.rand(1, 1, 9, 11, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        for max_val in (1.0, 1e2, 1e4):
            unfolded = dilation(tensor, kernel, max_val=max_val, engine="unfold")
            convolved = dilation(tensor, kernel, max_val=max_val, engine="convolution")
            self.assert_close(convolved, unfolded, atol=4.0 * torch.finfo(dtype).eps * max_val, rtol=0.0)

    def test_wart_dilation_max_val_sentinel_leaks_4734(self, device):
        # `max_val` is a finite stand-in for infinity, not an infinity: it is SUBTRACTED from the
        # masked-out neighbourhood cells and, under the default geodesic border, padded into the border,
        # so it reaches the output whenever the window is empty or the range of the image plus the
        # structuring element approaches `max_val`. scipy and scikit-image's `mode="ignore"` return a true
        # -inf/+inf in the empty window and the true dilation in the range cases. Tracked in #4734.
        # float32 only: 40000 is not representable in bfloat16 and the claim is about the float32
        # default `max_val=1e4`, so the dtype is explicit rather than taken from the fixture.
        # Generated with kornia in this worktree (torch 2.14.0, CPU, float32):
        #   dilation(torch.ones(1, 1, 1, 1), [[1., 0., 0.]], origin=[0, 2]).item()  -> -9999.0
        #   erosion(torch.ones(1, 1, 1, 1), [[1., 0., 0.]], origin=[0, 2]).item()   ->  10000.0
        #   dilation([[0., 5e4, 0.]], [[1., 0., 1.]])  -> [50000., 40000., 50000.]  (true: [5e4, 0, 5e4])
        #   dilation([[-5e4, -6e4]], torch.ones(1, 3)) -> [-10000., -10000.]        (true: [-5e4, -5e4])
        dtype = torch.float32
        single = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        side_kernel = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)

        # The window is empty at this origin, so the output is the sentinel plus the pixel, not the pixel.
        assert dilation(single, side_kernel, origin=[0, 2]).item() == -9999.0
        assert erosion(single, side_kernel, origin=[0, 2]).item() == 10000.0

        big = torch.tensor([[0.0, 5e4, 0.0]], device=device, dtype=dtype)[None, None]
        gapped_kernel = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=dtype)
        assert dilation(big, gapped_kernel).flatten().tolist() == [50000.0, 40000.0, 50000.0]
        # With a `max_val` above the image range the same call is correct, which is the diagnosis.
        assert dilation(big, gapped_kernel, max_val=1e6).flatten().tolist() == [50000.0, 0.0, 50000.0]

        negative = torch.tensor([[-5e4, -6e4]], device=device, dtype=dtype)[None, None]
        assert dilation(negative, torch.ones(1, 3, device=device, dtype=dtype)).flatten().tolist() == [
            -10000.0,
            -10000.0,
        ]

        # An empty window is not `max_val` either: a masked-out in-image cell contributes `x + max_val`
        # to `erosion` (`x - max_val` to `dilation`), so a negative pixel under it pulls the result below
        # the sentinel. Generated with kornia in this worktree (torch 2.14.0, CPU and MPS, float32):
        #   erosion([[-2.]], [[0., 1.]], origin=[0, 0])  -> 9998.0   (scipy, cval=inf, origin=(0, -1): inf)
        #   dilation([[5.]], [[0., 1.]], origin=[0, 0])  -> -9995.0  (scipy, cval=-inf, origin=(0, -1): -inf)
        # The sum is `s + m` with `s` the value the dtype STORES for `max_val`, rounded once: bfloat16 stores
        # 1e4 as 9984, so -34 gives 9920 (9984 - 34 = 9950 rounds down), not 9984 (1e4 - 34 = 9966 rounds up).
        # Generated with kornia in this worktree (torch 2.14.0 and 2.5.1, CPU and MPS):
        #   erosion([[-34.]], [[0, 1]], origin=[0, 0]) -> bfloat16 9920, float16 9968, float32/float64 9966
        #   dilation([[34.]], [[0, 1]], origin=[0, 0]) -> the negatives of those
        corner_kernel = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
        minus_two = torch.full((1, 1, 1, 1), -2.0, device=device, dtype=dtype)
        assert erosion(minus_two, corner_kernel, origin=[0, 0]).item() == 9998.0
        five = torch.full((1, 1, 1, 1), 5.0, device=device, dtype=dtype)
        assert dilation(five, corner_kernel, origin=[0, 0]).item() == -9995.0
        # This test fixes `dtype` to float32 above, so the rounding claim loops over the dtypes itself.
        stored_minus_34 = {torch.bfloat16: 9920.0, torch.float16: 9968.0, torch.float32: 9966.0, torch.float64: 9966.0}
        for sum_dtype, expected_sum in stored_minus_34.items():
            if sum_dtype == torch.float64 and device.type == "mps":
                continue
            minus_34 = torch.full((1, 1, 1, 1), -34.0, device=device, dtype=sum_dtype)
            assert erosion(minus_34, corner_kernel.to(sum_dtype), origin=[0, 0]).item() == expected_sum, sum_dtype
            assert dilation(-minus_34, corner_kernel.to(sum_dtype), origin=[0, 0]).item() == -expected_sum, sum_dtype
        # The same sentinel is stored into a structuring element when one is given.
        flat_corner = torch.zeros_like(corner_kernel)
        assert erosion(minus_two, corner_kernel, structuring_element=flat_corner, origin=[0, 0]).item() == 9998.0
        assert dilation(five, corner_kernel, structuring_element=flat_corner, origin=[0, 0]).item() == -9995.0

        # The same empty window breaks the adjunction `dilation(x) <= y  <=>  x <= erosion(y)`, which holds
        # while no window is empty (see test_convention_adjunction_without_empty_windows in test_erosion.py).
        #   max_val=1, x = y = [[-2.]]: dilation(x) -> -1.0, erosion(y) -> -1.0
        dilated = dilation(minus_two, corner_kernel, origin=[0, 0], max_val=1.0)
        eroded = erosion(minus_two, corner_kernel, origin=[0, 0], max_val=1.0)
        assert dilated.item() == -1.0
        assert eroded.item() == -1.0
        assert not bool((dilated <= minus_two).all())
        assert bool((minus_two <= eroded).all())

        # Only the geodesic border is padded with the sentinel (`constant` pads `border_value`); a non-flat
        # structuring element that reaches `max_val` lets that border back in even on a zero image; and an
        # all-zero kernel has no member in the pad, so its empty window is `x + max_val`, not
        # `max_val + min(0, x)`. Generated with kornia in this worktree (torch 2.14.0, CPU and MPS, float32):
        #   dilation([[3., 4., 5.]], [[0., 1.]], origin=[0, 0])              -> [-9997, 3, 4]
        #   dilation(..., border_type="constant")                            -> [0, 3, 4]
        #   dilation(zeros(1, 1, 1, 4), ones(1, 3), structuring_element=[[0., 0., 15000.]])
        #       -> [5000, 15000, 15000, 15000]   (scipy, cval=-inf: [0, 15000, 15000, 15000])
        #   erosion([[5.]], [[0.]])                                          -> 10005.0
        ramp = torch.tensor([[3.0, 4.0, 5.0]], device=device, dtype=dtype)[None, None]
        assert dilation(ramp, corner_kernel, origin=[0, 0]).flatten().tolist() == [-9997.0, 3.0, 4.0]
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
        assert leaked.flatten().tolist() == [5000.0, 15000.0, 15000.0, 15000.0]
        assert erosion(five, torch.zeros(1, 1, device=device, dtype=dtype)).item() == 10005.0

        # float16 cannot hold a `max_val` above 65504, and what happens depends on where the sentinel is stored.
        # The geodesic pad of a float16 image raises on CPU and CUDA but rounds to -65504 on MPS. Storing it
        # into a float16 kernel raises on MPS and on CPU with torch 2.5.1, and rounds on CPU and CUDA with
        # torch 2.14 (to -65504 below 65520, to -inf from there), which leaves the value unchanged. Generated
        # with kornia in this worktree (torch 2.14.0 and 2.5.1, CPU and MPS; torch 2.14.0, CUDA 13.0):
        #   dilation(half image, ones(3, 3) float32, max_val=65510.)    CPU, CUDA -> overflow; MPS -> runs
        #   dilation(half image, [[1., 0., 1.]] half, max_val=65510., "constant")
        #       CPU 2.14, CUDA 2.14 -> runs; CPU 2.5.1 and MPS -> overflow
        half_image = torch.rand(1, 1, 4, 4, generator=torch.Generator().manual_seed(0)).to(device, torch.float16)
        # A float32 kernel keeps the kernel store out of float16, so only the pad is exercised here, and the
        # window of `[[0, 1]]` at origin [0, 0] on a 1x1 image is empty, so the result shows the pad itself:
        # -65504 on MPS, where the masked in-image cell contributes -65510 in float32 and a -inf pad would lose.
        # The pad is a backend cast, so other backends are left unasserted until someone measures them.
        empty_window = torch.zeros(1, 1, 1, 1, device=device, dtype=torch.float16)
        corner = torch.tensor([[0.0, 1.0]], device=device)
        if device.type == "mps":
            assert dilation(empty_window, corner, origin=[0, 0], max_val=65510.0).item() == -65504.0
        elif device.type in ("cpu", "cuda"):
            with pytest.raises(RuntimeError, match="overflow"):
                dilation(empty_window, corner, origin=[0, 0], max_val=65510.0)
        half_gapped = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=torch.float16)
        above = None
        try:
            above = dilation(half_image, half_gapped, max_val=65510.0, border_type="constant")
        except RuntimeError as err:
            assert "overflow" in str(err), str(err)
        else:
            assert torch.equal(above, dilation(half_image, half_gapped, max_val=65504.0, border_type="constant"))
        # A `bool` or integer kernel on a float16 image is given the image's dtype (#4744), so it stores the
        # sentinel in float16 as well and takes the same branch as the float16 kernel above.
        for mask_dtype in (torch.bool, torch.int64):
            try:
                above_mask = dilation(half_image, half_gapped.to(mask_dtype), max_val=65510.0, border_type="constant")
            except RuntimeError as err:
                assert "overflow" in str(err), str(err)
                assert above is None, mask_dtype
            else:
                assert above is not None and torch.equal(above_mask, above), mask_dtype
        # The store converts the scalar before it looks for a zero cell, so a kernel of all ones, which stores
        # nothing, takes that same branch: it raises where the gapped kernel raises and runs where it runs.
        half_full = torch.ones(1, 3, device=device, dtype=torch.float16)
        try:
            above_full = dilation(half_image, half_full, max_val=65510.0, border_type="constant")
        except RuntimeError as err:
            assert "overflow" in str(err), str(err)
            assert above is None
        else:
            # Under `constant` an all-ones kernel never lets `max_val` into the output, so the branch is the pin.
            assert above is not None
            assert above_full.dtype == torch.float16

    def test_wart_integer_and_bool_input_4735(self, device):
        # The `max_val` sentinel is written into a tensor of the INPUT's dtype, so only floating-point
        # input is supported. `uint8` does not survive the GEODESIC pad (which stores -/+ max_val);
        # under the other `border_type` values the call runs and silently returns `float32`. `int64` is
        # silently wrong near `max_val`. `bool` is NOT silently all-`True`: on CPU the geodesic pad is `True`
        # in `bool`, so the correct dilation comes back with a `True` border ring, as wide on each side as
        # the kernel's members reach past that edge (the whole pad for a rectangle of ones). The
        # ring alone fills only an image no larger than itself (an all-False 1x5 keeps three False pixels);
        # the 1x5 of the issue is all `True` because the ring and its centre pixel's dilation cover it.
        # scipy, scikit-image and OpenCV all accept these dtypes. Tracked in #4735. Dtypes are
        # explicit here (the claim is about non-float dtypes), so this pin takes `device` only.
        # Generated with kornia in this worktree (torch 2.14.0):
        #   CPU, CUDA: dilation(torch.zeros(1, 1, 1, 5, dtype=torch.uint8), torch.ones(1, 3))
        #        -> RuntimeError: value cannot be converted to type uint8_t without overflow
        #   MPS: the same call does NOT raise; the store wraps (-1e4 mod 256 == 240) and returns
        #        float32 [240, 0, 0, 0, 240]. The backends disagree, so the pin asserts the consequence
        #        they share: an all-zero uint8 image does not come back all zero from `dilation`.
        #        `erosion` pads +max_val, which wraps to 16, and a min against 16 happens to leave an
        #        all-zero image alone on MPS -- so only `dilation` is asserted here.
        #   both: dilation(zeros uint8, ones(1, 3), border_type="constant") -> float32 zeros, no raise
        #   x = zeros(1, 1, 1, 9, dtype=bool); x[..., 4] = True   (CPU; MPS differs, see below)
        #   dilation(x, ones(1, 3, dtype=bool))       -> [T,F,F,T,T,T,F,F,T]  (ring + correct interior)
        #   dilation(x, ones(1, 1, dtype=bool))       -> [F,F,F,F,T,F,F,F,F]  (no pad, exact)
        #   dilation(x, ones(1, 3, ...), "constant")  -> [F,F,F,T,T,T,F,F,F]  (False pad, exact)
        #   y = zeros(1, 1, 1, 5, dtype=bool); y[..., 2] = True
        #   dilation(y, ones(1, 3, dtype=bool))       -> [T,T,T,T,T]          (#4735's own case)
        #   erosion(x, ones(1, 3, dtype=torch.bool))  -> NotImplementedError (torch 2.14) / RuntimeError
        #     (torch <= 2.9); the default `shift` engine says "Negation, the `-` operator, on a bool
        #     tensor is not supported", `unfold` "Subtraction ... with two bool tensors is not supported"
        #   dilation([[0, 50000, 0]] int64, [[1, 0, 1]] int64) -> [50000, 40000, 50000] (true: [5e4,0,5e4])
        float_kernel = torch.ones(1, 3, device=device)

        try:
            out_uint8 = dilation(torch.zeros(1, 1, 1, 5, dtype=torch.uint8, device=device), float_kernel)
        except RuntimeError as err:
            assert "overflow" in str(err), str(err)
        else:
            assert out_uint8.abs().max().item() != 0.0

        # Only the geodesic pad stores the sentinel, so `constant` runs -- and silently changes dtype.
        constant_uint8 = dilation(
            torch.zeros(1, 1, 1, 5, dtype=torch.uint8, device=device), float_kernel, border_type="constant"
        )
        assert constant_uint8.dtype == torch.float32
        assert constant_uint8.abs().max().item() == 0.0

        hot_bool = torch.zeros(1, 1, 1, 9, dtype=torch.bool, device=device)
        hot_bool[..., 4] = True
        bool_kernel = torch.ones(1, 3, dtype=torch.bool, device=device)
        # The ring is a CPU and CUDA fact (CUDA measured on torch 2.14.0 only, where it also holds at
        # max_val=9984). MPS stores the raw byte of -max_val mod 256 (240 for 1e4) in the bool pad:
        # torch 2.14 reads it as `True` unless it is 0, torch 2.5.1 as a signed integer, so at the default
        # max_val=1e4 torch 2.5.1 shows no ring (and -16 with a floating kernel), and at max_val=9984 neither
        # release does. Measured with kornia in this worktree (torch 2.14.0 and 2.5.1). Other backends are
        # unmeasured and left unasserted.
        pads_true = device.type in ("cpu", "cuda")
        if device.type == "mps":
            no_ring = dilation(hot_bool, bool_kernel, max_val=9984.0)
            assert no_ring.flatten().tolist() == [False, False, False, True, True, True, False, False, False]
        # The correct dilation is cols {3, 4, 5}; cols 0 and 8 are the `True` ring left by the pad.
        ring_plus_dilation = [True, False, False, True, True, True, False, False, True]
        exact_dilation = [False, False, False, True, True, True, False, False, False]
        if device.type == "mps":
            # At the default `max_val` the pad byte is 240: a ring on torch 2.14, none on 2.5.1, never a third thing.
            mps_default = dilation(hot_bool, bool_kernel).flatten().tolist()
            assert mps_default in (ring_plus_dilation, exact_dilation), mps_default
        if pads_true:
            assert dilation(hot_bool, bool_kernel).flatten().tolist() == ring_plus_dilation
        # A 1x1 kernel needs no pad, a `constant` pad of 0.0 is `False`, and a `circular` pad is the
        # image's own values: all three are exact. (`reflect` and `replicate` raise on a `bool` image on
        # CPU: "reflection_pad2d" / "replication_pad2d" not implemented for 'Bool'.)
        one_cell = torch.ones(1, 1, dtype=torch.bool, device=device)
        hot_pixel_only = [False, False, False, False, True, False, False, False, False]
        assert dilation(hot_bool, one_cell).flatten().tolist() == hot_pixel_only
        assert dilation(hot_bool, bool_kernel, border_type="constant").flatten().tolist() == exact_dilation
        assert dilation(hot_bool, bool_kernel, border_type="circular").flatten().tolist() == exact_dilation
        # The ring alone fills only an image no larger than itself: an all-False 1x5 keeps its three
        # interior pixels and an all-False 1x2 is nothing but ring. #4735's own 1x5 comes back all `True`
        # because the ring (cols 0 and 4) and the true dilation of its centre pixel (cols 1-3) cover it.
        # Generated with kornia in this worktree (torch 2.14.0 and 2.5.1, CPU):
        #   dilation(zeros(1, 1, 1, 5, bool), ones(1, 3, bool)) -> [T, F, F, F, T]
        #   dilation(zeros(1, 1, 1, 2, bool), ones(1, 3, bool)) -> [T, T]
        cold_small = torch.zeros(1, 1, 1, 5, dtype=torch.bool, device=device)
        cold_tiny = torch.zeros(1, 1, 1, 2, dtype=torch.bool, device=device)
        small_bool = cold_small.clone()
        small_bool[..., 2] = True
        # The ring is only as wide as the members reach past each edge, not the pad kornia applies:
        # `[[1, 1, 0, 0, 0]]` is padded by 2 on both sides, but after the reflection its members read
        # `x(p + 1)` and `x(p + 2)`, so the ring is two cells on the right and none on the left.
        # Generated with kornia in this worktree (torch 2.14.0 and 2.5.1, CPU):
        #   dilation(zeros(1, 1, 1, 9, bool), [[1., 1., 0., 0., 0.]]) -> [0, 0, 0, 0, 0, 0, 0, 1, 1]
        one_sided = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]], device=device)
        cold_row = torch.zeros(1, 1, 1, 9, dtype=torch.bool, device=device)
        if pads_true:
            assert dilation(cold_small, bool_kernel).flatten().tolist() == [True, False, False, False, True]
            assert dilation(cold_tiny, bool_kernel).flatten().tolist() == [True, True]
            assert dilation(small_bool, bool_kernel).flatten().tolist() == [True, True, True, True, True]
            assert dilation(cold_row, one_sided).flatten().tolist() == [0.0] * 7 + [1.0, 1.0]

        # torch 2.14 raises NotImplementedError, torch 2.5.1 / 2.9.1 raise RuntimeError, same message.
        with pytest.raises((NotImplementedError, RuntimeError), match="bool"):
            erosion(hot_bool, bool_kernel)

        # The sentinel is also stored into the KERNEL (into the structuring element when one is given), and
        # under `unfold` and `shift` the result takes the promoted dtype of image and kernel, so the image
        # dtype alone does not decide the outcome. A floating-point image lends its dtype to a `bool` or
        # integer kernel (#4744), so only a non-float image stores the sentinel in the kernel's own dtype.
        # Generated with kornia in this worktree (torch 2.14.0 and 2.5.1; CPU and MPS agree on every line except
        # the two ring lines, which MPS on torch 2.5.1 returns without the ring, as the ring comment above
        # records and the `mps_default` assertion admits):
        #   dilation(zeros uint8, ones(1, 3, float16), "constant")       -> float16 zeros (not float32)
        #   dilation(zeros int64, ones(1, 3, uint8), "constant")         -> RuntimeError overflow
        #   dilation(zeros float, ones(1, 3, uint8), "constant")         -> float32 zeros (exact)
        #   dilation([[0, 3, 0, 0, 7]] int64, [[T, F, T]])               -> [3, 4, 3, 7, 8]  (true: [3, 0, 3, 7, 0])
        #   dilation([[0, 3, 0, 0, 7]] float, [[T, F, T]])               -> [3, 0, 3, 7, 0]  (exact)
        #   dilation(hot_bool, [[T, F, T]]), geodesic and "constant"     -> all True
        #   erosion(int64 image, bool kernel)                            -> NotImplementedError / RuntimeError
        #   dilation(hot_bool, ones(1, 3) float)                         -> float32 ring + correct interior
        #   erosion(~hot_bool, [[1., 0., 1.]] float)                     -> float32, exact under geodesic
        #   gradient(hot_bool, ones(1, 3) float)                         -> float32, the dilation's ring
        zeros_uint8 = torch.zeros(1, 1, 1, 5, dtype=torch.uint8, device=device)
        half_kernel = torch.ones(1, 3, dtype=torch.float16, device=device)
        assert dilation(zeros_uint8, half_kernel, border_type="constant").dtype == torch.float16
        int_zeros = torch.zeros(1, 1, 1, 5, dtype=torch.int64, device=device)
        float_zeros = torch.zeros(1, 1, 1, 5, device=device)
        uint8_kernel = torch.ones(1, 3, dtype=torch.uint8, device=device)
        for border_type in ("constant", "circular"):
            with pytest.raises(RuntimeError, match="overflow"):
                dilation(int_zeros, uint8_kernel, border_type=border_type)
            with pytest.raises(RuntimeError, match="overflow"):
                erosion(int_zeros, uint8_kernel, border_type=border_type)
            # The raise is the store of `-max_val` into a dtype that cannot hold it: any positive value for uint8.
            with pytest.raises(RuntimeError, match="overflow"):
                dilation(int_zeros, uint8_kernel, border_type=border_type, max_val=0.5)
            for op in (dilation, erosion):
                float_out = op(float_zeros, uint8_kernel, border_type=border_type)
                assert float_out.dtype == torch.float32
                assert float_out.flatten().tolist() == [0.0] * 5

        # On a non-float image a `False` kernel cell stores -max_val as `True`, so it contributes `x + 1`
        # instead of dropping out. A floating-point image makes the same kernel a plain membership mask.
        sparse_bool_kernel = torch.tensor([[True, False, True]], device=device)
        int_ramp = torch.tensor([[0, 3, 0, 0, 7]], dtype=torch.int64, device=device)[None, None]
        ramp = int_ramp.float()
        assert dilation(int_ramp, sparse_bool_kernel).flatten().tolist() == [3, 4, 3, 7, 8]
        exact_ramp = [3.0, 0.0, 3.0, 7.0, 0.0]
        assert dilation(ramp, torch.tensor([[1.0, 0.0, 1.0]], device=device)).flatten().tolist() == exact_ramp
        assert dilation(ramp, sparse_bool_kernel).flatten().tolist() == exact_ramp
        for border_type in ("geodesic", "constant"):
            assert dilation(hot_bool, sparse_bool_kernel, border_type=border_type).all()
        with pytest.raises((NotImplementedError, RuntimeError), match="bool"):
            erosion(int_ramp, sparse_bool_kernel)
        assert erosion(ramp, sparse_bool_kernel).flatten().tolist() == [3.0, 0.0, 0.0, 0.0, 0.0]
        # `engine="convolution"` casts the kernel to the image's dtype before negating it, so on CPU the
        # erosion of an integer image runs instead of raising and a `False` cell contributes `x - 1`, which
        # can lower the minimum; the `x - 1` is computed in the image's dtype and wraps at its bounds, so a
        # `uint8` image turns a zero pad's `0 - 1` into 255. MPS and CUDA reject the integer image instead (see
        # #4762's pin below). Generated with kornia in this worktree (torch 2.14.0 and 2.5.1):
        #   erosion([[5, 1, 5, 5, 5]] int64, [[T, F, T]], engine="convolution") -> [1, 0, 1, 4, 4] (CPU)
        #   erosion([[5., 1., 5., 5., 5.]], [[T, F, T]], engine="convolution") -> [1, 5, 1, 5, 5] (exact)
        #   erosion([[10, 20, 30]] int64, [[T, T, F]], "constant", engine="convolution") -> [0, 10, -1] (CPU)
        #   erosion([[10, 20, 30]] uint8, [[T, T, F]], "constant", engine="convolution") -> [0, 10, 20] (CPU)
        int_dip = torch.tensor([[5, 1, 5, 5, 5]], dtype=torch.int64, device=device)[None, None]
        if device.type == "cpu":
            assert erosion(int_dip, sparse_bool_kernel, engine="convolution").flatten().tolist() == [1, 0, 1, 4, 4]
            steps = torch.tensor([[10, 20, 30]], dtype=torch.int64, device=device)[None, None]
            wide = torch.tensor([[True, True, False]], device=device)
            assert erosion(steps, wide, border_type="constant", engine="convolution").flatten().tolist() == [0, 10, -1]
            wrapped = erosion(steps.to(torch.uint8), wide, border_type="constant", engine="convolution")
            assert wrapped.flatten().tolist() == [0, 10, 20]
        elif device.type == "mps":
            with pytest.raises(RuntimeError, match="Floating"):
                erosion(int_dip, sparse_bool_kernel, engine="convolution")
        dip = int_dip.float()
        assert erosion(dip, sparse_bool_kernel, engine="convolution").flatten().tolist() == [1.0, 5.0, 1.0, 5.0, 5.0]
        # With a floating structuring element the sentinel is stored there instead, the kernel is only the
        # `kernel == 0` mask, and a `uint8` or `bool` kernel returns what the floating kernel does.
        frame = torch.rand(1, 1, 4, 5, generator=torch.Generator().manual_seed(0)).to(device)
        flat_se = torch.zeros(1, 3, device=device)
        for mask_kernel in (torch.tensor([[1, 0, 1]], dtype=torch.uint8, device=device), sparse_bool_kernel):
            for op in (dilation, erosion):
                assert torch.equal(
                    op(frame, mask_kernel, structuring_element=flat_se),
                    op(frame, mask_kernel.float(), structuring_element=flat_se),
                ), (mask_kernel.dtype, op.__name__)

        # A floating kernel lets every function run on a `bool` image under the `shift` engine (the default
        # off CUDA without autograd, named explicitly so the pin does not depend on the device's `auto`) and
        # return the floating dtype.
        ring_dilation = dilation(hot_bool, float_kernel, engine="shift")
        assert ring_dilation.dtype == torch.float32
        if pads_true:
            assert ring_dilation.flatten().tolist() == [float(v) for v in ring_plus_dilation]
        cold_bool = ~hot_bool
        gapped_float_kernel = torch.tensor([[1.0, 0.0, 1.0]], device=device)
        # `True` cannot lower a minimum, so the geodesic `True` pad leaves erosion exact. (MPS's raw +max_val
        # byte, 16 for the default 1e4, cannot lower it either on torch 2.14 or 2.5.1; a byte of 0, as for
        # max_val=9984, erodes the border to 0 there.)
        cold_erosion = erosion(cold_bool, gapped_float_kernel, engine="shift")
        assert cold_erosion.flatten().tolist() == [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        assert torch.equal(cold_erosion, erosion(cold_bool.float(), gapped_float_kernel, engine="shift"))
        ring_gradient = gradient(hot_bool, float_kernel, engine="shift")
        assert ring_gradient.dtype == torch.float32
        if pads_true:
            assert ring_gradient.flatten().tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        # The other engines do not: `unfold` (the `auto` choice on CUDA) subtracts from the `bool` image in
        # the erosion, and `convolution` rejects a `bool` image outright ("slow_conv2d_cpu" not implemented
        # for 'Bool' on CPU, "Convolution is supported only for Floating types" on MPS,
        # "getCudnnDataTypeFromScalarType() not supported for Bool" on CUDA) because it casts the
        # kernel to the image's dtype (#4762), so a fix for #4762 has to revisit this line too.
        with pytest.raises((NotImplementedError, RuntimeError), match="bool"):
            erosion(cold_bool, gapped_float_kernel, engine="unfold")
        with pytest.raises((NotImplementedError, RuntimeError)):
            dilation(hot_bool, float_kernel, engine="convolution")

        big_int = torch.tensor([[0, 50000, 0]], dtype=torch.int64, device=device)[None, None]
        int_kernel = torch.tensor([[1, 0, 1]], dtype=torch.int64, device=device)
        assert dilation(big_int, int_kernel).flatten().tolist() == [50000, 40000, 50000]
        # A floating kernel promotes an `int64` image to its own dtype, and float32 cannot hold 2**24 + 1.
        # A 1x1 kernel needs no pad, so the geodesic sentinel is never stored and even `uint8` runs.
        just_above = torch.tensor([[16777217, 0, 0]], dtype=torch.int64, device=device)[None, None]
        one_by_one = torch.ones(1, 1, device=device)
        assert dilation(just_above, one_by_one).flatten().tolist() == [16777216.0, 0.0, 0.0]
        small_uint8 = torch.tensor([[0, 7, 200]], dtype=torch.uint8, device=device)[None, None]
        unpadded = dilation(small_uint8, one_by_one)
        assert unpadded.dtype == torch.float32
        assert unpadded.flatten().tolist() == [0.0, 7.0, 200.0]

    def test_wart_convolution_engine_keeps_image_dtype_4762(self, device):
        # `unfold` and `shift` compute in, and return, the promoted dtype of the image and the kernel (or
        # the structuring element); `engine="convolution"` casts the kernel to the image's dtype instead,
        # so the result dtype depends on the engine. Tracked in #4762. Dtypes are explicit (the claim is
        # about a float16 image with a float32 kernel), so this pin takes `device` only.
        # Generated with kornia in this worktree (torch 2.14.0; CPU and MPS agree):
        #   dilation(float16 image, ones(3, 3) float32, engine=e).dtype -> unfold/shift float32, convolution float16
        #   erosion(...) likewise
        tensor = torch.rand(1, 1, 4, 5, generator=torch.Generator().manual_seed(0)).to(device, torch.float16)
        kernel = torch.ones(3, 3, device=device)
        for op in (dilation, erosion):
            assert op(tensor, kernel, engine="unfold").dtype == torch.float32, op.__name__
            assert op(tensor, kernel, engine="shift").dtype == torch.float32, op.__name__
            assert op(tensor, kernel, engine="convolution").dtype == torch.float16, op.__name__

        # Cast to a `uint8` image, the `-max_val` sentinel goes through an out-of-range float-to-uint8 cast,
        # which C leaves undefined: it wraps (-1e4 mod 256 == 240) or saturates to 0 depending on the platform,
        # the torch release and whether the cell lands in a vectorised chunk (on macOS arm64 with torch 2.14.0,
        # `torch.full((9,), -1e4).to(torch.uint8)` is 0 in its first eight cells and 240 in the ninth). So the
        # pin replays torch's own cast of the same three-element bias rather than hardcoding one outcome, and
        # rebuilds the one-hot `conv2d` in uint8 arithmetic: the masked cell contributes `(x + s) mod 256` for
        # whatever byte `s` the cast produced (240 when it wraps, making it a member when it saturates to 0).
        # `unfold` returns the correct float32 [20, 30, 30]. `conv2d` rejects integer images on MPS
        # ("Convolution is supported only for Floating types") and on CUDA (cuDNN: "unable to find an engine
        # to execute this computation"); other backends are unmeasured and left unasserted.
        #   dilation([[10, 20, 30]] uint8, [[1., 1., 0.]], "constant", engine="convolution")
        #       -> [240, 250, 30] on CPU, macOS arm64, torch 2.14.0 and 2.5.1
        small = torch.tensor([[10, 20, 30]], dtype=torch.uint8, device=device)[None, None]
        pair = torch.tensor([[1.0, 1.0, 0.0]], device=device)
        if device.type == "cpu":
            wrapped = dilation(small, pair, border_type="constant", engine="convolution")
            assert wrapped.dtype == torch.uint8
            # The bias is the flipped neighborhood [-max_val, 0, 0]; the padded row is [0, 10, 20, 30, 0].
            bias = torch.tensor([-1e4, 0.0, 0.0]).to(torch.uint8).tolist()
            padded = [0, 10, 20, 30, 0]
            expected = [max((padded[i + j] + bias[j]) % 256 for j in range(3)) for i in range(3)]
            assert wrapped.flatten().tolist() == expected
        elif device.type in ("mps", "cuda"):
            with pytest.raises((NotImplementedError, RuntimeError)):
                dilation(small, pair, border_type="constant", engine="convolution")
        assert dilation(small, pair, border_type="constant", engine="unfold").flatten().tolist() == [20.0, 30.0, 30.0]

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
        # In general the empty window returns `max_val + min(0, m)` (`-max_val + max(0, M)` in `dilation`),
        # `m` (`M`) the smallest (largest) in-image pixel under a masked-out cell, rounded in the image's dtype.
        corner_kernel = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
        minus_two = torch.full((1, 1, 1, 1), -2.0, device=device, dtype=dtype)
        assert erosion(minus_two, corner_kernel, origin=[0, 0]).item() == 9998.0
        five = torch.full((1, 1, 1, 1), 5.0, device=device, dtype=dtype)
        assert dilation(five, corner_kernel, origin=[0, 0]).item() == -9995.0
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

        # `max_val` has to be finite in the operands' dtype, and a representable one works in float16, both in
        # the geodesic pad and in the kernel's masked-out cells. (A larger one raises or rounds depending on the
        # backend and the torch release, which is torch's cast and is left unasserted.)
        half_image = torch.rand(1, 1, 4, 4, generator=torch.Generator().manual_seed(0)).to(device, torch.float16)
        half_cross = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float16
        )
        assert torch.equal(dilation(half_image, half_cross, max_val=65504.0), dilation(half_image, half_cross))

    def test_wart_non_float_image_is_not_rejected_4735(self, device):
        # Only floating-point images are supported, but a non-float image is not rejected. What it then does
        # depends on the border, the engine, the backend and the torch release, so the pin records only the
        # portable symptom that a fix for #4735 flips: a `uint8` image under `border_type="constant"` runs
        # and returns the float kernel's dtype. Dtypes are explicit, so this pin takes `device` only.
        # Generated with kornia in this worktree (torch 2.14.0 and 2.5.1, CPU and MPS):
        #   dilation(zeros(1, 1, 1, 5, uint8), ones(1, 3), border_type="constant") -> float32 zeros
        zeros_uint8 = torch.zeros(1, 1, 1, 5, dtype=torch.uint8, device=device)
        float_kernel = torch.ones(1, 3, device=device)
        for op in (dilation, erosion):
            out = op(zeros_uint8, float_kernel, border_type="constant")
            assert out.dtype == torch.float32, op.__name__
            assert out.flatten().tolist() == [0.0] * 5, op.__name__

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

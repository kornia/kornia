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

from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_DM_DESC,
    KORNIA_CHECK_IS_COLOR,
    KORNIA_CHECK_IS_COLOR_OR_GRAY,
    KORNIA_CHECK_IS_GRAY,
    KORNIA_CHECK_IS_IMAGE,
    KORNIA_CHECK_IS_LIST_OF_TENSOR,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_LAF,
    KORNIA_CHECK_SAME_DEVICE,
    KORNIA_CHECK_SAME_DEVICES,
    KORNIA_CHECK_SAME_SHAPE,
    KORNIA_CHECK_SHAPE,
    KORNIA_CHECK_TYPE,
    are_checks_enabled,
    disable_checks,
    enable_checks,
)
from kornia.core.exceptions import (
    BaseError,
    DeviceError,
    ImageError,
    ShapeError,
    TypeCheckError,
    ValueCheckError,
)

from testing.base import BaseTester


class TestCheck:
    def test_valid(self):
        assert KORNIA_CHECK(True, "This is a test") is True

    def test_invalid(self):
        with pytest.raises(BaseError):
            KORNIA_CHECK(False, "This is a test")

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK(False, "This should not raise", raises=False) is False

    def test_jit(self):
        op_jit = torch.jit.script(KORNIA_CHECK)
        assert op_jit is not None
        assert op_jit(True, "Testing") is True


class TestCheckShape:
    @pytest.mark.parametrize(
        "data,shape",
        [
            (torch.rand(2, 3), ["*", "H", "W"]),
            (torch.rand(3, 2, 3), ["3", "H", "W"]),
            (torch.rand(1, 1, 2, 3), ["1", "1", "H", "W"]),
            (torch.rand(2, 3, 2, 3), ["2", "3", "H", "W"]),
        ],
    )
    def test_valid(self, data, shape):
        assert KORNIA_CHECK_SHAPE(data, shape) is True

    @pytest.mark.parametrize(
        "data,shape",
        [
            (torch.rand(2, 3), ["1", "H", "W"]),
            (torch.rand(3, 2, 3), ["H", "W"]),
            (torch.rand(1, 2, 3), ["3", "H", "W"]),
            (torch.rand(1, 3, 2, 3), ["2", "C", "H", "W"]),
        ],
    )
    def test_invalid(self, data, shape):
        with pytest.raises(ShapeError):
            KORNIA_CHECK_SHAPE(data, shape)

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_SHAPE(torch.rand(2, 3), ["1", "H", "W"], raises=False) is False

    def test_jit(self):
        op_jit = torch.jit.script(KORNIA_CHECK_SHAPE)
        assert op_jit is not None
        assert op_jit(torch.rand(2, 3, 2, 3), ["2", "3", "H", "W"]) is True


class TestCheckSameShape:
    def test_valid(self):
        assert KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(2, 3)) is True
        assert KORNIA_CHECK_SAME_SHAPE(torch.rand(1, 2, 3), torch.rand(1, 2, 3)) is True
        assert KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3, 3), torch.rand(2, 3, 3)) is True

    def test_jit(self):
        op_jit = torch.jit.script(KORNIA_CHECK_SAME_SHAPE)
        assert op_jit is not None
        assert op_jit(torch.rand(2, 3), torch.rand(2, 3)) is True

    def test_invalid(self):
        with pytest.raises(ShapeError):
            KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(2, 2, 3))
        with pytest.raises(ShapeError):
            KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(1, 2, 3))
        with pytest.raises(ShapeError):
            KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(2, 3, 3))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(2, 2, 3), raises=False) is False


class TestCheckType:
    def test_valid(self):
        assert KORNIA_CHECK_TYPE("hello", str) is True
        assert KORNIA_CHECK_TYPE(23, int) is True
        assert KORNIA_CHECK_TYPE(torch.rand(1), torch.Tensor) is True
        assert KORNIA_CHECK_TYPE(torch.rand(1), (int, torch.Tensor)) is True

    def test_invalid(self):
        with pytest.raises(TypeCheckError):
            KORNIA_CHECK_TYPE("world", int)
        with pytest.raises(TypeCheckError):
            KORNIA_CHECK_TYPE(23, float)
        with pytest.raises(TypeCheckError):
            KORNIA_CHECK_TYPE(23, (float, str))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_TYPE("world", int, raises=False) is False


class TestCheckIsTensor:
    def test_valid(self):
        assert KORNIA_CHECK_IS_TENSOR(torch.rand(1)) is True

    def test_invalid(self):
        with pytest.raises(TypeCheckError):
            KORNIA_CHECK_IS_TENSOR([1, 2, 3])

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_IS_TENSOR([1, 2, 3], raises=False) is False


class TestCheckIsListOfTensor:
    def test_valid(self):
        assert KORNIA_CHECK_IS_LIST_OF_TENSOR([torch.rand(1), torch.rand(1), torch.rand(1)]) is True

    def test_invalid(self):
        with pytest.raises(TypeCheckError):
            KORNIA_CHECK_IS_LIST_OF_TENSOR([torch.rand(1), [2, 3], torch.rand(1)])
        with pytest.raises(TypeCheckError):
            KORNIA_CHECK_IS_LIST_OF_TENSOR([1, 2, 3])

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_IS_LIST_OF_TENSOR([torch.rand(1), [2, 3], torch.rand(1)], raises=False) is False


class TestCheckSameDevice:
    def test_valid(self, device):
        assert KORNIA_CHECK_SAME_DEVICE(torch.rand(1, device=device), torch.rand(1, device=device)) is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU.")
    def test_invalid(self):
        with pytest.raises(DeviceError):
            KORNIA_CHECK_SAME_DEVICE(torch.rand(1, device="cpu"), torch.rand(1, device="cuda"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU.")
    def test_invalid_raises_false(self):
        assert (
            KORNIA_CHECK_SAME_DEVICE(torch.rand(1, device="cpu"), torch.rand(1, device="cuda"), raises=False) is False
        )


class TestCheckSameDevices:
    def test_valid(self, device):
        assert KORNIA_CHECK_SAME_DEVICES([torch.rand(1, device=device), torch.rand(1, device=device)]) is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU.")
    def test_invalid(self):
        with pytest.raises(DeviceError):
            KORNIA_CHECK_SAME_DEVICES([torch.rand(1, device="cpu"), torch.rand(1, device="cuda")])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU.")
    def test_invalid_raises_false(self):
        assert (
            KORNIA_CHECK_SAME_DEVICES([torch.rand(1, device="cpu"), torch.rand(1, device="cuda")], raises=False)
            is False
        )


class TestCheckIsColor:
    def test_valid(self):
        assert KORNIA_CHECK_IS_COLOR(torch.rand(3, 4, 4)) is True
        assert KORNIA_CHECK_IS_COLOR(torch.rand(1, 3, 4, 4)) is True
        assert KORNIA_CHECK_IS_COLOR(torch.rand(2, 3, 4, 4)) is True

    def test_invalid(self):
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_COLOR(torch.rand(1, 4, 4))
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_COLOR(torch.rand(2, 4, 4))
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_COLOR(torch.rand(3, 4, 4, 4))
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_COLOR(torch.rand(1, 3, 4, 4, 4))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_IS_COLOR(torch.rand(1, 4, 4), raises=False) is False


class TestCheckIsGray:
    def test_valid(self):
        assert KORNIA_CHECK_IS_GRAY(torch.rand(1, 4, 4)) is True
        assert KORNIA_CHECK_IS_GRAY(torch.rand(2, 1, 4, 4)) is True
        assert KORNIA_CHECK_IS_GRAY(torch.rand(3, 1, 4, 4)) is True

    def test_invalid(self):
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_GRAY(torch.rand(3, 4, 4))
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_GRAY(torch.rand(1, 4, 4, 4))
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_GRAY(torch.rand(1, 3, 4, 4))
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_GRAY(torch.rand(1, 3, 4, 4, 4))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_IS_GRAY(torch.rand(1, 3, 4, 4, 4), raises=False) is False


class TestCheckIsColorOrGray:
    def test_valid(self):
        assert KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(3, 4, 4)) is True
        assert KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 3, 4, 4)) is True
        assert KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(2, 3, 4, 4)) is True
        assert KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 4, 4)) is True
        assert KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(2, 1, 4, 4)) is True
        assert KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(3, 1, 4, 4)) is True

    def test_invalid(self):
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 4, 4, 4))
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 3, 4, 4, 4))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 4, 4, 4), raises=False) is False


class TestCheckDmDesc:
    def test_valid(self):
        assert KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 8)) is True

    def test_invalid(self):
        with pytest.raises(ShapeError):
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 7))
        with pytest.raises(ShapeError):
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(3, 8))
        with pytest.raises(ShapeError):
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(3, 7))
        with pytest.raises(ShapeError):
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 3, 8))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 7), raises=False) is False


class TestCheckLaf:
    def test_valid(self):
        assert KORNIA_CHECK_LAF(torch.rand(4, 2, 2, 3)) is True

    def test_invalid(self):
        with pytest.raises(ShapeError):
            KORNIA_CHECK_LAF(torch.rand(4, 2, 2))
        with pytest.raises(ShapeError):
            KORNIA_CHECK_LAF(torch.rand(4, 2, 3, 2))
        with pytest.raises(ShapeError):
            KORNIA_CHECK_LAF(torch.rand(4, 2, 2, 2))
        with pytest.raises(ShapeError):
            KORNIA_CHECK_LAF(torch.rand(4, 2, 3, 3, 3))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_LAF(torch.rand(4, 2, 2), raises=False) is False


class TestCheckIsImage(BaseTester):
    def test_valid_float(self):
        assert KORNIA_CHECK_IS_IMAGE(torch.rand(3, 4, 4)) is True
        assert KORNIA_CHECK_IS_IMAGE(torch.rand(2, 3, 4, 4)) is True
        assert KORNIA_CHECK_IS_IMAGE(torch.rand(1, 1, 4, 4)) is True

    def test_valid_int(self):
        x = torch.randint(0, 256, (3, 4, 4), dtype=torch.uint8)
        assert KORNIA_CHECK_IS_IMAGE(x) is True
        y = torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)
        assert KORNIA_CHECK_IS_IMAGE(y) is True

    def test_invalid_float_range(self):
        with pytest.raises(ValueCheckError):
            KORNIA_CHECK_IS_IMAGE(torch.tensor([[[-0.5, 1.2]]], dtype=torch.float32))

    def test_invalid_int_range(self):
        bad = torch.tensor([[[300]]], dtype=torch.int32)
        with pytest.raises(ValueCheckError):
            KORNIA_CHECK_IS_IMAGE(bad)

    def test_invalid_shape(self):
        x = torch.rand(1, 4, 4, 4)
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_IMAGE(x)

    def test_invalid_range_no_raise(self):
        bad = torch.tensor([[[-0.1, 2.0]]], dtype=torch.float32)
        assert KORNIA_CHECK_IS_IMAGE(bad, raises=False) is False

    def test_invalid_shape_no_raise(self):
        bad = torch.rand(1, 4, 4, 4)
        assert KORNIA_CHECK_IS_IMAGE(bad, raises=False) is False

    @pytest.mark.parametrize("shape", [(5, 4, 5), (2, 4, 4, 4), (4, 5), (7,), ()])
    def test_shape_verdict_does_not_depend_on_raises(self, device, dtype, shape):
        # The values lie in [0, 1], so only the shape can fail the check.
        x = torch.rand(shape, device=device, dtype=dtype)
        assert KORNIA_CHECK_IS_IMAGE(x, raises=False) is False
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_IMAGE(x)

    @pytest.mark.parametrize("nan_count", ["one", "all"])
    def test_nan_fails_the_range_check(self, device, dtype, nan_count):
        x = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        if nan_count == "all":
            x.fill_(float("nan"))
        else:
            x[1, 2, 3, 4] = float("nan")
        assert KORNIA_CHECK_IS_IMAGE(x, raises=False) is False
        with pytest.raises(ValueCheckError, match=r"expected \[0, 1\]"):
            KORNIA_CHECK_IS_IMAGE(x)

    def test_float_range_error_reports_the_unit_range(self, device, dtype):
        x = torch.tensor([[[-0.5, 0.25, 1.5]]], device=device, dtype=dtype)
        with pytest.raises(ValueCheckError, match=r"expected \[0, 1\], got \[-0\.5, 1\.5\]\.") as err:
            KORNIA_CHECK_IS_IMAGE(x)
        assert err.value.actual_value == (-0.5, 1.5)
        assert err.value.expected_range == (0.0, 1.0)

    @pytest.mark.parametrize("shape", [(0, 3, 4, 5), (0, 1, 4, 5), (3, 0, 0)])
    def test_empty_image_passes(self, device, dtype, shape):
        # An empty image holds no value that could be out of range.
        x = torch.rand(shape, device=device, dtype=dtype)
        assert KORNIA_CHECK_IS_IMAGE(x, raises=False) is True
        assert KORNIA_CHECK_IS_IMAGE(x) is True
        u = torch.zeros(shape, device=device, dtype=torch.uint8)
        assert KORNIA_CHECK_IS_IMAGE(u, raises=False) is True
        assert KORNIA_CHECK_IS_IMAGE(u) is True

    def test_empty_tensor_with_a_bad_shape_fails(self, device, dtype):
        x = torch.rand(0, 5, 4, 5, device=device, dtype=dtype)
        assert KORNIA_CHECK_IS_IMAGE(x, raises=False) is False
        with pytest.raises(ImageError):
            KORNIA_CHECK_IS_IMAGE(x)

    @pytest.mark.parametrize("bits", [4, 8, 10, 16])
    def test_integer_range_is_zero_to_two_to_the_bits_minus_one(self, device, bits):
        top = 2**bits - 1
        assert KORNIA_CHECK_IS_IMAGE(torch.full((3, 4, 5), top, device=device, dtype=torch.int32), bits=bits) is True
        x = torch.full((3, 4, 5), top + 1, device=device, dtype=torch.int32)
        assert KORNIA_CHECK_IS_IMAGE(x, bits=bits, raises=False) is False
        with pytest.raises(ValueCheckError, match=rf"expected \[0, {top}\], got \[{top + 1}, {top + 1}\]\.") as err:
            KORNIA_CHECK_IS_IMAGE(x, bits=bits)
        assert err.value.actual_value == (top + 1, top + 1)
        assert err.value.expected_range == (0, top)

    @pytest.mark.parametrize(
        ("int_dtype", "bits"),
        [(torch.int8, 8), (torch.int8, 9), (torch.int16, 16), (torch.int32, 32), (torch.int64, 64), (torch.int64, 63)],
    )
    def test_signed_dtype_accepts_every_nonnegative_value_that_fits_the_bits(self, device, int_dtype, bits):
        # 2**bits - 1 exceeds these dtypes' maximum, so every non-negative value of the dtype is in range.
        top = torch.iinfo(int_dtype).max
        x = torch.tensor([[[0, 5, top]]], device=device, dtype=int_dtype)
        assert KORNIA_CHECK_IS_IMAGE(x, bits=bits, raises=False) is True
        assert KORNIA_CHECK_IS_IMAGE(x, bits=bits) is True
        x[0, 0, 0] = -1
        assert KORNIA_CHECK_IS_IMAGE(x, bits=bits, raises=False) is False
        with pytest.raises(ValueCheckError) as err:
            KORNIA_CHECK_IS_IMAGE(x, bits=bits)
        assert err.value.actual_value == (-1, top)
        assert err.value.expected_range == (0, 2**bits - 1)

    @pytest.mark.parametrize("uint_dtype", [torch.uint16, torch.uint32, torch.uint64])
    def test_wide_unsigned_dtype_is_range_checked(self, device, uint_dtype):
        info = torch.iinfo(uint_dtype)
        x = torch.tensor([[[0, 1000, info.max]]], dtype=uint_dtype).to(device)
        assert KORNIA_CHECK_IS_IMAGE(x, bits=info.bits, raises=False) is True
        assert KORNIA_CHECK_IS_IMAGE(x, bits=info.bits) is True
        # With one bit fewer the largest value is out of range, and the error reports it exactly.
        assert KORNIA_CHECK_IS_IMAGE(x, bits=info.bits - 1, raises=False) is False
        with pytest.raises(ValueCheckError) as err:
            KORNIA_CHECK_IS_IMAGE(x, bits=info.bits - 1)
        assert err.value.actual_value == (0, info.max)
        assert err.value.expected_range == (0, 2 ** (info.bits - 1) - 1)


class TestChecksEnableDisable:
    """Tests for the runtime enable/disable check control API."""

    def setup_method(self):
        # Always restore checks after each test to avoid side effects
        enable_checks()

    def teardown_method(self):
        enable_checks()

    def test_are_checks_enabled_default(self):
        enable_checks()
        assert are_checks_enabled() is True

    def test_disable_checks(self):
        disable_checks()
        assert are_checks_enabled() is False

    def test_enable_checks(self):
        disable_checks()
        enable_checks()
        assert are_checks_enabled() is True

    def test_disabled_checks_bypass_kornia_check(self):
        disable_checks()
        # With checks disabled, KORNIA_CHECK should return True even for False condition
        result = KORNIA_CHECK(False, "should be bypassed")
        assert result is True

    def test_disabled_checks_bypass_shape_check(self):
        disable_checks()
        # With checks disabled, wrong shape should not raise
        result = KORNIA_CHECK_SHAPE(torch.rand(2, 3), ["1", "H", "W"])
        assert result is True

    def test_enabled_checks_enforce_shape_check(self):
        enable_checks()
        with pytest.raises(ShapeError):
            KORNIA_CHECK_SHAPE(torch.rand(2, 3), ["1", "H", "W"])

    def test_env_var_disables_checks(self, monkeypatch):
        monkeypatch.setenv("KORNIA_CHECKS", "0")
        from kornia.core.check import _should_enable_checks

        assert _should_enable_checks() is False

    def test_env_var_enables_checks(self, monkeypatch):
        from kornia.core.check import _should_enable_checks

        monkeypatch.setenv("KORNIA_CHECKS", "1")
        assert _should_enable_checks() is True

    def test_env_var_true_string_variants(self, monkeypatch):
        from kornia.core.check import _should_enable_checks

        for val in ("true", "yes", "on", "1"):
            monkeypatch.setenv("KORNIA_CHECKS", val)
            assert _should_enable_checks() is True

    def test_env_var_false_string(self, monkeypatch):
        from kornia.core.check import _should_enable_checks

        monkeypatch.setenv("KORNIA_CHECKS", "false")
        assert _should_enable_checks() is False

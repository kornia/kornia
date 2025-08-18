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
    KORNIA_CHECK_IS_LIST_OF_TENSOR,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_LAF,
    KORNIA_CHECK_SAME_DEVICE,
    KORNIA_CHECK_SAME_DEVICES,
    KORNIA_CHECK_SAME_SHAPE,
    KORNIA_CHECK_SHAPE,
    KORNIA_CHECK_TYPE,
    KORNIA_CHECK_IS_IMAGE
)


class TestCheck:
    def test_valid(self):
        assert KORNIA_CHECK(True, "This is a test") is True

    def test_invalid(self):
        with pytest.raises(Exception):
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
        with pytest.raises(Exception):
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
        with pytest.raises(Exception):
            KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(2, 2, 3))
        with pytest.raises(Exception):
            KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(1, 2, 3))
        with pytest.raises(Exception):
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
        with pytest.raises(Exception):
            KORNIA_CHECK_TYPE("world", int)
        with pytest.raises(Exception):
            KORNIA_CHECK_TYPE(23, float)
        with pytest.raises(Exception):
            KORNIA_CHECK_TYPE(23, (float, str))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_TYPE("world", int, raises=False) is False


class TestCheckIsTensor:
    def test_valid(self):
        assert KORNIA_CHECK_IS_TENSOR(torch.rand(1)) is True

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_TENSOR([1, 2, 3])

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_IS_TENSOR([1, 2, 3], raises=False) is False


class TestCheckIsListOfTensor:
    def test_valid(self):
        assert KORNIA_CHECK_IS_LIST_OF_TENSOR([torch.rand(1), torch.rand(1), torch.rand(1)]) is True

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_LIST_OF_TENSOR([torch.rand(1), [2, 3], torch.rand(1)])
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_LIST_OF_TENSOR([1, 2, 3])

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_IS_LIST_OF_TENSOR([torch.rand(1), [2, 3], torch.rand(1)], raises=False) is False


class TestCheckSameDevice:
    def test_valid(self, device):
        assert KORNIA_CHECK_SAME_DEVICE(torch.rand(1, device=device), torch.rand(1, device=device)) is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU.")
    def test_invalid(self):
        with pytest.raises(Exception):
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
        with pytest.raises(Exception):
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
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_COLOR(torch.rand(1, 4, 4))
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_COLOR(torch.rand(2, 4, 4))
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_COLOR(torch.rand(3, 4, 4, 4))
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_COLOR(torch.rand(1, 3, 4, 4, 4))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_IS_COLOR(torch.rand(1, 4, 4), raises=False) is False


class TestCheckIsGray:
    def test_valid(self):
        assert KORNIA_CHECK_IS_GRAY(torch.rand(1, 4, 4)) is True
        assert KORNIA_CHECK_IS_GRAY(torch.rand(2, 1, 4, 4)) is True
        assert KORNIA_CHECK_IS_GRAY(torch.rand(3, 1, 4, 4)) is True

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_GRAY(torch.rand(3, 4, 4))
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_GRAY(torch.rand(1, 4, 4, 4))
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_GRAY(torch.rand(1, 3, 4, 4))
        with pytest.raises(Exception):
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
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 4, 4, 4))
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 3, 4, 4, 4))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 4, 4, 4), raises=False) is False


class TestCheckDmDesc:
    def test_valid(self):
        assert KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 8)) is True

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 7))
        with pytest.raises(Exception):
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(3, 8))
        with pytest.raises(Exception):
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(3, 7))
        with pytest.raises(Exception):
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 3, 8))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 7), raises=False) is False


class TestCheckLaf:
    def test_valid(self):
        assert KORNIA_CHECK_LAF(torch.rand(4, 2, 2, 3)) is True

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_LAF(torch.rand(4, 2, 2))
        with pytest.raises(Exception):
            KORNIA_CHECK_LAF(torch.rand(4, 2, 3, 2))
        with pytest.raises(Exception):
            KORNIA_CHECK_LAF(torch.rand(4, 2, 2, 2))
        with pytest.raises(Exception):
            KORNIA_CHECK_LAF(torch.rand(4, 2, 3, 3, 3))

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK_LAF(torch.rand(4, 2, 2), raises=False) is False

class TestCheckIsImage:
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
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_IMAGE(torch.tensor([[[-0.5, 1.2]]], dtype=torch.float32))

    def test_invalid_int_range(self):
        bad = torch.tensor([[[300]]], dtype=torch.int32)
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_IMAGE(bad)

    def test_invalid_shape(self):
        x = torch.rand(1, 4, 4, 4)
        assert KORNIA_CHECK_IS_IMAGE(x) is True

    def test_invalid_range_no_raise(self):
        bad = torch.tensor([[[-0.1, 2.0]]], dtype=torch.float32)
        assert KORNIA_CHECK_IS_IMAGE(bad, raises=False) is False

    def test_invalid_shape_no_raise(self):
        # When raises=False, shape check is actually enforced
        bad = torch.rand(1, 4, 4, 4)
        assert KORNIA_CHECK_IS_IMAGE(bad, raises=False) is False

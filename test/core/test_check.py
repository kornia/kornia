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
)


class TestCheck:
    def test_valid(self):
        assert KORNIA_CHECK(True, "This is a test") is True

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK(False, "This is a test")

    def test_invalid_raises_false(self):
        assert KORNIA_CHECK(False, "This should not raise", raises=False) is False


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


class TestCheckSameShape:
    def test_valid(self):
        assert KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(2, 3)) is True
        assert KORNIA_CHECK_SAME_SHAPE(torch.rand(1, 2, 3), torch.rand(1, 2, 3)) is True
        assert KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3, 3), torch.rand(2, 3, 3)) is True

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

import pytest
import torch

from kornia.core.logger import (
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
        KORNIA_CHECK(True, "This is a test")

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK(False, "This is a test")


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
        KORNIA_CHECK_SHAPE(data, shape)

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


class TestCheckSameShape:
    def test_valid(self):
        KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(2, 3))
        KORNIA_CHECK_SAME_SHAPE(torch.rand(1, 2, 3), torch.rand(1, 2, 3))
        KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3, 3), torch.rand(2, 3, 3))

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(2, 2, 3))
            KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(1, 2, 3))
            KORNIA_CHECK_SAME_SHAPE(torch.rand(2, 3), torch.rand(2, 3, 3))


class TestCheckType:
    def test_valid(self):
        KORNIA_CHECK_TYPE("hello", str)
        KORNIA_CHECK_TYPE(23, int)
        KORNIA_CHECK_TYPE(torch.rand(1), torch.Tensor)

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_TYPE("world", int)
            KORNIA_CHECK_TYPE(23, float)


class TestCheckIsTensor:
    def test_valid(self):
        KORNIA_CHECK_IS_TENSOR(torch.rand(1))

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_TENSOR([1, 2, 3])


class TestCheckIsListOfTensor:
    def test_valid(self):
        assert KORNIA_CHECK_IS_LIST_OF_TENSOR([torch.rand(1), torch.rand(1), torch.rand(1)])

    def test_invalid(self):
        with pytest.raises(Exception):
            assert KORNIA_CHECK_IS_LIST_OF_TENSOR([torch.rand(1), [2, 3], torch.rand(1)])
            assert KORNIA_CHECK_IS_LIST_OF_TENSOR([1, 2, 3])


class TestCheckSameDevice:
    def test_valid(self, device):
        KORNIA_CHECK_SAME_DEVICE(torch.rand(1, device=device), torch.rand(1, device=device))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU.")
    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_SAME_DEVICE(torch.rand(1, device="cpu"), torch.rand(1, device="cuda"))


class TestCheckSameDevices:
    def test_valid(self, device):
        KORNIA_CHECK_SAME_DEVICES([torch.rand(1, device=device), torch.rand(1, device=device)])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU.")
    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_SAME_DEVICES([torch.rand(1, device="cpu"), torch.rand(1, device="cuda")])


class TestCheckIsColor:
    def test_valid(self):
        KORNIA_CHECK_IS_COLOR(torch.rand(3, 4, 4))
        KORNIA_CHECK_IS_COLOR(torch.rand(1, 3, 4, 4))
        KORNIA_CHECK_IS_COLOR(torch.rand(2, 3, 4, 4))

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_COLOR(torch.rand(1, 4, 4))
            KORNIA_CHECK_IS_COLOR(torch.rand(2, 4, 4))
            KORNIA_CHECK_IS_COLOR(torch.rand(3, 4, 4, 4))
            KORNIA_CHECK_IS_COLOR(torch.rand(1, 3, 4, 4, 4))


class TestCheckIsGray:
    def test_valid(self):
        KORNIA_CHECK_IS_GRAY(torch.rand(1, 4, 4))
        KORNIA_CHECK_IS_GRAY(torch.rand(2, 1, 4, 4))
        KORNIA_CHECK_IS_GRAY(torch.rand(3, 1, 4, 4))

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_GRAY(torch.rand(3, 4, 4))
            KORNIA_CHECK_IS_GRAY(torch.rand(1, 4, 4, 4))
            KORNIA_CHECK_IS_GRAY(torch.rand(1, 3, 4, 4))
            KORNIA_CHECK_IS_GRAY(torch.rand(1, 3, 4, 4, 4))


class TestCheckIsColorOrGray:
    def test_valid(self):
        KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(3, 4, 4))
        KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 3, 4, 4))
        KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(2, 3, 4, 4))
        KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 4, 4))
        KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(2, 1, 4, 4))
        KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(3, 1, 4, 4))

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 4, 4, 4))
            KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.rand(1, 3, 4, 4, 4))


class TestCheckDmDesc:
    def test_valid(self):
        KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 8))

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 7))
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(3, 8))
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(3, 7))
            KORNIA_CHECK_DM_DESC(torch.rand(4), torch.rand(8), torch.rand(4, 8, 3))


class TestCheckLaf:
    def test_valid(self):
        KORNIA_CHECK_LAF(torch.rand(4, 2, 2, 3))

    def test_invalid(self):
        with pytest.raises(Exception):
            KORNIA_CHECK_LAF(torch.rand(4, 2, 2))
            KORNIA_CHECK_LAF(torch.rand(4, 2, 3, 2))
            KORNIA_CHECK_LAF(torch.rand(4, 2, 2, 2))
            KORNIA_CHECK_LAF(torch.rand(4, 2, 3, 3, 3))

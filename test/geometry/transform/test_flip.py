import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestVflip:
    def smoke_test(self, device, dtype):
        f = kornia.geometry.transform.Vflip()
        repr = "Vflip()"
        assert str(f) == repr

    def test_vflip(self, device, dtype):

        f = kornia.geometry.transform.Vflip()
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype
        )  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_vflip(self, device, dtype):

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.geometry.transform.Vflip()
        expected = torch.tensor(
            [[[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.geometry.transform.vflip(data)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input,))

        # Create new inputs
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 5.0, 0.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor(
            [[[5.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device, dtype=dtype
        )  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_close(actual, expected)

    def test_gradcheck(self, device, dtype):

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.geometry.transform.Vflip(), (input,), raise_exception=True)


class TestHflip:
    def smoke_test(self, device, dtype):
        f = kornia.geometry.transform.Hflip()
        repr = "Hflip()"
        assert str(f) == repr

    def test_hflip(self, device, dtype):

        f = kornia.geometry.transform.Hflip()
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], device=device, dtype=dtype
        )  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_hflip(self, device, dtype):

        input = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.geometry.transform.Hflip()
        expected = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]], device=device, dtype=dtype
        )  # 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.geometry.transform.hflip(data)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input,))

        # Create new inputs
        input = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 5.0, 5.0], [0.0, 0.0, 0.0]]], device=device, dtype=dtype
        )  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_close(actual, expected)

    def test_gradcheck(self, device, dtype):

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.geometry.transform.Hflip(), (input,), raise_exception=True)


class TestRot180:
    def smoke_test(self, device, dtype):
        f = kornia.geometry.transform.Rot180()
        repr = "Rot180()"
        assert str(f) == repr

    def test_rot180(self, device, dtype):

        f = kornia.geometry.transform.Rot180()
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype
        )  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_rot180(self, device, dtype):

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.geometry.transform.Rot180()
        expected = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.geometry.transform.rot180(data)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input,))

        # Create new inputs
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 5.0, 0.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor(
            [[[0.0, 5.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device, dtype=dtype
        )  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_close(actual, expected)

    def test_gradcheck(self, device, dtype):

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.geometry.transform.Rot180(), (input,), raise_exception=True)

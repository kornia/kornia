import kornia
import torch
import pytest

import kornia.testing as utils  # test utils

from test.utils import assert_close
from torch.autograd import gradcheck


class TestVflip:
    def smoke_test(self, device, dtype):
        f = kornia.Vflip()
        repr = "Vflip()"
        assert str(f) == repr

    def test_vflip(self, device, dtype):

        f = kornia.Vflip()
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]], device=device, dtype=dtype)  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_vflip(self, device, dtype):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.Vflip()
        expected = torch.tensor([[[0., 1., 1.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.vflip(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [5., 5., 0.]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[5., 5., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]], device=device, dtype=dtype)  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_close(actual, expected)

    def test_gradcheck(self, device, dtype):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.Vflip(), (input,), raise_exception=True)


class TestHflip:

    def smoke_test(self, device, dtype):
        f = kornia.Hflip()
        repr = "Hflip()"
        assert str(f) == repr

    def test_hflip(self, device, dtype):

        f = kornia.Hflip()
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor([[0., 0., 0.],
                                 [0., 0., 0.],
                                 [1., 1., 0.]], device=device, dtype=dtype)  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_hflip(self, device, dtype):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 1 x 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.Hflip()
        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 0., 0.],
                                  [1., 1., 0.]]], device=device, dtype=dtype)  # 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.hflip(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [5., 5., 0.],
                              [0., 0., 0.]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 5., 5.],
                                  [0., 0., 0.]]], device=device, dtype=dtype)  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_close(actual, expected)

    def test_gradcheck(self, device, dtype):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.Hflip(), (input,), raise_exception=True)


class TestRot180:

    def smoke_test(self, device, dtype):
        f = kornia.Rot180()
        repr = "Rot180()"
        assert str(f) == repr

    def test_rot180(self, device, dtype):

        f = kornia.Rot180()
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor([[1., 1., 0.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]], device=device, dtype=dtype)  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_rot180(self, device, dtype):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.Rot180()
        expected = torch.tensor([[1., 1., 0.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.rot180(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [5., 5., 0.]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 5., 5.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]], device=device, dtype=dtype)  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_close(actual, expected)

    def test_gradcheck(self, device, dtype):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.Rot180(), (input,), raise_exception=True)

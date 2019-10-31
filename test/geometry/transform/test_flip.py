import kornia
import torch
import pytest

import kornia.testing as utils  # test utils
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestVflip:
    def smoke_test(self):
        f = kornia.Vflip()
        repr = "Vflip()"
        assert str(f) == repr

    def test_vflip(self):

        f = kornia.Vflip()
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_vflip(self):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.Vflip()
        expected = torch.tensor([[[0., 1., 1.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]])  # 1 x 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.vflip(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [5., 5., 0.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[5., 5., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]]).double()  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.Vflip(), (input,), raise_exception=True)


class TestHflip:

    def smoke_test(self):
        f = kornia.Hflip()
        repr = "Hflip()"
        assert str(f) == repr

    def test_hflip(self):

        f = kornia.Hflip()
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        expected = torch.tensor([[0., 0., 0.],
                                [0., 0., 0.],
                                [1., 1., 0.]])  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_hflip(self):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 1 x 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.Hflip()
        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 0., 0.],
                                  [1., 1., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.hflip(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [5., 5., 0.],
                              [0., 0., 0.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 5., 5.],
                                  [0., 0., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]]).double()  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.Hflip(), (input,), raise_exception=True)


class TestRot180:

    def smoke_test(self):
        f = kornia.Rot180()
        repr = "Rot180()"
        assert str(f) == repr

    def test_rot180(self):

        f = kornia.Rot180()
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        expected = torch.tensor([[1., 1., 0.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_rot180(self):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.Rot180()
        expected = torch.tensor([[1., 1., 0.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 1 x 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.rot180(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [5., 5., 0.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 5., 5.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]]).double()  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.Rot180(), (input,), raise_exception=True)

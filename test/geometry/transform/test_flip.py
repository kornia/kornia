import kornia
import torch
import pytest

import kornia.testing as utils  # test utils
from torch.testing import assert_allclose
from torch.autograd import gradcheck
from typing import Union, Tuple, List


class TestFlip:
    def smoke_test(self):
        f = kornia.Flip(dims=[-2, -1])
        repr = "Flip(dims=[-2, -1])"
        assert str(f) == repr

    def test_flip(self):

        f = kornia.Flip(-1)
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        expected = torch.tensor([[0., 0., 0.],
                                 [0., 0., 0.],
                                 [1., 1., 0.]])  # 3 x 3

        assert (f(input) == expected).all()

    def test_batch_flip(self):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.Flip(-1)
        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 0., 0.],
                                  [1., 1., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        assert (f(input) == expected).all()

    def test_list_flip(self):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        expected = torch.tensor([[[1., 1., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]])

        f = kornia.Flip([-2, -1])

        assert (f(input) == expected).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor, dims=Union[Tuple[int, ...], List[int], int]) -> torch.Tensor:

            return kornia.flip(data, dims)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 0., 0.],
                                  [1., 1., 0.]]])

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, -1))
        dims = [-2, -1]

        # Create new inputs
        input = torch.tensor([[1., 1., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 1., 1.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input, dims)

        assert_allclose(actual, expected)

    def test_gradcheck(self):

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]]).double()  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.Flip([-2, -1]), (input,), raise_exception=True)

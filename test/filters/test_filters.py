import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestFilter2D:
    def test_smoke(self, device):
        kernel = torch.rand(1, 3, 3).to(device)
        input = torch.ones(1, 1, 7, 8).to(device)

        assert kornia.filter2D(input, kernel).shape == input.shape

    def test_mean_filter(self, device):
        kernel = torch.ones(1, 3, 3).to(device)
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 5., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)
        expected = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 5., 5., 5., 0.],
            [0., 5., 5., 5., 0.],
            [0., 5., 5., 5., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)

        actual = kornia.filter2D(input, kernel)
        assert_allclose(actual, expected)

    def test_mean_filter_2batch_2ch(self, device):
        kernel = torch.ones(1, 3, 3).to(device)
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 5., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).expand(2, 2, -1, -1).to(device)
        expected = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 5., 5., 5., 0.],
            [0., 5., 5., 5., 0.],
            [0., 5., 5., 5., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)

        actual = kornia.filter2D(input, kernel)
        assert_allclose(actual, expected)

    def test_normalized_mean_filter(self, device):
        kernel = torch.ones(1, 3, 3).to(device)
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 5., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).expand(2, 2, -1, -1).to(device)
        expected = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 5. / 9., 5. / 9., 5. / 9., 0.],
            [0., 5. / 9., 5. / 9., 5. / 9., 0.],
            [0., 5. / 9., 5. / 9., 5. / 9., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)
        actual = kornia.filter2D(input, kernel, normalized=True)
        assert_allclose(actual, expected)

    def test_even_sized_filter(self, device):
        kernel = torch.ones(1, 4, 4).to(device)
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 5., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)
        expected = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 5., 5., 0., 0.],
            [0., 5., 5., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)

        actual = kornia.filter2D(input, kernel)
        assert_allclose(actual, expected)

    def test_gradcheck(self, device):
        kernel = torch.rand(1, 3, 3).to(device)
        input = torch.ones(1, 1, 7, 8).to(device)

        # evaluate function gradient
        input = utils.tensor_to_gradcheck_var(input)  # to var
        kernel = utils.tensor_to_gradcheck_var(kernel)  # to var
        assert gradcheck(kornia.filter2D, (input, kernel),
                         raise_exception=True)

    @pytest.mark.skip(reason="not found compute_padding()")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        op = kornia.filter2D
        op = torch.jit.script(op)

        kernel = torch.rand(1, 3, 3)
        input = torch.ones(1, 1, 7, 8)
        expected = op(input, kernel)
        actual = op_script(input, kernel)
        assert_allclose(actual, expected)

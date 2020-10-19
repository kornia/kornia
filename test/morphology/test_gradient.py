import pytest
import torch
import kornia.morphology.gradient as gradient
import kornia.testing as utils  # test utils
from torch.autograd import gradcheck


class TestGradient:
    def test_shape_channels(self, device):
        input = torch.rand(1, 3, 4, 6).to(device)
        kernel = torch.rand(3, 3).to(device)
        test = gradient(input, kernel)
        assert test.shape == (1, 3, 4, 6)

    def test_shape_batch(self, device):
        input = torch.rand(3, 2, 6, 10).to(device)
        kernel = torch.rand(3, 3).to(device)
        test = gradient(input, kernel)
        assert test.shape == (3, 2, 6, 10)

    def test_gradcheck(self, device):
        input = torch.rand(2, 3, 4, 4, requires_grad=True).to(device)
        kernel = torch.rand(3, 3, requires_grad=True).to(device).double()
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(gradient, (input, kernel), raise_exception=True)

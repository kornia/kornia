import pytest
import torch
import kornia.morphology.black_hat as black_hat
import kornia.testing as utils  # test utils
from torch.autograd import gradcheck


class TestBlackHat:
    def test_shape_channels(self, device):
        input = torch.rand(1, 3, 4, 6).to(device)
        kernel = torch.rand(3, 3).to(device)
        test = black_hat(input, kernel)
        assert test.shape == (1, 3, 4, 6)

    def test_shape_batch(self, device):
        input = torch.rand(3, 2, 6, 10).to(device)
        kernel = torch.rand(3, 3).to(device)
        test = black_hat(input, kernel)
        assert test.shape == (3, 2, 6, 10)

    def test_gradcheck(self, device):
        input = torch.rand(2, 3, 4, 4, requires_grad=True).to(device)
        kernel = torch.rand(3, 3, requires_grad=True).to(device).double()
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(black_hat, (input, kernel), raise_exception=True)

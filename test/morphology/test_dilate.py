import pytest
import torch
import kornia.morphology.dilation as dilation
import kornia.testing as utils  # test utils
from torch.autograd import gradcheck


class TestDilate:
    def test_shape_channels(self, device, dtype):
        input = torch.rand(1, 3, 4, 6, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        test = dilation(input, kernel)
        assert test.shape == (1, 3, 4, 6)

    def test_shape_batch(self, device, dtype):
        input = torch.rand(3, 2, 6, 10, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        test = dilation(input, kernel)
        assert test.shape == (3, 2, 6, 10)

    def test_gradcheck(self, device, dtype):
        input = torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, requires_grad=True, device=device, dtype=dtype)
        assert gradcheck(dilation, (input, kernel), raise_exception=True)

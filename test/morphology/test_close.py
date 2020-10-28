import pytest
import torch
import kornia.morphology.close as close
import kornia.testing as utils  # test utils
from torch.autograd import gradcheck


class TestClose:
    def test_shape_channels(self, device, dtype):
        input = torch.rand(1, 3, 4, 6, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        test = close(input, kernel)
        assert test.shape == (1, 3, 4, 6)

    def test_shape_batch(self, device, dtype):
        input = torch.rand(3, 2, 6, 10, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        test = close(input, kernel)
        assert test.shape == (3, 2, 6, 10)

    def test_gradcheck(self, device, dtype):
        input = torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, requires_grad=True, device=device, dtype=dtype)
        assert gradcheck(close, (input, kernel), raise_exception=True)

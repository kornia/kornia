import pytest
import torch
import kornia.morphology.erosion as erosion
import kornia.testing as utils  # test utils
from torch.autograd import gradcheck


class TestErode:
    def test_shape_channels(self, device, dtype):
        input = torch.rand(1, 3, 4, 6, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        test = erosion(input, kernel)
        assert test.shape == (1, 3, 4, 6)

    def test_shape_batch(self, device, dtype):
        input = torch.rand(3, 2, 6, 10, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        test = erosion(input, kernel)
        assert test.shape == (3, 2, 6, 10)

    def test_gradcheck(self, device, dtype):
        input = torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, requires_grad=True, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(erosion, (input, kernel), raise_exception=True)

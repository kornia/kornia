import pytest
import torch

from kornia.filters import UnsharpMask, unsharp_mask
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class Testunsharp(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [3, (5, 3)])
    @pytest.mark.parametrize("sigma", [(3.0, 1.0), (0.5, -0.1)])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_smoke(self, shape, kernel_size, sigma, params_as_tensor, device, dtype):
        if params_as_tensor is True:
            sigma = torch.tensor([sigma], device=device, dtype=dtype).repeat(shape[0], 1)

        inpt = torch.ones(shape, device=device, dtype=dtype)
        actual = unsharp_mask(inpt, kernel_size, sigma, 'replicate')
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_cardinality(self, shape, device, dtype):
        kernel_size = (5, 7)
        sigma = (1.5, 2.1)

        inpt = torch.ones(shape, device=device, dtype=dtype)
        actual = unsharp_mask(inpt, kernel_size, sigma, "replicate")
        assert actual.shape == shape

    @pytest.mark.skip(reason='nothing to test')
    def test_exception(self):
        ...

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inpt = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = (3, 3)
        sigma = (1.5, 2.1)
        actual = unsharp_mask(inpt, kernel_size, sigma, "replicate")
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        # test parameters
        shape = (1, 3, 5, 5)
        kernel_size = (3, 3)
        sigma = (1.5, 2.1)

        # evaluate function gradient
        inpt = torch.rand(shape, device=device)
        inpt = tensor_to_gradcheck_var(inpt)  # to var
        self.gradcheck(unsharp_mask, (inpt, kernel_size, sigma, "replicate"))

    def test_module(self, device, dtype):
        params = [(3, 3), (1.5, 1.5)]
        op = unsharp_mask
        op_module = UnsharpMask(*params)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.parametrize("sigma", [(3.0, 1.0), (0.5, -0.1)])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_dynamo(self, sigma, params_as_tensor, device, dtype, torch_optimizer):
        if params_as_tensor is True:
            sigma = torch.tensor([sigma], device=device, dtype=dtype)

        inpt = torch.ones(1, 3, 10, 10, device=device, dtype=dtype)
        op = UnsharpMask(3, sigma)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))

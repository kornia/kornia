import math
import pytest

import kornia
import kornia.testing as utils  # test utils


import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestZCA:

    @pytest.mark.parametrize("unbiased", [True, False])
    def test_zca_unbiased(self, unbiased, device):

        data = torch.tensor([[0, 1],
                             [1, 0],
                             [-1, 0],
                             [0, -1]], dtype=torch.float32).to(device)

        if unbiased:
            expected = torch.sqrt(1.5 * torch.abs(data)) * torch.sign(data)
        else:
            expected = torch.sqrt(2 * torch.abs(data)) * torch.sign(data)

        expected = expected.to(device)

        zca = kornia.color.ZCAWhitening(unbiased=unbiased).fit(data)

        actual = zca(data)

        assert_allclose(actual, expected)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_dim_args(self, dim, device):

        data = torch.tensor([[0, 1],
                             [1, 0],
                             [-1, 0],
                             [0, -1]], dtype=torch.float32).to(device)

        if dim == 1:
            expected = torch.tensor([[-0.35360718, 0.35360718],
                                     [0.35351562, -0.35351562],
                                     [-0.35353088, 0.35353088],
                                     [0.35353088, -0.35353088]], dtype=torch.float32)
        elif dim == 0:
            expected = torch.tensor([[0., 1.2247448],
                                     [1.2247448, 0.],
                                     [-1.2247448, 0.],
                                     [0., -1.2247448]], dtype=torch.float32)
        expected = expected.to(device)

        zca = kornia.color.ZCAWhitening(dim=dim)
        actual = zca(data, True)

        assert_allclose(actual, expected)

    @pytest.mark.parametrize("input_shape", [(15, 2, 2, 2), (10, 4), (20, 3, 2, 2)])
    def test_identity(self, input_shape, device):
        """

        Assert that data can be recovered by the inverse transform

        """

        data = torch.randn(*input_shape, dtype=torch.float32).to(device)

        zca = kornia.color.ZCAWhitening(compute_inv=True).fit(data)

        data_w = zca(data)

        data_hat = zca.inverse_transform(data_w)

        assert_allclose(data, data_hat)

    def test_grad_zca_individual_transforms(self, device):
        """

        Checks if the gradients of the transforms are correct w.r.t to the input data

        """

        data = torch.tensor([[2, 0],
                             [0, 1],
                             [-2, 0],
                             [0, -1]],
                            dtype=torch.float32).to(device)

        data = utils.tensor_to_gradcheck_var(data)

        def zca_T(x):
            return kornia.color.zca_mean(x)[0]

        def zca_mu(x):
            return kornia.color.zca_mean(x)[1]

        def zca_T_inv(x):
            return kornia.color.zca_mean(x, return_inverse=True)[2]

        assert gradcheck(zca_T, (data,), raise_exception=True)
        assert gradcheck(zca_mu, (data,), raise_exception=True)
        assert gradcheck(zca_T_inv, (data,), raise_exception=True)

    def test_grad_zca_with_fit(self, device):

        data = torch.tensor([[2, 0],
                             [0, 1],
                             [-2, 0],
                             [0, -1]],
                            dtype=torch.float32).to(device)

        data = utils.tensor_to_gradcheck_var(data)

        def zca_fit(x):
            zca = kornia.color.ZCAWhitening(detach_transforms=False)
            return zca(x, include_fit=True)

        assert gradcheck(zca_fit, (data,), raise_exception=True)

    def test_grad_detach_zca(self, device):

        data = torch.tensor([[1, 0],
                             [0, 1],
                             [-2, 0],
                             [0, -1]],
                            dtype=torch.float32).to(device)

        data = utils.tensor_to_gradcheck_var(data)
        zca = kornia.color.ZCAWhitening()

        zca.fit(data)

        assert gradcheck(zca,
                         (data,), raise_exception=True)

    def test_not_fitted(self, device):

        with pytest.raises(RuntimeError):
            data = torch.rand(10, 2).to(device)

            zca = kornia.color.ZCAWhitening()
            zca(data)

    def test_not_fitted_inv(self, device):

        with pytest.raises(RuntimeError):
            data = torch.rand(10, 2).to(device)

            zca = kornia.color.ZCAWhitening()
            zca.inverse_transform(data)

    def test_jit(self, device, dtype):

        data = torch.rand((10, 3, 1, 2)).to(device)
        zca = kornia.color.ZCAWhitening().fit(data)
        zca_jit = kornia.color.ZCAWhitening().fit(data)
        zca_jit = torch.jit.script(zca_jit)
        assert_allclose(zca_jit(data), zca(data))

    @pytest.mark.parametrize("unbiased", [True, False])
    def test_zca_whiten_func_unbiased(self, unbiased, device):

        data = torch.tensor([[0, 1],
                             [1, 0],
                             [-1, 0],
                             [0, -1]], dtype=torch.float32).to(device)

        if unbiased:
            expected = torch.sqrt(1.5 * torch.abs(data)) * torch.sign(data)
        else:
            expected = torch.sqrt(2 * torch.abs(data)) * torch.sign(data)

        expected = expected.to(device)

        actual = kornia.zca_whiten(data, unbiased=unbiased)

        assert_allclose(actual, expected)

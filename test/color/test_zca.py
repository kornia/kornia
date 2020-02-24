import math
import pytest

import kornia
import kornia.testing as utils  # test utils

import numpy as np

import torch
from torchvision.transforms import LinearTransformation
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from test.common import device


class TestZCA:

    @pytest.mark.parametrize("biased", [True, False])
    def test_zca(self, biased, device):
        """

        Checks to see if check zca transformed data and the corresponding transform matrices
        are correctly calculated


        """

        eps = 1e-7
        x = np.random.rand(10, 3)
        x_center = x - np.mean(x, axis=0, keepdims=True)

        if biased:
            cov = np.dot(x_center.T, x_center) / (x.shape[0])
        else:
            cov = np.dot(x_center.T, x_center) / (x.shape[0] - 1)

        U, S, _ = np.linalg.svd(cov)
        s = np.sqrt(S + eps)
        s_inv = np.diag(1. / s)
        s = np.diag(s)
        T = np.dot(np.dot(U, s_inv), U.T).T

        data = torch.tensor(x, dtype=torch.float32).to(device)
        T_expected = torch.tensor(T, dtype=torch.float32).to(device)

        zca = kornia.color.ZCAWhiten(biased=biased, eps=eps).fit(data)

        assert_allclose(zca.T, T_expected)

    @pytest.mark.parametrize("input_shape", [(10, 2, 2, 2), (10, 4), (15, 3, 1, 3)])
    def test_identity(self, input_shape, device):
        """

        Assert that data can be recovered by the inverse transform

        """

        data = torch.rand(*input_shape, dtype=torch.float32).to(device)

        zca = kornia.color.ZCAWhiten(compute_inv=True).fit(data)

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
            return kornia.color.zca_whiten_transforms(x)[0]

        def zca_mu(x):
            return kornia.color.zca_whiten_transforms(x)[1]

        def zca_T_inv(x):
            return kornia.color.zca_whiten_transforms(x, compute_inv=True)[2]

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
            zca = kornia.color.ZCAWhiten(detach_transforms=False)
            return zca(x, include_fit=True)

        assert gradcheck(zca_fit, (data,), raise_exception=True)

    def test_grad_detach_zca(self, device):

        data = torch.tensor([[2, 0],
                             [0, 1],
                             [-2, 0],
                             [0, -1]],
                            dtype=torch.float32).to(device)

        data = utils.tensor_to_gradcheck_var(data)
        zca = kornia.color.ZCAWhiten(detach_transforms=True).fit(data)

        assert gradcheck(zca,
                         (data,), raise_exception=True)

    def test_not_fitted(self, device):

        with pytest.raises(RuntimeError):
            data = torch.rand(10, 2).to(device)

            zca = kornia.color.ZCAWhiten()
            zca(data)

    def test_not_fitted_inv(self, device):

        with pytest.raises(RuntimeError):
            data = torch.rand(10, 2).to(device)

            zca = kornia.color.ZCAWhiten()
            zca.inverse_transform(data)

    def test_with_linear_transform(self, device):
        data = torch.tensor([[1, 0],
                             [0, 1],
                             [-1, 0],
                             [0, -1]],
                            dtype=torch.float32).to(device)
        data = data.view(4, 1, 2, 1)
        expected = math.sqrt(3 / 2) * data

        T, mu = kornia.color.zca_whiten_transforms(data)

        lt = LinearTransformation(T, mu)

        out = torch.stack([lt(data[0]), lt(data[1]), lt(data[2]), lt(data[3])], axis=0)

        assert_allclose(out, expected)

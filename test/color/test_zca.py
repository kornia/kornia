import math
import pytest

import kornia
import kornia.testing as utils  # test utils

import numpy as np

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from test.common import device


class TestZCA:

    @pytest.mark.parametrize("biased",[True, False])
    def test_zca(self, biased, device):


        eps = 1e-7
        x = np.random.rand(10,3)
        x_center = x - np.mean(x, axis=0, keepdims=True)


        if biased:
            cov = np.dot(x_center.T, x_center) / (x.shape[0])
        else:
            cov = np.dot(x_center.T, x_center) / (x.shape[0]-1)

        U, S, _ = np.linalg.svd(cov)
        s = np.sqrt(S+eps)
        s_inv = np.diag(1./s)
        s = np.diag(s)
        T = np.dot(np.dot(U, s_inv), U.T).T

        data = torch.tensor(x, dtype=torch.float32).to(device)
        T_expected = torch.tensor(T, dtype=torch.float32).to(device)

        zca = kornia.color.ZCAWhiten(biased=biased, eps = eps).fit(data)

        assert_allclose(zca.T,T_expected)


    @pytest.mark.parametrize("input_shape", [(10, 2, 2, 2), (10, 4), (15, 3, 1, 3)])
    def test_identity(self, input_shape, device):

        data = torch.rand(*input_shape, dtype = torch.float32).to(device)

        zca = kornia.color.ZCAWhiten(compute_inv=True).fit(data)

        data_w = zca(data)

        data_hat = zca(data_w, True)

        assert_allclose(data, data_hat)

    def test_grad_zca_individual_transforms(self, device):

        data = torch.tensor([[2,0],
                        [0,1],
                        [-2,0],
                        [0, -1]],
                        dtype=torch.float32).to(device)

        data = utils.tensor_to_gradcheck_var(data) 

        zca_T = lambda x: kornia.color.zca_whiten_transforms(x)[0]
        zca_mu = lambda x: kornia.color.zca_whiten_transforms(x)[1]
        zca_T_inv = lambda x: kornia.color.zca_whiten_transforms(x, compute_inv=True)[2]

        assert gradcheck(zca_T, (data,), raise_exception=True)
        assert gradcheck(zca_mu, (data,), raise_exception=True)
        assert gradcheck(zca_T_inv, (data,), raise_exception=True)

    def test_grad_full_zca(self, device):

        data = torch.tensor([[2,0],
                [0,1],
                [-2,0],
                [0, -1]],
                dtype=torch.float32).to(device)

        data = utils.tensor_to_gradcheck_var(data) 

        assert gradcheck(kornia.color.ZCAWhiten(fit_in_forward=True, detach_transforms=False),
                                                 (data,), raise_exception=True)












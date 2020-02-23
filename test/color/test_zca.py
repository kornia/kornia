import math
import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from test.common import device


class TestZCA:

    @pytest.mark.parametrize("biased",[True, False])
    def test_zca(self, biased, device):

        data = torch.tensor([[1,0],
                             [0,1],
                             [-1,0],
                             [0, -1]],
                             dtype=torch.float32).to(device)


        expected = torch.tensor([[1,0],
                                    [0,1],
                                    [-1,0],
                                    [0, -1]],
                                    dtype=torch.float32)

        if not biased:
            expected *= math.sqrt(3/2)
        else:
            expected *= math.sqrt(2) 

        expected.to(device)

        zca = kornia.color.ZCAWhiten(biased=biased).fit(data)

        data_w = zca(data)
        assert_allclose(data_w, expected)



    @pytest.mark.parametrize("input_shape", [(10, 2, 2, 2), (10, 4), (15, 3, 1, 3)])
    def test_identity(self, input_shape, device):

        data = torch.rand(*input_shape, dtype = torch.float32).to(device)

        zca = kornia.color.ZCAWhiten(compute_inv=True).fit(data)

        data_w = zca(data)

        data_hat = zca(data_w, True)

        assert_allclose(data, data_hat)

    def test_grad(self, device):

        data = torch.tensor([[1,0],
                        [0,1],
                        [-1,0],
                        [0, -1]],
                        dtype=torch.float32).to(device)
                        
        data = utils.tensor_to_gradcheck_var(data) 

        zca = kornia.color.ZCAWhiten().fit(data)

        assert gradcheck(zca, (data,), raise_exception=True)



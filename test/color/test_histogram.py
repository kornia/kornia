import pytest
import numpy as np
import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

from kornia.color import histogram, histogram2d
import kornia.testing as utils  # test utils


class TestHistogram:
    def test_shape(self, device):
        inp = torch.ones(1, 32, device=device)
        bins = torch.linspace(0, 255, 128).to(device)
        bandwidth = torch.Tensor(np.array(0.9)).to(device)
        pdf = histogram(inp, bins, bandwidth)
        assert pdf.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(8, 32, device=device)
        bins = torch.linspace(0, 255, 128).to(device)
        bandwidth = torch.Tensor(np.array(0.9)).to(device)
        pdf = histogram(inp, bins, bandwidth)
        assert pdf.shape == (8, 128)

    def test_gradcheck(self, device):
        inp = torch.ones(8, 32, device=device)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        bins = torch.linspace(0, 255, 128).to(device)
        bins = utils.tensor_to_gradcheck_var(bins)
        bandwidth = torch.Tensor(np.array(0.9)).to(device)
        bandwidth = utils.tensor_to_gradcheck_var(bandwidth)
        assert gradcheck(histogram, (inp, bins, bandwidth), raise_exception=True)

    def test_uniform_dist(self, device):
        input1 = torch.linspace(0, 255, 10).unsqueeze(0).to(device)

        pdf = histogram(input1, torch.linspace(0, 255, 10).to(device), torch.Tensor(np.array(2 * 0.4**2)))
        ans = torch.ones((1, 10)) * 0.1
        assert((ans.cpu() - pdf.cpu()).sum() < 1e-6)


class TestHistogram2d:
    def test_shape(self, device):
        inp1 = torch.ones(1, 32, device=device)
        inp2 = torch.ones(1, 32, device=device)
        bins = torch.linspace(0, 255, 128).to(device)
        bandwidth = torch.Tensor(np.array(0.9)).to(device)
        pdf = histogram2d(inp1, inp2, bins, bandwidth)
        assert pdf.shape == (1, 128, 128)

    def test_shape_batch(self, device):
        inp1 = torch.ones(8, 32, device=device)
        inp2 = torch.ones(8, 32, device=device)
        bins = torch.linspace(0, 255, 128).to(device)
        bandwidth = torch.Tensor(np.array(0.9)).to(device)
        pdf = histogram2d(inp1, inp2, bins, bandwidth)
        assert pdf.shape == (8, 128, 128)

    def test_gradcheck(self, device):
        inp1 = torch.ones(3, 16, device=device)
        inp2 = torch.ones(3, 16, device=device)
        inp1 = utils.tensor_to_gradcheck_var(inp1)  # to var
        inp2 = utils.tensor_to_gradcheck_var(inp2)  # to var
        bins = torch.linspace(0, 255, 64).to(device)
        bins = utils.tensor_to_gradcheck_var(bins)
        bandwidth = torch.Tensor(np.array(0.9)).to(device)
        bandwidth = utils.tensor_to_gradcheck_var(bandwidth)
        assert gradcheck(histogram2d, (inp1, inp2, bins, bandwidth), raise_exception=True)

    def test_uniform_dist(self, device):
        input1 = torch.linspace(0, 255, 10).unsqueeze(0).to(device)
        input2 = torch.linspace(0, 255, 10).unsqueeze(0).to(device)

        joint_pdf = histogram2d(
            input1,
            input2,
            torch.linspace(0, 255, 10).to(device),
            torch.Tensor(np.array(2 * 0.4**2)))

        ans = torch.eye(10).unsqueeze(0) * 0.1
        assert((ans.cpu() - joint_pdf.cpu()).sum() < 1e-6)

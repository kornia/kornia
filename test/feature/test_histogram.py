import pytest
from test.common import device

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

from kornia.feature import histogram, histogram2d
import kornia.testing as utils  # test utils


class TestHistogram:
    def test_shape(self, device):
        inp = torch.ones(1, 32, device=device)
        bins = torch.linspace(0, 255, 128).to(device)
        pdf = histogram(inp, bins, 1)
        assert pdf.shape == (1, 128)

    def test_shape_batch(self, device):
        inp = torch.ones(8, 32, device=device)
        bins = torch.linspace(0, 255, 128).to(device)
        pdf = histogram(inp, bins, 1)
        assert pdf.shape == (8, 128)

    def test_gradcheck(self, device):
        inp = torch.ones(8, 32, device=device)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        bins = torch.linspace(0, 255, 128).to(device)

        assert gradcheck(histogram, (inp, bins, 1), raise_exception=True)


class TestHistogram2d:
    def test_shape(self, device):
        inp1 = torch.ones(1, 32, device=device)
        inp2 = torch.ones(1, 32, device=device)
        bins = torch.linspace(0, 255, 128).to(device)
        pdf = histogram2d(inp1, inp2, bins, 1)
        assert pdf.shape == (1, 128, 128)

    def test_shape_batch(self, device):
        inp1 = torch.ones(8, 32, device=device)
        inp2 = torch.ones(8, 32, device=device)
        bins = torch.linspace(0, 255, 128).to(device)
        pdf = histogram2d(inp1, inp2, bins, 1)
        assert pdf.shape == (8, 128, 128)

    def test_gradcheck(self, device):
        inp1 = torch.ones(3, 16, device=device)
        inp2 = torch.ones(3, 16, device=device)
        inp1 = utils.tensor_to_gradcheck_var(inp1)  # to var
        inp2 = utils.tensor_to_gradcheck_var(inp2)  # to var
        bins = torch.linspace(0, 255, 64).to(device)

        assert gradcheck(histogram2d, (inp1, inp2, bins, 1), raise_exception=True)

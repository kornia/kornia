import pytest
import numpy as np
import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils


class TestHistogram:

    fcn = kornia.enhance.histogram

    def test_shape(self, device, dtype):
        inp = torch.ones(1, 32, device=device, dtype=dtype)
        bins = torch.linspace(0, 255, 128, device=device, dtype=dtype)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        pdf = TestHistogram.fcn(inp, bins, bandwidth)
        assert pdf.shape == (1, 128)

    def test_shape_batch(self, device, dtype):
        inp = torch.ones(8, 32, device=device, dtype=dtype)
        bins = torch.linspace(0, 255, 128, device=device, dtype=dtype)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        pdf = TestHistogram.fcn(inp, bins, bandwidth)
        assert pdf.shape == (8, 128)

    def test_gradcheck(self, device, dtype):
        inp = torch.ones(1, 8, device=device, dtype=dtype)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        bins = torch.linspace(0, 255, 8, device=device, dtype=dtype)
        bins = utils.tensor_to_gradcheck_var(bins)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        bandwidth = utils.tensor_to_gradcheck_var(bandwidth)
        assert gradcheck(TestHistogram.fcn, (inp, bins, bandwidth), raise_exception=True)

    def test_jit(self, device, dtype):
        input1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        bins = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4 ** 2, device=device, dtype=dtype)
        inputs = (input1, bins, bandwidth)

        op = TestHistogram.fcn
        op_script = torch.jit.script(op)

        assert_allclose(op(*inputs), op_script(*inputs))

    def test_uniform_dist(self, device, dtype):
        input1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        input2 = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4 ** 2, device=device, dtype=dtype)

        pdf = TestHistogram.fcn(input1, input2, bandwidth)
        ans = 0.1 * torch.ones(1, 10, device=device, dtype=dtype)
        assert_allclose(ans, pdf)


class TestHistogram2d:

    fcn = kornia.enhance.histogram2d

    def test_shape(self, device, dtype):
        inp1 = torch.ones(1, 32, device=device, dtype=dtype)
        inp2 = torch.ones(1, 32, device=device, dtype=dtype)
        bins = torch.linspace(0, 255, 128, device=device, dtype=dtype)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        pdf = TestHistogram2d.fcn(inp1, inp2, bins, bandwidth)
        assert pdf.shape == (1, 128, 128)

    def test_shape_batch(self, device, dtype):
        inp1 = torch.ones(8, 32, device=device, dtype=dtype)
        inp2 = torch.ones(8, 32, device=device, dtype=dtype)
        bins = torch.linspace(0, 255, 128, device=device, dtype=dtype)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        pdf = TestHistogram2d.fcn(inp1, inp2, bins, bandwidth)
        assert pdf.shape == (8, 128, 128)

    def test_gradcheck(self, device, dtype):
        inp1 = torch.ones(1, 8, device=device, dtype=dtype)
        inp2 = torch.ones(1, 8, device=device, dtype=dtype)
        inp1 = utils.tensor_to_gradcheck_var(inp1)  # to var
        inp2 = utils.tensor_to_gradcheck_var(inp2)  # to var
        bins = torch.linspace(0, 255, 8, device=device, dtype=dtype)
        bins = utils.tensor_to_gradcheck_var(bins)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        bandwidth = utils.tensor_to_gradcheck_var(bandwidth)
        assert gradcheck(TestHistogram2d.fcn, (inp1, inp2, bins, bandwidth), raise_exception=True)

    def test_jit(self, device, dtype):
        input1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        input2 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        bins = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4 ** 2, device=device, dtype=dtype)
        inputs = (input1, input2, bins, bandwidth)

        op = TestHistogram2d.fcn
        op_script = torch.jit.script(op)

        assert_allclose(op(*inputs), op_script(*inputs))

    def test_uniform_dist(self, device, dtype):
        input1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        input2 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        bins = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4 ** 2, device=device, dtype=dtype)

        pdf = TestHistogram2d.fcn(input1, input2, bins, bandwidth)
        ans = 0.1 * kornia.eye_like(10, pdf)
        assert_allclose(ans, pdf)

import pytest
import torch
from packaging import version

import kornia

from testing.base import BaseTester


class TestImageHistogram2d(BaseTester):
    fcn = kornia.enhance.image_histogram2d

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_shape(self, device, dtype, kernel):
        sample = torch.ones(32, 32, device=device, dtype=dtype)
        hist, pdf = TestImageHistogram2d.fcn(sample, 0.0, 1.0, 256, kernel=kernel)
        assert hist.shape == (256,)
        assert pdf.shape == (256,)

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_shape_channels(self, device, dtype, kernel):
        sample = torch.ones(3, 32, 32, device=device, dtype=dtype)
        hist, pdf = TestImageHistogram2d.fcn(sample, 0.0, 1.0, 256, kernel=kernel)
        assert hist.shape == (3, 256)
        assert pdf.shape == (3, 256)

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_shape_batch(self, device, dtype, kernel):
        sample = torch.ones(8, 3, 32, 32, device=device, dtype=dtype)
        hist, pdf = TestImageHistogram2d.fcn(sample, 0.0, 1.0, 256, kernel=kernel)
        assert hist.shape == (8, 3, 256)
        assert pdf.shape == (8, 3, 256)

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_gradcheck(self, device, kernel):
        sample = torch.ones(32, 32, device=device, dtype=torch.float64)
        centers = torch.linspace(0, 255, 8, device=device, dtype=torch.float64)
        self.gradcheck(TestImageHistogram2d.fcn, (sample, 0.0, 255.0, 256, None, centers, True, kernel))

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("1.9"), reason="Tuple cannot be jitted with PyTorch < v1.9"
    )
    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_jit(self, device, dtype, kernel):
        sample = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        sample_x, _ = torch.meshgrid(sample, sample)
        samples = (sample_x, 0.0, 255.0, 10, None, None, False, kernel)

        op = TestImageHistogram2d.fcn
        op_script = torch.jit.script(op)

        out, out_script = op(*samples), op_script(*samples)
        self.assert_close(out[0], out_script[0])
        self.assert_close(out[1], out_script[1])

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    @pytest.mark.parametrize("size", [(1, 1), (3, 1, 1), (8, 3, 1, 1)])
    def test_uniform_hist(self, device, dtype, kernel, size):
        sample = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        sample_x, _ = torch.meshgrid(sample, sample)
        sample_x = sample_x.repeat(*size)
        if kernel == "gaussian":
            bandwidth = 2 * 0.4**2
        else:
            bandwidth = None
        hist, _ = TestImageHistogram2d.fcn(sample_x, 0.0, 255.0, 10, bandwidth=bandwidth, centers=sample, kernel=kernel)
        ans = 10 * torch.ones_like(hist)
        self.assert_close(ans, hist)

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    @pytest.mark.parametrize("size", [(1, 1), (3, 1, 1), (8, 3, 1, 1)])
    def test_uniform_dist(self, device, dtype, kernel, size):
        sample = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        sample_x, _ = torch.meshgrid(sample, sample)
        sample_x = sample_x.repeat(*size)
        if kernel == "gaussian":
            bandwidth = 2 * 0.4**2
        else:
            bandwidth = None
        hist, pdf = TestImageHistogram2d.fcn(
            sample_x, 0.0, 255.0, 10, bandwidth=bandwidth, centers=sample, kernel=kernel, return_pdf=True
        )
        ans = 0.1 * torch.ones_like(hist)
        self.assert_close(ans, pdf)


class TestHistogram2d(BaseTester):
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

    def test_gradcheck(self, device):
        inp1 = torch.ones(1, 8, device=device, dtype=torch.float64)
        inp2 = torch.ones(1, 8, device=device, dtype=torch.float64)
        bins = torch.linspace(0, 255, 8, device=device, dtype=torch.float64)
        bandwidth = torch.tensor(0.9, device=device, dtype=torch.float64)
        self.gradcheck(TestHistogram2d.fcn, (inp1, inp2, bins, bandwidth))

    def test_jit(self, device, dtype):
        sample1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        sample2 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        bins = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4**2, device=device, dtype=dtype)
        samples = (sample1, sample2, bins, bandwidth)

        op = TestHistogram2d.fcn
        op_script = torch.jit.script(op)

        self.assert_close(op(*samples), op_script(*samples))

    def test_uniform_dist(self, device, dtype):
        sample1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        sample2 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        bins = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4**2, device=device, dtype=dtype)

        pdf = TestHistogram2d.fcn(sample1, sample2, bins, bandwidth)
        ans = 0.1 * kornia.eye_like(10, pdf)
        self.assert_close(ans, pdf)


class TestHistogram(BaseTester):
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

    def test_gradcheck(self, device):
        inp = torch.ones(1, 8, device=device, dtype=torch.float64)
        bins = torch.linspace(0, 255, 8, device=device, dtype=torch.float64)
        bandwidth = torch.tensor(0.9, device=device, dtype=torch.float64)
        self.gradcheck(TestHistogram.fcn, (inp, bins, bandwidth))

    def test_jit(self, device, dtype):
        input1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        bins = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4**2, device=device, dtype=dtype)
        inputs = (input1, bins, bandwidth)

        op = TestHistogram.fcn
        op_script = torch.jit.script(op)

        self.assert_close(op(*inputs), op_script(*inputs))

    def test_uniform_dist(self, device, dtype):
        input1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        input2 = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4**2, device=device, dtype=dtype)

        pdf = TestHistogram.fcn(input1, input2, bandwidth)
        ans = 0.1 * torch.ones(1, 10, device=device, dtype=dtype)
        self.assert_close(ans, pdf)

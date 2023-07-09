import pytest
import torch
from packaging import version
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestImageHistogram2d:
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
    def test_gradcheck(self, device, dtype, kernel):
        sample = torch.ones(32, 32, device=device, dtype=dtype)
        sample = utils.tensor_to_gradcheck_var(sample)  # to var
        centers = torch.linspace(0, 255, 8, device=device, dtype=dtype)
        centers = utils.tensor_to_gradcheck_var(centers)
        assert gradcheck(
            TestImageHistogram2d.fcn,
            (sample, 0.0, 255.0, 256, None, centers, True, kernel),
            raise_exception=True,
            fast_mode=True,
        )

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
        assert_close(out[0], out_script[0])
        assert_close(out[1], out_script[1])

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
        assert_close(ans, hist)

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
        assert_close(ans, pdf)


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
        assert gradcheck(TestHistogram2d.fcn, (inp1, inp2, bins, bandwidth), raise_exception=True, fast_mode=True)

    def test_jit(self, device, dtype):
        sample1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        sample2 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        bins = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4**2, device=device, dtype=dtype)
        samples = (sample1, sample2, bins, bandwidth)

        op = TestHistogram2d.fcn
        op_script = torch.jit.script(op)

        assert_close(op(*samples), op_script(*samples))

    def test_uniform_dist(self, device, dtype):
        sample1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        sample2 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        bins = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4**2, device=device, dtype=dtype)

        pdf = TestHistogram2d.fcn(sample1, sample2, bins, bandwidth)
        ans = 0.1 * kornia.eye_like(10, pdf)
        assert_close(ans, pdf)


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
        assert gradcheck(TestHistogram.fcn, (inp, bins, bandwidth), raise_exception=True, fast_mode=True)

    def test_jit(self, device, dtype):
        input1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        bins = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4**2, device=device, dtype=dtype)
        inputs = (input1, bins, bandwidth)

        op = TestHistogram.fcn
        op_script = torch.jit.script(op)

        assert_close(op(*inputs), op_script(*inputs))

    def test_uniform_dist(self, device, dtype):
        input1 = torch.linspace(0, 255, 10, device=device, dtype=dtype).unsqueeze(0)
        input2 = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        bandwidth = torch.tensor(2 * 0.4**2, device=device, dtype=dtype)

        pdf = TestHistogram.fcn(input1, input2, bandwidth)
        ans = 0.1 * torch.ones(1, 10, device=device, dtype=dtype)
        assert_close(ans, pdf)

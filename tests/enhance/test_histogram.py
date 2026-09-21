# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

import kornia
from kornia.core._compat import torch_version_lt

from testing.base import BaseTester


class TestImageHistogram2d(BaseTester):
    fcn = kornia.enhance.image_histogram2d

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_shape(self, device, dtype, kernel):
        sample = torch.ones(16, 16, device=device, dtype=dtype)
        hist, pdf = TestImageHistogram2d.fcn(sample, 0.0, 1.0, 32, kernel=kernel)
        assert hist.shape == (32,)
        assert pdf.shape == (32,)

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_shape_channels(self, device, dtype, kernel):
        sample = torch.ones(3, 16, 16, device=device, dtype=dtype)
        hist, pdf = TestImageHistogram2d.fcn(sample, 0.0, 1.0, 32, kernel=kernel)
        assert hist.shape == (3, 32)
        assert pdf.shape == (3, 32)

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_shape_batch(self, device, dtype, kernel):
        sample = torch.ones(4, 3, 16, 16, device=device, dtype=dtype)
        hist, pdf = TestImageHistogram2d.fcn(sample, 0.0, 1.0, 32, kernel=kernel)
        assert hist.shape == (4, 3, 32)
        assert pdf.shape == (4, 3, 32)

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_gradcheck(self, device, kernel):
        sample = torch.ones(8, 8, device=device, dtype=torch.float64)
        centers = torch.linspace(0, 255, 8, device=device, dtype=torch.float64)
        self.gradcheck(TestImageHistogram2d.fcn, (sample, 0.0, 255.0, 8, None, centers, True, kernel))

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    def test_jit(self, device, dtype, kernel):
        sample = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        sample_x, _ = torch.meshgrid(sample, sample, indexing="ij")
        samples = (sample_x, 0.0, 255.0, 10, None, None, False, kernel)

        op = TestImageHistogram2d.fcn
        op_script = torch.jit.script(op)

        out, out_script = op(*samples), op_script(*samples)
        self.assert_close(out[0], out_script[0])
        self.assert_close(out[1], out_script[1])

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    @pytest.mark.parametrize("size", [(1, 1), (3, 1, 1), (4, 3, 1, 1)])
    def test_uniform_hist(self, device, dtype, kernel, size):
        sample = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        sample_x, _ = torch.meshgrid(sample, sample, indexing="ij")
        sample_x = sample_x.repeat(*size)
        if kernel == "gaussian":
            bandwidth = 2 * 0.4**2
        else:
            bandwidth = None
        hist, _ = TestImageHistogram2d.fcn(sample_x, 0.0, 255.0, 10, bandwidth=bandwidth, centers=sample, kernel=kernel)
        ans = 10 * torch.ones_like(hist)
        self.assert_close(ans, hist)

    @pytest.mark.parametrize("kernel", ["triangular", "gaussian", "uniform", "epanechnikov"])
    @pytest.mark.parametrize("size", [(1, 1), (3, 1, 1), (4, 3, 1, 1)])
    def test_uniform_dist(self, device, dtype, kernel, size):
        sample = torch.linspace(0, 255, 10, device=device, dtype=dtype)
        sample_x, _ = torch.meshgrid(sample, sample, indexing="ij")
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

    def test_large_n_bins_float16_centers_not_collapsed(self, device):
        """image_histogram2d built its bin-center arange at image.dtype -- float16 can only
        exactly represent integers up to 2048 (torch.arange(4096, dtype=torch.float16) has only
        3073 distinct values, not 4096), so with enough bins, DIFFERENT bin indices silently
        collapsed onto the SAME center value, giving those bins numerically IDENTICAL kernel-
        density output regardless of the real underlying data spread.

        Compares the function's own auto-constructed centers against EXPLICITLY-supplied
        float32 centers on the *same* float16 image tensor (the `centers=` argument bypasses
        internal construction entirely) -- this isolates the bug mechanism cleanly, with no
        confound from float16 also quantizing the image's own pixel VALUES (a separate, benign,
        expected precision limit unrelated to how centers are built). float32 is an exact
        reference here: n_bins=4096 is far below float32's 2**24 integer limit (MPS has no
        float64). The tolerance is far below the error the collapse causes."""
        torch.manual_seed(0)
        n_bins = 4096
        image = torch.rand(1, 64, 64, device=device, dtype=torch.float16)

        hist_auto, _ = TestImageHistogram2d.fcn(image, 0.0, 1.0, n_bins, kernel="gaussian")

        explicit_centers = 0.0 + (1.0 / n_bins) * (torch.arange(n_bins, device=device, dtype=torch.float32) + 0.5)
        hist_explicit, _ = TestImageHistogram2d.fcn(
            image, 0.0, 1.0, n_bins, centers=explicit_centers, kernel="gaussian"
        )

        self.assert_close(hist_auto.float(), hist_explicit.float(), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("return_pdf", [False, True])
    def test_auto_centers_preserve_input_dtype(self, device, dtype, return_pdf):
        """The float32-centers fix above (test_large_n_bins_float16_centers_not_collapsed)
        must not itself change the function's public return dtype: before that fix, a
        float16/bfloat16 image with centers=None returned float16/bfloat16 hist/pdf. Building
        centers at float32 promotes u/kernel_values through ordinary PyTorch type promotion,
        so hist/pdf must be explicitly cast back to image.dtype at the return boundary."""
        if device.type == "mps" and dtype == torch.bfloat16 and torch_version_lt(2, 14, 0):
            pytest.skip("bfloat16 on MPS is only exercised on torch >= 2.14 (the version of the blocking MPS job)")
        torch.manual_seed(0)
        image = torch.rand(1, 16, 16, device=device, dtype=dtype)

        hist, pdf = TestImageHistogram2d.fcn(image, 0.0, 1.0, 64, kernel="gaussian", return_pdf=return_pdf)

        assert hist.dtype == dtype
        assert pdf.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
    def test_integer_image_is_not_cast_into_an_integer_histogram(self, device, dtype):
        """An integer image has always given a float32 histogram (the bin centers are float and promote the result).
        Casting the KDE result back to the image dtype would truncate it and, for uint8, wrap it modulo 256."""
        torch.manual_seed(3)
        image = (torch.rand(1, 64, 64, device=device) * 255).to(dtype)

        hist, _ = TestImageHistogram2d.fcn(image, 0.0, 255.0, 16)
        reference, _ = TestImageHistogram2d.fcn(image.to(torch.float32), 0.0, 255.0, 16)

        assert hist.dtype == torch.float32
        assert hist.max() > 255, "the bin counts must exceed what uint8 can hold for this input to discriminate"
        self.assert_close(hist, reference, atol=0.0, rtol=0.0)

    def test_float64_image_keeps_float64_bin_centers(self, device):
        """A hard-coded float32 arange would give a float64 image float32-precision centers. With a bandwidth that is
        not a power of two (1 / 1000) the centers differ from the float64 ones by about 1e-8 and the histogram moves
        by ~1e-4, so this discriminates; a power-of-two n_bins would not."""
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        torch.manual_seed(0)
        n_bins = 1000
        image = torch.rand(1, 32, 32, device=device, dtype=torch.float64)

        hist_auto, _ = TestImageHistogram2d.fcn(image, 0.0, 1.0, n_bins, kernel="triangular")
        centers = 0.0 + (1.0 / n_bins) * (torch.arange(n_bins, device=device, dtype=torch.float64) + 0.5)
        hist_explicit, _ = TestImageHistogram2d.fcn(image, 0.0, 1.0, n_bins, centers=centers, kernel="triangular")

        assert hist_auto.dtype == torch.float64
        self.assert_close(hist_auto, hist_explicit, atol=1e-12, rtol=0.0)

    def test_explicit_centers_keep_the_promoted_dtype(self, device):
        """Only the auto-built-centers path is cast back to the image dtype. With explicit float64 centers a float32
        image has always given float64 outputs, and the caller who supplied them to get that precision must keep it."""
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        torch.manual_seed(2)
        image = torch.rand(1, 8, 8, device=device, dtype=torch.float32)
        centers = torch.linspace(0, 1, 16, device=device, dtype=torch.float64)

        hist, pdf = TestImageHistogram2d.fcn(image, 0.0, 1.0, 16, centers=centers, return_pdf=True)

        assert hist.dtype == torch.float64
        assert pdf.dtype == torch.float64


class TestHistogram2d(BaseTester):
    fcn = kornia.enhance.histogram2d

    def test_shape(self, device, dtype):
        inp1 = torch.ones(1, 16, device=device, dtype=dtype)
        inp2 = torch.ones(1, 16, device=device, dtype=dtype)
        bins = torch.linspace(0, 255, 32, device=device, dtype=dtype)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        pdf = TestHistogram2d.fcn(inp1, inp2, bins, bandwidth)
        assert pdf.shape == (1, 32, 32)

    def test_shape_batch(self, device, dtype):
        inp1 = torch.ones(4, 16, device=device, dtype=dtype)
        inp2 = torch.ones(4, 16, device=device, dtype=dtype)
        bins = torch.linspace(0, 255, 32, device=device, dtype=dtype)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        pdf = TestHistogram2d.fcn(inp1, inp2, bins, bandwidth)
        assert pdf.shape == (4, 32, 32)

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
        ans = 0.1 * kornia.core.ops.eye_like(10, pdf)
        self.assert_close(ans, pdf)


class TestHistogram(BaseTester):
    fcn = kornia.enhance.histogram

    def test_shape(self, device, dtype):
        inp = torch.ones(1, 16, device=device, dtype=dtype)
        bins = torch.linspace(0, 255, 32, device=device, dtype=dtype)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        pdf = TestHistogram.fcn(inp, bins, bandwidth)
        assert pdf.shape == (1, 32)

    def test_shape_batch(self, device, dtype):
        inp = torch.ones(4, 16, device=device, dtype=dtype)
        bins = torch.linspace(0, 255, 32, device=device, dtype=dtype)
        bandwidth = torch.tensor(0.9, device=device, dtype=dtype)
        pdf = TestHistogram.fcn(inp, bins, bandwidth)
        assert pdf.shape == (4, 32)

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

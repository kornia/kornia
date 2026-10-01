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

from kornia.filters.otsu_thresholding import OtsuThreshold, otsu_threshold

from testing.base import BaseTester, assert_close


class TestOtsuThreshold(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)
        op = OtsuThreshold()
        thresh_result, _thresh_value = op(img, nbins=4)
        assert thresh_result.shape == img.shape

    @pytest.mark.parametrize("input_shape", [(3, 3), (1, 3, 3), (1, 1, 3, 3), (2, 1, 1, 3, 3)])
    def test_transform_input_shapes(self, input_shape, device, dtype):
        img = torch.rand(input_shape, device=device, dtype=dtype)
        op = OtsuThreshold()
        flat, orig_shape = op.transform_input(img)
        assert orig_shape == img.shape
        assert flat.ndim == 2

    def test_otsu_threshold_consistency(self, device, dtype):
        torch.manual_seed(0)
        img = torch.rand(1, 4, 6, 1, device=device, dtype=dtype)
        out_func_tensor, _out_func_value = otsu_threshold(img, nbins=3, return_mask=False)
        out_class_tensor, _out_class_value = OtsuThreshold()(img, nbins=3)
        assert_close(out_func_tensor, out_class_tensor)

    def test_invalid_dim(self, device, dtype):
        img = torch.rand(1, 1, 1, 1, 3, 3, device=device, dtype=dtype)
        op = OtsuThreshold()
        with pytest.raises(ValueError, match="Unsupported tensor dimensionality"):
            op.transform_input(img)

    def test_gradcheck(self, device, dtype):
        img = torch.rand(1, 1, 5, 5, device=device, dtype=dtype, requires_grad=True)
        # The thresholded image only: the threshold's gradient is a straight-through surrogate that finite differences
        # cannot check (see TestOtsuThresholdDifferentiable).
        self.gradcheck(lambda img: otsu_threshold(img, 3, True, False)[0], (img,))

    def test_differentiable_tensor_otsu(self, device, dtype):
        differentiable_input = torch.rand(1, 1, 5, 5, device=device, dtype=dtype, requires_grad=True)

        input = differentiable_input.clone().detach().requires_grad_(False)

        op = OtsuThreshold()
        diff_thresh_result, _diff_thresh_value = op(input, slow_and_differentiable=True)
        thresh_result, _thresh_value = op(input)
        self.assert_close(diff_thresh_result, thresh_result)

    def test_threshold_result(self, device, dtype):
        input = torch.tensor(
            [[10, 10, 10, 10], [10, 10, 10, 10], [200, 200, 200, 200], [200, 200, 200, 200]], device=device, dtype=dtype
        )

        expected = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 0], [200, 200, 200, 200], [200, 200, 200, 200]], device=device, dtype=dtype
        )

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_gradual_threshold(self, device, dtype):
        input = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

        expected = torch.tensor([[0, 0, 0], [0, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_uniform_result(self, device, dtype):
        input = torch.tensor(
            [[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]], device=device, dtype=dtype
        )

        # a constant image has no split: its threshold is its value, so no pixel is above it
        expected = torch.zeros(4, 4, device=device, dtype=dtype)

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)


def test_mask(device, dtype):
    input = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

    expected = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1]], device=device, dtype=torch.bool)

    thresh_result, _thresh_value = otsu_threshold(input, return_mask=True)
    assert_close(thresh_result, expected)


@pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 1, 10, 10)])
def test_otsu_threshold_basic(shape, device, dtype):
    img = torch.rand(shape, device=device, dtype=dtype)
    thresh_result, _thresh_value = otsu_threshold(img)
    assert thresh_result.shape == img.shape


# Reference thresholds from scikit-image 0.26.0 and OpenCV 5.0.0, one image at a time:
#
#     import cv2
#     import numpy as np
#     from skimage.filters import threshold_otsu
#
#     u8 = np.concatenate([np.arange(256), np.full(300, 60), np.full(300, 190)]).astype(np.uint8).reshape(8, 107)
#     threshold_otsu(u8)  # 125
#     cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]  # 125.0
#     threshold_otsu(np.array([0.0, 0.1, 0.2, 0.55, 0.9, 1.0], np.float32), nbins=2)  # 0.25
#     threshold_otsu(np.linspace(0, 1, 60, dtype=np.float32) ** 2)  # 0.39257812
#     threshold_otsu(np.linspace(0.6, 4.3, 60, dtype=np.float32))  # 2.4138675
#     threshold_otsu(np.full((4, 5), 0.4, np.float32)), threshold_otsu(np.full((4, 5), -0.4, np.float32))  # 0.4, -0.4
#
# scikit-image returns the centre of the last bin below the split; kornia returns that bin's upper edge,
# min + (t + 1) * (max - min) / nbins, half a bin higher. For an integer image scikit-image bins the integer values
# themselves, and kornia's edge truncates to the same integer.
_SKIMAGE_UINT8 = 125
_OPENCV_UINT8 = 125.0
_SKIMAGE_NBINS_2 = 0.25
_SKIMAGE_RAMP_A = 0.39257812
_SKIMAGE_RAMP_B = 2.4138675


def _bimodal_uint8_image(device):
    # every value 0..255 once, plus 300 pixels at 60 and 300 at 190
    values = torch.cat([torch.arange(256), torch.full((300,), 60), torch.full((300,), 190)])
    return values.to(device=device, dtype=torch.uint8).view(1, 1, 8, 107)


def _skewed_ramps(device, dtype):
    a = (torch.linspace(0, 1, 60, dtype=torch.float64) ** 2).view(1, 1, 6, 10)
    b = torch.linspace(0.6, 4.3, 60, dtype=torch.float64).view(1, 1, 6, 10)
    return a.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


def _gaussian_mixture(device, dtype, bright_shift=0.0):
    # 700 dark and 300 bright pixels at the quantiles of N(0.3, 0.06) and N(0.7 + bright_shift, 0.1): overlapping
    # modes, so the threshold sits between them and follows them, unlike a threshold in an empty gap. The two modes
    # differ in size and spread, so a rule symmetric in them (the midrange, say) does not give Otsu's response.
    def quantiles(n):
        p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        return 2**0.5 * torch.erfinv(2 * p - 1)

    x = torch.cat([0.3 + 0.06 * quantiles(700), 0.7 + bright_shift + 0.1 * quantiles(300)])
    return x.view(1, 1, 20, 50).to(device=device, dtype=dtype)


@pytest.mark.parametrize("slow_and_differentiable", [False, True])
class TestOtsuThresholdBinEdges(BaseTester):
    def test_threshold_is_the_upper_edge_of_the_last_background_bin(self, slow_and_differentiable, device, dtype):
        # nbins=4 on [0, 1]: bins [0, .25), [.25, .5), [.5, .75), [.75, 1] hold 3, 1, 1, 3 pixels, and the split falls
        # after bin 1, so the threshold is its upper edge 0.5. The pixel 0.45, inside the last bin below the split, is
        # background; 0.55, inside the first bin above it, is foreground. A threshold read from
        # linspace(0, 1, nbins) would be 2/3 and drop 0.55.
        x = torch.tensor([[0.0, 0.05, 0.1, 0.45, 0.55, 0.9, 0.95, 1.0]], device=device, dtype=dtype)
        out, threshold = otsu_threshold(x, nbins=4, slow_and_differentiable=slow_and_differentiable)
        self.assert_close(threshold, torch.tensor([0.5], device=device, dtype=dtype), rtol=0.0, atol=0.0)
        assert (x > threshold).flatten().tolist() == [False] * 4 + [True] * 4
        self.assert_close(out.detach(), x * (x > threshold))

        # nbins=2 has one split, after bin 0: the threshold is 0.5, not the maximum, and keeps the 3 pixels that
        # scikit-image keeps (its threshold is the centre of bin 0, half a bin lower)
        x = torch.tensor([[0.0, 0.1, 0.2, 0.55, 0.9, 1.0]], device=device, dtype=dtype)
        _, threshold = otsu_threshold(x, nbins=2, slow_and_differentiable=slow_and_differentiable)
        self.assert_close(threshold, torch.tensor([_SKIMAGE_NBINS_2 + 0.25], device=device, dtype=dtype))
        assert (x > threshold).flatten().tolist() == (x > _SKIMAGE_NBINS_2).flatten().tolist()
        assert (x > threshold).flatten().tolist() == [False] * 3 + [True] * 3

    def test_uint8_threshold_matches_scikit_image_and_opencv(self, slow_and_differentiable, device, dtype):
        img = _bimodal_uint8_image(device)
        out, threshold = otsu_threshold(img, slow_and_differentiable=slow_and_differentiable)
        assert threshold.dtype == torch.uint8
        assert threshold.item() == _SKIMAGE_UINT8 == _OPENCV_UINT8
        flat = out.flatten()
        assert flat[125].item() == 0  # the pixel equal to the threshold is background
        assert flat[126].item() == 126  # the next value is foreground
        assert torch.equal(out > 0, img > _SKIMAGE_UINT8)

        # The same image in a float dtype: the threshold is the bin edge 126 * 255 / 256 = 125.508, in input units, and
        # the foreground is the same
        out, threshold = otsu_threshold(img.to(dtype), slow_and_differentiable=slow_and_differentiable)
        self.assert_close(threshold, torch.tensor([126 * 255 / 256], device=device, dtype=dtype))
        assert torch.equal(out > 0, img > _SKIMAGE_UINT8)

    def test_each_plane_is_thresholded_on_its_own_range(self, slow_and_differentiable, device, dtype):
        a, b = _skewed_ramps(device, dtype)
        out_a, threshold_a = otsu_threshold(a, slow_and_differentiable=slow_and_differentiable)
        out_b, threshold_b = otsu_threshold(b, slow_and_differentiable=slow_and_differentiable)
        # a batched with b, and a and b as the two channels of one image: neither changes a's or b's result
        for joint in (torch.cat([a, b]), torch.cat([a, b], dim=1)):
            out, thresholds = otsu_threshold(joint, slow_and_differentiable=slow_and_differentiable)
            assert torch.equal(thresholds, torch.cat([threshold_a, threshold_b]))
            assert torch.equal(out.reshape(2, 1, 6, 10), torch.cat([out_a, out_b]))

        if not slow_and_differentiable:
            # Half a bin above scikit-image's bin centre, on each ramp's own range. Each ramp has one pixel inside the
            # last bin below the split, above that bin's centre: scikit-image keeps it (23 and 31 pixels), kornia does
            # not.
            self.assert_close(threshold_a, torch.tensor([_SKIMAGE_RAMP_A + 1 / 512], device=device, dtype=dtype))
            self.assert_close(threshold_b, torch.tensor([_SKIMAGE_RAMP_B + 3.7 / 512], device=device, dtype=dtype))
            assert int((a > threshold_a).sum()) == 22
            assert int((b > threshold_b).sum()) == 30

    def test_constant_plane_returns_its_value(self, slow_and_differentiable, device, dtype):
        # a constant plane has no split: its threshold is its value, so none of its pixels is kept, whatever its sign;
        # its batch mate keeps its own threshold
        a, _ = _skewed_ramps(device, dtype)
        _, threshold_a = otsu_threshold(a, slow_and_differentiable=slow_and_differentiable)
        for value in (0.4, -0.4):
            plane = torch.full((1, 1, 6, 10), value, device=device, dtype=dtype)
            out, thresholds = otsu_threshold(torch.cat([plane, a]), slow_and_differentiable=slow_and_differentiable)
            self.assert_close(thresholds, torch.cat([plane[0, 0, 0, :1], threshold_a]), rtol=0.0, atol=0.0)
            assert out[0].count_nonzero().item() == 0
            if slow_and_differentiable:
                # the plane's threshold is its minimum, whose gradient is shared by its 60 tied pixels, and the empty
                # classes of its histogram put no nan into the batch's gradient
                plane = plane.clone().requires_grad_(True)
                _, thresholds = otsu_threshold(torch.cat([plane, a]), slow_and_differentiable=True)
                (grad,) = torch.autograd.grad(thresholds[0], plane)
                self.assert_close(grad, torch.full_like(grad, 1 / 60))

        plane = torch.full((4, 5), 7, device=device, dtype=torch.uint8)
        out, threshold = otsu_threshold(plane, slow_and_differentiable=slow_and_differentiable)
        assert threshold.tolist() == [7]
        assert out.count_nonzero().item() == 0


class TestOtsuThresholdDifferentiable(BaseTester):
    # No gradcheck on the threshold: its value is the hard split, which jumps between bin edges, while its gradient is
    # that of a soft-argmax over the between-class variance curve (straight-through), so finite differences of the
    # threshold do not match it by design. These tests check that the gradient exists, is finite and points the right
    # way.
    def test_threshold_has_a_gradient(self, device, dtype):
        x = _gaussian_mixture(device, dtype).requires_grad_(True)
        out, threshold = otsu_threshold(x, slow_and_differentiable=True)
        assert threshold.requires_grad
        (grad,) = torch.autograd.grad(threshold.sum(), x)
        assert grad.isfinite().all()
        assert grad.count_nonzero() > 0
        # moving every pixel by c moves the threshold by c, so the gradient sums to 1
        self.assert_close(grad.float().sum(), torch.tensor(1.0, device=device), rtol=0.0, atol=1e-3)

        # the thresholded image is x * (x > threshold) as before: its gradient is the mask
        (grad,) = torch.autograd.grad(out.sum(), x)
        self.assert_close(grad, (x.detach() > threshold.detach()).to(dtype))

        # the default path's threshold has no gradient
        _, threshold = otsu_threshold(x)
        assert not threshold.requires_grad

    def test_threshold_gradient_follows_the_bright_mode(self, device, dtype):
        x = _gaussian_mixture(device, dtype).requires_grad_(True)
        _, threshold = otsu_threshold(x, slow_and_differentiable=True)
        (grad,) = torch.autograd.grad(threshold.sum(), x)
        _, up = otsu_threshold(_gaussian_mixture(device, dtype, bright_shift=0.05), slow_and_differentiable=True)
        _, down = otsu_threshold(_gaussian_mixture(device, dtype, bright_shift=-0.05), slow_and_differentiable=True)
        # shifting the bright mode moves the threshold the same way ...
        assert up.item() > threshold.item() > down.item()
        # ... and the gradient along that shift, the sum over the bright pixels, agrees in sign and is close to the
        # threshold's actual response, (up - down) / 0.1 = 0.45 (0.47 in bfloat16); the gradient is 0.39 (0.55)
        along_shift = grad.flatten()[700:].float().sum().item()
        response = (up.item() - down.item()) / 0.1
        assert along_shift > 0
        assert abs(along_shift - response) < 0.1
        assert grad.flatten()[:700].float().sum().item() > 0

    def test_slow_and_fast_thresholds_agree_within_one_bin(self, device, dtype):
        # The slow histogram is the mass a Gaussian kernel density estimate of bandwidth 0.1 bin puts in each bin. It
        # differs from histc's only for pixels within a fraction of a bin of an edge, which leave part of their mass in
        # the neighbouring bin; on the images of the two bug reports below the best split moves up by at most one bin.
        # Pinned with the rounding of both thresholds to the dtype on top. Not a property of every image: two near-tied
        # splits far apart can swap under any change of histogram estimator.
        generator = torch.Generator().manual_seed(0)
        noise = torch.rand(1000, generator=generator, dtype=torch.float64)
        a, b = _skewed_ramps("cpu", torch.float64)
        cases = [
            ("bimodal", torch.cat([0.3 + 0.1 * noise[:500], 0.6 + 0.1 * noise[500:]]).view(1, 1, 20, 50), 256),
            ("mass at 0.3", torch.tensor([0.0] * 10 + [0.3] * 80 + [1.0] * 10).view(1, 1, 1, -1), 16),
            ("mass at 0.7", torch.tensor([0.0] * 10 + [0.7] * 80 + [1.0] * 10).view(1, 1, 1, -1), 16),
            ("uint8 values", _bimodal_uint8_image("cpu").double(), 256),
            ("nbins=2", torch.tensor([[0.0, 0.1, 0.2, 0.55, 0.9, 1.0]]), 2),
            ("ramp a", a, 256),
            ("ramp b", b, 256),
        ]
        thresholds = {}
        for name, x, nbins in cases:
            if name == "ramp a" and dtype == torch.bfloat16:
                # bfloat16 rounds one pixel of this ramp onto the bin edge 106 / 256: histc counts it in bin 106, the
                # kernel splits it between bins 105 and 106, and the splits on its two sides are within 0.003 % of each
                # other in between-class variance, so the slow path takes the other one, 6 bins up
                continue
            x = x.to(device=device, dtype=dtype)
            _, fast = otsu_threshold(x, nbins=nbins)
            _, slow = otsu_threshold(x, nbins=nbins, slow_and_differentiable=True)
            bin_width = (x.max().item() - x.min().item()) / nbins
            rounding = torch.finfo(dtype).eps * max(abs(fast.item()), abs(slow.item()))
            assert -rounding <= slow.item() - fast.item() <= bin_width + rounding, name
            thresholds[name] = slow.item()
        # the kernel density estimate sees the 80 pixels between its old sample points: the two images differ
        assert thresholds["mass at 0.3"] != thresholds["mass at 0.7"]

    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    def test_dynamo(self, slow_and_differentiable, device, dtype, torch_optimizer):
        x = _gaussian_mixture(device, dtype)
        # the slow path has no .item() and compiles as one graph; the default path passes each plane's minimum and
        # maximum to torch.histc through .item(), which breaks the graph before torch 2.14
        op = torch_optimizer(
            lambda x: otsu_threshold(x, nbins=64, slow_and_differentiable=slow_and_differentiable),
            fullgraph=slow_and_differentiable,
        )
        expected = otsu_threshold(x, nbins=64, slow_and_differentiable=slow_and_differentiable)
        actual = op(x)
        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1])

        if slow_and_differentiable:
            x = x.clone().requires_grad_(True)
            (expected_grad,) = torch.autograd.grad(otsu_threshold(x, 64, True)[1].sum(), x)
            (actual_grad,) = torch.autograd.grad(op(x)[1].sum(), x)
            self.assert_close(actual_grad, expected_grad)

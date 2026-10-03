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

    @staticmethod
    def _separated_image(device, dtype):
        """A fixed 5x5 image whose ``nbins=3`` Otsu threshold is 0.5 and keeps 14 of 25 pixels.

        Every pixel is at least 1.9e-3 from the threshold in each tested dtype (the smallest margin is 1.95e-3, in
        float16), so a finite-difference step cannot move a pixel across it. The values are fixed rather than drawn: a
        random draw often puts the threshold at the data maximum (empty output) or at a near-tie between two splits,
        and gradcheck then fails.
        """
        values = [
            [0.00, 0.00, 0.00, 0.502, 1.00],
            [1.00, 0.08, 0.13, 0.21, 0.27],
            [0.33, 0.38, 0.44, 0.47, 0.56],
            [0.61, 0.66, 0.71, 0.77, 0.83],
            [0.88, 0.91, 0.94, 0.96, 0.98],
        ]
        return torch.tensor(values, device=device, dtype=dtype).view(1, 1, 5, 5)

    def test_gradcheck(self, device, dtype):
        img = self._separated_image(device, dtype)
        # gradcheck evaluates the op in float64 on the (possibly half-rounded) fixture, so check that input.
        img64 = img.to(torch.float64)
        out, threshold = otsu_threshold(img64, 3, True)
        # Guard against a vacuous pass: pixels must lie on both sides of the threshold, and none within 1e-3 of it,
        # which is far above the finite-difference step.
        assert (out > 0).any()
        assert not (img64 > threshold).all()
        assert (img64 - threshold).abs().min() > 1e-3
        # The analytical gradient of the output is just the mask (no gradient flows through the threshold), so
        # this only checks that no pixel crosses the threshold under the finite-difference perturbation.
        self.gradcheck(otsu_threshold, (img, 3, True, False))

    def test_differentiable_tensor_otsu(self, device, dtype):
        img = self._separated_image(device, dtype).requires_grad_(True)

        op = OtsuThreshold()
        result, threshold = op(img, slow_and_differentiable=True)
        mask = (img.detach() > threshold).to(dtype)
        # A non-degenerate fixture: some pixels are kept and some are dropped.
        assert mask.any()
        assert not mask.all()

        # The slow path returns the pixels above the threshold and zeros elsewhere.
        self.assert_close(result.detach(), img.detach() * mask)

        result.sum().backward()
        # The thresholded image is ``mask * x`` with a constant mask, so its gradient is the mask.
        assert img.grad is not None
        self.assert_close(img.grad, mask)

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

        expected = torch.zeros_like(input)

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_large_image_threshold_matches_float32(self, device, dtype):
        # #5196: the histogram counts of a 300 x 300 image (90000 pixels) came back in the input dtype, so in float16
        # their sum overflowed to inf, the normalised histogram was all zeros and the threshold was 0; bfloat16 counts
        # were rounded to 8 bits. The pixel levels k / 256 are exact in every dtype, so the float32 call sees the same
        # pixels. For float16, bfloat16 and float32 the histogram arithmetic is then the same float32 computation, and
        # the two thresholds agree up to rounding the float32 one to `dtype`. float64 bins and sums in float64, and
        # the two best splits of this image differ by only about 50 float32 eps, so it is allowed one bin (1 / 256).
        levels = torch.randint(0, 256, (1, 1, 300, 300), generator=torch.Generator().manual_seed(0))
        img = (levels / 256).to(device=device, dtype=dtype)

        _, threshold = otsu_threshold(img)
        _, expected = otsu_threshold(img.float())

        assert 0.25 < expected.item() < 0.75
        atol = 1 / 256 if dtype == torch.float64 else 0.0
        self.assert_close(threshold, expected.to(dtype), rtol=0.0, atol=atol)

    def test_two_bins_uses_histogram_split_5172(self, device, dtype):
        image = torch.tensor([[0.0, 0.1, 0.2, 0.55, 0.9, 1.0]], device=device, dtype=dtype)
        mask, threshold = otsu_threshold(image, nbins=2, return_mask=True)
        # The two histc bins split [0, 1] at 0.5. The previous nbins-point linspace returned 1 instead.
        self.assert_close(threshold, image.new_tensor([0.5]), rtol=0, atol=0)
        assert mask.tolist() == [[False, False, False, True, True, True]]

    def test_uint8_pixel_above_selected_bin_5172(self, device):
        image = torch.cat([torch.arange(256), torch.full((300,), 60), torch.full((300,), 190)])
        image = image.to(device=device, dtype=torch.uint8).view(1, 1, 8, 107)
        mask, threshold = otsu_threshold(image, return_mask=True)
        assert threshold.item() == 125
        assert mask.flatten()[126]

    @pytest.mark.parametrize("shape", [(2, 6, 10), (2, 1, 6, 10), (1, 2, 6, 10), (1, 2, 1, 6, 10)])
    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    def test_planes_use_independent_ranges_5172(self, shape, slow_and_differentiable, device, dtype):
        first = torch.linspace(0, 1, 60, device=device, dtype=dtype).square().view(6, 10)
        second = torch.linspace(0.6, 4.3, 60, device=device, dtype=dtype).view(6, 10)
        image = torch.stack([first, second]).reshape(shape)
        expected = [otsu_threshold(plane, slow_and_differentiable=slow_and_differentiable) for plane in [first, second]]
        result, threshold = otsu_threshold(image, slow_and_differentiable=slow_and_differentiable)
        self.assert_close(threshold, torch.cat([item[1] for item in expected]), rtol=0, atol=0)
        self.assert_close(result, torch.stack([item[0] for item in expected]).reshape(shape), rtol=0, atol=0)

    @pytest.mark.parametrize("value", [-0.4, 0.0, 0.4])
    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    def test_constant_threshold_is_plane_value_5172(self, value, slow_and_differentiable, device, dtype):
        image = torch.full((1, 1, 4, 5), value, device=device, dtype=dtype)
        mask, threshold = otsu_threshold(image, slow_and_differentiable=slow_and_differentiable, return_mask=True)
        self.assert_close(threshold, image.flatten()[:1], rtol=0, atol=0)
        assert not mask.any()

    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    @pytest.mark.parametrize(
        "input_dtype,value", [(torch.int32, 2**24 + 1), (torch.int64, 2**24 + 1), (torch.int64, 2**53 + 1)]
    )
    @pytest.mark.parametrize("sign", [-1, 1])
    def test_constant_integer_preserves_exact_value_5172(
        self, input_dtype, value, sign, slow_and_differentiable, device
    ):
        image = torch.full((2, 3), sign * value, dtype=input_dtype, device=device)
        mask, threshold = otsu_threshold(image, slow_and_differentiable=slow_and_differentiable, return_mask=True)
        assert torch.equal(threshold, image.flatten()[:1])
        assert not mask.any()


def test_mask(device, dtype):
    input = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

    expected = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1]], device=device, dtype=torch.bool)

    thresh_result, _thresh_value = otsu_threshold(input, return_mask=True)
    assert_close(thresh_result, expected)


@pytest.mark.parametrize("slow_and_differentiable", [False, True])
@pytest.mark.parametrize(
    "values",
    [(-1.0, 0.0), (-2.0, -1.0), (0.0, 1.0), (1.0, 2.0)],
)
def test_mask_is_the_comparison_5173(values, slow_and_differentiable, device, dtype):
    # #5173: the mask used to be `result > 0`, so a foreground pixel of value 0 or below, above the threshold but not
    # above 0, came out False. The mask is `x > threshold`, whatever the sign of the data.
    low, high = values
    x = torch.tensor([[low, low, high, high]], device=device, dtype=dtype)
    mask, threshold = otsu_threshold(x, slow_and_differentiable=slow_and_differentiable, return_mask=True)
    assert mask.dtype == torch.bool
    assert_close(mask, x > threshold)
    assert mask.tolist() == [[False, False, True, True]]


def test_mask_per_channel_threshold_5173(device, dtype):
    # Each channel of a (B, C, H, W) input has its own threshold; the mask follows its channel's comparison and keeps
    # the input shape. Channel 0 is non-positive data, channel 1 positive.
    x = torch.tensor([[[[-1.0, -1.0], [0.0, 0.0]], [[1.0, 1.0], [2.0, 2.0]]]], device=device, dtype=dtype)
    mask, threshold = otsu_threshold(x, return_mask=True)
    assert mask.shape == x.shape
    assert threshold.shape == (2,)
    assert_close(mask, x > threshold.reshape(1, 2, 1, 1))
    assert mask[0, 0].tolist() == [[False, False], [True, True]]
    assert mask[0, 1].tolist() == [[False, False], [True, True]]


@pytest.mark.parametrize("slow_and_differentiable", [False, True])
def test_mask_agrees_with_the_thresholded_image_5173(slow_and_differentiable, device):
    # Integer input: the threshold is truncated to the input dtype, so on the fast path one pixel (0) equals it, and a
    # `>=` in the mask would keep that pixel while the thresholded image drops it.
    x = torch.tensor([[-30, -20, -10], [0, 10, 20], [30, 40, 50]], device=device)
    image, threshold = otsu_threshold(x, slow_and_differentiable=slow_and_differentiable)
    mask, _ = otsu_threshold(x, slow_and_differentiable=slow_and_differentiable, return_mask=True)
    assert torch.equal(mask, x > threshold)
    assert torch.equal(image, torch.where(mask, x, torch.zeros_like(x)))
    assert slow_and_differentiable or (x == threshold).any()  # the fixture does contain a pixel equal to the threshold


@pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 1, 10, 10)])
def test_otsu_threshold_basic(shape, device, dtype):
    img = torch.rand(shape, device=device, dtype=dtype)
    thresh_result, _thresh_value = otsu_threshold(img)
    assert thresh_result.shape == img.shape

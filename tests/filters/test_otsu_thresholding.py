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

    @pytest.mark.parametrize(
        "input_dtype,sign,offset,nbins,expected,kept,dropped",
        [
            (torch.int16, 1, -128, 256, -3, 126, 125),
            (torch.int16, -1, 0, 256, -126, 125, 126),
            (torch.uint8, 1, 0, 255, 125, 126, 125),
        ],
        ids=["shifted_below_zero", "mirrored", "edge_on_an_integer"],
    )
    def test_integer_threshold_is_the_largest_integer_below_the_edge_5172(
        self, input_dtype, sign, offset, nbins, expected, kept, dropped, device
    ):
        # The image above, shifted below zero, mirrored, or with 255 bins, whose selected upper edge is 126 exactly.
        # Truncating the edge toward zero rounds a negative edge up and keeps an integer edge, so the level just
        # above the split (scikit-image and OpenCV keep it) was dropped from the foreground.
        values = torch.cat([torch.arange(256), torch.full((300,), 60), torch.full((300,), 190)])
        image = (sign * values + offset).to(device=device, dtype=input_dtype).view(1, 1, 8, 107)
        mask, threshold = otsu_threshold(image, nbins=nbins, return_mask=True)
        assert threshold.item() == expected
        assert mask.flatten()[kept]
        assert not mask.flatten()[dropped]

    def test_float64_threshold_uses_float64_edges_5172(self, device):
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        # The single split of two bins is the midpoint 0.5 + 2**-31, which float32 rounds to 0.5.
        image = torch.tensor([[0.0, 0.1, 0.2, 0.55, 0.9, 1.0 + 2**-30]], device=device, dtype=torch.float64)
        _, threshold = otsu_threshold(image, nbins=2)
        assert threshold.item() == 0.5 + 2**-31

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


def _uint8_bimodal(device):
    # every value 0..255 once, plus 300 pixels at 60 and 300 at 190
    values = torch.cat([torch.arange(256), torch.full((300,), 60), torch.full((300,), 190)])
    return values.to(device=device, dtype=torch.uint8).view(1, 1, 8, 107)


class TestConventionsOtsuThreshold(BaseTester):
    def test_convention_otsu_threshold_per_plane_in_input_units_foreground_strictly_above(self, device, dtype):
        # two well-separated clusters per plane, in input units (not [0, 1]); B = 2, C = 3, H != W
        values = torch.tensor([20.0, 30.0, 40.0, 160.0, 170.0, 180.0])
        plane = torch.stack([values.roll(i) for i in range(4)]).view(1, 1, 4, 6)
        img = torch.cat([plane, plane.flip(-1) + 5, plane * 0.5], 1)
        img = torch.cat([img, img + 7]).to(device=device, dtype=dtype).requires_grad_(True)
        out, thresholds = otsu_threshold(img)
        # one threshold per (b, c) plane, returned flat, in the input dtype
        assert thresholds.shape == (6,)
        assert thresholds.dtype == out.dtype == dtype
        ordered = img.detach().view(6, -1).sort(dim=1).values  # 12 low and 12 high pixels per plane
        low_max, high_min = ordered[:, 11], ordered[:, 12]
        # between the clusters (a threshold equal to the low cluster's top still drops it), at the lowest tied split
        assert (thresholds >= low_max).all()
        assert (thresholds < low_max + (high_min - low_max) / 4).all()
        # the first output is x * (x > threshold), and its gradient is that mask
        mask = img.detach() > thresholds.view(2, 3, 1, 1)
        self.assert_close(out, img.detach() * mask)
        (grad,) = torch.autograd.grad(out.sum(), img)
        self.assert_close(grad, mask.to(dtype))
        # relabel: transposing the image leaves the thresholds and transposes the output
        out_t, thresholds_t = otsu_threshold(img.detach().transpose(-1, -2))
        self.assert_close(thresholds_t, thresholds)
        self.assert_close(out_t, out.detach().transpose(-1, -2))
        # strictly above: on 0..255 the pixel equal to the threshold is dropped, the next value is kept
        u8 = _uint8_bimodal(device)
        out8, t8 = otsu_threshold(u8)
        assert t8.dtype == torch.uint8
        level = int(t8.item())
        assert out8.flatten()[level].item() == 0
        assert out8.flatten()[level + 1].item() == level + 1

    def test_wart_otsu_threshold_one_bin_above_its_split_5172(self, device, dtype):
        """#5172: the threshold is read from linspace(min, max, nbins), not from the histc bin edges."""
        # nbins=2 has one split, and the returned threshold is the data maximum, so nothing is kept
        x = torch.tensor([[0.0, 0.1, 0.2, 0.55, 0.9, 1.0]], device=device, dtype=dtype)
        out, threshold = otsu_threshold(x, nbins=2)
        assert threshold.item() == x.max().item()
        assert out.count_nonzero().item() == 0
        # uint8: skimage.filters.threshold_otsu and cv2.THRESH_OTSU give 125 on this image
        _, t8 = otsu_threshold(_uint8_bimodal(device))
        assert t8.item() == 126

    def test_wart_otsu_threshold_depends_on_batch_mates_5172(self, device, dtype):
        """#5172: one histogram range, [min, max] of the whole call, is shared by every image and channel."""
        a = (torch.linspace(0, 1, 60) ** 2).view(1, 1, 6, 10).to(device=device, dtype=dtype)
        b = torch.linspace(0.6, 4.3, 60).view(1, 1, 6, 10).to(device=device, dtype=dtype)
        _, alone = otsu_threshold(a)
        _, batched = otsu_threshold(torch.cat([a, b]))
        _, channels = otsu_threshold(torch.cat([a, b], 1))
        assert batched[0] != alone[0]
        assert channels[0] != alone[0]

    def test_wart_otsu_constant_image_threshold_is_zero_5172(self, device, dtype):
        """#5172: a constant plane has no split and gets threshold 0 in input units."""
        for value, kept in ((0.4, 20), (-0.4, 0)):
            img = torch.full((1, 1, 4, 5), value, device=device, dtype=dtype)
            out, threshold = otsu_threshold(img)
            assert threshold.item() == 0
            assert out.count_nonzero().item() == kept

    def test_wart_otsu_return_mask_drops_nonpositive_foreground_5173(self, device, dtype):
        """#5173: the mask is computed as result > 0, so foreground pixels <= 0 are reported as background."""
        for values in ([-1.0, -1.0, 0.0, 0.0], [-2.0, -2.0, -1.0, -1.0]):
            x = torch.tensor([values], device=device, dtype=dtype)
            mask, threshold = otsu_threshold(x, return_mask=True)
            assert (x > threshold).flatten().tolist() == [False, False, True, True]
            assert not mask.any()

    def test_wart_otsu_slow_path_threshold_has_no_gradient_and_kde_skips_pixels_5174(self, device, dtype):
        """#5174: slow_and_differentiable gives no threshold gradient, and its 1e-3 KDE skips most pixels."""
        generator = torch.Generator().manual_seed(0)
        noise = torch.rand(1000, generator=generator)
        x = torch.cat([0.3 + 0.1 * noise[:500], 0.6 + 0.1 * noise[500:]]).view(1, 1, 20, 50)
        x = x.to(device=device, dtype=dtype).requires_grad_(True)
        _, threshold = otsu_threshold(x, slow_and_differentiable=True)
        assert not threshold.requires_grad
        # the KDE is evaluated at linspace(0, 1, 16) with bandwidth 1e-3: a mass at 0.3 or at 0.7 lies between those
        # points, so the slow path gives both images the same threshold while the fast path separates them
        fast, slow = [], []
        for mid in (0.3, 0.7):
            img = torch.tensor([0.0] * 10 + [mid] * 80 + [1.0] * 10, device=device, dtype=dtype).view(1, 1, 1, -1)
            fast.append(otsu_threshold(img, nbins=16)[1])
            slow.append(otsu_threshold(img, nbins=16, slow_and_differentiable=True)[1])
        assert fast[0] != fast[1]
        assert slow[0] == slow[1]

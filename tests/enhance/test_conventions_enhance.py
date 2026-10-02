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
from kornia.core.exceptions import BaseError, ShapeError

from testing.base import BaseTester


class TestEnhanceConventions(BaseTester):
    def test_convention_add_weighted_scalar_tensor_coefficient_5325(self, device, dtype):
        src = torch.arange(6, device=device, dtype=dtype).reshape(2, 3)
        alpha = torch.tensor(0.5, device=device, dtype=dtype)
        self.assert_close(
            kornia.enhance.add_weighted(src, alpha, src, 1.0, 0.0), kornia.enhance.add_weighted(src, 0.5, src, 1.0, 0.0)
        )

    def test_convention_normalize_channel_axis_is_one(self, device, dtype):
        data = torch.tensor([[[1.0], [4.0]], [[2.0], [8.0]]], device=device, dtype=dtype)
        mean = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        std = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        expected = torch.tensor([[[0.0], [1.0]], [[1.0], [3.0]]], device=device, dtype=dtype)
        self.assert_close(kornia.enhance.normalize(data, mean, std), expected)

    def test_convention_normalize_accepts_per_batch_statistics(self, device, dtype):
        data = torch.tensor([[[1.0], [4.0]], [[3.0], [8.0]]], device=device, dtype=dtype)
        mean = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device=device, dtype=dtype)
        std = torch.tensor([[1.0, 2.0], [1.0, 2.0]], device=device, dtype=dtype)
        expected = torch.tensor([[[0.0], [1.0]], [[1.0], [2.0]]], device=device, dtype=dtype)
        self.assert_close(kornia.enhance.normalize(data, mean, std), expected)

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 4),
            (2, 4, 3, 2, 2),
        ],
    )
    def test_convention_denormalize_is_inverse_for_channel_vector_5318(self, shape, device, dtype):
        data = torch.arange(torch.tensor(shape).prod(), device=device, dtype=dtype).reshape(shape) / 32.0
        mean = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        std = torch.tensor([0.5, 0.6, 0.7, 0.8], device=device, dtype=dtype)
        normalized = kornia.enhance.normalize(data, mean, std)
        self.assert_close(kornia.enhance.denormalize(normalized, mean, std), data)
        self.assert_close(kornia.enhance.Denormalize(mean, std)(normalized), data)

    def test_convention_denormalize_rank5_batch_channel_statistics_workaround(self, device, dtype):
        data = torch.arange(2 * 4 * 3 * 2 * 2, device=device, dtype=dtype).reshape(2, 4, 3, 2, 2) / 32.0
        mean = torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=device, dtype=dtype)
        std = torch.tensor([[0.5, 0.6, 0.7, 0.8]], device=device, dtype=dtype)
        normalized = kornia.enhance.normalize(data, mean, std)
        self.assert_close(kornia.enhance.denormalize(normalized, mean, std), data)

    def test_convention_image_histogram_triangular_range(self, device, dtype):
        # Centers are 0.25 and 0.75; 1.1 lies outside [0, 1] and keeps its triangular weight
        # 1 - 0.35 / 0.5 = 0.3 for the 0.75 bin instead of being clipped to 1.0 (weight 0.5) or dropped.
        image = torch.tensor([[0.25, 1.1]], device=device, dtype=dtype)
        hist, pdf = kornia.enhance.image_histogram2d(
            image, min=0.0, max=1.0, n_bins=2, bandwidth=0.5, kernel="triangular", return_pdf=True
        )
        self.assert_close(hist, torch.tensor([1.0, 0.3], device=device, dtype=dtype))
        self.assert_close(pdf, torch.tensor([1.0, 0.3], device=device, dtype=dtype) / 1.3)

    def test_convention_image_histogram_automatic_centers_follow_bandwidth(self, device, dtype):
        image = torch.tensor([[0.125]], device=device, dtype=dtype)
        hist, _ = kornia.enhance.image_histogram2d(
            image, min=0.0, max=1.0, n_bins=2, bandwidth=0.25, kernel="triangular"
        )
        # Centers are 0.125 and 0.375, generated as min + (i + 0.5) * bandwidth.
        self.assert_close(hist, torch.tensor([1.0, 0.0], device=device, dtype=dtype))

    @pytest.mark.parametrize("explicit_center", [False, True])
    @pytest.mark.parametrize("return_pdf", [False, True])
    def test_convention_image_histogram_rank2_single_bin_preserves_bin_axis_5363(
        self, device, dtype, explicit_center, return_pdf
    ):
        image = torch.tensor([[0.25, 0.75]], device=device, dtype=dtype)
        centers = torch.tensor([0.5], device=device, dtype=dtype) if explicit_center else None

        hist, pdf = kornia.enhance.image_histogram2d(
            image, min=0.0, max=1.0, n_bins=1, centers=centers, return_pdf=return_pdf
        )

        assert hist.shape == (1,)
        assert pdf.shape == (1,)

    def test_convention_jpeg_quality_is_one_dimensional(self, device, dtype):
        image = torch.zeros(1, 3, 16, 17, device=device, dtype=dtype)
        with pytest.raises(ShapeError):
            kornia.enhance.jpeg_codec_differentiable(image, torch.tensor(50.0, device=device, dtype=dtype))
        quality = torch.tensor([50.0], device=device, dtype=dtype)
        assert kornia.enhance.jpeg_codec_differentiable(image[0], quality).shape == (3, 16, 17)

    def test_convention_brightness_is_additive_and_optionally_clipped(self, device, dtype):
        image = torch.tensor([[[[0.25, 0.75]]]], device=device, dtype=dtype)
        self.assert_close(kornia.enhance.adjust_brightness(image, 0.5, clip_output=False), image + 0.5)
        self.assert_close(
            kornia.enhance.adjust_brightness(image, 0.5), torch.tensor([[[[0.75, 1.0]]]], device=device, dtype=dtype)
        )

    def test_convention_brightness_accumulative_is_multiplicative(self, device, dtype):
        image = torch.tensor([[[[0.25, 0.75]]]], device=device, dtype=dtype)
        self.assert_close(kornia.enhance.adjust_brightness_accumulative(image, 0.5, clip_output=False), image * 0.5)

    def test_convention_gamma_applies_gain_then_clamps(self, device, dtype):
        # gain * x ** gamma = 2 * (0.0625, 0.5625) = (0.125, 1.125), then clamped to 1.
        image = torch.tensor([[[[0.25, 0.75]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[0.125, 1.0]]]], device=device, dtype=dtype)
        self.assert_close(kornia.enhance.adjust_gamma(image, gamma=2.0, gain=2.0), expected)

    def test_convention_hue_raw_wraps_and_preserves_saturation_value(self, device, dtype):
        hsv = torch.tensor([[[[6.0]], [[0.25]], [[0.75]]]], device=device, dtype=dtype)
        result = kornia.enhance.adjust_hue_raw(hsv, 1.0)
        expected_hue = torch.fmod(torch.tensor(7.0, device=device, dtype=dtype), 2 * torch.pi)
        self.assert_close(result[0, 0, 0, 0], expected_hue)
        self.assert_close(result[:, 1:], hsv[:, 1:])

    def test_convention_hue_raw_wraps_negative_sum_5326(self, device, dtype):
        hsv = torch.tensor([[[[0.2]], [[0.25]], [[0.75]]]], device=device, dtype=dtype)
        result = kornia.enhance.adjust_hue_raw(hsv, -1.0)
        self.assert_close(result[0, 0, 0, 0], torch.tensor(2 * torch.pi - 0.8, device=device, dtype=dtype))

    def test_convention_saturation_raw_only_clamps_saturation(self, device, dtype):
        # Hue 4.0 is outside [0, 1], so clamping every channel would change it.
        hsv = torch.tensor([[[[4.0]], [[0.8]], [[0.6]]]], device=device, dtype=dtype)
        result = kornia.enhance.adjust_saturation_raw(hsv, 2.0)
        expected = torch.tensor([[[[4.0]], [[1.0]], [[0.6]]]], device=device, dtype=dtype)
        self.assert_close(result, expected)

    def test_convention_shift_rgb_uses_per_batch_rgb_shifts_and_clamps(self, device, dtype):
        image = torch.tensor([[[[0.1]], [[0.2]], [[0.3]]], [[[0.8]], [[0.7]], [[0.6]]]], device=device, dtype=dtype)
        result = kornia.enhance.shift_rgb(
            image,
            torch.tensor([0.2, -0.9], device=device, dtype=dtype),
            torch.tensor([-0.3, 0.4], device=device, dtype=dtype),
            torch.tensor([0.8, -0.2], device=device, dtype=dtype),
        )
        # RGB shifts are (B, 3, 1, 1); expected values include the [0, 1] clamp.
        expected = torch.tensor([[[[0.3]], [[0.0]], [[1.0]]], [[[0.0]], [[1.0]], [[0.4]]]], device=device, dtype=dtype)
        self.assert_close(result, expected)

    def test_convention_integral_uses_requested_axes(self, device, dtype):
        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]], device=device, dtype=dtype)
        self.assert_close(kornia.enhance.integral_tensor(data, (0,)), expected)

    def test_convention_threshold_modes_use_strict_comparison(self, device, dtype):
        data = torch.tensor([[0.5, 0.6]], device=device, dtype=dtype)
        expected = {
            kornia.enhance.ThresholdType.THRESH_BINARY: [0.0, 2.0],
            kornia.enhance.ThresholdType.THRESH_BINARY_INV: [2.0, 0.0],
            kornia.enhance.ThresholdType.THRESH_TRUNC: [0.5, 0.5],
            kornia.enhance.ThresholdType.THRESH_TOZERO: [0.0, 0.6],
            kornia.enhance.ThresholdType.THRESH_TOZERO_INV: [0.5, 0.0],
        }
        for mode, values in expected.items():
            self.assert_close(
                kornia.enhance.threshold(data, 0.5, 2.0, mode), torch.tensor([values], device=device, dtype=dtype)
            )

    def test_convention_zca_inverse_preserves_nonzero_sample_axis_5311(self, device, dtype):
        data = torch.tensor([[1.0, 2.0, 4.0], [2.0, 0.0, 3.0]], device=device, dtype=dtype)
        zca = kornia.enhance.ZCAWhitening(dim=1, compute_inv=True).fit(data)
        self.assert_close(zca.inverse_transform(zca(data)), data, low_tolerance=True)

    def test_convention_zca_fitted_state_round_trips_5312(self, device, dtype):
        data = torch.tensor([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0]], device=device, dtype=dtype)
        fitted = kornia.enhance.ZCAWhitening(compute_inv=True).fit(data)
        # Fit the target on other data first, so the load has to replace an existing fit.
        restored = kornia.enhance.ZCAWhitening(compute_inv=True).fit(data.flip(0) * 2.0)
        restored.load_state_dict(fitted.state_dict())
        self.assert_close(restored(data), fitted(data), low_tolerance=True)

    def test_convention_zca_fitted_state_migrates_dtype_5312(self, device):
        data = torch.tensor([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0]], device=device)
        fitted = kornia.enhance.ZCAWhitening().fit(data).to(dtype=torch.float16)
        output = fitted(data.to(dtype=torch.float16))
        assert output.dtype == torch.float16

    def test_convention_zca_unbiased_singleton_is_rejected_5313(self, device, dtype):
        with pytest.raises(ValueError, match="at least two samples"):
            kornia.enhance.zca_whiten(torch.ones(1, 2, device=device, dtype=dtype), unbiased=True)

    def test_convention_integral_duplicate_axis_is_rejected_5314(self, device, dtype):
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        with pytest.raises(BaseError, match="unique"):
            kornia.enhance.integral_tensor(data, (1, -1))

    def test_convention_histogram_zero_bandwidth_is_rejected_5315(self, device, dtype):
        if device.type != "cpu":
            # The check is torch._assert_async: skipped on MPS by design, a device-side assert on CUDA.
            pytest.skip("bandwidth check raises synchronously only on CPU")
        values = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
        bins = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        with pytest.raises(RuntimeError, match="Bandwidth must be finite"):
            kornia.enhance.histogram(values, bins, torch.tensor(0.0, device=device, dtype=dtype))

    def test_convention_image_histogram_empty_range_is_rejected_5315(self, device, dtype):
        with pytest.raises(ValueError, match="range must be finite and non-empty"):
            kornia.enhance.image_histogram2d(torch.ones(2, 2, device=device, dtype=dtype), min=0.0, max=0.0)

    def test_convention_image_histogram_rank1_is_rejected_5316(self, device, dtype):
        with pytest.raises(ShapeError):
            kornia.enhance.image_histogram2d(torch.ones(4, device=device, dtype=dtype))

    def test_convention_image_histogram_rank5_is_rejected_5316(self, device, dtype):
        with pytest.raises(ShapeError):
            kornia.enhance.image_histogram2d(torch.ones(1, 1, 1, 2, 3, device=device, dtype=dtype), n_bins=2)

    def test_convention_shift_rgb_rank5_is_rejected_5327(self, device, dtype):
        image = torch.zeros(2, 2, 3, 1, 1, device=device, dtype=dtype)
        shifts = torch.tensor([0.1, 0.2], device=device, dtype=dtype), torch.zeros(2, device=device, dtype=dtype)
        with pytest.raises(ShapeError):
            kornia.enhance.shift_rgb(image, shifts[0], shifts[1], shifts[1])

    def test_convention_normalize_min_max_rescales_each_spatial_plane(self, device, dtype):
        # (B, C, D, H, W): every depth slice is its own (H, W) plane, so each one spans [0, 1].
        data = torch.tensor([[[[[0.0, 1.0]], [[0.0, 10.0]]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[[0.0, 1.0]], [[0.0, 1.0]]]]], device=device, dtype=dtype)
        self.assert_close(kornia.enhance.normalize_min_max(data), expected, low_tolerance=True)

    @pytest.mark.xfail(strict=True, reason="#5220: equalize differs from float32 reference for float16 input")
    @pytest.mark.device_agnostic
    def test_wart_equalize_float16_matches_float32_reference_5220(self):
        image = torch.zeros(1, 1, 300, 300, dtype=torch.float16)
        image[..., :, 150:] = 1.0
        self.assert_close(kornia.enhance.equalize(image), kornia.enhance.equalize(image.float()).half())

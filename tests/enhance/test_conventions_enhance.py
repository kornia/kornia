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
    def test_convention_add_weighted_tensor_coefficients_are_exact_shape(self, device, dtype):
        src = torch.arange(6, device=device, dtype=dtype).reshape(2, 3)
        alpha = torch.ones(1, 3, device=device, dtype=dtype)
        with pytest.raises(BaseError):
            kornia.enhance.add_weighted(src, alpha, src, 1.0, 0.0)

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
            pytest.param((2, 4), marks=pytest.mark.xfail(strict=True, reason="#5318: C vector indexes missing axis")),
            pytest.param((2, 4, 3, 2, 2), marks=pytest.mark.xfail(strict=True, reason="#5318: C vector checks depth")),
        ],
    )
    def test_wart_denormalize_is_inverse_for_channel_vector_5318(self, shape, device, dtype):
        data = torch.arange(torch.tensor(shape).prod(), device=device, dtype=dtype).reshape(shape) / 32.0
        mean = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        std = torch.tensor([0.5, 0.6, 0.7, 0.8], device=device, dtype=dtype)
        normalized = kornia.enhance.normalize(data, mean, std)
        self.assert_close(kornia.enhance.denormalize(normalized, mean, std), data)

    def test_convention_denormalize_rank5_batch_channel_statistics_workaround(self, device, dtype):
        data = torch.arange(2 * 4 * 3 * 2 * 2, device=device, dtype=dtype).reshape(2, 4, 3, 2, 2) / 32.0
        mean = torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=device, dtype=dtype)
        std = torch.tensor([[0.5, 0.6, 0.7, 0.8]], device=device, dtype=dtype)
        normalized = kornia.enhance.normalize(data, mean, std)
        self.assert_close(kornia.enhance.denormalize(normalized, mean, std), data)

    def test_convention_image_histogram_triangular_range(self, device, dtype):
        image = torch.tensor([[0.25, 1.25]], device=device, dtype=dtype)
        hist, pdf = kornia.enhance.image_histogram2d(
            image, min=0.0, max=1.0, n_bins=2, bandwidth=0.5, kernel="triangular", return_pdf=True
        )
        self.assert_close(hist, torch.tensor([1.0, 0.0], device=device, dtype=dtype))
        self.assert_close(pdf, torch.tensor([1.0, 0.0], device=device, dtype=dtype))

    def test_convention_image_histogram_automatic_centers_follow_bandwidth(self, device, dtype):
        image = torch.tensor([[0.125]], device=device, dtype=dtype)
        hist, _ = kornia.enhance.image_histogram2d(
            image, min=0.0, max=1.0, n_bins=2, bandwidth=0.25, kernel="triangular"
        )
        # Centers are 0.125 and 0.375, generated as min + (i + 0.5) * bandwidth.
        self.assert_close(hist, torch.tensor([1.0, 0.0], device=device, dtype=dtype))

    def test_convention_jpeg_quality_is_one_dimensional(self, device, dtype):
        image = torch.zeros(1, 3, 16, 17, device=device, dtype=dtype)
        with pytest.raises(ShapeError):
            kornia.enhance.jpeg_codec_differentiable(image, torch.tensor(50.0, device=device, dtype=dtype))

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
        image = torch.tensor([[[[0.25, 0.75]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[0.5, 1.0]]]], device=device, dtype=dtype)
        self.assert_close(kornia.enhance.adjust_gamma(image, gamma=1.0, gain=2.0), expected)

    def test_convention_hue_raw_wraps_and_preserves_saturation_value(self, device, dtype):
        hsv = torch.tensor([[[[6.0]], [[0.25]], [[0.75]]]], device=device, dtype=dtype)
        result = kornia.enhance.adjust_hue_raw(hsv, 1.0)
        expected_hue = torch.fmod(torch.tensor(7.0, device=device, dtype=dtype), 2 * torch.pi)
        self.assert_close(result[0, 0, 0, 0], expected_hue)
        self.assert_close(result[:, 1:], hsv[:, 1:])

    def test_convention_hue_raw_keeps_negative_remainder(self, device, dtype):
        hsv = torch.tensor([[[[0.2]], [[0.25]], [[0.75]]]], device=device, dtype=dtype)
        result = kornia.enhance.adjust_hue_raw(hsv, -1.0)
        # Expected from torch.fmod(0.2 - 1.0, 2*pi), Kornia's signed hue remainder.
        self.assert_close(result[0, 0, 0, 0], torch.tensor(-0.8, device=device, dtype=dtype))

    def test_convention_saturation_raw_only_clamps_saturation(self, device, dtype):
        hsv = torch.tensor([[[[0.2]], [[0.8]], [[0.6]]]], device=device, dtype=dtype)
        result = kornia.enhance.adjust_saturation_raw(hsv, 2.0)
        expected = torch.tensor([[[[0.2]], [[1.0]], [[0.6]]]], device=device, dtype=dtype)
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
        expected = torch.tensor([[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]], device=device, dtype=dtype)
        self.assert_close(kornia.enhance.integral_tensor(data, (1,)), expected)

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

    @pytest.mark.xfail(strict=True, reason="#5311: inverse transform loses the configured sample axis")
    def test_wart_zca_inverse_preserves_nonzero_sample_axis_5311(self, device, dtype):
        data = torch.tensor([[1.0, 2.0, 4.0], [2.0, 0.0, 3.0]], device=device, dtype=dtype)
        zca = kornia.enhance.ZCAWhitening(dim=1, compute_inv=True).fit(data)
        self.assert_close(zca.inverse_transform(zca(data)), data, low_tolerance=True)

    @pytest.mark.xfail(strict=True, reason="#5312: fitted ZCA state is not serializable")
    def test_wart_zca_fitted_state_round_trips_5312(self, device, dtype):
        data = torch.tensor([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0]], device=device, dtype=dtype)
        fitted = kornia.enhance.ZCAWhitening(compute_inv=True).fit(data)
        restored = kornia.enhance.ZCAWhitening(compute_inv=True)
        restored.load_state_dict(fitted.state_dict())
        self.assert_close(restored(data), fitted(data), low_tolerance=True)

    @pytest.mark.xfail(strict=True, reason="#5312: fitted tensors do not migrate with module dtype")
    def test_wart_zca_fitted_state_migrates_dtype_5312(self, device):
        data = torch.tensor([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0]], device=device)
        fitted = kornia.enhance.ZCAWhitening().fit(data).to(dtype=torch.float16)
        output = fitted(data.to(dtype=torch.float16))
        assert output.dtype == torch.float16

    @pytest.mark.xfail(strict=True, reason="#5313: unbiased one-sample covariance is singular")
    def test_wart_zca_unbiased_singleton_is_finite_5313(self, device, dtype):
        try:
            output = kornia.enhance.zca_whiten(torch.ones(1, 2, device=device, dtype=dtype), unbiased=True)
        except ValueError:
            return
        assert torch.isfinite(output).all()

    @pytest.mark.xfail(strict=True, reason="#5314: duplicate integral axes are silently applied twice")
    def test_wart_integral_duplicate_axis_is_rejected_5314(self, device, dtype):
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        with pytest.raises(ValueError):
            kornia.enhance.integral_tensor(data, (1, -1))

    @pytest.mark.xfail(strict=True, reason="#5315: zero KDE bandwidth produces NaNs")
    def test_wart_histogram_zero_bandwidth_is_rejected_5315(self, device, dtype):
        values = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
        bins = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        with pytest.raises(ValueError):
            kornia.enhance.histogram(values, bins, torch.tensor(0.0, device=device, dtype=dtype))

    @pytest.mark.xfail(strict=True, reason="#5315: zero image-histogram range produces NaNs")
    def test_wart_image_histogram_empty_range_is_rejected_5315(self, device, dtype):
        with pytest.raises(ValueError):
            kornia.enhance.image_histogram2d(torch.ones(2, 2, device=device, dtype=dtype), min=0.0, max=0.0)

    @pytest.mark.xfail(strict=True, reason="#5316: image histogram does not validate documented ranks")
    @pytest.mark.parametrize("shape", [(1,), (1, 1, 1, 1, 1)])
    def test_wart_image_histogram_rank_is_rejected_5316(self, shape, device, dtype):
        with pytest.raises(ValueError):
            kornia.enhance.image_histogram2d(torch.ones(*shape, device=device, dtype=dtype))

    @pytest.mark.xfail(strict=True, reason="#5220: equalize differs from float32 reference for float16 input")
    @pytest.mark.device_agnostic
    def test_wart_equalize_float16_matches_float32_reference_5220(self):
        image = torch.zeros(1, 1, 300, 300, dtype=torch.float16)
        image[..., :, 150:] = 1.0
        self.assert_close(kornia.enhance.equalize(image), kornia.enhance.equalize(image.float()).half())

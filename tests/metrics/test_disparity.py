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
from kornia.core.exceptions import BaseError, ShapeError, TypeCheckError

from testing.base import BaseTester


def _dynamo_inputs(device, dtype):
    """Build inputs that exercise the masked reduction, the path that must stay compile-friendly."""
    input = torch.rand(2, 4, 5, device=device, dtype=dtype) * 10.0
    target = torch.rand(2, 4, 5, device=device, dtype=dtype) * 10.0 + 1.0
    mask = torch.rand(2, 4, 5, device=device) > 0.3
    return input, target, mask


_LARGE_VALID = 100_000
_LARGE_INVALID = 20_000


def _large_masked_inputs(device, dtype):
    """Build a masked disparity pair large enough to overflow a half-precision accumulator.

    ``float16`` saturates at 65504, so summing a map this size in the input dtype yields ``inf``.
    Exactly 3/4 of the 100k valid pixels are off by 100 px on a ground truth disparity of 100, and
    the masked-out tail is wildly wrong so that it changes the result if it ever leaks into a
    reduction.
    """
    n_outliers = 3 * _LARGE_VALID // 4
    target = torch.full((_LARGE_VALID + _LARGE_INVALID,), 100.0, device=device, dtype=dtype)
    input = target.clone()
    input[:n_outliers] = 200.0
    input[_LARGE_VALID:] = 10000.0

    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:_LARGE_VALID] = True
    return input, target, mask


_DISPARITY_METRICS = [
    kornia.metrics.mean_absolute_disparity_error,
    kornia.metrics.root_mean_squared_disparity_error,
    kornia.metrics.mean_bad_pixel_error,
    kornia.metrics.kitti_d1_error,
]


@pytest.mark.parametrize("metric", _DISPARITY_METRICS, ids=lambda m: m.__name__)
@pytest.mark.parametrize("int_dtype", [torch.uint8, torch.int32, torch.int64])
def test_convention_integer_disparity_maps_are_rejected(metric, int_dtype, device):
    """An integer disparity map must raise rather than return a truncated number.

    The reductions accumulate in float32 and cast back to the map's dtype, which for an integer map
    truncates the ratio: every metric here returned 0 for a true value of 0.5 before the guard.
    That is not a hypothetical input, since KITTI ships disparity as a uint16 PNG and
    ``torch.from_numpy(imread(...))`` hands back an integer tensor, so the failure would show up as
    a plausible-looking score rather than an error.
    """
    target = torch.tensor([10, 10], device=device, dtype=int_dtype)
    input = torch.tensor([10, 20], device=device, dtype=int_dtype)

    with pytest.raises(BaseError) as errinfo:
        metric(input, target)
    assert "floating point" in str(errinfo.value)

    # the same maps as floats still work; the integer path truncated each of these toward zero
    expected = {
        kornia.metrics.mean_absolute_disparity_error: 5.0,  # mean(|0|, |10|), truncated to 5
        kornia.metrics.root_mean_squared_disparity_error: 50.0**0.5,  # sqrt(mean(0, 100)), truncated to 7
        kornia.metrics.mean_bad_pixel_error: 0.5,  # 1 of 2 pixels over 3 px, truncated to 0
        kornia.metrics.kitti_d1_error: 0.5,  # 1 of 2 pixels over both thresholds, truncated to 0
    }[metric]
    assert metric(input.float(), target.float()).item() == pytest.approx(expected, rel=1e-5)


def _can_hold(device: torch.device, dtype: torch.dtype) -> bool:
    """Probe whether a backend can allocate a dtype at all, rather than hardcoding per backend.

    MPS has no float64, so the float32-by-float64 pair below cannot even be built there. Probing
    at runtime means the wider pair starts being exercised on its own once a backend gains the
    dtype, instead of staying permanently skipped behind a device name.
    """
    try:
        torch.zeros(1, device=device, dtype=dtype)
    except (TypeError, RuntimeError, NotImplementedError):
        return False
    return True


@pytest.mark.parametrize("metric", _DISPARITY_METRICS, ids=lambda m: m.__name__)
def test_convention_result_dtype_is_the_promoted_one(metric, device):
    """A mixed-dtype pair returns the promoted dtype, the same for every metric in the module.

    float32 by float64 is the pair that regressed: it returned float64 before the half-precision
    reduction fix and float32 after, while mean_absolute_disparity_error kept returning float64.
    float16 by float32 makes the same point on a backend that cannot hold float64, so the rule is
    checked everywhere rather than only where the regression was first measured.
    """
    pairs = [(torch.float16, torch.float32, torch.float32)]
    if _can_hold(device, torch.float64):
        pairs.append((torch.float32, torch.float64, torch.float64))

    for input_dtype, target_dtype, expected in pairs:
        target = torch.rand(8, device=device, dtype=target_dtype) * 100.0
        input = (target + 4.0).to(input_dtype)

        assert metric(input, target).dtype == expected
        assert metric(input.to(target_dtype), target).dtype == target_dtype
        assert metric(input, target.to(input_dtype)).dtype == input_dtype


@pytest.mark.parametrize(
    "metric", [kornia.metrics.mean_bad_pixel_error, kornia.metrics.kitti_d1_error], ids=lambda m: m.__name__
)
def test_convention_indicator_metrics_have_no_gradient(metric, device):
    """Both indicator metrics are built from comparisons, so they carry no gradient.

    Documented rather than fixed: a counting metric has no useful derivative. The contrast with
    ``mean_absolute_disparity_error``, which does propagate gradients, is what makes it worth
    pinning, since the two live in the same module and look interchangeable.
    """
    target = torch.rand(8, device=device, dtype=torch.float32) * 100.0
    input = (target.clone() + 4.0).requires_grad_()

    assert not metric(input, target).requires_grad
    assert kornia.metrics.mean_absolute_disparity_error(input, target).requires_grad


class TestMeanAbsoluteDisparityError(BaseTester):
    def test_smoke(self, device, dtype):
        input = torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
        target = torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
        actual = kornia.metrics.mean_absolute_disparity_error(input, target)
        assert actual.shape == torch.Size([])

    def test_metric_mean_reduction(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)
        expected = torch.tensor(0.5, device=device, dtype=dtype)
        actual = kornia.metrics.mean_absolute_disparity_error(sample, 1.5 * sample, reduction="mean")
        self.assert_close(actual, expected)

    def test_metric_sum_reduction(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)
        expected = torch.tensor(8.0, device=device, dtype=dtype)
        actual = kornia.metrics.mean_absolute_disparity_error(sample, 1.5 * sample, reduction="sum")
        self.assert_close(actual, expected)

    def test_metric_no_reduction(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)
        expected = torch.full((4, 4), 0.5, device=device, dtype=dtype)
        actual = kornia.metrics.mean_absolute_disparity_error(sample, 1.5 * sample, reduction="none")
        self.assert_close(actual, expected)

    def test_perfect_prediction(self, device, dtype):
        sample = torch.rand(4, 4, device=device, dtype=dtype)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        actual = kornia.metrics.mean_absolute_disparity_error(sample, sample)
        self.assert_close(actual, expected)

    def test_valid_mask(self, device, dtype):
        input = torch.zeros(2, 2, device=device, dtype=dtype)
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        mask = torch.tensor([[True, False], [True, True]], device=device)

        actual_mean = kornia.metrics.mean_absolute_disparity_error(input, target, mask, reduction="mean")
        self.assert_close(actual_mean, torch.tensor(8.0 / 3.0, device=device, dtype=dtype))

        actual_sum = kornia.metrics.mean_absolute_disparity_error(input, target, mask, reduction="sum")
        self.assert_close(actual_sum, torch.tensor(8.0, device=device, dtype=dtype))

        actual_none = kornia.metrics.mean_absolute_disparity_error(input, target, mask, reduction="none")
        expected_none = torch.tensor([[1.0, 0.0], [3.0, 4.0]], device=device, dtype=dtype)
        self.assert_close(actual_none, expected_none)

    def test_valid_mask_numeric(self, device, dtype):
        input = torch.zeros(2, 2, device=device, dtype=dtype)
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
        actual = kornia.metrics.mean_absolute_disparity_error(input, target, mask)
        self.assert_close(actual, torch.tensor(8.0 / 3.0, device=device, dtype=dtype))

    def test_valid_mask_broadcast(self, device, dtype):
        input = torch.zeros(2, 2, 2, device=device, dtype=dtype)
        target = torch.ones(2, 2, 2, device=device, dtype=dtype)
        mask = torch.tensor([[True, False], [False, False]], device=device)
        actual = kornia.metrics.mean_absolute_disparity_error(input, target, mask, reduction="sum")
        self.assert_close(actual, torch.tensor(2.0, device=device, dtype=dtype))

    def test_empty_valid_mask(self, device, dtype):
        input = torch.zeros(2, 2, device=device, dtype=dtype)
        target = torch.ones(2, 2, device=device, dtype=dtype)
        mask = torch.zeros(2, 2, device=device, dtype=torch.bool)

        actual_mean = kornia.metrics.mean_absolute_disparity_error(input, target, mask, reduction="mean")
        assert torch.isnan(actual_mean)

        actual_sum = kornia.metrics.mean_absolute_disparity_error(input, target, mask, reduction="sum")
        self.assert_close(actual_sum, torch.tensor(0.0, device=device, dtype=dtype))

    def test_exception(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.metrics.mean_absolute_disparity_error(None, sample)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            kornia.metrics.mean_absolute_disparity_error(sample, sample[..., :2])
        assert "Shape mismatch" in str(errinfo.value)

        with pytest.raises(BaseError) as errinfo:
            mask = torch.ones(3, device=device, dtype=torch.bool)
            kornia.metrics.mean_absolute_disparity_error(sample, sample, mask)
        assert "broadcastable" in str(errinfo.value)

        with pytest.raises(NotImplementedError) as errinfo:
            kornia.metrics.mean_absolute_disparity_error(sample, 2.0 * sample, reduction="foo")
        assert "Invalid reduction option." in str(errinfo.value)

    def test_large_masked_reduction(self, device, dtype):
        # 3/4 of the valid pixels are off by 100 px, the rest are exact.
        input, target, mask = _large_masked_inputs(device, dtype)
        actual = kornia.metrics.mean_absolute_disparity_error(input, target, mask)
        self.assert_close(actual, torch.tensor(75.0, device=device, dtype=dtype))
        assert actual.dtype == dtype

    def test_dynamo(self, device, dtype, torch_optimizer):
        input, target, mask = _dynamo_inputs(device, dtype)
        op = kornia.metrics.mean_absolute_disparity_error
        op_optimized = torch_optimizer(op)
        self.assert_close(op(input, target, mask), op_optimized(input, target, mask))


class TestRootMeanSquaredDisparityError(BaseTester):
    def test_smoke(self, device, dtype):
        input = torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
        target = torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
        actual = kornia.metrics.root_mean_squared_disparity_error(input, target)
        assert actual.shape == torch.Size([])

    def test_metric_mean_reduction(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)
        expected = torch.tensor(0.5, device=device, dtype=dtype)
        actual = kornia.metrics.root_mean_squared_disparity_error(sample, sample + 0.5, reduction="mean")
        self.assert_close(actual, expected)

    def test_metric_sum_reduction(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)
        expected = torch.tensor(2.0, device=device, dtype=dtype)
        actual = kornia.metrics.root_mean_squared_disparity_error(sample, sample + 0.5, reduction="sum")
        self.assert_close(actual, expected)

    def test_metric_no_reduction(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)
        expected = torch.full((4, 4), 0.5, device=device, dtype=dtype)
        actual = kornia.metrics.root_mean_squared_disparity_error(sample, sample + 0.5, reduction="none")
        self.assert_close(actual, expected)

    def test_perfect_prediction(self, device, dtype):
        sample = torch.rand(4, 4, device=device, dtype=dtype)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        actual = kornia.metrics.root_mean_squared_disparity_error(sample, sample)
        self.assert_close(actual, expected)

    def test_valid_mask(self, device, dtype):
        input = torch.zeros(2, 2, device=device, dtype=dtype)
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        mask = torch.tensor([[True, True], [True, False]], device=device)

        # sqrt((1 + 4 + 9) / 3)
        actual_mean = kornia.metrics.root_mean_squared_disparity_error(input, target, mask, reduction="mean")
        self.assert_close(actual_mean, torch.tensor(2.1602468994, device=device, dtype=dtype))

        # sqrt(1 + 4 + 9)
        actual_sum = kornia.metrics.root_mean_squared_disparity_error(input, target, mask, reduction="sum")
        self.assert_close(actual_sum, torch.tensor(3.7416573867, device=device, dtype=dtype))

        actual_none = kornia.metrics.root_mean_squared_disparity_error(input, target, mask, reduction="none")
        expected_none = torch.tensor([[1.0, 2.0], [3.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(actual_none, expected_none)

    def test_exception(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.metrics.root_mean_squared_disparity_error(None, sample)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(NotImplementedError) as errinfo:
            kornia.metrics.root_mean_squared_disparity_error(sample, 2.0 * sample, reduction="foo")
        assert "Invalid reduction option." in str(errinfo.value)

    def test_large_masked_reduction(self, device, dtype):
        # sqrt(0.75 * 100^2)
        input, target, mask = _large_masked_inputs(device, dtype)
        actual = kornia.metrics.root_mean_squared_disparity_error(input, target, mask)
        self.assert_close(actual, torch.tensor(86.6025403784, device=device, dtype=dtype))
        assert actual.dtype == dtype

    def test_large_error_does_not_saturate_the_square(self, device, dtype):
        # 300^2 = 90000 is past the float16 ceiling, so the squared error map must not be
        # accumulated in the input dtype.
        target = torch.full((64,), 100.0, device=device, dtype=dtype)
        input = target + 300.0
        actual = kornia.metrics.root_mean_squared_disparity_error(input, target)
        self.assert_close(actual, torch.tensor(300.0, device=device, dtype=dtype))

        actual_none = kornia.metrics.root_mean_squared_disparity_error(input, target, reduction="none")
        self.assert_close(actual_none, torch.full((64,), 300.0, device=device, dtype=dtype))

    def test_dynamo(self, device, dtype, torch_optimizer):
        input, target, mask = _dynamo_inputs(device, dtype)
        op = kornia.metrics.root_mean_squared_disparity_error
        op_optimized = torch_optimizer(op)
        self.assert_close(op(input, target, mask), op_optimized(input, target, mask))


class TestMeanBadPixelError(BaseTester):
    def test_smoke(self, device, dtype):
        input = torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
        target = torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
        actual = kornia.metrics.mean_bad_pixel_error(input, target)
        assert actual.shape == torch.Size([])

    def test_metric_mean_reduction(self, device, dtype):
        input = torch.zeros(1, 6, device=device, dtype=dtype)
        target = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], device=device, dtype=dtype)
        expected = torch.tensor(2.0 / 6.0, device=device, dtype=dtype)
        actual = kornia.metrics.mean_bad_pixel_error(input, target, reduction="mean")
        self.assert_close(actual, expected)

    def test_metric_sum_reduction(self, device, dtype):
        input = torch.zeros(1, 6, device=device, dtype=dtype)
        target = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], device=device, dtype=dtype)
        expected = torch.tensor(2.0, device=device, dtype=dtype)
        actual = kornia.metrics.mean_bad_pixel_error(input, target, reduction="sum")
        self.assert_close(actual, expected)

    def test_metric_no_reduction(self, device, dtype):
        input = torch.zeros(1, 6, device=device, dtype=dtype)
        target = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], device=device, dtype=dtype)
        actual = kornia.metrics.mean_bad_pixel_error(input, target, reduction="none")
        self.assert_close(actual, expected)

    def test_perfect_prediction(self, device, dtype):
        sample = torch.rand(4, 4, device=device, dtype=dtype)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        actual = kornia.metrics.mean_bad_pixel_error(sample, sample)
        self.assert_close(actual, expected)

    def test_threshold(self, device, dtype):
        input = torch.zeros(1, 6, device=device, dtype=dtype)
        target = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], device=device, dtype=dtype)

        actual = kornia.metrics.mean_bad_pixel_error(input, target, threshold=1.0)
        self.assert_close(actual, torch.tensor(4.0 / 6.0, device=device, dtype=dtype))

        actual = kornia.metrics.mean_bad_pixel_error(input, target, threshold=4.5)
        self.assert_close(actual, torch.tensor(1.0 / 6.0, device=device, dtype=dtype))

        # an error exactly equal to the threshold is not a bad pixel
        actual = kornia.metrics.mean_bad_pixel_error(input, target, threshold=5.0)
        self.assert_close(actual, torch.tensor(0.0, device=device, dtype=dtype))

    def test_valid_mask(self, device, dtype):
        input = torch.zeros(1, 6, device=device, dtype=dtype)
        target = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], device=device, dtype=dtype)
        mask = torch.tensor([[True, True, True, True, False, True]], device=device)

        actual_mean = kornia.metrics.mean_bad_pixel_error(input, target, valid_mask=mask, reduction="mean")
        self.assert_close(actual_mean, torch.tensor(1.0 / 5.0, device=device, dtype=dtype))

        actual_none = kornia.metrics.mean_bad_pixel_error(input, target, valid_mask=mask, reduction="none")
        expected_none = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(actual_none, expected_none)

    def test_exception(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.metrics.mean_bad_pixel_error(None, sample)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(NotImplementedError) as errinfo:
            kornia.metrics.mean_bad_pixel_error(sample, 2.0 * sample, reduction="foo")
        assert "Invalid reduction option." in str(errinfo.value)

    def test_large_masked_reduction(self, device, dtype):
        # 3/4 of the valid pixels are off by 100 px, well past the 3 px threshold.
        input, target, mask = _large_masked_inputs(device, dtype)
        actual = kornia.metrics.mean_bad_pixel_error(input, target, valid_mask=mask)
        self.assert_close(actual, torch.tensor(0.75, device=device, dtype=dtype))
        assert actual.dtype == dtype

    def test_dynamo(self, device, dtype, torch_optimizer):
        input, target, mask = _dynamo_inputs(device, dtype)
        op = kornia.metrics.mean_bad_pixel_error
        op_optimized = torch_optimizer(op)
        self.assert_close(op(input, target, 0.5, mask), op_optimized(input, target, 0.5, mask))


class TestKittiD1Error(BaseTester):
    # D1 marks a pixel as an outlier only when BOTH criteria hold:
    #   |d - d_gt| > abs_threshold  AND  |d - d_gt| / |d_gt| > rel_threshold
    # errors    = [0.0, 4.0, 4.0,  10.0]
    # relative  = [0.0, 4.0, 0.04, 1.0]
    # abs > 3   = [F,   T,   T,    T]
    # rel > .05 = [F,   T,   F,    T]
    # outlier   = [0,   1,   0,    1]  -> mean 0.5, sum 2.0
    INPUT = [1.0, 5.0, 104.0, 20.0]
    TARGET = [1.0, 1.0, 100.0, 10.0]

    def _sample(self, device, dtype):
        return (
            torch.tensor(self.INPUT, device=device, dtype=dtype),
            torch.tensor(self.TARGET, device=device, dtype=dtype),
        )

    def test_smoke(self, device, dtype):
        # Scaled to a realistic disparity range on purpose: on maps drawn from [0, 1) the error can
        # never exceed the 3 px absolute threshold, so the outlier branch would never be taken and
        # the metric would be a constant 0 whatever the implementation did.
        input = 100.0 * torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
        target = 100.0 * torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
        actual = kornia.metrics.kitti_d1_error(input, target)
        assert actual.shape == torch.Size([])
        assert 0.0 <= actual.item() <= 1.0
        per_pixel = kornia.metrics.kitti_d1_error(input, target, reduction="none")
        assert per_pixel.shape == input.shape
        assert bool((per_pixel > 0).any()), "the outlier branch was never exercised"

    def test_metric_mean_reduction(self, device, dtype):
        input, target = self._sample(device, dtype)
        expected = torch.tensor(0.5, device=device, dtype=dtype)
        actual = kornia.metrics.kitti_d1_error(input, target, reduction="mean")
        self.assert_close(actual, expected)

    def test_metric_sum_reduction(self, device, dtype):
        input, target = self._sample(device, dtype)
        expected = torch.tensor(2.0, device=device, dtype=dtype)
        actual = kornia.metrics.kitti_d1_error(input, target, reduction="sum")
        self.assert_close(actual, expected)

    def test_metric_no_reduction(self, device, dtype):
        input, target = self._sample(device, dtype)
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device, dtype=dtype)
        actual = kornia.metrics.kitti_d1_error(input, target, reduction="none")
        self.assert_close(actual, expected)

    def test_perfect_prediction(self, device, dtype):
        sample = torch.rand(4, 4, device=device, dtype=dtype) + 1.0
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        actual = kornia.metrics.kitti_d1_error(sample, sample)
        self.assert_close(actual, expected)

    def test_absolute_criterion_alone_is_not_an_outlier(self, device, dtype):
        # 4 px of error on a disparity of 100 exceeds the 3 px threshold but is only 4% relative
        # error, so D1 does not count it, while the plain bad-pixel metric does.
        input = torch.tensor([104.0], device=device, dtype=dtype)
        target = torch.tensor([100.0], device=device, dtype=dtype)
        self.assert_close(kornia.metrics.kitti_d1_error(input, target), torch.tensor(0.0, device=device, dtype=dtype))
        self.assert_close(
            kornia.metrics.mean_bad_pixel_error(input, target), torch.tensor(1.0, device=device, dtype=dtype)
        )

    def test_relative_criterion_alone_is_not_an_outlier(self, device, dtype):
        # 2 px of error on a disparity of 10 is 20% relative error but stays within the 3 px
        # absolute threshold, so D1 does not count it either.
        input = torch.tensor([12.0], device=device, dtype=dtype)
        target = torch.tensor([10.0], device=device, dtype=dtype)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(kornia.metrics.kitti_d1_error(input, target), expected)

    # Snippet used to generate expected (the devkit predicate, evaluate_scene_flow.cpp:13-14,51):
    #   d_err = fabs(d_gt - d_est) > 3.0 && fabs(d_gt - d_est) / fabs(d_gt) > 0.05
    #   (10, 13):    err 3.0,  rel 0.300 -> 3.0 > 3.0 is False            -> 0.0
    #   (100, 105):  err 5.0,  rel 0.050 -> 0.05 > 0.05 is False          -> 0.0
    #   (78, 82):    err 4.0,  rel 4/78 = 0.05128 > 0.05, 4/82 = 0.04878  -> 1.0
    #   (-100, -90): err 10.0, rel 10/|-100| = 0.100                      -> 1.0
    @pytest.mark.parametrize(
        ("target_value", "input_value", "expected_value"),
        [(10.0, 13.0, 0.0), (100.0, 105.0, 0.0), (78.0, 82.0, 1.0), (-100.0, -90.0, 1.0)],
        ids=["abs-exactly-at-threshold", "rel-exactly-at-threshold", "gt-is-the-denominator", "negative-gt"],
    )
    def test_convention_boundary_cases(self, target_value, input_value, expected_value, device, dtype):
        """Pin the two details that distinguish D1 from a near-miss implementation.

        Both comparisons are strict, so a pixel sitting exactly on either threshold is an inlier.
        Making either one non-strict flips the first two cases. The relative error divides by the
        **ground truth**, not the prediction: at (78, 82) the two give 0.05128 and 0.04878, which
        straddle the 0.05 threshold, so swapping the denominator flips that case. The negative pair
        pins the ``fabs(d_gt)`` in the devkit, since a signed denominator makes the ratio negative
        and never greater than the threshold.
        """
        input = torch.tensor([input_value], device=device, dtype=dtype)
        target = torch.tensor([target_value], device=device, dtype=dtype)
        expected = torch.tensor(expected_value, device=device, dtype=dtype)
        self.assert_close(kornia.metrics.kitti_d1_error(input, target), expected)

    def test_thresholds(self, device, dtype):
        input = torch.tensor([104.0], device=device, dtype=dtype)
        target = torch.tensor([100.0], device=device, dtype=dtype)

        # lowering the relative threshold below the 4% relative error makes the pixel an outlier
        actual = kornia.metrics.kitti_d1_error(input, target, rel_threshold=0.01)
        self.assert_close(actual, torch.tensor(1.0, device=device, dtype=dtype))

        # raising the absolute threshold above the 4 px error rules it out again
        actual = kornia.metrics.kitti_d1_error(input, target, abs_threshold=5.0, rel_threshold=0.01)
        self.assert_close(actual, torch.tensor(0.0, device=device, dtype=dtype))

    def test_valid_mask(self, device, dtype):
        input, target = self._sample(device, dtype)
        mask = torch.tensor([True, True, True, False], device=device)

        actual_mean = kornia.metrics.kitti_d1_error(input, target, valid_mask=mask, reduction="mean")
        self.assert_close(actual_mean, torch.tensor(1.0 / 3.0, device=device, dtype=dtype))

        actual_sum = kornia.metrics.kitti_d1_error(input, target, valid_mask=mask, reduction="sum")
        self.assert_close(actual_sum, torch.tensor(1.0, device=device, dtype=dtype))

        actual_none = kornia.metrics.kitti_d1_error(input, target, valid_mask=mask, reduction="none")
        expected_none = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=dtype)
        self.assert_close(actual_none, expected_none)

    def test_zero_target_disparity(self, device, dtype):
        # A zero ground truth disparity makes the relative error non-finite. Because both criteria
        # must hold, such pixels fall back to the absolute threshold and the output stays finite.
        input = torch.tensor([0.0, 5.0], device=device, dtype=dtype)
        target = torch.tensor([0.0, 0.0], device=device, dtype=dtype)

        actual = kornia.metrics.kitti_d1_error(input, target, reduction="none")
        expected = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        self.assert_close(actual, expected)
        assert torch.isfinite(actual).all()

    def test_exception(self, device, dtype):
        sample = torch.ones(4, 4, device=device, dtype=dtype)

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.metrics.kitti_d1_error(None, sample)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            kornia.metrics.kitti_d1_error(sample, sample[..., :2])
        assert "Shape mismatch" in str(errinfo.value)

        with pytest.raises(BaseError) as errinfo:
            mask = torch.ones(3, device=device, dtype=torch.bool)
            kornia.metrics.kitti_d1_error(sample, sample, valid_mask=mask)
        assert "broadcastable" in str(errinfo.value)

        with pytest.raises(NotImplementedError) as errinfo:
            kornia.metrics.kitti_d1_error(sample, 2.0 * sample, reduction="foo")
        assert "Invalid reduction option." in str(errinfo.value)

    def test_large_masked_reduction(self, device, dtype):
        # 3/4 of the valid pixels are off by 100 px on a disparity of 100: 100 % relative error,
        # so both criteria hold and the outlier ratio is exactly 0.75.
        input, target, mask = _large_masked_inputs(device, dtype)
        actual = kornia.metrics.kitti_d1_error(input, target, valid_mask=mask)
        self.assert_close(actual, torch.tensor(0.75, device=device, dtype=dtype))
        assert actual.dtype == dtype

    def test_dynamo(self, device, dtype, torch_optimizer):
        input, target, mask = _dynamo_inputs(device, dtype)
        op = kornia.metrics.kitti_d1_error
        op_optimized = torch_optimizer(op)
        self.assert_close(op(input, target, 3.0, 0.05, mask), op_optimized(input, target, 3.0, 0.05, mask))

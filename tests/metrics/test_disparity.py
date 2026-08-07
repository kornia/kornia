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

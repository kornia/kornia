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

from testing.base import BaseTester


class TestAepe(BaseTester):
    def test_metric_mean_reduction(self, device, dtype):
        sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
        expected = torch.tensor(0.565685424, device=device, dtype=dtype)
        actual = kornia.metrics.aepe(sample, 1.4 * sample, reduction="mean")
        self.assert_close(actual, expected)

    def test_metric_sum_reduction(self, device, dtype):
        sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
        expected = torch.tensor(1.4142, device=device, dtype=dtype) * 4**2
        actual = kornia.metrics.aepe(sample, 2.0 * sample, reduction="sum")
        self.assert_close(actual, expected)

    def test_metric_no_reduction(self, device, dtype):
        sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
        expected = torch.zeros(4, 4, device=device, dtype=dtype) + 1.4142
        actual = kornia.metrics.aepe(sample, 2.0 * sample, reduction="none")
        self.assert_close(actual, expected)

    def test_perfect_fit(self, device, dtype):
        sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
        expected = torch.zeros(4, 4, device=device, dtype=dtype)
        actual = kornia.metrics.aepe(sample, sample, reduction="none")
        self.assert_close(actual, expected)

    def test_aepe_alias(self, device, dtype):
        sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
        expected = torch.zeros(4, 4, device=device, dtype=dtype)
        actual_aepe = kornia.metrics.aepe(sample, sample, reduction="none")
        actual_alias = kornia.metrics.average_endpoint_error(sample, sample, reduction="none")
        self.assert_close(actual_aepe, expected)
        self.assert_close(actual_alias, expected)
        self.assert_close(actual_aepe, actual_alias)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError) as errinfo:
            criterion = kornia.metrics.AEPE()
            criterion(None, torch.ones(4, 4, 2, device=device, dtype=dtype))
        assert "Not a Tensor type. Got" in str(errinfo)

        with pytest.raises(NotImplementedError) as errinfo:
            sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
            _ = kornia.metrics.aepe(sample, 2.0 * sample, reduction="foo")
        assert "Invalid reduction option." in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
            _ = kornia.metrics.aepe(sample, 2.0 * sample[..., 0], reduction="mean")
        assert "shape must be [['*', '2']]. Got" in str(errinfo)

    def test_smoke(self, device, dtype):
        input = torch.rand(3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(3, 3, 2, device=device, dtype=dtype)

        criterion = kornia.metrics.AEPE()
        assert criterion(input, target) is not None

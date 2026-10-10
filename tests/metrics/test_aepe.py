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
from kornia.core.exceptions import BaseError

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
        from kornia.core.exceptions import TypeCheckError

        criterion = kornia.metrics.AEPE()
        with pytest.raises(TypeCheckError) as errinfo:
            criterion(None, torch.ones(4, 4, 2, device=device, dtype=dtype))
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
        with pytest.raises(NotImplementedError) as errinfo:
            _ = kornia.metrics.aepe(sample, 2.0 * sample, reduction="foo")
        assert "Invalid reduction option." in str(errinfo)

        from kornia.core.exceptions import ShapeError

        sample = torch.ones(4, 4, 2, device=device, dtype=dtype)
        with pytest.raises(ShapeError) as errinfo:
            _ = kornia.metrics.aepe(sample, 2.0 * sample[..., 0], reduction="mean")
        assert (
            "Shape dimension mismatch" in str(errinfo.value)
            or "Expected shape" in str(errinfo.value)
            or "shape must be" in str(errinfo.value)
        )

    def test_smoke(self, device, dtype):
        input = torch.rand(3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(3, 3, 2, device=device, dtype=dtype)

        criterion = kornia.metrics.AEPE()
        assert criterion(input, target) is not None


class TestConventionsAepe(BaseTester):
    def test_convention_aepe_flow_is_channel_last_and_pools_every_element(self, device, dtype):
        """aepe reads (*, 2) flow in pixels; 'mean' averages the endpoint error over every pixel of every sample."""
        # (B, H, W, 2) = (2, 3, 4, 2). Sample 0: one pixel off by (3, 4), error 5; sample 1: two pixels off by 1.
        target = torch.zeros(2, 3, 4, 2, device=device, dtype=dtype)
        flow = target.clone()
        flow[0, 1, 2] = flow.new_tensor([3.0, 4.0])
        flow[1, 2, 3] = flow.new_tensor([0.0, -1.0])
        flow[1, 0, 0] = flow.new_tensor([1.0, 0.0])
        # (5 + 1 + 1) / 24 pixels; the sum is 7 px, not normalised by the image size
        mean = kornia.metrics.aepe(flow, target)
        assert mean.shape == ()
        self.assert_close(mean, torch.tensor(7.0 / 24.0, device=device, dtype=dtype))
        self.assert_close(
            kornia.metrics.aepe(flow, target, reduction="sum"), torch.tensor(7.0, device=device, dtype=dtype)
        )
        epe = kornia.metrics.aepe(flow, target, reduction="none")
        assert epe.shape == (2, 3, 4)
        expected = torch.zeros(2, 3, 4, device=device, dtype=dtype)
        expected[0, 1, 2], expected[1, 2, 3], expected[1, 0, 0] = 5.0, 1.0, 1.0
        self.assert_close(epe, expected)
        # a channel-first (B, 2, H, W) flow, as RAFT and torchvision store it, is rejected here because W = 4 != 2 (with
        # W == 2 it is accepted and read wrongly): permute it to (B, H, W, 2)
        with pytest.raises((ValueError, BaseError)):
            kornia.metrics.aepe(flow.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
        # average_endpoint_error is aepe itself, and AEPE(reduction) calls it
        assert kornia.metrics.average_endpoint_error is kornia.metrics.aepe
        self.assert_close(kornia.metrics.AEPE("sum")(flow, target), torch.tensor(7.0, device=device, dtype=dtype))

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

from __future__ import annotations

import pytest
import torch

import kornia
from kornia.metrics import AverageMeter

from testing.base import BaseTester, supports_topk


class TestAverageMeter:
    def test_initial_state(self):
        m = AverageMeter()
        assert m.val == 0
        assert m.avg == 0
        assert m.count == 0

    def test_update_scalar(self):
        m = AverageMeter()
        m.update(0.8, n=1)
        m.update(0.4, n=1)
        assert abs(m.avg - 0.6) < 1e-6

    def test_update_weighted(self):
        m = AverageMeter()
        m.update(1.0, n=3)
        m.update(0.0, n=1)
        assert abs(m.avg - 0.75) < 1e-6

    def test_update_tensor(self):
        m = AverageMeter()
        m.update(torch.tensor(0.9), n=1)
        # avg property converts tensor to float
        assert isinstance(m.avg, float)
        assert abs(m.avg - 0.9) < 1e-6

    def test_reset(self):
        m = AverageMeter()
        m.update(1.0, n=5)
        m.reset()
        assert m.val == 0
        assert m.avg == 0
        assert m.count == 0


class TestConventionsAverageMeter(BaseTester):
    def test_low_precision_updates_keep_the_running_sum(self, device, dtype):
        """Repeated low-precision updates retain the full sum and raw last value."""
        meter = AverageMeter()
        value = torch.tensor(1.0, device=device, dtype=dtype)
        for _ in range(1000):
            meter.update(value)

        assert meter.avg == 1.0
        assert meter.count == 1000
        assert meter.val is value
        self.assert_close(
            meter.sum, torch.tensor(1000.0, device=device, dtype=torch.promote_types(dtype, torch.float32))
        )

    def test_weighting_promotes_before_multiplication(self, device, dtype):
        """Promote before weighting so sums and gradients avoid half-precision loss."""
        meter = AverageMeter()
        value = torch.tensor(1000.0, device=device, dtype=dtype, requires_grad=True)
        meter.update(value, n=100)

        assert meter.avg == 1000.0
        self.assert_close(
            meter.sum, torch.tensor(100000.0, device=device, dtype=torch.promote_types(dtype, torch.float32))
        )
        meter.sum.backward()
        self.assert_close(value.grad, torch.tensor(100.0, device=device, dtype=dtype))

    @pytest.mark.applies_to(when=lambda p: p["device_name"] != "mps", reason="MPS does not support float64")
    def test_later_update_preserves_a_wider_dtype(self, device):
        """A later float64 update widens the accumulator without rounding it away."""
        meter = AverageMeter()
        meter.update(torch.tensor(2**24, device=device, dtype=torch.float32))
        meter.update(torch.tensor(1.0, device=device, dtype=torch.float64))

        self.assert_close(meter.sum, torch.tensor(2**24 + 1, device=device, dtype=torch.float64))
        assert meter.avg == (2**24 + 1) / 2

    def test_convention_average_meter_weights_each_value_by_n(self, device, dtype):
        """update(val, n) adds val * n to sum and n to count: avg is the n-weighted mean, in the caller's units."""
        meter = AverageMeter()
        meter.update(torch.tensor(0.75, device=device, dtype=dtype), n=4)
        meter.update(torch.tensor(0.5, device=device, dtype=dtype), n=1)
        # (0.75 * 4 + 0.5 * 1) / 5 = 0.7, not the unweighted (0.75 + 0.5) / 2 = 0.625
        assert isinstance(meter.avg, float)
        self.assert_close(torch.tensor(meter.avg, dtype=dtype), torch.tensor(0.7, dtype=dtype))
        assert meter.count == 5
        self.assert_close(meter.sum, torch.tensor(3.5, device=device, dtype=torch.promote_types(dtype, torch.float32)))
        # val is the last raw value, not weighted
        self.assert_close(meter.val, torch.tensor(0.5, device=device, dtype=dtype))
        # a tensor val keeps its autograd graph in sum
        graph = AverageMeter()
        graph.update(torch.tensor(0.5, device=device, dtype=dtype, requires_grad=True) * 2, n=3)
        assert graph.sum.requires_grad

    def test_convention_average_meter_keeps_the_callers_scale(self, device, dtype):
        """AverageMeter does not rescale: accuracy's percentage stays a percentage."""
        if not supports_topk(device, dtype):
            pytest.skip(f"this torch build has no topk kernel for {dtype} on {device.type}")
        percent = AverageMeter()
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]], device=device, dtype=dtype)
        percent.update(kornia.metrics.accuracy(logits, torch.tensor([1, 1], device=device))[0], n=2)
        assert percent.avg == 50.0

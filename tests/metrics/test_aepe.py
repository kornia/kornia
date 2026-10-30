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
    @pytest.mark.parametrize("integer_dtype", [torch.uint8, torch.int8, torch.int16, torch.int32])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("route", ["function", "alias", "module"])
    def test_integer_flow(self, integer_dtype, reduction, route, device):
        distance = {torch.int8: 120, torch.int32: 50_000}.get(integer_dtype, 200)
        input = torch.tensor([[distance, 0], [0, distance], [3, 4]], device=device, dtype=integer_dtype)
        target = torch.zeros_like(input)
        expected = torch.tensor([distance, distance, 5], device=device, dtype=torch.get_default_dtype())
        if reduction != "none":
            expected = getattr(expected, reduction)()
        op = {"function": kornia.metrics.aepe, "alias": kornia.metrics.average_endpoint_error}.get(route)
        actual = kornia.metrics.AEPE(reduction)(input, target) if op is None else op(input, target, reduction)
        assert actual.dtype == torch.get_default_dtype()
        self.assert_close(actual, expected, atol=0, rtol=0)
        self.assert_close(kornia.metrics.aepe(target, input, reduction), expected, atol=0, rtol=0)

    def test_large_int64_origin(self, device):
        input = torch.tensor([[2**60 + 3, 2**60 + 4]], device=device, dtype=torch.int64)
        target = torch.full_like(input, 2**60)
        self.assert_close(kornia.metrics.aepe(input, target), torch.tensor(5.0, device=device), atol=0, rtol=0)

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


class TestAepeHalfPrecision(BaseTester):
    @pytest.mark.parametrize("scale", [1e-4, 100.0])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_representable_distance_and_gradient(self, device, dtype, scale, reduction):
        flow = torch.tensor([[3 * scale, 4 * scale], [0.0, 5 * scale]], device=device, dtype=dtype, requires_grad=True)
        target = torch.zeros_like(flow, requires_grad=True)
        reference = flow.detach().cpu().double()
        norm = torch.hypot(reference[:, 0], reference[:, 1])
        expected = norm if reduction == "none" else norm.mean() if reduction == "mean" else norm.sum()
        actual = kornia.metrics.aepe(flow, target, reduction=reduction)
        assert actual.dtype == dtype
        assert torch.isfinite(actual).all()
        self.assert_close(actual, expected.to(flow))
        self.assert_close(kornia.metrics.AEPE(reduction)(flow, target), actual)
        grad_flow, grad_target = torch.autograd.grad(actual.sum(), (flow, target))
        expected_grad = reference / norm[:, None]
        if reduction == "mean":
            expected_grad = expected_grad / norm.numel()
        expected_grad = expected_grad.to(flow)
        assert torch.isfinite(grad_flow).all()
        assert torch.isfinite(grad_target).all()
        self.assert_close(grad_flow, expected_grad)
        self.assert_close(grad_target, -expected_grad)


class TestAepeGradientAtExactMatch(BaseTester):
    """The derivative of sqrt at 0 is infinite; an exact match must get a zero gradient, not NaN (#6295)."""

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_gradient_is_zero_at_exact_match_and_unchanged_elsewhere(self, device, dtype, reduction):
        target = torch.tensor([[[1.0, 2.0], [3.5, 4.0], [0.0, 0.0]]], device=device, dtype=dtype)
        flow = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 3.0]]], device=device, dtype=dtype, requires_grad=True)
        epe = kornia.metrics.aepe(flow, target, reduction=reduction)
        (grad,) = torch.autograd.grad(epe.sum(), flow)
        scale = 1.0 / 3.0 if reduction == "mean" else 1.0
        expected = torch.tensor([[[0.0, 0.0], [-1.0 * scale, 0.0], [0.0, 1.0 * scale]]], device=device, dtype=dtype)
        assert torch.isfinite(grad).all()
        self.assert_close(grad, expected)

    def test_masked_loss_gradient_is_finite(self, device, dtype):
        flow = torch.zeros(1, 4, 4, 2, device=device, dtype=dtype, requires_grad=True)
        target = torch.zeros(1, 4, 4, 2, device=device, dtype=dtype)
        target[0, 0, 0] = 1.0
        mask = torch.zeros(1, 4, 4, device=device, dtype=dtype)
        mask[0, 0, 0] = 1.0  # only the mismatched pixel counts; NaN * 0 must not leak in from the matching ones
        loss = (kornia.metrics.aepe(flow, target, reduction="none") * mask).sum()
        (grad,) = torch.autograd.grad(loss, flow)
        assert torch.isfinite(grad).all()
        assert (grad[0, 1:] == 0).all()

    def test_target_gradient_is_zero_at_exact_match(self, device, dtype):
        flow = torch.ones(2, 3, 2, device=device, dtype=dtype)
        target = flow.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(kornia.metrics.aepe(flow, target), target)
        assert (grad == 0).all()

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_value_matches_the_plain_sqrt_formula(self, device, dtype, reduction):
        gen = torch.Generator(device="cpu").manual_seed(0)
        flow = torch.randn(2, 8, 8, 2, generator=gen).to(device=device, dtype=dtype)
        target = torch.randn(2, 8, 8, 2, generator=gen).to(device=device, dtype=dtype)
        target[:, :3] = flow[:, :3]  # exact matches, where the safe argument is substituted
        f32_flow, f32_target = flow.float(), target.float()
        sq = (f32_flow[..., 0] - f32_target[..., 0]) ** 2 + (f32_flow[..., 1] - f32_target[..., 1]) ** 2
        expected = sq.sqrt() if dtype not in (torch.float32, torch.float64) else None
        if expected is None:
            expected = ((flow[..., 0] - target[..., 0]) ** 2 + (flow[..., 1] - target[..., 1]) ** 2).sqrt()
        expected = expected if reduction == "none" else expected.mean() if reduction == "mean" else expected.sum()
        actual = kornia.metrics.aepe(flow, target, reduction=reduction)
        if dtype in (torch.float32, torch.float64):
            assert torch.equal(actual, expected)
        else:
            self.assert_close(actual, expected.to(dtype))

    def test_gradcheck(self, device):
        flow = torch.randn(2, 3, 2, device=device, dtype=torch.float64) + 2.0
        target = torch.randn(2, 3, 2, device=device, dtype=torch.float64)
        self.gradcheck(kornia.metrics.aepe, (flow, target), requires_grad=[True, True])

    def test_dynamo(self, device, dtype, torch_optimizer):
        flow = torch.rand(2, 4, 4, 2, device=device, dtype=dtype, requires_grad=True)
        target = flow.detach().clone()
        target[:, 0] += 0.5
        op = torch_optimizer(kornia.metrics.aepe, fullgraph=True)
        compiled = op(flow, target)
        self.assert_close(compiled, kornia.metrics.aepe(flow, target))
        (grad,) = torch.autograd.grad(compiled.sum(), flow)
        expected = torch.zeros_like(flow)
        expected[:, 0] = -(2.0**-0.5) / (flow.shape[0] * flow.shape[1] * flow.shape[2])
        assert torch.isfinite(grad).all()
        self.assert_close(grad, expected)

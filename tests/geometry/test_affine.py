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

from kornia.geometry.solvers.affine import _find_affine_linear, find_affine

from testing.base import BaseTester


class TestFindAffine(BaseTester):
    @staticmethod
    def _points(device, dtype):
        points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [3.0, 2.0]], device=device, dtype=dtype)[None]
        matrix = torch.tensor([[1.5, -0.25, 4.0], [0.5, -2.0, -3.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)[None]
        target = points @ matrix[:, :2, :2].transpose(-1, -2) + matrix[:, None, :2, 2]
        return points, target, matrix

    def test_exact(self, device, dtype):
        points, target, expected = self._points(device, dtype)
        actual = find_affine(points, target)
        tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(actual, expected, rtol=tol, atol=tol)

    def test_minimal_three_points(self, device, dtype):
        points, target, expected = self._points(device, dtype)
        actual = find_affine(points[:, :3], target[:, :3])
        tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(actual, expected, rtol=tol, atol=tol)

    def test_weighted_ignores_zero_weight_outlier(self, device, dtype):
        points, target, expected = self._points(device, dtype)
        target[..., -1, :] += 100
        weights = torch.tensor([[1.0, 1.0, 1.0, 0.0]], device=device, dtype=dtype)
        actual = find_affine(points, target, weights)
        tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(actual, expected, rtol=tol, atol=tol)

    def test_weighted_noisy_fit_matches_lstsq_reference(self, device, dtype):
        points, target, _ = self._points(device, dtype)
        target = target + torch.tensor(
            [[[0.2, -0.1], [-0.1, 0.15], [0.05, 0.2], [-0.2, -0.05]]], device=device, dtype=dtype
        )
        weights = torch.tensor([[0.25, 1.5, 2.0, 0.75]], device=device, dtype=dtype)
        # Use CPU float64 for an independent, portable reference; MPS lacks lstsq.
        points_cpu, target_cpu, weights_cpu = points.cpu().double(), target.cpu().double(), weights.cpu().double()
        design = torch.cat((points_cpu, torch.ones_like(points_cpu[..., :1])), dim=-1)
        reference = torch.linalg.lstsq(
            design * weights_cpu.sqrt()[..., None], target_cpu * weights_cpu.sqrt()[..., None]
        ).solution.to(device=device, dtype=dtype)
        expected = torch.eye(3, device=device, dtype=dtype)[None]
        expected[:, :2, :] = reference.transpose(-1, -2)
        actual = find_affine(points, target, weights)
        tol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(actual, expected, rtol=tol, atol=tol)
        weight_scale = 1.0e-2 if dtype in (torch.float16, torch.bfloat16) else 1.0e-10
        self.assert_close(find_affine(points, target, weights * weight_scale), actual, rtol=tol, atol=tol)

    def test_weights_do_not_narrow_before_normalization(self, device, dtype):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("Exercises mixed-precision weight normalization.")
        points, target, expected = self._points(device, dtype)
        tolerance = 3e-3
        for scale in (1.0e-8, 1.0e10):
            weights = torch.full((1, 4), scale, device=device, dtype=torch.float32)
            actual = find_affine(points, target, weights)
            self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_float64_extreme_weight_scale_invariance(self, device):
        if device.type != "cpu":
            pytest.skip("CPU float64 regression for extreme floating-point weights.")
        points, target, expected = self._points(device, torch.float64)
        weights = torch.full((1, 4), 1.0e300, device=device, dtype=torch.float64)
        actual = find_affine(points, target, weights)
        self.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("angle", [0.0, 45.0, 120.0])
    def test_elongated_source_any_direction(self, device, dtype, angle):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("Input rounding dominates in half precision.")
        # A 1000 px by 6 px strip of exact correspondences, centred at (700, 500) and rotated by `angle`.
        u = torch.linspace(-500.0, 500.0, 64, dtype=torch.float64)
        v = 3.0 * torch.tensor([1.0, -1.0], dtype=torch.float64).repeat(32)
        theta = torch.tensor(angle, dtype=torch.float64).deg2rad()
        points = torch.stack(
            (700 + u * theta.cos() - v * theta.sin(), 500 + u * theta.sin() + v * theta.cos()), dim=-1
        )[None]
        _, _, matrix = self._points("cpu", torch.float64)
        target = points @ matrix[:, :2, :2].transpose(-1, -2) + matrix[:, None, :2, 2]
        actual = find_affine(points.to(device, dtype), target.to(device, dtype)).cpu().double()
        transferred = points @ actual[:, :2, :2].transpose(-1, -2) + actual[:, None, :2, 2]
        tol = 1e-2 if dtype == torch.float32 else 1e-8
        self.assert_close(transferred, target, rtol=0, atol=tol)

    def test_scale_and_translation_conditioning(self, device, dtype):
        points, target, expected = self._points(device, dtype)
        scale = 10.0 if dtype in (torch.float16, torch.bfloat16) else 1.0e3
        offset_value = 100.0 if dtype in (torch.float16, torch.bfloat16) else 1.0e5
        offset = torch.tensor([offset_value, -2.0 * offset_value], device=device, dtype=dtype)
        points = points * scale + offset
        target = target * scale - offset
        expected = expected.clone()
        expected[:, :2, 2] = scale * expected[:, :2, 2] - offset - (expected[:, :2, :2] @ offset[..., None])[..., 0]
        actual = find_affine(points, target)
        tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
        self.assert_close(actual, expected, rtol=tol, atol=tol)

    def test_invalid_batches_are_nan(self, device, dtype):
        good, target, _ = self._points(device, dtype)
        bad = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], device=device, dtype=dtype)[None]
        actual = find_affine(torch.cat((good, bad)), torch.cat((target, bad)))
        assert torch.isfinite(actual[0]).all()
        assert torch.isnan(actual[1, :2]).all()
        zero_weight = find_affine(good, target, torch.zeros(1, 4, device=device, dtype=dtype))
        assert torch.isnan(zero_weight[:, :2]).all()

    def test_errors(self, device, dtype):
        points = torch.zeros(1, 2, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="At least 3"):
            find_affine(points, points)
        with pytest.raises(ValueError, match="same shape"):
            find_affine(torch.zeros(1, 3, 2, device=device, dtype=dtype), points)
        points = torch.zeros(1, 3, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="weights"):
            find_affine(points, points, torch.ones(1, 2, device=device, dtype=dtype))
        with pytest.raises(ValueError, match="non-negative"):
            find_affine(points, points, -torch.ones(1, 3, device=device, dtype=dtype))
        with pytest.raises(ValueError, match="shape"):
            find_affine(points[:, None], points[:, None])
        other_dtype = torch.float16 if dtype == torch.float32 else torch.float32
        with pytest.raises(TypeError, match="same dtype"):
            find_affine(points, points.to(other_dtype))

    def test_origin_fixed_linear(self, device, dtype):
        matrix = torch.tensor([[2.0, -1.0], [0.5, 3.0]], device=device, dtype=dtype)
        points = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=device, dtype=dtype)
        target = points @ matrix.T
        actual = _find_affine_linear(points, target)
        tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(actual, matrix, rtol=tol, atol=tol)
        padded = _find_affine_linear(points[:1], target[:1])
        assert torch.isnan(padded).all()

    def test_origin_fixed_prefix_and_padding(self, device, dtype):
        matrix = torch.tensor([[2.0, -1.0], [0.5, 3.0]], device=device, dtype=dtype)
        points = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=device, dtype=dtype)
        target = points @ matrix.T
        padded_points = torch.cat((points, torch.zeros(2, 2, device=device, dtype=dtype)))[None, None]
        padded_target = torch.cat((target, torch.zeros(2, 2, device=device, dtype=dtype)))[None, None]
        weights = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]], device=device, dtype=dtype)
        actual = _find_affine_linear(padded_points, padded_target, weights)
        tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(actual, matrix[None, None], rtol=tol, atol=tol)

    def test_gradcheck(self, device):
        points, target, _ = self._points(device, torch.float64)
        self.gradcheck(find_affine, (points.requires_grad_(), target.requires_grad_()), rtol=1e-5, atol=1e-6)

    def test_weighted_gradcheck_and_constant_target(self, device):
        points, _, _ = self._points(device, torch.float64)
        target = torch.tensor([[[2.0, -3.0]]], device=device, dtype=torch.float64).expand_as(points).clone()
        weights = torch.tensor([[0.5, 1.0, 1.5, 2.0]], device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(find_affine, (points.requires_grad_(), target.requires_grad_(), weights), rtol=1e-5, atol=1e-6)
        find_affine(points, target).square().sum().backward()
        assert torch.isfinite(points.grad).all()
        assert torch.isfinite(target.grad).all()

    def test_empty_batch(self, device, dtype):
        points = torch.empty(0, 3, 2, device=device, dtype=dtype)
        assert find_affine(points, points).shape == (0, 3, 3)

    def test_dynamo(self, device, dtype, torch_optimizer):
        points, target, _ = self._points(device, dtype)
        actual = torch_optimizer(find_affine)(points, target)
        self.assert_close(actual, find_affine(points, target))

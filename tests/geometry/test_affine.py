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

from kornia.geometry.solvers.affine import (
    _find_affine_linear,
    _find_affine_linear_pair,
    _find_affine_triplet,
    find_affine,
)

from testing.base import BaseTester, dynamo_is_available


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

    def test_meta_device_infers_the_shape(self):
        # torch.autocast raises for the meta device type, so the solver enters it only inside an autocast region.
        points = torch.empty(2, 5, 2, device="meta")
        assert find_affine(points, points).shape == (2, 3, 3)

    def test_autocast_preserves_float32_solver_precision(self, device):
        """Float32 inputs must not let ambient AMP narrow the affine matrix products."""
        if device.type not in ("cpu", "cuda", "mps"):
            pytest.skip("CPU/CUDA/MPS mixed-precision coverage")
        autocast_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        u = torch.linspace(-500.0, 500.0, 64, device=device)
        v = 100.0 * torch.tensor([1.0, -1.0], device=device).repeat(32)
        theta = torch.tensor(45.0, device=device).deg2rad()
        points = torch.stack(
            (700.0 + u * theta.cos() - v * theta.sin(), 500.0 + u * theta.sin() + v * theta.cos()), dim=-1
        )[None]
        expected = torch.tensor([[1.5, -0.25, 4.0], [0.5, -2.0, -3.0], [0.0, 0.0, 1.0]], device=device)[None]
        target = points @ expected[:, :2, :2].transpose(-1, -2) + expected[:, None, :2, 2]
        weights = torch.linspace(0.1, 2.0, points.shape[1], device=device)[None]

        reference = find_affine(points, target, weights)
        reference_refit = _find_affine_linear(points - points[:, :1], target - target[:, :1])
        reference_pair = _find_affine_linear_pair(points[:, 1:3] - points[:, :1], target[:, 1:3] - target[:, :1])
        reference_triplet = _find_affine_triplet(points[:, :3], target[:, :3])

        with torch.autocast(device.type, dtype=autocast_dtype):
            actual = find_affine(points, target, weights)
            refit = _find_affine_linear(points - points[:, :1], target - target[:, :1])
            pair = _find_affine_linear_pair(points[:, 1:3] - points[:, :1], target[:, 1:3] - target[:, :1])
            triplet = _find_affine_triplet(points[:, :3], target[:, :3])
            assert torch.is_autocast_enabled(device.type)

        assert actual.dtype == refit.dtype == pair.dtype == triplet.dtype == points.dtype
        self.assert_close(actual, reference, rtol=0, atol=0)
        self.assert_close(refit, reference_refit, rtol=0, atol=0)
        self.assert_close(pair, reference_pair, rtol=0, atol=0)
        self.assert_close(triplet, reference_triplet, rtol=0, atol=0)
        self.assert_close(actual, expected, rtol=0, atol=3e-4)
        self.assert_close(refit, expected[:, :2, :2], rtol=0, atol=1e-5)
        self.assert_close(pair, expected[:, :2, :2], rtol=0, atol=1e-5)
        # The three-point fit extrapolates its translation to the origin, 350 px from points 32 px apart, which
        # amplifies the input rounding 23 to 58 times: the float32 rounding of its inputs alone puts the exact fit
        # 1.7e-4 from expected on the CPU and 3.5e-4 on CUDA, whose float32 sin(pi / 4) is one ulp above the CPU's.
        # Its float32 accuracy is therefore measured against a float64 solve of the same inputs, on the CPU because
        # MPS has no float64.
        triplet64 = _find_affine_triplet(points[:, :3].cpu().double(), target[:, :3].cpu().double())
        self.assert_close(triplet.cpu().double(), triplet64, rtol=0, atol=2e-4)

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

    def test_origin_fixed_least_squares_with_zero_padding(self, device, dtype):
        generator = torch.Generator().manual_seed(5)
        points = torch.rand(2, 10, 2, generator=generator, dtype=torch.float64) - 0.5
        matrix = torch.tensor([[1.1, -0.2], [0.1, 0.9]], dtype=torch.float64)
        target = points @ matrix.T + 0.01 * torch.randn(2, 10, 2, generator=generator, dtype=torch.float64)
        # Origin-fixed least squares, as AdaLAM refits relative to its seed: no intercept column.
        expected = torch.linalg.lstsq(points, target).solution.transpose(-1, -2)
        # Zero rows, AdaLAM's padding of neighborhoods with fewer inliers, contribute nothing.
        padded_points = torch.cat((points, torch.zeros(2, 4, 2, dtype=torch.float64)), dim=1)
        padded_target = torch.cat((target, torch.zeros(2, 4, 2, dtype=torch.float64)), dim=1)
        actual = _find_affine_linear(padded_points.to(device, dtype), padded_target.to(device, dtype))
        assert actual.dtype == dtype
        tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(actual, expected.to(device, dtype), rtol=tol, atol=tol)

    def test_origin_fixed_pair(self, device, dtype):
        matrix = torch.tensor([[2.0, -1.0], [0.5, 3.0]], device=device, dtype=dtype)
        points = torch.tensor(
            [[[1.0, 0.0], [0.0, 2.0]], [[0.5, -0.25], [0.25, 0.75]], [[-1.0, 1.0], [1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )[None].expand(2, -1, -1, -1)
        target = points @ matrix.T
        actual = _find_affine_linear_pair(points, target)
        assert actual.shape == (2, 3, 2, 2)
        assert actual.dtype == dtype
        tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(actual, matrix.expand_as(actual), rtol=tol, atol=tol)

    def test_origin_fixed_pair_rank_deficient(self, device, dtype):
        # Collinear with the origin, a repeated point, a point at the origin, and a near-collinear pair whose
        # |det X| / ||X||_F**2 = 1e-5 lies below sqrt(eps) of float32.
        points = torch.tensor(
            [[[1.0, 2.0], [-2.0, -4.0]], [[1.0, 2.0], [1.0, 2.0]], [[0.0, 0.0], [1.0, 2.0]], [[1.0, 0.0], [1.0, 2e-5]]],
            device=device,
            dtype=dtype,
        )
        actual = _find_affine_linear_pair(points, points + 1)
        assert torch.isnan(actual[:3]).all()
        if dtype != torch.float64:
            assert torch.isnan(actual[3]).all()

    def test_origin_fixed_pair_matches_least_squares(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("Compares two float32 solvers on the same half-rounded inputs.")
        generator = torch.Generator().manual_seed(3)
        points = (torch.rand(64, 2, 2, generator=generator) - 0.5).to(device=device, dtype=dtype)
        target = (torch.rand(64, 2, 2, generator=generator) - 0.5).to(device=device, dtype=dtype)
        # Two exactly singular pairs: collinear with the origin, and one point at the origin.
        points[0, 1] = -3 * points[0, 0]
        points[1, 0] = 0
        expected = _find_affine_linear(points, target)
        actual = _find_affine_linear_pair(points, target)
        self.assert_close(torch.isnan(actual), torch.isnan(expected), atol=0, rtol=0)
        assert torch.isnan(actual[:2]).all()
        valid = ~torch.isnan(expected)
        # The normal equation squares the condition number; the direct solve does not.
        self.assert_close(actual[valid], expected[valid], rtol=1e-3, atol=1e-3)

    def test_triplet(self, device, dtype):
        points, target, expected = self._points(device, dtype)
        # Move the first point off the origin: the translation follows from it. The shifted targets stay
        # integers or quarters below 256, exact in bfloat16.
        offset = torch.tensor([30.0, -20.0], device=device, dtype=dtype)
        points, target = points[:, :3] + offset, target[:, :3] + offset @ expected[0, :2, :2].T
        actual = _find_affine_triplet(points.expand(4, -1, -1), target.expand(4, -1, -1), torch.ones(4, 3))
        assert actual.shape == (4, 3, 3)
        assert actual.dtype == dtype
        self.assert_close(actual[:, 2], expected[:, 2].expand(4, -1), atol=0, rtol=0)
        tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        self.assert_close(actual, expected.expand(4, -1, -1), rtol=tol, atol=tol)

    def test_triplet_matches_find_affine(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("Compares two float32 solvers on the same half-rounded inputs.")
        generator = torch.Generator().manual_seed(7)
        points = (1000 * torch.rand(64, 3, 2, generator=generator)).to(device=device, dtype=dtype)
        target = (1000 * torch.rand(64, 3, 2, generator=generator)).to(device=device, dtype=dtype)
        points[0, 2] = 2 * points[0, 1] - points[0, 0]  # collinear
        points[1, 1] = points[1, 0]  # repeated
        expected = find_affine(points, target)
        actual = _find_affine_triplet(points, target)
        self.assert_close(torch.isnan(actual), torch.isnan(expected), atol=0, rtol=0)
        assert torch.isnan(actual[:2, :2]).all()
        # Both are exact fits: each maps its own three source points onto their targets.
        valid = ~torch.isnan(actual).any(dim=(-2, -1))
        transferred = points[valid] @ actual[valid, :2, :2].transpose(-1, -2) + actual[valid, None, :2, 2]
        tol = 1e-2 if dtype == torch.float32 else 1e-8
        self.assert_close(transferred, target[valid], rtol=0, atol=tol)

    def test_origin_fixed_pair_errors(self, device, dtype):
        points = torch.zeros(4, 3, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Exactly 2"):
            _find_affine_linear_pair(points, points)
        with pytest.raises(ValueError, match="At least 2"):
            _find_affine_linear_pair(points[:, :1], points[:, :1])
        with pytest.raises(ValueError, match="same shape"):
            _find_affine_linear_pair(points[:, :2], points[:2, :2])
        quadruples = torch.zeros(4, 4, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Exactly 3"):
            _find_affine_triplet(quadruples, quadruples)

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

    @pytest.mark.skipif(not dynamo_is_available(), reason="no Dynamo on this torch/python pair")
    def test_eager_backend_traces_weighted_fit(self, device, dtype):
        # Weight values are not read back on the host, so a weighted fit traces as one graph. The name keeps
        # "compile" and "dynamo" out, so the ordinary jobs run it.
        points, target, _ = self._points(device, dtype)
        weights = torch.tensor([[0.5, 1.0, 1.5, 2.0]], device=device, dtype=dtype)
        torch._dynamo.reset()
        autocast_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        with torch.autocast(device.type, dtype=autocast_dtype):
            actual = torch.compile(find_affine, backend="eager", fullgraph=True)(points, target, weights)
        self.assert_close(actual, find_affine(points, target, weights))

    def test_weighted_fit_without_host_sync(self, device, dtype):
        if device.type != "cuda":
            pytest.skip("Host synchronization is observable on CUDA only.")
        points, target, _ = self._points(device, dtype)
        weights = torch.ones(1, 4, device=device, dtype=dtype)
        find_affine(points, target, weights)  # lazy CUDA and cuBLAS initialization may synchronize
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            find_affine(points, target, weights)
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    def test_dynamo(self, device, dtype, torch_optimizer):
        points, target, _ = self._points(device, dtype)
        actual = torch_optimizer(find_affine)(points, target)
        self.assert_close(actual, find_affine(points, target))

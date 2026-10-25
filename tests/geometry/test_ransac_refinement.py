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

from kornia.geometry.epipolar.essential import _refine_essential_lm
from kornia.geometry.epipolar.fundamental import _refine_fundamental_lm
from kornia.geometry.homography import _refine_homography_lm

from testing.base import BaseTester


class TestRANSACRefinementCPU(BaseTester):
    def scene(self, device, dtype, model):
        if device.type != "cpu" or dtype in (torch.float16, torch.bfloat16):
            pytest.skip("CPU LM fast path uses float32 or float64")
        generator = torch.Generator(device=device).manual_seed(42)
        x1 = torch.randn(60, 3, generator=generator, device=device, dtype=dtype)
        x1[:, 2] = 1.0
        if model == "homography":
            matrix = torch.eye(3, device=device, dtype=dtype)
            x2 = x1[:, :2].clone()
            refine = _refine_homography_lm
        else:
            matrix = x1.new_tensor([[0.0, -0.3, -0.2], [0.3, 0.0, -1.0], [0.2, 1.0, 0.0]])
            depth = torch.rand(60, 1, generator=generator, device=device, dtype=dtype) + 2.0
            x2 = x1 * depth + x1.new_tensor([1.0, -0.2, 0.3])
            x2 = x2 / x2[:, 2:]
            refine = _refine_fundamental_lm if model == "fundamental" else _refine_essential_lm
        matrix = matrix + 0.01 * torch.randn(3, 3, generator=generator, device=device, dtype=dtype)
        x2[:, :2] += 0.001 * torch.randn(60, 2, generator=generator, device=device, dtype=dtype)
        return refine, matrix[None], x1, x2

    @pytest.mark.parametrize("model", ["homography", "fundamental", "essential"])
    @pytest.mark.parametrize("loss", ["truncated", "cauchy"])
    def test_mask_excludes_nonfinite_points(self, device, dtype, model, loss):
        refine, matrix, x1, x2 = self.scene(device, dtype, model)
        mask = torch.arange(len(x1), device=device) % 3 != 0
        with torch.no_grad():
            expected = refine(matrix, x1[mask], x2[mask], None, loss, 0.01, 3)
            x1[~mask] = float("nan")
            actual = refine(matrix, x1, x2, mask[None], loss, 0.01, 3)
        self.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    @pytest.mark.parametrize("model", ["homography", "fundamental", "essential"])
    @pytest.mark.parametrize("loss", ["truncated", "cauchy"])
    @pytest.mark.parametrize("iterations", [1, 3, 20])
    def test_matches_differentiable_refinement(self, device, dtype, model, loss, iterations):
        refine, matrix, x1, x2 = self.scene(device, dtype, model)
        # Sixteen starts, perturbed from the scene's model with per-entry deviations from 0.01 to 1. From the far ones
        # some last trials raise the cost and the reference rejects them, so a shortcut that accepted its last trial
        # unconditionally fails here for the fundamental and essential matrices. The homography steps rejected here
        # are too small to tell apart from rounding.
        generator = torch.Generator(device=device).manual_seed(0)
        scales = torch.logspace(-2, 0, 16, device=device, dtype=dtype)[:, None, None]
        models = matrix + scales * torch.randn(16, 3, 3, generator=generator, device=device, dtype=dtype)
        # The differentiable path retains the fully vectorized implementation as a reference.
        with torch.enable_grad():
            expected = refine(models, x1, x2, None, loss, 0.01, iterations)
        with torch.no_grad():
            actual = refine(models, x1, x2, None, loss, 0.01, iterations)
        tolerance = 5e-4 if dtype == torch.float32 else 2e-6
        self.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize("loss", ["truncated", "cauchy"])
    def test_sampson_trial_cost_matches_normal_equations(self, device, dtype, loss):
        from kornia.geometry.epipolar import fundamental

        _, matrix, x1, x2 = self.scene(device, dtype, "fundamental")
        algebraic = fundamental._epipolar_design_rows(x1, x2).T
        quadratic = torch.cat(
            [fundamental._epipolar_design_rows(x1, x1), fundamental._epipolar_design_rows(x2, x2)], 1
        ).T
        tangent = matrix[:, None].expand(-1, 7, -1, -1)
        for mask in (None, (torch.arange(len(x1), device=device) % 3 != 0)[None]):
            _, expected = fundamental._sampson_normal_equations(matrix, tangent, algebraic, quadratic, mask, loss, 0.01)
            actual = fundamental._sampson_cost(matrix, algebraic, quadratic, mask, loss, 0.01)
            self.assert_close(actual, expected)

    @pytest.mark.parametrize("model", ["homography", "fundamental", "essential"])
    def test_batched_weighted_mask(self, device, dtype, model):
        refine, matrix, x1, x2 = self.scene(device, dtype, model)
        models = matrix.repeat(3, 1, 1)
        models[1] += 0.1
        weights = x1.new_ones(3, len(x1))
        weights[0, :20], weights[1, 20:40], weights[2, 40:] = 0.0, 0.5, 0.0
        with torch.enable_grad():
            expected = refine(models, x1, x2, weights, "cauchy", 0.01, 5)
        with torch.no_grad():
            actual = refine(models, x1, x2, weights, "cauchy", 0.01, 5)
        tolerance = 5e-4 if dtype == torch.float32 else 2e-6
        self.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize("model", ["homography", "fundamental", "essential"])
    @pytest.mark.parametrize("loss", ["truncated", "cauchy"])
    def test_numeric_zero_mask_excludes_singular_correspondence(self, device, dtype, model, loss):
        # A zero numeric mask is how the compiled RANSAC program drops its padding rows.  It cannot use the CPU
        # boolean-mask compaction, so the residual and Jacobian must be made finite before masking.  The appended
        # correspondence lies at a projective horizon for H and at both epipoles for F/E.
        refine, matrix, x1, x2 = self.scene(device, dtype, model)
        if model == "homography":
            matrix = matrix.new_tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
            singular_x1 = x1.new_tensor([[1.0, 0.0, 1.0]])
            singular_x2 = x2.new_tensor([[0.0, 0.0]])
        else:
            matrix = matrix.new_tensor([[[0.0, -0.8, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
            if model == "essential":
                matrix = matrix / matrix.norm(dim=(-2, -1), keepdim=True)
            singular_x1 = x1.new_tensor([[0.0, 0.0, 1.0]])
            singular_x2 = x2.new_tensor([[0.0, 0.0, 1.0]])
        weights = x1.new_ones(1, len(x1) + 1)
        weights[0, -1] = 0.0
        with torch.no_grad():
            # The control masks a regular correspondence in the same slot, so both runs reduce over the same number
            # of rows. Five unconverged LM steps amplify a change in float32 reduction order past the tolerance.
            initial = refine(matrix, x1, x2, None, loss, 0.1, 0)
            expected = refine(matrix, torch.cat([x1, x1[:1]]), torch.cat([x2, x2[:1]]), weights, loss, 0.1, 5)
            actual = refine(matrix, torch.cat([x1, singular_x1]), torch.cat([x2, singular_x2]), weights, loss, 0.1, 5)
        assert torch.isfinite(actual).all()
        assert (expected - initial).norm() > 1e-3
        tolerance = 5e-4 if dtype == torch.float32 else 2e-6
        self.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize("model", ["homography", "fundamental", "essential"])
    def test_empty_mask(self, device, dtype, model):
        refine, matrix, x1, x2 = self.scene(device, dtype, model)
        mask = torch.zeros(1, len(x1), device=device, dtype=torch.bool)
        with torch.enable_grad():
            expected = refine(matrix, x1, x2, mask, "cauchy", 0.01, 3)
        with torch.no_grad():
            actual = refine(matrix, x1, x2, mask, "cauchy", 0.01, 3)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("model", ["homography", "fundamental", "essential"])
    def test_differentiable_path(self, device, dtype, model):
        refine, matrix, x1, x2 = self.scene(device, dtype, model)
        matrix.requires_grad_()
        x1.requires_grad_()
        x2.requires_grad_()
        result = refine(matrix, x1, x2, None, "cauchy", 0.01, 3)
        gradients = torch.autograd.grad(result.square().sum(), (matrix, x1, x2))
        assert all(torch.isfinite(gradient).all() for gradient in gradients)

    @pytest.mark.parametrize("model", ["homography", "fundamental", "essential"])
    def test_keeps_input_tensors(self, device, dtype, model):
        refine, matrix, x1, x2 = self.scene(device, dtype, model)
        inputs = (matrix, x1, x2)
        copies = tuple(value.clone() for value in inputs)
        with torch.no_grad():
            refine(matrix, x1, x2, None, "cauchy", 0.01, 3)
        for value, copy in zip(inputs, copies):
            self.assert_close(value, copy, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("model", ["fundamental", "essential"])
    def test_partial_acceptance_keeps_zero_weight_model(self, device, dtype, model):
        refine, matrix, x1, x2 = self.scene(device, dtype, model)
        models = matrix.repeat(2, 1, 1)
        weights = x1.new_ones(2, len(x1))
        weights[0] = 0.0
        with torch.enable_grad():
            initial = refine(models, x1, x2, weights, "cauchy", 0.01, 0)
            expected = refine(models, x1, x2, weights, "cauchy", 0.01, 5)
        with torch.no_grad():
            actual = refine(models, x1, x2, weights, "cauchy", 0.01, 5)
        # The zero-weight row cannot lower its zero cost. The other row must take at least one accepted step,
        # exercising a mixture of accepted and rejected models without observing the refiner's internals.
        self.assert_close(actual[0], initial[0], atol=0.0, rtol=0.0)
        assert (actual[1] - initial[1]).norm() > 1e-3
        tolerance = 5e-4 if dtype == torch.float32 else 2e-6
        self.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

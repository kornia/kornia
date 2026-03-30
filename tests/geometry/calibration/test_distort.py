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

from kornia.geometry.calibration.distort import distort_points

from testing.base import BaseTester


class TestDistortPoints(BaseTester):
    def test_smoke(self, device, dtype):
        points = torch.rand(1, 2, device=device, dtype=dtype)
        K = torch.rand(3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(4, device=device, dtype=dtype)
        pointsu = distort_points(points, K, distCoeff)
        assert points.shape == pointsu.shape

        new_K = torch.rand(3, 3, device=device, dtype=dtype)
        pointsu = distort_points(points, K, distCoeff, new_K)
        assert points.shape == pointsu.shape

    def test_smoke_batch(self, device, dtype):
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(1, 4, device=device, dtype=dtype)
        pointsu = distort_points(points, K, distCoeff)
        assert points.shape == pointsu.shape

        new_K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        pointsu = distort_points(points, K, distCoeff, new_K)
        assert points.shape == pointsu.shape

    @pytest.mark.parametrize(
        "batch_size, num_points, num_distcoeff", [(1, 3, 4), (2, 4, 5), (3, 5, 8), (4, 6, 12), (5, 7, 14)]
    )
    def test_shape(self, batch_size, num_points, num_distcoeff, device, dtype):
        B, N, Ndist = batch_size, num_points, num_distcoeff

        points = torch.rand(B, N, 2, device=device, dtype=dtype)
        K = torch.rand(B, 3, 3, device=device, dtype=dtype)
        new_K = torch.rand(B, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(B, Ndist, device=device, dtype=dtype)

        pointsu = distort_points(points, K, distCoeff, new_K)
        assert pointsu.shape == (B, N, 2)

    def test_gradcheck(self, device):
        # ── ORIGINAL (partial): only points had requires_grad ──────────────
        # This left distCoeff, K, and new_K untested — meaning nobody had
        # verified that gradients actually flow through distortion coefficients.
        # That matters: without coefficient gradients you cannot optimise camera
        # intrinsics end-to-end via gradient descent (NeRF, bundle adjustment).
        #
        # ── FIX: enable requires_grad on ALL differentiable inputs ──────────
        points = torch.rand(1, 8, 2, device=device, dtype=torch.float64, requires_grad=True)
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        new_K = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        distCoeff = torch.rand(1, 4, device=device, dtype=torch.float64, requires_grad=True)

        assert self.gradcheck(distort_points, (points, K, distCoeff, new_K), raise_exception=True, fast_mode=True)

    def test_gradcheck_distcoeff_only(self, device):
        """Gradient flows through distortion coefficients independently.

        This is the critical use-case test: optimising distCoeff via gradient
        descent (e.g. camera calibration refinement during NeRF training) requires
        that d(output)/d(distCoeff) is non-zero and correctly computed.

        We fix points and K, and only differentiate through distCoeff — isolating
        the coefficient gradient path from the point gradient path.
        """
        points = torch.rand(1, 8, 2, device=device, dtype=torch.float64)
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        new_K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        distCoeff = torch.rand(1, 4, device=device, dtype=torch.float64, requires_grad=True)

        assert self.gradcheck(distort_points, (points, K, distCoeff, new_K), raise_exception=True, fast_mode=True)

    def test_gradcheck_K_only(self, device):
        """Gradient flows through the camera intrinsic matrix K independently.

        Verifies that d(output)/d(K) is correctly computed, which is required
        for joint optimisation of intrinsics and distortion parameters.
        """
        points = torch.rand(1, 8, 2, device=device, dtype=torch.float64)
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        new_K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        distCoeff = torch.rand(1, 4, device=device, dtype=torch.float64)

        assert self.gradcheck(distort_points, (points, K, distCoeff, new_K), raise_exception=True, fast_mode=True)

    def test_jit(self, device, dtype):
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        new_K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(1, 4, device=device, dtype=dtype)
        inputs = (points, K, distCoeff, new_K)

        op = distort_points
        op_jit = torch.jit.script(op)
        self.assert_close(op(*inputs), op_jit(*inputs))
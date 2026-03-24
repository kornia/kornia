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

import functools
from typing import Dict

import pytest
import torch

import kornia
import kornia.geometry.epipolar as epi

from testing.base import BaseTester

SOLVERS = ["svd", "eigh", "cofactor"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scene(device, dtype, num_views: int = 2, num_points: int = 10):
    """Return a consistent synthetic two-view scene in float64 then cast."""
    scene: Dict[str, torch.Tensor] = epi.generate_scene(num_views, num_points)
    return {k: v.to(device=device, dtype=dtype) for k, v in scene.items()}


class TestTriangulation(BaseTester):
    # ------------------------------------------------------------------
    # Smoke — verify all three solvers produce the right output shape
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_smoke(self, solver, device, dtype):
        P1 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        points1 = torch.rand(1, 1, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 1, 2, device=device, dtype=dtype)
        pts3d = epi.triangulate_points(P1, P2, points1, points2, solver=solver)
        assert pts3d.shape == (1, 1, 3)

    # ------------------------------------------------------------------
    # Shape / cardinality
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size, num_points", [(1, 3), (2, 4), (3, 5)])
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_shape(self, batch_size, num_points, solver, device, dtype):
        B, N = batch_size, num_points
        P1 = torch.rand(B, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        points1 = torch.rand(1, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        pts3d = epi.triangulate_points(P1, P2, points1, points2, solver=solver)
        assert pts3d.shape == (B, N, 3)

    # ------------------------------------------------------------------
    # Exception: unknown solver
    # ------------------------------------------------------------------

    def test_exception_unknown_solver(self, device, dtype):
        P1 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        pts = torch.rand(1, 4, 2, device=device, dtype=dtype)
        with pytest.raises(NotImplementedError, match="Unknown solver"):
            epi.triangulate_points(P1, P2, pts, pts, solver="unknown")

    # ------------------------------------------------------------------
    # Two-view accuracy on a noise-free synthetic scene
    # ------------------------------------------------------------------

    def test_two_view(self, device, dtype):
        torch.manual_seed(0)
        num_views: int = 2
        num_points: int = 10
        scene = _make_scene(device, dtype, num_views, num_points)

        P1 = scene["P"][0:1]
        P2 = scene["P"][1:2]
        x1 = scene["points2d"][0:1]
        x2 = scene["points2d"][1:2]

        X = epi.triangulate_points(P1, P2, x1, x2)
        x_reprojected = kornia.geometry.transform_points(scene["P"], X.expand(num_views, -1, -1))

        atol = {torch.float16: 1e-2, torch.bfloat16: 0.25, torch.float32: 1e-4}.get(dtype, 1e-4)
        self.assert_close(scene["points3d"], X, rtol=atol, atol=atol)
        self.assert_close(scene["points2d"], x_reprojected, rtol=atol, atol=atol)

    # ------------------------------------------------------------------
    # All solvers produce collinear results for noise-free data
    # ------------------------------------------------------------------

    def test_solver_consistency(self, device, dtype):
        """All three solvers should agree (up to sign) on noise-free data."""
        torch.manual_seed(0)
        B, N = 2, 20

        # Build a pair of cameras and project known 3-D points.
        R2 = torch.tensor(
            [[0.9998, -0.0175, 0.0], [0.0175, 0.9998, 0.0], [0.0, 0.0, 1.0]],
            device=device, dtype=dtype,
        )
        t2 = torch.tensor([[-0.5], [0.0], [0.0]], device=device, dtype=dtype)

        P1 = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        P2 = torch.cat([R2, t2], dim=-1).unsqueeze(0).expand(B, -1, -1)

        X_true = torch.rand(B, N, 3, device=device, dtype=dtype) + torch.tensor(
            [0.0, 0.0, 3.0], device=device, dtype=dtype
        )

        def project(P, X):
            Xh = torch.cat([X, torch.ones(B, N, 1, device=device, dtype=dtype)], dim=-1)
            px = (P @ Xh.mT).mT
            return px[..., :2] / px[..., 2:3]

        pts1 = project(P1, X_true)
        pts2 = project(P2, X_true)

        results = {s: epi.triangulate_points(P1, P2, pts1, pts2, solver=s) for s in SOLVERS}

        ref = results["svd"]
        for _name, pts in results.items():
            # Check that the recovered direction matches (cosine similarity ≈ 1).
            cos = (pts * ref).sum(-1) / (
                pts.norm(dim=-1).clamp(min=1e-8) * ref.norm(dim=-1).clamp(min=1e-8)
            )
            atol = {torch.float16: 1e-2, torch.bfloat16: 1e-2, torch.float32: 1e-3}.get(dtype, 1e-6)
            self.assert_close(cos.abs(), torch.ones_like(cos), atol=atol, rtol=0.0)

    # ------------------------------------------------------------------
    # Gradcheck — default solver
    # ------------------------------------------------------------------

    def test_gradcheck(self, device):
        points1 = torch.rand(1, 8, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(1, 8, 2, device=device, dtype=torch.float64)
        P1 = kornia.core.ops.eye_like(3, points1)
        P1 = torch.nn.functional.pad(P1, [0, 1])
        P2 = kornia.core.ops.eye_like(3, points2)
        P2 = torch.nn.functional.pad(P2, [0, 1])
        assert self.gradcheck(epi.triangulate_points, (P1, P2, points1, points2), raise_exception=True, fast_mode=True)

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_gradcheck_all_solvers(self, solver, device):
        points1 = torch.rand(1, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(1, 4, 2, device=device, dtype=torch.float64)
        P1 = kornia.core.ops.eye_like(3, points1)
        P1 = torch.nn.functional.pad(P1, [0, 1])
        P2 = kornia.core.ops.eye_like(3, points2)
        P2 = torch.nn.functional.pad(P2, [0, 1])
        fn = functools.partial(epi.triangulate_points, solver=solver)
        assert self.gradcheck(fn, (P1, P2, points1, points2), raise_exception=True, fast_mode=True)

    # ------------------------------------------------------------------
    # Noisy correspondences — compare against OpenCV DLT reference
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_noisy_correspondences_vs_opencv(self, solver, device, dtype):
        """Triangulation from noisy 2-D correspondences should agree with OpenCV's DLT.

        The test uses a simple synthetic two-view setup:
          - Camera 1 at the origin with canonical projection :math:`[I | 0]`.
          - Camera 2 with a pure rightward translation and a small rotation.
          - Ground-truth 3-D points at depth ~5 in front of both cameras.

        Gaussian pixel noise (sigma = 0.5 px) is added to the projected 2-D coordinates
        before triangulation.  The test verifies that the Kornia result agrees with
        ``cv2.triangulatePoints`` to within the expected noise-induced triangulation
        error (~0.3 units at this configuration).

        The test is skipped when OpenCV is not installed.
        """
        cv2 = pytest.importorskip("cv2")

        # Only run on float32 / float64 — half-precision is not a realistic use-case
        # for noisy-correspondence triangulation and cv2 uses float64 internally.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("noisy-correspondence test only runs for float32/float64")

        torch.manual_seed(42)

        N = 50  # number of 3-D points

        # --- Projection matrices ----------------------------------------
        # Camera 1: canonical  [I | 0]
        P1 = torch.eye(3, 4, dtype=dtype, device=device)

        # Camera 2: small rotation + rightward translation of 1 unit
        angle = 0.05  # ~2.9 degrees
        c, s = float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle)))
        R2 = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=dtype, device=device)
        t2 = torch.tensor([[-1.0], [0.0], [0.0]], dtype=dtype, device=device)
        P2 = torch.cat([R2, t2], dim=-1)  # (3, 4)

        # --- 3-D ground-truth points at depth 5 -------------------------
        X_world = torch.rand(N, 3, dtype=dtype, device=device)
        X_world[..., 2] = X_world[..., 2] + 4.5  # depth in [4.5, 5.5]

        # --- Project and add Gaussian noise (sigma = 0.5 px) ----------------
        def project(P: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            """(3,4) x (N,3) → (N,2) normalised image coords."""
            Xh = torch.cat([X, torch.ones(N, 1, dtype=dtype, device=device)], dim=-1)  # (N,4)
            px = (P @ Xh.mT).mT  # (N,3)
            return px[..., :2] / px[..., 2:3]

        noise_sigma = 0.5
        pts1_clean = project(P1, X_world)
        pts2_clean = project(P2, X_world)
        pts1 = pts1_clean + noise_sigma * torch.randn_like(pts1_clean)
        pts2 = pts2_clean + noise_sigma * torch.randn_like(pts2_clean)

        # --- Triangulate with Kornia (add batch dimension) ---------------
        X_kornia = epi.triangulate_points(
            P1.unsqueeze(0), P2.unsqueeze(0),
            pts1.unsqueeze(0), pts2.unsqueeze(0),
            solver=solver,
        ).squeeze(0)  # (N, 3)

        # --- Triangulate with OpenCV (reference) -------------------------
        # cv2.triangulatePoints expects float64 2xN arrays and 3x4 matrices.
        P1_np = P1.cpu().to(torch.float64).numpy()
        P2_np = P2.cpu().to(torch.float64).numpy()
        pts1_np = pts1.cpu().to(torch.float64).numpy().T  # (2, N)
        pts2_np = pts2.cpu().to(torch.float64).numpy().T  # (2, N)

        X_cv_h = cv2.triangulatePoints(P1_np, P2_np, pts1_np, pts2_np)  # (4, N)
        X_cv = (X_cv_h[:3] / X_cv_h[3:4]).T  # (N, 3)
        X_cv_t = torch.tensor(X_cv, dtype=dtype, device=device)

        # --- Both results must agree within noise-induced triangulation error ---
        # At depth~5 and baseline~1, sigma=0.5 px noise gives ~0.3 unit 3-D error.
        atol = 0.5
        self.assert_close(X_kornia, X_cv_t, atol=atol, rtol=0.0)

    # ------------------------------------------------------------------
    # Module-level import check
    # ------------------------------------------------------------------

    def test_module(self, device, dtype):
        assert hasattr(epi, "triangulate_points")
        P1 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        pts = torch.rand(1, 3, 2, device=device, dtype=dtype)
        out = epi.triangulate_points(P1, P2, pts, pts)
        assert out.shape == (1, 3, 3)

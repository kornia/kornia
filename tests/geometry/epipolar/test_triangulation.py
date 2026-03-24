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

    @pytest.mark.parametrize("solver", ["svd", "eigh"])
    def test_noisy_correspondences_dlt(self, solver, device, dtype):
        """Triangulation from noisy correspondences matches the numpy DLT reference.

        Two-view setup: camera 1 at [I|0], camera 2 with ~5.7-degree rotation and
        1-unit rightward translation, 8 points at depth 3-4, Gaussian noise sigma=0.05.

        Expected output was pre-computed once with a point-by-point numpy SVD
        implementation identical to ``cv2.triangulatePoints`` (seed 7).

        # Snippet used to generate X_expected (requires numpy only):
        # import numpy as np, torch
        # torch.manual_seed(7)
        # ... (see test body for the full scene construction)
        # for i in range(N):
        #     A = np.array([pts1[0,i]*P1[2]-P1[0], pts1[1,i]*P1[2]-P1[1],
        #                   pts2[0,i]*P2[2]-P2[0], pts2[1,i]*P2[2]-P2[1]])
        #     _, _, V = np.linalg.svd(A)
        #     X_expected[i] = V[-1, :3] / V[-1, 3]
        """
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("noisy-correspondence test only runs for float32/float64")

        # Hardcoded inputs (torch.manual_seed(7), noise sigma=0.05)
        P1 = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=dtype, device=device,
        )
        P2 = torch.tensor(
            [[0.9950041770935059, 0.0, 0.0998334214091301, -1.0],
             [0.0, 1.0, 0.0, 0.0],
             [-0.0998334214091301, 0.0, 0.9950041770935059, 0.0]],
            dtype=dtype, device=device,
        )
        pts1 = torch.tensor(
            [[-0.024943894271020096, 0.023484865267389687],
             [0.0284050326736405, 0.22463705371976522],
             [0.05363684307356541, 0.1995784905718564],
             [0.1202179211117687, -0.002082513366088421],
             [-0.05603666748438946, -0.07320451037059378],
             [0.2234642952593915, 0.03488261645314251],
             [0.04132917198453165, 0.19063330422731778],
             [0.07586561872234336, 0.12222851867867218]],
            dtype=dtype, device=device,
        )
        pts2 = torch.tensor(
            [[-0.2005761556307751, 0.08063633664743011],
             [-0.05218266808824606, 0.09602012894912612],
             [-0.14514520805453554, 0.11014017584343802],
             [-0.14273667377657795, 0.051925094545445555],
             [-0.2219682806714787, -0.09313464409688944],
             [-0.061311792967568134, 0.04181839051261021],
             [-0.1400088642080817, 0.12810430456996066],
             [-0.16590968636404868, 0.08471239170067128]],
            dtype=dtype, device=device,
        )
        # Expected: numpy DLT (same algorithm as cv2.triangulatePoints)
        X_expected = torch.tensor(
            [[-0.08818743120343768, 0.188334626330844, 3.6220817756133172],
             [0.16144758240818913, 0.8807090389866558, 5.4844829932547015],
             [0.18216324334542242, 0.517464929751309, 3.3477856439676588],
             [0.3350949547897918, 0.06786420490996858, 2.770735605435314],
             [-0.20989477003241103, -0.3123833872441908, 3.754403690255068],
             [0.5852768065399436, 0.0989402354921456, 2.6188660940360844],
             [0.1480216292032729, 0.5668535581804692, 3.5667031501090993],
             [0.22379169128122922, 0.3033298174456883, 2.9457656267879586]],
            dtype=dtype, device=device,
        )

        X_kornia = epi.triangulate_points(
            P1.unsqueeze(0), P2.unsqueeze(0),
            pts1.unsqueeze(0), pts2.unsqueeze(0),
            solver=solver,
        ).squeeze(0)

        atol = 1e-4 if dtype == torch.float64 else 1e-3
        self.assert_close(X_kornia, X_expected, atol=atol, rtol=0.0)

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

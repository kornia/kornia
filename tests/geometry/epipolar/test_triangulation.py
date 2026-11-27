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
import math
from typing import Dict

import pytest
import torch

import kornia
import kornia.geometry.epipolar as epi
from kornia.geometry.conversions import axis_angle_to_rotation_matrix

from testing.base import BaseTester
from testing.two_view import two_view_scene

SOLVERS = ["svd", "eigh", "cofactor"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scene(device, dtype, num_views: int = 2, num_points: int = 10):
    """Return a consistent synthetic two-view scene in float64 then cast."""
    scene: Dict[str, torch.Tensor] = epi.generate_scene(num_views, num_points)
    return {k: v.to(device=device, dtype=dtype) for k, v in scene.items()}


def _project(P: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Project (B, N, 3) points through a (B, 3, 4) camera matrix → (B, N, 2)."""
    B, N = X.shape[:2]
    Xh = torch.cat([X, torch.ones(B, N, 1, device=X.device, dtype=X.dtype)], dim=-1)
    px = (P @ Xh.mT).mT
    return px[..., :2] / px[..., 2:3]


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
            device=device,
            dtype=dtype,
        )
        t2 = torch.tensor([[-0.5], [0.0], [0.0]], device=device, dtype=dtype)

        P1 = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        P2 = torch.cat([R2, t2], dim=-1).unsqueeze(0).expand(B, -1, -1)

        X_true = torch.rand(B, N, 3, device=device, dtype=dtype) + torch.tensor(
            [0.0, 0.0, 3.0], device=device, dtype=dtype
        )

        pts1 = _project(P1, X_true)
        pts2 = _project(P2, X_true)

        results = {s: epi.triangulate_points(P1, P2, pts1, pts2, solver=s) for s in SOLVERS}

        ref = results["svd"]
        for _name, pts in results.items():
            # Check that the recovered direction matches (cosine similarity ≈ 1).
            cos = (pts * ref).sum(-1) / (pts.norm(dim=-1).clamp(min=1e-8) * ref.norm(dim=-1).clamp(min=1e-8))
            atol = {torch.float16: 1e-2, torch.bfloat16: 1e-2, torch.float32: 1e-3}.get(dtype, 1e-6)
            self.assert_close(cos.abs(), torch.ones_like(cos), atol=atol, rtol=0.0)

    # ------------------------------------------------------------------
    # Cofactor sign-alignment regression
    # ------------------------------------------------------------------

    @pytest.mark.applies_to(dtypes=("float32", "float64"), reason="cofactor sign test only runs for float32/float64")
    def test_cofactor_sign_alignment(self, device, dtype):
        """Cofactor solver must not produce NaN/Inf due to sign cancellation.

        Before the sign-alignment fix, the two 3x4 sub-systems could yield
        opposite-signed null vectors whose sum cancelled to ~0, producing NaN
        after dehomogenisation.  This test constructs a noise-free two-view
        scene with a modest baseline, triangulates with both the cofactor and
        SVD solvers, and checks that the cofactor output is finite and
        directionally consistent with the SVD result.
        """

        torch.manual_seed(3)
        B, N = 1, 8

        P1 = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        R2 = torch.tensor(
            [[0.9998, -0.0175, 0.0], [0.0175, 0.9998, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        t2 = torch.tensor([[-1.0], [0.0], [0.0]], device=device, dtype=dtype)
        P2 = torch.cat([R2, t2], dim=-1).unsqueeze(0).expand(B, -1, -1)

        X_true = torch.rand(B, N, 3, device=device, dtype=dtype) + torch.tensor(
            [0.0, 0.0, 3.0], device=device, dtype=dtype
        )

        pts1 = _project(P1, X_true)
        pts2 = _project(P2, X_true)

        X_cofactor = epi.triangulate_points(P1, P2, pts1, pts2, solver="cofactor")
        X_svd = epi.triangulate_points(P1, P2, pts1, pts2, solver="svd")

        # Output must be finite — NaN would indicate the pre-fix cancellation bug.
        assert torch.isfinite(X_cofactor).all(), "cofactor solver produced non-finite values"

        # Numerical closeness to SVD (noise-free → both solvers recover the same 3-D point).
        # This also catches incorrect scale/depth, unlike a direction-only check.
        atol = 1e-3 if dtype == torch.float32 else 1e-6
        self.assert_close(X_cofactor, X_svd, atol=atol, rtol=0.0)

    # ------------------------------------------------------------------
    # Gradcheck — default solver
    # ------------------------------------------------------------------

    def test_gradcheck(self, device):
        P1 = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        P2 = torch.tensor(
            [
                [
                    [0.98, 0.0, 0.198997487421324, -1.0],
                    [0.0, 1.0, 0.0, 0.1],
                    [-0.198997487421324, 0.0, 0.98, 0.2],
                ]
            ],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        points1 = torch.tensor(
            [[[0.05, -0.03], [-0.08, 0.12], [0.15, 0.07], [-0.11, -0.09]]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        points2 = torch.tensor(
            [[[-0.09, -0.01], [-0.19, 0.14], [0.03, 0.09], [-0.24, -0.06]]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        assert self.gradcheck(epi.triangulate_points, (P1, P2, points1, points2), raise_exception=True, fast_mode=True)

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_gradcheck_all_solvers(self, solver, device):
        P1 = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        P2 = torch.tensor(
            [
                [
                    [0.98, 0.0, 0.198997487421324, -1.0],
                    [0.0, 1.0, 0.0, 0.1],
                    [-0.198997487421324, 0.0, 0.98, 0.2],
                ]
            ],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        points1 = torch.tensor(
            [[[0.05, -0.03], [-0.08, 0.12], [0.15, 0.07], [-0.11, -0.09]]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        points2 = torch.tensor(
            [[[-0.09, -0.01], [-0.19, 0.14], [0.03, 0.09], [-0.24, -0.06]]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        fn = functools.partial(epi.triangulate_points, solver=solver)
        assert self.gradcheck(fn, (P1, P2, points1, points2), raise_exception=True, fast_mode=True)

    @pytest.mark.parametrize("solver", ["svd", "eigh"])
    def test_gradcheck_zero_baseline_points(self, solver, device):
        # At an exact zero baseline the camera-centre normalization is discontinuous wrt P1/P2, while
        # gradients wrt the image correspondences remain smooth.
        P1 = torch.eye(3, 4, device=device, dtype=torch.float64)[None]
        P2 = torch.tensor(
            [[[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            device=device,
            dtype=torch.float64,
        )
        points1 = torch.tensor(
            [[[0.05, -0.03], [-0.08, 0.12], [0.15, 0.07], [-0.11, -0.09]]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        points2 = torch.tensor(
            [[[-0.09, -0.01], [-0.19, 0.14], [0.03, 0.09], [-0.24, -0.06]]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        fn = functools.partial(epi.triangulate_points, solver=solver)
        assert self.gradcheck(
            fn,
            (P1, P2, points1, points2),
            requires_grad=[False, False, True, True],
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.applies_to(dtypes=("float32", "float64"), reason="backward stability on degenerate poses")
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_degenerate_camera_backward_finite(self, solver, device, dtype):
        pts1 = torch.tensor(
            [[[0.05, -0.03], [-0.08, 0.12], [0.15, 0.07]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        pts2 = torch.tensor(
            [[[-0.09, -0.01], [-0.19, 0.14], [0.03, 0.09]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        roll = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        for C_val in ([1.0, 2.0, -3.0], [2001.0, -1998.0, 2003.0]):
            C = torch.tensor([[C_val]], device=device, dtype=dtype).transpose(-2, -1)
            P1_zb = torch.cat([eye, -C], dim=-1).detach().clone().requires_grad_(True)
            P2_zb = torch.cat([roll, -roll @ C], dim=-1).detach().clone().requires_grad_(True)
            p1_in = pts1.detach().clone().requires_grad_(True)
            p2_in = pts2.detach().clone().requires_grad_(True)
            out = epi.triangulate_points(P1_zb, P2_zb, p1_in, p2_in, solver=solver)
            valid = out[~out.isnan().any(-1)]
            if valid.numel() > 0:
                valid.sum().backward()
                for t in (P1_zb, P2_zb, p1_in, p2_in):
                    assert t.grad is None or torch.isfinite(t.grad).all()

        P_aff1_u = torch.tensor(
            [[[1.0, 0.0, 0.0, -10.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        P_aff2_u = torch.tensor(
            [[[0.0, 0.0, 1.0, -20.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        P_aff2 = torch.tensor(
            [[[0.0, 0.0, 500.0, -100000.0], [0.0, 500.0, 0.0, 25000.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        P_persp = torch.tensor(
            [[[800.0, 0.0, 320.0, -79360.0], [0.0, 800.0, 240.0, 41200.0], [0.0, 0.0, 1.0, -195.0]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        xa1_u = torch.tensor([[[1.5, 0.8], [-0.2, -0.4]]], device=device, dtype=dtype, requires_grad=True)
        xa2_u = torch.tensor([[[2.0, 0.8], [-3.0, -0.4]]], device=device, dtype=dtype, requires_grad=True)
        xa2 = torch.tensor([[[2500.0, 400.0], [-1500.0, -200.0]]], device=device, dtype=dtype, requires_grad=True)
        xp = torch.tensor([[[396.0, 272.0], [312.0, 210.0]]], device=device, dtype=dtype, requires_grad=True)
        for p_a, p_b, q_a, q_b in ((P_aff1_u, P_aff2_u, xa1_u, xa2_u), (P_persp, P_aff2, xp, xa2)):
            for t in (p_a, p_b, q_a, q_b):
                if t.grad is not None:
                    t.grad.zero_()
            out_m = epi.triangulate_points(p_a, p_b, q_a, q_b, solver=solver)
            out_m.sum().backward()
            for t in (p_a, p_b, q_a, q_b):
                assert t.grad is not None
                assert torch.isfinite(t.grad).all()

        # Parallel rays produce NaN in one row; masking invalid rows must keep finite gradients for valid rows/cameras.
        P1_p = torch.tensor(
            [[[1.0, 0.0, 0.0, -10.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, -20.0]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        P2_p = torch.tensor(
            [[[1.0, 0.0, 0.0, -11.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, -20.0]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        q1_p = torch.tensor([[[0.25, -0.15], [0.2, 0.1]]], device=device, dtype=dtype, requires_grad=True)
        q2_p = torch.tensor([[[0.25, -0.15], [-0.05, 0.1]]], device=device, dtype=dtype, requires_grad=True)
        out_p = epi.triangulate_points(P1_p, P2_p, q1_p, q2_p, solver=solver)
        assert torch.isnan(out_p[:, 0]).all()
        valid_p = out_p[~out_p.isnan().any(-1)]
        assert valid_p.shape[0] == 1
        valid_p.sum().backward()
        for t in (P1_p, P2_p, q1_p, q2_p):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_mixed_input_dtypes(self, solver, device):
        P1 = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            device=device,
            dtype=torch.float32,
        )
        P2 = torch.tensor(
            [
                [
                    [0.98, 0.0, 0.198997487421324, -1.0],
                    [0.0, 1.0, 0.0, 0.1],
                    [-0.198997487421324, 0.0, 0.98, 0.2],
                ]
            ],
            device=device,
            dtype=torch.float32,
        )
        pts1 = torch.tensor([[[0.05, -0.03], [-0.08, 0.12]]], device=device, dtype=torch.float32)
        pts2 = torch.tensor([[[-0.09, -0.01], [-0.19, 0.14]]], device=device, dtype=torch.float32)
        ref = epi.triangulate_points(P1, P2, pts1, pts2, solver=solver)

        out_half_pts32 = epi.triangulate_points(P1.half(), P2.half(), pts1, pts2, solver=solver)
        assert out_half_pts32.dtype == torch.float32
        self.assert_close(out_half_pts32, ref, atol=5e-3, rtol=0.0)

        if device.type != "mps":
            out_f32_pts64 = epi.triangulate_points(P1, P2, pts1.double(), pts2.double(), solver=solver)
            assert out_f32_pts64.dtype == torch.float64
            self.assert_close(out_f32_pts64, ref.double(), atol=1e-6, rtol=0.0)

    # ------------------------------------------------------------------
    # Noisy correspondences — compare against OpenCV DLT reference
    # ------------------------------------------------------------------

    @pytest.mark.applies_to(
        dtypes=("float32", "float64"), reason="noisy-correspondence test only runs for float32/float64"
    )
    @pytest.mark.parametrize("solver", ["svd", "eigh"])
    def test_noisy_correspondences_dlt(self, solver, device, dtype):
        """Triangulation from noisy correspondences matches the camera-midpoint numpy DLT reference.

        Two-view setup: camera 1 at [I|0], camera 2 with ~5.7-degree rotation and
        1-unit rightward translation, 8 points at depth 3-4, Gaussian noise sigma=0.05.

        Expected output was pre-computed independently in numpy by centering at the camera-centre
        midpoint and scaling by the baseline before solving point-by-point DLT with SVD (seed 7).

        # Snippet used to generate X_expected (requires numpy only):
        # import numpy as np
        # C1 = -np.linalg.solve(P1[:3, :3], P1[:3, 3:4])
        # C2 = -np.linalg.solve(P2[:3, :3], P2[:3, 3:4])
        # c = 0.5 * (C1 + C2)
        # s = float(np.linalg.norm(C2 - C1))
        # P1_loc = np.concatenate([P1[:3, :3], (P1[:3, :3] @ c + P1[:3, 3:4]) / s], axis=-1)
        # P2_loc = np.concatenate([P2[:3, :3], (P2[:3, :3] @ c + P2[:3, 3:4]) / s], axis=-1)
        # for i in range(N):
        #     A = np.array([pts1[i, 0]*P1_loc[2]-P1_loc[0], pts1[i, 1]*P1_loc[2]-P1_loc[1],
        #                   pts2[i, 0]*P2_loc[2]-P2_loc[0], pts2[i, 1]*P2_loc[2]-P2_loc[1]])
        #     _, _, Vt = np.linalg.svd(A)
        #     X_expected[i] = s * (Vt[-1, :3] / Vt[-1, 3]) + c[:, 0]
        """

        # Hardcoded inputs (torch.manual_seed(7), noise sigma=0.05)
        P1 = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        P2 = torch.tensor(
            [
                [0.9950041770935059, 0.0, 0.0998334214091301, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0998334214091301, 0.0, 0.9950041770935059, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        pts1 = torch.tensor(
            [
                [-0.024943894271020096, 0.023484865267389687],
                [0.0284050326736405, 0.22463705371976522],
                [0.05363684307356541, 0.1995784905718564],
                [0.1202179211117687, -0.002082513366088421],
                [-0.05603666748438946, -0.07320451037059378],
                [0.2234642952593915, 0.03488261645314251],
                [0.04132917198453165, 0.19063330422731778],
                [0.07586561872234336, 0.12222851867867218],
            ],
            dtype=dtype,
            device=device,
        )
        pts2 = torch.tensor(
            [
                [-0.2005761556307751, 0.08063633664743011],
                [-0.05218266808824606, 0.09602012894912612],
                [-0.14514520805453554, 0.11014017584343802],
                [-0.14273667377657795, 0.051925094545445555],
                [-0.2219682806714787, -0.09313464409688944],
                [-0.061311792967568134, 0.04181839051261021],
                [-0.1400088642080817, 0.12810430456996066],
                [-0.16590968636404868, 0.08471239170067128],
            ],
            dtype=dtype,
            device=device,
        )
        # Expected: independent numpy midpoint-normalized DLT
        X_expected = torch.tensor(
            [
                [-0.08880389658945637, 0.18841153670999625, 3.62351131790014],
                [0.15757177732838928, 0.885600174897823, 5.514482304342408],
                [0.18066248631664222, 0.5183937975066767, 3.3535967578367316],
                [0.33467099874663286, 0.06791158161820093, 2.7725291146801854],
                [-0.20996360737018555, -0.31239319362076123, 3.754519968575197],
                [0.5852741626823669, 0.09894127466277042, 2.6188917064690562],
                [0.14724987297728892, 0.5673514694486397, 3.5697398583427917],
                [0.2235387840429543, 0.30342518833918153, 2.9466516048459175],
            ],
            dtype=dtype,
            device=device,
        )

        X_kornia = epi.triangulate_points(
            P1.unsqueeze(0),
            P2.unsqueeze(0),
            pts1.unsqueeze(0),
            pts2.unsqueeze(0),
            solver=solver,
        ).squeeze(0)

        atol = 1e-4 if dtype == torch.float64 else 1e-3
        self.assert_close(X_kornia, X_expected, atol=atol, rtol=0.0)

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_a_mixed_precision_region_does_not_reach_the_solver(self, solver, device):
        # A caller's autocast region ran the solvers' products in half precision: cofactor lost accuracy, and eigh
        # raised on MPS, which has no half-precision eigh (#6174). No "autocast" in the name: conftest skips those on
        # MPS.
        if device.type not in ("cpu", "cuda", "mps"):
            pytest.skip("CPU, CUDA and MPS autocast")
        # MPS autocasts to float16, and so does a CUDA device without bfloat16, where bfloat16 autocast raises.
        bfloat16_ok = device.type == "cpu" or (device.type == "cuda" and torch.cuda.is_bf16_supported())
        scene = _make_scene(device, torch.float32)
        args = (scene["P"][0:1], scene["P"][1:2], scene["points2d"][0:1], scene["points2d"][1:2])
        expected = epi.triangulate_points(*args, solver=solver)
        with torch.autocast(device.type, dtype=torch.bfloat16 if bfloat16_ok else torch.float16):
            actual = epi.triangulate_points(*args, solver=solver)
        assert actual.dtype == torch.float32
        self.assert_close(actual, expected, atol=0, rtol=0)

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


def _dehom(x: torch.Tensor) -> torch.Tensor:
    return x[..., :2] / x[..., 2:]


# Error bound for svd/eigh on the exact two-view fixture: float16/bfloat16 build the DLT rows in the input dtype.
_TRIANGULATION_ATOL = {torch.float16: 5e-2, torch.bfloat16: 0.5, torch.float32: 1e-3, torch.float64: 1e-9}
# Bound on the distance from the line of sight, relative to |X|, at zero baseline (the rows carry the input dtype's
# roundoff; a point off the line, such as a roundoff direction normalised to unit length, is at 9 or more).
_OFF_RAY_TOL = {torch.float16: 2e-2, torch.bfloat16: 0.3, torch.float32: 1e-2, torch.float64: 1e-2}
# Distance, in the scale of t (|t| = 0.5), at which a point still reads as finite: the rows' roundoff puts a point
# beyond a few tens of units at infinity in half precision.
_FAR_BUT_FINITE = {torch.float16: 20.0, torch.bfloat16: 8.0, torch.float32: 1e4, torch.float64: 1e6}


class TestConventionTriangulation(BaseTester):
    def test_convention_triangulate_points_argument_pairing(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        P1, P2, x1, x2, X = two_view["P1"], two_view["P2"], two_view["x1"], two_view["x2"], two_view["X"]
        atol = _TRIANGULATION_ATOL[dtype]
        results = {}
        for solver in ("svd", "eigh"):
            # P1 pairs with points1 and P2 with points2; the output is Euclidean (B, N, 3) in the input dtype.
            out = epi.triangulate_points(P1, P2, x1, x2, solver=solver)
            assert out.shape == (1, 12, 3)
            assert out.dtype == dtype
            self.assert_close(out, X, rtol=0.0, atol=atol)
            # Relabelling: swapping the two views as a whole recovers the same points; swapping only the cameras
            # or only the points misplaces every point, by more than 1.0.
            self.assert_close(epi.triangulate_points(P2, P1, x2, x1, solver=solver), X, rtol=0.0, atol=atol)
            for wrong in (
                epi.triangulate_points(P2, P1, x1, x2, solver=solver),
                epi.triangulate_points(P1, P2, x2, x1, solver=solver),
            ):
                assert (wrong - X).norm(dim=-1).min() > 1.0
            results[solver] = out
        # svd and eigh agree to roundoff on this well-conditioned fixture.
        self.assert_close(results["svd"], results["eigh"], rtol=0.0, atol=atol)

    def test_convention_triangulate_points_unchecked_cheirality_and_baseline(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        P1, P2, R, t, X = (two_view[k] for k in ("P1", "P2", "R", "t", "X"))

        def project(P: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
            return _dehom(torch.cat([Y, torch.ones_like(Y[..., :1])], -1) @ P.transpose(-2, -1))

        for solver in SOLVERS:
            # A point behind both cameras is returned, not rejected: its depth is negative in both. The fixture
            # points, in front, are the control.
            front = epi.triangulate_points(P1, P2, two_view["x1"], two_view["x2"], solver=solver)
            assert (front[..., 2] > 0).all()
            assert (epi.depth_from_point(R, t, front) > 0).all()
            behind = epi.triangulate_points(P1, P2, project(P1, -X), project(P2, -X), solver=solver)
            assert (behind[..., 2] < 0).all()
            assert (epi.depth_from_point(R, t, behind) < 0).all()
        # An exact shared camera centre C in every dtype. The second camera rolls 90 degrees:
        # (u, v) goes to (-v, u). Rounded rotations or independent projection products can introduce
        # an unintended baseline or inconsistent rays in half precision and make the null space 1-D.
        C = torch.tensor([1.0, 2.0, -3.0], device=device, dtype=dtype)
        P1_C = torch.tensor([[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, 3]], device=device, dtype=dtype)
        P2_C = torch.stack([-P1_C[1], P1_C[0], P1_C[2]])
        x1 = (
            torch.tensor(
                [[-4, 1], [-3, 2], [-2, -1], [-1, 3], [1, -3], [2, 1], [3, -2], [4, 3]], device=device, dtype=dtype
            )
            / 8
        )
        x2 = torch.stack([-x1[..., 1], x1[..., 0]], dim=-1)
        rays = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1).cpu().double()
        for solver in ("svd", "eigh"):
            out = epi.triangulate_points(P1_C, P2_C, x1, x2, solver=solver).cpu().double()
            finite = torch.isfinite(out).all(-1)
            assert finite.sum() >= 6  # the line-of-sight check below is not vacuous
            v, direction = out[finite] - C.cpu().double(), rays[finite]
            assert (
                torch.linalg.cross(v, direction, dim=-1).norm(dim=-1) / direction.norm(dim=-1)
            ).max() <= _OFF_RAY_TOL[dtype]
            assert (v.norm(dim=-1) > 1e-3).any()
        assert torch.isnan(epi.triangulate_points(P1_C, P2_C, x1, x2, solver="cofactor")).all()

    def test_convention_triangulate_points_infinity_is_nan(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        # A correspondence at infinity (the images of a direction (x, y, z, 0)) has a homogeneous w at roundoff
        # and comes back as NaN, in every solver and dtype; it used to be a finite point at a roundoff-set
        # distance. The fixture points, at depth 4 to 6, and points a few tens of units away are the control.
        P1, P2, d = two_view["P1"], two_view["P2"], two_view["X"]  # the fixture points read as directions
        x1 = _dehom(d @ P1[..., :3].transpose(-2, -1))
        x2 = _dehom(d @ P2[..., :3].transpose(-2, -1))
        far = _FAR_BUT_FINITE[dtype] * d / d.norm(dim=-1, keepdim=True)
        far_h = torch.cat([far, torch.ones_like(far[..., :1])], -1)
        x1_far, x2_far = _dehom(far_h @ P1.transpose(-2, -1)), _dehom(far_h @ P2.transpose(-2, -1))
        for solver in SOLVERS:
            assert torch.isnan(epi.triangulate_points(P1, P2, x1, x2, solver=solver)).all()
            assert torch.isfinite(epi.triangulate_points(P1, P2, two_view["x1"], two_view["x2"], solver=solver)).all()
            assert torch.isfinite(epi.triangulate_points(P1, P2, x1_far, x2_far, solver=solver)).all()

    @pytest.mark.applies_to(
        dtypes=("float32", "float64"), reason="the gradient check of the NaN mask runs in float32 and float64"
    )
    def test_convention_triangulate_points_masked_infinity_keeps_gradients_finite(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        # A point at infinity is NaN, but masking its row before the loss leaves every gradient finite, including
        # the cameras' gradients, which sum over all rows: the NaN is substituted, not produced by the solve.
        P1, P2, d = two_view["P1"], two_view["P2"], two_view["X"]
        x1_inf = _dehom(d[..., :1, :] @ P1[..., :3].transpose(-2, -1))
        x2_inf = _dehom(d[..., :1, :] @ P2[..., :3].transpose(-2, -1))
        for solver in SOLVERS:
            p1, p2 = P1.clone().requires_grad_(), P2.clone().requires_grad_()
            x1 = torch.cat([two_view["x1"], x1_inf], -2).requires_grad_()
            x2 = torch.cat([two_view["x2"], x2_inf], -2).requires_grad_()
            out = epi.triangulate_points(p1, p2, x1, x2, solver=solver)
            finite = ~out.isnan().any(-1)
            assert not finite[..., -1].any()
            assert finite[..., :-1].all()
            for grad in torch.autograd.grad(out[finite].sum(), (p1, p2, x1, x2)):
                assert torch.isfinite(grad).all()

    @pytest.mark.applies_to(
        dtypes=("float16", "bfloat16"), reason="the half-precision overflow of the unnormalised cofactor null vector"
    )
    def test_convention_triangulate_cofactor_half_precision_is_finite(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        # The cofactor null vector of pixel-scale rows is normalised in float32 before the cast back to the input
        # dtype; cast first, it overflowed float16 and every point was NaN. bfloat16 keeps float32's exponent range
        # but only 8 bits of mantissa, hence its looser bound.
        P1, P2, x1, x2, X = (two_view[k] for k in ("P1", "P2", "x1", "x2", "X"))
        out = epi.triangulate_points(P1, P2, x1, x2, solver="cofactor")
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
        self.assert_close(out, X, rtol=0.0, atol={torch.float16: 5e-2, torch.bfloat16: 1.0}[dtype])

    @pytest.mark.applies_to(dtypes=("float16",), reason="overflow of a float16 product before subtraction")
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_triangulate_points_avoids_float16_row_overflow(self, solver, device, dtype):
        # Rectified stereo with focal length 384, principal point (320, 200), and baseline 0.5.
        # The two projections are u_left = 320 + 384*x/z, u_right = 320 + 384*(x-0.5)/z, v = 200 + 384*y/z.
        projection = torch.tensor([[384, 0, 320, 0], [0, 384, 200, 0], [0, 0, 1, 0]], device=device, dtype=dtype)
        right = projection.clone()
        right[0, 3] = -192
        left_pixels = torch.tensor([[344, 152], [576, 264]], device=device, dtype=dtype)
        right_pixels = torch.tensor([[296, 152], [512, 264]], device=device, dtype=dtype)
        expected = torch.tensor([[0.25, -0.5, 4.0], [2.0, 0.5, 3.0]], device=device, dtype=dtype)
        # Rescaling both matrices by 128 preserves the reconstruction. For the second point, 576 * 128
        # overflows float16 before subtraction, even though the resulting coefficient is representable.
        result = epi.triangulate_points(128 * projection, 128 * right, left_pixels, right_pixels, solver=solver)
        self.assert_close(result, expected)

    @pytest.mark.applies_to(dtypes=("bfloat16",), reason="roundoff of a bfloat16 product before subtraction")
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_triangulate_points_avoids_bfloat16_row_roundoff(self, solver, device, dtype):
        # Rectified cameras with focal length 512, baseline 1 and camera depth z + 13.
        # From disparity 16, camera depth is 512/16 = 32. With principal point (512, 256),
        # x = (668 - 512)/16 = 9.75 and y = (284 - 256)/16 = 1.75.
        left = torch.tensor([[512, 0, 512, 6656], [0, 512, 256, 3328], [0, 0, 1, 13]], device=device, dtype=dtype)
        right = left.clone()
        right[0, 3] = 6144
        left_pixels = torch.tensor([[668, 284]], device=device, dtype=dtype)
        right_pixels = torch.tensor([[652, 284]], device=device, dtype=dtype)
        expected = torch.tensor([[9.75, 1.75, 19.0]], device=device, dtype=dtype)
        self.assert_close(epi.triangulate_points(left, right, left_pixels, right_pixels, solver=solver), expected)

    @pytest.mark.applies_to(dtypes=("float16",), reason="float16 preserves the solution at an offset world origin")
    def test_triangulate_eigh_preserves_half_solution_at_offset_world_origin(self, device, dtype):
        # Focal length 512, principal point (512, 208), baseline 1. A disparity of 8 gives camera
        # depth 64 and world z = 64 - 80 = -16. World x and y are the principal-point offsets divided by 8.
        left = torch.tensor([[512, 0, 512, 40960], [0, 512, 208, 16640], [0, 0, 1, 80]], device=device, dtype=dtype)
        right = left.clone()
        right[0, 3] = 40448
        left_pixels = torch.tensor([[704, 264], [576, 232]], device=device, dtype=dtype)
        right_pixels = torch.tensor([[696, 264], [568, 232]], device=device, dtype=dtype)
        expected = torch.tensor([[24.0, 7.0, -16.0], [8.0, 3.0, -16.0]], device=device, dtype=dtype)
        self.assert_close(epi.triangulate_points(left, right, left_pixels, right_pixels), expected)

    @pytest.mark.applies_to(
        dtypes=("float32", "float64"), reason="in half precision the rows' roundoff leaves the sub-systems full rank"
    )
    def test_convention_triangulate_cofactor_rank_deficient_sub_system(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        # A 3x4 sub-system of the DLT matrix that is rank-deficient, or nearly so with noise, has a null vector set
        # by roundoff or noise; cofactor used to average a fixed pair of them after normalising (#4900). It now
        # refines the longest sub-system null vector through the adjugate and returns the svd point: NaN when every
        # sub-system is rank-deficient (zero baseline), and svd's point for a rectified, vertical or rolled pair.
        K1, K2, R, X = two_view["K1"], two_view["K2"], two_view["R"], two_view["X"]
        eye = torch.eye(3, device=device, dtype=dtype)[None]

        def project(P: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
            return _dehom(torch.cat([Y, torch.ones_like(Y[..., :1])], -1) @ P.transpose(-2, -1))

        # Zero baseline: both cameras at C. The answer is any point on the ray from C in the direction X.
        C = torch.tensor([[[1.0], [2.0], [-3.0]]], device=device, dtype=dtype)
        P1, P2 = epi.projection_from_KRt(K1, eye, -C), epi.projection_from_KRt(K2, R, -R @ C)
        x1, x2 = project(P1, X + C.transpose(-2, -1)), project(P2, X + C.transpose(-2, -1))
        X64 = X.cpu().double()  # float64 on the CPU: MPS has no float64
        v = (epi.triangulate_points(P1, P2, x1, x2, solver="svd") - C.transpose(-2, -1)).cpu().double()
        assert (torch.linalg.cross(v, X64, dim=-1).norm(dim=-1) / X64.norm(dim=-1)).max() <= 1e-2
        assert torch.isnan(epi.triangulate_points(P1, P2, x1, x2, solver="cofactor")).all()
        # One K, R = I and a pure x (rectified) or y (vertical) translation: DLT rows 1 and 3 (0 and 2) coincide
        # without noise, so sub-system {1, 2, 3} ({0, 1, 2}) is rank-deficient. Camera 2 rolled by 90 degrees
        # about its axis with an x baseline makes rows 1 and 2 coincide, so {0, 1, 2} and {1, 2, 3} both are.
        # With 0.5 px of noise the rows stay nearly dependent; the old average was 5 to 6 units off on every
        # point of all three pairs and NaN on every point of the exact rolled pair.
        roll = axis_angle_to_rotation_matrix(torch.tensor([[0.0, 0.0, math.pi / 2]], device=device, dtype=dtype))
        b = 0.5 * eye[:, :, :1]
        pairs = {"rectified": (eye, -b), "vertical": (eye, -0.5 * eye[:, :, 1:2]), "rolled": (roll, -roll @ b)}
        g = torch.Generator().manual_seed(0)
        noise = [0.5 * torch.randn(X.shape[:-1] + (2,), generator=g, dtype=torch.float64) for _ in range(2)]
        n1, n2 = (n.to(device, dtype) for n in noise)
        for name, (R2, t2) in pairs.items():
            P1, P2 = epi.projection_from_KRt(K1, eye, torch.zeros_like(C)), epi.projection_from_KRt(K1, R2, t2)
            x1, x2 = project(P1, X), project(P2, X)
            if name == "rectified":
                assert torch.equal(x1[..., 1], x2[..., 1])
            for solver in ("svd", "cofactor"):
                out = epi.triangulate_points(P1, P2, x1, x2, solver=solver)
                assert (out - X).norm(dim=-1).max() <= _TRIANGULATION_ATOL[dtype], (name, solver)
            svd = epi.triangulate_points(P1, P2, x1 + n1, x2 + n2, solver="svd")
            cofactor = epi.triangulate_points(P1, P2, x1 + n1, x2 + n2, solver="cofactor")
            assert (svd - X).norm(dim=-1).max() <= 0.25, name
            assert (cofactor - svd).norm(dim=-1).max() <= 1e-4, name

    @staticmethod
    def _shifted_scaled_noisy_fixture(device: torch.device, dtype: torch.dtype):
        """Return camera poses, intrinsics, noisy projections, and independent midpoint DLT reference."""
        K1 = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        K2 = torch.tensor([[820.0, 0.0, 310.0], [0.0, 820.0, 250.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        R1 = torch.tensor(
            [
                [0.9968017063026194, 0.0, 0.0799146939691727],
                [0.0, 1.0, 0.0],
                [-0.0799146939691727, 0.0, 0.9968017063026194],
            ],
            device=device,
            dtype=dtype,
        )
        R2 = torch.tensor(
            [
                [0.9780309147241483, 0.0, -0.2084598998460996],
                [0.0, 1.0, 0.0],
                [0.2084598998460996, 0.0, 0.9780309147241483],
            ],
            device=device,
            dtype=dtype,
        )
        C1 = torch.tensor([[0.4], [-0.2], [0.1]], device=device, dtype=dtype)
        C2 = torch.tensor([[1.4], [0.1], [-0.2]], device=device, dtype=dtype)
        x1 = torch.tensor(
            [
                [346.960492734637, 222.457937028172],
                [495.44079372896, 335.34058417263],
                [262.764810314054, 347.137414337379],
                [471.256362376846, 174.682530915698],
                [572.28250093614, 307.155903216317],
                [296.64024135684, 255.215374023558],
            ],
            device=device,
            dtype=dtype,
        )
        x2 = torch.tensor(
            [
                [-96.625850918629, 174.989704810013],
                [86.793889451062, 297.106666519409],
                [-145.44730443701, 321.711187767982],
                [2.676362180344, 120.782985532275],
                [152.623258016905, 266.221325507181],
                [-120.270653957701, 217.898905519145],
            ],
            device=device,
            dtype=dtype,
        )
        # Expected unshifted/unit-scale DLT points computed independently with numpy SVD in the midpoint frame
        # (np.random.default_rng(12345).normal(0.0, 0.5) pixel noise around angles a1=0.08, a2=-0.21).
        X_ref = torch.tensor(
            [
                [0.19595262223655785, -0.29748185587550513, 4.507761362426817],
                [1.0982065645199677, 0.4000909131491279, 5.2068001362796394],
                [-0.5018611181220574, 0.5978606830822574, 6.013199335118363],
                [0.798715107131686, -0.499675690509741, 3.8174583478459088],
                [1.5048699031693211, 0.1976998878836682, 4.9191248995025925],
                [-0.1863552625212399, -0.1006820826299922, 5.4578439508466605],
            ],
            device=device,
            dtype=dtype,
        )
        return K1, K2, R1, R2, C1, C2, x1, x2, X_ref

    @pytest.mark.applies_to(dtypes=("float32", "float64"), reason="precision test for shifted/scaled scenes")
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_convention_triangulate_origin_shift_and_unit_scale(self, solver, device, dtype):
        K1, K2, R1, R2, C1, C2, x1, x2, X_ref = self._shifted_scaled_noisy_fixture(device, dtype)
        shift_dir = torch.tensor([[1.0], [-0.7], [0.5]], device=device, dtype=dtype)
        # In float32, the 1/1000 scene shifted by 500 m stores C_shifted ~ 0.5 against a 1e-3 baseline,
        # quantizing P's translation by ~6e-8 / 1e-3 ~ 6e-5 in normalized coordinates. Cofactor uses one step
        # of inverse iteration and matches SVD within O((s4/s3)^3) ~ 1.5e-8 in float64 on noisy input.
        ref_tol = 1e-6 if solver == "cofactor" else 1e-10
        tol = ref_tol if dtype == torch.float64 else 1e-3
        self_tol = 1e-10 if dtype == torch.float64 else 1e-3
        rec_baseline = None
        for scale in (1.0, 1000.0, 0.001):
            for shift_mag in (0.0, 87.0, 500.0):
                shift = shift_dir * shift_mag
                C1_s = (C1 + shift) * scale
                C2_s = (C2 + shift) * scale
                P1 = (K1 @ torch.cat([R1, -R1 @ C1_s], dim=-1)).unsqueeze(0)
                P2 = (K2 @ torch.cat([R2, -R2 @ C2_s], dim=-1)).unsqueeze(0)
                out = epi.triangulate_points(P1, P2, x1.unsqueeze(0), x2.unsqueeze(0), solver=solver).squeeze(0)
                rec = out / scale - shift.squeeze(-1)
                self.assert_close(rec, X_ref, atol=tol, rtol=0.0)
                if rec_baseline is None:
                    rec_baseline = rec
                else:
                    self.assert_close(rec, rec_baseline, atol=self_tol, rtol=0.0)

    @pytest.mark.applies_to(dtypes=("float32", "float64"), reason="noisy view-swap invariance across shift and scale")
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_convention_triangulate_noisy_view_swap_symmetry(self, solver, device, dtype):
        K1, K2, R1, R2, C1, C2, x1, x2, _ = self._shifted_scaled_noisy_fixture(device, dtype)
        shift = torch.tensor([[500.0], [-350.0], [250.0]], device=device, dtype=dtype)
        # cof inverse iteration picks the longest sub-system and can differ slightly at float32 input roundoff.
        tol = 1e-10 if dtype == torch.float64 else 5e-5
        for scale in (1.0, 1000.0, 0.001):
            P1 = (K1 @ torch.cat([R1, -R1 @ ((C1 + shift) * scale)], dim=-1)).unsqueeze(0)
            P2 = (K2 @ torch.cat([R2, -R2 @ ((C2 + shift) * scale)], dim=-1)).unsqueeze(0)
            out12 = epi.triangulate_points(P1, P2, x1.unsqueeze(0), x2.unsqueeze(0), solver=solver).squeeze(0)
            out21 = epi.triangulate_points(P2, P1, x2.unsqueeze(0), x1.unsqueeze(0), solver=solver).squeeze(0)
            self.assert_close(out12 / scale - shift.squeeze(-1), out21 / scale - shift.squeeze(-1), atol=tol, rtol=0.0)

    @pytest.mark.applies_to(dtypes=("float32", "float64"), reason="singular M and shifted zero-baseline convention")
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_convention_triangulate_singular_M_and_shifted_zero_baseline(self, solver, device, dtype):
        # Singular M (affine cameras at infinity) and mixed perspective+affine far from origin: finite result.
        P_aff1_u = torch.tensor(
            [[[1.0, 0.0, 0.0, -10.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        P_aff2_u = torch.tensor(
            [[[0.0, 0.0, 1.0, -20.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        P_aff2 = torch.tensor(
            [[[0.0, 0.0, 500.0, -100000.0], [0.0, 500.0, 0.0, 25000.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        P_persp = torch.tensor(
            [[[800.0, 0.0, 320.0, -79360.0], [0.0, 800.0, 240.0, 41200.0], [0.0, 0.0, 1.0, -195.0]]],
            device=device,
            dtype=dtype,
        )
        X_aff_u = torch.tensor([[[11.5, -4.2, 22.0], [9.8, -5.4, 17.0]]], device=device, dtype=dtype)
        X_aff = torch.tensor([[[101.5, -49.2, 205.0], [99.8, -50.4, 197.0]]], device=device, dtype=dtype)

        def project(P: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
            return _dehom(torch.cat([Y, torch.ones_like(Y[..., :1])], -1) @ P.transpose(-2, -1))

        xa1_u, xa2_u = project(P_aff1_u, X_aff_u), project(P_aff2_u, X_aff_u)
        xp, xa2 = project(P_persp, X_aff), project(P_aff2, X_aff)
        tol = 1e-6 if dtype == torch.float64 else 5e-2
        self.assert_close(
            epi.triangulate_points(P_aff1_u, P_aff2_u, xa1_u, xa2_u, solver=solver), X_aff_u, atol=tol, rtol=0.0
        )
        self.assert_close(epi.triangulate_points(P_persp, P_aff2, xp, xa2, solver=solver), X_aff, atol=tol, rtol=0.0)

        # Shifted zero baseline: coincident camera centers at [2001, -1998, 2003].
        two_view = two_view_scene(device, dtype)
        K1, K2, R, X = two_view["K1"], two_view["K2"], two_view["R"], two_view["X"]
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        C_far = torch.tensor([[[2001.0], [-1998.0], [2003.0]]], device=device, dtype=dtype)
        P1_zb = epi.projection_from_KRt(K1, eye, -C_far)
        P2_zb = epi.projection_from_KRt(K2, R, -R @ C_far)
        x1_zb, x2_zb = project(P1_zb, X + C_far.transpose(-2, -1)), project(P2_zb, X + C_far.transpose(-2, -1))
        out_zb = epi.triangulate_points(P1_zb, P2_zb, x1_zb, x2_zb, solver=solver)
        if solver == "cofactor":
            assert torch.isnan(out_zb).all()
        else:
            X64 = X.cpu().double()
            out_cpu = out_zb.cpu().double()
            finite = torch.isfinite(out_cpu).all(-1)
            assert finite.sum() >= 6
            v, direction = out_cpu[finite] - C_far.cpu().double().transpose(-2, -1)[0], X64[finite]
            assert (v.norm(dim=-1) > 1e-3).sum() >= 6
            assert (torch.linalg.cross(v, direction, dim=-1).norm(dim=-1) / direction.norm(dim=-1)).max() <= 5e-2
            displaced = v.norm(dim=-1) > 1e-3
            sin_angle = torch.linalg.cross(v[displaced], direction[displaced], dim=-1).norm(dim=-1) / (
                v[displaced].norm(dim=-1) * direction[displaced].norm(dim=-1)
            )
            assert sin_angle.max() <= 5e-2

    @pytest.mark.applies_to(dtypes=("float16",), reason="resolvable unit baseline at offset origin in float16")
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_convention_triangulate_half_nonzero_baseline_large_offset(self, solver, device, dtype):
        expected = torch.tensor([[[0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        P1 = torch.tensor(
            [[[1.0, 0.0, 0.0, 200.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        P2 = torch.tensor(
            [[[1.0, 0.0, 0.0, 201.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        pt1 = torch.tensor([[[200.0, 0.0]]], device=device, dtype=dtype)
        pt2 = torch.tensor([[[201.0, 0.0]]], device=device, dtype=dtype)
        out = epi.triangulate_points(P1, P2, pt1, pt2, solver=solver)
        assert torch.isfinite(out).all()
        self.assert_close(out, expected, atol=5e-2, rtol=0.0)

        # Even at C_x = -500 where (2 * in_eps) fallback triggers, cofactor stays finite without artificial NaN.
        P1_500 = torch.tensor(
            [[[1.0, 0.0, 0.0, 500.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], device=device, dtype=dtype
        )
        P2_500 = torch.tensor(
            [[[1.0, 0.0, 0.0, 501.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], device=device, dtype=dtype
        )
        pt1_500 = torch.tensor([[[0.0, 0.0]]], device=device, dtype=dtype)
        pt2_500 = torch.tensor([[[0.1, 0.0]]], device=device, dtype=dtype)
        expected_500 = torch.tensor([[[-500.0, 0.0, 10.0]]], device=device, dtype=dtype)
        out_500 = epi.triangulate_points(P1_500, P2_500, pt1_500, pt2_500, solver=solver)
        assert torch.isfinite(out_500).all()
        self.assert_close(out_500, expected_500, atol=0.5, rtol=0.0)

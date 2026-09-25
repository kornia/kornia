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

    def test_cofactor_sign_alignment(self, device, dtype):
        """Cofactor solver must not produce NaN/Inf due to sign cancellation.

        Before the sign-alignment fix, the two 3x4 sub-systems could yield
        opposite-signed null vectors whose sum cancelled to ~0, producing NaN
        after dehomogenisation.  This test constructs a noise-free two-view
        scene with a modest baseline, triangulates with both the cofactor and
        SVD solvers, and checks that the cofactor output is finite and
        directionally consistent with the SVD result.
        """
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("cofactor sign test only runs for float32/float64")

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
        # Expected: numpy DLT (same algorithm as cv2.triangulatePoints)
        X_expected = torch.tensor(
            [
                [-0.08818743120343768, 0.188334626330844, 3.6220817756133172],
                [0.16144758240818913, 0.8807090389866558, 5.4844829932547015],
                [0.18216324334542242, 0.517464929751309, 3.3477856439676588],
                [0.3350949547897918, 0.06786420490996858, 2.770735605435314],
                [-0.20989477003241103, -0.3123833872441908, 3.754403690255068],
                [0.5852768065399436, 0.0989402354921456, 2.6188660940360844],
                [0.1480216292032729, 0.5668535581804692, 3.5667031501090993],
                [0.22379169128122922, 0.3033298174456883, 2.9457656267879586],
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
        P1, P2, K2, R, t, X = (two_view[k] for k in ("P1", "P2", "K2", "R", "t", "X"))

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
        # Zero baseline raises nothing: svd and eigh return a point on the line of sight, or NaN when their arbitrary
        # choice in the 2-D null space lands at infinity (one bfloat16 point here), and cofactor NaN (every 3x4
        # sub-system is rank-deficient). Both cameras sit at C, away from the world origin, so that a solver
        # collapsing to the origin cannot pass; the line of sight through X + C is the ray from C in the direction X.
        C = torch.tensor([[[1.0], [2.0], [-3.0]]], device=device, dtype=dtype)
        P1_C = epi.projection_from_KRt(two_view["K1"], torch.eye(3, device=device, dtype=dtype)[None], -C)
        P2_C = epi.projection_from_KRt(K2, R, -R @ C)
        X_C = X + C.transpose(-2, -1)
        for solver in ("svd", "eigh"):
            out = epi.triangulate_points(P1_C, P2_C, project(P1_C, X_C), project(P2_C, X_C), solver=solver)
            finite = torch.isfinite(out).all(-1)
            assert finite.sum() >= 10  # of 12: the line-of-sight check below is not vacuous
            # float64 on the CPU: MPS has no float64
            finite = finite.cpu()
            v, X64 = (out - C.transpose(-2, -1)).cpu().double()[finite], X.cpu().double()[finite]
            assert (torch.linalg.cross(v, X64, dim=-1).norm(dim=-1) / X64.norm(dim=-1)).max() <= _OFF_RAY_TOL[dtype]
            assert (v.norm(dim=-1) / X64.norm(dim=-1)).min() > 1e-3  # not the centre itself
        if dtype in (torch.float32, torch.float64):  # in half precision the rows' roundoff leaves the rank
            out = epi.triangulate_points(P1_C, P2_C, project(P1_C, X_C), project(P2_C, X_C), solver="cofactor")
            assert torch.isnan(out).all()

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

    def test_convention_triangulate_cofactor_half_precision_is_finite(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("the half-precision overflow of the unnormalised cofactor null vector")
        # The cofactor null vector of pixel-scale rows is normalised in float32 before the cast back to the input
        # dtype; cast first, it overflowed float16 and every point was NaN. bfloat16 keeps float32's exponent range
        # but only 8 bits of mantissa, hence its looser bound.
        P1, P2, x1, x2, X = (two_view[k] for k in ("P1", "P2", "x1", "x2", "X"))
        out = epi.triangulate_points(P1, P2, x1, x2, solver="cofactor")
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
        self.assert_close(out, X, rtol=0.0, atol={torch.float16: 5e-2, torch.bfloat16: 1.0}[dtype])

    def test_convention_triangulate_cofactor_rank_deficient_sub_system(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("in half precision the rows' roundoff leaves the sub-systems full rank")
        # A rank-deficient 3x4 sub-system of the DLT matrix is left out of the cofactor average (its null vector
        # used to be roundoff normalised to unit length): NaN when both are, exact from the other one when only one
        # is. svd is the control on the same input.
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
        # A noise-free pure x translation with R = I and one K: y1 == y2 exactly, so DLT rows 1 and 3 coincide.
        P1, P2 = (
            epi.projection_from_KRt(K1, eye, torch.zeros_like(C)),
            epi.projection_from_KRt(K1, eye, -0.5 * eye[:, :, :1]),
        )
        x1, x2 = project(P1, X), project(P2, X)
        assert torch.equal(x1[..., 1], x2[..., 1])
        for solver in ("svd", "cofactor"):
            out = epi.triangulate_points(P1, P2, x1, x2, solver=solver)
            assert (out - X).norm(dim=-1).max() <= _TRIANGULATION_ATOL[dtype]
        # The same rectified pair with 0.5 px of noise: the rows no longer coincide but stay nearly dependent, so
        # dropping only an exactly rank-deficient sub-system does not fix it. svd stays within 0.1.
        g = torch.Generator().manual_seed(0)
        n1, n2 = (0.5 * torch.randn(x1.shape, generator=g, dtype=torch.float64).to(device, dtype) for _ in range(2))
        err = {
            s: (epi.triangulate_points(P1, P2, x1 + n1, x2 + n2, solver=s) - X).norm(dim=-1).max()
            for s in ("svd", "cofactor")
        }
        assert err["svd"] <= 0.25
        assert err["cofactor"] > 1.0

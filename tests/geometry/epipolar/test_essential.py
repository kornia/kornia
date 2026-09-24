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
import kornia.geometry.epipolar as epi

from testing.base import BaseTester
from testing.geometry.create import generate_two_view_random_scene


class TestFindEssential(BaseTester):
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 5, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 5, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 5, device=device, dtype=dtype)
        E_mat = epi.essential.find_essential(points1, points2, weights)
        assert E_mat.shape == (1, 10, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 5), (2, 6), (3, 7), (16, 5)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        E_mat = epi.essential.find_essential(points1, points2, weights)
        assert E_mat.shape == (B, 10, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 5), (2, 6), (3, 7)])
    def test_shape_noweights(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = None
        E_mat = epi.essential.find_essential(points1, points2, weights)
        assert E_mat.shape == (B, 10, 3, 3)

    def test_epipolar_constraint(self, device, dtype):
        calibrated_x1 = torch.tensor(
            [[[0.0640, 0.7799], [-0.2011, 0.2836], [-0.1355, 0.2907], [0.0520, 1.0086], [-0.0361, 0.6533]]],
            device=device,
            dtype=dtype,
        )
        calibrated_x2 = torch.tensor(
            [[[0.3470, -0.4274], [-0.1818, -0.1281], [-0.1766, -0.1617], [0.4066, -0.0706], [0.1137, 0.0363]]],
            device=device,
            dtype=dtype,
        )

        E = epi.essential.find_essential(calibrated_x1, calibrated_x2)
        if torch.all(E != 0):
            distance = epi.symmetrical_epipolar_distance(calibrated_x1, calibrated_x2, E)
            distance = torch.nan_to_num(distance, nan=1e8)
            # Note : here we check only the best model, although all solutions are returned
            mean_error = distance.mean(-1).min()
            self.assert_close(mean_error, torch.tensor(0.0, device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    def test_synthetic_sampson(self, device, dtype, monkeypatch):
        calibrated_x1 = torch.tensor(
            [[[0.0640, 0.7799], [-0.2011, 0.2836], [-0.1355, 0.2907], [0.0520, 1.0086], [-0.0361, 0.6533]]],
            device=device,
            dtype=dtype,
        )
        calibrated_x2 = torch.tensor(
            [[[0.3470, -0.4274], [-0.1818, -0.1281], [-0.1766, -0.1617], [0.4066, -0.0706], [0.1137, 0.0363]]],
            device=device,
            dtype=dtype,
        )

        weights = torch.ones_like(calibrated_x2)[..., 0]
        E_est = epi.essential.find_essential(calibrated_x1, calibrated_x2, weights)
        error = epi.sampson_epipolar_distance(calibrated_x1, calibrated_x2, E_est)
        error = torch.nan_to_num(error, nan=1e8)
        self.assert_close(
            error[:, torch.argmin(error.mean(-1))],
            torch.zeros((calibrated_x1.shape[:2]), device=device, dtype=dtype),
            atol=1e-4,
            rtol=1e-4,
        )

        # Noise-free minimal samples of random scenes: the pose is recovered for nearly all of them. About
        # half have a negative leading coefficient in the degree-10 polynomial, which a floor on that
        # coefficient such as clamp_min turns into the wrong roots (#4847).
        g = torch.Generator().manual_seed(0)
        n = 64
        points = torch.rand(n, 5, 3, generator=g, dtype=torch.float64) * torch.tensor(
            [2.0, 2.0, 4.0], dtype=torch.float64
        )
        points = points + torch.tensor([-1.0, -1.0, 3.0], dtype=torch.float64)
        axis_angle = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.4
        trans = torch.nn.functional.normalize(torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5, dim=-1)
        points, axis_angle, trans = (v.to(device=device, dtype=dtype) for v in (points, axis_angle, trans))
        R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)
        points2 = points @ R.transpose(-1, -2) + trans[:, None]
        x1, x2 = points[..., :2] / points[..., 2:], points2[..., :2] / points2[..., 2:]
        E_gt = epi.essential_from_Rt(
            torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3),
            torch.zeros(n, 3, 1, device=device, dtype=dtype),
            R,
            trans[..., None],
        )
        weights = torch.ones(n, 5, device=device, dtype=dtype)
        E_est = epi.essential.find_essential(x1, x2, weights)

        def unit(M):
            return M / M.flatten(-2).norm(dim=-1)[..., None, None]

        err = torch.minimum(
            (unit(E_est) - unit(E_gt)[:, None]).flatten(-2).norm(dim=-1),
            (unit(E_est) + unit(E_gt)[:, None]).flatten(-2).norm(dim=-1),
        )
        best = torch.nan_to_num(err, nan=1.0).min(dim=-1).values
        tol, at_least = (1e-8, 60) if dtype == torch.float64 else (1e-3, 48)
        assert int((best < tol).sum()) >= at_least

        # The roots of a polynomial do not depend on its sign or scale, and IEEE division makes that exact
        # for a power-of-two scale, so negating every coefficient, or scaling them all by 2**-40, must leave
        # the candidates unchanged bit for bit. This holds on every platform, whichever sign the SVD's
        # null-space basis gives a sample. The scale takes every leading coefficient here below 1e-8, so it
        # also pins that only an exactly zero one is replaced, not a small one.
        determinant = epi.essential._determinant_to_polynomial_jit
        for factor in (-1.0, 2.0**-40):
            monkeypatch.setattr(
                epi.essential, "_determinant_to_polynomial_jit", lambda A, *args, f=factor: f * determinant(A, *args)
            )
            E_scaled = epi.essential.find_essential(x1, x2, weights)
            assert torch.equal(torch.isnan(E_scaled), torch.isnan(E_est))
            self.assert_close(torch.nan_to_num(E_scaled), torch.nan_to_num(E_est), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("num_points", [5, 6, 8])
    def test_gradcheck(self, num_points, device):
        # For fewer than 9 points some or all of the four null-space vectors lie past min(N, 9), where
        # torch.linalg.svd gives no gradient, so the gradient to the correspondences was dropped: exactly
        # zero for 5 points (#4855). Candidates from complex roots are NaN and are zeroed here; they stay
        # complex under the small perturbations the check makes.
        g = torch.Generator().manual_seed(1)
        points1 = torch.rand(1, num_points, 2, generator=g, dtype=torch.float64).to(device)
        points2 = torch.rand(1, num_points, 2, generator=g, dtype=torch.float64).to(device)

        def proxy(points1, points2):
            return epi.essential.find_essential(points1, points2).nan_to_num()

        self.gradcheck(proxy, (points1, points2))

    def test_null_space_gradient_of_a_discarded_sample(self, device):
        # Points all at the origin give a rank-1 design matrix, so the null space the solver uses is not
        # unique and its derivative has no gap to divide by. A sample like that has its candidates
        # discarded, so no gradient reaches its basis, and it must contribute zero rather than 0 / 0.
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        design = torch.zeros(1, 5, 9, device=device, dtype=torch.float64)
        design[..., 8] = 1.0
        design.requires_grad_()
        basis = epi.essential._NullSpaceBasis.apply(design)
        (basis * 0.0).sum().backward()
        assert torch.isfinite(design.grad).all()
        assert (design.grad == 0).all()

    @pytest.mark.parametrize("batch_size, num_points", [(5, 5), (10, 5)])
    def test_degenerate_case(self, batch_size, num_points, device, dtype, monkeypatch):
        B, N = batch_size, num_points
        eye = torch.eye(3, device=device, dtype=dtype)

        # Points all at the origin give a design matrix whose SVD returns unit vectors for its null
        # space, and every basis of four unit vectors makes the 10x10 elimination matrix exactly
        # singular. A singular sample has no solution, so find_essential returns run_5point's fallback
        # for an element without candidates: the identity, for all 10.
        zeros = torch.zeros(B, N, 2, device=device, dtype=dtype)
        E_zeros = epi.essential.find_essential(zeros, zeros, torch.ones(B, N, device=device, dtype=dtype))
        self.assert_close(E_zeros, eye.expand(B, 10, 3, 3), atol=0.0, rtol=0.0)

        # A singular element does not disturb the rest of its batch: next to one, a regular sample
        # returns exactly what it returns next to a regular sample, NaN candidates from complex roots
        # included. The reference is a batch of the same size, because the batch size alone can change
        # the last bits of a result (by 4.8e-15 at float64 on macOS arm64).
        x1 = torch.tensor(
            [[0.0640, 0.7799], [-0.2011, 0.2836], [-0.1355, 0.2907], [0.0520, 1.0086], [-0.0361, 0.6533]],
            device=device,
            dtype=dtype,
        )
        x2 = torch.tensor(
            [[0.3470, -0.4274], [-0.1818, -0.1281], [-0.1766, -0.1617], [0.4066, -0.0706], [0.1137, 0.0363]],
            device=device,
            dtype=dtype,
        )
        weights = torch.ones(2, 5, device=device, dtype=dtype)
        regular = epi.essential.find_essential(torch.stack((x1, x1)), torch.stack((x2, x2)), weights)[1]
        mixed = epi.essential.find_essential(
            torch.stack((zeros[0, :5], x1)),
            torch.stack((zeros[0, :5], x2)),
            weights,
        )
        self.assert_close(mixed[0], eye.expand(10, 3, 3), atol=0.0, rtol=0.0)
        assert torch.equal(torch.isnan(mixed[1]), torch.isnan(regular))
        self.assert_close(torch.nan_to_num(mixed[1]), torch.nan_to_num(regular), atol=0.0, rtol=0.0)

        # Whether any other degenerate set is exactly singular, or ill-conditioned enough to make the
        # companion matrix non-finite, depends on the platform's LAPACK. These sets and a random draw
        # with the same points in both images exercise those paths where they occur, and must return the
        # documented shape either way. The last two give a non-finite companion matrix at float32 on
        # Linux x86 and on macOS arm64 respectively.
        lshape = torch.tensor([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.5]], device=device, dtype=dtype)
        draw = torch.rand(B, N, 2, generator=torch.Generator().manual_seed(79)).to(device=device, dtype=dtype)
        on_axis = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], device=device, dtype=dtype)
        repeated = torch.tensor(
            [[2.0, 0.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 1.0]], device=device, dtype=dtype
        )
        for points in (lshape.expand(B, 5, 2), draw, on_axis.expand(B, 5, 2), repeated.expand(B, 5, 2)):
            weights = torch.ones(points.shape[:2], device=device, dtype=dtype)
            assert epi.essential.find_essential(points, points, weights).shape == (B, 10, 3, 3)

        # For a design matrix whose rows are unit vectors, the SVD returns a null space of unit vectors,
        # which makes the elimination matrix exactly singular. For this one, solving against the identity
        # leaves finite candidates that fail the essential-matrix constraints, so they must be dropped.
        design = torch.eye(9, device=device, dtype=dtype)[[0, 1, 3, 5, 6]].expand(B, 5, 9)
        assert torch.isnan(epi.essential.null_to_Nister_solution(design, B)).all()

        # A companion matrix that is not finite has no roots, and torch.linalg.eigvals aborts on one, which
        # used to take the whole batch down. Which inputs produce it is platform-dependent, so make the
        # determinant polynomial of one element non-finite instead: that element takes the identity
        # fallback, and the other returns exactly what it returns in the unpatched batch above.
        determinant = epi.essential._determinant_to_polynomial_jit

        def overflowed(A, *args):
            cs = determinant(A, *args).clone()
            cs[0] = float("nan")
            return cs

        monkeypatch.setattr(epi.essential, "_determinant_to_polynomial_jit", overflowed)
        patched = epi.essential.find_essential(torch.stack((x1, x1)), torch.stack((x2, x2)), weights)
        self.assert_close(patched[0], eye.expand(10, 3, 3), atol=0.0, rtol=0.0)
        assert torch.equal(torch.isnan(patched[1]), torch.isnan(regular))
        self.assert_close(torch.nan_to_num(patched[1]), torch.nan_to_num(regular), atol=0.0, rtol=0.0)


class TestEssentialFromFundamental(BaseTester):
    def test_smoke(self, device, dtype):
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        E_mat = epi.essential_from_fundamental(F_mat, K1, K2)
        assert E_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        F_mat = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        E_mat = epi.essential_from_fundamental(F_mat, K1, K2)
        assert E_mat.shape == (B, 3, 3)

    @pytest.mark.xfail(reason="TODO: fix #685")
    def test_from_to_fundamental(self, device, dtype):
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        E_mat = epi.essential_from_fundamental(F_mat, K1, K2)
        F_hat = epi.fundamental_from_essential(E_mat, K1, K2)
        self.assert_close(F_mat, F_hat, atol=1e-4, rtol=1e-4)

    def test_shape_large(self, device, dtype):
        F_mat = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        E_mat = epi.essential_from_fundamental(F_mat, K1, K2)
        assert E_mat.shape == (1, 2, 3, 3)

    def test_from_fundamental(self, device, dtype):
        scene = generate_two_view_random_scene(device, dtype)

        F_mat = scene["F"]

        K1 = scene["K1"]
        K2 = scene["K2"]

        E_mat = epi.essential_from_fundamental(F_mat, K1, K2)
        F_hat = epi.fundamental_from_essential(E_mat, K1, K2)

        F_mat_norm = epi.normalize_transformation(F_mat)
        F_hat_norm = epi.normalize_transformation(F_hat)
        self.assert_close(F_mat_norm, F_hat_norm)

    def test_gradcheck(self, device):
        F_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        K1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        K2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        self.gradcheck(epi.essential_from_fundamental, (F_mat, K1, K2), requires_grad=(True, False, False))


class TestRelativeCameraMotion(BaseTester):
    def test_smoke(self, device, dtype):
        R1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        t1 = torch.rand(1, 3, 1, device=device, dtype=dtype)
        R2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        t2 = torch.rand(1, 3, 1, device=device, dtype=dtype)
        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        assert R.shape == (1, 3, 3)
        assert t.shape == (1, 3, 1)

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 8])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        R1 = torch.rand(B, 3, 3, device=device, dtype=dtype)
        t1 = torch.rand(B, 3, 1, device=device, dtype=dtype)
        R2 = torch.rand(1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        t2 = torch.rand(B, 3, 1, device=device, dtype=dtype)
        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        assert R.shape == (B, 3, 3)
        assert t.shape == (B, 3, 1)

    def test_translation(self, device, dtype):
        R1 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        t1 = torch.tensor([[[10.0], [0.0], [0.0]]]).type_as(R1)

        R2 = kornia.core.ops.eye_like(3, R1)
        t2 = kornia.core.ops.vec_like(3, t1)

        R_expected = R1.clone()
        t_expected = -t1

        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        self.assert_close(R_expected, R)
        self.assert_close(t_expected, t)

    def test_rotate_z(self, device, dtype):
        R1 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        R2 = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        t1 = kornia.core.ops.vec_like(3, R1)
        t2 = kornia.core.ops.vec_like(3, R2)

        R_expected = R2.clone()
        t_expected = t1

        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        self.assert_close(R_expected, R)
        self.assert_close(t_expected, t)

    def test_gradcheck(self, device):
        R1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        R2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        t1 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        t2 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        self.gradcheck(epi.relative_camera_motion, (R1, t1, R2, t2), requires_grad=(True, False, False, False))


class TestEssentalFromRt(BaseTester):
    def test_smoke(self, device, dtype):
        R1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        t1 = torch.rand(1, 3, 1, device=device, dtype=dtype)
        R2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        t2 = torch.rand(1, 3, 1, device=device, dtype=dtype)
        E_mat = epi.essential_from_Rt(R1, t1, R2, t2)
        assert E_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 8])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        R1 = torch.rand(B, 3, 3, device=device, dtype=dtype)
        t1 = torch.rand(B, 3, 1, device=device, dtype=dtype)
        R2 = torch.rand(1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        t2 = torch.rand(B, 3, 1, device=device, dtype=dtype)
        E_mat = epi.essential_from_Rt(R1, t1, R2, t2)
        assert E_mat.shape == (B, 3, 3)

    @pytest.mark.xfail(reason="TODO: fix #685")
    def test_from_fundamental_Rt(self, device, dtype):
        scene = generate_two_view_random_scene(device, dtype)

        E_from_Rt = epi.essential_from_Rt(scene["R1"], scene["t1"], scene["R2"], scene["t2"])

        E_from_F = epi.essential_from_fundamental(scene["F"], scene["K1"], scene["K2"])

        E_from_Rt_norm = epi.normalize_transformation(E_from_Rt)
        E_from_F_norm = epi.normalize_transformation(E_from_F)
        # TODO: occasionally failed with error > 0.04
        self.assert_close(E_from_Rt_norm, E_from_F_norm, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device):
        R1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        R2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        t1 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        t2 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        self.gradcheck(epi.essential_from_Rt, (R1, t1, R2, t2), requires_grad=(True, False, False, False))


class TestDecomposeEssentialMatrix(BaseTester):
    def test_smoke(self, device, dtype):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        R1, R2, t = epi.decompose_essential_matrix(E_mat)
        assert R1.shape == (1, 3, 3)
        assert R2.shape == (1, 3, 3)
        assert t.shape == (1, 3, 1)

    @pytest.mark.parametrize("batch_shape", [(1, 3, 3), (2, 3, 3), (2, 1, 3, 3), (3, 2, 1, 3, 3)])
    def test_shape(self, batch_shape, device, dtype):
        E_mat = torch.rand(batch_shape, device=device, dtype=dtype)
        R1, R2, t = epi.decompose_essential_matrix(E_mat)
        assert R1.shape == batch_shape
        assert R2.shape == batch_shape
        assert t.shape == batch_shape[:-1] + (1,)

    def test_gradcheck(self, device):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)

        def eval_rot1(input):
            return epi.decompose_essential_matrix(input)[0]

        def eval_rot2(input):
            return epi.decompose_essential_matrix(input)[1]

        def eval_vec(input):
            return epi.decompose_essential_matrix(input)[2]

        self.gradcheck(eval_rot1, (E_mat,))
        self.gradcheck(eval_rot2, (E_mat,))
        self.gradcheck(eval_vec, (E_mat,))


class TestDecomposeEssentialMatrixNoSVD(BaseTester):
    def test_smoke(self, device, dtype):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        R1, R2, t = epi.decompose_essential_matrix_no_svd(E_mat)
        assert R1.shape == (1, 3, 3)
        assert R2.shape == (1, 3, 3)
        assert t.shape == (1, 3, 1)

    @pytest.mark.parametrize("batch_shape", [(3, 3), (1, 3, 3), (2, 3, 3), (2, 1, 3, 3), (3, 2, 1, 3, 3)])
    def test_shape(self, batch_shape, device, dtype):
        E_mat = torch.rand(batch_shape, device=device, dtype=dtype)
        R1, R2, t = epi.decompose_essential_matrix_no_svd(E_mat)
        if len(batch_shape) >= 2:
            batch_shape = E_mat.view(-1, 3, 3).shape
        assert R1.shape == batch_shape
        assert R2.shape == batch_shape
        assert t.shape == batch_shape[:-1] + (1,)

    def test_gradcheck(self, device):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)

        def eval_rot1(input):
            return epi.decompose_essential_matrix_no_svd(input)[0]

        def eval_rot2(input):
            return epi.decompose_essential_matrix_no_svd(input)[1]

        def eval_vec(input):
            return epi.decompose_essential_matrix_no_svd(input)[2]

        self.gradcheck(eval_rot1, (E_mat,))
        self.gradcheck(eval_rot2, (E_mat,))
        self.gradcheck(eval_vec, (E_mat,))

    def test_correct_decompose(self):
        E_mat = torch.tensor([[[0.2057, -3.8266, 3.1615], [4.5417, -1.0707, -2.2023], [-1.0975, 1.6386, -0.6590]]])
        R1, R2, t = epi.decompose_essential_matrix(E_mat)
        R1_1, R2_1, t_1 = epi.decompose_essential_matrix_no_svd(E_mat)
        # As the orders of two R solutions and t solutions might be different from epi.decompose_essential_matrix(),
        # we have to check on the correct ones
        rtol: float = 1e-4
        if (R1 - R1_1).abs().sum() < rtol:
            self.assert_close(R1, R1_1)
            self.assert_close(R2, R2_1)
        else:
            self.assert_close(R1, R2_1)
            self.assert_close(R2, R1_1)
            R1_1, R2_1 = R2_1, R1_1

        if (t - t_1).abs().sum() < rtol:
            self.assert_close(t, t_1)
        else:
            self.assert_close(t, -t_1)
            t_1 = -t_1.clone()
        self.assert_close(
            epi.essential_from_Rt(R1_1, t_1, R2_1, -t_1), epi.essential_from_Rt(R1, t, R2, -t), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.xfail(reason="skip the tests where there are no solutions.")
    def test_consistency(self, device, dtype):
        scene = generate_two_view_random_scene(device, dtype)

        R1, t1 = scene["R1"], scene["t1"]
        R2, t2 = scene["R2"], scene["t2"]

        E_mat = epi.essential_from_Rt(R1, t1, R2, t2)

        # compare the decomposed R and t to the method with svd
        R1, R2, t = epi.decompose_essential_matrix(E_mat)
        R1_1, R2_1, t_1 = epi.decompose_essential_matrix_no_svd(E_mat)
        # As the orders of two R solutions and t solutions might be different from epi.decompose_essential_matrix(),
        # we have to check on the correct ones
        rtol: float = 1e-4
        if (R1 - R1_1).abs().sum() < rtol:
            self.assert_close(R1, R1_1)
            self.assert_close(R2, R2_1)
        else:
            self.assert_close(R1, R2_1)
            self.assert_close(R2, R1_1)
            R1_1, R2_1 = R2_1, R1_1

        if (t - t_1).abs().sum() < rtol:
            self.assert_close(t, t_1)
        else:
            self.assert_close(t, -t_1)
            t_1 = -t_1.clone()

        self.assert_close(epi.essential_from_Rt(R1_1, t_1, R2_1, -t_1), epi.essential_from_Rt(R1, t, R2, -t))


class TestMotionFromEssential(BaseTester):
    def test_smoke(self, device, dtype):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        Rs, Ts = epi.motion_from_essential(E_mat)
        assert Rs.shape == (1, 4, 3, 3)
        assert Ts.shape == (1, 4, 3, 1)

    @pytest.mark.parametrize("batch_shape", [(1, 3, 3), (2, 3, 3), (2, 1, 3, 3), (3, 2, 1, 3, 3)])
    def test_shape(self, batch_shape, device, dtype):
        E_mat = torch.rand(batch_shape, device=device, dtype=dtype)
        Rs, Ts = epi.motion_from_essential(E_mat)
        assert Rs.shape == batch_shape[:-2] + (4, 3, 3)
        assert Ts.shape == batch_shape[:-2] + (4, 3, 1)

    def test_two_view(self, device, dtype):
        scene = generate_two_view_random_scene(device, dtype)

        R1, t1 = scene["R1"], scene["t1"]
        R2, t2 = scene["R2"], scene["t2"]

        E_mat = epi.essential_from_Rt(R1, t1, R2, t2)

        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        t = torch.nn.functional.normalize(t, dim=1)

        Rs, ts = epi.motion_from_essential(E_mat)

        rot_error = (Rs - R).abs().sum((-2, -1))
        vec_error = (ts - t).abs().sum(-1)

        rtol: float = 1e-4
        assert (rot_error < rtol).any() & (vec_error < rtol).any()

    def test_gradcheck(self, device):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)

        def eval_rot(input):
            return epi.motion_from_essential(input)[0]

        def eval_vec(input):
            return epi.motion_from_essential(input)[1]

        self.gradcheck(eval_rot, (E_mat,))
        self.gradcheck(eval_vec, (E_mat,))


class TestMotionFromEssentialChooseSolution(BaseTester):
    def test_smoke(self, device, dtype):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        x1 = torch.rand(1, 1, 2, device=device, dtype=dtype)
        x2 = torch.rand(1, 1, 2, device=device, dtype=dtype)
        R, t, X = epi.motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2)
        assert R.shape == (1, 3, 3)
        assert t.shape == (1, 3, 1)
        assert X.shape == (1, 1, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 3), (2, 3), (2, 8), (3, 2)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        E_mat = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)  # check for broadcasting
        x1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        x2 = torch.rand(B, 1, 2, device=device, dtype=dtype)  # check for broadcasting
        R, t, X = epi.motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2)
        assert R.shape == (B, 3, 3)
        assert t.shape == (B, 3, 1)
        assert X.shape == (B, N, 3)

    def test_masking(self, device, dtype):
        E_mat = torch.rand(2, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(2, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(2, 3, 3, device=device, dtype=dtype)
        x1 = torch.rand(2, 10, 2, device=device, dtype=dtype)
        x2 = torch.rand(2, 10, 2, device=device, dtype=dtype)

        R, t, X = epi.motion_from_essential_choose_solution(E_mat, K1, K2, x1[:, 1:-1, :], x2[:, 1:-1, :])

        mask = torch.zeros(2, 10, dtype=torch.bool, device=device)
        mask[:, 1:-1] = True
        Rm, tm, Xm = epi.motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=mask)

        self.assert_close(R, Rm)
        self.assert_close(t, tm)
        self.assert_close(X, Xm[:, 1:-1, :])

    @pytest.mark.parametrize("num_points", [10, 15, 20])
    def test_unbatched(self, num_points, device, dtype):
        N = num_points
        E_mat = torch.rand(3, 3, device=device, dtype=dtype)
        K1 = torch.rand(3, 3, device=device, dtype=dtype)
        K2 = torch.rand(3, 3, device=device, dtype=dtype)
        x1 = torch.rand(N, 2, device=device, dtype=dtype)
        x2 = torch.rand(N, 2, device=device, dtype=dtype)

        R, t, X = epi.motion_from_essential_choose_solution(E_mat, K1, K2, x1[1:-1, :], x2[1:-1, :])
        assert R.shape == (3, 3)
        assert t.shape == (3, 1)
        assert X.shape == (N - 2, 3)

        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[1:-1] = True
        Rm, tm, Xm = epi.motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=mask)

        self.assert_close(R, Rm)
        self.assert_close(t, tm)
        self.assert_close(X, Xm[1:-1, :])

    def test_two_view(self, device, dtype):
        scene = generate_two_view_random_scene(device, dtype)

        E_mat = epi.essential_from_Rt(scene["R1"], scene["t1"], scene["R2"], scene["t2"])

        R, t = epi.relative_camera_motion(scene["R1"], scene["t1"], scene["R2"], scene["t2"])
        t = torch.nn.functional.normalize(t, dim=1)

        R_hat, t_hat, _ = epi.motion_from_essential_choose_solution(
            E_mat, scene["K1"], scene["K2"], scene["x1"], scene["x2"]
        )

        self.assert_close(t, t_hat)
        self.assert_close(R, R_hat, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        K1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        K2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        x1 = torch.rand(1, 2, 2, device=device, dtype=torch.float64)
        x2 = torch.rand(1, 2, 2, device=device, dtype=torch.float64)

        self.gradcheck(
            epi.motion_from_essential_choose_solution,
            (E_mat, K1, K2, x1, x2),
            requires_grad=(True, False, False, False, False),
        )

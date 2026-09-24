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


_NO_HALF_LU = "{} calls torch.det (LU), which has no float16/bfloat16 kernel"
_NO_HALF_FIND_ESSENTIAL = "find_essential calls torch.linalg.lu_factor_ex, which has no float16/bfloat16 kernel"
_HALF_PIXEL_F = (
    "a pixel-unit F spans eight decades (entries down to ~1e-8): float16 flushes the small entries to zero and "
    "bfloat16's 8-bit mantissa cannot resolve the epipolar residual, which kornia evaluates in the input dtype"
)

# A five-point sample of normalised coordinates with no real root (recipe in the #4883 pin).
_NO_REAL_ROOT_P1 = [
    [-0.094936139146062, 0.3761027702259922],
    [-0.13508466816570844, -0.15520684765969847],
    [0.4518000060718804, 0.7491498583169849],
    [-0.021505790125727963, 0.25308016814257595],
    [-0.021008959788924204, -0.29327032505050715],
]
_NO_REAL_ROOT_P2 = [
    [0.5908350760196772, -0.024407062108416273],
    [0.4805667988287366, -0.025019944402247606],
    [-0.5701912982679379, 0.4911702947428827],
    [-0.8380815108891836, 0.2981499818513886],
    [-0.8246696433582323, 0.08361235498654443],
]


def _skip_half(dtype: torch.dtype, reason: str) -> None:
    if dtype in (torch.float16, torch.bfloat16):
        pytest.skip(reason)


def _skip_find_essential(device: torch.device, dtype: torch.dtype) -> None:
    _skip_half(dtype, _NO_HALF_FIND_ESSENTIAL)
    if device.type == "mps":
        pytest.skip("find_essential calls torch.linalg.eigvals, which has no MPS kernel (#4528)")


def _hom(p: torch.Tensor) -> torch.Tensor:
    return torch.cat([p, torch.ones_like(p[..., :1])], -1)


def _epipolar_residual(F: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """|pts2^T F pts1| per correspondence."""
    return (_hom(pts2) * (_hom(pts1) @ F.transpose(-2, -1))).sum(-1).abs()


def _normalized(K: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Normalised camera coordinates K^-1 [u, v, 1]^T, dehomogenised."""
    return (_hom(pts) @ torch.linalg.inv(K).transpose(-2, -1))[..., :2]


def _gt_essential(scene):
    eye = torch.eye(3, device=scene["R"].device, dtype=scene["R"].dtype)[None]
    return epi.essential_from_Rt(eye, torch.zeros_like(scene["t"]), scene["R"], scene["t"])


def _first_camera(device, dtype):
    """A non-identity first camera, as world-to-camera extrinsics (R, t)."""
    Ra = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
        torch.tensor([[-0.3, 0.15, 0.2]], device=device, dtype=dtype)
    )
    ta = torch.tensor([[[0.2], [-0.4], [1.1]]], device=device, dtype=dtype)
    return Ra, ta


class TestConventionEssential(BaseTester):
    def test_convention_find_essential_returns_ten_candidates(self, two_view, device, dtype):
        _skip_find_essential(device, dtype)
        n1, n2 = _normalized(two_view["K1"], two_view["x1"]), _normalized(two_view["K2"], two_view["x2"])
        real = {}
        for num_points in (5, 12):
            E = epi.find_essential(n1[:, :num_points], n2[:, :num_points])
            # Always ten slots; each is either a real solution of unit Frobenius norm or all-NaN (complex root).
            assert E.shape == (1, 10, 3, 3)
            finite = torch.isfinite(E).all(dim=-1).all(dim=-1)[0]
            assert (finite | torch.isnan(E).all(dim=-1).all(dim=-1)[0]).all()
            real[num_points] = E[0, finite]
            assert real[num_points].shape[0] >= 1
            self.assert_close(real[num_points].norm(dim=(-2, -1)), torch.ones_like(real[num_points][:, 0, 0]))
        # float32 loses the minimal sample's true E to roundoff (#4884), so the truth is checked on all twelve points.
        # Minimal sample: every real candidate satisfies x2^T E x1 = 0 on the normalised coordinates of points1
        # (first image) and points2 (second image); the swapped product is the control.
        p1, p2 = n1[:, :5], n2[:, :5]
        for cand in real[5]:
            assert _epipolar_residual(cand[None], p1, p2).max() < 1e-3 * _epipolar_residual(cand[None], p2, p1).max()
        # All twelve points: one candidate is the true E up to sign, on the same side.
        E_gt = _gt_essential(two_view)
        E_gt = E_gt / E_gt.norm()
        dist = torch.minimum((real[12] - E_gt).abs().amax(dim=(-2, -1)), (real[12] + E_gt).abs().amax(dim=(-2, -1)))
        assert dist.min() < 1e-4
        best = real[12][dist.argmin()][None]
        assert _epipolar_residual(best, n1, n2).max() < 1e-3 * _epipolar_residual(best, n2, n1).max()

    def test_convention_essential_from_Rt_is_tx_R_of_relative_motion(self, two_view, device, dtype):
        R, t = two_view["R"], two_view["t"]
        Ra, ta = _first_camera(device, dtype)
        # E = [t]x R of relative_camera_motion(R1, t1, R2, t2), here with a non-identity first camera.
        E = epi.essential_from_Rt(Ra, ta, R @ Ra, R @ ta + t)
        self.assert_close(E, epi.cross_product_matrix(t[..., 0]) @ R, low_tolerance=True)
        # Not normalised: ||E||_F = sqrt(2) ||t||.
        self.assert_close(E.norm(dim=(-2, -1)), (2.0**0.5) * t.norm(dim=(-2, -1)), low_tolerance=True)
        # Swapping the cameras gives E^T, which is far from E on this fixture.
        E_swapped = epi.essential_from_Rt(R @ Ra, R @ ta + t, Ra, ta)
        self.assert_close(E_swapped, E.transpose(-2, -1), low_tolerance=True)
        assert (E - E.transpose(-2, -1)).abs().max() > 0.5

    def test_convention_essential_from_fundamental_K_sides(self, two_view, device, dtype):
        _skip_half(dtype, _HALF_PIXEL_F)
        K1, K2 = two_view["K1"], two_view["K2"]
        E_gt = _gt_essential(two_view)
        F = epi.fundamental_from_essential(E_gt, K1, K2)
        F = F / F[..., 2:, 2:]  # the scale an estimator returns
        # E = K2^T F K1: K1 is the camera of points1.
        E = epi.essential_from_fundamental(F, K1, K2)
        self.assert_close(E, K2.transpose(-2, -1) @ F @ K1)
        n1, n2 = _normalized(K1, two_view["x1"]), _normalized(K2, two_view["x2"])
        E_swapped = epi.essential_from_fundamental(F, K2, K1)
        resid = _epipolar_residual(E / E.norm(), n1, n2).max()
        assert resid < 1e-3 * _epipolar_residual(E_swapped / E_swapped.norm(), n1, n2).max()

    def test_convention_motion_from_essential_candidate_order(self, two_view, device, dtype):
        _skip_half(dtype, _NO_HALF_LU.format("motion_from_essential"))
        R, t = two_view["R"], two_view["t"]
        t_unit = t / t.norm(dim=-2, keepdim=True)
        E = _gt_essential(two_view)
        candidate_sets = []
        for E_in in (E, -E, 3.0 * E):
            Rs, ts = epi.motion_from_essential(E_in)
            R1, R2, t_dec = epi.decompose_essential_matrix(E_in)
            assert Rs.shape == (1, 4, 3, 3) and ts.shape == (1, 4, 3, 1)
            # Order [(R1, t), (R1, -t), (R2, t), (R2, -t)] of decompose_essential_matrix, with a unit t.
            self.assert_close(Rs[:, 0], R1)
            self.assert_close(Rs[:, 1], R1)
            self.assert_close(Rs[:, 2], R2)
            self.assert_close(Rs[:, 3], R2)
            self.assert_close(ts[:, 0], t_dec)
            self.assert_close(ts[:, 1], -t_dec)
            self.assert_close(ts[:, 2], t_dec)
            self.assert_close(ts[:, 3], -t_dec)
            self.assert_close(ts.norm(dim=(-2, -1)), torch.ones(1, 4, device=device, dtype=dtype))
            # Candidates 1 and 2 rebuild E_in itself as [t]x R (up to a positive scale), candidates 0 and 3 its
            # negative. Which of them is the true pose follows the sign and scale of E_in, so no index is fixed.
            unit = E_in * (2.0**0.5) / E_in.norm()
            for i, sign in enumerate((-1.0, 1.0, 1.0, -1.0)):
                self.assert_close(epi.cross_product_matrix(ts[:, i, :, 0]) @ Rs[:, i], sign * unit)
            hits = [
                bool((Rs[:, i] - R).abs().max() < 1e-4 and (ts[:, i] - t_unit).abs().max() < 1e-4) for i in range(4)
            ]
            assert sum(hits) == 1
            candidate_sets.append((Rs[0], ts[0]))
        # E, -E and 3E give the same candidate set.
        Rs0, ts0 = candidate_sets[0]
        for Rs_k, ts_k in candidate_sets[1:]:
            for i in range(4):
                match = (Rs_k - Rs0[i]).abs().amax(dim=(-2, -1)) + (ts_k - ts0[i]).abs().amax(dim=(-2, -1))
                assert match.min() < 1e-4

    def test_convention_choose_solution_recovers_relative_motion(self, two_view, device, dtype):
        _skip_half(dtype, _NO_HALF_LU.format("motion_from_essential_choose_solution"))
        R, t, X = two_view["R"], two_view["t"], two_view["X"]
        K1, K2, x1, x2 = two_view["K1"], two_view["K2"], two_view["x1"], two_view["x2"]
        t_norm = t.norm(dim=-2, keepdim=True)
        E = _gt_essential(two_view)
        # Pixel coordinates in, K1 and K2 applied inside: the pose of camera 2 relative to camera 1 with a unit t,
        # and the points triangulated in the first camera's frame at that scale, all at positive depth.
        for E_in in (E, -E):
            R_out, t_out, X_out = epi.motion_from_essential_choose_solution(E_in, K1, K2, x1, x2)
            self.assert_close(R_out, R, rtol=1e-4, atol=1e-4)
            self.assert_close(t_out, t / t_norm, rtol=1e-4, atol=1e-4)
            self.assert_close(X_out, X / t_norm, rtol=1e-4, atol=1e-3)
            assert (X_out[..., 2] > 0).all()
        # Swapping the images (E^T, K2, K1, x2, x1) returns the inverse motion.
        R_sw, t_sw, _ = epi.motion_from_essential_choose_solution(E.transpose(-2, -1), K2, K1, x2, x1)
        self.assert_close(R_sw, R.transpose(-2, -1), rtol=1e-4, atol=1e-4)
        self.assert_close(t_sw, -R.transpose(-2, -1) @ t / t_norm, rtol=1e-4, atol=1e-4)
        # Control: with K1 and K2 swapped the triangulated points are wrong.
        _, _, X_swapped_k = epi.motion_from_essential_choose_solution(E, K2, K1, x1, x2)
        assert (X_swapped_k - X / t_norm).abs().max() > 1.0

    def test_convention_relative_camera_motion_world_to_camera(self, two_view, device, dtype):
        R, t = two_view["R"], two_view["t"]
        Ra, ta = _first_camera(device, dtype)
        # World-to-camera extrinsics: camera 2 = (R Ra, R ta + t) sits at (R, t) from camera 1 = (Ra, ta), and the
        # result is (R2 R1^T, t2 - R2 R1^T t1).
        R_rel, t_rel = epi.relative_camera_motion(Ra, ta, R @ Ra, R @ ta + t)
        self.assert_close(R_rel, R, low_tolerance=True)
        self.assert_close(t_rel, t, low_tolerance=True)
        # Swapping the cameras gives the inverse motion, far from (R, t) on this fixture.
        R_sw, t_sw = epi.relative_camera_motion(R @ Ra, R @ ta + t, Ra, ta)
        self.assert_close(R_sw, R.transpose(-2, -1), low_tolerance=True)
        self.assert_close(t_sw, -R.transpose(-2, -1) @ t, low_tolerance=True)
        assert (R_sw - R).abs().max() > 0.1 and (t_sw - t).abs().max() > 0.5

    def test_wart_find_essential_ignores_weights_4876(self, two_view, device, dtype):
        _skip_find_essential(device, dtype)
        # #4876: weights is documented per correspondence but ignored: an outlier with weight 0, all-zero weights and
        # all-one weights give the same output (NaN slots compared as 0). Once weights are used these differ.
        n1, n2 = _normalized(two_view["K1"], two_view["x1"]), _normalized(two_view["K2"], two_view["x2"])
        p1 = torch.cat([n1, torch.tensor([[[0.4, -0.3]]], device=device, dtype=dtype)], 1)
        p2 = torch.cat([n2, torch.tensor([[[-0.35, 0.25]]], device=device, dtype=dtype)], 1)
        ones = torch.ones(1, 13, device=device, dtype=dtype)
        outlier_off = ones.clone()
        outlier_off[0, 12] = 0.0
        E_ones = epi.find_essential(p1, p2, ones)
        assert torch.equal(epi.find_essential(p1, p2, outlier_off).nan_to_num(0.0), E_ones.nan_to_num(0.0))
        assert torch.equal(epi.find_essential(p1, p2, torch.zeros_like(ones)).nan_to_num(0.0), E_ones.nan_to_num(0.0))
        # Control: the outlier does move the estimate, so a working weight would change the result.
        E_clean = epi.find_essential(n1, n2)

        def best(E):
            real = E[0, torch.isfinite(E[0]).all(dim=-1).all(dim=-1)]
            return min(_epipolar_residual(e[None], n1, n2).max() for e in real)

        assert best(E_ones) > 1e3 * best(E_clean)

    def test_wart_decompose_unbatched_t_shape_4878(self, two_view, device, dtype):
        _skip_half(dtype, _NO_HALF_LU.format("decompose_essential_matrix"))
        # #4878: an unbatched (3, 3) input gains a batch dim on the rotations but not on t.
        E = _gt_essential(two_view)[0]
        R1, R2, t = epi.decompose_essential_matrix(E)
        assert R1.shape == (1, 3, 3) and R2.shape == (1, 3, 3)
        assert t.shape == (3, 1)
        Rs, ts = epi.motion_from_essential(E)
        assert Rs.shape == (1, 4, 3, 3)
        assert ts.shape == (4, 3, 1)

    def test_wart_choose_solution_all_masked_returns_candidate0_4879(self, two_view, device, dtype):
        _skip_half(dtype, _NO_HALF_LU.format("motion_from_essential_choose_solution"))
        K1, K2, x1, x2 = two_view["K1"], two_view["K2"], two_view["x1"], two_view["x2"]
        R, t = two_view["R"], two_view["t"]
        E = _gt_essential(two_view)
        Rs, ts = epi.motion_from_essential(E)
        t_unit = t / t.norm(dim=-2, keepdim=True)
        # On this E candidate 0 is not the true pose, so returning it is visibly wrong.
        assert (Rs[:, 0] - R).abs().max() > 0.1 or (ts[:, 0] - t_unit).abs().max() > 0.1
        # #4879: with every point masked out no candidate passes the depth test, and candidate 0 comes back unflagged.
        mask = torch.zeros(1, 12, dtype=torch.bool, device=device)
        R_out, t_out, _ = epi.motion_from_essential_choose_solution(E, K1, K2, x1, x2, mask=mask)
        assert torch.equal(R_out, Rs[:, 0]) and torch.equal(t_out, ts[:, 0])
        # Control: one unmasked point is enough to select the true pose.
        mask[0, 5] = True
        R_one, t_one, _ = epi.motion_from_essential_choose_solution(E, K1, K2, x1, x2, mask=mask)
        self.assert_close(R_one, R, rtol=1e-4, atol=1e-4)
        self.assert_close(t_one, t_unit, rtol=1e-4, atol=1e-4)

    def test_wart_decompose_no_svd_batch_non_rotations_4880(self, two_view, device, dtype):
        # #4880: the rotation normaliser sums over the whole batch, so a batch of two copies of E returns non-rotations,
        # while the same E alone returns rotations.
        E = _gt_essential(two_view)
        eye = torch.eye(3, device=device, dtype=torch.float32)

        def orthogonality_error(Rm):
            Rm = Rm.float()
            return (Rm @ Rm.transpose(-2, -1) - eye).norm(dim=(-2, -1))

        R1, R2, _ = epi.decompose_essential_matrix_no_svd(E)
        assert orthogonality_error(R1).max() < 0.25 and orthogonality_error(R2).max() < 0.25
        R1b, R2b, _ = epi.decompose_essential_matrix_no_svd(torch.cat([E, E]))
        assert (orthogonality_error(R1b) > 1.0).all() and (orthogonality_error(R2b) > 1.0).all()

    def test_wart_choose_solution_batched_uses_element0_index_2198(self, two_view, device, dtype):
        _skip_half(dtype, _NO_HALF_LU.format("motion_from_essential_choose_solution"))
        K1, K2, x1, x2 = two_view["K1"], two_view["K2"], two_view["x1"], two_view["x2"]
        R, t = two_view["R"], two_view["t"]
        t_unit = t / t.norm(dim=-2, keepdim=True)
        E = _gt_essential(two_view)

        def is_truth(R_out, t_out):
            return bool((R_out - R[0]).abs().max() < 1e-4 and (t_out - t_unit[0]).abs().max() < 1e-4)

        # E and -E have their true pose at different candidate indices; each alone is recovered.
        for E_in in (E, -E):
            R_out, t_out, _ = epi.motion_from_essential_choose_solution(E_in, K1, K2, x1, x2)
            assert is_truth(R_out[0], t_out[0])
        # #2198: in the batch [E, -E] with the same correspondences, element 1 gets element 0's index: a wrong pose
        # with points behind a camera.
        R_b, t_b, X_b = epi.motion_from_essential_choose_solution(
            torch.cat([E, -E]), K1.expand(2, 3, 3), K2.expand(2, 3, 3), x1.expand(2, 12, 2), x2.expand(2, 12, 2)
        )
        assert is_truth(R_b[0], t_b[0])
        assert not is_truth(R_b[1], t_b[1])
        assert (X_b[1, :, 2] < 0).any()

    def test_wart_find_essential_float32_minimal_sample_4884(self, two_view, device, dtype):
        if dtype != torch.float32:
            pytest.skip("the defect is float32-specific")
        if device.type == "mps":
            pytest.skip("find_essential calls torch.linalg.eigvals, which has no MPS kernel (#4528)")
        # #4884: on the exact five-point sample of the fixture, float32 returns no candidate near the true E, while
        # float64, and float32 with six or more points, recover it to roundoff. Once fixed the nearest is close.
        n1, n2 = _normalized(two_view["K1"], two_view["x1"]), _normalized(two_view["K2"], two_view["x2"])
        E = epi.find_essential(n1[:, :5], n2[:, :5])
        real = E[0, torch.isfinite(E[0]).all(dim=-1).all(dim=-1)]
        E_gt = _gt_essential(two_view)
        E_gt = E_gt / E_gt.norm()
        nearest = torch.minimum((real - E_gt).norm(dim=(-2, -1)), (real + E_gt).norm(dim=(-2, -1))).min()
        assert nearest > 0.05

    def test_wart_find_essential_no_real_root_identity_4883(self, device, dtype):
        _skip_find_essential(device, dtype)
        # #4883: a five-point sample with no real root returns ten identity matrices instead of NaN slots. Sample:
        #   g = torch.Generator().manual_seed(0)
        #   p1 = torch.randn(4000, 5, 2, generator=g, dtype=torch.float64) * 0.5  # then p2 from the same g
        #   p1[1371], p2[1371]
        p1 = torch.tensor(_NO_REAL_ROOT_P1, device=device, dtype=dtype)[None]
        p2 = torch.tensor(_NO_REAL_ROOT_P2, device=device, dtype=dtype)[None]
        E = epi.find_essential(p1, p2)
        assert torch.equal(E, torch.eye(3, device=device, dtype=dtype).expand(1, 10, 3, 3))

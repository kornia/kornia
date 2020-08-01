import pytest

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia.geometry.epipolar as epi

import test_common as utils


class TestEssentialFromFundamental:
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

    def test_from_to_fundamental(self, device, dtype):
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        E_mat = epi.essential_from_fundamental(F_mat, K1, K2)
        F_hat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert_allclose(F_mat, F_hat, atol=1e-4, rtol=1e-4)

    def test_shape_large(self, device, dtype):
        F_mat = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        E_mat = epi.essential_from_fundamental(F_mat, K1, K2)
        assert E_mat.shape == (1, 2, 3, 3)

    def test_from_fundamental(self, device, dtype):

        scene = utils.generate_two_view_random_scene(device, dtype)

        F_mat = scene['F']

        K1 = scene['K1']
        K2 = scene['K2']

        E_mat = epi.essential_from_fundamental(F_mat, K1, K2)
        F_hat = epi.fundamental_from_essential(E_mat, K1, K2)

        F_mat_norm = epi.normalize_transformation(F_mat)
        F_hat_norm = epi.normalize_transformation(F_hat)
        assert_allclose(F_mat_norm, F_hat_norm)

    def test_gradcheck(self, device):
        F_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        K1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        K2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        assert gradcheck(epi.essential_from_fundamental,
                         (F_mat, K1, K2,), raise_exception=True)


class TestRelativeCameraMotion:
    def test_smoke(self, device, dtype):
        R1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        t1 = torch.rand(1, 3, 1, device=device, dtype=dtype)
        R2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        t2 = torch.rand(1, 3, 1, device=device, dtype=dtype)
        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        assert R.shape == (1, 3, 3)
        assert t.shape == (1, 3, 1)

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 8, ])
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
        R1 = torch.tensor([[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]], device=device, dtype=dtype)

        t1 = torch.tensor([[[10.], [0.], [0.]]]).type_as(R1)

        R2 = epi.eye_like(3, R1)
        t2 = epi.vec_like(3, t1)

        R_expected = R1.clone()
        t_expected = -t1

        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        assert_allclose(R_expected, R)
        assert_allclose(t_expected, t)

    def test_rotate_z(self, device, dtype):
        R1 = torch.tensor([[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]], device=device, dtype=dtype)

        R2 = torch.tensor([[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
        ]], device=device, dtype=dtype)

        t1 = epi.vec_like(3, R1)
        t2 = epi.vec_like(3, R2)

        R_expected = R2.clone()
        t_expected = t1

        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        assert_allclose(R_expected, R)
        assert_allclose(t_expected, t)

    def test_gradcheck(self, device):
        R1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        R2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        t1 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        t2 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        assert gradcheck(epi.relative_camera_motion,
                         (R1, t1, R2, t2,), raise_exception=True)


class TestEssentalFromRt:
    def test_smoke(self, device, dtype):
        R1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        t1 = torch.rand(1, 3, 1, device=device, dtype=dtype)
        R2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        t2 = torch.rand(1, 3, 1, device=device, dtype=dtype)
        E_mat = epi.essential_from_Rt(R1, t1, R2, t2)
        assert E_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 8, ])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        R1 = torch.rand(B, 3, 3, device=device, dtype=dtype)
        t1 = torch.rand(B, 3, 1, device=device, dtype=dtype)
        R2 = torch.rand(1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        t2 = torch.rand(B, 3, 1, device=device, dtype=dtype)
        E_mat = epi.essential_from_Rt(R1, t1, R2, t2)
        assert E_mat.shape == (B, 3, 3)

    def test_from_fundamental_Rt(self, device, dtype):

        scene = utils.generate_two_view_random_scene(device, dtype)

        E_from_Rt = epi.essential_from_Rt(
            scene['R1'], scene['t1'], scene['R2'], scene['t2'])

        E_from_F = epi.essential_from_fundamental(
            scene['F'], scene['K1'], scene['K2'])

        E_from_Rt_norm = epi.normalize_transformation(E_from_Rt)
        E_from_F_norm = epi.normalize_transformation(E_from_F)
        assert_allclose(E_from_Rt_norm, E_from_F_norm, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device):
        R1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        R2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        t1 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        t2 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        assert gradcheck(epi.essential_from_Rt,
                         (R1, t1, R2, t2,), raise_exception=True)


class TestDecomposeEssentialMatrix:
    def test_smoke(self, device, dtype):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        R1, R2, t = epi.decompose_essential_matrix(E_mat)
        assert R1.shape == (1, 3, 3)
        assert R2.shape == (1, 3, 3)
        assert t.shape == (1, 3, 1)

    @pytest.mark.parametrize("batch_shape", [
        (1, 3, 3), (2, 3, 3), (2, 1, 3, 3), (3, 2, 1, 3, 3),
    ])
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

        assert gradcheck(eval_rot1, (E_mat,), raise_exception=True)
        assert gradcheck(eval_rot2, (E_mat,), raise_exception=True)
        assert gradcheck(eval_vec, (E_mat,), raise_exception=True)


class TestMotionFromEssential:
    def test_smoke(self, device, dtype):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        Rs, Ts = epi.motion_from_essential(E_mat)
        assert Rs.shape == (1, 4, 3, 3)
        assert Ts.shape == (1, 4, 3, 1)

    @pytest.mark.parametrize("batch_shape", [
        (1, 3, 3), (2, 3, 3), (2, 1, 3, 3), (3, 2, 1, 3, 3),
    ])
    def test_shape(self, batch_shape, device, dtype):
        E_mat = torch.rand(batch_shape, device=device, dtype=dtype)
        Rs, Ts = epi.motion_from_essential(E_mat)
        assert Rs.shape == batch_shape[:-2] + (4, 3, 3)
        assert Ts.shape == batch_shape[:-2] + (4, 3, 1)

    def test_two_view(self, device, dtype):
        scene = utils.generate_two_view_random_scene(device, dtype)

        R1, t1 = scene['R1'], scene['t1']
        R2, t2 = scene['R2'], scene['t2']

        E_mat = epi.essential_from_Rt(R1, t1, R2, t2)

        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        t = torch.nn.functional.normalize(t, dim=1)

        Rs, ts = epi.motion_from_essential(E_mat)

        rot_error = (Rs - R).abs().sum((-2, -1))
        vec_error = (ts - t).abs().sum((-1))

        rtol: float = 1e-4
        assert (rot_error < rtol).any() & (vec_error < rtol).any()

    def test_gradcheck(self, device):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)

        def eval_rot(input):
            return epi.motion_from_essential(input)[0]

        def eval_vec(input):
            return epi.motion_from_essential(input)[1]

        assert gradcheck(eval_rot, (E_mat,), raise_exception=True)
        assert gradcheck(eval_vec, (E_mat,), raise_exception=True)


class TestMotionFromEssentialChooseSolution:
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

    @pytest.mark.parametrize("batch_size, num_points", [
        (1, 3), (2, 3), (2, 8), (3, 2),
    ])
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

    def test_two_view(self, device, dtype):

        scene = utils.generate_two_view_random_scene(device, dtype)

        E_mat = epi.essential_from_Rt(
            scene['R1'], scene['t1'], scene['R2'], scene['t2'])

        R, t = epi.relative_camera_motion(
            scene['R1'], scene['t1'], scene['R2'], scene['t2'])
        t = torch.nn.functional.normalize(t, dim=1)

        R_hat, t_hat, X_hat = epi.motion_from_essential_choose_solution(
            E_mat, scene['K1'], scene['K2'], scene['x1'], scene['x2'])

        assert_allclose(t, t_hat)
        assert_allclose(R, R_hat, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        K1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        K2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        x1 = torch.rand(1, 2, 2, device=device, dtype=torch.float64)
        x2 = torch.rand(1, 2, 2, device=device, dtype=torch.float64)

        assert gradcheck(epi.motion_from_essential_choose_solution,
                         (E_mat, K1, K2, x1, x2,), raise_exception=True)

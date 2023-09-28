from typing import Dict

import pytest
import test_common as utils
import torch
from torch.autograd import gradcheck

import kornia
import kornia.geometry.epipolar as epi
from kornia.testing import assert_close


class TestFindEssential:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 5, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 5, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 5, device=device, dtype=dtype)
        E_mat = epi.essential.find_essential(points1, points2, weights)
        assert E_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 5), (2, 6), (3, 7)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        E_mat = epi.essential.find_essential(points1, points2, weights)
        assert E_mat.shape == (B, 3, 3)

    @pytest.mark.parametrize("batch_size, num_points", [(1, 5), (2, 6), (3, 7)])
    def test_shape_noweights(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = None
        E_mat = epi.essential.find_essential(points1, points2, weights)
        assert E_mat.shape == (B, 3, 3)

    def test_epipolar_constraint(self, device, dtype):
        scene: Dict[str, torch.Tensor] = utils.generate_two_view_random_scene(device, dtype)
        x1 = scene['x1'][:, :5, :]
        x2 = scene['x2'][:, :5, :]
        K1 = scene['K1']
        K2 = scene['K2']

        try:
            import cv2

            calibrated_x1 = torch.from_numpy(
                cv2.undistortPoints(x1.numpy(), cameraMatrix=K1.squeeze().numpy(), distCoeffs=None)
            ).transpose(0, 1)
            calibrated_x2 = torch.from_numpy(
                cv2.undistortPoints(x2.numpy(), cameraMatrix=K2.squeeze().numpy(), distCoeffs=None)
            ).transpose(0, 1)

        except Exception:
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
            mean_error = distance.mean()
            assert_close(mean_error, torch.tensor(0.0, device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    def test_synthetic_sampson(self, device, dtype):
        scene: Dict[str, torch.Tensor] = utils.generate_two_view_random_scene(device, dtype)

        x1 = scene['x1'][:, :5, :]
        x2 = scene['x2'][:, :5, :]
        K1 = scene['K1']
        K2 = scene['K2']

        try:
            import cv2

            calibrated_x1 = torch.from_numpy(
                cv2.undistortPoints(x1.numpy(), cameraMatrix=K1[0].numpy(), distCoeffs=None)
            ).transpose(0, 1)
            calibrated_x2 = torch.from_numpy(
                cv2.undistortPoints(x2.numpy(), cameraMatrix=K2[0].numpy(), distCoeffs=None)
            ).transpose(0, 1)

        except Exception:
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

        weights = torch.ones_like(x1)[..., 0]
        E_est = epi.essential.find_essential(calibrated_x1, calibrated_x2, weights)
        error = epi.sampson_epipolar_distance(calibrated_x1, calibrated_x2, E_est)
        assert_close(error, torch.zeros((x1.shape[:2]), device=device, dtype=dtype), atol=1e-4, rtol=1e-4)


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

    @pytest.mark.xfail(reason="TODO: fix #685")
    def test_from_to_fundamental(self, device, dtype):
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        E_mat = epi.essential_from_fundamental(F_mat, K1, K2)
        F_hat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert_close(F_mat, F_hat, atol=1e-4, rtol=1e-4)

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
        assert_close(F_mat_norm, F_hat_norm)

    def test_gradcheck(self, device):
        F_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        K1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        K2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        assert gradcheck(epi.essential_from_fundamental, (F_mat, K1, K2), raise_exception=True, fast_mode=True)


class TestRelativeCameraMotion:
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

        R2 = kornia.eye_like(3, R1)
        t2 = kornia.vec_like(3, t1)

        R_expected = R1.clone()
        t_expected = -t1

        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        assert_close(R_expected, R)
        assert_close(t_expected, t)

    def test_rotate_z(self, device, dtype):
        R1 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        R2 = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        t1 = kornia.vec_like(3, R1)
        t2 = kornia.vec_like(3, R2)

        R_expected = R2.clone()
        t_expected = t1

        R, t = epi.relative_camera_motion(R1, t1, R2, t2)
        assert_close(R_expected, R)
        assert_close(t_expected, t)

    def test_gradcheck(self, device):
        R1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        R2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        t1 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        t2 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        assert gradcheck(epi.relative_camera_motion, (R1, t1, R2, t2), raise_exception=True, fast_mode=True)


class TestEssentalFromRt:
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
        scene = utils.generate_two_view_random_scene(device, dtype)

        E_from_Rt = epi.essential_from_Rt(scene['R1'], scene['t1'], scene['R2'], scene['t2'])

        E_from_F = epi.essential_from_fundamental(scene['F'], scene['K1'], scene['K2'])

        E_from_Rt_norm = epi.normalize_transformation(E_from_Rt)
        E_from_F_norm = epi.normalize_transformation(E_from_F)
        # TODO: occasionally failed with error > 0.04
        assert_close(E_from_Rt_norm, E_from_F_norm, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device):
        R1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        R2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        t1 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        t2 = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        assert gradcheck(epi.essential_from_Rt, (R1, t1, R2, t2), raise_exception=True, fast_mode=True)


class TestDecomposeEssentialMatrix:
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

        assert gradcheck(eval_rot1, (E_mat,), raise_exception=True, fast_mode=True)
        assert gradcheck(eval_rot2, (E_mat,), raise_exception=True, fast_mode=True)
        assert gradcheck(eval_vec, (E_mat,), raise_exception=True, fast_mode=True)


class TestMotionFromEssential:
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
        scene = utils.generate_two_view_random_scene(device, dtype)

        R1, t1 = scene['R1'], scene['t1']
        R2, t2 = scene['R2'], scene['t2']

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

        assert gradcheck(eval_rot, (E_mat,), raise_exception=True, fast_mode=True)
        assert gradcheck(eval_vec, (E_mat,), raise_exception=True, fast_mode=True)


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

        assert_close(R, Rm)
        assert_close(t, tm)
        assert_close(X, Xm[:, 1:-1, :])

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

        assert_close(R, Rm)
        assert_close(t, tm)
        assert_close(X, Xm[1:-1, :])

    def test_two_view(self, device, dtype):
        scene = utils.generate_two_view_random_scene(device, dtype)

        E_mat = epi.essential_from_Rt(scene['R1'], scene['t1'], scene['R2'], scene['t2'])

        R, t = epi.relative_camera_motion(scene['R1'], scene['t1'], scene['R2'], scene['t2'])
        t = torch.nn.functional.normalize(t, dim=1)

        R_hat, t_hat, _ = epi.motion_from_essential_choose_solution(
            E_mat, scene['K1'], scene['K2'], scene['x1'], scene['x2']
        )

        assert_close(t, t_hat)
        assert_close(R, R_hat, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        K1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        K2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        x1 = torch.rand(1, 2, 2, device=device, dtype=torch.float64)
        x2 = torch.rand(1, 2, 2, device=device, dtype=torch.float64)

        assert gradcheck(
            epi.motion_from_essential_choose_solution, (E_mat, K1, K2, x1, x2), raise_exception=True, fast_mode=True
        )

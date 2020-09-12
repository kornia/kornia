from typing import Optional

import pytest
import numpy as np

import kornia
from kornia.testing import tensor_to_gradcheck_var, create_eye_batch

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


# based on:
# https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/rotation_test.cc#L271

class TestAngleAxisToQuaternion:

    def test_smoke(self, device):
        angle_axis = torch.zeros(3)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert quaternion.shape == (4,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, device, batch_size):
        angle_axis = torch.zeros(batch_size, 3).to(device)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert quaternion.shape == (batch_size, 4)

    def test_zero_angle(self, device):
        angle_axis = torch.tensor([0., 0., 0.]).to(device)
        expected = torch.tensor([1., 0., 0., 0.]).to(device)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion, expected)

    def test_small_angle(self, device, dtype):
        theta = 1e-2
        angle_axis = torch.tensor([theta, 0., 0.]).to(device, dtype)
        expected = torch.tensor([np.cos(theta / 2), np.sin(theta / 2), 0., 0.]).to(device, dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion, expected)

    def test_x_rotation(self, device, dtype):
        half_sqrt2 = 0.5 * np.sqrt(2)
        angle_axis = torch.tensor([kornia.pi / 2, 0., 0.]).to(device, dtype)
        expected = torch.tensor([half_sqrt2, half_sqrt2, 0., 0.]).to(device, dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion, expected)

    def test_gradcheck(self, device):
        eps = 1e-12
        angle_axis = torch.tensor([0., 0., 0.]).to(device) + eps
        angle_axis = tensor_to_gradcheck_var(angle_axis)
        # evaluate function gradient
        assert gradcheck(kornia.angle_axis_to_quaternion, (angle_axis,),
                         raise_exception=True)


class TestRotationMatrixToQuaternion:

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, device, batch_size):
        matrix = torch.zeros(batch_size, 3, 3).to(device)
        quaternion = kornia.rotation_matrix_to_quaternion(matrix)
        assert quaternion.shape == (batch_size, 4)

    def test_identity(self, device):
        matrix = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]).to(device)
        expected = torch.tensor(
            [0., 0., 0., 1.],
        ).to(device)
        quaternion = kornia.rotation_matrix_to_quaternion(matrix)
        assert_allclose(quaternion, expected)

    def test_rot_x_45(self, device):
        matrix = torch.tensor([
            [1., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.],
        ]).to(device)
        pi_half2 = torch.cos(kornia.pi / 4).to(device)
        expected = torch.tensor(
            [pi_half2, 0., 0., pi_half2],
        ).to(device)
        quaternion = kornia.rotation_matrix_to_quaternion(matrix)
        assert_allclose(quaternion, expected)

    def test_back_and_forth(self, device):
        matrix = torch.tensor([
            [1., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.],
        ]).to(device)
        quaternion = kornia.rotation_matrix_to_quaternion(matrix)
        matrix_hat = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(matrix, matrix_hat)

    def test_corner_case(self, device):
        matrix = torch.tensor([
            [-0.7799533010, -0.5432914495, 0.3106555045],
            [0.0492402576, -0.5481169224, -0.8349509239],
            [0.6238971353, -0.6359263659, 0.4542570710]
        ]).to(device)
        quaternion_true = torch.tensor([0.280136495828629, -0.440902262926102,
                                        0.834015488624573, 0.177614107728004]).to(device)
        quaternion = kornia.rotation_matrix_to_quaternion(matrix)
        torch.set_printoptions(precision=10)
        assert_allclose(quaternion_true, quaternion)

    def test_gradcheck(self, device):
        matrix = torch.eye(3).to(device)
        matrix = tensor_to_gradcheck_var(matrix)
        # evaluate function gradient
        assert gradcheck(kornia.rotation_matrix_to_quaternion, (matrix,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        op = kornia.quaternion_log_to_exp
        op_script = torch.jit.script(op)

        quaternion = torch.tensor([0., 0., 1.]).to(device)
        actual = op_script(quaternion)
        expected = op(quaternion)
        assert_allclose(actual, expected)


class TestQuaternionToRotationMatrix:

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, device, batch_size):
        quaternion = torch.zeros(batch_size, 4).to(device)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert matrix.shape == (batch_size, 3, 3)

    def test_unit_quaternion(self, device):
        quaternion = torch.tensor([0., 0., 0., 1.]).to(device)
        expected = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]).to(device)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(matrix, expected)

    def test_x_rotation(self, device):
        quaternion = torch.tensor([1., 0., 0., 0.]).to(device)
        expected = torch.tensor([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.],
        ]).to(device)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(matrix, expected)

    def test_y_rotation(self, device):
        quaternion = torch.tensor([0., 1., 0., 0.]).to(device)
        expected = torch.tensor([
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., -1.],
        ]).to(device)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(matrix, expected)

    def test_z_rotation(self, device):
        quaternion = torch.tensor([0., 0., 1., 0.]).to(device)
        expected = torch.tensor([
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
        ]).to(device)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(matrix, expected)

    def test_gradcheck(self, device):
        quaternion = torch.tensor([0., 0., 0., 1.]).to(device)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(kornia.quaternion_to_rotation_matrix, (quaternion,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(input):
            return kornia.quaternion_to_rotation_matrix(input)

        quaternion = torch.tensor([0., 0., 1., 0.]).to(device)
        actual = op_script(quaternion)
        expected = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(actual, expected)


class TestQuaternionLogToExp:

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, device, batch_size):
        quaternion_log = torch.zeros(batch_size, 3).to(device)
        quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log)
        assert quaternion_exp.shape == (batch_size, 4)

    def test_unit_quaternion(self, device):
        quaternion_log = torch.tensor([0., 0., 0.]).to(device)
        expected = torch.tensor([0., 0., 0., 1.]).to(device)
        assert_allclose(kornia.quaternion_log_to_exp(quaternion_log), expected)

    def test_pi_quaternion(self, device):
        one = torch.tensor(1.).to(device)
        quaternion_log = torch.tensor([1., 0., 0.]).to(device)
        expected = torch.tensor([torch.sin(one), 0., 0., torch.cos(one)]).to(device)
        assert_allclose(kornia.quaternion_log_to_exp(quaternion_log), expected)

    def test_back_and_forth(self, device):
        quaternion_log = torch.tensor([0., 0., 0.]).to(device)
        quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log)
        quaternion_log_hat = kornia.quaternion_exp_to_log(quaternion_exp)
        assert_allclose(quaternion_log, quaternion_log_hat)

    def test_gradcheck(self, device):
        quaternion = torch.tensor([0., 0., 1.]).to(device)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(kornia.quaternion_log_to_exp, (quaternion,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        op = kornia.quaternion_log_to_exp
        op_script = torch.jit.script(op)

        quaternion = torch.tensor([0., 0., 1.]).to(device)
        actual = op_script(quaternion)
        expected = op(quaternion)
        assert_allclose(actual, expected)


class TestQuaternionExpToLog:

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, device, batch_size):
        quaternion_exp = torch.zeros(batch_size, 4).to(device)
        quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp)
        assert quaternion_log.shape == (batch_size, 3)

    def test_unit_quaternion(self, device):
        quaternion_exp = torch.tensor([0., 0., 0., 1.]).to(device)
        expected = torch.tensor([0., 0., 0.]).to(device)
        assert_allclose(kornia.quaternion_exp_to_log(quaternion_exp), expected)

    def test_pi_quaternion(self, device):
        quaternion_exp = torch.tensor([1., 0., 0., 0.]).to(device)
        expected = torch.tensor([kornia.pi / 2, 0., 0.]).to(device)
        assert_allclose(kornia.quaternion_exp_to_log(quaternion_exp), expected)

    def test_back_and_forth(self, device):
        quaternion_exp = torch.tensor([1., 0., 0., 0.]).to(device)
        quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp)
        quaternion_exp_hat = kornia.quaternion_log_to_exp(quaternion_log)
        assert_allclose(quaternion_exp, quaternion_exp_hat)

    def test_gradcheck(self, device):
        quaternion = torch.tensor([1., 0., 0., 0.]).to(device)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(kornia.quaternion_exp_to_log, (quaternion,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        op = kornia.quaternion_exp_to_log
        op_script = torch.jit.script(op)

        quaternion = torch.tensor([0., 0., 1., 0.]).to(device)
        actual = op_script(quaternion)
        expected = op(quaternion)
        assert_allclose(actual, expected)


class TestQuaternionToAngleAxis:

    def test_smoke(self, device):
        quaternion = torch.zeros(4).to(device)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert angle_axis.shape == (3,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, device, batch_size):
        quaternion = torch.zeros(batch_size, 4).to(device)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert angle_axis.shape == (batch_size, 3)

    def test_unit_quaternion(self, device):
        quaternion = torch.tensor([1., 0., 0., 0.]).to(device)
        expected = torch.tensor([0., 0., 0.]).to(device)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_y_rotation(self, device):
        quaternion = torch.tensor([0., 0., 1., 0.]).to(device)
        expected = torch.tensor([0., kornia.pi, 0.]).to(device)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_z_rotation(self, device, dtype):
        quaternion = torch.tensor([np.sqrt(3) / 2, 0., 0., 0.5]).to(device, dtype)
        expected = torch.tensor([0., 0., kornia.pi / 3]).to(device, dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_small_angle(self, device, dtype):
        theta = 1e-2
        quaternion = torch.tensor([np.cos(theta / 2), np.sin(theta / 2), 0., 0.]).to(device, dtype)
        expected = torch.tensor([theta, 0., 0.]).to(device, dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_gradcheck(self, device):
        eps = 1e-12
        quaternion = torch.tensor([1., 0., 0., 0.]).to(device) + eps
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(kornia.quaternion_to_angle_axis, (quaternion,),
                         raise_exception=True)


def test_pi():
    assert_allclose(kornia.pi, 3.141592)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_rad2deg(batch_shape, device):
    # generate input data
    x_rad = kornia.pi * torch.rand(batch_shape)
    x_rad = x_rad.to(device)

    # convert radians/degrees
    x_deg = kornia.rad2deg(x_rad)
    x_deg_to_rad = kornia.deg2rad(x_deg)

    # compute error
    assert_allclose(x_rad, x_deg_to_rad)

    # evaluate function gradient
    assert gradcheck(kornia.rad2deg, (tensor_to_gradcheck_var(x_rad),),
                     raise_exception=True)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_deg2rad(batch_shape, device):
    # generate input data
    x_deg = 180. * torch.rand(batch_shape)
    x_deg = x_deg.to(device)

    # convert radians/degrees
    x_rad = kornia.deg2rad(x_deg)
    x_rad_to_deg = kornia.rad2deg(x_rad)

    assert_allclose(x_deg, x_rad_to_deg)

    assert gradcheck(kornia.deg2rad, (tensor_to_gradcheck_var(x_deg),),
                     raise_exception=True)


class TestConvertPointsToHomogeneous:
    def test_convert_points(self, device, dtype):
        # generate input data
        points_h = torch.tensor([
            [1., 2., 1.],
            [0., 1., 2.],
            [2., 1., 0.],
            [-1., -2., -1.],
            [0., 1., -2.],
        ], device=device, dtype=dtype)

        expected = torch.tensor([
            [1., 2., 1., 1.],
            [0., 1., 2., 1.],
            [2., 1., 0., 1.],
            [-1., -2., -1., 1.],
            [0., 1., -2., 1.],
        ], device=device, dtype=dtype)

        # to euclidean
        points = kornia.convert_points_to_homogeneous(points_h)
        assert_allclose(points, expected)

    def test_convert_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[
            [2., 1., 0.],
        ], [
            [0., 1., 2.],
        ], [
            [0., 1., -2.],
        ]], device=device, dtype=dtype)

        expected = torch.tensor([[
            [2., 1., 0., 1.],
        ], [
            [0., 1., 2., 1.],
        ], [
            [0., 1., -2., 1.],
        ]], device=device, dtype=dtype)

        # to euclidean
        points = kornia.convert_points_to_homogeneous(points_h)
        assert_allclose(points, expected)

    @pytest.mark.parametrize("batch_shape", [
        (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
    def test_gradcheck(self, device, batch_shape):
        points_h = torch.rand(batch_shape).to(device)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(kornia.convert_points_to_homogeneous, (points_h,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        op = kornia.convert_points_to_homogeneous
        op_script = torch.jit.script(op)

        points_h = torch.zeros(1, 2, 3).to(device)
        actual = op_script(points_h)
        expected = op(points_h)

        assert_allclose(actual, expected)


class TestConvertAtoH:
    def test_convert_points(self, device):
        # generate input data
        A = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
        ]).to(device).view(1, 2, 3)

        expected = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]).to(device).view(1, 3, 3)

        # to euclidean
        H = kornia.geometry.conversions.convert_affinematrix_to_homography(A)
        assert_allclose(H, expected)

    @pytest.mark.parametrize("batch_shape", [
        (10, 2, 3), (16, 2, 3)])
    def test_gradcheck(self, device, batch_shape):
        points_h = torch.rand(batch_shape).to(device)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(kornia.convert_affinematrix_to_homography, (points_h,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        op = kornia.convert_affinematrix_to_homography
        op_script = torch.jit.script(op)

        points_h = torch.zeros(1, 2, 3).to(device)
        actual = op_script(points_h)
        expected = op(points_h)

        assert_allclose(actual, expected)


class TestConvertPointsFromHomogeneous:
    def test_convert_points(self, device):
        # generate input data
        points_h = torch.tensor([
            [1., 2., 1.],
            [0., 1., 2.],
            [2., 1., 0.],
            [-1., -2., -1.],
            [0., 1., -2.],
        ]).to(device)

        expected = torch.tensor([
            [1., 2.],
            [0., 0.5],
            [2., 1.],
            [1., 2.],
            [0., -0.5],
        ]).to(device)

        # to euclidean
        points = kornia.convert_points_from_homogeneous(points_h)
        assert_allclose(points, expected)

    def test_convert_points_batch(self, device):
        # generate input data
        points_h = torch.tensor([[
            [2., 1., 0.],
        ], [
            [0., 1., 2.],
        ], [
            [0., 1., -2.],
        ]]).to(device)

        expected = torch.tensor([[
            [2., 1.],
        ], [
            [0., 0.5],
        ], [
            [0., -0.5],
        ]]).to(device)

        # to euclidean
        points = kornia.convert_points_from_homogeneous(points_h)
        assert_allclose(points, expected)

    @pytest.mark.parametrize("batch_shape", [
        (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
    def test_gradcheck(self, device, batch_shape):
        points_h = torch.rand(batch_shape).to(device)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(kornia.convert_points_from_homogeneous, (points_h,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        op = kornia.convert_points_from_homogeneous
        op_script = torch.jit.script(op)

        points_h = torch.zeros(1, 2, 3).to(device)
        actual = op_script(points_h)
        expected = op(points_h)

        assert_allclose(actual, expected)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_angle_axis_to_rotation_matrix(batch_size, device):
    # generate input data
    angle_axis = torch.rand(batch_size, 3).to(device)
    eye_batch = create_eye_batch(batch_size, 3).to(device)

    # apply transform
    rotation_matrix = kornia.angle_axis_to_rotation_matrix(angle_axis)

    rotation_matrix_eye = torch.matmul(
        rotation_matrix, rotation_matrix.transpose(1, 2))
    assert_allclose(rotation_matrix_eye, eye_batch)

    # evaluate function gradient
    angle_axis = tensor_to_gradcheck_var(angle_axis)  # to var
    assert gradcheck(kornia.angle_axis_to_rotation_matrix, (angle_axis,),
                     raise_exception=True)


'''@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rotation_matrix_to_angle_axis_gradcheck(batch_size, device_type):
    # generate input data
    rmat = torch.rand(batch_size, 3, 3).to(torch.device(device_type))

    # evaluate function gradient
    rmat = tensor_to_gradcheck_var(rmat)  # to var
    assert gradcheck(kornia.rotation_matrix_to_angle_axis,
                     (rmat,), raise_exception=True)'''


'''def test_rotation_matrix_to_angle_axis(device_type):
    device = torch.device(device_type)
    rmat_1 = torch.tensor([[-0.30382753, -0.95095137, -0.05814062],
                           [-0.71581715, 0.26812278, -0.64476041],
                           [0.62872461, -0.15427791, -0.76217038]])
    rvec_1 = torch.tensor([1.50485376, -2.10737739, 0.7214174])

    rmat_2 = torch.tensor([[0.6027768, -0.79275544, -0.09054801],
                           [-0.67915707, -0.56931658, 0.46327563],
                           [-0.41881476, -0.21775548, -0.88157628]])
    rvec_2 = torch.tensor([-2.44916812, 1.18053411, 0.4085298])
    rmat = torch.stack([rmat_2, rmat_1], dim=0).to(device)
    rvec = torch.stack([rvec_2, rvec_1], dim=0).to(device)

    assert_allclose(kornia.rotation_matrix_to_angle_axis(rmat), rvec)'''


class TestNormalizePixelCoordinates:
    def test_tensor_bhw2(self, device):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False).to(device)

        expected = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True).to(device)

        grid_norm = kornia.normalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(grid_norm, expected)

    def test_list(self, device):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False).to(device)
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True).to(device)
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.normalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(grid_norm, expected)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        op = kornia.normalize_pixel_coordinates
        op_script = torch.jit.script(op)

        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True).to(device)

        actual = op_script(grid, height, width)
        expected = op(grid, height, width)

        assert_allclose(actual, expected)


class TestDenormalizePixelCoordinates:
    def test_tensor_bhw2(self, device):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True).to(device)

        expected = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False).to(device)

        grid_norm = kornia.denormalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(grid_norm, expected)

    def test_list(self, device):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True).to(device)
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False).to(device)
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.denormalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(grid_norm, expected)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        op = kornia.denormalize_pixel_coordinates
        op_script = torch.jit.script(op)

        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True).to(device)

        actual = op_script(grid, height, width)
        expected = op(grid, height, width)

        assert_allclose(actual, expected)

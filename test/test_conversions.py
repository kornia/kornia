import pytest
import numpy as np

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck
from torch.testing import assert_allclose

from torch.testing import assert_allclose
from utils import check_equal_torch, check_equal_numpy
from common import device_type
import utils


# based on:
# https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/rotation_test.cc#L271

class TestAngleAxisToQuaternion:

    def test_smoke(self):
        angle_axis = torch.zeros(3)
        quaternion = tgm.angle_axis_to_quaternion(angle_axis)
        assert quaternion.shape == (4,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size):
        angle_axis = torch.zeros(batch_size, 3)
        quaternion = tgm.angle_axis_to_quaternion(angle_axis)
        assert quaternion.shape == (batch_size, 4)

    def test_zero_angle(self):
        angle_axis = torch.Tensor([0, 0, 0])
        expected = torch.Tensor([1, 0, 0, 0])
        quaternion = tgm.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion, expected)

    def test_small_angle(self):
        theta = 1e-2
        angle_axis = torch.Tensor([theta, 0, 0])
        expected = torch.Tensor([np.cos(theta / 2), np.sin(theta / 2), 0, 0])
        quaternion = tgm.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion, expected)

    def test_x_rotation(self):
        half_sqrt2 = 0.5 * np.sqrt(2)
        angle_axis = torch.Tensor([tgm.pi / 2, 0, 0])
        expected = torch.Tensor([half_sqrt2, half_sqrt2, 0, 0])
        quaternion = tgm.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion, expected)

    def test_gradcheck(self):
        eps = 1e-12
        angle_axis = torch.Tensor([0, 0, 0]) + eps
        angle_axis = utils.tensor_to_gradcheck_var(angle_axis)
        # evaluate function gradient
        assert gradcheck(tgm.angle_axis_to_quaternion, (angle_axis,),
                         raise_exception=True)


class TestQuaternionToAngleAxis:

    def test_smoke(self):
        quaternion = torch.zeros(4)
        angle_axis = tgm.quaternion_to_angle_axis(quaternion)
        assert angle_axis.shape == (3,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size):
        quaternion = torch.zeros(batch_size, 4)
        angle_axis = tgm.quaternion_to_angle_axis(quaternion)
        assert angle_axis.shape == (batch_size, 3)

    def test_unit_quaternion(self):
        quaternion = torch.Tensor([1, 0, 0, 0])
        expected = torch.Tensor([0, 0, 0])
        angle_axis = tgm.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_y_rotation(self):
        quaternion = torch.Tensor([0, 0, 1, 0])
        expected = torch.Tensor([0, tgm.pi, 0])
        angle_axis = tgm.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_z_rotation(self):
        quaternion = torch.Tensor([np.sqrt(3) / 2, 0, 0, 0.5])
        expected = torch.Tensor([0, 0, tgm.pi / 3])
        angle_axis = tgm.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_small_angle(self):
        theta = 1e-2
        quaternion = torch.Tensor([np.cos(theta / 2), np.sin(theta / 2), 0, 0])
        expected = torch.Tensor([theta, 0, 0])
        angle_axis = tgm.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_gradcheck(self):
        eps = 1e-12
        quaternion = torch.Tensor([1, 0, 0, 0]) + eps
        quaternion = utils.tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(tgm.quaternion_to_angle_axis, (quaternion,),
                         raise_exception=True)


def test_pi():
    assert pytest.approx(tgm.pi.item(), 3.141592)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_rad2deg(batch_shape, device_type):
    # generate input data
    x_rad = tgm.pi * torch.rand(batch_shape)
    x_rad = x_rad.to(torch.device(device_type))

    # convert radians/degrees
    x_deg = tgm.rad2deg(x_rad)
    x_deg_to_rad = tgm.deg2rad(x_deg)

    # compute error
    error = utils.compute_mse(x_rad, x_deg_to_rad)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(x_deg, tgm.RadToDeg()(x_rad))

    # evaluate function gradient
    assert gradcheck(tgm.rad2deg, (utils.tensor_to_gradcheck_var(x_rad),),
                     raise_exception=True)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_deg2rad(batch_shape, device_type):
    # generate input data
    x_deg = 180. * torch.rand(batch_shape)
    x_deg = x_deg.to(torch.device(device_type))

    # convert radians/degrees
    x_rad = tgm.deg2rad(x_deg)
    x_rad_to_deg = tgm.rad2deg(x_rad)

    # compute error
    error = utils.compute_mse(x_deg, x_rad_to_deg)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(x_rad, tgm.DegToRad()(x_deg))

    assert gradcheck(tgm.deg2rad, (utils.tensor_to_gradcheck_var(x_deg),),
                     raise_exception=True)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_convert_points_to_homogeneous(batch_shape, device_type):
    # generate input data
    points = torch.rand(batch_shape)
    points = points.to(torch.device(device_type))

    # to homogeneous
    points_h = tgm.convert_points_to_homogeneous(points)

    assert points_h.shape[-2] == batch_shape[-2]
    assert (points_h[..., -1] == torch.ones(points_h[..., -1].shape)).all()

    # functional
    assert torch.allclose(points_h, tgm.ConvertPointsToHomogeneous()(points))

    # evaluate function gradient
    points = utils.tensor_to_gradcheck_var(points)  # to var
    assert gradcheck(tgm.convert_points_to_homogeneous, (points,),
                     raise_exception=True)


class TestConvertPointsFromHomogeneous:
    def test_convert_points(self, device_type):
        # generate input data
        points_h = torch.FloatTensor([
            [1, 2, 1],
            [0, 1, 2],
            [2, 1, 0],
        ]).to(torch.device(device_type))

        expected = torch.FloatTensor([
            [1, 2],
            [0, 0.5],
            [2e6, 1e6],
        ]).to(torch.device(device_type))

        # to euclidean
        points = tgm.convert_points_from_homogeneous(points_h)
        assert_allclose(points, expected)

    def test_convert_points_batch(self, device_type):
        # generate input data
        points_h = torch.FloatTensor([[
            [2, 1, 0],
        ], [
            [0, 1, 2],
        ]]).to(torch.device(device_type))

        expected = torch.FloatTensor([[
            [2e6, 1e6],
        ], [
            [0, 0.5],
        ]]).to(torch.device(device_type))

        # to euclidean
        points = tgm.convert_points_from_homogeneous(points_h)
        assert_allclose(points, expected)

    @pytest.mark.parametrize("batch_shape", [
        (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
    def test_gradcheck(self, batch_shape):
        points_h = torch.rand(batch_shape)

        # evaluate function gradient
        points_h = utils.tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(tgm.convert_points_from_homogeneous, (points_h,),
                         raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_angle_axis_to_rotation_matrix(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)
    angle_axis = torch.rand(batch_size, 3).to(device)
    eye_batch = utils.create_eye_batch(batch_size, 4).to(device)

    # apply transform
    rotation_matrix = tgm.angle_axis_to_rotation_matrix(angle_axis)

    rotation_matrix_eye = torch.matmul(
        rotation_matrix, rotation_matrix.transpose(1, 2))
    assert check_equal_torch(rotation_matrix_eye, eye_batch)

    # evaluate function gradient
    angle_axis = utils.tensor_to_gradcheck_var(angle_axis)  # to var
    assert gradcheck(tgm.angle_axis_to_rotation_matrix, (angle_axis,),
                     raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rtvec_to_pose_gradcheck(batch_size, device_type):
    # generate input data
    rtvec = torch.rand(batch_size, 6).to(torch.device(device_type))

    # evaluate function gradient
    rtvec = utils.tensor_to_gradcheck_var(rtvec)  # to var
    assert gradcheck(tgm.rtvec_to_pose, (rtvec,), raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rotation_matrix_to_angle_axis_gradcheck(batch_size, device_type):
    # generate input data
    rmat = torch.rand(batch_size, 3, 4).to(torch.device(device_type))

    # evaluate function gradient
    rmat = utils.tensor_to_gradcheck_var(rmat)  # to var
    assert gradcheck(tgm.rotation_matrix_to_angle_axis,
                     (rmat,), raise_exception=True)


def test_rotation_matrix_to_angle_axis(device_type):
    device = torch.device(device_type)
    rmat_1 = torch.tensor([[-0.30382753, -0.95095137, -0.05814062, 0.],
                           [-0.71581715, 0.26812278, -0.64476041, 0.],
                           [0.62872461, -0.15427791, -0.76217038, 0.]])
    rvec_1 = torch.tensor([1.50485376, -2.10737739, 0.7214174])

    rmat_2 = torch.tensor([[0.6027768, -0.79275544, -0.09054801, 0.],
                           [-0.67915707, -0.56931658, 0.46327563, 0.],
                           [-0.41881476, -0.21775548, -0.88157628, 0.]])
    rvec_2 = torch.tensor([-2.44916812, 1.18053411, 0.4085298])
    rmat = torch.stack([rmat_2, rmat_1], dim=0).to(device)
    rvec = torch.stack([rvec_2, rvec_1], dim=0).to(device)

    assert check_equal_torch(tgm.rotation_matrix_to_angle_axis(rmat), rvec)

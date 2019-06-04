import pytest
import numpy as np

import kornia
from kornia.testing import tensor_to_gradcheck_var, create_eye_batch
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


# based on:
# https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/rotation_test.cc#L271

class TestAngleAxisToQuaternion:

    def test_smoke(self):
        angle_axis = torch.zeros(3)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert quaternion.shape == (4,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size):
        angle_axis = torch.zeros(batch_size, 3)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert quaternion.shape == (batch_size, 4)

    def test_zero_angle(self):
        angle_axis = torch.Tensor([0, 0, 0])
        expected = torch.Tensor([1, 0, 0, 0])
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion, expected)

    def test_small_angle(self):
        theta = 1e-2
        angle_axis = torch.Tensor([theta, 0, 0])
        expected = torch.Tensor([np.cos(theta / 2), np.sin(theta / 2), 0, 0])
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion, expected)

    def test_x_rotation(self):
        half_sqrt2 = 0.5 * np.sqrt(2)
        angle_axis = torch.Tensor([kornia.pi / 2, 0, 0])
        expected = torch.Tensor([half_sqrt2, half_sqrt2, 0, 0])
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion, expected)

    def test_gradcheck(self):
        eps = 1e-12
        angle_axis = torch.Tensor([0, 0, 0]) + eps
        angle_axis = tensor_to_gradcheck_var(angle_axis)
        # evaluate function gradient
        assert gradcheck(kornia.angle_axis_to_quaternion, (angle_axis,),
                         raise_exception=True)


class TestQuaternionToRotationMatrix:

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size):
        quaternion = torch.zeros(batch_size, 4)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert matrix.shape == (batch_size, 3, 3)

    def test_unit_quaternion(self):
        quaternion = torch.tensor([0., 0., 0., 1.])
        expected = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(matrix, expected)

    def test_x_rotation(self):
        quaternion = torch.tensor([1., 0., 0., 0.])
        expected = torch.tensor([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.],
        ])
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(matrix, expected)

    def test_y_rotation(self):
        quaternion = torch.tensor([0., 1., 0., 0.])
        expected = torch.tensor([
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., -1.],
        ])
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(matrix, expected)

    def test_z_rotation(self):
        quaternion = torch.tensor([0., 0., 1., 0.])
        expected = torch.tensor([
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
        ])
        matrix = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(matrix, expected)

    def test_gradcheck(self):
        quaternion = torch.tensor([0., 0., 0., 1.])
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(kornia.quaternion_to_rotation_matrix, (quaternion,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.quaternion_to_rotation_matrix(input)

        quaternion = torch.tensor([0., 0., 1., 0.])
        actual = op_script(quaternion)
        expected = kornia.quaternion_to_rotation_matrix(quaternion)
        assert_allclose(actual, expected)


class TestQuaternionToAngleAxis:

    def test_smoke(self):
        quaternion = torch.zeros(4)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert angle_axis.shape == (3,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size):
        quaternion = torch.zeros(batch_size, 4)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert angle_axis.shape == (batch_size, 3)

    def test_unit_quaternion(self):
        quaternion = torch.Tensor([1, 0, 0, 0])
        expected = torch.Tensor([0, 0, 0])
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_y_rotation(self):
        quaternion = torch.Tensor([0, 0, 1, 0])
        expected = torch.Tensor([0, kornia.pi, 0])
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_z_rotation(self):
        quaternion = torch.Tensor([np.sqrt(3) / 2, 0, 0, 0.5])
        expected = torch.Tensor([0, 0, kornia.pi / 3])
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_small_angle(self):
        theta = 1e-2
        quaternion = torch.Tensor([np.cos(theta / 2), np.sin(theta / 2), 0, 0])
        expected = torch.Tensor([theta, 0, 0])
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis, expected)

    def test_gradcheck(self):
        eps = 1e-12
        quaternion = torch.Tensor([1, 0, 0, 0]) + eps
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(kornia.quaternion_to_angle_axis, (quaternion,),
                         raise_exception=True)


def test_pi():
    assert_allclose(kornia.pi, 3.141592)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_rad2deg(batch_shape, device_type):
    # generate input data
    x_rad = kornia.pi * torch.rand(batch_shape)
    x_rad = x_rad.to(torch.device(device_type))

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
def test_deg2rad(batch_shape, device_type):
    # generate input data
    x_deg = 180. * torch.rand(batch_shape)
    x_deg = x_deg.to(torch.device(device_type))

    # convert radians/degrees
    x_rad = kornia.deg2rad(x_deg)
    x_rad_to_deg = kornia.rad2deg(x_rad)

    assert_allclose(x_deg, x_rad_to_deg)

    assert gradcheck(kornia.deg2rad, (tensor_to_gradcheck_var(x_deg),),
                     raise_exception=True)


class TestConvertPointsToHomogeneous:
    def test_convert_points(self, device_type):
        # generate input data
        points_h = torch.tensor([
            [1., 2., 1.],
            [0., 1., 2.],
            [2., 1., 0.],
            [-1., -2., -1.],
            [0., 1., -2.],
        ]).to(torch.device(device_type))

        expected = torch.tensor([
            [1., 2., 1., 1.],
            [0., 1., 2., 1.],
            [2., 1., 0., 1.],
            [-1., -2., -1., 1.],
            [0., 1., -2., 1.],
        ]).to(torch.device(device_type))

        # to euclidean
        points = kornia.convert_points_to_homogeneous(points_h)
        assert_allclose(points, expected)

    def test_convert_points_batch(self, device_type):
        # generate input data
        points_h = torch.tensor([[
            [2., 1., 0.],
        ], [
            [0., 1., 2.],
        ], [
            [0., 1., -2.],
        ]]).to(torch.device(device_type))

        expected = torch.tensor([[
            [2., 1., 0., 1.],
        ], [
            [0., 1., 2., 1.],
        ], [
            [0., 1., -2., 1.],
        ]]).to(torch.device(device_type))

        # to euclidean
        points = kornia.convert_points_to_homogeneous(points_h)
        assert_allclose(points, expected)

    @pytest.mark.parametrize("batch_shape", [
        (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
    def test_gradcheck(self, batch_shape):
        points_h = torch.rand(batch_shape)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(kornia.convert_points_to_homogeneous, (points_h,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.convert_points_to_homogeneous(input)

        points_h = torch.zeros(1, 2, 3)
        actual = op_script(points_h)
        expected = kornia.convert_points_to_homogeneous(points_h)

        assert_allclose(actual, expected)


class TestConvertPointsFromHomogeneous:
    def test_convert_points(self, device_type):
        # generate input data
        points_h = torch.FloatTensor([
            [1, 2, 1],
            [0, 1, 2],
            [2, 1, 0],
            [-1, -2, -1],
            [0, 1, -2],
        ]).to(torch.device(device_type))

        expected = torch.FloatTensor([
            [1, 2],
            [0, 0.5],
            [2, 1],
            [1, 2],
            [0, -0.5],
        ]).to(torch.device(device_type))

        # to euclidean
        points = kornia.convert_points_from_homogeneous(points_h)
        assert_allclose(points, expected)

    def test_convert_points_batch(self, device_type):
        # generate input data
        points_h = torch.FloatTensor([[
            [2, 1, 0],
        ], [
            [0, 1, 2],
        ], [
            [0, 1, -2],
        ]]).to(torch.device(device_type))

        expected = torch.FloatTensor([[
            [2, 1],
        ], [
            [0, 0.5],
        ], [
            [0, -0.5],
        ]]).to(torch.device(device_type))

        # to euclidean
        points = kornia.convert_points_from_homogeneous(points_h)
        assert_allclose(points, expected)

    @pytest.mark.parametrize("batch_shape", [
        (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
    def test_gradcheck(self, batch_shape):
        points_h = torch.rand(batch_shape)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(kornia.convert_points_from_homogeneous, (points_h,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.convert_points_from_homogeneous(input)

        points_h = torch.zeros(1, 2, 3)
        actual = op_script(points_h)
        expected = kornia.convert_points_from_homogeneous(points_h)

        assert_allclose(actual, expected)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_angle_axis_to_rotation_matrix(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)
    angle_axis = torch.rand(batch_size, 3).to(device)
    eye_batch = create_eye_batch(batch_size, 4).to(device)

    # apply transform
    rotation_matrix = kornia.angle_axis_to_rotation_matrix(angle_axis)

    rotation_matrix_eye = torch.matmul(
        rotation_matrix, rotation_matrix.transpose(1, 2))
    assert_allclose(rotation_matrix_eye, eye_batch)

    # evaluate function gradient
    angle_axis = tensor_to_gradcheck_var(angle_axis)  # to var
    assert gradcheck(kornia.angle_axis_to_rotation_matrix, (angle_axis,),
                     raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rtvec_to_pose_gradcheck(batch_size, device_type):
    # generate input data
    rtvec = torch.rand(batch_size, 6).to(torch.device(device_type))

    # evaluate function gradient
    rtvec = tensor_to_gradcheck_var(rtvec)  # to var
    assert gradcheck(kornia.rtvec_to_pose, (rtvec,), raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rotation_matrix_to_angle_axis_gradcheck(batch_size, device_type):
    # generate input data
    rmat = torch.rand(batch_size, 3, 4).to(torch.device(device_type))

    # evaluate function gradient
    rmat = tensor_to_gradcheck_var(rmat)  # to var
    assert gradcheck(kornia.rotation_matrix_to_angle_axis,
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

    assert_allclose(kornia.rotation_matrix_to_angle_axis(rmat), rvec)


class TestNormalizePixelCoordinates:
    def test_tensor_bhw2(self):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)

        expected = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True)

        grid_norm = kornia.normalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(grid_norm, expected)

    def test_list(self):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True)
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.normalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(grid_norm, expected)

    def test_jit(self):
        @torch.jit.script
        def op_script(input: torch.Tensor, height: int,
                      width: int) -> torch.Tensor:
            return kornia.normalize_pixel_coordinates(input, height, width)
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)

        actual = op_script(grid, height, width)
        expected = kornia.normalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(actual, expected)

    def test_jit_trace(self):
        @torch.jit.script
        def op_script(input, height, width):
            return kornia.normalize_pixel_coordinates(input, height, width)
        # 1. Trace op
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        op_traced = torch.jit.trace(
            op_script,
            (grid, torch.tensor(height), torch.tensor(width),))

        # 2. Generate new input
        height, width = 2, 5
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False).repeat(2, 1, 1, 1)

        # 3. Evaluate
        actual = op_traced(
            grid, torch.tensor(height), torch.tensor(width))
        expected = kornia.normalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(actual, expected)


class TestDenormalizePixelCoordinates:
    def test_tensor_bhw2(self):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True)

        expected = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)

        grid_norm = kornia.denormalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(grid_norm, expected)

    def test_list(self):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True)
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.denormalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(grid_norm, expected)

    def test_jit(self):
        @torch.jit.script
        def op_script(input: torch.Tensor, height: int,
                      width: int) -> torch.Tensor:
            return kornia.denormalize_pixel_coordinates(input, height, width)
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True)

        actual = op_script(grid, height, width)
        expected = kornia.denormalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(actual, expected)

from functools import partial

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.geometry.conversions import (
    ARKitQTVecs_to_ColmapQTVecs,
    QuaternionCoeffOrder,
    Rt_to_matrix4x4,
    camtoworld_graphics_to_vision_4x4,
    camtoworld_graphics_to_vision_Rt,
    camtoworld_to_worldtocam_Rt,
    camtoworld_vision_to_graphics_4x4,
    camtoworld_vision_to_graphics_Rt,
    euler_from_quaternion,
    matrix4x4_to_Rt,
    quaternion_from_euler,
    worldtocam_to_camtoworld_Rt,
)
from kornia.geometry.quaternion import Quaternion
from kornia.testing import BaseTester, assert_close, create_eye_batch, tensor_to_gradcheck_var


@pytest.fixture
def atol(device, dtype):
    """Lower tolerance for cuda-float16 only."""
    if 'cuda' in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


@pytest.fixture
def rtol(device, dtype):
    """Lower tolerance for cuda-float16 only."""
    if 'cuda' in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


# based on:
# https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/rotation_test.cc#L271


class TestAngleAxisToQuaternion:
    def test_smoke_xyzw(self, device, dtype):
        angle_axis = torch.zeros(3, dtype=dtype, device=device)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert quaternion.shape == (4,)

    def test_smoke(self, device, dtype):
        angle_axis = torch.zeros(3, dtype=dtype, device=device)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion.shape == (4,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        angle_axis = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert quaternion.shape == (batch_size, 4)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        angle_axis = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion.shape == (batch_size, 4)

    def test_zero_angle_xyzw(self, device, dtype, atol, rtol):
        angle_axis = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_zero_angle(self, device, dtype, atol, rtol):
        angle_axis = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_x_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.sin(theta / 2.0), 0.0, 0.0, np.cos(theta / 2.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_y_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, np.sin(theta / 2.0), 0.0, np.cos(theta / 2.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_z_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, np.sin(theta / 2.0), np.cos(theta / 2.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_x_rotation_xyzw(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, 0.0, half_sqrt2), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, half_sqrt2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_y_rotation_xyzw(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, half_sqrt2, 0.0, half_sqrt2), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, half_sqrt2, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_z_rotation_xyzw(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, half_sqrt2, half_sqrt2), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, 0.0, half_sqrt2), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        angle_axis = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        angle_axis = tensor_to_gradcheck_var(angle_axis)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.geometry.conversions.angle_axis_to_quaternion, order=QuaternionCoeffOrder.XYZW),
                (angle_axis,),
                raise_exception=True,
                fast_mode=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        angle_axis = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        angle_axis = tensor_to_gradcheck_var(angle_axis)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.geometry.conversions.angle_axis_to_quaternion, order=QuaternionCoeffOrder.WXYZ),
            (angle_axis,),
            raise_exception=True,
            fast_mode=True,
        )


class TestQuaternionToAngleAxis:
    def test_smoke_xyzw(self, device, dtype):
        quaternion = torch.zeros(4, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert angle_axis.shape == (3,)

    def test_smoke(self, device, dtype):
        quaternion = torch.zeros(4, device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert angle_axis.shape == (3,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        quaternion = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert angle_axis.shape == (batch_size, 3)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert angle_axis.shape == (batch_size, 3)

    def test_unit_quaternion_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_x_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi, 0.0, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_y_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_z_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.5, np.sqrt(3.0) / 2.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 3.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((np.sqrt(3.0) / 2.0, 0.0, 0.0, 0.5), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 3.0), device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_x_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.sin(theta / 2.0), 0.0, 0.0, np.cos(theta / 2.0)), device=device, dtype=dtype)
        expected = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_y_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((0.0, np.sin(theta / 2), 0.0, np.cos(theta / 2)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_z_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((0.0, 0.0, np.sin(theta / 2), np.cos(theta / 2)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype) + eps
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.geometry.conversions.quaternion_to_angle_axis, order=QuaternionCoeffOrder.XYZW),
                (quaternion,),
                raise_exception=True,
                fast_mode=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.geometry.conversions.quaternion_to_angle_axis, order=QuaternionCoeffOrder.WXYZ),
            (quaternion,),
            raise_exception=True,
            fast_mode=True,
        )


class TestRotationMatrixToQuaternion:
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                matrix, order=QuaternionCoeffOrder.XYZW
            )
        assert quaternion.shape == (batch_size, 4)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion.shape == (batch_size, 4)

    def test_identity_xyzw(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                matrix, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_identity(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_rot_x_45_xyzw(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        pi_half2 = torch.cos(kornia.pi / 4.0).to(device=device, dtype=dtype)
        expected = torch.tensor((pi_half2, 0.0, 0.0, pi_half2), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                matrix, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_rot_x_45(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        pi_half2 = torch.cos(kornia.pi / 4.0).to(device=device, dtype=dtype)
        expected = torch.tensor((pi_half2, pi_half2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_back_and_forth_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                matrix, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
            matrix_hat = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(matrix, matrix_hat, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
            matrix, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        matrix_hat = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            quaternion, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(matrix, matrix_hat, atol=atol, rtol=rtol)

    def test_corner_case_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(
            (
                (-0.7799533010, -0.5432914495, 0.3106555045),
                (0.0492402576, -0.5481169224, -0.8349509239),
                (0.6238971353, -0.6359263659, 0.4542570710),
            ),
            device=device,
            dtype=dtype,
        )
        quaternion_true = torch.tensor(
            (0.280136495828629, -0.440902262926102, 0.834015488624573, 0.177614107728004), device=device, dtype=dtype
        )
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                matrix, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        torch.set_printoptions(precision=10)
        assert_close(quaternion_true, quaternion, atol=atol, rtol=rtol)

    def test_corner_case(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(
            (
                (-0.7799533010, -0.5432914495, 0.3106555045),
                (0.0492402576, -0.5481169224, -0.8349509239),
                (0.6238971353, -0.6359263659, 0.4542570710),
            ),
            device=device,
            dtype=dtype,
        )
        quaternion_true = torch.tensor(
            (0.177614107728004, 0.280136495828629, -0.440902262926102, 0.834015488624573), device=device, dtype=dtype
        )
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
            matrix, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        torch.set_printoptions(precision=10)
        assert_close(quaternion_true, quaternion, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        matrix = torch.eye(3, device=device, dtype=dtype)
        matrix = tensor_to_gradcheck_var(matrix)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(
                    kornia.geometry.conversions.rotation_matrix_to_quaternion, eps=eps, order=QuaternionCoeffOrder.XYZW
                ),
                (matrix,),
                raise_exception=True,
                fast_mode=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        matrix = torch.eye(3, device=device, dtype=dtype)
        matrix = tensor_to_gradcheck_var(matrix)
        # evaluate function gradient
        assert gradcheck(
            partial(
                kornia.geometry.conversions.rotation_matrix_to_quaternion, eps=eps, order=QuaternionCoeffOrder.WXYZ
            ),
            (matrix,),
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.parametrize('order', ['wxyz', 'xyzw'])
    def test_dynamo(self, device, dtype, torch_optimizer, order):
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion, order=QuaternionCoeffOrder(order))
        expected = op(quaternion, order=QuaternionCoeffOrder(order))

        assert_close(actual, expected)


class TestQuaternionToRotationMatrix:
    @pytest.mark.parametrize("batch_dims", ((), (1,), (3,), (8,), (1, 1), (5, 6)))
    def test_smoke_batch_xyzw(self, batch_dims, device, dtype):
        quaternion = torch.zeros(*batch_dims, 4, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert matrix.shape == (*batch_dims, 3, 3)

    @pytest.mark.parametrize("batch_dims", ((), (1,), (3,), (8,), (1, 1), (5, 6)))
    def test_smoke_batch(self, batch_dims, device, dtype):
        quaternion = torch.zeros(*batch_dims, 4, device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert matrix.shape == (*batch_dims, 3, 3)

    def test_unit_quaternion_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_x_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_y_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)

        with pytest.warns(UserWarning):
            matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_z_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.geometry.conversions.quaternion_to_rotation_matrix, order=QuaternionCoeffOrder.XYZW),
                (quaternion,),
                raise_exception=True,
                fast_mode=True,
            )

    def test_gradcheck(self, device, dtype):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.geometry.conversions.quaternion_to_rotation_matrix, order=QuaternionCoeffOrder.WXYZ),
            (quaternion,),
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.parametrize('order', ['wxyz', 'xyzw'])
    def test_dynamo(self, device, dtype, torch_optimizer, order):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_to_rotation_matrix
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion, order=QuaternionCoeffOrder(order))
        expected = op(quaternion, order=QuaternionCoeffOrder(order))

        assert_close(actual, expected)


class TestQuaternionLogToExp:
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        quaternion_log = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
                quaternion_log, order=QuaternionCoeffOrder.XYZW
            )
        assert quaternion_exp.shape == (batch_size, 4)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion_log = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
            quaternion_log, order=QuaternionCoeffOrder.WXYZ
        )
        assert quaternion_exp.shape == (batch_size, 4)

    def test_unit_quaternion_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
                quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
            quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.sin(one), 0.0, 0.0, torch.cos(one)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
                quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), torch.sin(one), 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
            quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, torch.sin(one), 0.0, torch.cos(one)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
                quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, torch.sin(one), 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
            quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, torch.sin(one), torch.cos(one)), device=device, dtype=dtype)

        with pytest.warns(UserWarning):
            quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
                quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, 0.0, torch.sin(one)), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
            quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_back_and_forth_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)

        with pytest.warns(UserWarning):
            quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
                quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        with pytest.warns(UserWarning):
            quaternion_log_hat = kornia.geometry.conversions.quaternion_exp_to_log(
                quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_log, quaternion_log_hat, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)

        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(
            quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        quaternion_log_hat = kornia.geometry.conversions.quaternion_exp_to_log(
            quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_log, quaternion_log_hat, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.geometry.conversions.quaternion_log_to_exp, eps=eps, order=QuaternionCoeffOrder.XYZW),
                (quaternion,),
                raise_exception=True,
                fast_mode=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.geometry.conversions.quaternion_log_to_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ),
            (quaternion,),
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.parametrize('order', ['wxyz', 'xyzw'])
    def test_dynamo(self, device, dtype, torch_optimizer, order):
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion, order=QuaternionCoeffOrder(order))
        expected = op(quaternion, order=QuaternionCoeffOrder(order))

        assert_close(actual, expected)


class TestQuaternionExpToLog:
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
                quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert quaternion_log.shape == (batch_size, 3)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
            quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert quaternion_log.shape == (batch_size, 3)

    def test_unit_quaternion_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
                quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
            quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
                quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
            quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
                quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
            quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
                quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
            quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_back_and_forth_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)

        with pytest.warns(UserWarning):
            quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
                quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        with pytest.warns(UserWarning):
            quaternion_exp_hat = kornia.geometry.conversions.quaternion_log_to_exp(
                quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_exp, quaternion_exp_hat, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(
            quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        quaternion_exp_hat = kornia.geometry.conversions.quaternion_log_to_exp(
            quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_exp, quaternion_exp_hat, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.geometry.conversions.quaternion_exp_to_log, eps=eps, order=QuaternionCoeffOrder.XYZW),
                (quaternion,),
                raise_exception=True,
                fast_mode=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.geometry.conversions.quaternion_exp_to_log, eps=eps, order=QuaternionCoeffOrder.WXYZ),
            (quaternion,),
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.parametrize('order', ['wxyz', 'xyzw'])
    def test_dynamo(self, device, dtype, torch_optimizer, order):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_exp_to_log
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion, order=QuaternionCoeffOrder(order))
        expected = op(quaternion, order=QuaternionCoeffOrder(order))

        assert_close(actual, expected)


class TestAngleAxisToRotationMatrix:
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_angle_axis_gradcheck(self, batch_size, device, dtype, atol, rtol):
        # generate input data
        angle_axis = torch.rand(batch_size, 3, device=device, dtype=dtype)
        eye_batch = create_eye_batch(batch_size, 3, device=device, dtype=dtype)

        # apply transform
        rotation_matrix = kornia.geometry.conversions.angle_axis_to_rotation_matrix(angle_axis)

        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        assert_close(rotation_matrix_eye, eye_batch, atol=atol, rtol=rtol)

        # evaluate function gradient
        angle_axis = tensor_to_gradcheck_var(angle_axis)  # to var
        assert gradcheck(
            kornia.geometry.conversions.angle_axis_to_rotation_matrix,
            (angle_axis,),
            raise_exception=True,
            fast_mode=True,
        )

    def test_angle_axis_to_rotation_matrix(self, device, dtype, atol, rtol):
        rmat_1 = torch.tensor(
            (
                (-0.30382753, -0.95095137, -0.05814062),
                (-0.71581715, 0.26812278, -0.64476041),
                (0.62872461, -0.15427791, -0.76217038),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_1 = torch.tensor((1.50485376, -2.10737739, 0.7214174), device=device, dtype=dtype)

        rmat_2 = torch.tensor(
            (
                (0.6027768, -0.79275544, -0.09054801),
                (-0.67915707, -0.56931658, 0.46327563),
                (-0.41881476, -0.21775548, -0.88157628),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_2 = torch.tensor((-2.44916812, 1.18053411, 0.4085298), device=device, dtype=dtype)
        rmat = torch.stack((rmat_2, rmat_1), dim=0)
        rvec = torch.stack((rvec_2, rvec_1), dim=0)

        assert_close(kornia.geometry.conversions.angle_axis_to_rotation_matrix(rvec), rmat, atol=atol, rtol=rtol)


class TestRotationMatrixToAngleAxis:
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_quaternion_gradcheck(self, batch_size, device, dtype, atol, rtol):
        # generate input data
        quaternion = torch.rand(batch_size, 4, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.normalize_quaternion(quaternion + 1e-6)
        rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            quaternion=quaternion, order=QuaternionCoeffOrder.WXYZ
        )

        eye_batch = create_eye_batch(batch_size, 3, device=device, dtype=dtype)
        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        # This didn't pass with atol=0.001, rtol=0.001 for float16 Cuda 11.2 GeForce 1080 Ti
        assert_close(rotation_matrix_eye, eye_batch, atol=atol * 10.0, rtol=rtol * 10.0)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device, dtype):
        quaternion = torch.rand(batch_size, 4, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.normalize_quaternion(quaternion + 1e-6)
        rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            quaternion=quaternion, order=QuaternionCoeffOrder.WXYZ
        )
        # evaluate function gradient
        rotation_matrix = tensor_to_gradcheck_var(rotation_matrix)  # to var
        assert gradcheck(
            kornia.geometry.conversions.rotation_matrix_to_angle_axis,
            (rotation_matrix,),
            raise_exception=True,
            fast_mode=True,
        )

    def test_rotation_matrix_to_angle_axis(self, device, dtype, atol, rtol):
        rmat_1 = torch.tensor(
            (
                (-0.30382753, -0.95095137, -0.05814062),
                (-0.71581715, 0.26812278, -0.64476041),
                (0.62872461, -0.15427791, -0.76217038),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_1 = torch.tensor((1.50485376, -2.10737739, 0.7214174), device=device, dtype=dtype)

        rmat_2 = torch.tensor(
            (
                (0.6027768, -0.79275544, -0.09054801),
                (-0.67915707, -0.56931658, 0.46327563),
                (-0.41881476, -0.21775548, -0.88157628),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_2 = torch.tensor((-2.44916812, 1.18053411, 0.4085298), device=device, dtype=dtype)
        rmat = torch.stack((rmat_2, rmat_1), dim=0)
        rvec = torch.stack((rvec_2, rvec_1), dim=0)

        assert_close(kornia.geometry.conversions.rotation_matrix_to_angle_axis(rmat), rvec, atol=atol, rtol=rtol)


def test_pi():
    assert_close(kornia.constants.pi.item(), 3.141592)


@pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
def test_rad2deg(batch_shape, device, dtype):
    # generate input data
    x_rad = kornia.constants.pi * torch.rand(batch_shape, device=device, dtype=dtype)

    # convert radians/degrees
    x_deg = kornia.geometry.conversions.rad2deg(x_rad)
    x_deg_to_rad = kornia.geometry.conversions.deg2rad(x_deg)

    # compute error
    assert_close(x_rad, x_deg_to_rad)


@pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
def test_rad2deg_gradcheck(batch_shape, device, dtype):
    x_rad = torch.rand(batch_shape, device=device, dtype=dtype)
    # evaluate function gradient
    assert gradcheck(
        kornia.geometry.conversions.rad2deg, (tensor_to_gradcheck_var(x_rad),), raise_exception=True, fast_mode=True
    )


@pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
def test_deg2rad(batch_shape, device, dtype, atol, rtol):
    # generate input data
    x_deg = 180.0 * torch.rand(batch_shape, device=device, dtype=dtype)

    # convert radians/degrees
    x_rad = kornia.geometry.conversions.deg2rad(x_deg)
    x_rad_to_deg = kornia.geometry.conversions.rad2deg(x_rad)

    assert_close(x_deg, x_rad_to_deg, atol=atol, rtol=rtol)


@pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
def test_deg2rad_gradcheck(batch_shape, device, dtype):
    x_deg = 180.0 * torch.rand(batch_shape, device=device, dtype=dtype)
    assert gradcheck(
        kornia.geometry.conversions.deg2rad, (tensor_to_gradcheck_var(x_deg),), raise_exception=True, fast_mode=True
    )


class TestPolCartConversions:
    def test_smoke(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        assert kornia.geometry.conversions.pol2cart(x, x) is not None
        assert kornia.geometry.conversions.cart2pol(x, x) is not None

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_pol2cart(self, batch_shape, device, dtype):
        # generate input data
        rho = torch.rand(batch_shape, dtype=dtype)
        phi = kornia.constants.pi * torch.rand(batch_shape, dtype=dtype)
        rho = rho.to(device)
        phi = phi.to(device)

        # convert pol/cart
        x_pol2cart, y_pol2cart = kornia.geometry.conversions.pol2cart(rho, phi)
        rho_pol2cart, phi_pol2cart = kornia.geometry.conversions.cart2pol(x_pol2cart, y_pol2cart, 0)

        assert_close(rho, rho_pol2cart)
        assert_close(phi, phi_pol2cart)

    @pytest.mark.parametrize("batch_shape", [(2, 3)])
    def test_gradcheck(self, batch_shape, device, dtype):
        rho = torch.rand(batch_shape, dtype=dtype, device=device)
        phi = kornia.constants.pi * torch.rand(batch_shape, dtype=dtype, device=device)
        assert gradcheck(
            kornia.geometry.conversions.pol2cart,
            (tensor_to_gradcheck_var(rho), tensor_to_gradcheck_var(phi)),
            raise_exception=True,
            fast_mode=True,
        )
        assert gradcheck(
            kornia.geometry.conversions.cart2pol,
            (tensor_to_gradcheck_var(rho), tensor_to_gradcheck_var(phi)),
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_cart2pol(self, batch_shape, device, dtype):
        # generate input data
        x = torch.rand(batch_shape, dtype=dtype)
        y = torch.rand(batch_shape, dtype=dtype)
        x = x.to(device)
        y = y.to(device)

        # convert cart/pol
        rho_cart2pol, phi_cart2pol = kornia.geometry.conversions.cart2pol(x, y, 0)
        x_cart2pol, y_cart2pol = kornia.geometry.conversions.pol2cart(rho_cart2pol, phi_cart2pol)

        assert_close(x, x_cart2pol)
        assert_close(y, y_cart2pol)


class TestConvertPointsToHomogeneous:
    def test_convert_points(self, device, dtype):
        # generate input data
        points_h = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [-1.0, -2.0, -1.0], [0.0, 1.0, -2.0]],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [1.0, 2.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 1.0],
                [2.0, 1.0, 0.0, 1.0],
                [-1.0, -2.0, -1.0, 1.0],
                [0.0, 1.0, -2.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_to_homogeneous(points_h)
        assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_convert_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[2.0, 1.0, 0.0, 1.0]], [[0.0, 1.0, 2.0, 1.0]], [[0.0, 1.0, -2.0, 1.0]]], device=device, dtype=dtype
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_to_homogeneous(points_h)
        assert_close(points, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_gradcheck(self, batch_shape, device, dtype):
        points_h = torch.rand(batch_shape, device=device, dtype=dtype)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(
            kornia.geometry.conversions.convert_points_to_homogeneous, (points_h,), raise_exception=True, fast_mode=True
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_points_to_homogeneous
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        assert_close(actual, expected)


class TestConvertAtoH:
    def test_convert_points(self, device, dtype):
        # generate input data
        A = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype).view(1, 2, 3)

        expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype).view(
            1, 3, 3
        )

        # to euclidean
        H = kornia.geometry.conversions.convert_affinematrix_to_homography(A)
        assert_close(H, expected)

    @pytest.mark.parametrize("batch_shape", [(10, 2, 3), (16, 2, 3)])
    def test_gradcheck(self, batch_shape, device, dtype):
        points_h = torch.rand(batch_shape, device=device, dtype=dtype)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(
            kornia.geometry.conversions.convert_affinematrix_to_homography,
            (points_h,),
            raise_exception=True,
            fast_mode=True,
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_affinematrix_to_homography
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        assert_close(actual, expected)


class TestConvertPointsFromHomogeneous:
    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_cardinality(self, device, dtype, batch_shape):
        points_h = torch.rand(batch_shape, device=device, dtype=dtype)
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        assert points.shape == points.shape[:-1] + (2,)

    def test_points(self, device, dtype):
        # generate input data
        points_h = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [-1.0, -2.0, -1.0], [0.0, 1.0, -2.0]],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [[1.0, 2.0], [0.0, 0.5], [2.0, 1.0], [1.0, 2.0], [0.0, -0.5]], device=device, dtype=dtype
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor([[[2.0, 1.0]], [[0.0, 0.5]], [[0.0, -0.5]]], device=device, dtype=dtype)

        # to euclidean
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        points_h = torch.ones(1, 10, 3, device=device, dtype=dtype)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(
            kornia.geometry.conversions.convert_points_from_homogeneous,
            (points_h,),
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.skip("RuntimeError: Jacobian mismatch for output 0 with respect to input 0,")
    def test_gradcheck_zvec_zeros(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.1], [2.0, 1.0, 0.1]], device=device, dtype=dtype)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(
            kornia.geometry.conversions.convert_points_from_homogeneous,
            (points_h,),
            raise_exception=True,
            fast_mode=True,
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_points_from_homogeneous
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        assert_close(actual, expected)


class TestNormalizePixelCoordinates:
    def test_tensor_bhw2(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(dtype=dtype)

        expected = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.geometry.conversions.normalize_pixel_coordinates(grid, height, width, eps=eps)

        assert_close(grid_norm, expected, atol=atol, rtol=rtol)

    def test_list(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(dtype=dtype)
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.geometry.conversions.normalize_pixel_coordinates(grid, height, width, eps=eps)

        assert_close(grid_norm, expected, atol=atol, rtol=rtol)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if device == torch.device('cpu'):
            pytest.skip('NormalizePixelCoordinates not working on CPU with dynamo!')

        op = kornia.geometry.conversions.normalize_pixel_coordinates
        op_optimized = torch_optimizer(op)

        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(dtype=dtype)

        actual = op_optimized(grid, height, width)
        expected = op(grid, height, width)

        assert_close(actual, expected)


class TestDenormalizePixelCoordinates:
    def test_tensor_bhw2(self, device, dtype):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(dtype=dtype)

        expected = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.geometry.conversions.denormalize_pixel_coordinates(grid, height, width)

        assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_list(self, device, dtype):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(dtype=dtype)
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.geometry.conversions.denormalize_pixel_coordinates(grid, height, width)

        assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if device == torch.device('cpu'):
            pytest.xfail('DenormalizePixelCoordinates not working on CPU with dynamo!')

        op = kornia.geometry.conversions.denormalize_pixel_coordinates
        op_optimized = torch_optimizer(op)

        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(dtype=dtype)

        actual = op_optimized(grid, height, width)
        expected = op(grid, height, width)

        assert_close(actual, expected)


class TestProjectPoints:
    def test_smoke(self, device, dtype):
        point_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        point_3d = torch.zeros(2, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 2)

    def test_smoke_batch_multi(self, device, dtype):
        point_3d = torch.zeros(2, 4, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, 4, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 4, 2)

    def test_project_and_unproject(self, device, dtype):
        point_3d = torch.tensor([[10.0, 2.0, 30.0]], device=device, dtype=dtype)
        depth = point_3d[..., -1:]
        camera_matrix = torch.tensor(
            [[[2746.0, 0.0, 991.0], [0.0, 2748.0, 619.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        point_3d_hat = kornia.geometry.camera.unproject_points(point_2d, depth, camera_matrix)
        assert_close(point_3d, point_3d_hat, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        # TODO: point [0, 0, 0] crashes
        points_3d = torch.ones(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)

        # evaluate function gradient
        points_3d = tensor_to_gradcheck_var(points_3d)
        camera_matrix = tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(
            kornia.geometry.camera.project_points, (points_3d, camera_matrix), raise_exception=True, fast_mode=True
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.camera.project_points
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_3d, camera_matrix)
        expected = op(points_3d, camera_matrix)

        assert_close(actual, expected)


class TestDenormalizePointsWithIntrinsics:
    def test_smoke(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        points_norm = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        points_2d = torch.zeros(2, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 2)

    def test_toy(self, device, dtype):
        point_2d = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.denormalize_points_with_intrinsics
        expected = torch.tensor([[192.0, 192.0]], device=device, dtype=dtype)
        assert_close(op(point_2d, camera_matrix), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)

        # evaluate function gradient
        points_2d = tensor_to_gradcheck_var(points_2d)
        camera_matrix = tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(
            kornia.geometry.conversions.denormalize_points_with_intrinsics,
            (points_2d, camera_matrix),
            raise_exception=True,
            fast_mode=True,
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.conversions.denormalize_points_with_intrinsics
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_2d, camera_matrix)
        expected = op(points_2d, camera_matrix)

        assert_close(actual, expected)


class TestNormalizePointsWithIntrinsics:
    def test_smoke(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        points_norm = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        points_2d = torch.zeros(2, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 2)

    def test_norm_unnorm(self, device, dtype):
        point_2d = torch.tensor([[128.0, 128.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        back = kornia.geometry.conversions.denormalize_points_with_intrinsics
        point_2d_norm = op(point_2d, camera_matrix)
        point_2d_hat = back(point_2d_norm, camera_matrix)
        assert_close(point_2d, point_2d_hat, atol=1e-4, rtol=1e-4)

    def test_toy(self, device, dtype):
        point_2d = torch.tensor([[192.0, 192.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        out = op(point_2d, camera_matrix)
        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)

        # evaluate function gradient
        points_2d = tensor_to_gradcheck_var(points_2d)
        camera_matrix = tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(
            kornia.geometry.conversions.normalize_points_with_intrinsics,
            (points_2d, camera_matrix),
            raise_exception=True,
            fast_mode=True,
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        op_optimized = torch_optimizer(op)

        op_optimized(points_2d, camera_matrix)
        op(points_2d, camera_matrix)


class TestRt2Extrinsics:
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        R = torch.rand(batch_size, 3, 3, dtype=dtype, device=device)
        t = torch.rand(batch_size, 3, 1, dtype=dtype, device=device)

        Rt = Rt_to_matrix4x4(R, t)
        assert Rt.shape == (batch_size, 4, 4)

        R2, t2 = matrix4x4_to_Rt(Rt)
        assert R2.shape == (batch_size, 3, 3)
        assert t2.shape == (batch_size, 3, 1)

        assert_close(R, R2, rtol=1e-4, atol=1e-5)
        assert_close(t, t2, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [5])
    def test_gradcheck(self, batch_size, device, dtype):
        R = torch.rand(batch_size, 3, 3, dtype=dtype, device=device)
        t = torch.rand(batch_size, 3, 1, dtype=dtype, device=device)
        assert gradcheck(
            kornia.geometry.conversions.Rt_to_matrix4x4,
            (tensor_to_gradcheck_var(R), tensor_to_gradcheck_var(t)),
            raise_exception=True,
            fast_mode=True,
        )


class TestCamtoworldGraphicsToVision:
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        t_vis = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R_vis = kornia.geometry.angle_axis_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        K_vis = Rt_to_matrix4x4(R_vis, t_vis)
        K_graf = camtoworld_vision_to_graphics_4x4(K_vis)

        expected = torch.tensor(
            [[0, 0, -1, 2], [0, -1, 0, 3], [-1, 0, 0, 4], [0, 0, 0, 1]], device=device, dtype=dtype
        )[None].repeat(batch_size, 1, 1)

        assert_close(K_graf, expected, rtol=1e-4, atol=1e-5)
        R_graf, t_graf = camtoworld_vision_to_graphics_Rt(R_vis, t_vis)
        expected_R = torch.tensor([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], device=device, dtype=dtype)[None].repeat(
            batch_size, 1, 1
        )
        expected_t = torch.tensor([2, 3, 4], device=device, dtype=dtype).reshape(1, 3, 1).repeat(batch_size, 1, 1)

        assert_close(t_graf, expected_t, rtol=1e-4, atol=1e-5)
        assert_close(R_graf, expected_R, rtol=1e-4, atol=1e-5)

        Kvis_back = camtoworld_graphics_to_vision_4x4(K_graf)
        assert_close(Kvis_back, K_vis, rtol=1e-4, atol=1e-5)

        R_vis_back, t_vis_back = camtoworld_graphics_to_vision_Rt(R_graf, t_graf)
        assert_close(R_vis_back, R_vis, rtol=1e-4, atol=1e-5)
        assert_close(t_vis_back, t_vis, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device, dtype):
        t_vis = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R_vis = kornia.geometry.angle_axis_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        K_vis = Rt_to_matrix4x4(R_vis, t_vis)
        assert gradcheck(
            camtoworld_graphics_to_vision_4x4, (tensor_to_gradcheck_var(K_vis)), raise_exception=True, fast_mode=True
        )
        assert gradcheck(
            camtoworld_vision_to_graphics_4x4, (tensor_to_gradcheck_var(K_vis)), raise_exception=True, fast_mode=True
        )


class TestCamtoworldRtToPoseRt:
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        t = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R = kornia.geometry.angle_axis_to_rotation_matrix(angles).repeat(batch_size, 1, 1)

        Rp, tp = camtoworld_to_worldtocam_Rt(R, t)

        expected_Rp = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=device, dtype=dtype)[None].repeat(
            batch_size, 1, 1
        )
        expected_tp = torch.tensor([4, -3, -2], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        assert_close(Rp, expected_Rp, rtol=1e-4, atol=1e-5)
        assert_close(tp, expected_tp, rtol=1e-4, atol=1e-5)

        Rback, tback = worldtocam_to_camtoworld_Rt(Rp, tp)
        assert_close(Rback, R, rtol=1e-4, atol=1e-5)
        assert_close(tback, t, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device, dtype):
        t = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R = kornia.geometry.angle_axis_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        assert gradcheck(
            camtoworld_to_worldtocam_Rt,
            (tensor_to_gradcheck_var(R), tensor_to_gradcheck_var(t)),
            raise_exception=True,
            fast_mode=True,
        )
        assert gradcheck(
            worldtocam_to_camtoworld_Rt,
            (tensor_to_gradcheck_var(R), tensor_to_gradcheck_var(t)),
            raise_exception=True,
            fast_mode=True,
        )


class TestCARKitToColmap:
    def test_everything(self, device, dtype):
        # generate input data
        t = torch.tensor([1, 0, 0], device=device, dtype=dtype).view(1, 3, 1)
        ang_deg = torch.tensor([45, 60.0, 0.0], device=device, dtype=dtype)[None]
        ang_rad = kornia.geometry.conversions.deg2rad(ang_deg)
        qvec = kornia.geometry.angle_axis_to_quaternion(ang_rad, order=QuaternionCoeffOrder.WXYZ)

        q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(qvec, t)

        angles_colmap = kornia.geometry.conversions.quaternion_to_angle_axis(q_colmap, order=QuaternionCoeffOrder.WXYZ)
        angles_colmap = kornia.geometry.conversions.rad2deg(angles_colmap)
        expected_angles = torch.tensor([[116.8870620728, 0.0, -71.7524719238]], device=device, dtype=dtype)
        expected_t = torch.tensor([[[-0.5256], [0.3558], [0.7727]]], device=device, dtype=dtype)

        assert_close(angles_colmap, expected_angles, rtol=1e-4, atol=1e-5)
        assert_close(t_colmap, expected_t, rtol=1e-4, atol=1e-5)


class TestEulerFromQuaternion(BaseTester):
    def test_smoke(self, device, dtype):
        q = Quaternion.random(batch_size=1)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        assert roll.shape == pitch.shape
        assert pitch.shape == yaw.shape

    @pytest.mark.parametrize("batch_size", ((1, 3, 4)))
    def test_cardinality(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size=batch_size)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        assert roll.shape[0] == batch_size
        assert pitch.shape[0] == batch_size
        assert yaw.shape[0] == batch_size

    def test_exception(self, device, dtype):
        q = Quaternion.random(batch_size=2)
        q = q.to(device, dtype)
        with pytest.raises(Exception):
            euler_from_quaternion(q.w, torch.rand(1), q.y, q.z)

    def test_gradcheck(self, device):
        q = Quaternion.random(batch_size=1).to(device, torch.float64)
        assert gradcheck(euler_from_quaternion, (q.w, q.x, q.y, q.z), raise_exception=True, fast_mode=True)

    def test_module(self, device, dtype):
        pass

    def test_dynamo(self, device, dtype, torch_optimizer):
        q = Quaternion.random(batch_size=1)
        q = q.to(device, dtype)
        op = euler_from_quaternion
        op_optimized = torch_optimizer(op)
        assert_close(op(q.w, q.x, q.y, q.z), op_optimized(q.w, q.x, q.y, q.z))

    def test_forth_and_back(self, device, dtype):
        q = Quaternion.random(batch_size=2)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        # TODO: check hwo to prevent getting inverted angles sometimes
        assert_close(q.w.abs(), qw.abs())
        assert_close(q.x.abs(), qx.abs())
        assert_close(q.y.abs(), qy.abs())
        assert_close(q.z.abs(), qz.abs())


class TestQuaternionFromEuler(BaseTester):
    def test_smoke(self, device, dtype):
        roll, pitch, yaw = torch.rand(3, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        assert qw.shape == qx.shape
        assert qx.shape == qy.shape
        assert qy.shape == qz.shape

    @pytest.mark.parametrize("batch_size", ((1, 3, 4)))
    def test_cardinality(self, device, dtype, batch_size):
        roll, pitch, yaw = torch.rand(3, batch_size, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        assert qw.shape[0] == batch_size
        assert qx.shape[0] == batch_size
        assert qy.shape[0] == batch_size
        assert qz.shape[0] == batch_size

    def test_exception(self, device, dtype):
        _, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)
        with pytest.raises(Exception):
            quaternion_from_euler(torch.rand(1), pitch, yaw)

    def test_gradcheck(self, device):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(quaternion_from_euler, (roll, pitch, yaw), raise_exception=True, fast_mode=True)

    def test_module(self, device, dtype):
        pass

    def test_dynamo(self, device, dtype, torch_optimizer):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)

        op = quaternion_from_euler
        op_optimized = torch_optimizer(op)

        actual = op_optimized(roll, pitch, yaw)
        expected = op(roll, pitch, yaw)

        assert_close(actual[0], expected[0])
        assert_close(actual[1], expected[1])
        assert_close(actual[2], expected[2])

    def test_forth_and_back(self, device, dtype):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        roll_new, pitch_new, yaw_new = euler_from_quaternion(qw, qx, qy, qz)
        assert_close(roll, roll_new)
        assert_close(pitch, pitch_new)
        assert_close(yaw, yaw_new)

    def test_values(self, device, dtype):
        # num_samples = 5
        # data = 2 * torch.rand(3, num_samples, device=device, dtype=dtype) - 1
        # roll, pitch, yaw = torch.pi * data
        roll = torch.tensor(
            [2.6518599987, 0.0612506270, 1.2417907715, 2.8829660416, -1.9961174726], device=device, dtype=dtype
        )

        pitch = torch.tensor(
            [2.3267219067, -2.7309591770, -1.4011553526, -2.1962766647, 2.1454355717], device=device, dtype=dtype
        )

        yaw = torch.tensor(
            [-0.8856627345, 0.2605336905, 0.4579202533, -1.3095731735, 0.6096843481], device=device, dtype=dtype
        )

        euler_expected = torch.tensor(
            [
                [-0.4897327125, 0.8148705959, 2.2559301853],
                [-3.0803420544, -0.4106334746, -2.8810589314],
                [1.2417914867, -1.4011553526, 0.4579201937],
                [-0.2586266696, -0.9453159571, 1.8320195675],
                [1.1454752684, 0.9961569905, -2.5319085121],
            ],
            device=device,
            dtype=dtype,
        )

        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        euler = euler_from_quaternion(qw, qx, qy, qz)
        euler = torch.stack(euler, -1)

        self.assert_close(euler, euler_expected, 1e-4, 1e-4)

        # this test is passing: pip install transforms3d
        # import transforms3d as tf3
        # out = [tf3.euler.euler2quat(roll[i], pitch[i], yaw[i]) for i in range(num_samples)]
        # out = torch.tensor(out, device=device, dtype=dtype)
        # self.assert_close(torch.stack((qw, qx, qy, qz), -1), out)

        # out = [tf3.euler.quat2euler((qw[i], qx[i], qy[i], qz[i])) for i in range(num_samples)]
        # out = torch.tensor(out, device=device, dtype=dtype)

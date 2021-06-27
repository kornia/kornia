from functools import partial

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.geometry.conversions import QuaternionCoeffOrder
from kornia.testing import assert_close, create_eye_batch, tensor_to_gradcheck_var


@pytest.fixture
def atol(device, dtype):
    """Lower tolerance for cuda-float16 only"""
    if 'cuda' in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


@pytest.fixture
def rtol(device, dtype):
    """Lower tolerance for cuda-float16 only"""
    if 'cuda' in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


# based on:
# https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/rotation_test.cc#L271


class TestAngleAxisToQuaternion:
    def test_smoke_xyzw(self, device, dtype):
        angle_axis = torch.zeros(3, dtype=dtype, device=device)
        with pytest.warns(UserWarning):
            quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.XYZW)
        assert quaternion.shape == (4,)

    def test_smoke(self, device, dtype):
        angle_axis = torch.zeros(3, dtype=dtype, device=device)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion.shape == (4,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        angle_axis = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.XYZW)
        assert quaternion.shape == (batch_size, 4)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        angle_axis = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion.shape == (batch_size, 4)

    def test_zero_angle_xyzw(self, device, dtype, atol, rtol):
        angle_axis = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_zero_angle(self, device, dtype, atol, rtol):
        angle_axis = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_x_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.sin(theta / 2.0), 0.0, 0.0, np.cos(theta / 2.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_y_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, np.sin(theta / 2.0), 0.0, np.cos(theta / 2.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0), device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_z_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, np.sin(theta / 2.0), np.cos(theta / 2.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        angle_axis = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)), device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_x_rotation_xyzw(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, 0.0, half_sqrt2), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, half_sqrt2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_y_rotation_xyzw(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, half_sqrt2, 0.0, half_sqrt2), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, half_sqrt2, 0.0), device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_z_rotation_xyzw(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, half_sqrt2, half_sqrt2), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        angle_axis = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, 0.0, half_sqrt2), device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        angle_axis = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        angle_axis = tensor_to_gradcheck_var(angle_axis)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.angle_axis_to_quaternion, order=QuaternionCoeffOrder.XYZW),
                (angle_axis,),
                raise_exception=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        angle_axis = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        angle_axis = tensor_to_gradcheck_var(angle_axis)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.angle_axis_to_quaternion, order=QuaternionCoeffOrder.WXYZ),
            (angle_axis,),
            raise_exception=True,
        )


class TestQuaternionToAngleAxis:
    def test_smoke_xyzw(self, device, dtype):
        quaternion = torch.zeros(4, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert angle_axis.shape == (3,)

    def test_smoke(self, device, dtype):
        quaternion = torch.zeros(4, device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert angle_axis.shape == (3,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        quaternion = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert angle_axis.shape == (batch_size, 3)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert angle_axis.shape == (batch_size, 3)

    def test_unit_quaternion_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_x_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi, 0.0, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_y_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_z_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.5, np.sqrt(3.0) / 2.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 3.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((np.sqrt(3.0) / 2.0, 0.0, 0.0, 0.5), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 3.0), device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_x_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.sin(theta / 2.0), 0.0, 0.0, np.cos(theta / 2.0)), device=device, dtype=dtype)
        expected = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_y_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((0.0, np.sin(theta / 2), 0.0, np.cos(theta / 2)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_z_xyzw(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((0.0, 0.0, np.sin(theta / 2), np.cos(theta / 2)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(angle_axis, expected, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype) + eps
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.quaternion_to_angle_axis, order=QuaternionCoeffOrder.XYZW),
                (quaternion,),
                raise_exception=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.quaternion_to_angle_axis, order=QuaternionCoeffOrder.WXYZ),
            (quaternion,),
            raise_exception=True,
        )


class TestRotationMatrixToQuaternion:
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.XYZW)
        assert quaternion.shape == (batch_size, 4)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        quaternion = kornia.rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion.shape == (batch_size, 4)

    def test_identity_xyzw(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_identity(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_rot_x_45_xyzw(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        pi_half2 = torch.cos(kornia.pi / 4.0).to(device=device, dtype=dtype)
        expected = torch.tensor((pi_half2, 0.0, 0.0, pi_half2), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_rot_x_45(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        pi_half2 = torch.cos(kornia.pi / 4.0).to(device=device, dtype=dtype)
        expected = torch.tensor((pi_half2, pi_half2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_back_and_forth_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion = kornia.rotation_matrix_to_quaternion(matrix, eps=eps, order=QuaternionCoeffOrder.XYZW)
            matrix_hat = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(matrix, matrix_hat, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        quaternion = kornia.rotation_matrix_to_quaternion(matrix, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        matrix_hat = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
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
            quaternion = kornia.rotation_matrix_to_quaternion(matrix, eps=eps, order=QuaternionCoeffOrder.XYZW)
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
        quaternion = kornia.rotation_matrix_to_quaternion(matrix, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        torch.set_printoptions(precision=10)
        assert_close(quaternion_true, quaternion, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        matrix = torch.eye(3, device=device, dtype=dtype)
        matrix = tensor_to_gradcheck_var(matrix)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.rotation_matrix_to_quaternion, eps=eps, order=QuaternionCoeffOrder.XYZW),
                (matrix,),
                raise_exception=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        matrix = torch.eye(3, device=device, dtype=dtype)
        matrix = tensor_to_gradcheck_var(matrix)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.rotation_matrix_to_quaternion, eps=eps, order=QuaternionCoeffOrder.WXYZ),
            (matrix,),
            raise_exception=True,
        )

    @pytest.mark.skipif(torch.__version__.startswith('1.6'), reason='JIT Enum not handled.')
    def test_jit_xyzw(self, device, dtype):
        op = kornia.quaternion_log_to_exp
        op_script = torch.jit.script(op)
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            actual = op_script(quaternion)
        with pytest.warns(UserWarning):
            expected = op(quaternion)
        assert_close(actual, expected)

    @pytest.mark.skipif(torch.__version__.startswith('1.6'), reason='JIT Enum not handled.')
    def test_jit(self, device, dtype):
        eps = torch.finfo(dtype).eps
        op = kornia.quaternion_log_to_exp
        op_script = torch.jit.script(op)
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        actual = op_script(quaternion, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        expected = op(quaternion, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(actual, expected)


class TestQuaternionToRotationMatrix:
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        quaternion = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert matrix.shape == (batch_size, 3, 3)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert matrix.shape == (batch_size, 3, 3)

    def test_unit_quaternion_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_x_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_y_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)

        with pytest.warns(UserWarning):
            matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_z_rotation_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.quaternion_to_rotation_matrix, order=QuaternionCoeffOrder.XYZW),
                (quaternion,),
                raise_exception=True,
            )

    def test_gradcheck(self, device, dtype):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.quaternion_to_rotation_matrix, order=QuaternionCoeffOrder.WXYZ),
            (quaternion,),
            raise_exception=True,
        )

    @pytest.mark.skipif(torch.__version__.startswith('1.6'), reason='JIT Enum not handled.')
    def test_jit_xyzw(self, device, dtype):
        op = kornia.geometry.conversions.quaternion_to_rotation_matrix
        op_jit = torch.jit.script(op)
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            assert_close(op(quaternion), op_jit(quaternion))

    @pytest.mark.skipif(torch.__version__.startswith('1.6'), reason='JIT Enum not handled.')
    def test_jit(self, device, dtype):
        op = kornia.geometry.conversions.quaternion_to_rotation_matrix
        op_jit = torch.jit.script(op)
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        assert_close(
            op(quaternion, order=QuaternionCoeffOrder.WXYZ), op_jit(quaternion, order=QuaternionCoeffOrder.WXYZ)
        )


class TestQuaternionLogToExp:
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        quaternion_log = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, order=QuaternionCoeffOrder.XYZW)
        assert quaternion_exp.shape == (batch_size, 4)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion_log = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion_exp.shape == (batch_size, 4)

    def test_unit_quaternion_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.sin(one), 0.0, 0.0, torch.cos(one)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), torch.sin(one), 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, torch.sin(one), 0.0, torch.cos(one)), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, torch.sin(one), 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, torch.sin(one), torch.cos(one)), device=device, dtype=dtype)

        with pytest.warns(UserWarning):
            quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, 0.0, torch.sin(one)), device=device, dtype=dtype)
        quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_back_and_forth_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)

        with pytest.warns(UserWarning):
            quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW)
        with pytest.warns(UserWarning):
            quaternion_log_hat = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_log, quaternion_log_hat, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)

        quaternion_exp = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        quaternion_log_hat = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_log, quaternion_log_hat, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.quaternion_log_to_exp, eps=eps, order=QuaternionCoeffOrder.XYZW),
                (quaternion,),
                raise_exception=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.quaternion_log_to_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ),
            (quaternion,),
            raise_exception=True,
        )

    @pytest.mark.skipif(torch.__version__.startswith('1.6'), reason='JIT Enum not handled.')
    def test_jit_xyzw(self, device, dtype):
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_jit = torch.jit.script(op)
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            assert_close(
                op(quaternion, order=QuaternionCoeffOrder.XYZW), op_jit(quaternion, order=QuaternionCoeffOrder.XYZW)
            )

    @pytest.mark.skipif(torch.__version__.startswith('1.6'), reason='JIT Enum not handled.')
    def test_jit(self, device, dtype):
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_jit = torch.jit.script(op)
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        assert_close(
            op(quaternion, order=QuaternionCoeffOrder.WXYZ), op_jit(quaternion, order=QuaternionCoeffOrder.WXYZ)
        )


class TestQuaternionExpToLog:
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch_xyzw(self, batch_size, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert quaternion_log.shape == (batch_size, 3)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion_log.shape == (batch_size, 3)

    def test_unit_quaternion_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_back_and_forth_xyzw(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)

        with pytest.warns(UserWarning):
            quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.XYZW)
        with pytest.warns(UserWarning):
            quaternion_exp_hat = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.XYZW)
        assert_close(quaternion_exp, quaternion_exp_hat, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.quaternion_exp_to_log(quaternion_exp, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        quaternion_exp_hat = kornia.quaternion_log_to_exp(quaternion_log, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(quaternion_exp, quaternion_exp_hat, atol=atol, rtol=rtol)

    def test_gradcheck_xyzw(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        with pytest.warns(UserWarning):
            assert gradcheck(
                partial(kornia.quaternion_exp_to_log, eps=eps, order=QuaternionCoeffOrder.XYZW),
                (quaternion,),
                raise_exception=True,
            )

    def test_gradcheck(self, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = tensor_to_gradcheck_var(quaternion)
        # evaluate function gradient
        assert gradcheck(
            partial(kornia.quaternion_exp_to_log, eps=eps, order=QuaternionCoeffOrder.WXYZ),
            (quaternion,),
            raise_exception=True,
        )

    @pytest.mark.skipif(torch.__version__.startswith('1.6'), reason='JIT Enum not handled.')
    def test_jit_xyzw(self, device, dtype, atol, rtol):
        op = kornia.geometry.conversions.quaternion_exp_to_log
        op_jit = torch.jit.script(op)
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            assert_close(op(quaternion), op_jit(quaternion), atol=atol, rtol=rtol)

    @pytest.mark.skipif(torch.__version__.startswith('1.6'), reason='JIT Enum not handled.')
    def test_jit(self, device, dtype, atol, rtol):
        op = kornia.quaternion_exp_to_log
        op_script = torch.jit.script(op)

        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        actual = op_script(quaternion, eps=torch.finfo(dtype).eps, order=QuaternionCoeffOrder.WXYZ)
        expected = op(quaternion, eps=torch.finfo(dtype).eps, order=QuaternionCoeffOrder.WXYZ)
        assert_close(actual, expected, atol=atol, rtol=rtol)


class TestAngleAxisToRotationMatrix:
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_angle_axis_gradcheck(self, batch_size, device, dtype, atol, rtol):
        # generate input data
        angle_axis = torch.rand(batch_size, 3, device=device, dtype=dtype)
        eye_batch = create_eye_batch(batch_size, 3, device=device, dtype=dtype)

        # apply transform
        rotation_matrix = kornia.angle_axis_to_rotation_matrix(angle_axis)

        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        assert_close(rotation_matrix_eye, eye_batch, atol=atol, rtol=rtol)

        # evaluate function gradient
        angle_axis = tensor_to_gradcheck_var(angle_axis)  # to var
        assert gradcheck(kornia.angle_axis_to_rotation_matrix, (angle_axis,), raise_exception=True)

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

        assert_close(kornia.angle_axis_to_rotation_matrix(rvec), rmat, atol=atol, rtol=rtol)


class TestRotationMatrixToAngleAxis:
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_quaternion_gradcheck(self, batch_size, device, dtype, atol, rtol):
        # generate input data
        quaternion = torch.rand(batch_size, 4, device=device, dtype=dtype)
        quaternion = kornia.normalize_quaternion(quaternion)
        rotation_matrix = kornia.quaternion_to_rotation_matrix(quaternion=quaternion, order=QuaternionCoeffOrder.WXYZ)

        eye_batch = create_eye_batch(batch_size, 3, device=device, dtype=dtype)
        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        # This didn't pass with atol=0.001, rtol=0.001 for float16 Cuda 11.2 GeForce 1080 Ti
        assert_close(rotation_matrix_eye, eye_batch, atol=atol * 10.0, rtol=rtol * 10.0)

        # evaluate function gradient
        rotation_matrix = tensor_to_gradcheck_var(rotation_matrix)  # to var
        assert gradcheck(kornia.rotation_matrix_to_angle_axis, (rotation_matrix,), raise_exception=True)

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

        assert_close(kornia.rotation_matrix_to_angle_axis(rmat), rvec, atol=atol, rtol=rtol)


def test_pi():
    assert_close(kornia.pi.item(), 3.141592)


@pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
def test_rad2deg(batch_shape, device, dtype):
    # generate input data
    x_rad = kornia.pi * torch.rand(batch_shape, device=device, dtype=dtype)

    # convert radians/degrees
    x_deg = kornia.rad2deg(x_rad)
    x_deg_to_rad = kornia.deg2rad(x_deg)

    # compute error
    assert_close(x_rad, x_deg_to_rad)

    # evaluate function gradient
    assert gradcheck(kornia.rad2deg, (tensor_to_gradcheck_var(x_rad),), raise_exception=True)


@pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
def test_deg2rad(batch_shape, device, dtype, atol, rtol):
    # generate input data
    x_deg = 180.0 * torch.rand(batch_shape, device=device, dtype=dtype)

    # convert radians/degrees
    x_rad = kornia.deg2rad(x_deg)
    x_rad_to_deg = kornia.rad2deg(x_rad)

    assert_close(x_deg, x_rad_to_deg, atol=atol, rtol=rtol)

    eps = torch.finfo(dtype).eps
    assert gradcheck(kornia.deg2rad, (tensor_to_gradcheck_var(x_deg),), raise_exception=True)


class TestPolCartConversions:
    def test_smoke(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        assert kornia.pol2cart(x, x) is not None
        assert kornia.cart2pol(x, x) is not None

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_pol2cart(self, batch_shape, device, dtype):
        # generate input data
        rho = torch.rand(batch_shape, dtype=dtype)
        phi = kornia.pi * torch.rand(batch_shape, dtype=dtype)
        rho = rho.to(device)
        phi = phi.to(device)

        # convert pol/cart
        x_pol2cart, y_pol2cart = kornia.pol2cart(rho, phi)
        rho_pol2cart, phi_pol2cart = kornia.cart2pol(x_pol2cart, y_pol2cart, 0)

        assert_close(rho, rho_pol2cart)
        assert_close(phi, phi_pol2cart)

        assert gradcheck(
            kornia.pol2cart, (tensor_to_gradcheck_var(rho), tensor_to_gradcheck_var(phi)), raise_exception=True
        )

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_cart2pol(self, batch_shape, device, dtype):
        # generate input data
        x = torch.rand(batch_shape, dtype=dtype)
        y = torch.rand(batch_shape, dtype=dtype)
        x = x.to(device)
        y = y.to(device)

        # convert cart/pol
        rho_cart2pol, phi_cart2pol = kornia.cart2pol(x, y, 0)
        x_cart2pol, y_cart2pol = kornia.pol2cart(rho_cart2pol, phi_cart2pol)

        assert_close(x, x_cart2pol)
        assert_close(y, y_cart2pol)

        assert gradcheck(
            kornia.cart2pol, (tensor_to_gradcheck_var(x), tensor_to_gradcheck_var(y)), raise_exception=True
        )


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
        points = kornia.convert_points_to_homogeneous(points_h)
        assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_convert_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[2.0, 1.0, 0.0, 1.0]], [[0.0, 1.0, 2.0, 1.0]], [[0.0, 1.0, -2.0, 1.0]]], device=device, dtype=dtype
        )

        # to euclidean
        points = kornia.convert_points_to_homogeneous(points_h)
        assert_close(points, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_gradcheck(self, batch_shape, device, dtype):
        points_h = torch.rand(batch_shape, device=device, dtype=dtype)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(kornia.convert_points_to_homogeneous, (points_h,), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.geometry.conversions.convert_points_to_homogeneous
        op_jit = torch.jit.script(op)
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)
        assert_close(op(points_h), op_jit(points_h))


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
        assert gradcheck(kornia.convert_affinematrix_to_homography, (points_h,), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.geometry.conversions.convert_affinematrix_to_homography
        op_jit = torch.jit.script(op)
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)
        assert_close(op(points_h), op_jit(points_h))


class TestConvertPointsFromHomogeneous:
    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_cardinality(self, device, dtype, batch_shape):
        points_h = torch.rand(batch_shape, device=device, dtype=dtype)
        points = kornia.convert_points_from_homogeneous(points_h)
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
        points = kornia.convert_points_from_homogeneous(points_h)
        assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor([[[2.0, 1.0]], [[0.0, 0.5]], [[0.0, -0.5]]], device=device, dtype=dtype)

        # to euclidean
        points = kornia.convert_points_from_homogeneous(points_h)
        assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        points_h = torch.ones(1, 10, 3, device=device, dtype=dtype)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(kornia.convert_points_from_homogeneous, (points_h,), raise_exception=True)

    @pytest.mark.skip("RuntimeError: Jacobian mismatch for output 0 with respect to input 0,")
    def test_gradcheck_zvec_zeros(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.1], [2.0, 1.0, 0.1]], device=device, dtype=dtype)

        # evaluate function gradient
        points_h = tensor_to_gradcheck_var(points_h)  # to var
        assert gradcheck(kornia.convert_points_from_homogeneous, (points_h,), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.geometry.conversions.convert_points_from_homogeneous
        op_jit = torch.jit.script(op)
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)
        assert_close(op(points_h), op_jit(points_h))


class TestNormalizePixelCoordinates:
    def test_tensor_bhw2(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(dtype=dtype)

        expected = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.normalize_pixel_coordinates(grid, height, width, eps=eps)

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

        grid_norm = kornia.normalize_pixel_coordinates(grid, height, width, eps=eps)

        assert_close(grid_norm, expected, atol=atol, rtol=rtol)

    def test_jit(self, device, dtype):
        op = kornia.geometry.conversions.normalize_pixel_coordinates
        op_script = torch.jit.script(op)

        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(dtype=dtype)

        actual = op_script(grid, height, width)
        expected = op(grid, height, width)
        assert_close(actual, expected)


class TestDenormalizePixelCoordinates:
    def test_tensor_bhw2(self, device, dtype):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(dtype=dtype)

        expected = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.denormalize_pixel_coordinates(grid, height, width)

        assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_list(self, device, dtype):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(dtype=dtype)
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.denormalize_pixel_coordinates(grid, height, width)

        assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_jit(self, device, dtype):
        op = kornia.geometry.conversions.denormalize_pixel_coordinates
        op_script = torch.jit.script(op)

        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(dtype=dtype)

        actual = op_script(grid, height, width)
        expected = op(grid, height, width)

        assert_close(actual, expected)

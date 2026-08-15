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

import inspect
import sys
from functools import partial

import numpy as np
import pytest
import torch

import kornia
from kornia.core._compat import torch_version
from kornia.core.ops import eye_like
from kornia.geometry.conversions import (
    ARKitQTVecs_to_ColmapQTVecs,
    Rt_to_matrix4x4,
    axis_angle_to_rotation_matrix,
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

from testing.base import BaseTester, assert_close


@pytest.fixture()
def atol(device, dtype):
    """Lower tolerance for cuda-float16 only."""
    if "cuda" in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


@pytest.fixture()
def rtol(device, dtype):
    """Lower tolerance for cuda-float16 only."""
    if "cuda" in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


class TestAngleAxisToQuaternion(BaseTester):
    # based on:
    # https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/rotation_test.cc#L271

    def test_smoke(self, device, dtype):
        axis_angle = torch.zeros(3, dtype=dtype, device=device)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.shape == (4,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        axis_angle = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.shape == (batch_size, 4)

    def test_zero_angle(self, device, dtype, atol, rtol):
        axis_angle = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        axis_angle = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        axis_angle = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        axis_angle = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        axis_angle = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, half_sqrt2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        axis_angle = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, half_sqrt2, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        axis_angle = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, 0.0, half_sqrt2), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("input_dtype", (torch.int16, torch.int32, torch.int64, torch.uint8))
    def test_integer_input(self, input_dtype, device):
        # an integer input used to be written back into an integer output buffer, which truncated
        # every component to zero. see https://github.com/kornia/kornia/issues/3948
        axis_angle = torch.tensor((1, 0, 0), device=device, dtype=input_dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.is_floating_point()
        expected = torch.tensor((np.cos(0.5), np.sin(0.5), 0.0, 0.0), device=device, dtype=quaternion.dtype)
        self.assert_close(quaternion, expected, atol=1.0e-4, rtol=1.0e-4)

    @pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
    def test_float_input_keeps_dtype(self, input_dtype, device):
        axis_angle = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=input_dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.dtype == input_dtype
        expected = torch.tensor((np.cos(0.5), np.sin(0.5), 0.0, 0.0), device=device, dtype=input_dtype)
        self.assert_close(quaternion, expected, atol=1.0e-2, rtol=1.0e-2)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        axis_angle = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.axis_angle_to_quaternion), (axis_angle,))


class TestQuaternionToAngleAxis(BaseTester):
    def test_smoke(self, device, dtype):
        quaternion = torch.zeros(4, device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        assert axis_angle.shape == (3,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        assert axis_angle.shape == (batch_size, 3)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi, 0.0, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((np.sqrt(3.0) / 2.0, 0.0, 0.0, 0.5), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 3.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_to_axis_angle), (quaternion,))


class TestRotationMatrixToQuaternion(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix)
        assert quaternion.shape == (batch_size, 4)

    def test_identity(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_rot_x_45(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        pi_half2 = torch.cos(kornia.pi / 4.0).to(device=device, dtype=dtype)
        expected = torch.tensor((pi_half2, pi_half2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        matrix_hat = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, matrix_hat, atol=atol, rtol=rtol)

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
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        torch.set_printoptions(precision=10)
        self.assert_close(quaternion_true, quaternion, atol=atol, rtol=rtol)

    def test_cond1_180_rot_x(self, device, dtype, atol, rtol):
        # 180° rotation around X: trace < 0, m00 > m11 and m00 > m22 → activates cond_1 branch.
        # R_x(π) = diag(1, -1, -1); expected quaternion (w,x,y,z) = (0, 1, 0, 0).
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        self.assert_close(quaternion.abs(), expected.abs(), atol=atol, rtol=rtol)
        # Round-trip: convert back and verify the rotation matrix is recovered.
        mat_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(mat_back, matrix, atol=atol, rtol=rtol)

    def test_cond2_180_rot_y(self, device, dtype, atol, rtol):
        # 180° rotation around Y: trace < 0, m11 > m22 and m00 not dominant → activates cond_2 branch.
        # R_y(π) = diag(-1, 1, -1); expected quaternion (w,x,y,z) = (0, 0, 1, 0).
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        self.assert_close(quaternion.abs(), expected.abs(), atol=atol, rtol=rtol)
        mat_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(mat_back, matrix, atol=atol, rtol=rtol)

    def test_all_four_branches_in_batch(self, device, dtype, atol, rtol):
        # Batch of 4 rotation matrices that each activate a different internal branch.
        # Verify consistency via round-trip: R → q → R must recover the original rotation.
        eps = torch.finfo(dtype).eps
        identity = torch.eye(3, device=device, dtype=dtype)  # trace > 0 → trace_positive_cond
        rot_x_180 = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        rot_y_180 = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        rot_z_180 = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        batch = torch.stack([identity, rot_x_180, rot_y_180, rot_z_180])  # (4, 3, 3)
        quaternions = kornia.geometry.conversions.rotation_matrix_to_quaternion(batch, eps=eps)
        mats_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternions)
        self.assert_close(mats_back, batch, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        matrix = torch.eye(3, device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.rotation_matrix_to_quaternion, eps=eps), (matrix,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)


class TestQuaternionToRotationMatrix(BaseTester):
    @pytest.mark.parametrize("batch_dims", ((), (1,), (3,), (8,), (1, 1), (5, 6)))
    def test_smoke_batch(self, batch_dims, device, dtype):
        quaternion = torch.zeros(*batch_dims, 4, device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        assert matrix.shape == (*batch_dims, 3, 3)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=torch.float64)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_to_rotation_matrix), (quaternion,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_to_rotation_matrix
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)


class TestQuaternionLogToExp(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion_log = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log)
        assert quaternion_exp.shape == (batch_size, 4)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), torch.sin(one), 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, torch.sin(one), 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, 0.0, torch.sin(one)), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)

        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        quaternion_log_hat = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, quaternion_log_hat, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_log_to_exp, eps=eps), (quaternion,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)


class TestQuaternionExpToLog(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        assert quaternion_log.shape == (batch_size, 3)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        quaternion_exp_hat = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, quaternion_exp_hat, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_exp_to_log, eps=eps), (quaternion,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_exp_to_log
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)


class TestAngleAxisToRotationMatrix(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_axis_angle_gradcheck(self, batch_size, device, atol, rtol):
        dtype = torch.float64
        # generate input data
        axis_angle = torch.rand(batch_size, 3, device=device, dtype=dtype)
        eye_batch = eye_like(3, axis_angle)

        # apply transform
        rotation_matrix = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)

        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        self.assert_close(rotation_matrix_eye, eye_batch, atol=atol, rtol=rtol)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.axis_angle_to_rotation_matrix, (axis_angle,))

    def test_axis_angle_to_rotation_matrix(self, device, dtype, atol, rtol):
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

        self.assert_close(kornia.geometry.conversions.axis_angle_to_rotation_matrix(rvec), rmat, atol=atol, rtol=rtol)


class TestRotationMatrixToAngleAxis(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_quaternion_gradcheck(self, batch_size, device, dtype, atol, rtol):
        # generate input data
        quaternion = torch.rand(batch_size, 4, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.normalize_quaternion(quaternion + 1e-6)
        rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion=quaternion)

        eye_batch = eye_like(3, rotation_matrix)
        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        # This didn't pass with atol=0.001, rtol=0.001 for float16 Cuda 11.2 GeForce 1080 Ti
        self.assert_close(rotation_matrix_eye, eye_batch, atol=atol * 10.0, rtol=rtol * 10.0)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device):
        dtype = torch.float64
        quaternion = torch.rand(batch_size, 4, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.normalize_quaternion(quaternion + 1e-6)
        rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion=quaternion)
        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.rotation_matrix_to_axis_angle, (rotation_matrix,))

    def test_rotation_matrix_to_axis_angle(self, device, dtype, atol, rtol):
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

        self.assert_close(kornia.geometry.conversions.rotation_matrix_to_axis_angle(rmat), rvec, atol=atol, rtol=rtol)


class TestRadDegConversions(BaseTester):
    def test_pi(self):
        self.assert_close(kornia.constants.pi.item(), 3.141592)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_rad2deg(self, batch_shape, device, dtype):
        # generate input data
        x_rad = kornia.constants.pi * torch.rand(batch_shape, device=device, dtype=dtype)

        # convert radians/degrees
        x_deg = kornia.geometry.conversions.rad2deg(x_rad)
        x_deg_to_rad = kornia.geometry.conversions.deg2rad(x_deg)

        # compute error
        self.assert_close(x_rad, x_deg_to_rad)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_rad2deg_gradcheck(self, batch_shape, device):
        dtype = torch.float64
        x_rad = torch.rand(batch_shape, device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.rad2deg, (x_rad,))

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_deg2rad(self, batch_shape, device, dtype, atol, rtol):
        # generate input data
        x_deg = 180.0 * torch.rand(batch_shape, device=device, dtype=dtype)

        # convert radians/degrees
        x_rad = kornia.geometry.conversions.deg2rad(x_deg)
        x_rad_to_deg = kornia.geometry.conversions.rad2deg(x_rad)

        self.assert_close(x_deg, x_rad_to_deg, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_deg2rad_gradcheck(self, batch_shape, device):
        x_deg = 180.0 * torch.rand(batch_shape, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.conversions.deg2rad, (x_deg,))

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="kornia.constants.pi is float32, so f64 loses ~7 digits (angle_to_rotation_matrix "
        "inherits it via deg2rad) — kornia#3937",
        strict=True,
    )
    @pytest.mark.parametrize(
        ("op_name", "arg", "expected"),
        [
            ("rad2deg", torch.pi, 180.0),
            ("deg2rad", 180.0, torch.pi),
            ("angle_to_rotation_matrix", 90.0, [[0.0, 1.0], [-1.0, 0.0]]),
        ],
    )
    def test_convention_float64_results_are_exact_3937(self, device, op_name, arg, expected):
        # Intended behavior: each op is exact to the precision of its input dtype, like
        # torch.rad2deg / torch.deg2rad; angle_to_rotation_matrix(90) is then the exact quarter
        # turn. It is not: all three multiply by kornia.constants.pi, a *float32* tensor merely
        # cast to the input dtype, so a float64 input carries a systematic ~2.8e-8 relative
        # error (#3937). float64 is hardcoded (like test_rad2deg_gradcheck above) because at
        # float32 the biased constant *is* the correctly rounded pi; MPS is skipped visibly
        # below because it has no float64 at all, so without the skip the xfail would be
        # satisfied by a TypeError instead of the precision assert it documents (hence also
        # raises=AssertionError on the mark). Marked xfail(strict=True) so fixing #3937 makes
        # every case XPASS and forces this mark out — a one-place edit.
        # Snippet used to generate expected (stdlib + torch):
        #   math.degrees(math.pi) == 180.0 and (180.0 * math.pi) / 180.0 == math.pi exactly
        #   kornia rad2deg(tensor(pi, f64)).item()   -> 179.99999499104382
        #   kornia deg2rad(tensor(180., f64)).item() -> 3.1415927410125732 (math.pi + 8.7e-08)
        #   kornia angle_to_rotation_matrix(tensor(90., f64)).flatten().tolist() ->
        #     [-4.371139000186241e-08, 0.999999999999999, -0.999999999999999, -4.371139e-08]
        # atol/rtol 1e-12 sits between the current ~4.4e-8 cosine error and the 6.123234e-17
        # an unbiased constant would give.
        if device.type == "mps":
            pytest.skip("MPS has no float64, and this pin is float64-only by construction")

        op = getattr(kornia.geometry.conversions, op_name)

        out = op(torch.tensor(arg, device=device, dtype=torch.float64))

        self.assert_close(out, torch.tensor(expected, device=device, dtype=torch.float64), atol=1e-12, rtol=1e-12)

    def test_convention_angle_to_rotation_matrix_takes_degrees(self, device, dtype):
        # Convention pin: angle_to_rotation_matrix reads its argument in DEGREES (not radians)
        # and returns [[cos, sin], [-sin, cos]] -- the transpose of the textbook math-frame CCW
        # matrix. Pinned on a small non-symmetric angle so a sign flip on the off-diagonal is
        # caught.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
        #   [[c, s], [-s, c]] -> [[0.8660254037844387, 0.49999999999999994],
        #                         [-0.49999999999999994, 0.8660254037844387]]
        out = kornia.geometry.conversions.angle_to_rotation_matrix(torch.tensor(30.0, device=device, dtype=dtype))
        expected = torch.tensor([[0.8660254, 0.5], [-0.5, 0.8660254]], device=device, dtype=dtype)
        self.assert_close(out, expected)

        # A radian-reading implementation would turn pi/2 into the quarter turn [[0, 1], [-1, 0]];
        # this one reads pi/2 as 1.5708 *degrees* and returns a near-identity matrix instead.
        # Snippet used to generate expected:
        #   c, s = math.cos(math.radians(math.pi / 2)), math.sin(math.radians(math.pi / 2))
        #   [[c, s], [-s, c]] -> [[0.9996242168385687, 0.027412134354665284], ...]
        out_rad = kornia.geometry.conversions.angle_to_rotation_matrix(
            torch.tensor(torch.pi / 2, device=device, dtype=dtype)
        )
        expected_rad = torch.tensor([[0.99962422, 0.02741213], [-0.02741213, 0.99962422]], device=device, dtype=dtype)
        self.assert_close(out_rad, expected_rad)

    @pytest.mark.parametrize(
        ("op_name", "arg", "expected"),
        [
            ("rad2deg", [1, 2, 3], [60.0, 120.0, 180.0]),
            ("deg2rad", [180, 90], [3.0, 1.5]),
            ("angle_to_rotation_matrix", [90], [[[0.07073720, 0.99749500], [-0.99749500, 0.07073720]]]),
        ],
    )
    def test_wart_integer_input_truncates_pi_to_3_3937(self, device, op_name, arg, expected):
        # Wart pins for #3937: assert the CURRENT broken outputs the docstring warnings document.
        # kornia.constants.pi is cast to the *integer* input dtype and truncates to 3, so rad2deg
        # divides by 3, deg2rad multiplies by 3 (90 degrees -> 1.5 radians), and the downstream
        # angle_to_rotation_matrix([90]) is nowhere near the quarter turn. If a case fails, #3937
        # was (partly) fixed -- update or remove the warnings in rad2deg, deg2rad and
        # angle_to_rotation_matrix and flip/remove the strict xfail above. NOT a contract that
        # int inputs must keep these values: what they *should* do (promote to float like
        # torch.rad2deg, or raise) is a maintainer decision, and a strict xfail asserting the
        # promoted-float answer would stay silently XFAIL forever if the fix chose to raise;
        # a wart pin flips loudly under either polarity.
        # Snippet used to generate expected (torch only):
        #   kornia rad2deg(torch.tensor([1, 2, 3])) -> tensor([ 60., 120., 180.]), dtype float32
        #     (torch.rad2deg gives [ 57.2958, 114.5916, 171.8873])
        #   kornia deg2rad(torch.tensor([180, 90])) -> tensor([3.0000, 1.5000]), dtype float32
        #     (torch.deg2rad gives [3.1416, 1.5708])
        #   kornia angle_to_rotation_matrix(torch.tensor([90])).flatten().tolist() ->
        #     [0.07073719799518585, 0.9974949955940247, -0.9974949955940247, 0.07073719799518585]
        #     (math.cos(1.5), math.sin(1.5) -> (0.0707372016677029, 0.9974949866040544))
        op = getattr(kornia.geometry.conversions, op_name)

        out = op(torch.tensor(arg, device=device))

        assert out.dtype == torch.float32
        self.assert_close(out, torch.tensor(expected, device=device, dtype=torch.float32), atol=1e-4, rtol=1e-4)


class TestPolCartConversions(BaseTester):
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

        self.assert_close(rho, rho_pol2cart)
        self.assert_close(phi, phi_pol2cart)

    @pytest.mark.parametrize("batch_shape", [(2, 3)])
    def test_gradcheck(self, batch_shape, device):
        rho = torch.rand(batch_shape, dtype=torch.float64, device=device)
        phi = kornia.constants.pi * torch.rand(batch_shape, dtype=torch.float64, device=device)
        self.gradcheck(kornia.geometry.conversions.pol2cart, (rho, phi))
        self.gradcheck(kornia.geometry.conversions.cart2pol, (rho, phi))

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

        self.assert_close(x, x_cart2pol)
        self.assert_close(y, y_cart2pol)

    def test_convention_pol2cart_takes_rho_phi_returns_x_y(self, device, dtype):
        # Convention pin: pol2cart's argument order is (rho, phi) and its return order is
        # (x, y), with phi in RADIANS measured from the +x axis: x = rho*cos(phi),
        # y = rho*sin(phi). The literal is deliberately off-axis (3 != 4) so that swapping
        # either the arguments or the returns is caught.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   phi = math.atan2(4.0, 3.0)  # 0.9272952180016122 rad
        #   5.0 * math.cos(phi), 5.0 * math.sin(phi) -> (3.0000000000000004, 4.0)
        rho = torch.tensor(5.0, device=device, dtype=dtype)
        phi = torch.tensor(0.9272952180016122, device=device, dtype=dtype)

        x, y = kornia.geometry.conversions.pol2cart(rho, phi)

        self.assert_close(x, torch.tensor(3.0, device=device, dtype=dtype))
        self.assert_close(y, torch.tensor(4.0, device=device, dtype=dtype))

    def test_convention_cart2pol_takes_x_y_and_phi_is_atan2_y_x(self, device, dtype):
        # Convention pin: cart2pol's argument order is (x, y) and its return order is
        # (rho, phi), with phi = atan2(y, x) in radians -- zero on the +x axis and increasing
        # toward +y (which is clockwise *as displayed* under kornia's y-down image axes).
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.hypot(3.0, 4.0), math.atan2(4.0, 3.0) -> (5.0, 0.9272952180016122)
        #   math.atan2(1.0, 0.0) -> 1.5707963267948966  (atan2(x, y) would give 0.0 here)
        rho, phi = kornia.geometry.conversions.cart2pol(
            torch.tensor(3.0, device=device, dtype=dtype), torch.tensor(4.0, device=device, dtype=dtype)
        )
        self.assert_close(rho, torch.tensor(5.0, device=device, dtype=dtype))
        self.assert_close(phi, torch.tensor(0.9272952180016122, device=device, dtype=dtype))

        phi_y_axis = kornia.geometry.conversions.cart2pol(
            torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
        )[1]
        self.assert_close(phi_y_axis, torch.tensor(1.5707963267948966, device=device, dtype=dtype))

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="cart2pol returns sqrt(x**2 + y**2 + eps), biasing rho — kornia#3939",
        strict=True,
    )
    def test_convention_cart2pol_rho_is_the_exact_radius(self, device, dtype):
        # Intended behavior: rho is the Euclidean radius, so rho(0, 0) == 0. It currently is
        # not: eps is added *inside* the sqrt, so rho = sqrt(x**2 + y**2 + eps) and the origin
        # maps to sqrt(1e-8) = 1e-4 (see #3939; eps belongs in the gradient path, not the
        # value). Marked xfail(strict=True) so fixing #3939 makes this XPASS loudly.
        # Snippet used to generate expected (stdlib only):
        #   math.hypot(0.0, 0.0) -> 0.0 ; kornia cart2pol(0., 0.)[0].item() -> 0.0001
        if dtype == torch.float16:
            pytest.skip("float16 cannot represent the default eps=1e-8, so the bias is invisible there")

        rho = kornia.geometry.conversions.cart2pol(
            torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
        )[0]
        self.assert_close(rho, torch.tensor(0.0, device=device, dtype=dtype), atol=1e-6, rtol=0.0)

    def test_wart_rho_is_biased_by_eps_inside_the_sqrt_3939(self, device, dtype):
        # Wart pin for kornia#3939, companion to the strict xfail above: assert the CURRENT
        # biased rho. The xfail pins the intended rho(0, 0) == 0 but cannot flip under every fix
        # polarity -- the equally standard sqrt(clamp(x**2 + y**2, min=eps)) (the shape
        # normalize_pixel_coordinates already uses) also returns 1e-4 at the origin, leaving the
        # mark silently XFAIL with a stale reason string. So two cells are pinned: the origin,
        # rho = sqrt(eps) = 1e-4, which flips under a grad-only eps (rho 0) and under eps**2
        # inside the sqrt (rho 1e-8); and a sub-eps point x = 5e-5, whose x**2 = 2.5e-9 < eps
        # gives rho = sqrt(1.25e-8) ~ 1.118e-4, which additionally flips under the clamp shape
        # (rho 1e-4, 10.6 % below, outside rtol 1e-2). If either assert fails, #3939 was
        # (partly) fixed -- update or remove the warning in cart2pol and flip/remove the strict
        # xfail above. eps=1e-8 is passed explicitly so the pinned literals do not silently
        # track a later change to the default.
        # Snippet used to generate expected (torch only, executed at each pinned dtype):
        #   c2p = kornia.geometry.conversions.cart2pol
        #   c2p(torch.tensor(0., dtype=torch.float64), torch.tensor(0., dtype=torch.float64),
        #       eps=1e-8)[0] -> 0.0001                    (f32: 9.999999747378752e-05)
        #   c2p(torch.tensor(5e-5, dtype=torch.float64), torch.tensor(0., dtype=torch.float64),
        #       eps=1e-8)[0] -> 0.00011180339887498949    (f32: 0.00011180339788552374)
        # At bfloat16 the outputs land within 0.3 % of the literals (1.00136e-4, 1.12057e-4),
        # inside rtol 1e-2, so the pin holds there too.
        if dtype == torch.float16:
            pytest.skip("float16 cannot represent eps=1e-8, so rho is 0 at both pinned points and the bias invisible")

        zero = torch.tensor(0.0, device=device, dtype=dtype)

        rho_origin = kornia.geometry.conversions.cart2pol(zero, zero, eps=1e-8)[0]
        rho_sub_eps = kornia.geometry.conversions.cart2pol(
            torch.tensor(5e-5, device=device, dtype=dtype), zero, eps=1e-8
        )[0]

        self.assert_close(rho_origin, torch.tensor(1e-4, device=device, dtype=dtype), atol=0.0, rtol=1e-2)
        self.assert_close(
            rho_sub_eps, torch.tensor(1.1180339887498949e-4, device=device, dtype=dtype), atol=0.0, rtol=1e-2
        )

    def test_convention_positive_rotation_decreases_cart2pol_phi(self, device, dtype):
        # Cross-symbol convention pin: enforces the opposite-sense relation between
        # angle_to_rotation_matrix and cart2pol stated canonically in cart2pol's Convention
        # block (phi decreases by theta modulo 2*pi).
        # First case: no branch-cut crossing, so the raw difference is -theta itself.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   -math.radians(30.0) -> -0.5235987755982988
        v = torch.tensor([3.0, 4.0], device=device, dtype=dtype)
        phi0 = kornia.geometry.conversions.cart2pol(v[0], v[1])[1]

        rot = kornia.geometry.conversions.angle_to_rotation_matrix(torch.tensor(30.0, device=device, dtype=dtype))
        v_rot = rot @ v
        phi1 = kornia.geometry.conversions.cart2pol(v_rot[0], v_rot[1])[1]

        expected_delta = torch.tensor(-0.5235987755982988, device=device, dtype=dtype)
        self.assert_close(phi1 - phi0, expected_delta)

        # Second case: crossing the -x branch cut, where the returned phi is re-wrapped into
        # [-pi, pi] and only the difference modulo 2*pi is -theta (the worked -170 + 30 example
        # lives in cart2pol's Convention block).
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   5 * math.cos(math.radians(-170.0)), 5 * math.sin(math.radians(-170.0))
        #     -> (-4.92403876506104, -0.8682408883346514)
        #   math.radians(160.0) -> 2.792526803190927
        w = torch.tensor([-4.9240388, -0.8682409], device=device, dtype=dtype)
        phi0_cut = kornia.geometry.conversions.cart2pol(w[0], w[1])[1]

        w_rot = rot @ w
        phi1_cut = kornia.geometry.conversions.cart2pol(w_rot[0], w_rot[1])[1]

        self.assert_close(phi1_cut, torch.tensor(2.7925268, device=device, dtype=dtype))

        raw_delta = phi1_cut - phi0_cut
        wrapped_delta = torch.atan2(torch.sin(raw_delta), torch.cos(raw_delta))
        # The re-wrap atan2(sin, cos) adds two more transcendental roundings on top of the two
        # atan2 outputs it differences, overshooting the central per-dtype tolerances in the half
        # dtypes. Measured against the dtype-cast expected tensor the assert compares with
        # (-0.5234375 in both halves): |err| is 1.953125e-3 in float16 (wrapped -0.525390625;
        # central allowance atol 1e-3 + rtol 1e-3 * 0.52 = 1.52e-3) and 1.171875e-2 in bfloat16
        # (wrapped -0.53515625; allowance 1.19e-2 -- a 1.4 % margin that torch rounding drift
        # could erase). atol 2.4e-2 is ~2x the bfloat16 error; a sign-flipped or unwrapped delta
        # would still be off by >= 1.0.
        wrap_tol = {"atol": 2.4e-2, "rtol": 0.0} if dtype in (torch.float16, torch.bfloat16) else {}
        self.assert_close(wrapped_delta, expected_delta, **wrap_tol)


class TestConvertPointsToHomogeneous(BaseTester):
    def test_convert_points(self, device, dtype):
        # Convention pin: the homogeneous 1.0 is appended as the *last* component (the
        # non-symmetric rows catch a prepend or a component reversal).
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
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_convert_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[2.0, 1.0, 0.0, 1.0]], [[0.0, 1.0, 2.0, 1.0]], [[0.0, 1.0, -2.0, 1.0]]], device=device, dtype=dtype
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_to_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_gradcheck(self, batch_shape, device):
        points_h = torch.rand(batch_shape, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_points_to_homogeneous, (points_h,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_points_to_homogeneous
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        self.assert_close(actual, expected)


class TestConvertAtoH(BaseTester):
    def test_convert_points(self, device, dtype):
        # Convention pin and its enforcement point: the (B, 2, 3) affine block is copied
        # verbatim into the top of the (B, 3, 3) result (no transpose, no reordering) and the
        # row [0, 0, 1] is appended at the *bottom*. The literal is non-symmetric so a
        # transpose is caught.
        # Snippet used to generate expected (torch only):
        #   convert_affinematrix_to_homography(torch.tensor([[[1., 2., 3.], [4., 5., 6.]]]))
        #     -> [[[1., 2., 3.], [4., 5., 6.], [0., 0., 1.]]]
        A = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], device=device, dtype=dtype)

        expected = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        H = kornia.geometry.conversions.convert_affinematrix_to_homography(A)
        self.assert_close(H, expected)

    @pytest.mark.parametrize("batch_shape", [(10, 2, 3), (16, 2, 3)])
    def test_gradcheck(self, batch_shape, device):
        points_h = torch.rand(batch_shape, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_affinematrix_to_homography, (points_h,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_affinematrix_to_homography
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        self.assert_close(actual, expected)

    def test_convention_homography3d_appends_bottom_row_0_0_0_1(self, device, dtype):
        # Convention pin (3-D sibling, which has no test class of its own): the (B, 3, 4) affine
        # block is copied verbatim and the row [0, 0, 0, 1] is appended at the bottom.
        # Snippet used to generate expected (by hand):
        #   [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] gains [0, 0, 0, 1]
        A = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.convert_affinematrix_to_homography3d(A)

        expected = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out, expected)


class TestConvertPointsFromHomogeneous(BaseTester):
    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_cardinality(self, device, dtype, batch_shape):
        points_h = torch.rand(batch_shape, device=device, dtype=dtype)
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        assert points.shape == points.shape[:-1] + (2,)

    def test_points(self, device, dtype):
        # Convention pins: the [2., 1., 0.] row is the |w| <= eps case (default eps 1e-8) --
        # returned *unchanged*, not zeros, not inf, no exception. The negative-w rows pin that
        # the sign of w is preserved (no abs): [0., 1., -2.] -> [0., -0.5].
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
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor([[[2.0, 1.0]], [[0.0, 0.5]], [[0.0, -0.5]]], device=device, dtype=dtype)

        # to euclidean
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_h = torch.ones(1, 10, 3, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_points_from_homogeneous, (points_h,))

    def test_gradcheck_zvec_zeros(self, device):
        # generate input data
        points_h = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.1], [2.0, 1.0, 0.1]], device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_points_from_homogeneous, (points_h,), eps=1e-8)

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_points_from_homogeneous
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        self.assert_close(actual, expected)

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="convert_points_from_homogeneous divides by w + eps — kornia#3938",
        strict=True,
    )
    def test_convention_divides_by_exactly_w(self, device, dtype):
        # Intended behavior: for |w| > eps the point is divided by exactly w. It currently is
        # divided by w + eps with no regard for sign, so the signed relative error is exactly
        # -eps / (w + eps): -1/3 at w = +2e-8 (33 % low) and +1 at w = -2e-8 (100 % high) (#3938).
        # Marked xfail(strict=True) so fixing #3938 makes this XPASS and forces the mark out.
        # Snippet used to generate expected (by hand):
        #   2 / 2e-8, 4 / 2e-8 -> [1e8, 2e8]  (kornia returns [6.6666667e7, 1.3333333e8])
        #   measured signed relative error in float64: -0.3333333333333334, and
        #   -eps / (w + eps) = -1e-8 / 3e-8 = -0.3333333333333333
        if dtype == torch.float16:
            pytest.skip("float16 underflows w=2e-8 to 0, which is the |w| <= eps passthrough branch, not the eps bias")

        points = torch.tensor([[2.0, 4.0, 2e-8]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.convert_points_from_homogeneous(points)

        expected = torch.tensor([[1e8, 2e8]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_wart_division_is_by_w_plus_eps_for_both_signs_3938(self, device, dtype):
        # Wart pin for kornia#3938, companion to the strict xfail above: assert the CURRENT
        # biased outputs for both signs of w. The xfail pins the intended behavior but cannot
        # flip under every fix polarity -- a sign-aware eps (w + sign(w) * eps) leaves the
        # positive-w case failing the intended [1e8, 2e8] and the mark silently XFAIL; here the
        # NEGATIVE-w cell (divided by -2e-8 + 1e-8 = -1e-8, doubling the point) flips under that
        # fix too, and both cells flip under an exact division or a grad-only eps. If either
        # assert fails, #3938 was (partly) fixed -- update or remove the warning in
        # convert_points_from_homogeneous and flip/remove the strict xfail above. eps=1e-8 is
        # passed explicitly so the pinned literals do not silently track a later change to the default.
        # Snippet used to generate expected (torch only, executed at each pinned dtype):
        #   cpfh = kornia.geometry.conversions.convert_points_from_homogeneous
        #   cpfh(torch.tensor([[2., 4., 2e-8]], dtype=torch.float64), eps=1e-8)
        #     -> [[66666666.66666666, 133333333.33333331]]   (f32: [[66666668.0, 133333336.0]])
        #   cpfh(torch.tensor([[2., 4., -2e-8]], dtype=torch.float64), eps=1e-8)
        #     -> [[-200000000.0, -400000000.0]]               (f32: identical)
        # At bfloat16 the outputs land within 0.15 % of the literals (66584576, -200278016),
        # inside rtol 1e-2, so the pin holds there too.
        if dtype == torch.float16:
            pytest.skip("float16 underflows w=2e-8 to 0, which is the |w| <= eps passthrough branch, not the eps bias")

        cpfh = kornia.geometry.conversions.convert_points_from_homogeneous

        out_pos = cpfh(torch.tensor([[2.0, 4.0, 2e-8]], device=device, dtype=dtype), eps=1e-8)
        out_neg = cpfh(torch.tensor([[2.0, 4.0, -2e-8]], device=device, dtype=dtype), eps=1e-8)

        expected_pos = torch.tensor([[6.6666668e7, 1.33333336e8]], device=device, dtype=dtype)
        expected_neg = torch.tensor([[-2e8, -4e8]], device=device, dtype=dtype)
        self.assert_close(out_pos, expected_pos, atol=0.0, rtol=1e-2)
        self.assert_close(out_neg, expected_neg, atol=0.0, rtol=1e-2)


def _skip_if_mps_clamp_caching(device):
    # Runtime probe instead of a torch-version pin, so the skip retires itself on any torch
    # build where the two clamps below return different values.
    if device.type == "mps" and torch.equal(
        torch.zeros(2, device=device).clamp(1e-8), torch.zeros(2, device=device).clamp(1e-7)
    ):
        pytest.skip(
            "this torch build caches clamp's scalar min per shape/dtype on MPS -- first value wins "
            "(seen on torch 2.9.1): z = torch.zeros(2, device='mps'); z.clamp(1e-8) then z.clamp(1e-7) "
            "both return 9.99999993922529e-09, while the same pair on cpu returns 1e-08 then "
            "1.0000000116860974e-07. The clamped eps this pin measures is therefore set by whichever "
            "earlier test clamped first, which is a torch defect, not a kornia one"
        )


def _assert_degenerate_size_cell(
    func_2d, func_3d, fill, ndim, arg_name, degenerate_size, expected, tols, device, dtype
):
    # Shared driver for the two kornia#3940 wart matrices below -- the normalize and
    # denormalize halves differ only in function pair, fill value, tolerances and expected
    # table -- so the eventual #3940 cleanup edits one body and one MPS-skip helper. eps=1e-8 is
    # passed explicitly so the pinned literals do not silently track a later change to the default eps
    # while the clamp bug itself is still present.
    _skip_if_mps_clamp_caching(device)

    if ndim == "2d":
        sizes = {"height": 5, "width": 7}
        func = func_2d
    else:
        sizes = {"depth": 5, "height": 7, "width": 9}
        func = func_3d
    sizes[arg_name] = degenerate_size
    pts = torch.full((1, len(sizes)), fill, device=device, dtype=dtype)

    out = func(pts, *sizes.values(), eps=1e-8)

    expected_t = torch.tensor([expected], device=device, dtype=dtype)
    if tols is None:
        assert_close(out, expected_t)
    else:
        atol, rtol = tols
        assert_close(out, expected_t, atol=atol, rtol=rtol)


class TestNormalizePixelCoordinates(BaseTester):
    def test_tensor_bhw2(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.geometry.conversions.normalize_pixel_coordinates(grid, height, width, eps=eps)

        self.assert_close(grid_norm, expected, atol=atol, rtol=rtol)

    def test_list(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.geometry.conversions.normalize_pixel_coordinates(grid, height, width, eps=eps)

        self.assert_close(grid_norm, expected, atol=atol, rtol=rtol)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if device == torch.device("cpu"):
            pytest.skip("NormalizePixelCoordinates not working on CPU with dynamo!")

        op = kornia.geometry.conversions.normalize_pixel_coordinates
        op_optimized = torch_optimizer(op)

        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        actual = op_optimized(grid, height, width)
        expected = op(grid, height, width)

        self.assert_close(actual, expected)

    def test_convention_corner_aligned_formula(self, device, dtype):
        # Convention pin: normalize_pixel_coordinates maps x -> 2*x/(W - 1) - 1 (corner-aligned,
        # i.e. the align_corners=True convention). grid_sample's *default* align_corners=False
        # convention, (2*x + 1)/W - 1, would give [-0.75, -0.25, 0.75] for the same input.
        # Snippet used to generate expected (stdlib only, W = 4):
        #   [2 * x / (4 - 1) - 1 for x in (0.0, 1.0, 3.0)] -> [-1.0, -0.3333333333333333, 1.0]
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 4, 4)

        expected = torch.tensor([[-1.0, -1.0], [-0.33333333, -0.33333333], [1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_positional_order_is_height_then_width(self, device, dtype):
        # Convention pin: the positional signature is (pixel_coordinates, height, width), which is
        # the reverse of the per-point (x, y) -> (width, height) scaling order: slot 0 is scaled by
        # width and slot 1 by height. Calling with H and W swapped would give [[5.0, -0.3333]].
        # Snippet used to generate expected (stdlib only, H = 2, W = 4):
        #   2 * 3.0 / (4 - 1) - 1, 2 * 1.0 / (2 - 1) - 1 -> (1.0, 1.0)
        pts = torch.tensor([[3.0, 1.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 2, 4)

        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_output_is_not_clamped(self, device, dtype):
        # Convention pin: nothing is clamped to [-1, 1] -- out-of-image coordinates extrapolate
        # linearly past it.
        # Snippet used to generate expected (stdlib only, H = W = 4):
        #   2 * 10.0 / (4 - 1) - 1, 2 * 0.0 / (4 - 1) - 1 -> (5.666666666666666, -1.0)
        pts = torch.tensor([[10.0, 0.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 4, 4)

        expected = torch.tensor([[5.6666667, -1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_grid_sample_needs_align_corners_true(self, device, dtype):
        # Convention pin: feeding normalized coordinates to torch.nn.functional.grid_sample
        # requires align_corners=True to be passed explicitly. With it, the three normalized
        # pixel centres sample back the exact pixel values; grid_sample's own default
        # (align_corners=None -> False) instead places u = -1, -1/3, 1 at pixels
        # ((u + 1) * 4 - 1) / 2 = -0.5, 0.8333, 3.5, i.e. half a pixel outside the image at
        # both ends, so every sampled value would be wrong.
        # Snippet used to generate expected (stdlib only, W = 4, img = arange(16).view(4, 4)):
        #   img[0, 0], img[1, 1], img[3, 3] -> 0.0, 5.0, 15.0
        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], device=device, dtype=dtype)
        grid = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 4, 4).view(1, 1, 3, 2)

        sampled_aligned = torch.nn.functional.grid_sample(img, grid, align_corners=True).flatten()

        self.assert_close(sampled_aligned, torch.tensor([0.0, 5.0, 15.0], device=device, dtype=dtype))

    def test_convention_3d_component_order_is_depth_x_y(self, device, dtype):
        # Convention pin (normalize_pixel_coordinates3d has no test class of its own): the
        # component order is (d, x, y) -- depth first, then x scaled by width, then y scaled by
        # height. It is NOT (x, y, z): reading the same three numbers that way sends the point
        # out of range instead of to the far corner.
        # Snippet used to generate expected (stdlib only, D = 3, H = 5, W = 9):
        #   2 * 2 / (3 - 1) - 1, 2 * 8 / (9 - 1) - 1, 2 * 4 / (5 - 1) - 1 -> (1.0, 1.0, 1.0)
        #   the (x, y, z) reading [2, 4, 8] gives 2 * 2 / 2 - 1, 2 * 4 / 8 - 1, 2 * 8 / 4 - 1
        #                                      -> (1.0, 0.0, 3.0)
        far_corner = torch.tensor([[2.0, 8.0, 4.0]], device=device, dtype=dtype)
        out = kornia.geometry.conversions.normalize_pixel_coordinates3d(far_corner, 3, 5, 9)
        self.assert_close(out, torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=dtype))

        swapped = torch.tensor([[2.0, 4.0, 8.0]], device=device, dtype=dtype)
        out_swapped = kornia.geometry.conversions.normalize_pixel_coordinates3d(swapped, 3, 5, 9)
        self.assert_close(out_swapped, torch.tensor([[1.0, 0.0, 3.0]], device=device, dtype=dtype))

    # Wart-pin matrix for kornia#3940, normalizing half: one cell per (size argument x
    # degenerate class) of normalize_pixel_coordinates and normalize_pixel_coordinates3d,
    # asserting the CURRENT broken output that the docstring warnings document. If any cell
    # fails, #3940 was (partly) fixed -- update or remove the degenerate-size warnings in
    # normalize_pixel_coordinates, denormalize_pixel_coordinates, normalize_pixel_coordinates3d
    # and denormalize_pixel_coordinates3d. The cells are NOT a contract that degenerate sizes
    # must keep returning these values.
    # They are regular tests rather than strict xfails on purpose: the intended behavior
    # (raise ValueError, or clamp the *output*, or keep the current pass-through) is a
    # maintainer decision, and a strict xfail asserting one of those answers would stay
    # silently XFAIL forever if a different one were chosen. A wart pin flips loudly under
    # every polarity, and covering the full matrix means any partial fix -- one function, one
    # argument, or one degenerate class -- flips at least one cell.
    # Exactly one size argument is degenerate per cell (the others stay at 5/7 in 2-D and
    # 5/7/9 in 3-D), so the finite components pin which argument was degenerated. All three
    # classes give the same output because the mechanism is `(size - 1).clamp(eps)`: size 1
    # gives 0, size 0 gives -1 and size -3 gives -4, all clamped up to eps = 1e-8, so the
    # factor becomes 2e8.
    # Snippet used to generate expected (torch only; same output for bad in (1, 0, -3)):
    #   npc = kornia.geometry.conversions.normalize_pixel_coordinates
    #   npc3 = kornia.geometry.conversions.normalize_pixel_coordinates3d
    #   npc(torch.tensor([[1., 1.]], dtype=torch.float64), bad, 7, eps=1e-8)
    #     -> [[-0.6666666666666667, 199999999.0]]
    #   npc(torch.tensor([[1., 1.]], dtype=torch.float64), 5, bad, eps=1e-8) -> [[199999999.0, -0.5]]
    #   npc3(torch.tensor([[1., 1., 1.]], dtype=torch.float64), bad, 7, 9, eps=1e-8)
    #     -> [[199999999.0, -0.75, -0.6666666666666667]]
    #   npc3(torch.tensor([[1., 1., 1.]], dtype=torch.float64), 5, bad, 9, eps=1e-8)
    #     -> [[-0.5, -0.75, 199999999.0]]
    #   npc3(torch.tensor([[1., 1., 1.]], dtype=torch.float64), 5, 7, bad, eps=1e-8)
    #     -> [[-0.5, 199999999.0, -0.6666666666666667]]
    # At float16 2e8 overflows to inf and the literal overflows identically; at bfloat16 both
    # sides round to 200278016.0, so the comparison stays meaningful at every dtype.
    @pytest.mark.parametrize("degenerate_size", [1, 0, -3], ids=["one", "zero", "negative"])
    @pytest.mark.parametrize(
        ("ndim", "arg_name", "expected"),
        [
            ("2d", "height", [-0.6666667, 199999999.0]),
            ("2d", "width", [199999999.0, -0.5]),
            ("3d", "depth", [199999999.0, -0.75, -0.6666667]),
            ("3d", "height", [-0.5, -0.75, 199999999.0]),
            ("3d", "width", [-0.5, 199999999.0, -0.6666667]),
        ],
        ids=["2d-height", "2d-width", "3d-depth", "3d-height", "3d-width"],
    )
    def test_wart_degenerate_size_matrix(self, ndim, arg_name, degenerate_size, expected, device, dtype):
        _assert_degenerate_size_cell(
            kornia.geometry.conversions.normalize_pixel_coordinates,
            kornia.geometry.conversions.normalize_pixel_coordinates3d,
            1.0,
            ndim,
            arg_name,
            degenerate_size,
            expected,
            None,
            device,
            dtype,
        )

    def test_wart_all_size_arguments_degenerate_together(self, device, dtype):
        # Wart pin for kornia#3940, companion to the matrix above: both sizes degenerate at
        # once also passes silently, exploding every component. Flips together with the matrix
        # when #3940 is fixed -- see the cleanup note above the matrix.
        # Snippet used to generate expected (torch only):
        #   npc = kornia.geometry.conversions.normalize_pixel_coordinates
        #   npc(torch.tensor([[1., 1.]], dtype=torch.float64), 1, 1, eps=1e-8)
        #     -> [[199999999.0, 199999999.0]]
        _skip_if_mps_clamp_caching(device)

        pts = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 1, 1, eps=1e-8)

        expected = torch.tensor([[199999999.0, 199999999.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)


def test_wart_default_eps_1e_8_backs_the_quoted_warning_numbers():
    # The wart pins in this file pass eps=1e-8 explicitly so their literals do not track the
    # default, which leaves the default itself pinned by nothing while six docstring warnings
    # quote numbers that hold only for eps=1e-8: cart2pol (rho = 1e-04 at the origin),
    # convert_points_from_homogeneous (the -1/3 and +1 relative errors at w = +/-2e-8),
    # normalize_pixel_coordinates and normalize_pixel_coordinates3d (the 199999999.0 / 2e8
    # blow-up factor) and denormalize_pixel_coordinates / denormalize_pixel_coordinates3d (the
    # 5e-09 collapse factor). If this fails, the default moved -- rework those warnings'
    # numbers together with this list.
    for op_name in (
        "cart2pol",
        "convert_points_from_homogeneous",
        "normalize_pixel_coordinates",
        "denormalize_pixel_coordinates",
        "normalize_pixel_coordinates3d",
        "denormalize_pixel_coordinates3d",
    ):
        op = getattr(kornia.geometry.conversions, op_name)
        assert inspect.signature(op).parameters["eps"].default == 1e-8, op_name


def test_wart_float16_underflowed_default_eps_flips_branches(device):
    # Wart pins for the float16 sentences of the #3939 and #3938 warnings. float16 is
    # hardcoded (no dtype fixture) so the pins run in every test configuration: the float16
    # legs of the wart pins above are skipped because the default eps=1e-8 underflows to 0
    # there, which is exactly the behavior pinned here. eps is left at its default on purpose
    # -- the underflow of the *default* is the claim. atol=rtol=0.0 because both claims are
    # exactness claims: with the float16 default tolerance (1e-3) the eps-biased
    # rho = 1e-4 of the other branch would still compare equal to 0.
    # Snippet used to generate expected (torch only, executed on cpu float16):
    #   cart2pol(torch.tensor(0., dtype=torch.float16), torch.tensor(0., dtype=torch.float16))[0]
    #     -> 0.0  (not sqrt(eps) = 1e-4: eps underflows the sum inside the sqrt)
    #   convert_points_from_homogeneous(torch.tensor([[2., 4., 2e-8]], dtype=torch.float16))
    #     -> [[2., 4.]]  (w underflows to 0 and takes the abs(w) <= eps passthrough branch)
    zero = torch.tensor(0.0, device=device, dtype=torch.float16)
    rho = kornia.geometry.conversions.cart2pol(zero, zero)[0]
    assert_close(rho, zero, atol=0.0, rtol=0.0)

    out = kornia.geometry.conversions.convert_points_from_homogeneous(
        torch.tensor([[2.0, 4.0, 2e-8]], device=device, dtype=torch.float16)
    )
    assert_close(out, torch.tensor([[2.0, 4.0]], device=device, dtype=torch.float16), atol=0.0, rtol=0.0)


def test_wart_float16_degenerate_roundtrip_is_inf_then_nan(device):
    # Wart pin for the float16 sentence of the #3940 warnings, float16-hardcoded like the test
    # above: with the default eps underflowed to 0, the clamp keeps the degenerate denominator
    # at 0, the normalized component is inf (not the 2e8 the other dtypes pin) and the
    # denormalize round trip of that is nan, not the input.
    # Snippet used to generate expected (torch only, executed on cpu float16):
    #   normalize_pixel_coordinates(torch.ones(1, 2, dtype=torch.float16), 1, 1) -> [[inf, inf]]
    #   denormalize_pixel_coordinates(<that>, 1, 1) -> [[nan, nan]]
    _skip_if_mps_clamp_caching(device)

    ones = torch.ones(1, 2, device=device, dtype=torch.float16)
    norm = kornia.geometry.conversions.normalize_pixel_coordinates(ones, 1, 1)
    assert (norm == torch.inf).all()

    denorm = kornia.geometry.conversions.denormalize_pixel_coordinates(norm, 1, 1)
    assert torch.isnan(denorm).all()


class TestDenormalizePixelCoordinates(BaseTester):
    def test_tensor_bhw2(self, device, dtype):
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.geometry.conversions.denormalize_pixel_coordinates(grid, height, width)

        self.assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_list(self, device, dtype):
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.geometry.conversions.denormalize_pixel_coordinates(grid, height, width)

        self.assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if device == torch.device("cpu"):
            pytest.xfail("DenormalizePixelCoordinates not working on CPU with dynamo!")

        op = kornia.geometry.conversions.denormalize_pixel_coordinates
        op_optimized = torch_optimizer(op)

        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        actual = op_optimized(grid, height, width)
        expected = op(grid, height, width)

        self.assert_close(actual, expected)

    def test_convention_corner_aligned_inverse(self, device, dtype):
        # Convention pin: denormalize_pixel_coordinates is the corner-aligned inverse,
        # x = (W - 1) * (x_norm + 1) / 2, taken positionally as (coords, height, width) with
        # (x, y) points. grid_sample's align_corners=False convention, ((x_norm + 1) * W - 1)/2,
        # would give [[3.5, -0.5]] for the same input.
        # Snippet used to generate expected (stdlib only, H = 2, W = 4):
        #   (4 - 1) * (1.0 + 1) / 2, (2 - 1) * (-1.0 + 1) / 2 -> (3.0, 0.0)
        pts_norm = torch.tensor([[1.0, -1.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.denormalize_pixel_coordinates(pts_norm, 2, 4)

        expected = torch.tensor([[3.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_roundtrip_denormalize_of_normalize(self, device, dtype):
        # Convention pin: denormalize(normalize(p)) == p on a non-degenerate, non-square,
        # non-identity image size, so the two formulas are exact mutual inverses.
        # Snippet used to generate expected (by hand): the input itself.
        pts = torch.tensor([[1.0, 2.0], [3.0, 0.0]], device=device, dtype=dtype)

        norm = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 5, 7)
        out = kornia.geometry.conversions.denormalize_pixel_coordinates(norm, 5, 7)

        self.assert_close(out, pts)

    def test_convention_3d_component_order_and_roundtrip(self, device, dtype):
        # Convention pin (denormalize_pixel_coordinates3d has no test class of its own): same
        # (d, x, y) order as the 3-D normalizer, so the normalized origin maps to the per-axis
        # centres ((D - 1)/2, (W - 1)/2, (H - 1)/2), and the pair round-trips exactly.
        # Snippet used to generate expected (stdlib only, D = 3, H = 5, W = 9):
        #   (3 - 1) / 2, (9 - 1) / 2, (5 - 1) / 2 -> (1.0, 4.0, 2.0)
        centre = kornia.geometry.conversions.denormalize_pixel_coordinates3d(
            torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype), 3, 5, 9
        )
        self.assert_close(centre, torch.tensor([[1.0, 4.0, 2.0]], device=device, dtype=dtype))

        pts = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        norm = kornia.geometry.conversions.normalize_pixel_coordinates3d(pts, 3, 5, 9)
        out = kornia.geometry.conversions.denormalize_pixel_coordinates3d(norm, 3, 5, 9)
        self.assert_close(out, pts)

    # Wart-pin matrix for kornia#3940, denormalizing half: one cell per (size argument x
    # degenerate class) of denormalize_pixel_coordinates and denormalize_pixel_coordinates3d,
    # asserting the CURRENT broken output that the docstring warnings document. If any cell
    # fails, #3940 was (partly) fixed -- update or remove the degenerate-size warnings in
    # normalize_pixel_coordinates, denormalize_pixel_coordinates, normalize_pixel_coordinates3d
    # and denormalize_pixel_coordinates3d. The cells are NOT a contract that degenerate sizes
    # must keep returning these values; see the polarity note on the normalize wart matrix,
    # which also explains why exactly one argument is degenerate per cell and why all three
    # classes (1, 0, -3) give the same output.
    # Here the clamped denominator multiplies instead of divides, so the degenerate axis
    # collapses to eps / 2 = 5e-09 rather than exploding to 2e8. The tolerance is tight
    # (rtol 1e-6, atol 0) on purpose: at atol 1e-2 the collapsed component would compare
    # equal to any small number, including the 0.0 a "clamp the output" fix might return.
    # It still holds at every dtype -- at float16 5e-09 underflows to 0.0 on both the
    # measured and the literal side, at bfloat16 both round to 5.005858838558197e-09, and the
    # finite components 2.0/3.0/4.0 are exact in every dtype.
    # Snippet used to generate expected (torch only; same output for bad in (1, 0, -3)):
    #   dpc = kornia.geometry.conversions.denormalize_pixel_coordinates
    #   dpc3 = kornia.geometry.conversions.denormalize_pixel_coordinates3d
    #   dpc(torch.tensor([[0., 0.]], dtype=torch.float64), bad, 7, eps=1e-8) -> [[3.0, 5e-09]]
    #   dpc(torch.tensor([[0., 0.]], dtype=torch.float64), 5, bad, eps=1e-8) -> [[5e-09, 2.0]]
    #   dpc3(torch.tensor([[0., 0., 0.]], dtype=torch.float64), bad, 7, 9, eps=1e-8)
    #     -> [[5e-09, 4.0, 3.0]]
    #   dpc3(torch.tensor([[0., 0., 0.]], dtype=torch.float64), 5, bad, 9, eps=1e-8)
    #     -> [[2.0, 4.0, 5e-09]]
    #   dpc3(torch.tensor([[0., 0., 0.]], dtype=torch.float64), 5, 7, bad, eps=1e-8)
    #     -> [[2.0, 5e-09, 3.0]]
    @pytest.mark.parametrize("degenerate_size", [1, 0, -3], ids=["one", "zero", "negative"])
    @pytest.mark.parametrize(
        ("ndim", "arg_name", "expected"),
        [
            ("2d", "height", [3.0, 5e-09]),
            ("2d", "width", [5e-09, 2.0]),
            ("3d", "depth", [5e-09, 4.0, 3.0]),
            ("3d", "height", [2.0, 4.0, 5e-09]),
            ("3d", "width", [2.0, 5e-09, 3.0]),
        ],
        ids=["2d-height", "2d-width", "3d-depth", "3d-height", "3d-width"],
    )
    def test_wart_degenerate_size_matrix(self, ndim, arg_name, degenerate_size, expected, device, dtype):
        _assert_degenerate_size_cell(
            kornia.geometry.conversions.denormalize_pixel_coordinates,
            kornia.geometry.conversions.denormalize_pixel_coordinates3d,
            0.0,
            ndim,
            arg_name,
            degenerate_size,
            expected,
            (0.0, 1e-6),
            device,
            dtype,
        )


class TestProjectPoints(BaseTester):
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
        self.assert_close(point_3d, point_3d_hat, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # TODO: point [0, 0, 0] crashes
        points_3d = torch.ones(1, 3, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.camera.project_points, (points_3d, camera_matrix))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.camera.project_points
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_3d, camera_matrix)
        expected = op(points_3d, camera_matrix)

        self.assert_close(actual, expected)


class TestDenormalizePointsWithIntrinsics(BaseTester):
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

    def test_smoke_batch_n(self, device, dtype):
        points_2d = torch.zeros(2, 9, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 9, 2)

    def test_toy(self, device, dtype):
        point_2d = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.denormalize_points_with_intrinsics
        expected = torch.tensor([[192.0, 192.0]], device=device, dtype=dtype)
        self.assert_close(op(point_2d, camera_matrix), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_2d = torch.zeros(1, 2, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.denormalize_points_with_intrinsics, (points_2d, camera_matrix))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.conversions.denormalize_points_with_intrinsics
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_2d, camera_matrix)
        expected = op(points_2d, camera_matrix)

        self.assert_close(actual, expected)

    def test_convention_maps_normalized_camera_points_to_pixels(self, device, dtype):
        # Convention pin: the input is in *normalized camera* coordinates and the output is in
        # *pixel* coordinates, using the pinhole layout u = x * fx + cx, v = y * fy + cy.
        # fx != fy and cx != cy so a transposed or swapped read of K is caught.
        # Snippet used to generate expected (stdlib only, fx=100, fy=200, cx=320, cy=240):
        #   1.0 * 100 + 320, 1.0 * 200 + 240 -> (420.0, 440.0)
        points_norm = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_norm, camera_matrix)

        expected = torch.tensor([[420.0, 440.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)


class TestNormalizePointsWithIntrinsics(BaseTester):
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

    def test_smoke_batch_n(self, device, dtype):
        points_2d = torch.zeros(2, 10, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 10, 2)

    def test_norm_unnorm(self, device, dtype):
        point_2d = torch.tensor([[128.0, 128.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        back = kornia.geometry.conversions.denormalize_points_with_intrinsics
        point_2d_norm = op(point_2d, camera_matrix)
        point_2d_hat = back(point_2d_norm, camera_matrix)
        self.assert_close(point_2d, point_2d_hat, atol=1e-4, rtol=1e-4)

    def test_toy(self, device, dtype):
        point_2d = torch.tensor([[192.0, 192.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        out = op(point_2d, camera_matrix)
        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_2d = torch.zeros(1, 2, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.normalize_points_with_intrinsics, (points_2d, camera_matrix))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_2d, camera_matrix)
        expected = op(points_2d, camera_matrix)

        self.assert_close(actual, expected)

    def test_convention_intrinsics_layout_fx_fy_cx_cy(self, device, dtype):
        # Convention pin: K is the standard row-major pinhole matrix, fx = K[..., 0, 0],
        # fy = K[..., 1, 1], cx = K[..., 0, 2], cy = K[..., 1, 2], and the point is (u, v) in
        # pixels, so x = (u - cx)/fx, y = (v - cy)/fy. fx != fy and cx != cy, so swapping the
        # two focal lengths would give [[0.5, 2.0]] and a transposed K would not divide at all.
        # Snippet used to generate expected (stdlib only, fx=100, fy=200, cx=320, cy=240):
        #   (420.0 - 320) / 100, (440.0 - 240) / 200 -> (1.0, 1.0)
        points_2d = torch.tensor([[420.0, 440.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)

        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_skew_term_is_ignored(self, device, dtype):
        # Convention pin: only the diagonal fx, fy and the [:2, 2] column of K are read -- the
        # skew entry K[..., 0, 1] is silently ignored, so a skewed K gives the same answer as
        # the skew-free one. A skew-aware implementation would return (1.0 - 7/100, 1.0).
        # Snippet used to generate expected (stdlib only): identical to the skew-free result,
        #   (420.0 - 320) / 100, (440.0 - 240) / 200 -> (1.0, 1.0)
        points_2d = torch.tensor([[420.0, 440.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 7.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)

        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)


class TestRt2Extrinsics(BaseTester):
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

        self.assert_close(R, R2, rtol=1e-4, atol=1e-5)
        self.assert_close(t, t2, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [5])
    def test_gradcheck(self, batch_size, device):
        R = torch.rand(batch_size, 3, 3, dtype=torch.float64, device=device)
        t = torch.rand(batch_size, 3, 1, dtype=torch.float64, device=device)
        self.gradcheck(kornia.geometry.conversions.Rt_to_matrix4x4, (R, t))


class TestCamtoworldGraphicsToVision(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        t_vis = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R_vis = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        K_vis = Rt_to_matrix4x4(R_vis, t_vis)
        K_graf = camtoworld_vision_to_graphics_4x4(K_vis)

        expected = torch.tensor(
            [[0, 0, -1, 2], [0, -1, 0, 3], [-1, 0, 0, 4], [0, 0, 0, 1]], device=device, dtype=dtype
        )[None].repeat(batch_size, 1, 1)

        self.assert_close(K_graf, expected, rtol=1e-4, atol=1e-5)
        R_graf, t_graf = camtoworld_vision_to_graphics_Rt(R_vis, t_vis)
        expected_R = torch.tensor([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], device=device, dtype=dtype)[None].repeat(
            batch_size, 1, 1
        )
        expected_t = torch.tensor([2, 3, 4], device=device, dtype=dtype).reshape(1, 3, 1).repeat(batch_size, 1, 1)

        self.assert_close(t_graf, expected_t, rtol=1e-4, atol=1e-5)
        self.assert_close(R_graf, expected_R, rtol=1e-4, atol=1e-5)

        Kvis_back = camtoworld_graphics_to_vision_4x4(K_graf)
        self.assert_close(Kvis_back, K_vis, rtol=1e-4, atol=1e-5)

        R_vis_back, t_vis_back = camtoworld_graphics_to_vision_Rt(R_graf, t_graf)
        self.assert_close(R_vis_back, R_vis, rtol=1e-4, atol=1e-5)
        self.assert_close(t_vis_back, t_vis, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device):
        t_vis = torch.tensor([2, 3, 4], device=device, dtype=torch.float64).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=torch.float64)[None]
        R_vis = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        K_vis = Rt_to_matrix4x4(R_vis, t_vis)
        self.gradcheck(camtoworld_graphics_to_vision_4x4, (K_vis,))
        self.gradcheck(camtoworld_vision_to_graphics_4x4, (K_vis,))


class TestCamtoworldRtToPoseRt(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        t = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)

        Rp, tp = camtoworld_to_worldtocam_Rt(R, t)

        expected_Rp = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=device, dtype=dtype)[None].repeat(
            batch_size, 1, 1
        )
        expected_tp = torch.tensor([4, -3, -2], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        self.assert_close(Rp, expected_Rp, rtol=1e-4, atol=1e-5)
        self.assert_close(tp, expected_tp, rtol=1e-4, atol=1e-5)

        Rback, tback = worldtocam_to_camtoworld_Rt(Rp, tp)
        self.assert_close(Rback, R, rtol=1e-4, atol=1e-5)
        self.assert_close(tback, t, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device):
        t = torch.tensor([2, 3, 4], device=device, dtype=torch.float64).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=torch.float64)[None]
        R = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        self.gradcheck(camtoworld_to_worldtocam_Rt, (R, t))
        self.gradcheck(worldtocam_to_camtoworld_Rt, (R, t))


class TestCARKitToColmap(BaseTester):
    def test_everything(self, device, dtype):
        # generate input data
        t = torch.tensor([1, 0, 0], device=device, dtype=dtype).view(1, 3, 1)
        ang_deg = torch.tensor([45, 60.0, 0.0], device=device, dtype=dtype)[None]
        ang_rad = kornia.geometry.conversions.deg2rad(ang_deg)
        qvec = kornia.geometry.axis_angle_to_quaternion(ang_rad)

        q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(qvec, t)

        angles_colmap = kornia.geometry.conversions.quaternion_to_axis_angle(q_colmap)
        angles_colmap = kornia.geometry.conversions.rad2deg(angles_colmap)
        expected_angles = torch.tensor([[116.8870620728, 0.0, -71.7524719238]], device=device, dtype=dtype)
        expected_t = torch.tensor([[[-0.5256], [0.3558], [0.7727]]], device=device, dtype=dtype)

        self.assert_close(angles_colmap, expected_angles, rtol=1e-4, atol=1e-5)
        self.assert_close(t_colmap, expected_t, rtol=1e-4, atol=1e-5)


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
        self.gradcheck(euler_from_quaternion, (q.w, q.x, q.y, q.z))

    @pytest.mark.skipif(
        torch_version() in {"2.0.1", "2.1.2", "2.2.2", "2.3.1"} and sys.version_info.minor == 8,
        reason="Not working on 2.0",
    )
    def test_dynamo(self, device, dtype, torch_optimizer):
        q = Quaternion.random(batch_size=1)
        q = q.to(device, dtype)
        op = euler_from_quaternion
        op_optimized = torch_optimizer(op)
        self.assert_close(op(q.w, q.x, q.y, q.z), op_optimized(q.w, q.x, q.y, q.z))

    def test_forth_and_back(self, device, dtype):
        q = Quaternion.random(batch_size=2)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        # TODO: check hwo to prevent getting inverted angles sometimes
        self.assert_close(q.w.abs(), qw.abs())
        self.assert_close(q.x.abs(), qx.abs())
        self.assert_close(q.y.abs(), qy.abs())
        self.assert_close(q.z.abs(), qz.abs())


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
        self.gradcheck(quaternion_from_euler, (roll, pitch, yaw))

    def test_dynamo(self, device, dtype, torch_optimizer):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)

        op = quaternion_from_euler
        op_optimized = torch_optimizer(op)

        actual = op_optimized(roll, pitch, yaw)
        expected = op(roll, pitch, yaw)

        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1])
        self.assert_close(actual[2], expected[2])

    def test_forth_and_back(self, device, dtype):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        roll_new, pitch_new, yaw_new = euler_from_quaternion(qw, qx, qy, qz)
        self.assert_close(roll, roll_new)
        self.assert_close(pitch, pitch_new)
        self.assert_close(yaw, yaw_new)

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


@pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
def test_vector_to_skew_symmetric_matrix(batch_size, device, dtype):
    if batch_size is None:
        vector = torch.rand(3, device=device, dtype=dtype)
    else:
        vector = torch.rand((batch_size, 3), device=device, dtype=dtype)
    skew_symmetric_matrix = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(vector)
    assert skew_symmetric_matrix.shape[-1] == 3
    assert skew_symmetric_matrix.shape[-2] == 3
    z = torch.zeros_like(vector[..., 0])
    assert_close(skew_symmetric_matrix[..., 0, 0], z)
    assert_close(skew_symmetric_matrix[..., 1, 1], z)
    assert_close(skew_symmetric_matrix[..., 2, 2], z)
    assert_close(skew_symmetric_matrix[..., 0, 1], -vector[..., 2])
    assert_close(skew_symmetric_matrix[..., 1, 0], vector[..., 2])
    assert_close(skew_symmetric_matrix[..., 0, 2], vector[..., 1])
    assert_close(skew_symmetric_matrix[..., 2, 0], -vector[..., 1])
    assert_close(skew_symmetric_matrix[..., 1, 2], -vector[..., 0])
    assert_close(skew_symmetric_matrix[..., 2, 1], vector[..., 0])

    # Convention's enforcement point: [v]x @ x == cross(v, x) -- the vector is the LEFT factor
    # of the cross product, NOT cross(x, v), which is the negation.
    # Snippet used to generate expected (stdlib only):
    #   v, x = (1, 2, 3), (4, 5, 6)
    #   cross(v, x) = (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) -> (-3, 6, -3)
    v = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
    x = torch.tensor([4.0, 5.0, 6.0], device=device, dtype=dtype)
    skew = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(v)
    expected_cross = torch.tensor([-3.0, 6.0, -3.0], device=device, dtype=dtype)
    assert_close(skew @ x, expected_cross)


class TestAxisAngleToRotationMatrix:
    def test_identity_rotation(self):
        aa = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
        R = axis_angle_to_rotation_matrix(aa)
        Id = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        assert torch.allclose(R, Id, atol=1e-6)

    def test_90deg_x_axis(self):
        aa = torch.tensor([[torch.pi / 2, 0.0, 0.0]], dtype=torch.float64)
        R = axis_angle_to_rotation_matrix(aa).squeeze(0)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(R, expected, atol=1e-6)

    def test_180deg_y_axis(self):
        aa = torch.tensor([[0.0, torch.pi, 0.0]], dtype=torch.float64)
        R = axis_angle_to_rotation_matrix(aa).squeeze(0)
        expected = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(R, expected, atol=1e-6)

    def test_batched_input(self):
        aa = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [torch.pi / 2, 0.0, 0.0],
                [0.0, torch.pi, 0.0],
            ],
            dtype=torch.float64,
        )
        R = axis_angle_to_rotation_matrix(aa)
        assert R.shape == (3, 3, 3)

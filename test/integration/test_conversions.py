from typing import Optional

import pytest
import numpy as np

import kornia
from kornia.testing import tensor_to_gradcheck_var, create_eye_batch

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestAngleAxisToQuaternionToAngleAxis:
    def test_zero_angle(self, device, dtype):
        angle_axis = torch.tensor([0., 0., 0.], device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        angle_axis_hat = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis_hat, angle_axis, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_small_angle(self, axis, device, dtype):
        theta = 1e-2
        array = [0., 0., 0.]
        array[axis] = theta
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        angle_axis_hat = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis_hat, angle_axis, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_rotation(self, axis, device, dtype):
        # half_sqrt2 = 0.5 * np.sqrt(2)
        array = [0., 0., 0.]
        array[axis] = kornia.pi / 2.
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        angle_axis_hat = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis_hat, angle_axis, atol=1e-4, rtol=1e-4)


class TestQuaternionToAngleAxisToQuaternion:
    def test_unit_quaternion(self, device, dtype):
        quaternion = torch.tensor([0., 0., 0., 1.], device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        quaternion_hat = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion_hat, quaternion, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_rotation(self, axis, device, dtype):
        array = [0., 0., 0., 0.]
        array[axis] = 1.
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        quaternion_hat = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion_hat, quaternion, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_small_angle(self, axis, device, dtype):
        theta = 1e-2
        array = [0., 0., 0., np.cos(theta / 2)]
        array[axis] = np.sin(theta / 2)
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        quaternion_hat = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion_hat, quaternion, atol=1e-4, rtol=1e-4)


class TestQuaternionToRotationMatrixToAngleAxis:
    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_qma(self, axis, device, dtype):
        array = [[0., 0., 0., 0.]]
        array[0][axis] = 1.
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        assert quaternion.shape[-1] == 4

        mm = kornia.quaternion_to_rotation_matrix(quaternion)
        assert mm.shape[-1] == 3
        assert mm.shape[-2] == 3

        angle_axis = kornia.rotation_matrix_to_angle_axis(mm)
        assert angle_axis.shape[-1] == 3

        quaternion_hat = kornia.angle_axis_to_quaternion(angle_axis)
        assert_allclose(quaternion_hat, quaternion, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_qam(self, axis, device, dtype):
        array = [[0., 0., 0., 0.]]
        array[0][axis] = 1.
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        assert quaternion.shape[-1] == 4

        angle_axis = kornia.quaternion_to_angle_axis(quaternion)
        assert angle_axis.shape[-1] == 3

        rot_m = kornia.angle_axis_to_rotation_matrix(angle_axis)
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3

        quaternion_hat = kornia.rotation_matrix_to_quaternion(rot_m)
        assert_allclose(quaternion_hat, quaternion, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_amq(self, axis, device, dtype):
        array = [[0., 0., 0.]]
        array[0][axis] = kornia.pi / 2.
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        assert angle_axis.shape[-1] == 3

        rot_m = kornia.angle_axis_to_rotation_matrix(angle_axis)
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3

        quaternion = kornia.rotation_matrix_to_quaternion(rot_m)
        assert quaternion.shape[-1] == 4

        angle_axis_hat = kornia.quaternion_to_angle_axis(quaternion)
        assert_allclose(angle_axis_hat, angle_axis, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_aqm(self, axis, device, dtype):
        array = [[0., 0., 0.]]
        array[0][axis] = kornia.pi / 2.
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        assert angle_axis.shape[-1] == 3

        quaternion = kornia.angle_axis_to_quaternion(angle_axis)
        assert quaternion.shape[-1] == 4

        rot_m = kornia.quaternion_to_rotation_matrix(quaternion)
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3

        angle_axis_hat = kornia.rotation_matrix_to_angle_axis(rot_m)
        assert_allclose(angle_axis_hat, angle_axis, atol=1e-4, rtol=1e-4)


class TestAngleOfRotations:
    """
    See: https://arxiv.org/pdf/1711.02508.pdf
    """

    @staticmethod
    def matrix_angle_abs(mx):
        """Unsigned rotation matrix angle"""
        return torch.arccos((torch.trace(mx[..., :3, :3]) - 1.) / 2.)

    @staticmethod
    def axis_and_angle_to_rotation_matrix(axis_name: str, angle: float, device, dtype):
        """
        See also: https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
        """
        axis_name = axis_name.lower()
        assert axis_name in ('x', 'y', 'z')
        sn = np.sin(angle)
        cs = np.cos(angle)
        if axis_name == 'x':
            axis = torch.tensor((1., 0., 0.), device=device, dtype=dtype)
            rot_m = torch.tensor(
                ((1., 0., 0.),
                 (0., cs, -sn),
                 (0., sn, cs)),
                device=device, dtype=dtype)
        elif axis_name == 'y':
            axis = torch.tensor((0., 1., 0.), device=device, dtype=dtype)
            rot_m = torch.tensor(
                ((cs, 0., sn),
                 (0., 1., 0.),
                 (-sn, 0., cs)),
                device=device, dtype=dtype)
        elif axis_name == 'z':
            axis = torch.tensor((0., 0., 1.), device=device, dtype=dtype)
            rot_m = torch.tensor(
                ((cs, -sn, 0.),
                 (sn, cs, 0.),
                 (0., 0., 1.)),
                device=device, dtype=dtype)
        else:
            raise NotImplementedError(f'Not prepared for axis with name {axis_name}')

        return rot_m, axis

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    def test_axis_angle_to_rotation_matrix(self, axis_name, device, dtype):
        # Random angle in [-pi..pi]
        angle = np.random.random() * 2. * np.pi - np.pi
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(
            axis_name=axis_name, angle=angle, device=device, dtype=dtype)

        # Make sure the returned axis matches the named one, and the appropriate column
        if axis_name == 'x':
            assert_allclose(axis, (1., 0., 0.))
            assert_allclose(axis, rot_m[:3, 0])
        elif axis_name == 'y':
            assert_allclose(axis, (0., 1., 0.))
            assert_allclose(axis, rot_m[:3, 1])
        elif axis_name == 'z':
            assert_allclose(axis, (0., 0., 1.))
            assert_allclose(axis, rot_m[:3, 2])
        else:
            raise NotImplementedError(f'Not prepared for axis_name {axis_name}')

        # Make sure axes are perpendicular
        assert_allclose(torch.dot(rot_m[:3, 0], rot_m[:3, 1]), 0., atol=1.e-4, rtol=1.e-4)
        assert_allclose(torch.dot(rot_m[:3, 1], rot_m[:3, 2]), 0., atol=1.e-4, rtol=1.e-4)
        assert_allclose(torch.dot(rot_m[:3, 0], rot_m[:3, 2]), 0., atol=1.e-4, rtol=1.e-4)

        # Make sure axes are unit norm
        one = torch.tensor((1.,), device=device, dtype=dtype)
        assert torch.isclose(torch.linalg.norm(rot_m[:3, 0]), one, atol=1.e-4, rtol=1.e-4)
        assert torch.isclose(torch.linalg.norm(rot_m[:3, 1]), one, atol=1.e-4, rtol=1.e-4)
        assert torch.isclose(torch.linalg.norm(rot_m[:3, 2]), one, atol=1.e-4, rtol=1.e-4)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -135., -90., -45., 0., 45, 90, 135, 179.9))
    def test_matrix_angle(self, axis_name, angle_deg, device, dtype):
        angle = (angle_deg * kornia.pi / 180.).to(dtype).to(device)
        rot_m, _ = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(axis_name=axis_name,
                                                                          angle=angle,
                                                                          device=device,
                                                                          dtype=dtype)
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_allclose(np.abs(angle), matrix_angle_abs)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -90., -45., 0., 45, 90, 179.9))
    def test_quaternion(self, axis_name, angle_deg, device, dtype):
        angle = (angle_deg * kornia.pi / 180.).to(dtype).to(device)
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(axis_name=axis_name,
                                                                             angle=angle,
                                                                             device=device,
                                                                             dtype=dtype)
        quaternion = kornia.rotation_matrix_to_quaternion(rot_m)
        # compute quaternion rotation angle
        # See Section 2.4.4 Equation (105a) in https://arxiv.org/pdf/1711.02508.pdf
        angle_hat = 2. * torch.atan2(torch.linalg.norm(quaternion[..., :3]), quaternion[..., 3])
        # make sure it lands between [-pi..pi)
        while kornia.pi <= angle_hat:
            angle_hat -= 2. * kornia.pi
        # invert angle, if quaternion axis points in the opposite direction of the original axis
        if torch.dot(quaternion[..., :3], axis) < 0.:
            angle_hat *= -1.
        # quaternion angle should match input angle
        assert_allclose(angle_hat, angle, atol=1.e-4, rtol=1.e-4)
        # magnitude of angle should match matrix rotation angle
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_allclose(torch.abs(angle_hat), matrix_angle_abs, atol=1.e-4, rtol=1.e-4)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -90., -45., 0, 45, 90, 179.9))
    def test_angle_axis(self, axis_name, angle_deg, device, dtype):
        angle = (angle_deg * kornia.pi / 180.).to(dtype).to(device)
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(axis_name=axis_name,
                                                                             angle=angle,
                                                                             device=device,
                                                                             dtype=dtype)
        angle_axis = kornia.rotation_matrix_to_angle_axis(rot_m)
        # compute angle_axis rotation angle
        angle_hat = torch.linalg.norm(angle_axis)
        # invert angle, if angle_axis axis points in the opposite direction of the original axis
        if torch.dot(angle_axis, axis) < 0.:
            angle_hat *= -1.
        # angle_axis angle should match input angle
        assert_allclose(angle_hat, angle, atol=1.e-4, rtol=1.e-4)
        # magnitude of angle should match matrix rotation angle
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_allclose(torch.abs(angle_hat), matrix_angle_abs, atol=1.e-4, rtol=1.e-4)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -90., -45., 0, 45, 90, 179.9))
    def test_log_quaternion(self, axis_name, angle_deg, device, dtype):
        angle = (angle_deg * kornia.pi / 180.).to(dtype).to(device)
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(axis_name=axis_name,
                                                                             angle=angle,
                                                                             device=device,
                                                                             dtype=dtype)
        quaternion = kornia.rotation_matrix_to_quaternion(rot_m)
        log_q = kornia.quaternion_exp_to_log(quaternion)
        # compute angle_axis rotation angle
        angle_hat = 2. * torch.linalg.norm(log_q)
        # make sure it lands between [-pi..pi)
        while kornia.pi <= angle_hat:
            angle_hat -= 2. * kornia.pi
        # invert angle, if angle_axis axis points in the opposite direction of the original axis
        if torch.dot(log_q, axis) < 0.:
            angle_hat *= -1.
        # angle_axis angle should match input angle
        assert_allclose(angle_hat, angle, atol=1.e-4, rtol=1.e-4)
        # magnitude of angle should match matrix rotation angle
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_allclose(torch.abs(angle_hat), matrix_angle_abs, atol=1.e-4, rtol=1.e-4)

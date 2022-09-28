from __future__ import annotations

import numpy as np
import pytest
import torch

import kornia
from kornia.geometry.conversions import QuaternionCoeffOrder
from kornia.testing import assert_close


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


class TestAngleAxisToQuaternionToAngleAxis:
    def test_zero_angle(self, device, dtype, atol, rtol):
        angle_axis = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        angle_axis_hat = kornia.geometry.conversions.quaternion_to_angle_axis(
            quaternion, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(angle_axis_hat, angle_axis, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_small_angle(self, axis, device, dtype, atol, rtol):
        theta = 1.0e-2
        array = [0.0, 0.0, 0.0]
        array[axis] = theta
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        angle_axis_hat = kornia.geometry.conversions.quaternion_to_angle_axis(
            quaternion, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(angle_axis_hat, angle_axis, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_rotation(self, axis, device, dtype, atol, rtol):
        # half_sqrt2 = 0.5 * np.sqrt(2)
        array = [0.0, 0.0, 0.0]
        array[axis] = kornia.pi / 2.0
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        angle_axis_hat = kornia.geometry.conversions.quaternion_to_angle_axis(
            quaternion, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(angle_axis_hat, angle_axis, atol=atol, rtol=rtol)


class TestQuaternionToAngleAxisToQuaternion:
    def test_unit_quaternion_xyzw(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        with pytest.warns(UserWarning):
            quaternion_hat = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        quaternion_hat = kornia.geometry.conversions.angle_axis_to_quaternion(
            angle_axis, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_rotation_xyzw(self, axis, device, dtype, atol, rtol):
        array = [0.0, 0.0, 0.0, 0.0]
        array[axis] = 1.0
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        with pytest.warns(UserWarning):
            quaternion_hat = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_rotation(self, axis, device, dtype, atol, rtol):
        array = [0.0, 0.0, 0.0, 0.0]
        array[1 + axis] = 1.0
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        quaternion_hat = kornia.geometry.conversions.angle_axis_to_quaternion(
            angle_axis, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

        # just to be sure, check that mixing orders fails
        with pytest.warns(UserWarning):
            quaternion_hat_wrong = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert not torch.allclose(quaternion_hat_wrong, quaternion, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_small_angle_xyzw(self, axis, device, dtype, atol, rtol):
        theta = 1.0e-2
        array = [0.0, 0.0, 0.0, np.cos(theta / 2)]
        array[axis] = np.sin(theta / 2.0)
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        with pytest.warns(UserWarning):
            quaternion_hat = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_small_angle(self, axis, device, dtype, atol, rtol):
        theta = 1.0e-2
        array = [np.cos(theta / 2), 0.0, 0.0, 0.0]
        array[1 + axis] = np.sin(theta / 2.0)
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        quaternion_hat = kornia.geometry.conversions.angle_axis_to_quaternion(
            angle_axis, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

        # just to be sure, check that mixing orders fails
        with pytest.warns(UserWarning):
            quaternion_hat_wrong = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert not torch.allclose(quaternion_hat_wrong, quaternion, atol=atol, rtol=rtol)


class TestQuaternionToRotationMatrixToAngleAxis:
    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_qma_xyzw(self, axis, device, dtype, atol, rtol):
        array = [[0.0, 0.0, 0.0, 0.0]]
        array[0][axis] = 1.0  # `0 + axis` should fail when WXYZ
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        assert quaternion.shape[-1] == 4

        with pytest.warns(UserWarning):
            mm = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.XYZW)
        assert mm.shape[-1] == 3
        assert mm.shape[-2] == 3

        angle_axis = kornia.geometry.conversions.rotation_matrix_to_angle_axis(mm)
        assert angle_axis.shape[-1] == 3
        angle_axis_expected = [[0.0, 0.0, 0.0]]
        angle_axis_expected[0][axis] = kornia.pi
        angle_axis_expected = torch.tensor(angle_axis_expected, device=device, dtype=dtype)
        assert_close(angle_axis, angle_axis_expected, atol=atol, rtol=rtol)

        with pytest.warns(UserWarning):
            quaternion_hat = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_qma(self, axis, device, dtype, atol, rtol):
        array = [[0.0, 0.0, 0.0, 0.0]]
        array[0][1 + axis] = 1.0  # `1 + axis` this should fail when XYZW
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        assert quaternion.shape[-1] == 4

        mm = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert mm.shape[-1] == 3
        assert mm.shape[-2] == 3

        angle_axis = kornia.geometry.conversions.rotation_matrix_to_angle_axis(mm)
        assert angle_axis.shape[-1] == 3
        angle_axis_expected = [[0.0, 0.0, 0.0]]
        angle_axis_expected[0][axis] = kornia.pi
        angle_axis_expected = torch.tensor(angle_axis_expected, device=device, dtype=dtype)
        assert_close(angle_axis, angle_axis_expected, atol=atol, rtol=rtol)

        quaternion_hat = kornia.geometry.conversions.angle_axis_to_quaternion(
            angle_axis, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_qam_xyzw(self, axis, device, dtype, atol, rtol):
        array = [[0.0, 0.0, 0.0, 0.0]]
        array[0][axis] = 1.0
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        assert quaternion.shape[-1] == 4

        with pytest.warns(UserWarning):
            angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert angle_axis.shape[-1] == 3

        rot_m = kornia.geometry.conversions.angle_axis_to_rotation_matrix(angle_axis)
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3

        with pytest.warns(UserWarning):
            quaternion_hat = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                rot_m, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_qam(self, axis, device, dtype, atol, rtol):
        array = [[0.0, 0.0, 0.0, 0.0]]
        array[0][1 + axis] = 1.0
        quaternion = torch.tensor(array, device=device, dtype=dtype)
        assert quaternion.shape[-1] == 4

        angle_axis = kornia.geometry.conversions.quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert angle_axis.shape[-1] == 3

        rot_m = kornia.geometry.conversions.angle_axis_to_rotation_matrix(angle_axis)
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3

        quaternion_hat = kornia.geometry.conversions.rotation_matrix_to_quaternion(
            rot_m, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(quaternion_hat, quaternion, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_amq_xyzw(self, axis, device, dtype, atol, rtol):
        array = [[0.0, 0.0, 0.0]]
        array[0][axis] = kornia.pi / 2.0
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        assert angle_axis.shape[-1] == 3

        rot_m = kornia.geometry.conversions.angle_axis_to_rotation_matrix(angle_axis)
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3

        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                rot_m, order=QuaternionCoeffOrder.XYZW
            )
        assert quaternion.shape[-1] == 4

        with pytest.warns(UserWarning):
            angle_axis_hat = kornia.geometry.conversions.quaternion_to_angle_axis(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert_close(angle_axis_hat, angle_axis, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_amq(self, axis, device, dtype, atol, rtol):
        array = [[0.0, 0.0, 0.0]]
        array[0][axis] = kornia.pi / 2.0
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        assert angle_axis.shape[-1] == 3

        rot_m = kornia.geometry.conversions.angle_axis_to_rotation_matrix(angle_axis)
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3

        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot_m, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion.shape[-1] == 4

        angle_axis_hat = kornia.geometry.conversions.quaternion_to_angle_axis(
            quaternion, order=QuaternionCoeffOrder.WXYZ
        )
        assert_close(angle_axis_hat, angle_axis, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_aqm_xyzw(self, axis, device, dtype, atol, rtol):
        array = [[0.0, 0.0, 0.0]]
        array[0][axis] = kornia.pi / 2.0
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        assert angle_axis.shape[-1] == 3

        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(
                angle_axis, order=QuaternionCoeffOrder.XYZW
            )
        assert quaternion.shape[-1] == 4

        with pytest.warns(UserWarning):
            rot_m = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                quaternion, order=QuaternionCoeffOrder.XYZW
            )
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3

        angle_axis_hat = kornia.geometry.conversions.rotation_matrix_to_angle_axis(rot_m)
        assert_close(angle_axis_hat, angle_axis, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_triplet_aqm(self, axis, device, dtype, atol, rtol):
        array = [[0.0, 0.0, 0.0]]
        array[0][axis] = kornia.pi / 2.0
        angle_axis = torch.tensor(array, device=device, dtype=dtype)
        assert angle_axis.shape[-1] == 3

        quaternion = kornia.geometry.conversions.angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        assert quaternion.shape[-1] == 4

        rot_m = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3

        angle_axis_hat = kornia.geometry.conversions.rotation_matrix_to_angle_axis(rot_m)
        assert_close(angle_axis_hat, angle_axis, atol=atol, rtol=rtol)


class TestAngleOfRotations:
    """
    See: https://arxiv.org/pdf/1711.02508.pdf
    """

    @staticmethod
    def matrix_angle_abs(mx: torch.Tensor):
        """Unsigned rotation matrix angle."""
        trace = torch.diagonal(mx[..., :3, :3], dim1=-1, dim2=-2).sum(-1, keepdim=True)
        return torch.acos((trace - 1.0) / 2.0)

    @staticmethod
    def axis_and_angle_to_rotation_matrix(axis_name: str, angle: torch.Tensor, device, dtype):
        """See also: https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations."""
        axis_name = axis_name.lower()
        assert axis_name in ('x', 'y', 'z')
        sn = torch.sin(angle)
        cs = torch.cos(angle)
        ones = torch.ones_like(sn)
        zeros = torch.zeros_like(sn)
        if axis_name == 'x':
            axis = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype).repeat(angle.size())
            rot_m = torch.stack((ones, zeros, zeros, zeros, cs, -sn, zeros, sn, cs), dim=2).view(-1, 3, 3)
        elif axis_name == 'y':
            axis = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype).repeat(angle.size())
            rot_m = torch.stack((cs, zeros, sn, zeros, ones, zeros, -sn, zeros, cs), dim=2).view(-1, 3, 3)
        elif axis_name == 'z':
            axis = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype).repeat(angle.size())
            rot_m = torch.stack((cs, -sn, zeros, sn, cs, zeros, zeros, zeros, ones), dim=2).view(-1, 3, 3)
        else:
            raise NotImplementedError(f'Not prepared for axis with name {axis_name}')

        return rot_m, axis

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    def test_axis_angle_to_rotation_matrix(self, axis_name, device, dtype, atol, rtol):
        # Random angle in [-pi..pi]
        angle = torch.tensor((np.random.random(size=(2, 1)) * 2.0 * np.pi - np.pi), device=device, dtype=dtype)
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(
            axis_name=axis_name, angle=angle, device=device, dtype=dtype
        )
        assert rot_m.dim() == 3
        assert rot_m.shape[-1] == 3
        assert rot_m.shape[-2] == 3
        assert rot_m.shape[-3] == angle.numel()
        assert axis.shape[-1] == 3
        assert axis.shape[-2] == angle.numel()

        # Make sure the returned axis matches the named one, and the appropriate column
        if axis_name == 'x':
            assert_close(axis, torch.tensor(((1.0, 0.0, 0.0),) * angle.numel(), device=device, dtype=dtype))
            assert_close(axis, rot_m[..., :3, 0])
        elif axis_name == 'y':
            assert_close(axis, torch.tensor(((0.0, 1.0, 0.0),) * angle.numel(), device=device, dtype=dtype))
            assert_close(axis, rot_m[..., :3, 1])
        elif axis_name == 'z':
            assert_close(axis, torch.tensor(((0.0, 0.0, 1.0),) * angle.numel(), device=device, dtype=dtype))
            assert_close(axis, rot_m[..., :3, 2])
        else:
            raise NotImplementedError(f'Not prepared for axis_name {axis_name}')

        # Make sure axes are perpendicular
        zero = torch.zeros_like(angle).unsqueeze(-1)
        assert_close(rot_m[..., :3, 1:2].permute((0, 2, 1)) @ rot_m[..., :3, 0:1], zero, atol=atol, rtol=rtol)
        assert_close(rot_m[..., :3, 2:3].permute((0, 2, 1)) @ rot_m[..., :3, 1:2], zero, atol=atol, rtol=rtol)
        assert_close(rot_m[..., :3, 2:3].permute((0, 2, 1)) @ rot_m[..., :3, 0:1], zero, atol=atol, rtol=rtol)

        # Make sure axes are unit norm
        one = torch.ones_like(angle)
        assert_close(rot_m[..., :3, 0].norm(p=2, dim=-1, keepdim=True), one, atol=atol, rtol=rtol)
        assert_close(rot_m[..., :3, 1].norm(p=2, dim=-1, keepdim=True), one, atol=atol, rtol=rtol)
        assert_close(rot_m[..., :3, 2].norm(p=2, dim=-1, keepdim=True), one, atol=atol, rtol=rtol)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -135.0, -90.0, -45.0, 0.0, 45, 90, 135, 179.9))
    def test_matrix_angle(self, axis_name, angle_deg, device, dtype):
        angle = (angle_deg * kornia.pi / 180.0).to(dtype).to(device).view(1, 1)
        rot_m, _ = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(
            axis_name=axis_name, angle=angle, device=device, dtype=dtype
        )
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_close(torch.abs(angle), matrix_angle_abs)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -90.0, -45.0, 0.0, 45, 90, 179.9))
    def test_quaternion_xyzw(self, axis_name, angle_deg, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        angle = torch.tensor((angle_deg * kornia.pi / 180.0,), device=device, dtype=dtype).repeat(2, 1)
        pi = torch.ones_like(angle) * kornia.pi
        assert 2 <= len(angle.shape)
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(
            axis_name=axis_name, angle=angle, device=device, dtype=dtype
        )
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                rot_m, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        # compute quaternion rotation angle
        # See Section 2.4.4 Equation (105a) in https://arxiv.org/pdf/1711.02508.pdf
        angle_hat = 2.0 * torch.atan2(quaternion[..., :3].norm(p=2, dim=-1, keepdim=True), quaternion[..., 3:4])
        # make sure it lands between [-pi..pi)
        mask = pi < angle_hat
        while torch.any(mask):
            angle_hat = torch.where(mask, angle_hat - 2.0 * kornia.pi, angle_hat)
            mask = pi < angle_hat
        # invert angle, if quaternion axis points in the opposite direction of the original axis
        dots = (quaternion[..., :3] * axis).sum(dim=-1, keepdim=True)
        angle_hat = torch.where(dots < 0.0, angle_hat * -1.0, angle_hat)
        # quaternion angle should match input angle
        assert_close(angle_hat, angle, atol=atol, rtol=rtol)
        # magnitude of angle should match matrix rotation angle
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_close(torch.abs(angle_hat), matrix_angle_abs, atol=atol, rtol=rtol)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -90.0, -45.0, 0.0, 45, 90, 179.9))
    def test_quaternion(self, axis_name, angle_deg, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        angle = torch.tensor((angle_deg * kornia.pi / 180.0,), device=device, dtype=dtype).repeat(2, 1)
        pi = torch.ones_like(angle) * kornia.pi
        assert 2 <= len(angle.shape)
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(
            axis_name=axis_name, angle=angle, device=device, dtype=dtype
        )
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
            rot_m, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        # compute quaternion rotation angle
        # See Section 2.4.4 Equation (105a) in https://arxiv.org/pdf/1711.02508.pdf
        angle_hat = 2.0 * torch.atan2(quaternion[..., 1:4].norm(p=2, dim=-1, keepdim=True), quaternion[..., 0:1])
        # make sure it lands between [-pi..pi)
        mask = pi < angle_hat
        while torch.any(mask):
            angle_hat = torch.where(mask, angle_hat - 2.0 * kornia.pi, angle_hat)
            mask = pi < angle_hat
        # invert angle, if quaternion axis points in the opposite direction of the original axis
        dots = (quaternion[..., 1:4] * axis).sum(dim=-1, keepdim=True)
        angle_hat = torch.where(dots < 0.0, angle_hat * -1.0, angle_hat)
        # quaternion angle should match input angle
        assert_close(angle_hat, angle, atol=atol, rtol=rtol)
        # magnitude of angle should match matrix rotation angle
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_close(torch.abs(angle_hat), matrix_angle_abs, atol=atol, rtol=rtol)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -90.0, -45.0, 0, 45, 90, 179.9))
    def test_angle_axis(self, axis_name, angle_deg, device, dtype, atol, rtol):
        angle = (angle_deg * kornia.pi / 180.0).to(dtype).to(device).repeat(2, 1)
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(
            axis_name=axis_name, angle=angle, device=device, dtype=dtype
        )
        angle_axis = kornia.geometry.conversions.rotation_matrix_to_angle_axis(rot_m)
        # compute angle_axis rotation angle
        angle_hat = angle_axis.norm(p=2, dim=-1, keepdim=True)
        # invert angle, if angle_axis axis points in the opposite direction of the original axis
        dots = (angle_axis * axis).sum(dim=-1, keepdim=True)
        angle_hat = torch.where(dots < 0.0, angle_hat * -1.0, angle_hat)
        # angle_axis angle should match input angle
        assert_close(angle_hat, angle, atol=atol, rtol=rtol)
        # magnitude of angle should match matrix rotation angle
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_close(torch.abs(angle_hat), matrix_angle_abs, atol=atol, rtol=rtol)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -90.0, -45.0, 0, 45, 90, 179.9))
    def test_log_quaternion_xyzw(self, axis_name, angle_deg, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        angle = (angle_deg * kornia.pi / 180.0).to(dtype).to(device).repeat(2, 1)
        pi = torch.ones_like(angle) * kornia.pi
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(
            axis_name=axis_name, angle=angle, device=device, dtype=dtype
        )
        with pytest.warns(UserWarning):
            quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
                rot_m, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        with pytest.warns(UserWarning):
            log_q = kornia.geometry.conversions.quaternion_exp_to_log(
                quaternion, eps=eps, order=QuaternionCoeffOrder.XYZW
            )
        # compute angle_axis rotation angle
        angle_hat = 2.0 * log_q.norm(p=2, dim=-1, keepdim=True)
        # make sure it lands between [-pi..pi)
        mask = pi < angle_hat
        while torch.any(mask):
            angle_hat = torch.where(mask, angle_hat - 2.0 * kornia.pi, angle_hat)
            mask = pi < angle_hat
        # invert angle, if angle_axis axis points in the opposite direction of the original axis
        dots = (log_q * axis).sum(dim=-1, keepdim=True)
        angle_hat = torch.where(dots < 0.0, angle_hat * -1.0, angle_hat)
        # angle_axis angle should match input angle
        assert_close(angle_hat, angle, atol=atol, rtol=rtol)
        # magnitude of angle should match matrix rotation angle
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_close(torch.abs(angle_hat), matrix_angle_abs, atol=atol, rtol=rtol)

    @pytest.mark.parametrize('axis_name', ('x', 'y', 'z'))
    @pytest.mark.parametrize("angle_deg", (-179.9, -90.0, -45.0, 0, 45, 90, 179.9))
    def test_log_quaternion(self, axis_name, angle_deg, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        angle = (angle_deg * kornia.pi / 180.0).to(dtype).to(device).repeat(2, 1)
        pi = torch.ones_like(angle) * kornia.pi
        rot_m, axis = TestAngleOfRotations.axis_and_angle_to_rotation_matrix(
            axis_name=axis_name, angle=angle, device=device, dtype=dtype
        )
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
            rot_m, eps=eps, order=QuaternionCoeffOrder.WXYZ
        )
        log_q = kornia.geometry.conversions.quaternion_exp_to_log(quaternion, eps=eps, order=QuaternionCoeffOrder.WXYZ)
        # compute angle_axis rotation angle
        angle_hat = 2.0 * log_q.norm(p=2, dim=-1, keepdim=True)
        # make sure it lands between [-pi..pi)
        mask = pi < angle_hat
        while torch.any(mask):
            angle_hat = torch.where(mask, angle_hat - 2.0 * kornia.pi, angle_hat)
            mask = pi < angle_hat
        # invert angle, if angle_axis axis points in the opposite direction of the original axis
        dots = (log_q * axis).sum(dim=-1, keepdim=True)
        angle_hat = torch.where(dots < 0.0, angle_hat * -1.0, angle_hat)
        # angle_axis angle should match input angle
        assert_close(angle_hat, angle, atol=atol, rtol=rtol)
        # magnitude of angle should match matrix rotation angle
        matrix_angle_abs = TestAngleOfRotations.matrix_angle_abs(rot_m)
        assert_close(torch.abs(angle_hat), matrix_angle_abs, atol=atol, rtol=rtol)

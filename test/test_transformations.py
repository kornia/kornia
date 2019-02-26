import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utilities
from common import TEST_DEVICES


def identity_matrix(batch_size):
    r"""Creates a batched homogeneous identity matrix"""
    return torch.eye(4).repeat(batch_size, 1, 1)  # Nx4x4


def euler_angles_to_rotation_matrix(x, y, z):
    r"""Create a rotation matrix from x, y, z angles"""
    assert x.dim() == 1, x.shape
    assert x.shape == y.shape == z.shape
    ones, zeros = torch.ones_like(x), torch.zeros_like(x)
    # the rotation matrix for the x-axis
    rx_tmp = [
        ones, zeros, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x), zeros,
        zeros, torch.sin(x), torch.cos(x), zeros,
        zeros, zeros, zeros, ones]
    rx = torch.stack(rx_tmp, dim=-1).view(-1, 4, 4)
    # the rotation matrix for the y-axis
    ry_tmp = [
        torch.cos(y), zeros, torch.sin(y), zeros,
        zeros, ones, zeros, zeros,
        -torch.sin(y), zeros, torch.cos(y), zeros,
        zeros, zeros, zeros, ones]
    ry = torch.stack(ry_tmp, dim=-1).view(-1, 4, 4)
    # the rotation matrix for the z-axis
    rz_tmp = [
        torch.cos(z), -torch.sin(z), zeros, zeros,
        torch.sin(z), torch.cos(z), zeros, zeros,
        zeros, zeros, ones, zeros,
        zeros, zeros, zeros, ones]
    rz = torch.stack(rz_tmp, dim=-1).view(-1, 4, 4)
    return torch.matmul(rz, torch.matmul(ry, rx))  # Bx4x4


class TestComposeTransforms:

    def test_translation_4x4(self):
        offset = 10
        trans_01 = identity_matrix(batch_size=1)[0]
        trans_12 = identity_matrix(batch_size=1)[0]
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = tgm.compose_transformations(trans_01, trans_12)
        assert utils.check_equal_torch(trans_02, trans_12)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, batch_size):
        offset = 10
        trans_01 = identity_matrix(batch_size)
        trans_12 = identity_matrix(batch_size)
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = tgm.compose_transformations(trans_01, trans_12)
        assert utils.check_equal_torch(trans_02, trans_12)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size):
        trans_01 = identity_matrix(batch_size)
        trans_12 = identity_matrix(batch_size)

        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        trans_12 = utils.tensor_to_gradcheck_var(trans_12)  # to var
        assert gradcheck(tgm.compose_transformations, (trans_01, trans_12,),
                         raise_exception=True)


class TestInverseTransformation:

    def test_translation_4x4(self):
        offset = 10
        trans_01 = identity_matrix(batch_size=1)[0]
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = tgm.inverse_transformation(trans_01)
        trans_01_hat = tgm.inverse_transformation(trans_10)
        assert utils.check_equal_torch(trans_01, trans_01_hat)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, batch_size):
        offset = 10
        trans_01 = identity_matrix(batch_size)
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = tgm.inverse_transformation(trans_01)
        trans_01_hat = tgm.inverse_transformation(trans_10)
        assert utils.check_equal_torch(trans_01, trans_01_hat)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_rotation_translation_Bx4x4(self, batch_size):
        offset = 10
        x, y, z = 0, 0, tgm.pi
        ones = torch.ones(batch_size)
        rmat_01 = euler_angles_to_rotation_matrix(x * ones, y * ones, z * ones)

        trans_01 = identity_matrix(batch_size)
        trans_01[..., :3, -1] += offset  # add offset to translation vector
        trans_01[..., :3, :3] = rmat_01[..., :3, :3]

        trans_10 = tgm.inverse_transformation(trans_01)
        trans_01_hat = tgm.inverse_transformation(trans_10)
        assert utils.check_equal_torch(trans_01, trans_01_hat)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size):
        trans_01 = identity_matrix(batch_size)
        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        assert gradcheck(tgm.inverse_transformation, (trans_01,),
                         raise_exception=True)


class TestRelativeTransformation:

    def test_translation_4x4(self):
        offset = 10.
        trans_01 = identity_matrix(batch_size=1)[0]
        trans_02 = identity_matrix(batch_size=1)[0]
        trans_02[..., :3, -1] += offset  # add offset to translation vector

        trans_12 = tgm.relative_transformation(trans_01, trans_02)
        trans_02_hat = tgm.compose_transformations(trans_01, trans_12)
        assert utils.check_equal_torch(trans_02_hat, trans_02)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_rotation_translation_Bx4x4(self, batch_size):
        offset = 10.
        x, y, z = 0., 0., tgm.pi
        ones = torch.ones(batch_size)
        rmat_02 = euler_angles_to_rotation_matrix(x * ones, y * ones, z * ones)

        trans_01 = identity_matrix(batch_size)
        trans_02 = identity_matrix(batch_size)
        trans_02[..., :3, -1] += offset  # add offset to translation vector
        trans_02[..., :3, :3] = rmat_02[..., :3, :3]

        trans_12 = tgm.relative_transformation(trans_01, trans_02)
        trans_02_hat = tgm.compose_transformations(trans_01, trans_12)
        assert utils.check_equal_torch(trans_02_hat, trans_02)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size):
        trans_01 = identity_matrix(batch_size)
        trans_02 = identity_matrix(batch_size)

        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        trans_02 = utils.tensor_to_gradcheck_var(trans_02)  # to var
        assert gradcheck(tgm.relative_transformation, (trans_01, trans_02,),
                         raise_exception=True)

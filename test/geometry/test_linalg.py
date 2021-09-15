import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


def identity_matrix(batch_size, device, dtype):
    r"""Create a batched homogeneous identity matrix"""
    return torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)  # Nx4x4


def euler_angles_to_rotation_matrix(x, y, z):
    r"""Create a rotation matrix from x, y, z angles"""
    assert x.dim() == 1, x.shape
    assert x.shape == y.shape == z.shape
    ones, zeros = torch.ones_like(x), torch.zeros_like(x)
    # the rotation matrix for the x-axis
    rx_tmp = [
        ones,
        zeros,
        zeros,
        zeros,
        zeros,
        torch.cos(x),
        -torch.sin(x),
        zeros,
        zeros,
        torch.sin(x),
        torch.cos(x),
        zeros,
        zeros,
        zeros,
        zeros,
        ones,
    ]
    rx = torch.stack(rx_tmp, dim=-1).view(-1, 4, 4)
    # the rotation matrix for the y-axis
    ry_tmp = [
        torch.cos(y),
        zeros,
        torch.sin(y),
        zeros,
        zeros,
        ones,
        zeros,
        zeros,
        -torch.sin(y),
        zeros,
        torch.cos(y),
        zeros,
        zeros,
        zeros,
        zeros,
        ones,
    ]
    ry = torch.stack(ry_tmp, dim=-1).view(-1, 4, 4)
    # the rotation matrix for the z-axis
    rz_tmp = [
        torch.cos(z),
        -torch.sin(z),
        zeros,
        zeros,
        torch.sin(z),
        torch.cos(z),
        zeros,
        zeros,
        zeros,
        zeros,
        ones,
        zeros,
        zeros,
        zeros,
        zeros,
        ones,
    ]
    rz = torch.stack(rz_tmp, dim=-1).view(-1, 4, 4)
    return torch.matmul(rz, torch.matmul(ry, rx))  # Bx4x4


class TestTransformPoints:
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_points", [2, 3, 5])
    @pytest.mark.parametrize("num_dims", [2, 3])
    def test_transform_points(self, batch_size, num_points, num_dims, device, dtype):
        # generate input data
        eye_size = num_dims + 1
        points_src = torch.rand(batch_size, num_points, num_dims, device=device, dtype=dtype)

        dst_homo_src = utils.create_random_homography(batch_size, eye_size).to(device=device, dtype=dtype)
        dst_homo_src = dst_homo_src.to(device)

        # transform the points from dst to ref
        points_dst = kornia.transform_points(dst_homo_src, points_src)

        # transform the points from ref to dst
        src_homo_dst = torch.inverse(dst_homo_src)
        points_dst_to_src = kornia.transform_points(src_homo_dst, points_dst)

        # projected should be equal as initial
        assert_close(points_src, points_dst_to_src, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        eye_size = num_dims + 1
        points_src = torch.rand(batch_size, num_points, num_dims, device=device, dtype=dtype)
        dst_homo_src = utils.create_random_homography(batch_size, eye_size).to(device=device, dtype=dtype)
        # evaluate function gradient
        points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
        dst_homo_src = utils.tensor_to_gradcheck_var(dst_homo_src)  # to var
        assert gradcheck(kornia.geometry.transform_points, (dst_homo_src, points_src), raise_exception=True)

    def test_jit(self, device, dtype):
        points = torch.ones(1, 2, 2, device=device, dtype=dtype)
        transform = kornia.eye_like(3, points)
        op = kornia.geometry.transform_points
        op_script = torch.jit.script(op)
        actual = op_script(transform, points)
        expected = op(transform, points)
        assert_close(actual, expected, atol=1e-4, rtol=1e-4)


class TestComposeTransforms:
    def test_translation_4x4(self, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_12 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = kornia.compose_transformations(trans_01, trans_12)
        assert_close(trans_02, trans_12, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_12 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = kornia.compose_transformations(trans_01, trans_12)
        assert_close(trans_02, trans_12, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size, device, dtype):
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_12 = identity_matrix(batch_size, device=device, dtype=dtype)

        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        trans_12 = utils.tensor_to_gradcheck_var(trans_12)  # to var
        assert gradcheck(kornia.compose_transformations, (trans_01, trans_12), raise_exception=True)


class TestInverseTransformation:
    def test_translation_4x4(self, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = kornia.inverse_transformation(trans_01)
        trans_01_hat = kornia.inverse_transformation(trans_10)
        assert_close(trans_01, trans_01_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = kornia.inverse_transformation(trans_01)
        trans_01_hat = kornia.inverse_transformation(trans_10)
        assert_close(trans_01, trans_01_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_rotation_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10
        x, y, z = 0, 0, kornia.pi
        ones = torch.ones(batch_size, device=device, dtype=dtype)
        rmat_01 = euler_angles_to_rotation_matrix(x * ones, y * ones, z * ones)

        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_01[..., :3, -1] += offset  # add offset to translation vector
        trans_01[..., :3, :3] = rmat_01[..., :3, :3]

        trans_10 = kornia.inverse_transformation(trans_01)
        trans_01_hat = kornia.inverse_transformation(trans_10)
        assert_close(trans_01, trans_01_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size, device, dtype):
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        assert gradcheck(kornia.inverse_transformation, (trans_01,), raise_exception=True)


class TestRelativeTransformation:
    def test_translation_4x4(self, device, dtype):
        offset = 10.0
        trans_01 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_02 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_02[..., :3, -1] += offset  # add offset to translation vector

        trans_12 = kornia.relative_transformation(trans_01, trans_02)
        trans_02_hat = kornia.compose_transformations(trans_01, trans_12)
        assert_close(trans_02_hat, trans_02, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_rotation_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10.0
        x, y, z = 0.0, 0.0, kornia.pi
        ones = torch.ones(batch_size, device=device, dtype=dtype)
        rmat_02 = euler_angles_to_rotation_matrix(x * ones, y * ones, z * ones)

        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_02 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_02[..., :3, -1] += offset  # add offset to translation vector
        trans_02[..., :3, :3] = rmat_02[..., :3, :3]

        trans_12 = kornia.relative_transformation(trans_01, trans_02)
        trans_02_hat = kornia.compose_transformations(trans_01, trans_12)
        assert_close(trans_02_hat, trans_02, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size, device, dtype):
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_02 = identity_matrix(batch_size, device=device, dtype=dtype)

        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        trans_02 = utils.tensor_to_gradcheck_var(trans_02)  # to var
        assert gradcheck(kornia.relative_transformation, (trans_01, trans_02), raise_exception=True)


class TestTransformLAFs:
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_points", [2, 3, 5])
    def test_transform_points(self, batch_size, num_points, device, dtype):
        # generate input data
        eye_size = 3
        lafs_src = torch.rand(batch_size, num_points, 2, 3, device=device, dtype=dtype)

        dst_homo_src = utils.create_random_homography(batch_size, eye_size).to(device=device, dtype=dtype)

        # transform the points from dst to ref
        lafs_dst = kornia.perspective_transform_lafs(dst_homo_src, lafs_src)

        # transform the points from ref to dst
        src_homo_dst = torch.inverse(dst_homo_src)
        lafs_dst_to_src = kornia.perspective_transform_lafs(src_homo_dst, lafs_dst)

        # projected should be equal as initial
        assert_close(lafs_src, lafs_dst_to_src)

    def test_gradcheck(self, device, dtype):
        # generate input data
        batch_size, num_points = 2, 3
        eye_size = 3
        points_src = torch.rand(batch_size, num_points, 2, 3, device=device, dtype=dtype)
        dst_homo_src = utils.create_random_homography(batch_size, eye_size).to(device=device, dtype=dtype)
        # evaluate function gradient
        points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
        dst_homo_src = utils.tensor_to_gradcheck_var(dst_homo_src)  # to var
        assert gradcheck(kornia.perspective_transform_lafs, (dst_homo_src, points_src), raise_exception=True)

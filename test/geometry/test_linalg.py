from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.geometry.linalg as kgl
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


def identity_matrix(batch_size, device, dtype):
    r"""Create a batched homogeneous identity matrix."""
    return torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)  # Nx4x4


def euler_angles_to_rotation_matrix(x, y, z):
    r"""Create a rotation matrix from x, y, z angles."""
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
        points_dst = kgl.transform_points(dst_homo_src, points_src)

        # transform the points from ref to dst
        src_homo_dst = torch.inverse(dst_homo_src)
        points_dst_to_src = kgl.transform_points(src_homo_dst, points_dst)

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
    def test_smoke(self, device, dtype):
        batch_size = 2
        trans_01 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)
        trans_12 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)

        to_check_1 = kornia.geometry.compose_transformations(trans_01, trans_12)
        to_check_2 = kornia.geometry.compose_transformations(trans_01[0], trans_12[0])

        assert to_check_1.shape == (batch_size, 4, 4)
        assert to_check_2.shape == (4, 4)

    def test_exception(self, device, dtype):
        to_check_1 = torch.rand((7, 4, 4, 3), device=device, dtype=dtype)
        to_check_2 = torch.rand((5, 10, 10), device=device, dtype=dtype)
        to_check_3 = torch.rand((6, 4, 4), device=device, dtype=dtype)
        to_check_4 = torch.rand((4, 4), device=device, dtype=dtype)
        to_check_5 = torch.rand((3, 3), device=device, dtype=dtype)

        # Testing if exception is thrown when both inputs have shape (3, 3)
        with pytest.raises(ValueError):
            _ = kornia.geometry.compose_transformations(to_check_5, to_check_5)

        # Testing if exception is thrown when both inputs have shape (5, 10, 10)
        with pytest.raises(ValueError):
            _ = kornia.geometry.compose_transformations(to_check_2, to_check_2)

        # Testing if exception is thrown when one input has shape (6, 4, 4)
        # whereas the other input has shape (4, 4)
        with pytest.raises(ValueError):
            _ = kornia.geometry.compose_transformations(to_check_3, to_check_4)

        # Testing if exception is thrown when one input has shape (7, 4, 4, 3)
        # whereas the other input has shape (4, 4)
        with pytest.raises(ValueError):
            _ = kornia.geometry.compose_transformations(to_check_1, to_check_4)

    def test_translation_4x4(self, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_12 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = kgl.compose_transformations(trans_01, trans_12)
        assert_close(trans_02, trans_12, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_12 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = kgl.compose_transformations(trans_01, trans_12)
        assert_close(trans_02, trans_12, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size, device, dtype):
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_12 = identity_matrix(batch_size, device=device, dtype=dtype)

        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        trans_12 = utils.tensor_to_gradcheck_var(trans_12)  # to var
        assert gradcheck(kgl.compose_transformations, (trans_01, trans_12), raise_exception=True)


class TestInverseTransformation:
    def test_smoke(self, device, dtype):
        batch_size = 2
        trans_01 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)

        to_check_1 = kornia.geometry.inverse_transformation(trans_01)
        to_check_2 = kornia.geometry.inverse_transformation(trans_01[0])

        assert to_check_1.shape == (batch_size, 4, 4)
        assert to_check_2.shape == (4, 4)

    def test_exception(self, device, dtype):
        to_check_1 = torch.rand((7, 4, 4, 3), device=device, dtype=dtype)
        to_check_2 = torch.rand((5, 10, 10), device=device, dtype=dtype)
        to_check_3 = torch.rand((3, 3), device=device, dtype=dtype)

        # Testing if exception is thrown when the input has shape (7, 4, 4, 3)
        with pytest.raises(ValueError):
            _ = kornia.geometry.inverse_transformation(to_check_1)

        # Testing if exception is thrown when the input has shape (5, 10, 10)
        with pytest.raises(ValueError):
            _ = kornia.geometry.inverse_transformation(to_check_2)

        # Testing if exception is thrown when the input has shape (3, 3)
        with pytest.raises(ValueError):
            _ = kornia.geometry.inverse_transformation(to_check_3)

    def test_translation_4x4(self, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = kgl.inverse_transformation(trans_01)
        trans_01_hat = kgl.inverse_transformation(trans_10)
        assert_close(trans_01, trans_01_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = kgl.inverse_transformation(trans_01)
        trans_01_hat = kgl.inverse_transformation(trans_10)
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

        trans_10 = kgl.inverse_transformation(trans_01)
        trans_01_hat = kgl.inverse_transformation(trans_10)
        assert_close(trans_01, trans_01_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size, device, dtype):
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        assert gradcheck(kgl.inverse_transformation, (trans_01,), raise_exception=True)


class TestRelativeTransformation:
    def test_smoke(self, device, dtype):
        batch_size = 2
        trans_01 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)
        trans_02 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)

        to_check_1 = kornia.geometry.relative_transformation(trans_01, trans_02)
        to_check_2 = kornia.geometry.relative_transformation(trans_01[0], trans_02[0])

        assert to_check_1.shape == (batch_size, 4, 4)
        assert to_check_2.shape == (4, 4)

    def test_exception(self, device, dtype):
        to_check_1 = torch.rand((7, 4, 4, 3), device=device, dtype=dtype)
        to_check_2 = torch.rand((5, 10, 10), device=device, dtype=dtype)
        to_check_3 = torch.rand((6, 4, 4), device=device, dtype=dtype)
        to_check_4 = torch.rand((4, 4), device=device, dtype=dtype)
        to_check_5 = torch.rand((3, 3), device=device, dtype=dtype)

        # Testing if exception is thrown when both inputs have shape (3, 3)
        with pytest.raises(ValueError):
            _ = kornia.geometry.relative_transformation(to_check_5, to_check_5)

        # Testing if exception is thrown when both inputs have shape (5, 10, 10)
        with pytest.raises(ValueError):
            _ = kornia.geometry.relative_transformation(to_check_2, to_check_2)

        # Testing if exception is thrown when one input has shape (6, 4, 4)
        # whereas the other input has shape (4, 4)
        with pytest.raises(ValueError):
            _ = kornia.geometry.relative_transformation(to_check_3, to_check_4)

        # Testing if exception is thrown when one input has shape (7, 4, 4, 3)
        # whereas the other input has shape (4, 4)
        with pytest.raises(ValueError):
            _ = kornia.geometry.relative_transformation(to_check_1, to_check_4)

    def test_translation_4x4(self, device, dtype):
        offset = 10.0
        trans_01 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_02 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_02[..., :3, -1] += offset  # add offset to translation vector

        trans_12 = kgl.relative_transformation(trans_01, trans_02)
        trans_02_hat = kgl.compose_transformations(trans_01, trans_12)
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

        trans_12 = kgl.relative_transformation(trans_01, trans_02)
        trans_02_hat = kgl.compose_transformations(trans_01, trans_12)
        assert_close(trans_02_hat, trans_02, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size, device, dtype):
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_02 = identity_matrix(batch_size, device=device, dtype=dtype)

        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        trans_02 = utils.tensor_to_gradcheck_var(trans_02)  # to var
        assert gradcheck(kgl.relative_transformation, (trans_01, trans_02), raise_exception=True)


class TestPointsLinesDistances:
    def test_smoke(self, device, dtype):
        pts = torch.rand(1, 1, 2, device=device, dtype=dtype)
        lines = torch.rand(1, 1, 3, device=device, dtype=dtype)
        distances = kgl.point_line_distance(pts, lines)
        assert distances.shape == (1, 1)

        # homogeneous
        pts = torch.rand(1, 1, 3, device=device, dtype=dtype)
        lines = torch.rand(1, 1, 3, device=device, dtype=dtype)
        distances = kgl.point_line_distance(pts, lines)
        assert distances.shape == (1, 1)

    @pytest.mark.parametrize(
        "batch_size, sample_size", [(1, 1), (2, 1), (4, 1), (7, 1), (1, 3), (2, 3), (4, 3), (7, 3)]
    )
    def test_shape(self, batch_size, sample_size, device, dtype):
        B, N = batch_size, sample_size
        pts = torch.rand(B, N, 2, device=device, dtype=dtype)
        lines = torch.rand(B, N, 3, device=device, dtype=dtype)
        distances = kgl.point_line_distance(pts, lines)
        assert distances.shape == (B, N)

    @pytest.mark.parametrize(
        "batch_size, extra_dim_size", [(1, 1), (2, 1), (4, 1), (7, 1), (1, 3), (2, 3), (4, 3), (7, 3)]
    )
    def test_shapes(self, batch_size, extra_dim_size, device, dtype):
        B, T, N = batch_size, extra_dim_size, 3
        pts = torch.rand(B, T, N, 2, device=device, dtype=dtype)
        lines = torch.rand(B, T, N, 3, device=device, dtype=dtype)
        distances = kgl.point_line_distance(pts, lines)
        assert distances.shape == (B, T, N)

    def test_functional(self, device):
        pts = torch.tensor([1.0, 0], device=device, dtype=torch.float64).view(1, 1, 2).tile(1, 6, 1)
        lines = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            device=device,
            dtype=torch.float64,
        ).view(1, 6, 3)
        distances = kgl.point_line_distance(pts, lines)
        distances_expected = torch.tensor(
            [
                0.0,
                1.0,
                1.0,
                2.0,
                torch.sqrt(torch.tensor(2, dtype=torch.float64)) / 2,
                torch.sqrt(torch.tensor(2, dtype=torch.float64)),
            ],
            device=device,
        ).view(1, 6)
        assert_close(distances, distances_expected, rtol=1e-6, atol=1e-6)

    def test_gradcheck(self, device):
        pts = torch.rand(2, 3, 2, device=device, requires_grad=True, dtype=torch.float64)
        lines = torch.rand(2, 3, 3, device=device, requires_grad=True, dtype=torch.float64)
        assert gradcheck(kgl.point_line_distance, (pts, lines), raise_exception=True)

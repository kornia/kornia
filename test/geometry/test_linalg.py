import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


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


class TestTransformPoints:

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_points", [2, 3, 5])
    @pytest.mark.parametrize("num_dims", [2, 3])
    def test_transform_points(
            self, device, batch_size, num_points, num_dims):
        # generate input data
        eye_size = num_dims + 1
        points_src = torch.rand(batch_size, num_points, num_dims)
        points_src = points_src.to(device)

        dst_homo_src = utils.create_random_homography(batch_size, eye_size)
        dst_homo_src = dst_homo_src.to(device)

        # transform the points from dst to ref
        points_dst = kornia.transform_points(dst_homo_src, points_src)

        # transform the points from ref to dst
        src_homo_dst = torch.inverse(dst_homo_src)
        points_dst_to_src = kornia.transform_points(src_homo_dst, points_dst)

        # projected should be equal as initial
        assert_allclose(points_src, points_dst_to_src)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        eye_size = num_dims + 1
        points_src = torch.rand(batch_size, num_points, num_dims).to(device)
        dst_homo_src = utils.create_random_homography(batch_size, eye_size).to(device)
        # evaluate function gradient
        points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
        dst_homo_src = utils.tensor_to_gradcheck_var(dst_homo_src)  # to var
        assert gradcheck(kornia.transform_points, (dst_homo_src, points_src,),
                         raise_exception=True)

    def test_jit(self, device):
        @torch.jit.script
        def op_script(transform, points):
            return kornia.transform_points(transform, points)

        points = torch.ones(1, 2, 2, device=device)
        transform = kornia.eye_like(3, points)
        op = kornia.transform_points
        op_script = torch.jit.script(op)
        actual = op_script(transform, points)
        expected = op(transform, points)
        assert_allclose(actual, expected)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit_trace(self, device):
        @torch.jit.script
        def op_script(transform, points):
            return kornia.transform_points(transform, points)

        points = torch.ones(1, 2, 2).to(device)
        transform = torch.eye(3)[None].to(device)
        op_script_trace = torch.jit.trace(op_script, (transform, points,))
        actual = op_script_trace(transform, points)
        expected = kornia.transform_points(transform, points)

        assert_allclose(actual, expected)


class TestTransformBoxes:

    def test_transform_boxes(self, device):

        boxes = torch.tensor([[139.2640, 103.0150, 397.3120, 410.5225]]).to(device)

        expected = torch.tensor([372.7360, 103.0150, 114.6880, 410.5225]).to(device)

        trans_mat = torch.tensor([[[-1., 0., 512.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]]]).to(device)

        out = kornia.transform_boxes(trans_mat, boxes)
        assert_allclose(out, expected)

    def test_transform_multiple_boxes(self, device):

        boxes = torch.tensor([[139.2640, 103.0150, 397.3120, 410.5225],
                              [1.0240, 80.5547, 512.0000, 512.0000],
                              [165.2053, 262.1440, 510.6347, 508.9280],
                              [119.8080, 144.2067, 257.0240, 410.1292]]).to(device)

        boxes = boxes.repeat(2, 1, 1)  # 2 x 4 x 4 two images 4 boxes each

        expected = torch.tensor([[[372.7360, 103.0150, 114.6880, 410.5225],
                                  [510.9760, 80.5547, 0.0000, 512.0000],
                                  [346.7947, 262.1440, 1.3653, 508.9280],
                                  [392.1920, 144.2067, 254.9760, 410.1292]],

                                 [[139.2640, 103.0150, 397.3120, 410.5225],
                                  [1.0240, 80.5547, 512.0000, 512.0000],
                                  [165.2053, 262.1440, 510.6347, 508.9280],
                                  [119.8080, 144.2067, 257.0240, 410.1292]]]).to(device)

        trans_mat = torch.tensor([[[-1., 0., 512.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]],

                                  [[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]]]).to(device)

        out = kornia.transform_boxes(trans_mat, boxes)
        assert_allclose(out, expected)

    def test_transform_boxes_wh(self, device):

        boxes = torch.tensor([[139.2640, 103.0150, 258.0480, 307.5075],
                              [1.0240, 80.5547, 510.9760, 431.4453],
                              [165.2053, 262.1440, 345.4293, 246.7840],
                              [119.8080, 144.2067, 137.2160, 265.9225]]).to(device)

        expected = torch.tensor([[372.7360, 103.0150, -258.0480, 307.5075],
                                 [510.9760, 80.5547, -510.9760, 431.4453],
                                 [346.7947, 262.1440, -345.4293, 246.7840],
                                 [392.1920, 144.2067, -137.2160, 265.9225]]).to(device)

        trans_mat = torch.tensor([[[-1., 0., 512.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]]]).to(device)

        out = kornia.transform_boxes(trans_mat, boxes, mode='xywh')
        assert_allclose(out, expected)

    def test_gradcheck(self, device):

        boxes = torch.tensor([[139.2640, 103.0150, 258.0480, 307.5075],
                              [1.0240, 80.5547, 510.9760, 431.4453],
                              [165.2053, 262.1440, 345.4293, 246.7840],
                              [119.8080, 144.2067, 137.2160, 265.9225]]).to(device)

        trans_mat = torch.tensor([[[-1., 0., 512.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]]]).to(device)

        trans_mat = utils.tensor_to_gradcheck_var(trans_mat)
        boxes = utils.tensor_to_gradcheck_var(boxes)

        assert gradcheck(kornia.transform_boxes, (trans_mat, boxes), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(transform, boxes):
            return kornia.transform_boxes(transform, boxes)

        boxes = torch.tensor([139.2640, 103.0150, 258.0480, 307.5075]).to(device)

        trans_mat = torch.tensor([[[-1., 0., 512.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]]]).to(device)

        actual = op_script(trans_mat, boxes)
        expected = kornia.transform_points(trans_mat, boxes)

        assert_allclose(actual, expected)


class TestComposeTransforms:

    def test_translation_4x4(self, device):
        offset = 10
        trans_01 = identity_matrix(batch_size=1)[0].to(device)
        trans_12 = identity_matrix(batch_size=1)[0].to(device)
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = kornia.compose_transformations(trans_01, trans_12)
        assert_allclose(trans_02, trans_12)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, device, batch_size):
        offset = 10
        trans_01 = identity_matrix(batch_size)
        trans_12 = identity_matrix(batch_size)
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = kornia.compose_transformations(trans_01, trans_12)
        assert_allclose(trans_02, trans_12)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, device, batch_size):
        trans_01 = identity_matrix(batch_size).to(device)
        trans_12 = identity_matrix(batch_size).to(device)

        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        trans_12 = utils.tensor_to_gradcheck_var(trans_12)  # to var
        assert gradcheck(kornia.compose_transformations, (trans_01, trans_12,),
                         raise_exception=True)


class TestInverseTransformation:

    def test_translation_4x4(self, device):
        offset = 10
        trans_01 = identity_matrix(batch_size=1)[0].to(device)
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = kornia.inverse_transformation(trans_01)
        trans_01_hat = kornia.inverse_transformation(trans_10)
        assert_allclose(trans_01, trans_01_hat)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, device, batch_size):
        offset = 10
        trans_01 = identity_matrix(batch_size).to(device)
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = kornia.inverse_transformation(trans_01)
        trans_01_hat = kornia.inverse_transformation(trans_10)
        assert_allclose(trans_01, trans_01_hat)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_rotation_translation_Bx4x4(self, device, batch_size):
        offset = 10
        x, y, z = 0, 0, kornia.pi
        ones = torch.ones(batch_size).to(device)
        rmat_01 = euler_angles_to_rotation_matrix(x * ones, y * ones, z * ones)

        trans_01 = identity_matrix(batch_size).to(device)
        trans_01[..., :3, -1] += offset  # add offset to translation vector
        trans_01[..., :3, :3] = rmat_01[..., :3, :3]

        trans_10 = kornia.inverse_transformation(trans_01)
        trans_01_hat = kornia.inverse_transformation(trans_10)
        assert_allclose(trans_01, trans_01_hat)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, device, batch_size):
        trans_01 = identity_matrix(batch_size).to(device)
        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        assert gradcheck(kornia.inverse_transformation, (trans_01,),
                         raise_exception=True)


class TestRelativeTransformation:

    def test_translation_4x4(self, device):
        offset = 10.
        trans_01 = identity_matrix(batch_size=1)[0].to(device)
        trans_02 = identity_matrix(batch_size=1)[0].to(device)
        trans_02[..., :3, -1] += offset  # add offset to translation vector

        trans_12 = kornia.relative_transformation(trans_01, trans_02)
        trans_02_hat = kornia.compose_transformations(trans_01, trans_12)
        assert_allclose(trans_02_hat, trans_02)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_rotation_translation_Bx4x4(self, device, batch_size):
        offset = 10.
        x, y, z = 0., 0., kornia.pi
        ones = torch.ones(batch_size).to(device)
        rmat_02 = euler_angles_to_rotation_matrix(x * ones, y * ones, z * ones)

        trans_01 = identity_matrix(batch_size).to(device)
        trans_02 = identity_matrix(batch_size).to(device)
        trans_02[..., :3, -1] += offset  # add offset to translation vector
        trans_02[..., :3, :3] = rmat_02[..., :3, :3]

        trans_12 = kornia.relative_transformation(trans_01, trans_02)
        trans_02_hat = kornia.compose_transformations(trans_01, trans_12)
        assert_allclose(trans_02_hat, trans_02)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, device, batch_size):
        trans_01 = identity_matrix(batch_size).to(device)
        trans_02 = identity_matrix(batch_size).to(device)

        trans_01 = utils.tensor_to_gradcheck_var(trans_01)  # to var
        trans_02 = utils.tensor_to_gradcheck_var(trans_02)  # to var
        assert gradcheck(kornia.relative_transformation, (trans_01, trans_02,),
                         raise_exception=True)


class TestTransformLAFs:

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_points", [2, 3, 5])
    def test_transform_points(
            self, batch_size, num_points, device):
        # generate input data
        eye_size = 3
        lafs_src = torch.rand(batch_size, num_points, 2, 3).to(device)

        dst_homo_src = utils.create_random_homography(batch_size, eye_size)
        dst_homo_src = dst_homo_src.to(device)

        # transform the points from dst to ref
        lafs_dst = kornia.perspective_transform_lafs(dst_homo_src, lafs_src)

        # transform the points from ref to dst
        src_homo_dst = torch.inverse(dst_homo_src)
        lafs_dst_to_src = kornia.perspective_transform_lafs(src_homo_dst, lafs_dst)

        # projected should be equal as initial
        assert_allclose(lafs_src, lafs_dst_to_src)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        eye_size = 3
        points_src = torch.rand(batch_size, num_points, 2, 3).to(device)
        dst_homo_src = utils.create_random_homography(batch_size, eye_size).to(device)
        # evaluate function gradient
        points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
        dst_homo_src = utils.tensor_to_gradcheck_var(dst_homo_src)  # to var
        assert gradcheck(kornia.perspective_transform_lafs, (dst_homo_src, points_src,),
                         raise_exception=True)

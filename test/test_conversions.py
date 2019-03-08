import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils
from utils import check_equal_torch, check_equal_numpy
from common import device_type


def test_pi():
    assert pytest.approx(tgm.pi.item(), 3.141592)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_rad2deg(batch_shape, device_type):
    # generate input data
    x_rad = tgm.pi * torch.rand(batch_shape)
    x_rad = x_rad.to(torch.device(device_type))

    # convert radians/degrees
    x_deg = tgm.rad2deg(x_rad)
    x_deg_to_rad = tgm.deg2rad(x_deg)

    # compute error
    error = utils.compute_mse(x_rad, x_deg_to_rad)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(x_deg, tgm.RadToDeg()(x_rad))

    # evaluate function gradient
    assert gradcheck(tgm.rad2deg, (utils.tensor_to_gradcheck_var(x_rad),),
                     raise_exception=True)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_deg2rad(batch_shape, device_type):
    # generate input data
    x_deg = 180. * torch.rand(batch_shape)
    x_deg = x_deg.to(torch.device(device_type))

    # convert radians/degrees
    x_rad = tgm.deg2rad(x_deg)
    x_rad_to_deg = tgm.rad2deg(x_rad)

    # compute error
    error = utils.compute_mse(x_deg, x_rad_to_deg)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(x_rad, tgm.DegToRad()(x_deg))

    assert gradcheck(tgm.deg2rad, (utils.tensor_to_gradcheck_var(x_deg),),
                     raise_exception=True)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_convert_points_to_homogeneous(batch_shape, device_type):
    # generate input data
    points = torch.rand(batch_shape)
    points = points.to(torch.device(device_type))

    # to homogeneous
    points_h = tgm.convert_points_to_homogeneous(points)

    assert points_h.shape[-2] == batch_shape[-2]
    assert (points_h[..., -1] == torch.ones(points_h[..., -1].shape)).all()

    # functional
    assert torch.allclose(points_h, tgm.ConvertPointsToHomogeneous()(points))

    # evaluate function gradient
    points = utils.tensor_to_gradcheck_var(points)  # to var
    assert gradcheck(tgm.convert_points_to_homogeneous, (points,),
                     raise_exception=True)


@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_convert_points_from_homogeneous(batch_shape, device_type):
    # generate input data
    points_h = torch.rand(batch_shape)
    points_h = points_h.to(torch.device(device_type))
    points_h[..., -1] = 1.0

    # to euclidean
    points = tgm.convert_points_from_homogeneous(points_h)

    error = utils.compute_mse(points_h[..., :2], points)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(points, tgm.ConvertPointsFromHomogeneous()(points_h))

    # evaluate function gradient
    points = utils.tensor_to_gradcheck_var(points)  # to var
    assert gradcheck(tgm.convert_points_from_homogeneous, (points,),
                     raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("num_points", [2, 3, 5])
@pytest.mark.parametrize("num_dims", [2, 3])
def test_transform_points(batch_size, num_points, num_dims, device_type):
    # generate input data
    eye_size = num_dims + 1
    points_src = torch.rand(batch_size, num_points, num_dims)
    points_src = points_src.to(torch.device(device_type))

    dst_homo_src = utils.create_random_homography(batch_size, eye_size)
    dst_homo_src = dst_homo_src.to(torch.device(device_type))

    # transform the points from dst to ref
    points_dst = tgm.transform_points(dst_homo_src, points_src)

    # transform the points from ref to dst
    src_homo_dst = torch.inverse(dst_homo_src)
    points_dst_to_src = tgm.transform_points(src_homo_dst, points_dst)

    # projected should be equal as initial
    error = utils.compute_mse(points_src, points_dst_to_src)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(points_dst,
                          tgm.TransformPoints(dst_homo_src)(points_src))

    # evaluate function gradient
    points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
    dst_homo_src = utils.tensor_to_gradcheck_var(dst_homo_src)  # to var
    assert gradcheck(tgm.transform_points, (dst_homo_src, points_src,),
                     raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_angle_axis_to_rotation_matrix(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)
    angle_axis = torch.rand(batch_size, 3).to(device)
    eye_batch = utils.create_eye_batch(batch_size, 4).to(device)

    # apply transform
    rotation_matrix = tgm.angle_axis_to_rotation_matrix(angle_axis)

    rotation_matrix_eye = torch.matmul(
        rotation_matrix, rotation_matrix.transpose(1, 2))
    assert check_equal_torch(rotation_matrix_eye, eye_batch)

    # evaluate function gradient
    angle_axis = utils.tensor_to_gradcheck_var(angle_axis)  # to var
    assert gradcheck(tgm.angle_axis_to_rotation_matrix, (angle_axis,),
                     raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rtvec_to_pose_gradcheck(batch_size, device_type):
    # generate input data
    rtvec = torch.rand(batch_size, 6).to(torch.device(device_type))

    # evaluate function gradient
    rtvec = utils.tensor_to_gradcheck_var(rtvec)  # to var
    assert gradcheck(tgm.rtvec_to_pose, (rtvec,), raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rotation_matrix_to_angle_axis_gradcheck(batch_size, device_type):
    # generate input data
    rmat = torch.rand(batch_size, 3, 4).to(torch.device(device_type))

    # evaluate function gradient
    rmat = utils.tensor_to_gradcheck_var(rmat)  # to var
    assert gradcheck(tgm.rotation_matrix_to_angle_axis,
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

    assert check_equal_torch(tgm.rotation_matrix_to_angle_axis(rmat), rvec)

import unittest
import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils
from common import TEST_DEVICES


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_shape",
                         [(1, 1, 7, 32), (2, 3, 16, 31)])
def test_warp_perspective_rotation(batch_shape, device_type):
    # generate input data
    batch_size, channels, height, width = batch_shape
    alpha = 0.5 * tgm.pi * torch.ones(batch_size)  # 90 deg rotation

    # create data patch
    device = torch.device(device_type)
    patch = torch.rand(batch_shape).to(device)

    # create transformation (rotation)
    M = torch.eye(3, device=device).repeat(batch_size, 1, 1)  # Bx3x3
    M[:, 0, 0] = torch.cos(alpha)
    M[:, 0, 1] = -torch.sin(alpha)
    M[:, 1, 0] = torch.sin(alpha)
    M[:, 1, 1] = torch.cos(alpha)

    # apply transformation and inverse
    _, _, h, w = patch.shape
    patch_warped = tgm.warp_perspective(patch, M, dsize=(height, width))
    patch_warped_inv = tgm.warp_perspective(patch_warped, torch.inverse(M),
                                            dsize=(height, width))

    # generate mask to compute error
    mask = torch.ones_like(patch)
    mask_warped_inv = tgm.warp_perspective(
        tgm.warp_perspective(patch, M, dsize=(height, width)),
        torch.inverse(M), dsize=(height, width))

    assert utils.check_equal_torch(mask_warped_inv * patch,
                                   mask_warped_inv * patch_warped_inv)

    # evaluate function gradient
    patch = utils.tensor_to_gradcheck_var(patch)  # to var
    M = utils.tensor_to_gradcheck_var(M, requires_grad=False)  # to var
    assert gradcheck(tgm.warp_perspective, (patch, M, (height, width,)),
                     raise_exception=True)


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_get_perspective_transform(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)

    h_max, w_max = 64, 32  # height, width
    h = torch.ceil(h_max * torch.rand(batch_size)).to(device)
    w = torch.ceil(w_max * torch.rand(batch_size)).to(device)

    norm = torch.rand(batch_size, 4, 2).to(device)
    points_src = torch.zeros_like(norm)
    points_src[:, 1, 0] = h
    points_src[:, 2, 1] = w
    points_src[:, 3, 0] = h
    points_src[:, 3, 1] = w
    points_dst = points_src + norm

    # compute transform from source to target
    dst_homo_src = tgm.get_perspective_transform(points_src, points_dst)

    assert utils.check_equal_torch(
        tgm.transform_points(dst_homo_src, points_src), points_dst)

    # compute gradient check
    points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
    points_dst = utils.tensor_to_gradcheck_var(points_dst)  # to var
    assert gradcheck(tgm.get_perspective_transform,
                     (points_src, points_dst,), raise_exception=True)


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rotation_matrix2d(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)
    center_base = torch.zeros(batch_size, 2).to(device)
    angle_base = torch.ones(batch_size).to(device)
    scale_base = torch.ones(batch_size).to(device)

    # 90 deg rotation
    center = center_base
    angle = 90. * angle_base
    scale = scale_base
    M = tgm.get_rotation_matrix2d(center, angle, scale)

    for i in range(batch_size):
        pytest.approx(M[i, 0, 0].item(), 0.0)
        pytest.approx(M[i, 0, 1].item(), 1.0)
        pytest.approx(M[i, 1, 0].item(), -1.0)
        pytest.approx(M[i, 1, 1].item(), 0.0)

    # 90 deg rotation + 2x scale
    center = center_base
    angle = 90. * angle_base
    scale = 2. * scale_base
    M = tgm.get_rotation_matrix2d(center, angle, scale)

    for i in range(batch_size):
        pytest.approx(M[i, 0, 0].item(), 0.0)
        pytest.approx(M[i, 0, 1].item(), 2.0)
        pytest.approx(M[i, 1, 0].item(), -2.0)
        pytest.approx(M[i, 1, 1].item(), 0.0)

    # 45 deg rotation
    center = center_base
    angle = 45. * angle_base
    scale = scale_base
    M = tgm.get_rotation_matrix2d(center, angle, scale)

    for i in range(batch_size):
        pytest.approx(M[i, 0, 0].item(), 0.7071)
        pytest.approx(M[i, 0, 1].item(), 0.7071)
        pytest.approx(M[i, 1, 0].item(), -0.7071)
        pytest.approx(M[i, 1, 1].item(), 0.7071)

    # evaluate function gradient
    center = utils.tensor_to_gradcheck_var(center)  # to var
    angle = utils.tensor_to_gradcheck_var(angle)  # to var
    scale = utils.tensor_to_gradcheck_var(scale)  # to var
    assert gradcheck(tgm.get_rotation_matrix2d, (center, angle, scale),
                     raise_exception=True)


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("channels", [1, 5])
def test_warp_perspective_crop(batch_size, device_type, channels):
    # generate input data
    src_h, src_w = 3, 4
    dst_h, dst_w = 3, 2
    device = torch.device(device_type)

    # [x, y] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src = torch.rand(batch_size, 4, 2).to(device)
    points_src[:, :, 0] *= dst_h
    points_src[:, :, 1] *= dst_w

    # [x, y] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst = torch.zeros_like(points_src)
    points_dst[:, 1, 0] = dst_w - 1
    points_dst[:, 2, 0] = dst_w - 1
    points_dst[:, 2, 1] = dst_h - 1
    points_dst[:, 3, 1] = dst_h - 1

    # compute transformation between points
    dst_pix_trans_src_pix = tgm.get_perspective_transform(
        points_src, points_dst)

    # create points grid in normalized coordinates
    grid_src_norm = tgm.create_meshgrid(src_h, src_w,
                                        normalized_coordinates=True)
    grid_src_norm = grid_src_norm.repeat(batch_size, 1, 1, 1).to(device)

    # create points grid in pixel coordinates
    grid_src_pix = tgm.create_meshgrid(src_h, src_w,
                                       normalized_coordinates=False)
    grid_src_pix = grid_src_pix.repeat(batch_size, 1, 1, 1).to(device)

    src_norm_trans_src_pix = tgm.normal_transform_pixel(src_h, src_w).repeat(
        batch_size, 1, 1).to(device)
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)

    dst_norm_trans_dst_pix = tgm.normal_transform_pixel(dst_h, dst_w).repeat(
        batch_size, 1, 1).to(device)

    # transform pixel grid
    grid_dst_pix = tgm.transform_points(
        dst_pix_trans_src_pix.unsqueeze(1), grid_src_pix)
    grid_dst_norm = tgm.transform_points(
        dst_norm_trans_dst_pix.unsqueeze(1), grid_dst_pix)

    # transform norm grid
    dst_norm_trans_src_norm = torch.matmul(
        dst_norm_trans_dst_pix, torch.matmul(
            dst_pix_trans_src_pix, src_pix_trans_src_norm))
    grid_dst_norm2 = tgm.transform_points(
        dst_norm_trans_src_norm.unsqueeze(1), grid_src_norm)

    # grids should be equal
    # TODO: investage why precision is that low
    assert utils.check_equal_torch(grid_dst_norm, grid_dst_norm2, 1e-2)

    # warp tensor
    patch = torch.rand(batch_size, channels, src_h, src_w)
    patch_warped = tgm.warp_perspective(
        patch, dst_pix_trans_src_pix, (dst_h, dst_w))
    assert patch_warped.shape == (batch_size, channels, dst_h, dst_w)

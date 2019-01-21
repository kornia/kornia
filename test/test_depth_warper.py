import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils


def test_depth_warper():
    # generate input data
    batch_size = 1
    height, width = 8, 8
    cx, cy = width / 2, height / 2
    fx, fy = 1., 1.
    rx, ry, rz = 0., 0., 0.
    tx, ty, tz = 0., 0., 0.
    offset = 1.  # we will apply a 1unit offset to `i` camera

    pinhole_ref = utils.create_pinhole(
        fx, fy, cx, cy, height, width, rx, ry, rx, tx, ty, tz)
    pinhole_ref = pinhole_ref.expand(batch_size, -1)

    pinhole_i = utils.create_pinhole(
        fx,
        fy,
        cx,
        cy,
        height,
        width,
        rx,
        ry,
        rx,
        tx + offset,
        ty + offset,
        tz)
    pinhole_i = pinhole_i.expand(batch_size, -1)

    # create checkerboard
    board = utils.create_checkerboard(height, width, 4)
    patch_i = torch.from_numpy(board).view(
        1, 1, height, width).expand(batch_size, 1, height, width)

    # instantiate warper and compute relative homographies
    warper = tgm.DepthWarper(pinhole_i)
    warper.compute_homographies(
        pinhole_ref, scale=torch.ones(
            batch_size))

    # generate synthetic inverse depth
    inv_depth_ref = torch.ones(batch_size, 1, height, width)

    # warpd source patch by depth
    patch_ref = warper(inv_depth_ref, patch_i)

    # compute error
    assert utils.check_equal_torch(
        patch_ref[..., :int(height - offset), :int(width - offset)],
        patch_i[..., int(offset):, int(offset):])

    # test functional
    patch_ref_functional = tgm.depth_warp(pinhole_i, pinhole_ref,
                                          inv_depth_ref, patch_i)
    assert utils.check_equal_torch(patch_ref, patch_ref_functional)

    # test compute_projection
    # TODO: this might need a better test
    xy_projected = warper._compute_projection(0.0, 0.0, 1.0)
    assert xy_projected.shape == (1, 2)

    # test compute_subpixel_step
    # TODO: this might need a better test
    subpixel_step = warper.compute_subpixel_step()
    assert pytest.approx(subpixel_step.item(), 0.3536)

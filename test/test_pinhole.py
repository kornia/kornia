import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utilities
from common import TEST_DEVICES


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 5, 6])
def test_scale_pinhole(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)
    pinholes = torch.rand(batch_size, 12).to(device)
    scales = torch.rand(batch_size).to(device)

    pinholes_scale = tgm.scale_pinhole(pinholes, scales)
    assert utils.check_equal_torch(
        pinholes_scale[..., :6] / scales.unsqueeze(-1), pinholes[..., :6])

    # evaluate function gradient
    pinholes = utils.tensor_to_gradcheck_var(pinholes)  # to var
    scales = utils.tensor_to_gradcheck_var(scales)  # to var
    assert gradcheck(tgm.scale_pinhole, (pinholes, scales,),
                     raise_exception=True)


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 5, 6])
def test_pinhole_matrix(batch_size, device_type):
    # generate input data
    image_height, image_width = 32., 32.
    cx, cy = image_width / 2, image_height / 2
    fx, fy = 1., 1.
    rx, ry, rz = 0., 0., 0.
    tx, ty, tz = 0., 0., 0.
    offset_x = 10.  # we will apply a 10units offset to `i` camera
    eps = 1e-6

    pinhole = utils.create_pinhole(
        fx, fy, cx, cy, image_height, image_width, rx, ry, rx, tx, ty, tz)
    pinhole = pinhole.repeat(batch_size, 1).to(torch.device(device_type))

    pinhole_matrix = tgm.pinhole_matrix(pinhole)

    ones = torch.ones(batch_size)
    assert bool((pinhole_matrix[:, 0, 0] == fx * ones).all())
    assert bool((pinhole_matrix[:, 1, 1] == fy * ones).all())
    assert bool((pinhole_matrix[:, 0, 2] == cx * ones).all())
    assert bool((pinhole_matrix[:, 1, 2] == cy * ones).all())

    # functional
    assert tgm.PinholeMatrix()(pinhole).shape == (batch_size, 4, 4)

    # evaluate function gradient
    pinhole = utils.tensor_to_gradcheck_var(pinhole)  # to var
    assert gradcheck(tgm.pinhole_matrix, (pinhole,),
                     raise_exception=True)


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 5, 6])
def test_inverse_pinhole_matrix(batch_size, device_type):
    # generate input data
    image_height, image_width = 32., 32.
    cx, cy = image_width / 2, image_height / 2
    fx, fy = 1., 1.
    rx, ry, rz = 0., 0., 0.
    tx, ty, tz = 0., 0., 0.
    offset_x = 10.  # we will apply a 10units offset to `i` camera
    eps = 1e-6

    pinhole = utils.create_pinhole(
        fx, fy, cx, cy, image_height, image_width, rx, ry, rx, tx, ty, tz)
    pinhole = pinhole.repeat(batch_size, 1).to(torch.device(device_type))

    pinhole_matrix = tgm.inverse_pinhole_matrix(pinhole)

    ones = torch.ones(batch_size)
    assert utils.check_equal_torch(pinhole_matrix[:, 0, 0], (1. / fx) * ones)
    assert utils.check_equal_torch(pinhole_matrix[:, 1, 1], (1. / fy) * ones)
    assert utils.check_equal_torch(
        pinhole_matrix[:, 0, 2], (-1. * cx / fx) * ones)
    assert utils.check_equal_torch(
        pinhole_matrix[:, 1, 2], (-1. * cy / fx) * ones)

    # functional
    assert tgm.InversePinholeMatrix()(pinhole).shape == (batch_size, 4, 4)

    # evaluate function gradient
    pinhole = utils.tensor_to_gradcheck_var(pinhole)  # to var
    assert gradcheck(tgm.pinhole_matrix, (pinhole,),
                     raise_exception=True)


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 5, 6])
def test_homography_i_H_ref(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)
    image_height, image_width = 32., 32.
    cx, cy = image_width / 2, image_height / 2
    fx, fy = 1., 1.
    rx, ry, rz = 0., 0., 0.
    tx, ty, tz = 0., 0., 0.
    offset_x = 10.  # we will apply a 10units offset to `i` camera
    eps = 1e-6

    pinhole_ref = utils.create_pinhole(
        fx, fy, cx, cy, image_height, image_width, rx, ry, rx, tx, ty, tz)
    pinhole_ref = pinhole_ref.repeat(batch_size, 1).to(device)

    pinhole_i = utils.create_pinhole(
        fx,
        fy,
        cx,
        cy,
        image_height,
        image_width,
        rx,
        ry,
        rx,
        tx + offset_x,
        ty,
        tz)
    pinhole_i = pinhole_i.repeat(batch_size, 1).to(device)

    # compute homography from ref to i
    i_H_ref = tgm.homography_i_H_ref(pinhole_i, pinhole_ref) + eps
    i_H_ref_inv = torch.inverse(i_H_ref)

    # compute homography from i to ref
    ref_H_i = tgm.homography_i_H_ref(pinhole_ref, pinhole_i) + eps
    assert utils.check_equal_torch(i_H_ref_inv, ref_H_i)

    # evaluate function gradient
    assert gradcheck(tgm.homography_i_H_ref,
                     (utils.tensor_to_gradcheck_var(pinhole_ref) + eps,
                      utils.tensor_to_gradcheck_var(pinhole_i) + eps,),
                     raise_exception=True)

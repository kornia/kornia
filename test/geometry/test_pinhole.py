import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestPinholeCamera:
    def _create_intrinsics(self, batch_size, fx, fy, cx, cy):
        intrinsics = torch.eye(4)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = cx
        intrinsics[..., 1, 2] = cy
        return intrinsics.expand(batch_size, -1, -1)

    def _create_extrinsics(self, batch_size, tx, ty, tz):
        extrinsics = torch.eye(4)
        extrinsics[..., 0, -1] = tx
        extrinsics[..., 1, -1] = ty
        extrinsics[..., 2, -1] = tz
        return extrinsics.expand(batch_size, -1, -1)

    def test_smoke(self, device):
        intrinsics = torch.eye(4)[None].to(device)
        extrinsics = torch.eye(4)[None].to(device)
        height = torch.ones(1).to(device)
        width = torch.ones(1).to(device)
        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)
        assert isinstance(pinhole, kornia.PinholeCamera)

    def test_pinhole_camera_attributes(self, device):
        batch_size = 1
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy).to(device)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz).to(device)
        height = torch.ones(batch_size).to(device) * height
        width = torch.ones(batch_size).to(device) * width

        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)

        assert pinhole.batch_size == batch_size
        assert pinhole.fx.item() == fx
        assert pinhole.fy.item() == fy
        assert pinhole.cx.item() == cx
        assert pinhole.cy.item() == cy
        assert pinhole.tx.item() == tx
        assert pinhole.ty.item() == ty
        assert pinhole.tz.item() == tz
        assert pinhole.height.item() == height
        assert pinhole.width.item() == width
        assert pinhole.rt_matrix.shape == (batch_size, 3, 4)
        assert pinhole.camera_matrix.shape == (batch_size, 3, 3)
        assert pinhole.rotation_matrix.shape == (batch_size, 3, 3)
        assert pinhole.translation_vector.shape == (batch_size, 3, 1)

    def test_pinhole_camera_translation_setters(self, device):
        batch_size = 1
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy).to(device)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz).to(device)
        height = torch.ones(batch_size).to(device) * height
        width = torch.ones(batch_size).to(device) * width

        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)

        assert pinhole.tx.item() == tx
        assert pinhole.ty.item() == ty
        assert pinhole.tz.item() == tz

        # add offset
        pinhole.tx += 3.
        pinhole.ty += 2.
        pinhole.tz += 1.

        assert pinhole.tx.item() == tx + 3.
        assert pinhole.ty.item() == ty + 2.
        assert pinhole.tz.item() == tz + 1.

        # set to zero
        pinhole.tx = 0.
        pinhole.ty = 0.
        pinhole.tz = 0.

        assert pinhole.tx.item() == 0.
        assert pinhole.ty.item() == 0.
        assert pinhole.tz.item() == 0.

    def test_pinhole_camera_attributes_batch2(self, device):
        batch_size = 2
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy).to(device)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz).to(device)
        height = torch.ones(batch_size).to(device) * height
        width = torch.ones(batch_size).to(device) * width

        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)

        assert pinhole.batch_size == batch_size
        assert pinhole.fx.shape[0] == batch_size
        assert pinhole.fy.shape[0] == batch_size
        assert pinhole.cx.shape[0] == batch_size
        assert pinhole.cy.shape[0] == batch_size
        assert pinhole.tx.shape[0] == batch_size
        assert pinhole.ty.shape[0] == batch_size
        assert pinhole.tz.shape[0] == batch_size
        assert pinhole.height.shape[0] == batch_size
        assert pinhole.width.shape[0] == batch_size
        assert pinhole.rt_matrix.shape == (batch_size, 3, 4)
        assert pinhole.camera_matrix.shape == (batch_size, 3, 3)
        assert pinhole.rotation_matrix.shape == (batch_size, 3, 3)
        assert pinhole.translation_vector.shape == (batch_size, 3, 1)

    def test_pinhole_camera_scale(self, device):
        batch_size = 2
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3
        scale_val = 2.0

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy).to(device)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz).to(device)
        height = torch.ones(batch_size).to(device) * height
        width = torch.ones(batch_size).to(device) * width
        scale_factor = torch.ones(batch_size).to(device) * scale_val

        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)
        pinhole_scale = pinhole.scale(scale_factor)

        assert_allclose(
            pinhole_scale.intrinsics[..., 0, 0],
            pinhole.intrinsics[..., 0, 0] * scale_val)  # fx
        assert_allclose(
            pinhole_scale.intrinsics[..., 1, 1],
            pinhole.intrinsics[..., 1, 1] * scale_val)  # fy
        assert_allclose(
            pinhole_scale.intrinsics[..., 0, 2],
            pinhole.intrinsics[..., 0, 2] * scale_val)  # cx
        assert_allclose(
            pinhole_scale.intrinsics[..., 1, 2],
            pinhole.intrinsics[..., 1, 2] * scale_val)  # cy
        assert_allclose(
            pinhole_scale.height,
            pinhole.height * scale_val)
        assert_allclose(
            pinhole_scale.width,
            pinhole.width * scale_val)

    def test_pinhole_camera_scale_inplace(self, device):
        batch_size = 2
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3
        scale_val = 2.0

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy).to(device)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz).to(device)
        height = torch.ones(batch_size).to(device) * height
        width = torch.ones(batch_size).to(device) * width
        scale_factor = torch.ones(batch_size).to(device) * scale_val

        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)
        pinhole_scale = pinhole.clone()
        pinhole_scale.scale_(scale_factor)

        assert_allclose(
            pinhole_scale.intrinsics[..., 0, 0],
            pinhole.intrinsics[..., 0, 0] * scale_val)  # fx
        assert_allclose(
            pinhole_scale.intrinsics[..., 1, 1],
            pinhole.intrinsics[..., 1, 1] * scale_val)  # fy
        assert_allclose(
            pinhole_scale.intrinsics[..., 0, 2],
            pinhole.intrinsics[..., 0, 2] * scale_val)  # cx
        assert_allclose(
            pinhole_scale.intrinsics[..., 1, 2],
            pinhole.intrinsics[..., 1, 2] * scale_val)  # cy
        assert_allclose(
            pinhole_scale.height, pinhole.height * scale_val)
        assert_allclose(
            pinhole_scale.width, pinhole.width * scale_val)

    def _make_example_camera(self):
        return kornia.PinholeCamera(
            torch.randn(3, 4, 4),
            torch.randn(3, 4, 4),
            torch.randn(3),
            torch.randn(3),
        )

    def test_to_dtype(self, device):
        camera = self._make_example_camera().to(device)
        assert camera.extrinsics.dtype == torch.float32

        camera = camera.to(torch.float64)
        assert camera.extrinsics.dtype == torch.float64
        assert camera.intrinsics.dtype == torch.float64
        assert camera.height.dtype == torch.float64
        assert camera.width.dtype == torch.float64

        camera = camera.to(torch.float32)
        assert camera.extrinsics.dtype == torch.float32
        assert camera.intrinsics.dtype == torch.float32
        assert camera.height.dtype == torch.float32
        assert camera.width.dtype == torch.float32

    def test_to_device(self, device):
        if 'cuda' not in str(device):
            pytest.skip('This test is cuda-specific')

        cpu = torch.device('cpu')
        gpu = torch.device('cuda')

        camera = self._make_example_camera()
        assert camera.device == cpu

        camera = camera.to(gpu)
        assert camera.device == gpu

        camera = camera.to(cpu)
        assert camera.device == cpu

    def test_pin_memory(self, device):
        if 'cuda' not in str(device):
            pytest.skip('This test is cuda-specific')

        camera = self._make_example_camera().to(device)
        assert not camera.intrinsics.is_pinned()
        assert not camera.extrinsics.is_pinned()
        assert not camera.height.is_pinned()
        assert not camera.width.is_pinned()

        camera = camera.pin_memory()
        assert camera.intrinsics.is_pinned()
        assert camera.extrinsics.is_pinned()
        assert camera.height.is_pinned()
        assert camera.width.is_pinned()

    def test_getitem_slice(self, device):
        camera = self._make_example_camera().to(device)

        sliced = camera[1:]
        assert (sliced.intrinsics == camera.intrinsics[1:]).all().item()
        assert (sliced.extrinsics == camera.extrinsics[1:]).all().item()
        assert (sliced.height == camera.height[1:]).all().item()
        assert (sliced.width == camera.width[1:]).all().item()

    def test_getitem_int(self, device):
        camera = self._make_example_camera().to(device)

        sliced = camera[1]
        assert (sliced.intrinsics[0] == camera.intrinsics[1]).all().item()
        assert (sliced.extrinsics[0] == camera.extrinsics[1]).all().item()
        assert (sliced.height[0] == camera.height[1]).all().item()
        assert (sliced.width[0] == camera.width[1]).all().item()

'''@pytest.mark.parametrize("batch_size", [1, 2, 5, 6])
def test_scale_pinhole(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)
    pinholes = torch.rand(batch_size, 12).to(device)
    scales = torch.rand(batch_size).to(device)

    pinholes_scale = kornia.scale_pinhole(pinholes, scales)
    assert_allclose(
        pinholes_scale[..., :6] / scales.unsqueeze(-1), pinholes[..., :6])

    # evaluate function gradient
    pinholes = utils.tensor_to_gradcheck_var(pinholes)  # to var
    scales = utils.tensor_to_gradcheck_var(scales)  # to var
    assert gradcheck(kornia.scale_pinhole, (pinholes, scales,),
                     raise_exception=True)


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

    pinhole_matrix = kornia.pinhole_matrix(pinhole)

    ones = torch.ones(batch_size)
    assert bool((pinhole_matrix[:, 0, 0] == fx * ones).all())
    assert bool((pinhole_matrix[:, 1, 1] == fy * ones).all())
    assert bool((pinhole_matrix[:, 0, 2] == cx * ones).all())
    assert bool((pinhole_matrix[:, 1, 2] == cy * ones).all())

    # functional
    assert kornia.PinholeMatrix()(pinhole).shape == (batch_size, 4, 4)

    # evaluate function gradient
    pinhole = utils.tensor_to_gradcheck_var(pinhole)  # to var
    assert gradcheck(kornia.pinhole_matrix, (pinhole,),
                     raise_exception=True)


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

    pinhole_matrix = kornia.inverse_pinhole_matrix(pinhole)

    ones = torch.ones(batch_size)
    assert_allclose(pinhole_matrix[:, 0, 0], (1. / fx) * ones)
    assert_allclose(pinhole_matrix[:, 1, 1], (1. / fy) * ones)
    assert_allclose(
        pinhole_matrix[:, 0, 2], (-1. * cx / fx) * ones)
    assert_allclose(
        pinhole_matrix[:, 1, 2], (-1. * cy / fx) * ones)

    # functional
    assert kornia.InversePinholeMatrix()(pinhole).shape == (batch_size, 4, 4)

    # evaluate function gradient
    pinhole = utils.tensor_to_gradcheck_var(pinhole)  # to var
    assert gradcheck(kornia.pinhole_matrix, (pinhole,),
                     raise_exception=True)


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
    i_H_ref = kornia.homography_i_H_ref(pinhole_i, pinhole_ref) + eps
    i_H_ref_inv = torch.inverse(i_H_ref)

    # compute homography from i to ref
    ref_H_i = kornia.homography_i_H_ref(pinhole_ref, pinhole_i) + eps
    assert_allclose(i_H_ref_inv, ref_H_i)

    # evaluate function gradient
    assert gradcheck(kornia.homography_i_H_ref,
                     (utils.tensor_to_gradcheck_var(pinhole_ref) + eps,
                      utils.tensor_to_gradcheck_var(pinhole_i) + eps,),
                     raise_exception=True)'''

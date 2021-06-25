import torch

import kornia
from kornia.testing import assert_close


class TestPinholeCamera:
    def _create_intrinsics(self, batch_size, fx, fy, cx, cy, device, dtype):
        intrinsics = torch.eye(4, device=device, dtype=dtype)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = cx
        intrinsics[..., 1, 2] = cy
        return intrinsics.expand(batch_size, -1, -1)

    def _create_extrinsics(self, batch_size, tx, ty, tz, device, dtype):
        extrinsics = torch.eye(4, device=device, dtype=dtype)
        extrinsics[..., 0, -1] = tx
        extrinsics[..., 1, -1] = ty
        extrinsics[..., 2, -1] = tz
        return extrinsics.expand(batch_size, -1, -1)

    def test_smoke(self, device, dtype):
        intrinsics = torch.eye(4, device=device, dtype=dtype)[None]
        extrinsics = torch.eye(4, device=device, dtype=dtype)[None]
        height = torch.ones(1, device=device, dtype=dtype)
        width = torch.ones(1, device=device, dtype=dtype)
        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)
        assert isinstance(pinhole, kornia.PinholeCamera)

    def test_pinhole_camera_attributes(self, device, dtype):
        batch_size = 1
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz, device=device, dtype=dtype)
        height = torch.ones(batch_size, device=device, dtype=dtype) * height
        width = torch.ones(batch_size, device=device, dtype=dtype) * width

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

    def test_pinhole_camera_translation_setters(self, device, dtype):
        batch_size = 1
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz, device=device, dtype=dtype)
        height = torch.ones(batch_size, device=device, dtype=dtype) * height
        width = torch.ones(batch_size, device=device, dtype=dtype) * width

        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)

        assert pinhole.tx.item() == tx
        assert pinhole.ty.item() == ty
        assert pinhole.tz.item() == tz

        # add offset
        pinhole.tx += 3.0
        pinhole.ty += 2.0
        pinhole.tz += 1.0

        assert pinhole.tx.item() == tx + 3.0
        assert pinhole.ty.item() == ty + 2.0
        assert pinhole.tz.item() == tz + 1.0

        # set to zero
        pinhole.tx = 0.0
        pinhole.ty = 0.0
        pinhole.tz = 0.0

        assert pinhole.tx.item() == 0.0
        assert pinhole.ty.item() == 0.0
        assert pinhole.tz.item() == 0.0

    def test_pinhole_camera_attributes_batch2(self, device, dtype):
        batch_size = 2
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz, device=device, dtype=dtype)
        height = torch.ones(batch_size, device=device, dtype=dtype) * height
        width = torch.ones(batch_size, device=device, dtype=dtype) * width

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

    def test_pinhole_camera_scale(self, device, dtype):
        batch_size = 2
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3
        scale_val = 2.0

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz, device=device, dtype=dtype)
        height = torch.ones(batch_size, device=device, dtype=dtype) * height
        width = torch.ones(batch_size, device=device, dtype=dtype) * width
        scale_factor = torch.ones(batch_size, device=device, dtype=dtype) * scale_val

        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)
        pinhole_scale = pinhole.scale(scale_factor)

        assert_close(
            pinhole_scale.intrinsics[..., 0, 0], pinhole.intrinsics[..., 0, 0] * scale_val, atol=1e-4, rtol=1e-4
        )  # fx
        assert_close(
            pinhole_scale.intrinsics[..., 1, 1], pinhole.intrinsics[..., 1, 1] * scale_val, atol=1e-4, rtol=1e-4
        )  # fy
        assert_close(
            pinhole_scale.intrinsics[..., 0, 2], pinhole.intrinsics[..., 0, 2] * scale_val, atol=1e-4, rtol=1e-4
        )  # cx
        assert_close(
            pinhole_scale.intrinsics[..., 1, 2], pinhole.intrinsics[..., 1, 2] * scale_val, atol=1e-4, rtol=1e-4
        )  # cy
        assert_close(pinhole_scale.height, pinhole.height * scale_val, atol=1e-4, rtol=1e-4)
        assert_close(pinhole_scale.width, pinhole.width * scale_val, atol=1e-4, rtol=1e-4)

    def test_pinhole_camera_scale_inplace(self, device, dtype):
        batch_size = 2
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3
        scale_val = 2.0

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz, device=device, dtype=dtype)
        height = torch.ones(batch_size, device=device, dtype=dtype) * height
        width = torch.ones(batch_size, device=device, dtype=dtype) * width
        scale_factor = torch.ones(batch_size, device=device, dtype=dtype) * scale_val

        pinhole = kornia.PinholeCamera(intrinsics, extrinsics, height, width)
        pinhole_scale = pinhole.clone()
        pinhole_scale.scale_(scale_factor)

        assert_close(
            pinhole_scale.intrinsics[..., 0, 0], pinhole.intrinsics[..., 0, 0] * scale_val, atol=1e-4, rtol=1e-4
        )  # fx
        assert_close(
            pinhole_scale.intrinsics[..., 1, 1], pinhole.intrinsics[..., 1, 1] * scale_val, atol=1e-4, rtol=1e-4
        )  # fy
        assert_close(
            pinhole_scale.intrinsics[..., 0, 2], pinhole.intrinsics[..., 0, 2] * scale_val, atol=1e-4, rtol=1e-4
        )  # cx
        assert_close(
            pinhole_scale.intrinsics[..., 1, 2], pinhole.intrinsics[..., 1, 2] * scale_val, atol=1e-4, rtol=1e-4
        )  # cy
        assert_close(pinhole_scale.height, pinhole.height * scale_val, atol=1e-4, rtol=1e-4)
        assert_close(pinhole_scale.width, pinhole.width * scale_val, atol=1e-4, rtol=1e-4)


'''@pytest.mark.parametrize("batch_size", [1, 2, 5, 6])
def test_scale_pinhole(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)
    pinholes = torch.rand(batch_size, 12, device=device, dtype=dtype)
    scales = torch.rand(batch_size, device=device, dtype=dtype)

    pinholes_scale = kornia.scale_pinhole(pinholes, scales)
    assert_close(
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
    assert_close(pinhole_matrix[:, 0, 0], (1. / fx) * ones)
    assert_close(pinhole_matrix[:, 1, 1], (1. / fy) * ones)
    assert_close(
        pinhole_matrix[:, 0, 2], (-1. * cx / fx) * ones)
    assert_close(
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
    pinhole_ref = pinhole_ref.repeat(batch_size, 1, device=device, dtype=dtype)

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
    pinhole_i = pinhole_i.repeat(batch_size, 1, device=device, dtype=dtype)

    # compute homography from ref to i
    i_H_ref = kornia.homography_i_H_ref(pinhole_i, pinhole_ref) + eps
    i_H_ref_inv = torch.inverse(i_H_ref)

    # compute homography from i to ref
    ref_H_i = kornia.homography_i_H_ref(pinhole_ref, pinhole_i) + eps
    assert_close(i_H_ref_inv, ref_H_i)

    # evaluate function gradient
    assert gradcheck(kornia.homography_i_H_ref,
                     (utils.tensor_to_gradcheck_var(pinhole_ref) + eps,
                      utils.tensor_to_gradcheck_var(pinhole_i) + eps,),
                     raise_exception=True)'''

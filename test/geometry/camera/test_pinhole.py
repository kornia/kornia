import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.testing import assert_close, tensor_to_gradcheck_var


class TestCam2Pixel:
    def _create_intrinsics(self, batch_size, fx, fy, cx, cy, device, dtype):
        temp = torch.eye(4, device=device, dtype=dtype)
        temp[0, 0], temp[0, 2] = fx, cx
        temp[1, 1], temp[1, 2] = fy, cy
        intrinsics = temp.expand(batch_size, -1, -1)
        return intrinsics

    def _create_intrinsics_inv(self, batch_size, fx, fy, cx, cy, device, dtype):
        temp = torch.eye(4, device=device, dtype=dtype)
        temp[0, 0], temp[0, 2] = 1 / fx, -cx / fx
        temp[1, 1], temp[1, 2] = 1 / fy, -cy / fy
        intrinsics_inv = temp.expand(batch_size, -1, -1)
        return intrinsics_inv

    def _get_samples(self, shape, low, high, device, dtype):
        """Return a tensor having the given shape and whose values are in the range [low, high)"""
        return ((high - low) * torch.rand(shape, device=device, dtype=dtype)) + low

    @pytest.mark.parametrize("batch_size", (1,))
    def test_smoke(self, batch_size, device, dtype):
        H, W = 250, 500
        fx, fy = W, H
        cx, cy = W / 2, H / 2
        eps = 1e-12
        seed = 77
        low, high = -500, 500

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)

        # Setting the projection matrix to the intrinsic matrix for
        # simplicity (i.e. assuming that the RT matrix is an identity matrix)
        proj_mat = intrinsics

        torch.manual_seed(seed)
        cam_coords_src = self._get_samples((batch_size, H, W, 3), low, high, device, dtype)

        pixel_coords_dst = kornia.geometry.camera.cam2pixel(
            cam_coords_src=cam_coords_src, dst_proj_src=proj_mat, eps=eps
        )
        assert pixel_coords_dst.shape == (batch_size, H, W, 2)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_consistency(self, batch_size, device, dtype):
        H, W = 250, 500
        fx, fy = W, H
        cx, cy = W / 2, H / 2
        eps = 1e-12
        seed = 77
        low, high = -500, 500

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        intrinsics_inv = self._create_intrinsics_inv(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)

        # Setting the projection matrix to the intrinsic matrix for
        # simplicity (i.e. assuming that the RT matrix is an identity matrix)
        proj_mat = intrinsics

        torch.manual_seed(seed)
        cam_coords_input = self._get_samples((batch_size, H, W, 3), low, high, device, dtype)

        pixel_coords_output = kornia.geometry.camera.cam2pixel(
            cam_coords_src=cam_coords_input, dst_proj_src=proj_mat, eps=eps
        )

        last_ch = torch.ones((batch_size, H, W, 1), device=device, dtype=dtype)
        pixel_coords_concat = torch.cat([pixel_coords_output, last_ch], axis=-1)

        depth = cam_coords_input[..., 2:3].permute(0, 3, 1, 2).contiguous()
        cam_coords_output = kornia.geometry.camera.pixel2cam(
            depth=depth, intrinsics_inv=intrinsics_inv, pixel_coords=pixel_coords_concat
        )

        assert_close(cam_coords_output, cam_coords_input, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", (1,))
    def test_gradcheck(self, batch_size, device, dtype):
        H, W = 10, 20
        fx, fy = W, H
        cx, cy = W / 2, H / 2
        eps = 1e-12
        seed = 77
        low, high = -500, 500
        atol, rtol = 1e-5, 1e-3

        # Different tolerances for the below case.
        if (device.type == "cuda") and (dtype == torch.float64):
            atol, rtol = 1e-4, 1e-2

        # If contiguous() is not called, gradcheck fails
        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype).contiguous()

        # Setting the projection matrix to the intrinsic matrix for
        # simplicity (i.e. assuming that the RT matrix is an identity matrix)
        proj_mat = intrinsics

        torch.manual_seed(seed)
        cam_coords_src = self._get_samples((batch_size, H, W, 3), low, high, device, dtype)

        cam_coords_src = tensor_to_gradcheck_var(cam_coords_src)
        proj_mat = tensor_to_gradcheck_var(proj_mat)

        assert gradcheck(
            kornia.geometry.camera.cam2pixel,
            (cam_coords_src, proj_mat, eps),
            raise_exception=True,
            atol=atol,
            rtol=rtol,
        )


class TestPixel2Cam:
    def _create_intrinsics(self, batch_size, fx, fy, cx, cy, device, dtype):
        temp = torch.eye(4, device=device, dtype=dtype)
        temp[0, 0], temp[0, 2] = fx, cx
        temp[1, 1], temp[1, 2] = fy, cy
        intrinsics = temp.expand(batch_size, -1, -1)
        return intrinsics

    def _create_intrinsics_inv(self, batch_size, fx, fy, cx, cy, device, dtype):
        temp = torch.eye(4, device=device, dtype=dtype)
        temp[0, 0], temp[0, 2] = 1 / fx, -cx / fx
        temp[1, 1], temp[1, 2] = 1 / fy, -cy / fy
        intrinsics_inv = temp.expand(batch_size, -1, -1)
        return intrinsics_inv

    def _get_samples(self, shape, low, high, device, dtype):
        """Return a tensor having the given shape and whose values are in the range [low, high)"""
        return ((high - low) * torch.rand(shape, device=device, dtype=dtype)) + low

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_smoke(self, batch_size, device, dtype):
        H, W = 250, 500
        fx, fy = W, H
        cx, cy = W / 2, H / 2
        seed = 77
        low_1, high_1 = -500, 500
        low_2, high_2 = -(max(W, H) * 3), (max(W, H) * 3)

        torch.manual_seed(seed)
        depth = self._get_samples((batch_size, 1, H, W), low_1, high_1, device, dtype)
        pixel_coords = self._get_samples((batch_size, H, W, 2), low_2, high_2, device, dtype)

        last_ch = torch.ones((batch_size, H, W, 1), device=device, dtype=dtype)
        pixel_coords_input = torch.cat([pixel_coords, last_ch], axis=-1)

        intrinsics_inv = self._create_intrinsics_inv(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)

        output = kornia.geometry.camera.pixel2cam(
            depth=depth, intrinsics_inv=intrinsics_inv, pixel_coords=pixel_coords_input
        )

        assert output.shape == (batch_size, H, W, 3)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_consistency(self, batch_size, device, dtype):
        H, W = 250, 500
        fx, fy = W, H
        cx, cy = W / 2, H / 2
        eps = 1e-12
        seed = 77
        low_1, high_1 = -500, 500
        low_2, high_2 = -(max(W, H) * 3), (max(W, H) * 3)

        torch.manual_seed(seed)
        depth = self._get_samples((batch_size, 1, H, W), low_1, high_1, device, dtype)
        pixel_coords = self._get_samples((batch_size, H, W, 2), low_2, high_2, device, dtype)

        last_ch = torch.ones((batch_size, H, W, 1), device=device, dtype=dtype)
        pixel_coords_input = torch.cat([pixel_coords, last_ch], axis=-1)

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        intrinsics_inv = self._create_intrinsics_inv(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)

        cam_coords = kornia.geometry.camera.pixel2cam(
            depth=depth, intrinsics_inv=intrinsics_inv, pixel_coords=pixel_coords_input
        )

        # Setting the projection matrix to the intrinsic matrix for
        # simplicity (i.e. assuming that the RT matrix is an identity matrix)
        proj_mat = intrinsics
        pixel_coords_output = kornia.geometry.camera.cam2pixel(
            cam_coords_src=cam_coords, dst_proj_src=proj_mat, eps=eps
        )
        pixel_coords_concat = torch.cat([pixel_coords_output, last_ch], axis=-1)

        assert_close(pixel_coords_concat, pixel_coords_input, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", (1,))
    def test_gradcheck(self, batch_size, device, dtype):
        H, W = 10, 20
        fx, fy = W, H
        cx, cy = W / 2, H / 2
        seed = 77
        low_1, high_1 = -500, 500
        low_2, high_2 = -(max(W, H) * 3), (max(W, H) * 3)

        torch.manual_seed(seed)
        depth = self._get_samples((batch_size, 1, H, W), low_1, high_1, device, dtype)
        pixel_coords = self._get_samples((batch_size, H, W, 2), low_2, high_2, device, dtype)

        last_ch = torch.ones((batch_size, H, W, 1), device=device, dtype=dtype)
        pixel_coords_input = torch.cat([pixel_coords, last_ch], axis=-1)

        # If contiguous() is not called, gradcheck fails
        intrinsics_inv = self._create_intrinsics_inv(
            batch_size, fx, fy, cx, cy, device=device, dtype=dtype
        ).contiguous()

        depth = tensor_to_gradcheck_var(depth)
        intrinsics_inv = tensor_to_gradcheck_var(intrinsics_inv)
        pixel_coords_input = tensor_to_gradcheck_var(pixel_coords_input)

        assert gradcheck(
            kornia.geometry.camera.pixel2cam, (depth, intrinsics_inv, pixel_coords_input), raise_exception=True
        )


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
        pinhole = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)
        assert isinstance(pinhole, kornia.geometry.camera.PinholeCamera)

    def test_pinhole_camera_attributes(self, device, dtype):
        batch_size = 1
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        tx, ty, tz = 1, 2, 3

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        extrinsics = self._create_extrinsics(batch_size, tx, ty, tz, device=device, dtype=dtype)
        height = torch.ones(batch_size, device=device, dtype=dtype) * height
        width = torch.ones(batch_size, device=device, dtype=dtype) * width

        pinhole = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)

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

        pinhole = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)

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

        pinhole = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)

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

        pinhole = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)
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

        pinhole = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)
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

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math

import pytest
import torch

import kornia
from kornia.geometry.camera.pinhole import (
    PinholeCamerasList,
    get_optical_pose_base,
    homography_i_H_ref,
    inverse_pinhole_matrix,
    pinhole_matrix,
)

from testing.base import BaseTester


def _k44(device, dtype, fx=100.0, fy=100.0, cx=4.0, cy=3.0):
    """Build a (1, 4, 4) intrinsics matrix in ``PinholeCamera``'s layout.

    ``cx != cy`` on purpose so a transposed reading of the matrix changes every literal below. Built from a
    nested list rather than by item assignment into ``torch.eye`` so the signs survive on ``mps``.
    """
    return torch.tensor(
        [[[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    )


def _e44(device, dtype, tx=0.0, ty=0.0, tz=0.0):
    """Build a (1, 4, 4) extrinsics matrix with rotation ``I`` and translation ``(tx, ty, tz)``.

    An identity pose is frame-invariant, so every frame/direction pin below passes a non-zero translation.
    """
    return torch.tensor(
        [[[1.0, 0.0, 0.0, tx], [0.0, 1.0, 0.0, ty], [0.0, 0.0, 1.0, tz], [0.0, 0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    )


class TestCam2Pixel(BaseTester):
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
        # Use normalized image-plane coords so that projected pixel values stay in [0,W) x [0,H).
        # cam_x/z in [-0.5, 0.5] gives pixel_x in [cx - fx/2, cx + fx/2] = [0, W).
        low_norm, high_norm = -0.45, 0.45
        low_z, high_z = 1.0, 500.0

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        intrinsics_inv = self._create_intrinsics_inv(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)

        # Setting the projection matrix to the intrinsic matrix for
        # simplicity (i.e. assuming that the RT matrix is an identity matrix)
        proj_mat = intrinsics

        torch.manual_seed(seed)
        # Generate z first, then x,y as z * normalized_coord so pixel coords stay in image bounds
        cam_coords_z = self._get_samples((batch_size, H, W, 1), low_z, high_z, device, dtype)
        cam_coords_xy = self._get_samples((batch_size, H, W, 2), low_norm, high_norm, device, dtype) * cam_coords_z
        cam_coords_input = torch.cat([cam_coords_xy, cam_coords_z], dim=-1)

        pixel_coords_output = kornia.geometry.camera.cam2pixel(
            cam_coords_src=cam_coords_input, dst_proj_src=proj_mat, eps=eps
        )

        last_ch = torch.ones((batch_size, H, W, 1), device=device, dtype=dtype)
        pixel_coords_concat = torch.cat([pixel_coords_output, last_ch], axis=-1)

        depth = cam_coords_input[..., 2:3].permute(0, 3, 1, 2).contiguous()
        cam_coords_output = kornia.geometry.camera.pixel2cam(
            depth=depth, intrinsics_inv=intrinsics_inv, pixel_coords=pixel_coords_concat
        )

        self.assert_close(cam_coords_output, cam_coords_input, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", (1,))
    def test_gradcheck(self, batch_size, device):
        dtype = torch.float64
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

        self.gradcheck(kornia.geometry.camera.cam2pixel, (cam_coords_src, proj_mat, eps), atol=atol, rtol=rtol)

    def test_wart_cam2pixel_epsilon_makes_the_singular_divide_finite_4267(self, device, dtype):
        # Wart pin for kornia#4267: cam2pixel divides by ``z + 1e-12`` instead
        # of guarding the singularity, so the camera-plane point (1, 2, 0) yields a finite 1e14-scale pixel rather
        # than inf (project_points_z1) or [[104, 203]] (project_points). The epsilon enters the arithmetic, so it
        # also biases every finite result below z ~ 1e-10.
        # Snippet used to generate expected: cam2pixel([[[[1., 2., 0.]]]], _k44(...)) executed 2026-09-05
        # (torch 2.14.0, cpu and mps) -> float32 [[[[1.00000000376832e14, 2.00000000753664e14]]]].
        # Pins the CURRENT value; NOT a contract; delete when #4267 is repaired.
        cam_coords = torch.tensor([[[[1.0, 2.0, 0.0], [1.0, 2.0, 1.0e-12]]]], device=device, dtype=dtype)
        uv = kornia.geometry.camera.cam2pixel(cam_coords, _k44(device, dtype))
        if dtype == torch.float16:
            assert bool(torch.isposinf(uv).all())
        else:
            assert bool(torch.isfinite(uv).all())
            # At z = eps the additive denominator halves the result; a guarded divide would not.
            expected = torch.tensor([[[[1.0e14, 2.0e14], [5.0e13, 1.0e14]]]], device=device, dtype=dtype)
            self.assert_close(uv, expected)

    @pytest.mark.parametrize("shape", [(), (4, 4), (3, 3), (1, 3, 3), (1, 3, 4), (1, 5, 4), (1, 1, 4, 4)])
    def test_invalid_projection_shape_4266(self, shape, device, dtype):
        cam_coords = torch.tensor([[[[1.0, 2.0, 4.0]]]], device=device, dtype=dtype)
        projection = torch.ones(shape, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Input dst_proj_src has to be in the shape of Bx4x4"):
            kornia.geometry.camera.cam2pixel(cam_coords, projection)

    @pytest.mark.parametrize("shape", [(), (3,), (1, 3), (1, 2, 3), (1, 2, 3, 2), (1, 2, 3, 4), (1, 2, 3, 1, 3)])
    def test_invalid_coordinate_shape_4266(self, shape, device, dtype):
        cam_coords = torch.ones(shape, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Input cam_coords_src has to be in the shape of BxHxWx3"):
            kornia.geometry.camera.cam2pixel(cam_coords, _k44(device, dtype))


class TestPixel2Cam(BaseTester):
    @pytest.mark.parametrize("shape", [(), (3,), (1, 3), (1, 2, 3), (1, 2, 3, 2), (1, 2, 3, 4), (1, 2, 3, 1, 3)])
    def test_invalid_coordinate_shape_4266(self, shape, device, dtype):
        depth = torch.ones(1, 1, 2, 3, device=device, dtype=dtype)
        pixel_coords = torch.ones(shape, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Input pixel_coords has to be in the shape of BxHxWx3") as exc_info:
            kornia.geometry.camera.pixel2cam(depth, _k44(device, dtype), pixel_coords)
        assert str(pixel_coords.shape) in str(exc_info.value)

    @pytest.mark.parametrize("depth_shape", [(), (2,), (2, 1, 3), (2, 2, 3, 4), (2, 3, 3, 4), (2, 1, 3, 4, 1)])
    def test_invalid_depth_shape(self, depth_shape, device, dtype):
        depth = torch.ones(depth_shape, device=device, dtype=dtype)
        intrinsics_inv = torch.eye(4, device=device, dtype=dtype).repeat(2, 1, 1)
        pixel_coords = torch.ones(2, 3, 4, 3, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Input depth has to be in the shape of Bx1xHxW"):
            kornia.geometry.camera.pixel2cam(depth, intrinsics_inv, pixel_coords)

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
        # Depth must be positive and bounded away from zero to avoid 1/z blow-up.
        # Pixel coords restricted to image bounds [0,W) x [0,H) to avoid TF32 precision issues
        # from large coordinate values in matrix multiplication.
        low_1, high_1 = 1.0, 500.0
        low_2x, high_2x = 0.0, float(W)
        low_2y, high_2y = 0.0, float(H)

        torch.manual_seed(seed)
        depth = self._get_samples((batch_size, 1, H, W), low_1, high_1, device, dtype)
        pixel_coords_x = self._get_samples((batch_size, H, W, 1), low_2x, high_2x, device, dtype)
        pixel_coords_y = self._get_samples((batch_size, H, W, 1), low_2y, high_2y, device, dtype)
        pixel_coords = torch.cat([pixel_coords_x, pixel_coords_y], dim=-1)

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

        self.assert_close(pixel_coords_concat, pixel_coords_input, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", (1,))
    @pytest.mark.slow
    def test_gradcheck(self, batch_size, device):
        dtype = torch.float64
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

        self.gradcheck(kornia.geometry.camera.pixel2cam, (depth, intrinsics_inv, pixel_coords_input), fast_mode=False)

    @pytest.mark.parametrize(
        "intrinsics_shape",
        [(), (4,), (4, 4), (1, 4, 4, 1), (1, 3, 3), (1, 2, 2), (1, 5, 5), (1, 3, 4), (1, 4, 3), (1, 5, 4)],
    )
    def test_invalid_intrinsics_shape_4266(self, intrinsics_shape, device, dtype):
        # A rank-only guard lets non-square matrices return the wrong number of coordinate components.
        depth = torch.ones(1, 1, 2, 3, device=device, dtype=dtype)
        pixel_coords = torch.ones(1, 2, 3, 3, device=device, dtype=dtype)
        intrinsics_inv = torch.ones(intrinsics_shape, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Input intrinsics_inv has to be in the shape of Bx4x4") as exc_info:
            kornia.geometry.camera.pixel2cam(depth, intrinsics_inv, pixel_coords)

        assert str(intrinsics_inv.shape) in str(exc_info.value)

    @pytest.mark.parametrize("intrinsics_batch, points_batch", [(1, 1), (2, 2), (1, 2)])
    def test_intrinsics_batch_broadcast(self, intrinsics_batch, points_batch, device, dtype):
        # fx=2, fy=4, cx=4, cy=3: pixel (6, 11) at depth 2 maps to camera point (2, 4, 2).
        intrinsics_inv = torch.tensor(
            [[[0.5, 0.0, -2.0, 0.0], [0.0, 0.25, -0.75, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).expand(intrinsics_batch, -1, -1)
        depth = torch.full((points_batch, 1, 1, 1), 2.0, device=device, dtype=dtype)
        pixel_coords = torch.tensor([[[[6.0, 11.0, 1.0]]]], device=device, dtype=dtype).expand(points_batch, -1, -1, -1)

        actual = kornia.geometry.camera.pixel2cam(depth, intrinsics_inv, pixel_coords)

        expected = torch.tensor([[[[2.0, 4.0, 2.0]]]], device=device, dtype=dtype).expand(points_batch, -1, -1, -1)
        self.assert_close(actual, expected)

    def test_convention_pixel2cam_rejects_multi_channel_depth_4266(self, device, dtype):
        # Regression pin for kornia#4266: multi-channel depth must be rejected instead of scaling
        # each camera coordinate by a different channel. Single-channel depth scales the whole ray.
        # An exact inverse of fx = fy = 1, cx = 4, cy = 3, so every literal below is exact in every dtype.
        intrinsics_inv = torch.tensor(
            [[[1.0, 0.0, -4.0, 0.0], [0.0, 1.0, -3.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        # (1, H = 3, W = 4, 3) grid of (u, v, 1): u = 5 + col, v = 5 + row, so the camera point is
        # (1 + col, 2 + row, 1).
        pixel_coords = torch.tensor(
            [
                [
                    [[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [7.0, 5.0, 1.0], [8.0, 5.0, 1.0]],
                    [[5.0, 6.0, 1.0], [6.0, 6.0, 1.0], [7.0, 6.0, 1.0], [8.0, 6.0, 1.0]],
                    [[5.0, 7.0, 1.0], [6.0, 7.0, 1.0], [7.0, 7.0, 1.0], [8.0, 7.0, 1.0]],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        depth_3ch = torch.tensor([2.0, 3.0, 5.0], device=device, dtype=dtype).reshape(1, 3, 1, 1).expand(1, 3, 3, 4)
        with pytest.raises(ValueError, match="Input depth has to be in the shape of Bx1xHxW"):
            kornia.geometry.camera.pixel2cam(depth_3ch.contiguous(), intrinsics_inv, pixel_coords)
        depth_1ch = torch.full((1, 1, 3, 4), 2.0, device=device, dtype=dtype)
        out_1ch = kornia.geometry.camera.pixel2cam(depth_1ch, intrinsics_inv, pixel_coords)
        self.assert_close(
            out_1ch[0, 2, 3], torch.tensor([8.0, 8.0, 2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )
        depth_2ch = torch.tensor([2.0, 3.0], device=device, dtype=dtype).reshape(1, 2, 1, 1).expand(1, 2, 3, 4)
        with pytest.raises(ValueError, match="Input depth has to be in the shape of Bx1xHxW"):
            kornia.geometry.camera.pixel2cam(depth_2ch.contiguous(), intrinsics_inv, pixel_coords)


class TestPinholeCamera(BaseTester):
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

    def _create_extrinsics_with_rotation(self, batch_size, alpha, beta, gamma, tx, ty, tz, device, dtype):
        Rx = torch.eye(3, device=device, dtype=dtype)
        Rx[1, 1] = math.cos(alpha)
        Rx[1, 2] = math.sin(alpha)
        Rx[2, 1] = -Rx[1, 2]
        Rx[2, 2] = Rx[1, 1]

        Ry = torch.eye(3, device=device, dtype=dtype)
        Ry[0, 0] = math.cos(beta)
        Ry[0, 2] = -math.sin(beta)
        Ry[2, 0] = -Ry[0, 2]
        Ry[2, 2] = Ry[0, 0]

        Rz = torch.eye(3, device=device, dtype=dtype)
        Rz[0, 0] = math.cos(gamma)
        Rz[0, 1] = math.sin(gamma)
        Rz[1, 0] = -Rz[0, 1]
        Rz[1, 1] = Rz[0, 0]

        Ryz = torch.matmul(Ry, Rz)
        R = torch.matmul(Rx, Ryz)

        extrinsics = torch.eye(4, device=device, dtype=dtype)
        extrinsics[..., 0, -1] = tx
        extrinsics[..., 1, -1] = ty
        extrinsics[..., 2, -1] = tz
        extrinsics[:3, :3] = R

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

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_from_parameters(self, batch_size, device, dtype):
        height, width = 6, 8
        fx = torch.arange(1, batch_size + 1, device=device, dtype=dtype) * 100.0
        fy = fx / 2.0
        cx = torch.full((batch_size,), width / 2, device=device, dtype=dtype)
        cy = torch.full((batch_size,), height / 2, device=device, dtype=dtype)
        tx = torch.arange(batch_size, device=device, dtype=dtype)
        ty, tz = tx + 1.0, tx + 2.0

        pinhole = kornia.geometry.camera.PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx, ty, tz, batch_size, device=device, dtype=dtype
        )

        assert pinhole.batch_size == batch_size
        # ``height`` and ``width`` are broadcast over the whole batch, like every other parameter
        self.assert_close(pinhole.height, torch.full((batch_size,), float(height), device=device, dtype=dtype))
        self.assert_close(pinhole.width, torch.full((batch_size,), float(width), device=device, dtype=dtype))
        self.assert_close(pinhole.fx, fx)
        self.assert_close(pinhole.fy, fy)
        self.assert_close(pinhole.cx, cx)
        self.assert_close(pinhole.cy, cy)
        self.assert_close(pinhole.tx, tx)
        self.assert_close(pinhole.ty, ty)
        self.assert_close(pinhole.tz, tz)

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

        self.assert_close(
            pinhole_scale.intrinsics[..., 0, 0], pinhole.intrinsics[..., 0, 0] * scale_val, atol=1e-4, rtol=1e-4
        )  # fx
        self.assert_close(
            pinhole_scale.intrinsics[..., 1, 1], pinhole.intrinsics[..., 1, 1] * scale_val, atol=1e-4, rtol=1e-4
        )  # fy
        self.assert_close(
            pinhole_scale.intrinsics[..., 0, 2], pinhole.intrinsics[..., 0, 2] * scale_val, atol=1e-4, rtol=1e-4
        )  # cx
        self.assert_close(
            pinhole_scale.intrinsics[..., 1, 2], pinhole.intrinsics[..., 1, 2] * scale_val, atol=1e-4, rtol=1e-4
        )  # cy
        self.assert_close(pinhole_scale.height, pinhole.height * scale_val, atol=1e-4, rtol=1e-4)
        self.assert_close(pinhole_scale.width, pinhole.width * scale_val, atol=1e-4, rtol=1e-4)

    def test_pinhole_camera_scale_does_not_alias_the_source(self, device, dtype):
        """scale() returns a new camera, so writing to it must not reach the source.

        The intrinsics were already cloned; the extrinsics were handed over by
        reference, and the constructor stores what it is given. Setting tx on
        the scaled camera therefore moved the source camera too.
        """
        batch_size = 2
        height, width = 4, 6
        intrinsics = self._create_intrinsics(batch_size, 1, 2, width / 2, height / 2, device=device, dtype=dtype)
        extrinsics = self._create_extrinsics(batch_size, 1, 2, 3, device=device, dtype=dtype)
        height_t = torch.ones(batch_size, device=device, dtype=dtype) * height
        width_t = torch.ones(batch_size, device=device, dtype=dtype) * width
        scale_factor = torch.ones(batch_size, device=device, dtype=dtype) * 2.0

        pinhole = kornia.geometry.camera.PinholeCamera(intrinsics.clone(), extrinsics.clone(), height_t, width_t)
        tx_before = pinhole.tx.clone()

        pinhole_scale = pinhole.scale(scale_factor)
        assert pinhole_scale.extrinsics is not pinhole.extrinsics
        assert pinhole_scale.extrinsics.data_ptr() != pinhole.extrinsics.data_ptr()

        pinhole_scale.tx = 7.0
        self.assert_close(pinhole.tx, tx_before)

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

        self.assert_close(
            pinhole_scale.intrinsics[..., 0, 0], pinhole.intrinsics[..., 0, 0] * scale_val, atol=1e-4, rtol=1e-4
        )  # fx
        self.assert_close(
            pinhole_scale.intrinsics[..., 1, 1], pinhole.intrinsics[..., 1, 1] * scale_val, atol=1e-4, rtol=1e-4
        )  # fy
        self.assert_close(
            pinhole_scale.intrinsics[..., 0, 2], pinhole.intrinsics[..., 0, 2] * scale_val, atol=1e-4, rtol=1e-4
        )  # cx
        self.assert_close(
            pinhole_scale.intrinsics[..., 1, 2], pinhole.intrinsics[..., 1, 2] * scale_val, atol=1e-4, rtol=1e-4
        )  # cy
        self.assert_close(pinhole_scale.height, pinhole.height * scale_val, atol=1e-4, rtol=1e-4)
        self.assert_close(pinhole_scale.width, pinhole.width * scale_val, atol=1e-4, rtol=1e-4)

    def test_pinhole_camera_project_and_unproject(self, device, dtype):
        batch_size = 5
        n = 2  # Point per batch
        height, width = 4, 6
        fx, fy, cx, cy = 1, 2, width / 2, height / 2
        alpha, beta, gamma = 0.0, 0.0, 0.4
        tx, ty, tz = 0, 0, 3

        intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
        extrinsics = self._create_extrinsics_with_rotation(
            batch_size, alpha, beta, gamma, tx, ty, tz, device=device, dtype=dtype
        )

        height = torch.ones(batch_size, device=device, dtype=dtype) * height
        width = torch.ones(batch_size, device=device, dtype=dtype) * width

        pinhole = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)

        point_3d = torch.rand((batch_size, n, 3), device=device, dtype=dtype)

        depth = point_3d[..., -1:] + tz

        point_2d = pinhole.project(point_3d)
        point_3d_hat = pinhole.unproject(point_2d, depth)
        self.assert_close(point_3d, point_3d_hat, atol=1e-4, rtol=1e-4)

    def test_pinhole_camera_device(self, device, dtype):
        batch_size = 5
        intrinsics = torch.rand((batch_size, 4, 4), device=device, dtype=dtype)
        extrinsics = torch.rand((batch_size, 4, 4), device=device, dtype=dtype)
        height = torch.randint(low=5, high=9, size=(batch_size,), device=device)
        width = torch.randint(low=5, high=9, size=(batch_size,), device=device)

        pinhole = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)
        assert pinhole.device() == intrinsics.device

    def test_convention_extrinsics_are_world_to_camera(self, device, dtype):
        # Convention pin: project() computes K (R X + t) -- the extrinsics map WORLD points INTO the camera frame
        # (OpenCV / COLMAP semantics). With R = I and t = (1, 0, 0) the world point (1, 2, 4) becomes (2, 2, 4) in
        # the camera frame and projects to u = 100*2/4 + 4 = 54, v = 100*2/4 + 3 = 53. A cam-to-world reading would
        # move the point to (0, 2, 4) and give (4, 53); cx != cy also breaks a transposed reading.
        # Snippet used to generate expected: hand arithmetic, re-executed 2026-09-05 (torch 2.14.0, cpu and mps,
        # every dtype): [[54., 53.]].
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype),
            _e44(device, dtype, tx=1.0),
            torch.tensor([6], device=device),
            torch.tensor([8], device=device),
        )
        uv = cam.project(torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype))
        self.assert_close(uv, torch.tensor([[54.0, 53.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_convention_unproject_returns_the_world_point(self, device, dtype):
        # Convention pin: unproject(uv, depth) inverts project() back into the WORLD frame (it applies the inverse
        # of K @ E), and ``depth`` is the CAMERA-frame z, not the ray length. Same tx = 1 extrinsics as above, so a
        # camera-frame result (2, 2, 4) fails this pin. Contrast the free function unproject_points, which takes no
        # extrinsics and therefore necessarily returns a camera-frame point.
        # Snippet used to generate expected: cam.unproject(cam.project(X), 4.) executed 2026-09-05 (torch 2.14.0)
        # -> float32 [[0.99999988, 1.9999999, 4.0]], max abs error 1.19e-07.
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype),
            _e44(device, dtype, tx=1.0),
            torch.tensor([6], device=device),
            torch.tensor([8], device=device),
        )
        X = torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype)
        back = cam.unproject(cam.project(X), torch.tensor([[4.0]], device=device, dtype=dtype))
        self.assert_close(back, X)

    def test_convention_clone_is_a_deep_copy(self, device, dtype):
        # Convention pin: clone() is the ONLY deep copy on PinholeCamera -- a new
        # object, new intrinsics and extrinsics tensors with different storage, and mutating the clone leaves the
        # source untouched. scale() also returns a camera with its own tensors; scale_() and the tx / ty / tz
        # setters are the ones that still write through to the caller (kornia#4264).
        # Snippet used to generate expected: build a tx = 2 camera, clone it, set clone.tx = 9; executed
        # 2026-09-05 (torch 2.14.0, every dtype) -> source tx [2.0], clone tx [9.0].
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype),
            _e44(device, dtype, tx=2.0),
            torch.tensor([6.0], device=device, dtype=dtype),
            torch.tensor([8.0], device=device, dtype=dtype),
        )
        cloned = cam.clone()
        assert cloned is not cam
        assert cloned.intrinsics is not cam.intrinsics
        assert cloned.extrinsics is not cam.extrinsics
        assert cloned.intrinsics.data_ptr() != cam.intrinsics.data_ptr()
        assert cloned.extrinsics.data_ptr() != cam.extrinsics.data_ptr()
        cloned.tx = torch.tensor([9.0], device=device, dtype=dtype)
        self.assert_close(cam.tx, torch.tensor([2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(cloned.tx, torch.tensor([9.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_convention_intrinsics_inverse_is_the_exact_inverse(self, device, dtype):
        # Convention pin: intrinsics_inverse() @ intrinsics is byte-exact eye(4) on an
        # asymmetric fx = 100, fy = 50, cx = 4, cy = 3 intrinsics -- this is the pair DepthWarper feeds pixel2cam.
        # The legacy 12-vector twins are NOT exact: inverse_pinhole_matrix @ pinhole_matrix is 1e-06 off
        # (kornia#4268, pinned in TestPinholeMatrix below).
        # Snippet used to generate expected: torch.equal(inv @ K, eye(4)[None]) executed 2026-09-05 (torch 2.14.0)
        # -> True for float32/float64/bfloat16 on cpu and float32 on mps.
        if dtype == torch.float16:
            pytest.skip("float16 cannot round-trip 1/fx: inv @ K differs from eye(4) by 1.5e-05 (2026-09-05)")
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype, fy=50.0),
            _e44(device, dtype, tx=1.0),
            torch.tensor([6.0], device=device, dtype=dtype),
            torch.tensor([8.0], device=device, dtype=dtype),
        )
        product = cam.intrinsics_inverse() @ cam.intrinsics
        self.assert_close(product, torch.eye(4, device=device, dtype=dtype)[None], atol=0.0, rtol=0.0)

    def test_wart_scale_rescales_the_principal_point_by_the_half_pixel_rule_4263(self, device, dtype):
        # Wart pin for kornia#4263: scale(s) gives cx' = s * cx (2.0 for cx = 4,
        # s = 0.5) -- the half-pixel / COLMAP convention -- although create_meshgrid and every unprojection path in
        # the library enumerate integer pixel CENTRES, under which the grid-consistent value is
        # cx' = s * cx + (s - 1) / 2 = 1.75. Pins the CURRENT value so the window's repair flips this loudly.
        # Snippet used to generate expected: cam.scale(0.5) executed 2026-09-05 (torch 2.14.0, every dtype)
        # -> cx [2.0], cy [1.5], fx [50.0]. NOT a contract; delete when #4263 is repaired.
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype),
            _e44(device, dtype, tx=1.0),
            torch.tensor([6.0], device=device, dtype=dtype),
            torch.tensor([8.0], device=device, dtype=dtype),
        )
        scaled = cam.scale(torch.tensor([0.5], device=device, dtype=dtype))
        self.assert_close(scaled.cx, torch.tensor([2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(scaled.cy, torch.tensor([1.5], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(scaled.fx, torch.tensor([50.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_wart_constructor_and_scale_inplace_write_through_to_the_caller_4264(self, device, dtype):
        # Wart pin for kornia#4264: the class stores the tensors it is constructed from instead of copying
        # them, so every mutating accessor writes into the CALLER's tensors -- the constructor keeps the
        # caller's ``intrinsics`` / ``extrinsics`` objects, the ``tx`` setter writes into the caller's
        # extrinsics, and the in-place ``scale_`` rewrites the caller's intrinsics and image size.
        # The fourth leg of #4264 -- ``scale()`` handing ``self.extrinsics`` to the new camera by reference --
        # is repaired, and is pinned the other way up by
        # ``test_pinhole_camera_scale_does_not_alias_the_source``; it is deliberately not asserted here.
        # Snippet used to generate expected: cam.intrinsics is K and cam.extrinsics is E -> True; after
        # ``cam.tx = 5.0`` E[0, 0, 3] reads 5.0; after ``cam.scale_(0.5)`` K[0, 0, 2] reads 2.0 (from 4.0),
        # K[0, 0, 0] reads 50.0 and the caller's height/width read [3.0] / [4.0] (from [6.0] / [8.0]);
        # executed 2026-09-05 (torch 2.14.0, cpu and mps, every dtype).
        # Pins the CURRENT behavior; NOT a contract; delete when the rest of #4264 is repaired.
        # The constructor stores, rather than copies, all four arguments.
        K = _k44(device, dtype)
        E = _e44(device, dtype, tx=1.0)
        height = torch.tensor([6.0], device=device, dtype=dtype)
        width = torch.tensor([8.0], device=device, dtype=dtype)
        source = kornia.geometry.camera.PinholeCamera(K, E, height, width)
        assert source.intrinsics is K
        assert source.extrinsics is E
        # The tx setter writes into the caller's extrinsics tensor.
        source.tx = 5.0
        self.assert_close(E[0, 0, 3], torch.tensor(5.0, device=device, dtype=dtype), atol=0.0, rtol=0.0)
        # scale_ rewrites the caller's intrinsics and image-size tensors in place.
        source.scale_(torch.tensor([0.5], device=device, dtype=dtype))
        self.assert_close(K[0, 0, 2], torch.tensor(2.0, device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(K[0, 0, 0], torch.tensor(50.0, device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(height, torch.tensor([3.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(width, torch.tensor([4.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_wart_scale_inplace_rejects_integer_image_size_4265(self, device, dtype):
        # Wart pin for kornia#4265: the constructor accepts int64 height/width --
        # that is what the class docstring's own example builds -- and a floating factor promotes them, but the
        # in-place twin scale_() writes the float result back into the int64 storage and raises.
        # Snippet used to generate expected: cam.scale_(0.5) on an int64 height executed 2026-09-05 (torch 2.14.0,
        # every dtype) -> RuntimeError("result type Float can't be cast to the desired output type Long").
        # Pins the CURRENT behavior; NOT a contract; delete when #4265 is repaired.
        K = _k44(device, dtype)
        cam = kornia.geometry.camera.PinholeCamera(
            K,
            _e44(device, dtype, tx=1.0),
            torch.tensor([6], device=device),
            torch.tensor([8], device=device),
        )
        for factor in (2, torch.tensor([2], device=device)):
            scaled = cam.scale(factor)
            assert scaled.height.dtype == scaled.width.dtype == torch.int64
            self.assert_close(scaled.height, torch.tensor([12], device=device))
            self.assert_close(scaled.width, torch.tensor([16], device=device))
            inplace = cam.clone()
            assert inplace.scale_(factor) is inplace
            self.assert_close(inplace.height, scaled.height)
            self.assert_close(inplace.width, scaled.width)
            self.assert_close(inplace.intrinsics, scaled.intrinsics)
        assert cam.scale(torch.tensor([0.5], device=device, dtype=dtype)).height.is_floating_point()
        with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
            cam.scale_(0.5)
        expected_K = _k44(device, dtype)
        expected_K[:, :2, :3] *= 0.5
        self.assert_close(K, expected_K, atol=0.0, rtol=0.0)
        self.assert_close(cam.intrinsics, expected_K, atol=0.0, rtol=0.0)
        self.assert_close(cam.height, torch.tensor([6], device=device))
        self.assert_close(cam.width, torch.tensor([8], device=device))

    @pytest.mark.parametrize("shape", [(4, 4), (1, 3, 3), (1, 3, 4), (1, 5, 5), (1, 2, 3, 3), (1, 1, 1, 4, 4)])
    @pytest.mark.parametrize("parameter", ["intrinsics", "extrinsics"])
    def test_invalid_parameter_shape_4266(self, shape, parameter, device, dtype):
        batch_size = shape[0]
        params = {
            "intrinsics": _k44(device, dtype).expand(batch_size, -1, -1),
            "extrinsics": _e44(device, dtype).expand(batch_size, -1, -1),
            "height": torch.full((batch_size,), 6, device=device, dtype=dtype),
            "width": torch.full((batch_size,), 8, device=device, dtype=dtype),
        }
        params[parameter] = torch.ones(shape, device=device, dtype=dtype)
        with pytest.raises(ValueError, match=f"Argument {parameter} shape must be"):
            kornia.geometry.camera.PinholeCamera(**params)

    def test_camera_list_preserves_rank_four_parameters(self, device, dtype):
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype), _e44(device, dtype), torch.tensor([6], device=device), torch.tensor([8], device=device)
        )
        cameras = PinholeCamerasList([cam, cam.clone()])
        assert cameras.intrinsics.shape == (1, 2, 4, 4)
        assert cameras.extrinsics.shape == (1, 2, 4, 4)
        assert cameras.num_cameras == 2
        points = torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype)
        self.assert_close(cameras.get_pinhole(1).project(points), cam.project(points))

    def test_project_and_unproject_reject_rank_one_points_4266(self, device, dtype):
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype),
            _e44(device, dtype, tx=1.0),
            torch.tensor([6], device=device),
            torch.tensor([8], device=device),
        )
        point = torch.tensor([1.0, 2.0, 4.0], device=device, dtype=dtype)
        with pytest.raises(ValueError) as cam_error:
            cam.project(point)
        with pytest.raises(ValueError) as function_error:
            kornia.geometry.camera.project_points(point, _k44(device, dtype)[:, :3, :3].contiguous())
        assert (
            str(cam_error.value)
            == str(function_error.value)
            == "Input must be at least a 2D tensor. Got torch.Size([3])"
        )

        point_2d = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        with pytest.raises(ValueError, match=r"Input must be at least a 2D tensor\. Got torch\.Size\(\[2\]\)"):
            cam.unproject(point_2d, torch.tensor([4.0], device=device, dtype=dtype))

    def test_constructor_accepts_an_empty_batch_4281(self, device, dtype):
        # Regression for kornia#4281: a consistent empty camera batch is valid.
        intrinsics = torch.zeros(0, 4, 4, device=device, dtype=dtype)
        extrinsics = torch.zeros(0, 4, 4, device=device, dtype=dtype)
        height = torch.zeros(0, device=device, dtype=dtype)
        width = torch.zeros(0, device=device, dtype=dtype)

        camera = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)

        assert camera.batch_size == 0
        assert camera.intrinsics.shape == (0, 4, 4)
        assert camera.extrinsics.shape == (0, 4, 4)
        assert camera.height.shape == (0,)
        assert camera.width.shape == (0,)
        for tensor in (camera.intrinsics, camera.extrinsics, camera.height, camera.width):
            assert tensor.dtype == dtype
            assert tensor.device == device

        # Free functions on the same surface follow the same empty-in/empty-out convention.
        empty = kornia.geometry.camera.project_points(
            torch.zeros(0, 1, 3, device=device, dtype=dtype), _k44(device, dtype)[:, :3, :3].contiguous()
        )
        assert empty.shape == (0, 1, 2)

    def test_project_an_empty_batch_with_a_point_axis_4466(self, device, dtype):
        # Regression for kornia#4466: an empty camera batch projected (0, 3) points but raised
        # ZeroDivisionError on (0, N, 3), inside transform_points.
        trans = torch.eye(4, device=device, dtype=dtype).expand(0, 4, 4)
        empty = torch.zeros(0, device=device, dtype=dtype)
        camera = kornia.geometry.camera.PinholeCamera(trans, trans, empty, empty)

        assert camera.project(torch.zeros(0, 3, device=device, dtype=dtype)).shape == (0, 2)
        assert camera.project(torch.zeros(0, 1, 3, device=device, dtype=dtype)).shape == (0, 1, 2)

    @pytest.mark.parametrize("batch_sizes", [(1, 2, 1, 1), (0, 1, 0, 0)])
    def test_constructor_rejects_mismatched_batch_sizes_4281(self, batch_sizes, device, dtype):
        with pytest.raises(ValueError, match="Arguments shapes must match"):
            kornia.geometry.camera.PinholeCamera(
                torch.zeros(batch_sizes[0], 4, 4, device=device, dtype=dtype),
                torch.zeros(batch_sizes[1], 4, 4, device=device, dtype=dtype),
                torch.zeros(batch_sizes[2], device=device, dtype=dtype),
                torch.zeros(batch_sizes[3], device=device, dtype=dtype),
            )

    def test_wart_project_and_project_points_disagree_at_z_zero_4267(self, device, dtype):
        # Wart pin for kornia#4267: PinholeCamera.project and the free function
        # project_points are documented as the same projection and agree exactly away from the singularity, but
        # they apply K on OPPOSITE sides of the masked |z| <= 1e-8 divide, so at z = 0 they return different
        # answers -- K @ [1, 2, 0] = [100, 200] for the method and fx*x + cx = [104, 203] for the function.
        # The divergence is the defect, not either value.
        # Snippet used to generate expected: both calls at (1, 2, 0) and (1, 2, 4) executed 2026-09-05
        # (torch 2.14.0, cpu and mps, every dtype). NOT a contract; delete when #4267 is repaired.
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype),
            _e44(device, dtype),
            torch.tensor([6], device=device),
            torch.tensor([8], device=device),
        )
        K3 = _k44(device, dtype)[:, :3, :3].contiguous()
        singular = torch.tensor([[1.0, 2.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(
            cam.project(singular), torch.tensor([[100.0, 200.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )
        self.assert_close(
            kornia.geometry.camera.project_points(singular, K3),
            torch.tensor([[104.0, 203.0]], device=device, dtype=dtype),
            atol=0.0,
            rtol=0.0,
        )
        regular = torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype)
        self.assert_close(cam.project(regular), kornia.geometry.camera.project_points(regular, K3))

    def test_convention_from_parameters_fills_every_batch_element_4279(self, device, dtype):
        # Regression pin for #4279: image size must be filled for every camera in the batch.
        cam = self._from_parameters_batch2(device, dtype)
        self.assert_close(cam.height, torch.tensor([6.0, 6.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(cam.width, torch.tensor([8.0, 8.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(cam.fx, torch.tensor([100.0, 200.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(cam.tx, torch.tensor([1.0, 2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def _from_parameters_batch2(self, device, dtype):
        # Asymmetric per-element parameters (fx 100/200, fy 100/50, cx 4/6, cy 3/2, tx 1/2) so a pin that reads the
        # wrong batch element, or transposes fx and fy, changes the literal.
        return kornia.geometry.camera.PinholeCamera.from_parameters(
            fx=torch.tensor([100.0, 200.0], device=device, dtype=dtype),
            fy=torch.tensor([100.0, 50.0], device=device, dtype=dtype),
            cx=torch.tensor([4.0, 6.0], device=device, dtype=dtype),
            cy=torch.tensor([3.0, 2.0], device=device, dtype=dtype),
            height=6,
            width=8,
            tx=torch.tensor([1.0, 2.0], device=device, dtype=dtype),
            ty=torch.tensor([0.0, 0.0], device=device, dtype=dtype),
            tz=torch.tensor([0.0, 0.0], device=device, dtype=dtype),
            batch_size=2,
            device=device,
            dtype=dtype,
        )


class TestPinholeMatrix(BaseTester):
    """Pins for the legacy 12-vector pinhole API in ``kornia.geometry.camera.pinhole``.

    None of these names is in ``kornia.geometry.camera.__all__`` or reachable as ``kornia.geometry.camera.X``,
    so they are imported from the module directly.
    """

    def _vec12(self, device, dtype):
        # (fx, fy, cx, cy, height, width, rx, ry, rz, tx, ty, tz); cx != cy and height != width so a swapped
        # reading of the vector changes the matrix.
        return torch.tensor(
            [[100.0, 100.0, 4.0, 3.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype
        )

    def test_wart_pinhole_matrix_perturbs_every_entry_4268(self, device, dtype):
        # Wart pin for kornia#4268: pinhole_matrix
        # adds its eps to the WHOLE identity before writing the parameters, so every entry is perturbed -- the
        # structural zero at [0, 0, 1] is 1e-06 and the structural one at [0, 3, 3] is 1.000001. Passing eps=0.0
        # gives the exact matrix, so the default is the only thing wrong. inverse_pinhole_matrix divides by
        # fx + eps, so the legacy pair is never an exact inverse (contrast PinholeCamera.intrinsics_inverse, which
        # is byte-exact). A (1, 4, 4) input raises a bare AssertionError carrying only a shape, not a ShapeError.
        # Snippet used to generate expected: pinhole_matrix(vec12) executed 2026-09-05 (torch 2.14.0) -> float32
        # [0, 0, 1] = 9.999999974752427e-07, [0, 3, 3] = 1.0000009536743164, inv[0, 0, 0] * 100 = 0.99999998.
        # Pins the CURRENT values; NOT a contract; delete when #4268 is repaired.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("half precision cannot hold 1 + 1e-06: [0, 3, 3] reads exactly 1.0 (executed 2026-09-05)")
        vec = self._vec12(device, dtype)
        matrix = pinhole_matrix(vec)
        assert matrix[0, 0, 1].item() != 0.0
        assert matrix[0, 3, 3].item() != 1.0
        self.assert_close(matrix[0, 0, 1], torch.tensor(1e-06, device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(matrix[0, 3, 3], torch.tensor(1.000001, device=device, dtype=dtype), atol=0.0, rtol=0.0)
        exact = pinhole_matrix(vec, eps=0.0)
        assert exact[0, 0, 1].item() == 0.0
        assert exact[0, 3, 3].item() == 1.0
        residual = (inverse_pinhole_matrix(vec) @ matrix - torch.eye(4, device=device, dtype=dtype)).abs().max()
        assert residual.item() > 0.0
        with pytest.raises(AssertionError, match=r"torch\.Size\(\[1, 4, 4\]\)"):
            inverse_pinhole_matrix(torch.eye(4, device=device, dtype=dtype)[None])

    def test_wart_dead_legacy_functions_always_raise_4283(self, device, dtype):
        # Wart pin for kornia#4283: get_optical_pose_base and homography_i_H_ref
        # carry full docstrings, Args/Returns blocks and a .. math:: block, validate their input, and then always
        # raise NotImplementedError -- get_optical_pose_base's dependency was removed from torchgeometry years ago
        # ("# TODO: where is rtvec_to_pose?"), and homography_i_H_ref is dead because it calls it.
        # Snippet used to generate expected: both calls on the documented (N, 12) vector executed 2026-09-05
        # (torch 2.14.0, every dtype) -> NotImplementedError('').
        # Pins the CURRENT behavior; NOT a contract; delete when #4283 is repaired.
        vec = self._vec12(device, dtype)
        with pytest.raises(NotImplementedError):
            get_optical_pose_base(vec)
        with pytest.raises(NotImplementedError):
            homography_i_H_ref(vec, vec)

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
        # Wart pin for kornia#4267 (audit labels 5a-pc2-07, 5a-pc2-15): cam2pixel divides by ``z + 1e-12`` instead
        # of guarding the singularity, so the camera-plane point (1, 2, 0) yields a finite 1e14-scale pixel rather
        # than inf (project_points_z1) or [[104, 203]] (project_points). The epsilon enters the arithmetic, so it
        # also biases every finite result below z ~ 1e-10.
        # Snippet used to generate expected: cam2pixel([[[[1., 2., 0.]]]], _k44(...)) executed 2026-09-05
        # (torch 2.14.0, cpu and mps) -> float32 [[[[1.00000000376832e14, 2.00000000753664e14]]]].
        # Pins the CURRENT value; NOT a contract; delete when #4267 is repaired.
        if dtype == torch.float16:
            pytest.skip("float16 overflows to inf at z = 0; that cell is pinned separately below")
        cam_coords = torch.tensor([[[[1.0, 2.0, 0.0]]]], device=device, dtype=dtype)
        uv = kornia.geometry.camera.cam2pixel(cam_coords, _k44(device, dtype))
        assert bool(torch.isfinite(uv).all())
        self.assert_close(uv, torch.tensor([[[[1.0e14, 2.0e14]]]], device=device, dtype=dtype))

    def test_wart_cam2pixel_overflows_to_inf_at_z_zero_in_float16_4267(self, device, dtype):
        # Wart pin for kornia#4267 (audit label 5a-pc2-08): the same ``z + 1e-12`` divide overflows float16's
        # 65504 range, so the float16 answer at z = 0 is inf while float32 is a finite 1e14 -- the epsilon does
        # not even achieve in half precision what it was added for.
        # Snippet used to generate expected: cam2pixel([[[[1., 2., 0.]]]].half(), _k44(..., float16)) executed
        # 2026-09-05 (torch 2.14.0, cpu and mps) -> [[[[inf, inf]]]].
        # Pins the CURRENT value; NOT a contract; delete when #4267 is repaired.
        if dtype != torch.float32:
            pytest.skip("float16-specific pin; the float32 cell runs it exactly once per device")
        cam_coords = torch.tensor([[[[1.0, 2.0, 0.0]]]], device=device, dtype=torch.float16)
        uv = kornia.geometry.camera.cam2pixel(cam_coords, _k44(device, torch.float16))
        assert bool(torch.isinf(uv).all())


class TestPixel2Cam(BaseTester):
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

    def test_wart_pixel2cam_guard_admits_a_3x3_inverse_4266(self, device, dtype):
        # Wart pin for kornia#4266 (audit label 5a-pc2-03): pixel2cam's guard is written
        # ``if not len(depth.shape) == 4 and depth.shape[1] == 1``, so the second clause is dead and the guard
        # never inspects ``intrinsics_inv`` at all. A (B, 3, 3) inverse -- the shape every free function on this
        # surface takes -- passes it, and the failure surfaces much later, from transform_points.
        # Snippet used to generate expected: pixel2cam(ones(1,1,2,3), (1,3,3) inverse, zeros(1,2,3,3)) executed
        # 2026-09-05 (torch 2.14.0) -> ValueError("Last input dimensions must differ by one unit Got...").
        # Pins the CURRENT behavior; NOT a contract; delete when #4266 is repaired.
        depth = torch.ones(1, 1, 2, 3, device=device, dtype=dtype)
        pixel_coords = torch.zeros(1, 2, 3, 3, device=device, dtype=dtype)
        intrinsics_inv_3x3 = torch.tensor(
            [[[0.01, 0.0, -0.04], [0.0, 0.01, -0.03], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )
        with pytest.raises(ValueError, match="Last input dimensions must differ by one unit"):
            kornia.geometry.camera.pixel2cam(depth, intrinsics_inv_3x3, pixel_coords)


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
        # Convention pin (audit labels 5a-al-05, 5a-al-06): clone() is the ONLY deep copy on PinholeCamera -- a new
        # object, new intrinsics and extrinsics tensors with different storage, and mutating the clone leaves the
        # source untouched. Contrast scale(), which hands the SAME extrinsics to the new camera (kornia#4264).
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
        # Convention pin (audit label 5a-al-13): intrinsics_inverse() @ intrinsics is byte-exact eye(4) on an
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
        # Wart pin for kornia#4263 (audit labels 5a-al-01, 5a-al-18): scale(s) gives cx' = s * cx (2.0 for cx = 4,
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

    def test_wart_scale_shares_extrinsics_with_the_source_4264(self, device, dtype):
        # Wart pin for kornia#4264 (audit label 5a-al-01): scale() clones the intrinsics but passes
        # ``self.extrinsics`` straight through, so the returned camera aliases the source's pose -- writing
        # ``scaled.tx = 7`` (the tx setter writes into extrinsics) changes the SOURCE camera too.
        # Snippet used to generate expected: scaled.extrinsics is cam.extrinsics -> True; after scaled.tx = 7 the
        # source tx reads [7.0]; executed 2026-09-05 (torch 2.14.0, every dtype).
        # Pins the CURRENT behavior; NOT a contract; delete when #4264 is repaired.
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype),
            _e44(device, dtype, tx=1.0),
            torch.tensor([6.0], device=device, dtype=dtype),
            torch.tensor([8.0], device=device, dtype=dtype),
        )
        scaled = cam.scale(torch.tensor([2.0], device=device, dtype=dtype))
        assert scaled.extrinsics is cam.extrinsics
        assert scaled.intrinsics is not cam.intrinsics
        scaled.tx = torch.tensor([7.0], device=device, dtype=dtype)
        self.assert_close(cam.tx, torch.tensor([7.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_wart_scale_inplace_rejects_integer_image_size_4265(self, device, dtype):
        # Wart pin for kornia#4265 (audit labels 5a-al-02, 5a-al-03): the constructor accepts int64 height/width --
        # that is what the class docstring's own example builds -- and scale() promotes them to float, but the
        # in-place twin scale_() writes the float result back into the int64 storage and raises.
        # Snippet used to generate expected: cam.scale_(0.5) on an int64 height executed 2026-09-05 (torch 2.14.0,
        # every dtype) -> RuntimeError("result type Float can't be cast to the desired output type Long").
        # Pins the CURRENT behavior; NOT a contract; delete when #4265 is repaired.
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype),
            _e44(device, dtype, tx=1.0),
            torch.tensor([6], device=device),
            torch.tensor([8], device=device),
        )
        assert cam.scale(torch.tensor([0.5], device=device, dtype=dtype)).height.is_floating_point()
        with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
            cam.scale_(0.5)

    def test_wart_constructor_accepts_3x3_intrinsics_whose_projection_is_garbage_4266(self, device, dtype):
        # Wart pin for kornia#4266 (audit labels 5a-pc-02, 5a-pc-03): _check_valid_params uses ``and`` where ``or``
        # was meant (the source even carries the author's "Shouldn't this be an OR logic than AND?"), so a
        # (1, 3, 3) intrinsics with a (1, 3, 4) extrinsics passes a validator whose message promises Bx4x4, and
        # .project returns a (1, 1) tensor of nonsense instead of a (1, 2) pixel.
        # Snippet used to generate expected: PinholeCamera(K[:, :3, :3], E[:, :3, :]).project([[1., 2., 4.]])
        # executed 2026-09-05 (torch 2.14.0) -> shape (1, 1), float32 value [[0.5471698]] (dtype-dependent, so
        # only the cardinality is pinned). NOT a contract; delete when #4266 is repaired.
        K = _k44(device, dtype)[:, :3, :3].contiguous()
        E = _e44(device, dtype, tx=1.0)[:, :3, :].contiguous()
        cam = kornia.geometry.camera.PinholeCamera(
            K, E, torch.tensor([6], device=device), torch.tensor([8], device=device)
        )
        assert cam.project(torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype)).shape == (1, 1)

    def test_wart_project_rejects_an_unbatched_point_with_indexerror_4266(self, device, dtype):
        # Wart pin for kornia#4266 (audit labels 5a-pj-03, 5a-pp-01): PinholeCamera.project documents ``(*, 3)``
        # and raises a bare IndexError("tuple index out of range") on a (3,) point, while the free function
        # project_points raises ValueError("Input must be at least a 2D tensor") on the same input -- two
        # exception types for one documented contract, and neither is the documented shape.
        # Snippet used to generate expected: both calls executed 2026-09-05 (torch 2.14.0, every dtype).
        # Pins the CURRENT behavior; NOT a contract; delete when #4266 is repaired.
        cam = kornia.geometry.camera.PinholeCamera(
            _k44(device, dtype),
            _e44(device, dtype, tx=1.0),
            torch.tensor([6], device=device),
            torch.tensor([8], device=device),
        )
        point = torch.tensor([1.0, 2.0, 4.0], device=device, dtype=dtype)
        with pytest.raises(IndexError, match="tuple index out of range"):
            cam.project(point)
        with pytest.raises(ValueError, match="at least a 2D tensor"):
            kornia.geometry.camera.project_points(point, _k44(device, dtype)[:, :3, :3].contiguous())

    def test_wart_constructor_rejects_an_empty_batch_4281(self, device, dtype):
        # Wart pin for kornia#4281 (audit label 5a-pc-09): _check_valid is ``all(data.shape[0] for ...)``, which
        # tests that each batch size is non-zero rather than that they are EQUAL, so a perfectly consistent B = 0
        # camera is rejected with a message about mismatched shapes. The free functions on the same surface follow
        # kornia's empty-in/empty-out convention: project_points(B = 0) returns a (0, 2) tensor.
        # Snippet used to generate expected: both calls executed 2026-09-05 (torch 2.14.0, every dtype)
        # -> ValueError("Arguments shapes must match") and shape (0, 1, 2).
        # Pins the CURRENT behavior; NOT a contract; delete when #4281 is repaired.
        with pytest.raises(ValueError, match="Arguments shapes must match"):
            kornia.geometry.camera.PinholeCamera(
                torch.zeros(0, 4, 4, device=device, dtype=dtype),
                torch.zeros(0, 4, 4, device=device, dtype=dtype),
                torch.zeros(0, device=device, dtype=dtype),
                torch.zeros(0, device=device, dtype=dtype),
            )
        empty = kornia.geometry.camera.project_points(
            torch.zeros(0, 1, 3, device=device, dtype=dtype), _k44(device, dtype)[:, :3, :3].contiguous()
        )
        assert empty.shape == (0, 1, 2)

    def test_wart_project_and_project_points_disagree_at_z_zero_4267(self, device, dtype):
        # Wart pin for kornia#4267 (audit labels Y4-01, Y4-03, Y4-05): PinholeCamera.project and the free function
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

    def test_wart_from_parameters_zeroes_the_size_of_later_batch_elements_4279(self, device, dtype):
        # Wart pin for kornia#4279 (audit label 5a-al-11): from_parameters does ``height_tmp[..., 0] += height`` on
        # a (B,) zero tensor, so height and width are filled only for batch element 0 while every other parameter
        # (fx, fy, cx, cy, tx, ty, tz) is broadcast correctly -- the camera looks healthy until something reads its
        # image size.
        # Snippet used to generate expected: from_parameters(height=6, width=8, batch_size=2) executed 2026-09-05
        # (torch 2.14.0, every dtype) -> height [6.0, 0.0], width [8.0, 0.0], fx [100.0, 200.0], tx [1.0, 2.0].
        # Pins the CURRENT value; NOT a contract; delete when #4279 is repaired.
        cam = self._from_parameters_batch2(device, dtype)
        self.assert_close(cam.height, torch.tensor([6.0, 0.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(cam.width, torch.tensor([8.0, 0.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(cam.fx, torch.tensor([100.0, 200.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(cam.tx, torch.tensor([1.0, 2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    @pytest.mark.xfail(strict=True, reason="kornia#4279: from_parameters fills height/width only for element 0")
    def test_convention_from_parameters_fills_every_batch_element_4279(self, device, dtype):
        # Intended contract, asserted as a strict xfail so the repair makes it XPASS and forces this mark out:
        # ``batch_size=B`` builds B cameras, so a height of 6 and a width of 8 apply to every element exactly as
        # fx, cx and tx already do. Settled by #4279's own Expected section ("height.tolist() == [6.0, 6.0]").
        cam = self._from_parameters_batch2(device, dtype)
        self.assert_close(cam.height, torch.tensor([6.0, 6.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(cam.width, torch.tensor([8.0, 8.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

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

    None of these names is in ``kornia.geometry.camera.__all__`` or reachable as ``kornia.geometry.camera.X``
    (audit labels 5a-lg-16, 5a-lg-17), so they are imported from the module directly.
    """

    def _vec12(self, device, dtype):
        # (fx, fy, cx, cy, height, width, rx, ry, rz, tx, ty, tz); cx != cy and height != width so a swapped
        # reading of the vector changes the matrix.
        return torch.tensor(
            [[100.0, 100.0, 4.0, 3.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype
        )

    def test_wart_pinhole_matrix_perturbs_every_entry_4268(self, device, dtype):
        # Wart pin for kornia#4268 (audit labels 5a-lg-01, 5a-lg-02, 5a-lg-03, 5a-lg-06, 5a-lg-07): pinhole_matrix
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
        with pytest.raises(AssertionError):
            inverse_pinhole_matrix(torch.eye(4, device=device, dtype=dtype)[None])

    def test_wart_dead_legacy_functions_always_raise_4283(self, device, dtype):
        # Wart pin for kornia#4283 (audit labels 5a-lg-12, 5a-lg-13): get_optical_pose_base and homography_i_H_ref
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

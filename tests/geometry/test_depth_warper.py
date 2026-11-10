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

import pytest
import torch

import kornia
from kornia.geometry.conversions import normalize_pixel_coordinates

from testing.base import BaseTester


class TestDepthWarper(BaseTester):
    eps = 1e-6

    def _create_pinhole_pair(self, batch_size, device, dtype):
        # prepare data
        fx, fy = 1.0, 1.0
        height, width = 3, 5
        cx, cy = width / 2, height / 2
        tx, ty, tz = 0, 0, 0

        # create pinhole cameras
        pinhole_src = kornia.geometry.camera.PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx, ty, tz, batch_size, device=device, dtype=dtype
        )
        pinhole_dst = kornia.geometry.camera.PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx, ty, tz, batch_size, device=device, dtype=dtype
        )
        return pinhole_src, pinhole_dst

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_compute_projection_matrix(self, batch_size, device, dtype):
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size, device, dtype)
        pinhole_dst.tx += 1.0  # apply offset to tx

        # create warper
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        assert warper._dst_proj_src is None

        # initialize projection matrices
        warper.compute_projection_matrix(pinhole_src)
        assert warper._dst_proj_src is not None

        # retrieve computed projection matrix and compare to expected
        dst_proj_src = warper._dst_proj_src
        dst_proj_src_expected = torch.eye(4, device=device, dtype=dtype)[None].repeat(batch_size, 1, 1)  # Bx4x4
        dst_proj_src_expected[..., 0, -2] += pinhole_src.cx
        dst_proj_src_expected[..., 1, -2] += pinhole_src.cy
        dst_proj_src_expected[..., 0, -1] += 1.0  # offset to x-axis
        # half-precision cameras compose in float32
        assert dst_proj_src.dtype == torch.promote_types(dtype, torch.float32)
        self.assert_close(dst_proj_src.to(dtype), dst_proj_src_expected)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_warp_grid_offset_x1_depth1(self, batch_size, device, dtype):
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size, device, dtype)
        pinhole_dst.tx += 1.0  # apply offset to tx

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        grid_warped = warper.warp_grid(depth_src)
        assert grid_warped.shape == (batch_size, height, width, 2)

        # normalize base meshgrid
        grid = warper.grid[..., :2].to(device=device, dtype=dtype)
        grid_norm = normalize_pixel_coordinates(grid, height, width)

        # check offset in x-axis
        self.assert_close(grid_warped[..., -2, 0], grid_norm[..., -1, 0].repeat(batch_size, 1), atol=1e-4, rtol=1e-4)
        # check that y-axis remain the same
        self.assert_close(grid_warped[..., -1, 1], grid_norm[..., -1, 1].repeat(batch_size, 1), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_warp_grid_offset_x1y1_depth1(self, batch_size, device, dtype):
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size, device, dtype)
        pinhole_dst.tx += 1.0  # apply offset to tx
        pinhole_dst.ty += 1.0  # apply offset to ty

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        grid_warped = warper.warp_grid(depth_src)
        assert grid_warped.shape == (batch_size, height, width, 2)

        # normalize base meshgrid
        grid = warper.grid[..., :2].to(device=device, dtype=dtype)
        grid_norm = normalize_pixel_coordinates(grid, height, width)

        # check offset in x-axis
        self.assert_close(grid_warped[..., -2, 0], grid_norm[..., -1, 0].repeat(batch_size, 1), atol=1e-4, rtol=1e-4)
        # check that y-axis remain the same
        self.assert_close(
            grid_warped[..., -2, :, 1], grid_norm[..., -1, :, 1].repeat(batch_size, 1), rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_warp_tensor_offset_x1y1(self, batch_size, device, dtype):
        channels, height, width = 3, 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size, device, dtype)
        pinhole_dst.tx += 1.0  # apply offset to tx
        pinhole_dst.ty += 1.0  # apply offset to ty

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        # create patch to warp
        patch_dst = (
            torch.arange(float(height * width), device=device, dtype=dtype)
            .view(1, 1, height, width)
            .expand(batch_size, channels, -1, -1)
        )

        # warpd source patch by depth
        patch_src = warper(depth_src, patch_dst)

        # compare patches
        self.assert_close(patch_dst[..., 1:, 1:], patch_src[..., :2, :4], atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_compute_projection(self, batch_size, device, dtype):
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size, device, dtype)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        # test compute_projection
        xy_projected = warper._compute_projection(0.0, 0.0, 1.0)
        assert xy_projected.shape == (batch_size, 2)

    @staticmethod
    def _create_stereo_warper(focal, height, width, baselines, device, dtype):
        pinhole_src = kornia.geometry.camera.PinholeCamera.from_parameters(
            focal, focal, (width - 1) / 2, (height - 1) / 2, height, width, 0.0, 0.0, 0.0, len(baselines), device, dtype
        )
        pinhole_dst = pinhole_src.clone()
        pinhole_dst.extrinsics[:, 0, 3] = -torch.tensor(baselines, device=device, dtype=dtype)
        return kornia.geometry.depth.DepthWarper(pinhole_dst, height, width).compute_projection_matrix(pinhole_src)

    @staticmethod
    def _step_rtol(dtype):
        # Half cameras compose in float32, but their intrinsics and the returned step are rounded to the half dtype.
        return 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4

    @pytest.mark.parametrize(
        ("focal", "height", "width", "baseline"),
        [
            (500.0, 48, 64, 0.1),
            (500.0, 48, 64, 0.2),
            (500.0, 24, 32, 0.1),
            (1000.0, 48, 64, 0.1),
            (500.0, 480, 640, 0.1),
        ],
    )
    def test_compute_subpixel_step_6603(self, focal, height, width, baseline, device, dtype):
        # A rectified pair: disparity is f * b * invd, so half a pixel is 0.5 / (f * b) of inverse depth.
        warper = self._create_stereo_warper(focal, height, width, [baseline], device, dtype)
        expected = 0.5 / (focal * baseline)
        self.assert_close(warper.compute_subpixel_step().item(), expected, rtol=self._step_rtol(dtype), atol=0.0)

    def test_compute_subpixel_step_takes_batch_minimum_6603(self, device, dtype):
        warper = self._create_stereo_warper(500.0, 48, 64, [0.1, 0.2], device, dtype)
        self.assert_close(warper.compute_subpixel_step().item(), 0.005, rtol=self._step_rtol(dtype), atol=0.0)

    # A non-rectified pair, 480x640: the destination camera is (610, 590, 330, 225) rotated by the axis-angle
    # (0.05, -0.08, 0.03) and translated by (0.2, -0.07, 0.04); the source extrinsics are the identity. The steps
    # come from numpy alone, as the central difference the method takes at the image centre:
    #   ray = inv(K_src) @ [319.5, 239.5, 1]; px(invd) = project(K_dst @ (R @ ray / invd + t))
    #   step = 0.5 / (norm(px(1.01) - px(0.99)) / 2 / 0.01)
    K_SRC_A = (520.0, 510.0, 300.0, 250.0)
    K_SRC_B = (260.0, 255.0, 150.0, 125.0)
    STEP_A, STEP_B = 0.0041753668629233645, 0.004781759015307951

    @staticmethod
    def _general_warper(k_srcs, device, dtype, src_intrinsics_dtype=None):
        def intrinsics(fx, fy, cx, cy):
            return [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

        batch = len(k_srcs)
        k_src = torch.tensor([intrinsics(*k) for k in k_srcs], device=device, dtype=src_intrinsics_dtype or dtype)
        k_dst = torch.tensor([intrinsics(610.0, 590.0, 330.0, 225.0)], device=device, dtype=dtype)
        # Built in float64 on the CPU, which MPS lacks, then moved to the device.
        rotation = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            torch.tensor([[0.05, -0.08, 0.03]], dtype=torch.float64)
        ).to(device=device, dtype=dtype)
        extrinsics = torch.eye(4, device=device, dtype=dtype)[None].clone()
        extrinsics[:, :3, :3] = rotation
        extrinsics[:, :3, 3] = torch.tensor([0.2, -0.07, 0.04], device=device, dtype=dtype)
        identity = torch.eye(4, device=device, dtype=dtype).expand(batch, 4, 4)
        height = torch.full((batch,), 480.0, device=device, dtype=dtype)
        width = torch.full((batch,), 640.0, device=device, dtype=dtype)
        pinhole_src = kornia.geometry.camera.PinholeCamera(k_src, identity, height, width)
        pinhole_dst = kornia.geometry.camera.PinholeCamera(k_dst, extrinsics, height[:1], width[:1])
        return kornia.geometry.depth.DepthWarper(pinhole_dst, 480, 640).compute_projection_matrix(pinhole_src)

    def test_convention_subpixel_step_general_pair_6603(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("The literal pins the float32 and float64 value.")
        warper = self._general_warper([self.K_SRC_A], device, dtype)
        self.assert_close(warper.compute_subpixel_step().item(), self.STEP_A, rtol=1e-4, atol=0.0)

    @pytest.mark.parametrize("order", ["AB", "BA"])
    def test_compute_subpixel_step_uses_each_source_camera_6603(self, device, dtype, order):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("The literals pin the float32 and float64 values.")
        k_srcs = [self.K_SRC_A, self.K_SRC_B] if order == "AB" else [self.K_SRC_B, self.K_SRC_A]
        warper = self._general_warper(k_srcs, device, dtype)
        expected = min(self.STEP_A, self.STEP_B)
        self.assert_close(warper.compute_subpixel_step().item(), expected, rtol=1e-4, atol=0.0)

    def test_compute_subpixel_step_source_intrinsics_in_another_dtype_6603(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64.")
        warper = self._general_warper([self.K_SRC_A], device, torch.float32, src_intrinsics_dtype=torch.float64)
        step = warper.compute_subpixel_step()
        assert step.dtype == torch.float32
        self.assert_close(step.item(), self.STEP_A, rtol=1e-4, atol=0.0)

    def test_compute_projection_value_6603(self, device, dtype):
        # The centre pixel of the source camera at inverse depth 1 lands f * b = 50 pixels to the left. The baseline
        # is the one the camera holds: 0.1 rounds in half precision, and that moves the pixel by a part of a pixel.
        warper = self._create_stereo_warper(500.0, 48, 64, [0.1], device, dtype)
        baseline = torch.tensor(0.1, dtype=dtype).item()
        expected = torch.tensor([[31.5 - 500.0 * baseline, 23.5]], device=device, dtype=warper._dst_proj_src.dtype)
        self.assert_close(warper._compute_projection(31.5, 23.5, 1.0), expected, rtol=1e-4, atol=1e-3)

    def test_compute_subpixel_step_gradient_beside_a_camera_without_baseline_6603(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64.")
        dtype = torch.float64

        def step(baselines, scale):
            batch = len(baselines)
            k = torch.tensor(
                [[500.0, 0.0, 31.5, 0.0], [0.0, 500.0, 23.5, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                device=device,
                dtype=dtype,
            )
            focal = torch.stack([scale, scale, scale.new_ones(()), scale.new_ones(())])
            k_dst = (k * focal[:, None]).expand(batch, 4, 4)
            extrinsics = torch.eye(4, device=device, dtype=dtype).repeat(batch, 1, 1)
            extrinsics[:, 0, 3] = -torch.tensor(baselines, device=device, dtype=dtype)
            identity = torch.eye(4, device=device, dtype=dtype).expand(batch, 4, 4)
            height = torch.full((batch,), 48.0, device=device, dtype=dtype)
            width = torch.full((batch,), 64.0, device=device, dtype=dtype)
            pinhole_src = kornia.geometry.camera.PinholeCamera(k.expand(batch, 4, 4), identity, height, width)
            pinhole_dst = kornia.geometry.camera.PinholeCamera(k_dst, extrinsics, height, width)
            warper = kornia.geometry.depth.DepthWarper(pinhole_dst, 48, 64)
            return warper.compute_projection_matrix(pinhole_src).compute_subpixel_step()

        # d/ds of 0.5 / (s * f * b) at s = 1 is -0.5 / (f * b) = -0.01; the camera without baseline adds nothing.
        scale = torch.ones((), device=device, dtype=dtype, requires_grad=True)
        value = step([0.0, 0.1], scale)
        (grad,) = torch.autograd.grad(value, scale)
        self.assert_close(value.item(), 0.01, rtol=1e-4, atol=0.0)
        self.assert_close(grad.item(), -0.01, rtol=1e-4, atol=0.0)

        # Only cameras without baseline: inf, with a zero gradient by convention.
        scale = torch.ones((), device=device, dtype=dtype, requires_grad=True)
        value = step([0.0, 0.0], scale)
        (grad,) = torch.autograd.grad(value, scale)
        assert torch.isinf(value)
        assert grad.item() == 0.0

    def test_compute_subpixel_step_matches_warp_grid_6603(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64.")
        # A general pose and different intrinsics: the step is what warp_grid measures at the image centre.
        dtype = torch.float64
        height, width = 31, 41
        k_src = torch.tensor(
            [[[50.0, 0.0, 18.0, 0.0], [0.0, 60.0, 17.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        k_dst = k_src * torch.tensor([1.2, 0.9, 1.0, 1.0], device=device, dtype=dtype)[:, None]
        pose = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            torch.tensor([[0.02, -0.03, 0.01]], device=device, dtype=dtype)
        )
        extrinsics = torch.eye(4, device=device, dtype=dtype)[None].clone()
        extrinsics[:, :3, :3] = pose
        extrinsics[:, :3, 3] = torch.tensor([0.1, -0.05, 0.02], device=device, dtype=dtype)
        identity = torch.eye(4, device=device, dtype=dtype)[None]
        size = torch.tensor([height], device=device, dtype=dtype), torch.tensor([width], device=device, dtype=dtype)
        pinhole_src = kornia.geometry.camera.PinholeCamera(k_src, identity, *size)
        pinhole_dst = kornia.geometry.camera.PinholeCamera(k_dst, extrinsics, *size)
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width).compute_projection_matrix(pinhole_src)

        def center(invd):
            depth = torch.full((1, 1, height, width), 1.0 / invd, device=device, dtype=dtype)
            grid = warper.warp_grid(depth)
            return kornia.geometry.conversions.denormalize_pixel_coordinates(grid, height, width)[
                0, height // 2, width // 2
            ]

        expected = 0.5 / ((center(1.01) - center(0.99)).norm() / 2 / 0.01)
        self.assert_close(warper.compute_subpixel_step(), expected, rtol=1e-9, atol=0.0)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_compute_subpixel_step_without_baseline_6603(self, batch_size, device, dtype):
        # No baseline, no parallax: no change of inverse depth moves a pixel.
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size, device, dtype)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        subpixel_step = warper.compute_subpixel_step()
        assert subpixel_step.dtype == dtype
        assert torch.isinf(subpixel_step)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_gradcheck(self, batch_size, device):
        dtype = torch.float64
        # prepare data
        channels, height, width = 3, 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size, device, dtype)

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)

        # create patch to warp
        img_dst = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.depth.depth_warp, (pinhole_dst, pinhole_src, depth_src, img_dst, height, width))

    # TODO(edgar): we should include a test showing some kind of occlusion
    # def test_warp_with_occlusion(self):
    #    pass

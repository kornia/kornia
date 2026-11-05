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

    @pytest.mark.parametrize(
        ("focal", "height", "width", "baseline"),
        [(500.0, 48, 64, 0.1), (500.0, 48, 64, 0.2), (500.0, 24, 32, 0.1), (1000.0, 48, 64, 0.1)],
    )
    def test_compute_subpixel_step_6603(self, focal, height, width, baseline, device, dtype):
        # A rectified pair: disparity is f * b * invd, so half a pixel is 0.5 / (f * b) of inverse depth. float16
        # rounds the projected pixels to 1/16, about 3% of this step.
        warper = self._create_stereo_warper(focal, height, width, [baseline], device, dtype)
        self.assert_close(warper.compute_subpixel_step().item(), 0.5 / (focal * baseline), rtol=0.05, atol=0.0)

    def test_compute_subpixel_step_takes_batch_minimum_6603(self, device, dtype):
        warper = self._create_stereo_warper(500.0, 48, 64, [0.1, 0.2], device, dtype)
        self.assert_close(warper.compute_subpixel_step().item(), 0.005, rtol=0.05, atol=0.0)

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

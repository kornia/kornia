import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils
from torchgeometry.core.depth_warper import PinholeCamera
from torchgeometry.core.depth_warper import normalize_pixel_coordinates


class TestDepthWarper:
    eps = 1e-6

    def _test_compute_projection_matrix(self):
        # prepare data
        batch_size, height, width = 1, 3, 5
        fx, fy = 1., 1.
        cx, cy = width / 2, height / 2
        tx, ty, tz = 0, 0, 0

        # create pinhole cameras
        pinhole_src = PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx, ty, tz)
        pinhole_dst = PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx + 1., ty, tz)

        # create warper
        warper = tgm.DepthWarper(None, height, width, pinhole_dst)
        assert warper._dst_proj_src is None

        # initialize projection matrices
        warper.compute_projection_matrix(pinhole_src)
        assert warper._dst_proj_src is not None

        # retreive computed projection matrix and compare to expected
        dst_proj_src = warper._dst_proj_src
        dst_proj_src_expected = torch.eye(4)[None]  # Bx4x4
        dst_proj_src_expected[..., 0, -2] += cx
        dst_proj_src_expected[..., 1, -2] += cy
        dst_proj_src_expected[..., 0, -1] += 1.  # offset to x-axis
        assert utils.check_equal_torch(dst_proj_src, dst_proj_src_expected)

    def _test_warp_grid_offset_x1_depth1(self):
        # prepare data
        batch_size, height, width = 1, 3, 5
        fx, fy = 1., 1.
        cx, cy = width / 2, height / 2
        tx, ty, tz = 0, 0, 0

        # create pinhole cameras
        pinhole_src = PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx, ty, tz)
        pinhole_dst = PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx + 1., ty, tz)

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width)

        # create warper, initialize projection matrices and warp grid
        warper = tgm.DepthWarper(None, height, width, pinhole_dst)
        warper.compute_projection_matrix(pinhole_src)
        grid_warped = warper.warp_grid(depth_src)

        # normalize base meshgrid
        grid = warper.grid[..., :2]
        grid_norm = normalize_pixel_coordinates(grid, height, width)

        # check offset in x-axis
        assert utils.check_equal_torch(
            grid_norm[..., -1, 0], grid_warped[..., -2, 0])
        # check that y-axis remain the same
        assert utils.check_equal_torch(
            grid_norm[..., -1, 1], grid_warped[..., -1, 1])

    def _test_warp_grid_offset_x1y1_depth1(self):
        # prepare data
        batch_size, height, width = 1, 3, 5
        fx, fy = 1., 1.
        cx, cy = width / 2, height / 2
        tx, ty, tz = 0, 0, 0

        # create pinhole cameras
        pinhole_src = PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx, ty, tz)
        pinhole_dst = PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx + 1., ty + 1., tz)

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width)

        # create warper, initialize projection matrices and warp grid
        warper = tgm.DepthWarper(None, height, width, pinhole_dst)
        warper.compute_projection_matrix(pinhole_src)
        grid_warped = warper.warp_grid(depth_src)

        # normalize base meshgrid
        grid = warper.grid[..., :2]
        grid_norm = normalize_pixel_coordinates(grid, height, width)

        # check offset in x-axis
        assert utils.check_equal_torch(
            grid_norm[..., -1, 0], grid_warped[..., -2, 0])
        # check that y-axis remain the same
        assert utils.check_equal_torch(
            grid_norm[..., -1, :, 1], grid_warped[..., -2, :, 1])

    def _test_depth_warper_offset_tensor(self):
        # prepare data
        batch_size, channels, height, width = 1, 3, 3, 5
        fx, fy = 1., 1.
        cx, cy = width / 2, height / 2
        tx, ty, tz = 0, 0, 0

        # create pinhole cameras
        pinhole_src = PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx, ty, tz)
        pinhole_dst = PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx + 1., ty + 1., tz)

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width)

        # create warper, initialize projection matrices and warp grid
        warper = tgm.DepthWarper(None, height, width, pinhole_dst)
        warper.compute_projection_matrix(pinhole_src)

        # create patch to warp
        patch_dst = torch.arange(float(height * width)).view(
            batch_size, 1, height, width).expand(batch_size, channels, -1, -1)

        # warpd source patch by depth
        patch_src = warper(depth_src, patch_dst)

        # compare patches
        assert utils.check_equal_torch(
            patch_dst[..., 1:, 1:], patch_src[..., :2, :4])

        # test compute_projection
        # TODO: this might need a better test
        xy_projected = warper._compute_projection(0.0, 0.0, 1.0)
        assert xy_projected.shape == (1, 2)

        # test compute_subpixel_step
        # TODO: this might need a better test
        subpixel_step = warper.compute_subpixel_step()
        assert pytest.approx(subpixel_step.item(), 0.3536)

    def test_run_all(self):
        self._test_compute_projection_matrix()
        self._test_warp_grid_offset_x1_depth1()
        self._test_warp_grid_offset_x1y1_depth1()
        self._test_depth_warper_offset_tensor()

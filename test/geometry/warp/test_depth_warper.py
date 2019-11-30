import pytest

import kornia
import kornia.testing as utils  # test utils
from kornia.geometry.conversions import normalize_pixel_coordinates
from test.common import device

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestDepthWarper:
    eps = 1e-6

    def _create_pinhole_pair(self, batch_size):
        # prepare data
        fx, fy = 1., 1.
        height, width = 3, 5
        cx, cy = width / 2, height / 2
        tx, ty, tz = 0, 0, 0

        # create pinhole cameras
        pinhole_src = kornia.PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx, ty, tz, batch_size)
        pinhole_dst = kornia.PinholeCamera.from_parameters(
            fx, fy, cx, cy, height, width, tx, ty, tz, batch_size)
        return pinhole_src, pinhole_dst

    @pytest.mark.parametrize("batch_size", (1, 2,))
    def test_compute_projection_matrix(self, batch_size):
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size)
        pinhole_dst.tx += 1.  # apply offset to tx

        # create warper
        warper = kornia.DepthWarper(pinhole_dst, height, width)
        assert warper._dst_proj_src is None

        # initialize projection matrices
        warper.compute_projection_matrix(pinhole_src)
        assert warper._dst_proj_src is not None

        # retreive computed projection matrix and compare to expected
        dst_proj_src = warper._dst_proj_src
        dst_proj_src_expected = torch.eye(
            4)[None].repeat(batch_size, 1, 1)  # Bx4x4
        dst_proj_src_expected[..., 0, -2] += pinhole_src.cx
        dst_proj_src_expected[..., 1, -2] += pinhole_src.cy
        dst_proj_src_expected[..., 0, -1] += 1.  # offset to x-axis
        assert_allclose(dst_proj_src, dst_proj_src_expected)

    @pytest.mark.parametrize("batch_size", (1, 2,))
    def test_warp_grid_offset_x1_depth1(self, device, batch_size):
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size)
        pinhole_dst.tx += 1.  # apply offset to tx

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        grid_warped = warper.warp_grid(depth_src)
        assert grid_warped.shape == (batch_size, height, width, 2)

        # normalize base meshgrid
        grid = warper.grid[..., :2]
        grid_norm = normalize_pixel_coordinates(grid, height, width)

        # check offset in x-axis
        assert_allclose(
            grid_warped[..., -2, 0], grid_norm[..., -1, 0])
        # check that y-axis remain the same
        assert_allclose(
            grid_warped[..., -1, 1], grid_norm[..., -1, 1])

    @pytest.mark.parametrize("batch_size", (1, 2,))
    def test_warp_grid_offset_x1y1_depth1(self, device, batch_size):
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size)
        pinhole_dst.tx += 1.  # apply offset to tx
        pinhole_dst.ty += 1.  # apply offset to ty

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        grid_warped = warper.warp_grid(depth_src)
        assert grid_warped.shape == (batch_size, height, width, 2)

        # normalize base meshgrid
        grid = warper.grid[..., :2]
        grid_norm = normalize_pixel_coordinates(grid, height, width)

        # check offset in x-axis
        assert_allclose(
            grid_warped[..., -2, 0], grid_norm[..., -1, 0])
        # check that y-axis remain the same
        assert_allclose(
            grid_warped[..., -2, :, 1], grid_norm[..., -1, :, 1])

    @pytest.mark.parametrize("batch_size", (1, 2,))
    def test_warp_tensor_offset_x1y1(self, device, batch_size):
        channels, height, width = 3, 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size)
        pinhole_dst.tx += 1.  # apply offset to tx
        pinhole_dst.ty += 1.  # apply offset to ty

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        # create patch to warp
        patch_dst = torch.arange(float(height * width)).view(
            1, 1, height, width).expand(batch_size, channels, -1, -1)

        # warpd source patch by depth
        patch_src = warper(depth_src, patch_dst)

        # compare patches
        assert_allclose(
            patch_dst[..., 1:, 1:], patch_src[..., :2, :4])

    @pytest.mark.parametrize("batch_size", (1, 2,))
    def test_compute_projection(self, device, batch_size):
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        # test compute_projection
        xy_projected = warper._compute_projection(0.0, 0.0, 1.0)
        assert xy_projected.shape == (batch_size, 2)

    @pytest.mark.parametrize("batch_size", (1, 2,))
    def test_compute_subpixel_step(self, device, batch_size):
        height, width = 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size)

        # create warper, initialize projection matrices and warp grid
        warper = kornia.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)

        # test compute_subpixel_step
        subpixel_step = warper.compute_subpixel_step()
        assert pytest.approx(subpixel_step.item(), 0.3536)

    @pytest.mark.parametrize("batch_size", (1, 2))
    def test_gradcheck(self, device, batch_size):
        # prepare data
        channels, height, width = 3, 3, 5  # output shape
        pinhole_src, pinhole_dst = self._create_pinhole_pair(batch_size)

        # initialize depth to one
        depth_src = torch.ones(batch_size, 1, height, width)
        depth_src = utils.tensor_to_gradcheck_var(depth_src)  # to var

        # create patch to warp
        img_dst = torch.ones(batch_size, channels, height, width)
        img_dst = utils.tensor_to_gradcheck_var(img_dst)  # to var

        # evaluate function gradient
        assert gradcheck(kornia.depth_warp,
                         (pinhole_dst,
                          pinhole_src,
                          depth_src,
                          img_dst,
                          height,
                          width,
                          ),
                         raise_exception=True)

    # TODO(edgar): we should include a test showing some kind of occlusion
    # def test_warp_with_occlusion(self):
    #    pass

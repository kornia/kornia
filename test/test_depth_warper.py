import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils


class TestDepthWarper:
    eps = 1e-6

    def _test_warp_grid_offset_x1_depth1(self):
        batch_size, channels, height, width = 1, 3, 3, 4
        depth = torch.ones(batch_size, 1, height, width)
        inv_depth = 1. / (depth + self.eps)

        i_trans_ref = torch.eye(4)[None] + self.eps   # 1x4x4
        i_trans_ref[..., 0, -1] += 1.

        warper = tgm.DepthWarper(None, height, width,
            normalized_coordinates=False)
        warper._i_Hs_ref = i_trans_ref

        grid = warper.grid
        grid_warped = warper.warp_grid(inv_depth)

        assert utils.check_equal_torch(grid[..., 1], grid_warped[..., 1])
        assert utils.check_equal_torch(
            grid[..., 1:4, 0], grid_warped[..., 0:3, 0])

    def _test_warp_grid_offset_x1_depth2(self):
        depth_val = 2.0
        batch_size, channels, height, width = 1, 3, 3, 4
        depth = torch.ones(batch_size, 1, height, width) * depth_val
        inv_depth = 1. / (depth + self.eps)

        i_trans_ref = torch.eye(4)[None] + self.eps   # 1x4x4
        i_trans_ref[..., 0, -1] += 1.

        warper = tgm.DepthWarper(None, height, width,
            normalized_coordinates=False)
        warper._i_Hs_ref = i_trans_ref

        grid = warper.grid
        grid_warped = warper.warp_grid(inv_depth)

        assert utils.check_equal_torch(
            grid[..., 1], grid_warped[..., 1] / depth_val)
        assert utils.check_equal_torch(
            grid[..., 1:4, 0], grid_warped[..., 0:3, 0] / depth_val)

    def _test_warp_grid_offset_x1y1_depth1(self):
        depth_val = 1.0
        batch_size, channels, height, width = 1, 3, 3, 4
        depth = torch.ones(batch_size, 1, height, width) * depth_val
        inv_depth = 1. / (depth + self.eps)

        i_trans_ref = torch.eye(4)[None] + self.eps   # 1x4x4
        i_trans_ref[..., 0, -1] += 1.
        i_trans_ref[..., 1, -1] += 1.

        warper = tgm.DepthWarper(None, height, width,
            normalized_coordinates=False)
        warper._i_Hs_ref = i_trans_ref

        grid = warper.grid
        grid_warped = warper.warp_grid(inv_depth)

        assert utils.check_equal_torch(
                grid[..., 1:3, :, 1], grid_warped[..., 0:2, :, 1] / depth_val)
        assert utils.check_equal_torch(
            grid[..., 1:4, 0], grid_warped[..., 0:3, 0] / depth_val)

    def _test_depth_warper_offset(self):
        # generate input data
        batch_size = 1
        height, width = 4, 5
        cx, cy = width / 2, height / 2
        fx, fy = 1., 1.
        rx, ry, rz = 0., 0., 0.
        tx, ty, tz = 0., 0., 0.
        offset = 1.  # we will apply a 1unit offset to `i` camera
    
        pinhole_ref = utils.create_pinhole(
            fx, fy, cx, cy, height, width, rx, ry, rx, tx, ty, tz)
        pinhole_ref = pinhole_ref.expand(batch_size, -1)
    
        pinhole_i = utils.create_pinhole(
            fx,
            fy,
            cx,
            cy,
            height,
            width,
            rx,
            ry,
            rx,
            tx + offset,
            ty + offset,
            tz)
        pinhole_i = pinhole_i.expand(batch_size, -1)
    
        # create checkerboard
        #board = utils.create_checkerboard(height, width, 4)
        #patch_i = torch.from_numpy(board).view(
        #    1, 1, height, width).expand(batch_size, 1, height, width)
        patch_i = torch.arange(float(height * width)).view(
            batch_size, 1, height, width).expand(batch_size, -1, -1, -1)
    
        # instantiate warper and compute relative homographies
        warper = tgm.DepthWarper(pinhole_i)
        warper.compute_homographies(
            pinhole_ref, scale=torch.ones(
                batch_size))
    
        # generate synthetic inverse depth
        inv_depth_ref = torch.ones(batch_size, 1, height, width)
        #inv_depth_ref[..., 1:-1, 1:-1] = 2.
    
        # warpd source patch by depth
        patch_ref = warper(inv_depth_ref, patch_i)
        import pdb;pdb.set_trace()
    
        # compute error
        assert utils.check_equal_torch(
            patch_ref[..., :int(height - offset), :int(width - offset)],
            patch_i[..., int(offset):, int(offset):])
    
        # test functional
        patch_ref_functional = tgm.depth_warp(pinhole_i, pinhole_ref,
                                              inv_depth_ref, patch_i)
        assert utils.check_equal_torch(patch_ref, patch_ref_functional)
    
        # test compute_projection
        # TODO: this might need a better test
        xy_projected = warper._compute_projection(0.0, 0.0, 1.0)
        assert xy_projected.shape == (1, 2)
    
        # test compute_subpixel_step
        # TODO: this might need a better test
        subpixel_step = warper.compute_subpixel_step()
        assert pytest.approx(subpixel_step.item(), 0.3536)

    def test_run_all(self):
        self._test_warp_grid_offset_x1_depth1()
        self._test_warp_grid_offset_x1_depth2()
        self._test_warp_grid_offset_x1y1_depth1()
        #self._test_depth_warper_offset()


import unittest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils


class Tester(unittest.TestCase):

    def test_depth_warper(self):
        # generate input data
        batch_size = 1
        height, width = 8, 8
        cx, cy = width / 2, height / 2
        fx, fy = 1., 1.
        rx, ry, rz = 0., 0., 0.
        tx, ty, tz = 0., 0., 0.
        offset = 1.  # we will apply a 1unit offset to `i` camera

        pinhole_src = utils.create_pinhole(
            fx, fy, cx, cy, height, width, rx, ry, rx, tx, ty, tz)
        pinhole_src = pinhole_src.expand(batch_size, -1)

        pinhole_dst = utils.create_pinhole(
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
        pinhole_dst = pinhole_dst.expand(batch_size, -1)

        # create checkerboard
        board = utils.create_checkerboard(height, width, 4)
        patch_src = torch.from_numpy(board).view(
            1, 1, height, width).expand(batch_size, 1, height, width)

        # instantiate warper and compute relative homographies
        warper = tgm.DepthWarper(pinhole_src, height, width)
        warper.compute_homographies(
            pinhole_dst, scale=torch.ones(
                batch_size, 1))

        # generate synthetic inverse depth
        inv_depth_src = torch.ones(batch_size, 1, height, width)

        # warpd source patch by depth
        patch_dst = warper(inv_depth_src, patch_src)

        # compute error
        res = utils.check_equal_torch(
            patch_src[..., :-int(offset), :-int(offset)],
            patch_dst[..., int(offset):, int(offset):])
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()

import unittest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils


class Tester(unittest.TestCase):

    def test_warp_perspective(self):
        # generate input data
        batch_size = 1
        height, width = 16, 32
        alpha = tgm.pi / 2  # 90 deg rotation

        # create data patch
        patch = torch.rand(batch_size, 1, height, width)

        # create transformation (rotation)
        M = torch.tensor([[
            [torch.cos(alpha), -torch.sin(alpha), 0.],
            [torch.sin(alpha), torch.cos(alpha), 0.],
            [0., 0., 1.],
        ]])  # Bx3x3

        # apply transformation and inverse
        _, _, h, w = patch.shape
        patch_warped = tgm.warp_perspective(patch, M, dsize=(height, width))
        patch_warped_inv = tgm.warp_perspective(patch_warped, tgm.inverse(M),
                                                dsize=(height, width))

        # generate mask to compute error
        mask = torch.ones_like(patch)
        mask_warped_inv = tgm.warp_perspective(
            tgm.warp_perspective(patch, M, dsize=(height, width)),
            tgm.inverse(M), dsize=(height, width))

        res = utils.check_equal_torch(mask_warped_inv * patch,
                                      mask_warped_inv * patch_warped_inv)
        self.assertTrue(res)

    def test_warp_perspective_gradcheck(self):
        # generate input data
        batch_size = 1
        height, width = 16, 32
        alpha = tgm.pi / 2  # 90 deg rotation

        # create data patch
        patch = torch.rand(batch_size, 1, height, width)
        patch = utils.tensor_to_gradcheck_var(patch)  # to var

        # create transformation (rotation)
        M = torch.tensor([[
            [torch.cos(alpha), -torch.sin(alpha), 0.],
            [torch.sin(alpha), torch.cos(alpha), 0.],
            [0., 0., 1.],
        ]])  # Bx3x3
        M = utils.tensor_to_gradcheck_var(M, requires_grad=False)  # to var

        # evaluate function gradient
        res = gradcheck(tgm.warp_perspective, (patch, M, (height, width,)),
                        raise_exception=True)
        self.assertTrue(res)

    def test_get_perspective_transform(self):
        # generate input data
        h, w = 64, 32  # height, width
        norm = torch.randn(1, 4, 2)
        points_src = torch.FloatTensor([[
            [0, 0], [h, 0], [0, w], [h, w],
        ]])
        points_dst = points_src + norm

        # compute transform from source to target
        dst_homo_src = tgm.get_perspective_transform(points_src, points_dst)

        res = utils.check_equal_torch(
            tgm.transform_points(dst_homo_src, points_src), points_dst)
        self.assertTrue(res)

    def test_get_perspective_transform_gradcheck(self):
        # generate input data
        h, w = 64, 32  # height, width
        norm = torch.randn(1, 4, 2)
        points_src = torch.FloatTensor([[
            [0, 0], [h, 0], [0, w], [h, w],
        ]])
        points_dst = points_src + norm
        points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
        points_dst = utils.tensor_to_gradcheck_var(points_dst)  # to var

        # compute transform from source to target
        res = gradcheck(tgm.get_perspective_transform,
            (points_src, points_dst,), raise_exception=True)
        self.assertTrue(res)
if __name__ == '__main__':
    unittest.main()

import unittest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils


class Tester(unittest.TestCase):

    def test_warp_perspective(self):
        # generate input data
        batch_size = 1
        height, width = 64, 64
        alpha = tgm.pi / 2  # 90 deg rotation

        # create data patch
        patch = torch.rand(batch_size, 1, height, width)

        # create transformation (rotation)
        M = torch.tensor([[
            [torch.cos(alpha), -torch.sin(alpha), 0.],
            [torch.sin(alpha),  torch.cos(alpha), 0.],
            [              0.,                0., 1.],
        ]])  # Bx3x3

        # apply transformation and inverse
        _, _, h, w = patch.shape
        patch_warped = tgm.warp_perspective(patch, M, dsize=(height, width))
        patch_warped_inv = tgm.warp_perspective(patch_warped, tgm.inverse(M),
                dsize=(height, width))

        res = utils.check_equal_torch(patch, patch_warped_inv)
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
            [torch.sin(alpha),  torch.cos(alpha), 0.],
            [              0.,                0., 1.],
        ]])  # Bx3x3
        M = utils.tensor_to_gradcheck_var(M, requires_grad=False)  # to var

        # evaluate function gradient
        res = gradcheck(tgm.warp_perspective, (patch, M, (height, width,)),
                        raise_exception=True)
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()

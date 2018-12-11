import unittest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils


class Tester(unittest.TestCase):

    num_tests = 10

    def test_homography_warper(self):
        # generate input data
        batch_size = 1
        height, width = 128, 64
        eye_size = 3  # identity 3x3

        # create checkerboard
        board = utils.create_checkerboard(height, width, 4)
        patch_src = torch.from_numpy(board).view(
            1, 1, height, width).expand(batch_size, 1, height, width)

        # create base homography
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size)

        # instantiate warper
        warper = tgm.HomographyWarper(height, width)

        for i in range(self.num_tests):
            # generate homography noise
            homo_delta = torch.zeros_like(dst_homo_src)
            homo_delta[:, -1, -1] = 0.0

            dst_homo_src_i = dst_homo_src + homo_delta

            # transform the points from dst to ref
            patch_dst = warper(patch_src, dst_homo_src_i)
            patch_dst_to_src = warper(patch_dst, torch.inverse(dst_homo_src_i))

            # projected should be equal as initial
            error = utils.compute_patch_error(
                patch_dst, patch_dst_to_src, height, width)

            threshold = 0.05
            self.assertTrue(error.item() < threshold)

            # check functional api
            patch_dst_to_src_functional = tgm.homography_warp(
                patch_dst, torch.inverse(dst_homo_src_i), (height, width))
            res = utils.check_equal_torch(patch_dst_to_src,
                                          patch_dst_to_src_functional)
            self.assertTrue(res)

    def test_local_homography_warper(self):
        # generate input data
        batch_size = 1
        height, width = 16, 32
        eye_size = 3  # identity 3x3

        # create checkerboard
        board = utils.create_checkerboard(height, width, 4)
        patch_src = torch.from_numpy(board).view(
            1, 1, height, width).expand(batch_size, 1, height, width)

        # create local homography
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size)
        dst_homo_src = dst_homo_src.view(batch_size, -1).unsqueeze(1)
        dst_homo_src = dst_homo_src.repeat(1, height * width, 1).view(
            1, height, width, 3, 3)  # NxHxWx3x3

        # warp reference patch
        patch_src_to_i = tgm.homography_warp(
            patch_src, dst_homo_src, (height, width))

    def test_homography_warper_gradcheck(self):
        # generate input data
        batch_size = 1
        height, width = 16, 32  # small patch, otherwise the test takes forever
        eye_size = 3  # identity 3x3

        # create checkerboard
        board = utils.create_checkerboard(height, width, 4)
        patch_src = torch.from_numpy(board).view(
            1, 1, height, width).expand(batch_size, 1, height, width)
        patch_src = utils.tensor_to_gradcheck_var(patch_src)  # to var

        # create base homography
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size)
        dst_homo_src = utils.tensor_to_gradcheck_var(
            dst_homo_src, requires_grad=False)  # to var

        # instantiate warper
        warper = tgm.HomographyWarper(height, width)

        # evaluate function gradient
        res = gradcheck(warper, (patch_src, dst_homo_src,),
                        raise_exception=True)
        self.assertTrue(res)

        # evaluate function gradient
        res = gradcheck(
            tgm.homography_warp,
            (patch_src,
             dst_homo_src,
             (height,
              width)),
            raise_exception=True)
        self.assertTrue(res)

if __name__ == '__main__':
    unittest.main()

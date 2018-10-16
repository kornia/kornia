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

    def test_warp_perspective_crop(self):
        # generate input data
        batch_size = 1
        src_h, src_w = 3, 4
        dst_h, dst_w = 3, 2

        # [x, y] origin
        # top-left, top-right, bottom-right, bottom-left
        points_src = torch.FloatTensor([[
            [1, 0], [2, 0], [2, 2], [1, 2],
        ]])

        # [x, y] destination
        # top-left, top-right, bottom-right, bottom-left
        points_dst = torch.FloatTensor([[
            [0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1],
        ]])

        # compute transformation between points
        dst_pix_trans_src_pix = tgm.get_perspective_transform(
            points_src, points_dst)

        # create points grid in normalized coordinates
        grid_src_norm = tgm.create_meshgrid(src_h, src_w,
            normalized_coordinates=True)
        grid_src_norm = torch.unsqueeze(grid_src_norm, dim=0)

        # create points grid in pixel coordinates
        grid_src_pix = tgm.create_meshgrid(src_h, src_w,
            normalized_coordinates=False)
        grid_src_pix = torch.unsqueeze(grid_src_pix, dim=0)

        src_norm_trans_src_pix = tgm.normal_transform_pixel(src_h, src_w)
        src_pix_trans_src_norm = tgm.inverse(src_norm_trans_src_pix)

        dst_norm_trans_dst_pix = tgm.normal_transform_pixel(dst_h, dst_w)

        # transform pixel grid
        grid_dst_pix = tgm.transform_points(
            dst_pix_trans_src_pix, grid_src_pix)
        grid_dst_norm = tgm.transform_points(
            dst_norm_trans_dst_pix, grid_dst_pix)

        # transform norm grid
        dst_norm_trans_src_norm = torch.matmul(dst_norm_trans_dst_pix,
            torch.matmul(dst_pix_trans_src_pix, src_pix_trans_src_norm))
        grid_dst_norm2 = tgm.transform_points(
            dst_norm_trans_src_norm, grid_src_norm)

        # grids should be equal
        self.assertTrue(utils.check_equal_torch(
            grid_dst_norm, grid_dst_norm2))

        # warp tensor
        patch = torch.rand(batch_size, 1, src_h, src_w)
        patch_warped = tgm.warp_perspective(patch,
            dst_pix_trans_src_pix, (dst_h, dst_w))
        self.assertTrue(utils.check_equal_torch(
            patch[:, :, :3, 1:3], patch_warped))

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

    def test_rotation_matrix2d(self):
        # generate input data
        batch_size = 1
        center_base = torch.zeros(batch_size, 2)
        angle_base = torch.ones(batch_size, 1)
        scale_base = torch.ones(batch_size, 1)

        # 90 deg rotation
        center = center_base
        angle = 90. * angle_base
        scale = scale_base
        M = tgm.get_rotation_matrix2d(center, angle, scale)

        self.assertAlmostEqual(M[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(M[0, 0, 1].item(), 1.0)
        self.assertAlmostEqual(M[0, 1, 0].item(), -1.0)
        self.assertAlmostEqual(M[0, 1, 1].item(), 0.0)

        # 90 deg rotation + 2x scale
        center = center_base
        angle = 90. * angle_base
        scale = 2. * scale_base
        M = tgm.get_rotation_matrix2d(center, angle, scale)

        self.assertAlmostEqual(M[0, 0, 0].item(), 0.0, 4)
        self.assertAlmostEqual(M[0, 0, 1].item(), 2.0, 4)
        self.assertAlmostEqual(M[0, 1, 0].item(), -2.0, 4)
        self.assertAlmostEqual(M[0, 1, 1].item(), 0.0, 4)

        # 45 deg rotation
        center = center_base
        angle = 45. * angle_base
        scale = scale_base
        M = tgm.get_rotation_matrix2d(center, angle, scale)

        self.assertAlmostEqual(M[0, 0, 0].item(), 0.7071, 4)
        self.assertAlmostEqual(M[0, 0, 1].item(), 0.7071, 4)
        self.assertAlmostEqual(M[0, 1, 0].item(), -0.7071, 4)
        self.assertAlmostEqual(M[0, 1, 1].item(), 0.7071, 4)

    def test_get_rotation_matrix2d_gradcheck(self):
        # generate input data
        batch_size = 1
        center = torch.zeros(batch_size, 2)
        angle = torch.ones(batch_size, 1)
        scale = torch.ones(batch_size, 1)

        center = utils.tensor_to_gradcheck_var(center)  # to var
        angle = utils.tensor_to_gradcheck_var(angle)  # to var
        scale = utils.tensor_to_gradcheck_var(scale)  # to var

        # evaluate function gradient
        res = gradcheck(tgm.get_rotation_matrix2d, (center, angle, scale),
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

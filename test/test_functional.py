import unittest

import torch
import torchgeometry as dgm
from torch.autograd import gradcheck

import utils  # test utilities


class Tester(unittest.TestCase):

    def test_convert_points_to_homogeneous(self):
        # generate input data
        batch_size = 2
        points = torch.rand(batch_size, 2, 3)

        # to homogeneous
        points_h = dgm.convert_points_to_homogeneous(points)
        self.assertTrue((points_h[..., -1] == torch.ones(1, 2, 1)).all())

    def test_convert_points_to_homogeneous_gradcheck(self):
        # generate input data
        batch_size = 2
        points = torch.rand(batch_size, 2, 3)
        points = utils.tensor_to_gradcheck_var(points)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.convert_points_to_homogeneous, (points,),
                        raise_exception=True)
        self.assertTrue(res)

    def test_convert_points_from_homogeneous(self):
        # generate input data
        batch_size = 2
        points_h = torch.rand(batch_size, 2, 3)
        points_h[..., -1] = 1.0

        # to euclidean
        points = dgm.convert_points_from_homogeneous(points_h)

        error = utils.compute_mse(points_h[..., :2], points)
        self.assertAlmostEqual(error.item(), 0.0, places=4)

    def test_convert_points_from_homogeneous_gradcheck(self):
        # generate input data
        batch_size = 2
        points = torch.rand(batch_size, 2, 3)
        points = utils.tensor_to_gradcheck_var(points)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.convert_points_from_homogeneous, (points,),
                        raise_exception=True)
        self.assertTrue(res)

    def test_inverse(self):
        # generate input data
        batch_size = 2
        eye_size = 3  # identity 3x3
        homographies = utils.create_random_homography(batch_size, eye_size)
        homographies_inv = dgm.inverse(homographies)

        # H_inv * H == I
        res = torch.matmul(homographies_inv, homographies)
        eye = utils.create_eye_batch(batch_size, eye_size)
        error = utils.compute_mse(res, eye)
        self.assertAlmostEqual(error.item(), 0.0, places=4)

    def test_inverse_gradcheck(self):
        # generate input data
        batch_size = 2
        eye_size = 3  # identity 3x3
        homographies = utils.create_random_homography(batch_size, eye_size)
        homographies = utils.tensor_to_gradcheck_var(homographies)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.inverse, (homographies,), raise_exception=True)
        self.assertTrue(res)

    def test_transform_points(self):
        # generate input data
        batch_size = 2
        num_points = 2
        num_dims = 2
        eye_size = 3  # identity 3x3
        points_src = torch.rand(batch_size, 2, num_dims)
        dst_homo_src = utils.create_random_homography(batch_size, eye_size)

        # transform the points from dst to ref
        points_dst = dgm.transform_points(dst_homo_src, points_src)

        # transform the points from ref to dst
        src_homo_dst = dgm.inverse(dst_homo_src)
        points_dst_to_src = dgm.transform_points(src_homo_dst, points_dst)

        # projected should be equal as initial
        error = utils.compute_mse(points_src, points_dst_to_src)
        self.assertAlmostEqual(error.item(), 0.0, places=4)

    def test_transform_points_gradcheck(self):
        # generate input data
        batch_size = 2
        num_points = 2
        num_dims = 2
        eye_size = 3  # identity 3x3
        points_src = torch.rand(batch_size, 2, num_dims)
        points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
        dst_homo_src = utils.create_random_homography(batch_size, eye_size)
        dst_homo_src = utils.tensor_to_gradcheck_var(dst_homo_src)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.transform_points, (dst_homo_src, points_src,),
                        raise_exception=True)
        self.assertTrue(res)

    def test_pi(self):
        self.assertAlmostEqual(dgm.pi.item(), 3.141592, places=4)

    def test_rad2deg(self):
        # generate input data
        x_rad = dgm.pi * torch.rand(2, 3, 4)

        # convert radians/degrees
        x_deg = dgm.rad2deg(x_rad)
        x_deg_to_rad = dgm.deg2rad(x_deg)

        # compute error
        error = utils.compute_mse(x_rad, x_deg_to_rad)
        self.assertAlmostEqual(error.item(), 0.0, places=4)

    def test_rad2deg_gradcheck(self):
        # generate input data
        x_rad = dgm.pi * torch.rand(2, 3, 4)

        # evaluate function gradient
        res = gradcheck(dgm.rad2deg, (utils.tensor_to_gradcheck_var(x_rad),),
                        raise_exception=True)
        self.assertTrue(res)

    def test_deg2rad(self):
        # generate input data
        x_deg = 180. * torch.rand(2, 3, 4)

        # convert radians/degrees
        x_rad = dgm.deg2rad(x_deg)
        x_rad_to_deg = dgm.rad2deg(x_rad)

        # compute error
        error = utils.compute_mse(x_deg, x_rad_to_deg)
        self.assertAlmostEqual(error.item(), 0.0, places=4)

    def test_deg2rad_gradcheck(self):
        # generate input data
        x_deg = 180. * torch.rand(2, 3, 4)

        # evaluate function gradient
        res = gradcheck(dgm.deg2rad, (utils.tensor_to_gradcheck_var(x_deg),),
                        raise_exception=True)
        self.assertTrue(res)

    @unittest.skip("")
    def test_inverse_pose(self):
        # generate input data
        batch_size = 1
        eye_size = 4  # identity 4x4
        dst_pose_src = utils.create_random_homography(batch_size, eye_size)
        dst_pose_src[:, -1] = 0.0
        dst_pose_src[:, -1, -1] = 1.0

        # compute the inverse of the pose
        src_pose_dst = dgm.inverse_pose(dst_pose_src)

        # H_inv * H == I
        res = torch.matmul(src_pose_dst, dst_pose_src)
        error = compute_mse(res, utils.create_eye_batch(batch_size, eye_size))
        self.assertAlmostEqual(error.item(), 0.0, places=4)

    def test_inverse_pose_gradcheck(self):
        # generate input data
        batch_size = 2
        eye_size = 4  # identity 4x4
        dst_pose_src = utils.create_random_homography(batch_size, eye_size)
        dst_pose_src = utils.tensor_to_gradcheck_var(dst_pose_src)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.inverse_pose, (dst_pose_src,),
                        raise_exception=True)
        self.assertTrue(res)

    def test_homography_i_H_ref(self):
        # generate input data
        image_height, image_width = 32., 32.
        cx, cy = image_width / 2, image_height / 2
        fx, fy = 1., 1.
        rx, ry, rz = 0., 0., 0.
        tx, ty, tz = 0., 0., 0.
        offset_x = 10.  # we will apply a 10units offset to `i` camera

        pinhole_ref = utils.create_pinhole(fx, fy, cx, cy, \
            image_height, image_width, rx, ry, rx, tx, ty, tz)

        pinhole_i = utils.create_pinhole(fx, fy, cx, cy, \
            image_height, image_width, rx, ry, rx, tx + offset_x, ty, tz)

        # compute homography from ref to i
        i_H_ref = dgm.homography_i_H_ref(pinhole_i, pinhole_ref)
        i_H_ref_inv = dgm.inverse(i_H_ref)

        # compute homography from i to ref
        ref_H_i = dgm.homography_i_H_ref(pinhole_ref, pinhole_i)

        res = utils.check_equal_torch(i_H_ref_inv, ref_H_i)
        self.assertTrue(res)

    @unittest.skip("Jacobian mismatch for output 0 with respect to input 0")
    def test_homography_i_H_ref_gradcheck(self):
        # generate input data
        image_height, image_width = 32., 32.
        cx, cy = image_width / 2, image_height / 2
        fx, fy = 1., 1.
        rx, ry, rz = 0., 0., 0.
        tx, ty, tz = 0., 0., 0.
        offset_x = 10.  # we will apply a 10units offset to `i` camera

        pinhole_ref = utils.create_pinhole(fx, fy, cx, cy, \
            image_height, image_width, rx, ry, rx, tx, ty, tz)
        pinhole_ref = utils.tensor_to_gradcheck_var(pinhole_ref)  # to var

        pinhole_i = utils.create_pinhole(fx, fy, cx, cy, \
            image_height, image_width, rx, ry, rx, tx + offset_x, ty, tz)
        pinhole_i = utils.tensor_to_gradcheck_var(pinhole_ref)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.homography_i_H_ref, (pinhole_i, pinhole_ref,),
                        raise_exception=True)
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()

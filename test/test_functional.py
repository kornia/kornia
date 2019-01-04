import unittest
import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utilities


class Tester(unittest.TestCase):

    def test_inverse_pose(self):
        # generate input data
        batch_size = 1
        eye_size = 4  # identity 4x4
        dst_pose_src = utils.create_random_homography(batch_size, eye_size)
        dst_pose_src[:, -1] = 0.0
        dst_pose_src[:, -1, -1] = 1.0

        # compute the inverse of the pose
        src_pose_dst = tgm.inverse_pose(dst_pose_src)

        # H_inv * H == I
        eye = torch.matmul(src_pose_dst, dst_pose_src)
        res = utils.check_equal_torch(eye, torch.eye(4), eps=1e-3)

    def test_inverse_pose_gradcheck(self):
        # generate input data
        batch_size = 2
        eye_size = 4  # identity 4x4
        dst_pose_src = utils.create_random_homography(batch_size, eye_size)
        dst_pose_src = utils.tensor_to_gradcheck_var(dst_pose_src)  # to var

        # evaluate function gradient
        res = gradcheck(tgm.inverse_pose, (dst_pose_src,),
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
        eps = 1e-6

        pinhole_ref = utils.create_pinhole(
            fx, fy, cx, cy, image_height, image_width, rx, ry, rx, tx, ty, tz)

        pinhole_i = utils.create_pinhole(
            fx,
            fy,
            cx,
            cy,
            image_height,
            image_width,
            rx,
            ry,
            rx,
            tx + offset_x,
            ty,
            tz)

        # compute homography from ref to i
        i_H_ref = tgm.homography_i_H_ref(pinhole_i, pinhole_ref) + eps
        i_H_ref_inv = torch.inverse(i_H_ref)

        # compute homography from i to ref
        ref_H_i = tgm.homography_i_H_ref(pinhole_ref, pinhole_i) + eps

        res = utils.check_equal_torch(i_H_ref_inv, ref_H_i)
        self.assertTrue(res)

    def test_homography_i_H_ref_gradcheck(self):
        # generate input data
        image_height, image_width = 32., 32.
        cx, cy = image_width / 2, image_height / 2
        fx, fy = 1., 1.
        rx, ry, rz = 0., 0., 0.
        tx, ty, tz = 0., 0., 0.
        offset_x = 10.  # we will apply a 10units offset to `i` camera
        eps = 1e-6

        pinhole_ref = utils.create_pinhole(
            fx, fy, cx, cy, image_height, image_width, rx, ry, rx, tx, ty, tz)
        pinhole_ref = utils.tensor_to_gradcheck_var(pinhole_ref)  # to var

        pinhole_i = utils.create_pinhole(
            fx,
            fy,
            cx,
            cy,
            image_height,
            image_width,
            rx,
            ry,
            rx,
            tx + offset_x,
            ty,
            tz)
        pinhole_i = utils.tensor_to_gradcheck_var(pinhole_ref)  # to var

        # evaluate function gradient
        res = gradcheck(
            tgm.homography_i_H_ref,
            (pinhole_i + eps,
             pinhole_ref + eps,
             ),
            raise_exception=True)
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()

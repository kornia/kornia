import unittest
import numpy as np

import torch
import torchgeometry.transforms as tf
from torch.autograd import gradcheck

import utils  # test utils
from utils import check_equal_torch, check_equal_numpy


class Tester(unittest.TestCase):

    def test_angle_axis_to_rotation_matrix_torch(self):
        # generate input data
        batch_size = 2 
        angle_axis = torch.rand(batch_size, 3)
        eye_batch = utils.create_eye_batch(batch_size, 4)

        # apply transform
        rotation_matrix = tf.angle_axis_to_rotation_matrix(angle_axis)

        rotation_matrix_eye = torch.matmul(
            rotation_matrix, rotation_matrix.transpose(1, 2))
        self.assertTrue(check_equal_torch(rotation_matrix_eye, eye_batch))
 
    def test_angle_axis_to_rotation_matrix_numpy(self):
        # generate input data
        angle_axis = np.random.rand(3)

        # apply transform
        rotation_matrix = tf.angle_axis_to_rotation_matrix(angle_axis)

        rotation_matrix_eye = rotation_matrix.dot(rotation_matrix.T)
        self.assertTrue(check_equal_numpy(rotation_matrix_eye, np.eye(4)))

    def test_angle_axis_to_rotation_matrix_gradcheck(self):
        # generate input data
        batch_size = 2
        angle_axis = torch.rand(batch_size, 3)
        angle_axis = utils.tensor_to_gradcheck_var(angle_axis)  # to var

        # apply transform
        rotation_matrix = tf.angle_axis_to_rotation_matrix(angle_axis)

        # evaluate function gradient
        res = gradcheck(tf.angle_axis_to_rotation_matrix, (angle_axis,),
                        raise_exception=True)
        self.assertTrue(res)

    def test_rtvec_to_pose_gradcheck(self):
        # generate input data
        batch_size = 2
        rtvec = torch.rand(batch_size, 6)
        rtvec = utils.tensor_to_gradcheck_var(rtvec)  # to var

        # evaluate function gradient
        res = gradcheck(tf.rtvec_to_pose, (rtvec,), raise_exception=True)
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()

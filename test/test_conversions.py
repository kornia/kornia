import unittest
import numpy as np

import torch
import torchgeometry as tgm
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
        rotation_matrix = tgm.angle_axis_to_rotation_matrix(angle_axis)

        rotation_matrix_eye = torch.matmul(
            rotation_matrix, rotation_matrix.transpose(1, 2))
        self.assertTrue(check_equal_torch(rotation_matrix_eye, eye_batch))

    def test_angle_axis_to_rotation_matrix_numpy(self):
        # generate input data
        angle_axis = np.random.rand(3)

        # apply transform
        rotation_matrix = tgm.angle_axis_to_rotation_matrix(angle_axis)

        rotation_matrix_eye = rotation_matrix.dot(rotation_matrix.T)
        self.assertTrue(check_equal_numpy(rotation_matrix_eye, np.eye(4)))

    def test_angle_axis_to_rotation_matrix_gradcheck(self):
        # generate input data
        batch_size = 2
        angle_axis = torch.rand(batch_size, 3)
        angle_axis = utils.tensor_to_gradcheck_var(angle_axis)  # to var

        # apply transform
        rotation_matrix = tgm.angle_axis_to_rotation_matrix(angle_axis)

        # evaluate function gradient
        res = gradcheck(tgm.angle_axis_to_rotation_matrix, (angle_axis,),
                        raise_exception=True)
        self.assertTrue(res)

    def test_rtvec_to_pose_gradcheck(self):
        # generate input data
        batch_size = 2
        rtvec = torch.rand(batch_size, 6)
        rtvec = utils.tensor_to_gradcheck_var(rtvec)  # to var

        # evaluate function gradient
        res = gradcheck(tgm.rtvec_to_pose, (rtvec,), raise_exception=True)
        self.assertTrue(res)


    def test_rotation_matrix_to_angle_axis_torch(self):
        rmat_1 = torch.tensor([[-0.30382753, -0.95095137, -0.05814062, 0.],
                             [-0.71581715,  0.26812278, -0.64476041,   0.],
                             [ 0.62872461, -0.15427791, -0.76217038,   0.],
                             [ 0.        ,  0.        ,  0.        ,   1.]])
        rvec_1 = torch.tensor([1.50485376, -2.10737739,  0.7214174 ])

        rmat_2 = torch.tensor([[ 0.6027768,  -0.79275544, -0.09054801, 0.],
                               [-0.67915707, -0.56931658,  0.46327563, 0.],
                               [-0.41881476, -0.21775548, -0.88157628, 0.],
                               [ 0.       ,   0.        ,  0.        , 1.]])
        rvec_2 = torch.tensor([-2.44916812, 1.18053411 , 0.4085298 ])
        rmat = torch.stack([rmat_2, rmat_1], dim=0)
        rvec = torch.stack([rvec_2, rvec_1], dim=0)
        self.assertTrue(check_equal_torch(tgm.rotation_matrix_to_angle_axis(rmat), rvec))


    def test_rotation_matrix_to_angle_axis_gradcheck(self):
        # generate input data
        batch_size = 2
        rmat = torch.rand(batch_size, 4, 4)
        rmat = utils.tensor_to_gradcheck_var(rmat)  # to var

        # evaluate function gradient
        res = gradcheck(tgm.rotation_matrix_to_angle_axis, (rmat,), raise_exception=True)
        self.assertTrue(res)


    def test_rotation_matrix_to_angle_axis_numpy(self):
        rmat = np.asarray([[-0.30382753, -0.95095137, -0.05814062, 0.],
                             [-0.71581715, 0.26812278, -0.64476041, 0.],
                             [0.62872461, -0.15427791, -0.76217038, 0.],
                             [0., 0., 0., 1.]])
        rvec = np.array([1.50485376, -2.10737739, 0.7214174])
        self.assertTrue(
            check_equal_numpy(
                tgm.rotation_matrix_to_angle_axis(rmat),
                rvec))

    @unittest.skip('')
    def test_rotation_matrix_to_angle_axis_gradcheck(self):
        print('test_rotation_matrix_to_angle_axis_gradcheck to be implemented :)')  # noqa
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

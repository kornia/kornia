import unittest
import pytest
import numpy as np

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils
from utils import check_equal_torch, check_equal_numpy
from common import TEST_DEVICES


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3),])
def test_convert_points_to_homogeneous(batch_shape, device_type):
    # generate input data
    points = torch.rand(batch_shape)
    points = points.to(torch.device(device_type))

    # to homogeneous
    points_h = tgm.convert_points_to_homogeneous(points)

    assert points_h.shape[-2] == batch_shape[-2]
    assert (points_h[..., -1] == torch.ones(points_h[..., -1].shape)).all()

    # functional
    assert torch.allclose(points_h, tgm.ConvertPointsToHomogeneous()(points))

    # evaluate function gradient
    points = utils.tensor_to_gradcheck_var(points)  # to var
    assert gradcheck(tgm.convert_points_to_homogeneous, (points,),
                     raise_exception=True)



class Tester(unittest.TestCase):

    def test_angle_axis_to_rotation_matrix(self):
        # generate input data
        batch_size = 2
        angle_axis = torch.rand(batch_size, 3)
        eye_batch = utils.create_eye_batch(batch_size, 4)

        # apply transform
        rotation_matrix = tgm.angle_axis_to_rotation_matrix(angle_axis)

        rotation_matrix_eye = torch.matmul(
            rotation_matrix, rotation_matrix.transpose(1, 2))
        self.assertTrue(check_equal_torch(rotation_matrix_eye, eye_batch))

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

    def test_rotation_matrix_to_angle_axis(self):
        rmat_1 = torch.tensor([[-0.30382753, -0.95095137, -0.05814062, 0.],
                               [-0.71581715, 0.26812278, -0.64476041, 0.],
                               [0.62872461, -0.15427791, -0.76217038, 0.]])
        rvec_1 = torch.tensor([1.50485376, -2.10737739, 0.7214174])

        rmat_2 = torch.tensor([[0.6027768, -0.79275544, -0.09054801, 0.],
                               [-0.67915707, -0.56931658, 0.46327563, 0.],
                               [-0.41881476, -0.21775548, -0.88157628, 0.]])
        rvec_2 = torch.tensor([-2.44916812, 1.18053411, 0.4085298])
        rmat = torch.stack([rmat_2, rmat_1], dim=0)
        rvec = torch.stack([rvec_2, rvec_1], dim=0)
        self.assertTrue(
            check_equal_torch(
                tgm.rotation_matrix_to_angle_axis(rmat),
                rvec))

    def test_rotation_matrix_to_angle_axis_gradcheck(self):
        # generate input data
        batch_size = 2
        rmat = torch.rand(batch_size, 3, 4)
        rmat = utils.tensor_to_gradcheck_var(rmat)  # to var

        # evaluate function gradient
        res = gradcheck(tgm.rotation_matrix_to_angle_axis,
                        (rmat,), raise_exception=True)
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()

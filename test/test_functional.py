import unittest

import torch
import torchgeometry as dgm
from torch.autograd import gradcheck


# test utilies

def create_eye_batch(batch_size):
    """Creates a batch of identity matrices of shape Bx3x3
    """
    return torch.eye(3).view(1, 3, 3).expand(batch_size, -1, -1)

def create_random_homography(batch_size, std_val=1e-1):
    """Creates a batch of random homographies of shape Bx3x3
    """
    std = std_val * torch.rand(batch_size, 3, 3)
    eye = create_eye_batch(batch_size)
    return eye + std

def tensor_to_gradcheck_var(tensor):
    """Converts the input tensor to a valid variable to check the gradient.
      `gradcheck` needs 64-bit floating point and requires gradient.
    """
    tensor = tensor.type(torch.DoubleTensor)
    return tensor.requires_grad_(True)


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
        points = tensor_to_gradcheck_var(points)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.convert_points_to_homogeneous, (points,),
                        raise_exception=True)

    def test_convert_points_from_homogeneous(self):
        # generate input data
        batch_size = 2
        points_h = torch.rand(batch_size, 2, 3)
        points_h[..., -1] = 1.0

        # to euclidean
        points = dgm.convert_points_from_homogeneous(points_h)

        error = torch.sum((points_h[..., :2] - points) ** 2)
        self.assertAlmostEqual(error.item(), 0.0)

    def test_convert_points_from_homogeneous_gradcheck(self):
        # generate input data
        batch_size = 2
        points = torch.rand(batch_size, 2, 3)
        points = tensor_to_gradcheck_var(points)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.convert_points_from_homogeneous, (points,),
                        raise_exception=True)

    def test_inverse(self):
        # generate input data
        batch_size = 2
        homographies = create_random_homography(batch_size)
        homographies_inv = dgm.inverse(homographies)

        # H_inv * H == I
        res = torch.matmul(homographies_inv, homographies)
        eye = create_eye_batch(batch_size)
        error = torch.sum((res - eye) ** 2)
        self.assertAlmostEqual(error.item(), 0.0)

    def test_inverse_gradcheck(self):
        # generate input data
        batch_size = 2
        homographies = create_random_homography(batch_size)
        homographies = tensor_to_gradcheck_var(homographies)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.inverse, (homographies,), raise_exception=True)

    def test_transform_points(self):
        # generate input data
        batch_size = 2
        num_points = 2
        num_dims = 2
        points_src = torch.rand(batch_size, 2, num_dims)
        dst_homo_src = create_random_homography(batch_size)

        # transform the points from dst to ref
        points_dst = dgm.transform_points(dst_homo_src, points_src)

        # transform the points from ref to dst
        src_homo_dst = dgm.inverse(dst_homo_src)
        points_dst_to_src = dgm.transform_points(src_homo_dst, points_dst)

        # projected should be equal as initial
        error = torch.sum((points_src - points_dst_to_src) ** 2)
        self.assertAlmostEqual(error.item(), 0.0)

    def test_transform_points_gradcheck(self):
        # generate input data
        batch_size = 2
        num_points = 2
        num_dims = 2
        points_src = torch.rand(batch_size, 2, num_dims)
        points_src = tensor_to_gradcheck_var(points_src)      # to var
        dst_homo_src = create_random_homography(batch_size)
        dst_homo_src = tensor_to_gradcheck_var(dst_homo_src)  # to var

        # evaluate function gradient
        res = gradcheck(dgm.transform_points, (dst_homo_src, points_src,),
                        raise_exception=True)

    def test_pi(self):
        self.assertAlmostEqual(dgm.pi.item(), 3.141592, places=4)

    def test_rad2deg(self):
        # generate input data
        x_rad = dgm.pi * torch.rand(2, 3, 4)

        # convert radians/degrees
        x_deg = dgm.rad2deg(x_rad)
        x_deg_to_rad = dgm.deg2rad(x_deg)

        # compute error
        error = torch.sum((x_rad - x_deg_to_rad) ** 2)
        self.assertAlmostEqual(error.item(), 0.0)
        
    def test_rad2deg_gradcheck(self):
        # generate input data
        x_rad = dgm.pi * torch.rand(2, 3, 4)

        # evaluate function gradient
        res = gradcheck(dgm.rad2deg, (tensor_to_gradcheck_var(x_rad),),
                        raise_exception=True)

    def test_deg2rad(self):
        # generate input data
        x_deg = 180. * torch.rand(2, 3, 4)

        # convert radians/degrees
        x_rad = dgm.deg2rad(x_deg)
        x_rad_to_deg = dgm.rad2deg(x_rad)

        # compute error
        error = torch.sum((x_deg - x_rad_to_deg) ** 2)
        self.assertAlmostEqual(error.item(), 0.0)
        
    def test_deg2rad_gradcheck(self):
        # generate input data
        x_deg = 180. * torch.rand(2, 3, 4)

        # evaluate function gradient
        res = gradcheck(dgm.deg2rad, (tensor_to_gradcheck_var(x_deg),),
                        raise_exception=True)


if __name__ == '__main__':
    unittest.main()

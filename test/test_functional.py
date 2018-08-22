import unittest

import torch
import torchgeometry as dgm

# test utilies

def create_eye_batch(batch_size):
    return torch.eye(3).view(1, 3, 3).expand(batch_size, -1, -1)

def create_random_homography(batch_size, std_val=1e-1):
    std = std_val * torch.rand(batch_size, 3, 3)
    eye = create_eye_batch(batch_size)
    return eye + std


class Tester(unittest.TestCase):

    def test_convert_points_to_homogeneous(self):
        # generate input data
        batch_size = 2
        points = torch.rand(batch_size, 2, 3)

        # to homogeneous
        points_h = dgm.convert_points_to_homogeneous(points)
        self.assertTrue((points_h[..., -1] == torch.ones(1, 2, 1)).all())

    def test_convert_points_from_homogeneous(self):
        # generate input data
        batch_size = 2
        points_h = torch.rand(batch_size, 2, 3)
        points_h[..., -1] = 1.0

        # to euclidean
        points = dgm.convert_points_from_homogeneous(points_h)

        error = torch.sum((points_h[..., :2] - points) ** 2)
        self.assertAlmostEqual(error, 0.0)

    def test_inverse(self):
        # generate input data
        batch_size = 2
        homographies = create_random_homography(batch_size)
        homographies_inv = dgm.inverse(homographies)

        # H_inv * H == I
        res = torch.matmul(homographies_inv, homographies)
        eye = create_eye_batch(batch_size)
        error = torch.sum((res - eye) ** 2)
        self.assertAlmostEqual(error, 0.0)

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
        self.assertAlmostEqual(error, 0.0)


if __name__ == '__main__':
    unittest.main()

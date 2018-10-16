import unittest
import numpy as np

import torch
import torchgeometry as tgm

import utils  # test utils


class Tester(unittest.TestCase):

    def test_tensor_to_image(self):
        tensor = torch.ones(3, 4, 4)
        image = tgm.tensor_to_image(tensor)

        self.assertTrue(image.shape == (4, 4, 3))
        self.assertIsInstance(image, np.ndarray)

    def test_image_to_tensor(self):
        image = np.ones((4, 4, 3))
        tensor = tgm.image_to_tensor(image)

        self.assertTrue(tensor.shape == (3, 4, 4))
        self.assertIsInstance(tensor, torch.Tensor)

    def test_create_meshgrid(self):
        height, width = 4, 6
        normalized_coordinates = False

        # create the meshgrid and verify shape
        grid = tgm.create_meshgrid(height, width, normalized_coordinates)
        self.assertTrue(grid.shape == (1, height, width, 2))

        # check grid corner values
        self.assertTrue(tuple(grid[0, 0, 0].numpy()) == (0., 0.))
        self.assertTrue(tuple(
            grid[0, height - 1, width - 1].numpy()) == (width - 1, height - 1))

    def test_normalize_pixel_grid(self):
        # generate input data
        batch_size = 1
        height, width = 2, 4

        # create points grid
        grid_norm = tgm.create_meshgrid(height, width,
                                        normalized_coordinates=True)
        grid_norm = torch.unsqueeze(grid_norm, dim=0)
        grid_pix = tgm.create_meshgrid(height, width,
                                       normalized_coordinates=False)
        grid_pix = torch.unsqueeze(grid_pix, dim=0)

        # grid from pixel space to normalized
        norm_trans_pix = tgm.normal_transform_pixel(height, width)  # 1x3x3
        pix_trans_norm = tgm.inverse(norm_trans_pix)  # 1x3x3
        # transform grids
        grid_pix_to_norm = tgm.transform_points(norm_trans_pix, grid_pix)
        grid_norm_to_pix = tgm.transform_points(pix_trans_norm, grid_norm)
        self.assertTrue(utils.check_equal_torch(grid_pix, grid_norm_to_pix))
        self.assertTrue(utils.check_equal_torch(grid_norm, grid_pix_to_norm))


if __name__ == '__main__':
    unittest.main()

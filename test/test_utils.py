import unittest
import numpy as np

import torch
import torchgeometry as tgm


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


if __name__ == '__main__':
    unittest.main()

import pytest
import os

import kornia
import torch
from torch.testing import assert_allclose


class TestSaveLoadPointCloud:
    def test_save_pointcloud(self):
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        # save to file
        filename = "pointcloud.ply"
        kornia.save_pointcloud_ply(filename, xyz_save)

        # load file
        xyz_load = kornia.load_pointcloud_ply(filename)
        assert_allclose(xyz_save.reshape(-1, 3), xyz_load)

        # remove the temporal file
        if os.path.exists(filename):
            os.remove(filename)

import os

import torch
from kornia.testing import assert_close

import kornia


class TestSaveLoadPointCloud:
    def test_save_pointcloud(self):
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        # save to file
        filename = "pointcloud.ply"
        kornia.save_pointcloud_ply(filename, xyz_save)

        # load file
        xyz_load = kornia.load_pointcloud_ply(filename)
        assert_close(xyz_save.reshape(-1, 3), xyz_load)

        # remove the temporal file
        if os.path.exists(filename):
            os.remove(filename)

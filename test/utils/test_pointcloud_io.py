from __future__ import annotations

import os

import torch

import kornia
from kornia.testing import assert_close


class TestSaveLoadPointCloud:
    def test_save_pointcloud(self):
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        # save to file
        filename = "pointcloud.ply"
        kornia.utils.save_pointcloud_ply(filename, xyz_save)

        # load file
        xyz_load = kornia.utils.load_pointcloud_ply(filename)
        assert_close(xyz_save.reshape(-1, 3), xyz_load)

        # remove the temporal file
        if os.path.exists(filename):
            os.remove(filename)

    @staticmethod
    def test_inf_coordinates_save_pointcloud():
        # create the tensor to save
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        # add nans or infinite values
        # point data (inf, inf, inf) skipped
        xyz_save[0, 0, :] = float('inf')
        # point data (inf, number, number) counted
        xyz_save[0, 1, 0] = float('inf')
        # point data (inf, inf, number) counted
        xyz_save[1, 0, :-1] = float('inf')

        # save to file
        filename = "pointcloud.ply"
        kornia.utils.save_pointcloud_ply(filename, xyz_save)

        # same as xyz_save but with dropping the first point
        xyz_correct = xyz_save.reshape(-1, 3)[1:, :]

        # load file
        xyz_load = kornia.utils.load_pointcloud_ply(filename)
        assert_close(xyz_correct, xyz_load)

        if os.path.exists(filename):
            os.remove(filename)

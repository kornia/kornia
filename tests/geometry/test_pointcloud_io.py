# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import numpy as np
import pytest
import torch

import kornia

from testing.base import BaseTester


class TestSaveLoadPointCloud(BaseTester):
    def test_save_pointcloud(self):
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        filename = "pointcloud.ply"
        kornia.geometry.save_pointcloud_ply(filename, xyz_save)

        xyz_load = kornia.geometry.load_pointcloud_ply(filename)
        self.assert_close(xyz_save.reshape(-1, 3), xyz_load)

        if os.path.exists(filename):
            os.remove(filename)

    def test_inf_coordinates_save_pointcloud(self):
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        xyz_save[0, 0, :] = float("inf")  # all inf → skipped
        xyz_save[0, 1, 0] = float("inf")  # partial inf → kept
        xyz_save[1, 0, :-1] = float("inf")  # partial inf → kept

        filename = "pointcloud.ply"
        kornia.geometry.save_pointcloud_ply(filename, xyz_save)

        xyz_correct = xyz_save.reshape(-1, 3)[1:, :]

        xyz_load = kornia.geometry.load_pointcloud_ply(filename)
        self.assert_close(xyz_correct, xyz_load)

        if os.path.exists(filename):
            os.remove(filename)

    def test_invalid_filename_type(self):
        xyz_save = torch.rand(10, 3)
        with pytest.raises(TypeError):
            kornia.geometry.save_pointcloud_ply(1234, xyz_save)

    def test_invalid_filename_extension(self):
        xyz_save = torch.rand(10, 3)
        with pytest.raises(TypeError):
            kornia.geometry.save_pointcloud_ply("pointcloud.txt", xyz_save)

    def test_invalid_pointcloud_type(self):
        with pytest.raises(TypeError):
            kornia.geometry.save_pointcloud_ply("pointcloud.ply", [[1, 2, 3]])

    def test_invalid_pointcloud_shape(self):
        xyz_save = torch.rand(10, 4)
        with pytest.raises(TypeError):
            kornia.geometry.save_pointcloud_ply("pointcloud.ply", xyz_save)

    def test_save_pointcloud_with_nan(self):
        xyz_save = torch.rand(5, 3)
        xyz_save[0, :] = float("nan")
        xyz_save[1, 0] = float("nan")
        filename = "pointcloud_nan.ply"
        kornia.geometry.save_pointcloud_ply(filename, xyz_save)
        xyz_load = kornia.geometry.load_pointcloud_ply(filename)
        expected = xyz_save[torch.isfinite(xyz_save).any(dim=1)]

        # Use numpy to compare with NaNs considered equal
        np.testing.assert_allclose(
            expected.detach().cpu().numpy(),
            xyz_load.detach().cpu().numpy(),
            atol=1e-9,
            equal_nan=True,
        )

        if os.path.exists(filename):
            os.remove(filename)

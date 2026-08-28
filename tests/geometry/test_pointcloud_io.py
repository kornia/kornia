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

    def test_save_pointcloud_binary(self):
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        filename = "pointcloud_binary.ply"
        kornia.geometry.save_pointcloud_ply_binary(filename, xyz_save)

        xyz_load = kornia.geometry.load_pointcloud_ply_binary(filename)
        self.assert_close(xyz_save.reshape(-1, 3), xyz_load)

        if os.path.exists(filename):
            os.remove(filename)

    def test_save_pointcloud_binary_with_nan_inf(self):
        xyz_save = torch.rand(5, 3)
        xyz_save[0, :] = float("nan")
        xyz_save[1, 0] = float("inf")
        filename = "pointcloud_binary_nan_inf.ply"
        kornia.geometry.save_pointcloud_ply_binary(filename, xyz_save)
        xyz_load = kornia.geometry.load_pointcloud_ply_binary(filename)
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

    def test_load_pointcloud_binary_stops_at_declared_vertex_count(self, tmp_path):
        xyz_save = torch.tensor([[1.0, 2.0, 3.0]])
        filename = tmp_path / "pointcloud_binary_with_trailing_data.ply"
        kornia.geometry.save_pointcloud_ply_binary(str(filename), xyz_save)

        # A PLY file may contain elements after its vertices (for example, faces).
        # Use 24 trailing bytes to expose loaders that infer the vertex count from
        # the entire remaining payload instead of the header's element declaration.
        with filename.open("ab") as file:
            file.write(bytes(24))

        xyz_load = kornia.geometry.load_pointcloud_ply_binary(str(filename))

        self.assert_close(xyz_load, xyz_save)

    def test_load_pointcloud_binary_finds_end_header(self, tmp_path):
        xyz_save = torch.tensor([[1.0, 2.0, 3.0]])
        filename = tmp_path / "pointcloud_binary_long_header.ply"
        kornia.geometry.save_pointcloud_ply_binary(str(filename), xyz_save)

        header, payload = filename.read_bytes().split(b"end_header\n", maxsplit=1)
        filename.write_bytes(header + b"comment extra header line\nend_header\n" + payload)

        xyz_load = kornia.geometry.load_pointcloud_ply_binary(str(filename))

        self.assert_close(xyz_load, xyz_save)

    @pytest.mark.parametrize("vertex_count", ["-1", "invalid"])
    def test_load_pointcloud_binary_rejects_invalid_vertex_count(self, tmp_path, vertex_count):
        filename = tmp_path / "pointcloud_binary_invalid_count.ply"
        filename.write_bytes(
            b"ply\n"
            b"format binary_little_endian 1.0\n"
            + f"element vertex {vertex_count}\n".encode()
            + b"property double x\nproperty double y\nproperty double z\nend_header\n"
        )

        with pytest.raises(ValueError, match="PLY vertex count"):
            kornia.geometry.load_pointcloud_ply_binary(str(filename))

    def test_load_pointcloud_binary_empty(self, tmp_path):
        xyz_save = torch.empty((0, 3))
        filename = tmp_path / "pointcloud_binary_empty.ply"
        kornia.geometry.save_pointcloud_ply_binary(str(filename), xyz_save)

        xyz_load = kornia.geometry.load_pointcloud_ply_binary(str(filename))

        self.assert_close(xyz_load, xyz_save)

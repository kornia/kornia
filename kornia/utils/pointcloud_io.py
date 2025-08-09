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
import torch


def save_pointcloud_ply(filename: str, pointcloud: torch.Tensor) -> None:
    r"""Save to disk a pointcloud in PLY format.

    Args:
        filename: the path to save the pointcloud.
        pointcloud: tensor containing the pointcloud to save.
          The tensor must be in the shape of :math:`(*, 3)` where the last
          component is assumed to be a 3d point coordinate :math:`(X, Y, Z)`.

    """
    if not (isinstance(filename, str) and filename.lower().endswith(".ply")):
        raise TypeError(f"Input filename must be a string with the .ply extension. Got {filename!r}")

    if not torch.is_tensor(pointcloud):
        raise TypeError(f"Input pointcloud type is not a torch.Tensor. Got {type(pointcloud)}")

    if pointcloud.ndim < 2 or pointcloud.shape[-1] != 3:
        raise TypeError(f"Input pointcloud must have shape (..., 3). Got {tuple(pointcloud.shape)}")

    # Flatten points
    xyz = pointcloud.reshape(-1, 3)

    valid_mask = torch.isfinite(xyz).any(dim=1)
    valid_count = int(valid_mask.sum().item())

    with open(filename, "w") as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment arraiy generated\n")
        f.write(f"element vertex {valid_count}\n")
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("end_header\n")

        if valid_count > 0:
            arr = xyz[valid_mask].detach().cpu().numpy()
            np.savetxt(f, arr, fmt="%.9g", delimiter=" ")


def load_pointcloud_ply(filename: str, header_size: int = 8) -> torch.Tensor:
    r"""Load from disk a pointcloud in PLY format.

    Args:
        filename: the path to the pointcloud.
        header_size: the size of the ply file header that will
          be skipped during loading.

    Return:
        tensor containing the loaded point with shape :math:`(*, 3)` where
        :math:`*` represents the number of points.

    """
    if not isinstance(filename, str) and filename[-3:] == ".ply":
        raise TypeError(f"Input filename must be a string in with the .ply  extension. Got {filename}")
    if not os.path.isfile(filename):
        raise ValueError("Input filename is not an existing file.")
    if not (isinstance(header_size, int) and header_size > 0):
        raise TypeError(f"Input header_size must be a positive integer. Got {header_size}.")
    # open the file and populate tensor
    with open(filename) as f:
        points = []

        # skip header
        lines = f.readlines()[header_size:]

        # iterate over the points
        for line in lines:
            x_str, y_str, z_str = line.split()
            points.append((torch.tensor(float(x_str)), torch.tensor(float(y_str)), torch.tensor(float(z_str))))

        # create tensor from list
        pointcloud: torch.Tensor = torch.tensor(points)
        return pointcloud

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
    valid_points = xyz[valid_mask]
    valid_count = valid_points.shape[0]

    with open(filename, "w", encoding="utf-8") as f:
        # Write PLY header
        f.writelines(
            [
                "ply\n",
                "format ascii 1.0\n",
                "comment arraiy generated\n",
                f"element vertex {valid_count}\n",
                "property double x\n",
                "property double y\n",
                "property double z\n",
                "end_header\n",
            ]
        )

        if valid_count > 0:
            # Move to CPU, convert to float64 for matching 'double' in header
            arr = valid_points.detach().cpu().to(torch.float64)
            # Write each row as space-separated floats
            for x, y, z in arr.tolist():
                f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")


def load_pointcloud_ply(filename: str, header_size: int = 8) -> torch.Tensor:
    r"""Load from disk a pointcloud in PLY format.

    Args:
        filename: the path to the pointcloud.
        header_size: the number of header lines to skip.

    Return:
        tensor containing the loaded points with shape :math:`(*, 3)` where
        :math:`*` represents the number of points.
    """
    if not (isinstance(filename, str) and filename.lower().endswith(".ply")):
        raise TypeError(f"Input filename must be a string with the .ply extension. Got {filename!r}")
    if not os.path.isfile(filename):
        raise ValueError("Input filename is not an existing file.")
    if not (isinstance(header_size, int) and header_size > 0):
        raise TypeError(f"Input header_size must be a positive integer. Got {header_size}.")

    # Read all file bytes
    with open(filename, "rb") as f:
        # Skip header lines
        for _ in range(header_size):
            f.readline()
        raw_data = f.read()

    # Decode once and split (faster than line-by-line parsing in Python)
    text = raw_data.decode("utf-8", errors="ignore")
    parts = text.split()

    # We only take the first 3 columns per point
    if len(parts) % 3 != 0:
        raise ValueError(f"Expected 3 columns per point, got a total of {len(parts)} values.")

    # Convert directly to a float32 tensor in one go
    tensor = torch.tensor(list(map(float, parts[: (len(parts) // 3) * 3])), dtype=torch.float32).view(-1, 3)
    return tensor

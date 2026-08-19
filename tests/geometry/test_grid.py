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

import pytest
import torch

import kornia

from testing.base import assert_close


def test_create_meshgrid(device, dtype):
    height, width = 4, 6
    normalized_coordinates = False

    # create the meshgrid and verify shape
    grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates, device=device, dtype=dtype)

    assert grid.device == device
    assert grid.dtype == dtype
    assert grid.shape == (1, height, width, 2)

    # check grid corner values
    assert tuple(grid[0, 0, 0].cpu().numpy()) == (0.0, 0.0)
    assert tuple(grid[0, height - 1, width - 1].cpu().numpy()) == (width - 1, height - 1)


def test_normalize_pixel_grid(device, dtype):
    if device.type == "cuda" and dtype == torch.float16:
        pytest.skip('"inverse_cuda" not implemented for "Half"')

    # generate input data
    height, width = 2, 4

    # create points grid
    grid_norm = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device, dtype=dtype)

    assert grid_norm.device == device
    assert grid_norm.dtype == dtype
    grid_norm = torch.unsqueeze(grid_norm, dim=0)

    grid_pix = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device, dtype=dtype)

    assert grid_pix.device == device
    assert grid_pix.dtype == dtype
    grid_pix = torch.unsqueeze(grid_pix, dim=0)

    # grid from pixel space to normalized
    norm_trans_pix = kornia.geometry.conversions.normal_transform_pixel(
        height, width, device=device, dtype=dtype
    )  # 1x3x3
    pix_trans_norm = torch.inverse(norm_trans_pix)  # 1x3x3
    # transform grids
    grid_pix_to_norm = kornia.geometry.linalg.transform_points(norm_trans_pix, grid_pix)
    grid_norm_to_pix = kornia.geometry.linalg.transform_points(pix_trans_norm, grid_norm)
    assert_close(grid_pix, grid_norm_to_pix)
    assert_close(grid_norm, grid_pix_to_norm)


def test_create_meshgrid_align_corners(device, dtype):
    # align_corners selects which grid_sample normalization the grid is built for: True puts pixel
    # centers 0 and W-1 at -1 and +1, False puts the outer pixel EDGES there (#3904).
    height, width = 2, 4

    grid_true = kornia.geometry.create_meshgrid(height, width, True, device=device, dtype=dtype, align_corners=True)
    grid_false = kornia.geometry.create_meshgrid(height, width, True, device=device, dtype=dtype, align_corners=False)

    # align_corners=True is the default, so the bare call must match it
    assert_close(kornia.geometry.create_meshgrid(height, width, True, device=device, dtype=dtype), grid_true)

    assert_close(grid_true[0, :, 0, 0], torch.tensor([-1.0, -1.0], device=device, dtype=dtype))
    assert_close(grid_true[0, :, -1, 0], torch.tensor([1.0, 1.0], device=device, dtype=dtype))
    # False: x centers land at (2x + 1)/W - 1, i.e. -0.75 and 0.75 for W = 4
    assert_close(grid_false[0, :, 0, 0], torch.tensor([-0.75, -0.75], device=device, dtype=dtype))
    assert_close(grid_false[0, :, -1, 0], torch.tensor([0.75, 0.75], device=device, dtype=dtype))

    # align_corners is only about the normalization, so it is a no-op for pixel coordinates
    assert_close(
        kornia.geometry.create_meshgrid(height, width, False, device=device, dtype=dtype, align_corners=True),
        kornia.geometry.create_meshgrid(height, width, False, device=device, dtype=dtype, align_corners=False),
    )


def test_create_meshgrid3d(device, dtype):
    depth, height, width = 5, 4, 6
    normalized_coordinates = False

    # create the meshgrid and verify shape
    grid = kornia.geometry.create_meshgrid3d(depth, height, width, normalized_coordinates, device=device, dtype=dtype)

    assert grid.device == device
    assert grid.dtype == dtype
    assert grid.shape == (1, depth, height, width, 3)

    # check grid corner values
    assert tuple(grid[0, 0, 0, 0].cpu().numpy()) == (0.0, 0.0, 0.0)
    assert tuple(grid[0, depth - 1, height - 1, width - 1].cpu().numpy()) == (depth - 1, width - 1, height - 1)

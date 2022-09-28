from __future__ import annotations

import pytest
import torch

import kornia
from kornia.testing import assert_close


def test_create_meshgrid(device, dtype):
    height, width = 4, 6
    normalized_coordinates = False

    # create the meshgrid and verify shape
    grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates, device=device, dtype=dtype)

    assert grid.device == device
    assert grid.dtype == dtype
    assert grid.shape == (1, height, width, 2)

    # check grid corner values
    assert tuple(grid[0, 0, 0].cpu().numpy()) == (0.0, 0.0)
    assert tuple(grid[0, height - 1, width - 1].cpu().numpy()) == (width - 1, height - 1)


def test_normalize_pixel_grid(device, dtype):
    if device.type == 'cuda' and dtype == torch.float16:
        pytest.skip('"inverse_cuda" not implemented for "Half"')

    # generate input data
    height, width = 2, 4

    # create points grid
    grid_norm = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True, device=device, dtype=dtype)

    assert grid_norm.device == device
    assert grid_norm.dtype == dtype
    grid_norm = torch.unsqueeze(grid_norm, dim=0)

    grid_pix = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device, dtype=dtype)

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


def test_create_meshgrid3d(device, dtype):
    depth, height, width = 5, 4, 6
    normalized_coordinates = False

    # create the meshgrid and verify shape
    grid = kornia.utils.create_meshgrid3d(depth, height, width, normalized_coordinates, device=device, dtype=dtype)

    assert grid.device == device
    assert grid.dtype == dtype
    assert grid.shape == (1, depth, height, width, 3)

    # check grid corner values
    assert tuple(grid[0, 0, 0, 0].cpu().numpy()) == (0.0, 0.0, 0.0)
    assert tuple(grid[0, depth - 1, height - 1, width - 1].cpu().numpy()) == (depth - 1, width - 1, height - 1)

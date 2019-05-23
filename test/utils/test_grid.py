import pytest

import torch
import kornia as kornia

import utils  # test utils


def test_create_meshgrid():
    height, width = 4, 6
    normalized_coordinates = False

    # create the meshgrid and verify shape
    grid = kornia.utils.create_meshgrid(
        height, width, normalized_coordinates)
    assert grid.shape == (1, height, width, 2)

    # check grid corner values
    assert tuple(grid[0, 0, 0].numpy()) == (0., 0.)
    assert tuple(
        grid[0, height - 1, width - 1].numpy()) == (width - 1, height - 1)


def test_normalize_pixel_grid():
    # generate input data
    batch_size = 1
    height, width = 2, 4

    # create points grid
    grid_norm = kornia.utils.create_meshgrid(
        height, width, normalized_coordinates=True)
    grid_norm = torch.unsqueeze(grid_norm, dim=0)
    grid_pix = kornia.utils.create_meshgrid(
        height, width, normalized_coordinates=False)
    grid_pix = torch.unsqueeze(grid_pix, dim=0)

    # grid from pixel space to normalized
    norm_trans_pix = kornia.normal_transform_pixel(height, width)  # 1x3x3
    pix_trans_norm = torch.inverse(norm_trans_pix)  # 1x3x3
    # transform grids
    grid_pix_to_norm = kornia.transform_points(norm_trans_pix, grid_pix)
    grid_norm_to_pix = kornia.transform_points(pix_trans_norm, grid_norm)
    assert utils.check_equal_torch(grid_pix, grid_norm_to_pix)
    assert utils.check_equal_torch(grid_norm, grid_pix_to_norm)

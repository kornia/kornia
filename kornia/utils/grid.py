from typing import Optional, Union

import torch
import kornia as K


def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: bool = True,
        device: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def create_meshgrid3d(
        depth: int,
        height: int,
        width: int,
        normalized_coordinates: bool = True,
        device: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        depth (int): the image depth (channels).
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, D, H, W, 3)`.
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    zs: torch.Tensor = torch.linspace(0, depth - 1, depth, device=device, dtype=torch.float)
    # Fix TracerWarning
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
        zs = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([zs, xs, ys])).transpose(1, 2)  # 3xHxW
    return base_grid.unsqueeze(0).permute(0, 3, 4, 2, 1)  # 1xHxWx3

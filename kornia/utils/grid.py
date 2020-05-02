from typing import Optional

import torch


@torch.jit.ignore
def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: Optional[bool] = True,
        device: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def create_meshgrid3d(
        depth: int,
        height: int,
        width: int,
        normalized_coordinates: Optional[bool] = True,
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
        normalized_coordinates (Optional[bool]): wether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, D, H, W, 3)`.
    """
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    zs: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float)
        zs = torch.linspace(-1, 1, depth, device=device, dtype=torch.float)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
        zs = torch.linspace(0, depth - 1, depth, device=device, dtype=torch.float)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([zs, xs, ys])).transpose(1, 2)  # 3xHxW
    return base_grid.unsqueeze(0).permute(0, 3, 4, 2, 1)  # 1xHxWx3

from typing import Union, Tuple, cast

import torch
import torch.nn as nn

import kornia


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _get_perspective_params(
    batch_size: int,
    width: int, height: int,
    distortion_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        batch_size (int): the tensor batch size.
        width (int): width of the image.
        height (int) : height of the image.
        distortion_scale (float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

    Returns:
        List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        The points are in -x order.
    """
    start_points: torch.Tensor = torch.tensor([[
        [0., 0],
        [0, width - 1],
        [height - 1, 0],
        [height - 1, width - 1],
    ]]).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx: float = distortion_scale * width / 2
    fy: float = distortion_scale * height / 2

    factor = torch.tensor([fy, fx]).view(-1, 1, 2)
    offset = factor * torch.rand(batch_size, 4, 2) * 2 - 1

    end_points = start_points + offset

    return start_points, end_points


# TODO: implement apply_perspective

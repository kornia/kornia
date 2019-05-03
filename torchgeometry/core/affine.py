from typing import Union

import torch
import torch.nn as nn

from torchgeometry.core import warp_affine, get_rotation_matrix2d


def compute_rotation_center(tensor: torch.Tensor) -> torch.Tensor:
    """Computes the center of tensor plane."""
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor([center_x, center_y],
        device=tensor.device, dtype=tensor.dtype)
    return center


def compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Computes a pure rotation matrix."""
    scale: torch.Tensor = torch.ones_like(angle)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166

def affine(tensor: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    r"""Apply an affine transformation to the image.
    
    Args:
        tensor (torch.Tensor): The image tensor to be warped.
        matrix (torch.Tensor): The 2x3 affine transformation matrix.
    
    Returns:
        torch.Tensor: The warped image.
    """
    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    warped: torch.Tensor = warp_affine(tensor, matrix, tensor.shape[-2:])

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185

def rotate(tensor: torch.Tensor, angle: torch.Tensor,
           center: Union[None, torch.Tensor] = None) -> torch.Tensor:
    r"""Rotate the image anti-clockwise about the centre.
    
    Args:
        tensor (torch.Tensor): The image tensor to be rotated.
        angle (torch.Tensor): The angle through which to rotate.
        center (torch.Tensor): The center through which to rotate. The tensor
          must have a shape of :math:(B, 2), where B is batch size and last
          dimension contains cx and cy.
    Returns:
        torch.Tensor: The rotated image tensor.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}"
                        .format(type(angle)))
    if center is not None and not torch.is_tensor(angle):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the rotation center
    if center is None:
        center: torch.Tensor = compute_rotation_center(tensor)
        # TODO: add broadcasting to get_rotation_matrix2d for center
        center = center.expand(angle.shape[0], -1)

    # compute the rotation matrix
    rotation_matrix: torch.Tensor = compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3])


class Rotate(nn.Module):
    r"""Rotate the tensor anti-clockwise about the centre.
    
    Args:
        angle (torch.Tensor): The angle through which to rotate.
        center (torch.Tensor): The center through which to rotate. The tensor
          must have a shape of :math:(B, 2), where B is batch size and last
          dimension contains cx and cy.
    Returns:
        torch.Tensor: The rotated tensor.
    """
    def __init__(self, angle: torch.Tensor,
            center: Union[None, torch.Tensor] = None) -> None:
        super(Rotate, self).__init__()
        self.angle: torch.Tensor = angle
        self.center: Union[None, torch.Tensor] = center

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rotate(input, self.angle, self.center)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            'angle={0}, center={1})' \
            .format(self.angle.item(), self.center)


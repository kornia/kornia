from typing import Union, Tuple, Optional
import numbers

import torch
import torch.nn as nn

from torchgeometry.core import get_rotation_matrix2d, warp_affine


class RandomRotation(nn.Module):
    def __init__(self, degrees, center=None):
        super(RandomRotation, self).__init__()
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                 raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.center = center
    
    @staticmethod
    def get_params(degrees):
        return torch.FloatTensor(1).uniform_(degrees[0], degrees[1]).item()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.rotate(img, self.get_params(self.degrees), self.center)


class Rotate(nn.Module):
    r"""Rotate the image anti-clockwise about the centre.
    
    Args:
        tensor (torch.Tensor): The image tensor to be rotated.
        degrees (float): The angle through which to rotate.

    Returns:
        torch.Tensor: The rotated image tensor.
    """
    def __init__(self,
            degrees: Union[float, Tuple[float]],
            center: Union[None, Tuple[float]] = None) -> None:
        super(Rotate, self).__init__()
        self.degrees: Union[float, Tuple[float]] = degrees
        self.center: Union[None, Tuple[float]] = center

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rotate(input, self.degrees, self.center)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            'degrees={0}, center={1})' \
            .format(self.degrees, self.center)


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185

def rotate(tensor: torch.Tensor, degrees: Union[float, Tuple[float]],
           center: Union[None, Tuple[float]] = None) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # degrees must be a list
    if isinstance(degrees, numbers.Number):
        degrees = [degrees]

    if len(tensor.shape) == 4 and tensor.shape[0] != len(degrees):
        raise ValueError("Input tensor shape a degrees length must match. "
                         "Got tensor: {0} and degrees length: {1}"
                         .format(tensor.shape, len(degrees)))

    # convert parameters to tensor
    degrees_t: torch.Tensor = torch.Tensor(degrees)
    scale_t: torch.Tensor = torch.ones_like(degrees_t)

    # compute rotation center
    center_t: Union[Tuple[float], torch.Tensor] = None
    if center is None:
        height, width = tensor.shape[-2:]
        center_x: float = float(width - 1) / 2
        center_y: float = float(height - 1) / 2
        center_t = [center_x, center_y]
    center_t = torch.Tensor([center_t])

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    center_t = center_t.expand(degrees_t.shape[0], -1)
    rotation_matrix: torch.Tensor = get_rotation_matrix2d(
        center_t, degrees_t, scale_t)

    return affine(tensor, rotation_matrix)

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

    # warp the input tensor
    warped: torch.Tensor = warp_affine(tensor, matrix, tensor.shape[-2:])

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped

from typing import Union, Tuple, Optional
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeometry.core import get_rotation_matrix2d, warp_affine


def compute_rotation_center(tensor: torch.Tensor) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.Tensor([center_x, center_y])
    return center


def convert_matrix2d_to_homogeneous(matrix: torch.Tensor) -> torch.Tensor:
    # pad last two dimensions with zeros
    matrix_h: torch.Tensor = F.pad(matrix, (0, 0, 0, 1), "constant", 0.0)
    matrix_h[..., -1, -1] += 1.
    return matrix_h


def identity_matrix() -> torch.Tensor:
    return torch.eye(3)[None]


def rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    scale: torch.Tensor = torch.ones_like(angle)
    matrix: torch.Tensor = convert_matrix2d_to_homogeneous(
        get_rotation_matrix2d(center, angle, scale))
    return matrix


class RotationMatrix(nn.Module):
    def __init__(self,
            angle: torch.Tensor, center: torch.Tensor) -> None:
        super(RotationMatrix, self).__init__()
        self.angle: torch.Tensor = angle
        self.center: torch.Tensor = center
        self.matrix: torch.Tensor = self._generate_matrix()

    def _generate_matrix(self) -> torch.Tensor:
        assert self.angle is not None
        assert self.center is not None
        return rotation_matrix(self.angle, self.center)

    def affine(self) -> torch.Tensor:
        assert self.matrix is not None
        return self.matrix[..., :2, :3]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self.matrix is not None
        return torch.matmul(input, self.matrix)


class RandomRotationMatrix(nn.Module):
    def __init__(self,
            degrees: torch.Tensor,
            center: Union[None, torch.Tensor] = None) -> None:
        super(RandomRotationMatrix, self).__init__()
        if len(degrees.shape) == 1:
            if bool(degrees < 0):
                 raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = torch.cat([-degrees, degrees])
        else:
            if len(degrees.shape) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.center = center
        self.angle = None
        self.matrix = None
    
    def _generate_params(self, num_samples: int) -> torch.Tensor:
        degrees: Tuple[float] = self.degrees.tolist()
        angle: torch.Tensor = torch.tensor(
            [float(num_samples)]).uniform_(degrees[0], degrees[1])
        return angle

    def _generate_matrix(self, num_samples: int) -> torch.Tensor:
        assert self.center is not None
        self.angle: torch.Tensor = self._generate_params(num_samples)
        return rotation_matrix(self.angle, self.center)

    def affine(self) -> torch.Tensor:
        assert self.matrix is not None
        return self.matrix[..., :2, :3]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 3, input.shape
        assert self.center is not None, self.center
        num_samples: int = input.shape[0]
        self.matrix: torch.Tensor = self._generate_matrix(num_samples)
        return torch.matmul(input, self.matrix)


class Rotate(nn.Module):
    r"""Rotate the image anti-clockwise about the centre.
    
    Args:
        tensor (torch.Tensor): The image tensor to be rotated.
        angle (torch.Tensor): The angle through which to rotate.

    Returns:
        torch.Tensor: The rotated image tensor.
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


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185

def rotate(tensor: torch.Tensor, angle: torch.Tensor,
           center: Union[None, torch.Tensor] = None) -> torch.Tensor:
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
    if (len(tensor.shape) == 3 and not len(angle.shape) == 1) or \
       (len(tensor.shape) == 4 and tensor.shape[0] != angle.shape[0]):
        raise ValueError("Input tensor and angle shapes must match. "
                         "Got tensor: {0} and angle: {1}"
                         .format(tensor.shape, angle.shape))

    # compute the rotation center
    if center is None:
        center = compute_rotation_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    center = center.expand(angle.shape[0], -1)
    rotation_matrix = RotationMatrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix.affine())

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

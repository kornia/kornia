from typing import Union, Tuple, Optional
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeometry.core import get_rotation_matrix2d, warp_affine
from torchgeometry.core import get_perspective_transform, warp_perspective


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


def scaling_matrix(scale: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    '''matrix: torch.Tensor = identity_matrix()
    matrix = matrix.repeat(scale.shape[0], 1, 1)

    sx, sy = torch.chunk(scale, chunks=2, dim=-1)
    cx, cy = torch.chunk(center, chunks=2, dim=-1)

    matrix[..., 0, 2:3] += cx
    matrix[..., 1, 2:3] += cy
    matrix[..., 0:1, 0] *= sx
    matrix[..., 1:2, 1] *= sy'''
    angle: torch.Tensor = torch.zeros_like(scale)
    matrix: torch.Tensor = convert_matrix2d_to_homogeneous(
        get_rotation_matrix2d(center, angle, scale))
    return matrix


def translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    matrix: torch.Tensor = identity_matrix()
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


class ScalingMatrix(nn.Module):
    def __init__(self, scale_factor: torch.Tensor, center: torch.Tensor) -> None:
        super(ScalingMatrix, self).__init__()
        self.scale_factor: torch.Tensor = scale_factor
        self.center: torch.Tensor = center
        self.matrix: torch.Tensor = self._generate_matrix()

    def _generate_matrix(self) -> torch.Tensor:
        assert self.center is not None
        assert self.scale_factor is not None
        return scaling_matrix(self.scale_factor, self.center)

    def affine(self) -> torch.Tensor:
        assert self.matrix is not None
        return self.matrix[..., :2, :3]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self.matrix is not None
        return torch.matmul(self.matrix, input)


class TranslationMatrix(nn.Module):
    def __init__(self, translation: torch.Tensor) -> None:
        super(TranslationMatrix, self).__init__()
        self.translation: torch.Tensor = translation
        self.matrix: torch.Tensor = self._generate_matrix()

    def _generate_matrix(self) -> torch.Tensor:
        assert self.translation is not None
        return translation_matrix(self.translation)

    def affine(self) -> torch.Tensor:
        assert self.matrix is not None
        return self.matrix[..., :2, :3]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self.matrix is not None
        return torch.matmul(self.matrix, input)


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
        return torch.matmul(self.matrix, input)


class RandomRotationMatrix(nn.Module):
    def __init__(self,
            degrees: torch.Tensor,
            center: Union[None, torch.Tensor] = None) -> None:
        super(RandomRotationMatrix, self).__init__()
        self.degrees: torch.Tensor = degrees
        self.center: torch.Tensor = center

        if len(degrees.shape) == 1:
            if bool(degrees < 0):
                 raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = torch.cat([-degrees, degrees])
        else:
            if len(degrees.shape) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
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


class RandomTranslationMatrix(nn.Module):
    def __init__(self, translation: torch.Tensor) -> None:
        super(RandomTranslationMatrix, self).__init__()
        if len(translation.shape) != 1:
            raise ValueError("Translation tensor must be of size of 2.")

        if translation.shape[0] == 1:
            self.translation = torch.cat([-translation, translation])
        elif translation.shape[0] == 2:
            self.translation = translation
        else:
            raise ValueError("If translation is a sequence, it must be of len 2.")

        self.matrix = None
    
    def _generate_params(self, num_samples: int) -> torch.Tensor:
        translation_vec: Tuple[float] = self.translation.tolist()
        translation: torch.Tensor = torch.zeros(num_samples, 2).uniform_(
            translation_vec[0], translation_vec[1])
        return translation

    def _generate_matrix(self, num_samples: int) -> torch.Tensor:
        self.translation: torch.Tensor = self._generate_params(num_samples)
        return translation_matrix(self.translation)

    def affine(self) -> torch.Tensor:
        assert self.matrix is not None
        return self.matrix[..., :2, :3]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 3, input.shape
        num_samples: int = input.shape[0]
        self.matrix: torch.Tensor = self._generate_matrix(num_samples)
        return torch.matmul(input, self.matrix)


class RandomScalingMatrix(nn.Module):
    def __init__(self,
            scaling_factor: torch.Tensor,
            center: Union[None, torch.Tensor] = None) -> None:
        super(RandomScalingMatrix, self).__init__()
        if len(scaling_factor.shape) != 1 and scaling_factor.shape[0] != 2:
            raise ValueError("If scaling_factor is a sequence, it must be of len 2.")

        if any(scaling_factor < 0):
            raise ValueError("scaling_factor must be positive.")

        self.scaling_factor: torch.Tensor = scaling_factor
        self.center: torch.Tensor = center
        self.scale = None
        self.matrix = None
    
    def _generate_params(self, num_samples: int) -> torch.Tensor:
        scaling_factor_vec: Tuple[float] = self.scaling_factor.tolist()
        scale: torch.Tensor = torch.tensor([float(num_samples)]).uniform_(
            scaling_factor_vec[0], scaling_factor_vec[1])
        return scale

    def _generate_matrix(self, num_samples: int) -> torch.Tensor:
        assert self.center is not None
        self.scale: torch.Tensor = self._generate_params(num_samples)
        return scaling_matrix(self.scale, self.center)

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


class Translate(nn.Module):
    r"""Translate the tensor in pixel units.

    Args:
        translation (torch.Tensor): tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          :math:(B, 2), where B is batch size, last dimension contains dx dy.

    Returns:
        torch.Tensor: The translated tensor.
    """
    def __init__(self, translation: torch.Tensor) -> None:
        super(Translate, self).__init__()
        self.translation: torch.Tensor = translation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return translate(input, self.translation)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            'translation={0})'.format(self.translation)


class Scale(nn.Module):
    r"""Scale the tensor by a factor.
    
    Args:
        scale_factor (torch.Tensor): The scale factor apply.
        center (torch.Tensor): The center through which to rotate. The tensor
          must have a shape of :math:(B, 2), where B is batch size and last
          dimension contains cx and cy.

    Returns:
        torch.Tensor: The scaled tensor.
    """
    def __init__(self, scale_factor: torch.Tensor,
            center: Union[None, torch.Tensor] = None) -> None:
        super(Scale, self).__init__()
        self.scale_factor: torch.Tensor = scale_factor
        self.center: torch.Tensor = center

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return scale(input, self.scale_factor, self.center)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            'scale_factor={0}, center={1})'  \
            .format(self.scale_factor, self.center)


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


def translate(tensor: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    r"""Translate the tensor in pixel units.

    Args:
        tensor (torch.Tensor): The image tensor to be translated.
        translation (torch.Tensor): tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          :math:(B, 2), where B is batch size, last dimension contains dx dy.

    Returns:
        torch.Tensor: The translated tensor.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(translation):
        raise TypeError("Input translation type is not a torch.Tensor. Got {}"
                        .format(type(translation)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the translation matrix
    translation_matrix = TranslationMatrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix.affine())


def scale(tensor: torch.Tensor, scale_factor: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    r"""Rotate the image anti-clockwise about the centre.
    
    Args:
        tensor (torch.Tensor): The image tensor to be scaled.
        scale_factor (torch.Tensor): The scale factor apply.
        center (torch.Tensor): The center through which to rotate. The tensor
          must have a shape of :math:(B, 2), where B is batch size and last
          dimension contains cx and cy.

    Returns:
        torch.Tensor: The scaled tensor.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(scale_factor):
        raise TypeError("Input scale_factor type is not a torch.Tensor. Got {}"
                        .format(type(scale_factor)))

    # compute the rotation center
    if center is None:
        center = compute_rotation_center(tensor)

    # compute the scaling matrix
    center = center.expand(scale_factor.shape[0], -1)
    scaling_matrix = ScalingMatrix(scale_factor, center)

    # warp using the affine transform
    return affine(tensor, scaling_matrix.affine())


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


def resized_crop(
        tensor: torch.Tensor,
        tl: torch.Tensor,
        tr: torch.Tensor,
        bl: torch.Tensor,
        br: torch.Tensor,
        size: Tuple[int]) -> torch.Tensor:
    dst_h, dst_w = size

    # [y, x] origin
    # top-left, top-right, bottom-left, bottom-right
    points_src: torch.Tensor = torch.stack([tl, tr, bl, br], dim=1)

    # [y, x] destination
    # top-left, top-right, bottom-left, bottom-right
    points_dst: torch.Tensor = torch.FloatTensor([[
        [0, 0],
        [0, dst_w - 1],
        [dst_h - 1, 0],
        [dst_h - 1, dst_w - 1],
    ]]).repeat(points_src.shape[0], 1, 1)

    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # compute transformation between points and warp
    dst_trans_src: torch.Tensor = get_perspective_transform(
        points_src, points_dst)

    warped: torch.Tensor = warp_perspective(
        tensor, dst_trans_src, (dst_h, dst_w))

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


def center_crop(tensor: torch.Tensor, size: Tuple[int]) -> torch.Tensor:

    dst_h, dst_w = size
    src_h, src_w = tensor.shape[-2:]

    dst_h_half: float = float(dst_h) / 2
    dst_w_half: float = float(dst_w) / 2
    src_h_half: float = float(src_h) / 2
    src_w_half: float = float(src_w) / 2

    start_x: float = src_h_half - dst_h_half
    start_y: float = src_w_half - dst_w_half

    end_x: float = start_x + dst_w - 1
    end_y: float = start_y + dst_h - 1

    # [y, x] origin
    # top-left, top-right, bottom-left, bottom-right
    points_src: torch.Tensor = torch.FloatTensor([[
        [start_y, start_x],
        [start_y, end_x],
        [end_y, start_x],
        [end_y, end_x],
    ]])

    # [y, x] destination
    # top-left, top-right, bottom-left, bottom-right
    points_dst: torch.Tensor = torch.FloatTensor([[
        [0, 0],
        [0, dst_w - 1],
        [dst_h - 1, 0],
        [dst_h - 1, dst_w - 1],
    ]])

    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # compute transformation between points and warp
    dst_trans_src: torch.Tensor = get_perspective_transform(
        points_src, points_dst)
    dst_trans_src = dst_trans_src.repeat(tensor.shape[0], 1, 1)

    warped: torch.Tensor = warp_perspective(
        tensor, dst_trans_src, (dst_h, dst_w))

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped

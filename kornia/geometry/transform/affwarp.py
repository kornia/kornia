from typing import Union, Tuple, Optional

import torch
import torch.nn as nn

from kornia.geometry.transform.imgwarp import (
    warp_affine, get_rotation_matrix2d, get_affine_matrix2d
)
from kornia.geometry.transform.projwarp import (
    warp_affine3d, get_projective_transform
)

__all__ = [
    "affine",
    "scale",
    "rotate",
    "rotate3d",
    "translate",
    "shear",
    "resize",
    "rescale",
    "Scale",
    "Rotate",
    "Translate",
    "Shear",
    "Resize",
    "Rescale",
    "Affine",
]

# utilities to compute affine matrices


def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """Computes the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    assert 2 <= len(tensor.shape) <= 4, f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}."
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor(
        [center_x, center_y],
        device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_tensor_center3d(tensor: torch.Tensor) -> torch.Tensor:
    """Computes the center of tensor plane for (D, H, W), (C, D, H, W) and (B, C, D, H, W)."""
    assert 3 <= len(tensor.shape) <= 5, f"Must be a 3D tensor as DHW, CDHW and BCDHW. Got {tensor.shape}."
    depth, height, width = tensor.shape[-3:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center_z: float = float(depth - 1) / 2
    center: torch.Tensor = torch.tensor(
        [center_x, center_y, center_z],
        device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_rotation_matrix(angle: torch.Tensor,
                             center: torch.Tensor) -> torch.Tensor:
    """Computes a pure affine rotation matrix."""
    scale: torch.Tensor = torch.ones_like(center)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_rotation_matrix3d(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor,
                               center: torch.Tensor) -> torch.Tensor:
    """Computes a pure affine rotation matrix."""
    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 0:
        yaw = yaw.unsqueeze(dim=0)
        pitch = pitch.unsqueeze(dim=0)
        roll = roll.unsqueeze(dim=0)

    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 1:
        yaw = yaw.unsqueeze(dim=1)
        pitch = pitch.unsqueeze(dim=1)
        roll = roll.unsqueeze(dim=1)

    assert len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 2, \
        f"Expected yaw, pitch, roll to be (B, 1). Got {yaw.shape}, {pitch.shape}, {roll.shape}."

    angles: torch.Tensor = torch.cat([yaw, pitch, roll], dim=1)
    scales: torch.Tensor = torch.ones_like(yaw)
    matrix: torch.Tensor = get_projective_transform(center, angles, scales)
    return matrix


def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for translation."""
    matrix: torch.Tensor = torch.eye(
        3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_scaling_matrix(scale: torch.Tensor,
                            center: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for scaling."""
    angle: torch.Tensor = torch.zeros(scale.shape[:1], device=scale.device, dtype=scale.dtype)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_shear_matrix(shear: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for shearing."""
    matrix: torch.Tensor = torch.eye(3, device=shear.device, dtype=shear.dtype)
    matrix = matrix.repeat(shear.shape[0], 1, 1)

    shx, shy = torch.chunk(shear, chunks=2, dim=-1)
    matrix[..., 0, 1:2] += shx
    matrix[..., 1, 0:1] += shy
    return matrix


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166

def affine(tensor: torch.Tensor, matrix: torch.Tensor, mode: str = 'bilinear',
           align_corners: bool = False) -> torch.Tensor:
    r"""Apply an affine transformation to the image.

    Args:
        tensor (torch.Tensor): The image tensor to be warped in shapes of
            :math:`(H, W)`, :math:`(D, H, W)` and :math:`(B, C, H, W)`.
        matrix (torch.Tensor): The 2x3 affine transformation matrix.
        mode (str): 'bilinear' | 'nearest'
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

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
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = warp_affine(tensor, matrix, (height, width), mode,
                                       align_corners=align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


def affine3d(tensor: torch.Tensor, matrix: torch.Tensor, mode: str = 'bilinear',
             align_corners: bool = False) -> torch.Tensor:
    r"""Apply an affine transformation to the 3d volume.

    Args:
        tensor (torch.Tensor): The image tensor to be warped in shapes of
            :math:`(D, H, W)`, :math:`(C, D, H, W)` and :math:`(B, C, D, H, W)`.
        matrix (torch.Tensor): The 3x4 affine transformation matrix.
        mode (str): 'bilinear' | 'nearest'
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Returns:
        torch.Tensor: The warped image.
    """
    # warping needs data in the shape of BCDHW
    is_unbatched: bool = tensor.ndimension() == 4
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    depth: int = tensor.shape[-3]
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = warp_affine3d(tensor, matrix, (depth, height, width), mode,
                                         align_corners=align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185

def rotate(tensor: torch.Tensor, angle: torch.Tensor,
           center: Union[None, torch.Tensor] = None, mode: str = 'bilinear',
           align_corners: bool = False) -> torch.Tensor:
    r"""Rotate the image anti-clockwise about the centre.

    See :class:`~kornia.Rotate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}"
                        .format(type(angle)))
    if center is not None and not torch.is_tensor(center):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    angle = angle.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3], mode, align_corners)


def rotate3d(tensor: torch.Tensor, yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor,
             center: Union[None, torch.Tensor] = None, mode: str = 'bilinear',
             align_corners: bool = False) -> torch.Tensor:
    r"""Rotate the image anti-clockwise about the centre.

    See :class:`~kornia.Rotate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(yaw):
        raise TypeError("yaw is not a torch.Tensor. Got {}".format(type(yaw)))
    if not torch.is_tensor(pitch):
        raise TypeError("pitch is not a torch.Tensor. Got {}".format(type(pitch)))
    if not torch.is_tensor(roll):
        raise TypeError("roll is not a torch.Tensor. Got {}".format(type(roll)))
    if center is not None and not torch.is_tensor(center):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if len(tensor.shape) not in (4, 5,):
        raise ValueError("Invalid tensor shape, we expect CxDxHxW or BxCxDxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center3d(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    yaw = yaw.expand(tensor.shape[0])
    pitch = pitch.expand(tensor.shape[0])
    roll = roll.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix3d(yaw, pitch, roll, center)

    # warp using the affine transform
    return affine3d(tensor, rotation_matrix[..., :3, :4], mode, align_corners)


def translate(tensor: torch.Tensor, translation: torch.Tensor,
              align_corners: bool = False) -> torch.Tensor:
    r"""Translate the tensor in pixel units.

    See :class:`~kornia.Translate` for details.
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
    translation_matrix: torch.Tensor = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3], align_corners=align_corners)


def scale(tensor: torch.Tensor, scale_factor: torch.Tensor,
          center: Union[None, torch.Tensor] = None,
          align_corners: bool = False) -> torch.Tensor:
    r"""Scales the input image.

    See :class:`~kornia.Scale` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(scale_factor):
        raise TypeError("Input scale_factor type is not a torch.Tensor. Got {}"
                        .format(type(scale_factor)))

    if len(scale_factor.shape) == 1:
        # convert isotropic scaling to x-y direction
        scale_factor = scale_factor.repeat(1, 2)

    # compute the tensor center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    center = center.expand(tensor.shape[0], -1)
    scale_factor = scale_factor.expand(tensor.shape[0], 2)
    scaling_matrix: torch.Tensor = _compute_scaling_matrix(scale_factor, center)

    # warp using the affine transform
    return affine(tensor, scaling_matrix[..., :2, :3], align_corners=align_corners)


def shear(tensor: torch.Tensor, shear: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
    r"""Shear the tensor.

    See :class:`~kornia.Shear` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(shear):
        raise TypeError("Input shear type is not a torch.Tensor. Got {}"
                        .format(type(shear)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the translation matrix
    shear_matrix: torch.Tensor = _compute_shear_matrix(shear)

    # warp using the affine transform
    return affine(tensor, shear_matrix[..., :2, :3], align_corners=align_corners)


def _side_to_image_size(
    side_size: int, aspect_ratio: float, side: str = "short"
) -> Tuple[int, int]:
    if side not in ("short", "long", "vert", "horz"):
        raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")
    if side == "vert":
        return side_size, int(side_size * aspect_ratio)
    elif side == "horz":
        return int(side_size / aspect_ratio), side_size
    elif (side == "short") ^ (aspect_ratio < 1.0):
        return side_size, int(side_size * aspect_ratio)
    else:
        return int(side_size / aspect_ratio), side_size


def resize(input: torch.Tensor, size: Union[int, Tuple[int, int]],
           interpolation: str = 'bilinear', align_corners: bool = False,
           side: str = "short") -> torch.Tensor:
    r"""Resize the input torch.Tensor to the given size.

    See :class:`~kornia.Resize` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    input_size = h, w = input.shape[-2:]
    if isinstance(size, int):
        aspect_ratio = w / h
        size = _side_to_image_size(size, aspect_ratio, side)

    if size == input_size:
        return input

    return torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)


class Resize(nn.Module):
    r"""Resize the input torch.Tensor to the given size.

    Args:
        size (int, tuple(int, int)): Desired output size. If size is a sequence like (h, w),
            output size will be matched to this. If size is an int, smaller edge of the image will
            be matched to this number. i.e, if height > width, then image will be rescaled
            to (size * height / width, size)
        interpolation (str):  algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' |
            'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'.
        align_corners(bool): interpolation flag. Default: False. See
            https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
        side (str): Corresponding side if ``size`` is an integer. Can be one of ``"short"``, ``"long"``, ``"vert"``,
            or ``"horz"``. Defaults to ``"short"``.

    Returns:
        torch.Tensor: The resized tensor.
    """

    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: str = 'bilinear',
                 align_corners: bool = False, side: str = "short") -> None:
        super(Resize, self).__init__()
        self.size: Union[int, Tuple[int, int]] = size
        self.interpolation: str = interpolation
        self.align_corners: bool = align_corners
        self.side = side

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return resize(input, self.size, self.interpolation, align_corners=self.align_corners, side=self.side)


class Affine(nn.Module):
    r"""Apply multiple elementary affine transforms simultaneously.

    Args:
        angle (torch.Tensor, optional): Angle in degrees for counter-clockwise rotation around the center. The tensor
            must have a shape of (B), where B is the batch size.
        translation (torch.Tensor, optional): Amount of pixels for translation in x- and y-direction. The tensor must
            have a shape of (B, 2), where B is the batch size and the last dimension contains dx and dy.
        scale_factor (torch.Tensor, optional): Factor for scaling. The tensor must have a shape of (B), where B is the
            batch size.
        shear (torch.Tensor, optional): Angles in degrees for shearing in x- and y-direction around the center. The
            tensor must have a shape of (B, 2), where B is the batch size and the last dimension contains sx and sy.
        center (torch.Tensor, optional): Transformation center in pixels. The tensor must have a shape of (B, 2), where
            B is the batch size and the last dimension contains cx and cy. Defaults to the center of image to be
            transformed.
        align_corners (bool): interpolation flag. Default: False. See :func:`~torch.nn.functional.interpolate` for
            details.

    Raises:
        RuntimeError: If not one of ``angle``, ``translation``, ``scale_factor``, or ``shear`` is set.

    Returns:
        torch.Tensor: The transformed tensor.
    """

    def __init__(
            self,
            angle: Optional[torch.Tensor] = None,
            translation: Optional[torch.Tensor] = None,
            scale_factor: Optional[torch.Tensor] = None,
            shear: Optional[torch.Tensor] = None,
            center: Optional[torch.Tensor] = None,
            align_corners: bool = False,
    ) -> None:
        batch_sizes = [arg.size()[0] for arg in (angle, translation, scale_factor, shear) if arg is not None]
        if not batch_sizes:
            msg = (
                "Affine was created without any affine parameter. At least one of angle, translation, scale_factor, or "
                "shear has to be set."
            )
            raise RuntimeError(msg)

        batch_size = batch_sizes[0]
        if not all(other == batch_size for other in batch_sizes[1:]):
            raise RuntimeError(f"The batch sizes of the affine parameters mismatch: {batch_sizes}")

        self._batch_size = batch_size

        super().__init__()

        if angle is None:
            angle = torch.zeros(batch_size)
        self.angle = angle

        if translation is None:
            translation = torch.zeros(batch_size, 2)
        self.translation = translation

        if scale_factor is None:
            scale_factor = torch.ones(batch_size, 2)
        self.scale_factor = scale_factor

        self.shear = shear
        self.center = center
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.shear is None:
            sx = sy = None
        else:
            sx, sy = self.shear[..., 0], self.shear[..., 1]

        if self.center is None:
            center = _compute_tensor_center(input).expand(input.size()[0], -1)
        else:
            center = self.center

        matrix = get_affine_matrix2d(self.translation, center, self.scale_factor, -self.angle, sx=sx, sy=sy)
        return affine(input, matrix[..., :2, :3], align_corners=self.align_corners)


def rescale(
    input: torch.Tensor,
    factor: Union[float, Tuple[float, float]],
    interpolation: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Rescale the input torch.Tensor with the given factor.

    See :class:`~kornia.Rescale` for details.
    """
    if isinstance(factor, float):
        factor_vert = factor_horz = factor
    else:
        factor_vert, factor_horz = factor

    height, width = input.size()[-2:]
    size = (int(height * factor_vert), int(width * factor_horz))
    return resize(input, size, interpolation=interpolation, align_corners=align_corners)


class Rescale(nn.Module):
    r"""Rescale the input torch.Tensor with the given factor.

    Args:
        factor (float, tuple(float, float)): Desired scaling factor in each direction. If scalar, the value is used
            for both the x- and y-direction.
        interpolation (str):  Algorithm used for upsampling. Can be one of ``"nearest"``, ``"linear"``, ``"bilinear"``,
            ``"bicubic"``, ``"trilinear"``, or ``"area"``. Default: ``"bilinear"``.
        align_corners(bool): Interpolation flag. Default: False. See :func:`~torch.nn.functional.interpolate` for
            details.

    Returns:
        torch.Tensor: The rescaled tensor.
    """

    def __init__(
        self, factor: Union[float, Tuple[float, float]], interpolation: str = "bilinear", align_corners: bool = False
    ) -> None:
        super().__init__()
        self.factor: Union[float, Tuple[float, float]] = factor
        self.interpolation: str = interpolation
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rescale(input, self.factor, self.interpolation, align_corners=self.align_corners)


class Rotate(nn.Module):
    r"""Rotate the tensor anti-clockwise about the centre.

    Args:
        angle (torch.Tensor): The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center (torch.Tensor): The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The rotated tensor.
    """

    def __init__(self, angle: torch.Tensor,
                 center: Union[None, torch.Tensor] = None,
                 align_corners: bool = False) -> None:
        super(Rotate, self).__init__()
        self.angle: torch.Tensor = angle
        self.center: Union[None, torch.Tensor] = center
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rotate(input, self.angle, self.center, align_corners=self.align_corners)


class Translate(nn.Module):
    r"""Translate the tensor in pixel units.

    Args:
        translation (torch.Tensor): tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The translated tensor.
    """

    def __init__(self, translation: torch.Tensor, align_corners: bool = False) -> None:
        super(Translate, self).__init__()
        self.translation: torch.Tensor = translation
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return translate(input, self.translation, self.align_corners)


class Scale(nn.Module):
    r"""Scale the tensor by a factor.

    Args:
        scale_factor (torch.Tensor): The scale factor apply. The tensor
          must have a shape of (B) or (B, 2), where B is batch size.
          If (B), isotropic scaling will perform.
          If (B, 2), x-y-direction specific scaling will perform.
        center (torch.Tensor): The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The scaled tensor.
    """

    def __init__(self, scale_factor: torch.Tensor,
                 center: Union[None, torch.Tensor] = None,
                 align_corners: bool = False) -> None:
        super(Scale, self).__init__()
        self.scale_factor: torch.Tensor = scale_factor
        self.center: Union[None, torch.Tensor] = center
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return scale(input, self.scale_factor, self.center, self.align_corners)


class Shear(nn.Module):
    r"""Shear the tensor.

    Args:
        tensor (torch.Tensor): The image tensor to be skewed.
        shear (torch.Tensor): tensor containing the angle to shear
          in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains shx shy.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The skewed tensor.
    """

    def __init__(self, shear: torch.Tensor,
                 align_corners: bool = False) -> None:
        super(Shear, self).__init__()
        self.shear: torch.Tensor = shear
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return shear(input, self.shear, self.align_corners)

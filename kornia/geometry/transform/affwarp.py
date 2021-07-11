from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import kornia
from kornia.geometry.transform.imgwarp import get_affine_matrix2d, get_rotation_matrix2d, warp_affine
from kornia.geometry.transform.projwarp import get_projective_transform, warp_affine3d
from kornia.utils import _extract_device_dtype
from kornia.utils.image import perform_keep_shape

__all__ = [
    "affine",
    "affine3d",
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
    center: torch.Tensor = torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_tensor_center3d(tensor: torch.Tensor) -> torch.Tensor:
    """Computes the center of tensor plane for (D, H, W), (C, D, H, W) and (B, C, D, H, W)."""
    assert 3 <= len(tensor.shape) <= 5, f"Must be a 3D tensor as DHW, CDHW and BCDHW. Got {tensor.shape}."
    depth, height, width = tensor.shape[-3:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center_z: float = float(depth - 1) / 2
    center: torch.Tensor = torch.tensor([center_x, center_y, center_z], device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Computes a pure affine rotation matrix."""
    scale: torch.Tensor = torch.ones_like(center)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_rotation_matrix3d(
    yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor, center: torch.Tensor
) -> torch.Tensor:
    """Computes a pure affine rotation matrix."""
    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 0:
        yaw = yaw.unsqueeze(dim=0)
        pitch = pitch.unsqueeze(dim=0)
        roll = roll.unsqueeze(dim=0)

    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 1:
        yaw = yaw.unsqueeze(dim=1)
        pitch = pitch.unsqueeze(dim=1)
        roll = roll.unsqueeze(dim=1)

    assert (
        len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 2
    ), f"Expected yaw, pitch, roll to be (B, 1). Got {yaw.shape}, {pitch.shape}, {roll.shape}."

    angles: torch.Tensor = torch.cat([yaw, pitch, roll], dim=1)
    scales: torch.Tensor = torch.ones_like(yaw)
    matrix: torch.Tensor = get_projective_transform(center, angles, scales)
    return matrix


def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for translation."""
    matrix: torch.Tensor = torch.eye(3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_scaling_matrix(scale: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
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


def affine(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    r"""Apply an affine transformation to the image.

    .. image:: _static/img/warp_affine.png

    Args:
        tensor: The image tensor to be warped in shapes of
            :math:`(H, W)`, :math:`(D, H, W)` and :math:`(B, C, H, W)`.
        matrix: The 2x3 affine transformation matrix.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The warped image with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 2, 3, 5)
        >>> aff = torch.eye(2, 3)[None]
        >>> out = affine(img, aff)
        >>> print(out.shape)
        torch.Size([1, 2, 3, 5])
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
    warped: torch.Tensor = warp_affine(tensor, matrix, (height, width), mode, padding_mode, align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


def affine3d(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Apply an affine transformation to the 3d volume.

    Args:
        tensor: The image tensor to be warped in shapes of
            :math:`(D, H, W)`, :math:`(C, D, H, W)` and :math:`(B, C, D, H, W)`.
        matrix: The affine transformation matrix with shape :math:`(B, 3, 4)`.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
         `` 'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The warped image.

    Example:
        >>> img = torch.rand(1, 2, 4, 3, 5)
        >>> aff = torch.eye(3, 4)[None]
        >>> out = affine3d(img, aff)
        >>> print(out.shape)
        torch.Size([1, 2, 4, 3, 5])
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
    warped: torch.Tensor = warp_affine3d(tensor, matrix, (depth, height, width), mode, padding_mode, align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185


def rotate(
    tensor: torch.Tensor,
    angle: torch.Tensor,
    center: Union[None, torch.Tensor] = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    r"""Rotate the tensor anti-clockwise about the center.

    .. image:: _static/img/rotate.png

    Args:
        tensor: The image tensor to be warped in shapes of :math:`(B, C, H, W)`.
        angle: The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center: The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The rotated tensor with shape as input.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       rotate_affine.html>`__.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> angle = torch.tensor([90.])
        >>> out = rotate(img, angle)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))

    if not isinstance(angle, torch.Tensor):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}".format(type(angle)))

    if center is not None and not isinstance(center, torch.Tensor):
        raise TypeError("Input center type is not a torch.Tensor. Got {}".format(type(center)))

    if len(tensor.shape) not in (3, 4):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape))

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    angle = angle.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3], mode, padding_mode, align_corners)


def rotate3d(
    tensor: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    roll: torch.Tensor,
    center: Union[None, torch.Tensor] = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Rotate 3D the tensor anti-clockwise about the centre.

    Args:
        tensor: The image tensor to be warped in shapes of :math:`(B, C, D, H, W)`.
        yaw: The yaw angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        pitch: The pitch angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        roll: The roll angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center: The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        torch.Tensor: The rotated tensor with shape as input.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))

    if not isinstance(yaw, torch.Tensor):
        raise TypeError("yaw is not a torch.Tensor. Got {}".format(type(yaw)))

    if not isinstance(pitch, torch.Tensor):
        raise TypeError("pitch is not a torch.Tensor. Got {}".format(type(pitch)))

    if not isinstance(roll, torch.Tensor):
        raise TypeError("roll is not a torch.Tensor. Got {}".format(type(roll)))

    if center is not None and not isinstance(center, torch.Tensor):
        raise TypeError("Input center type is not a torch.Tensor. Got {}".format(type(center)))

    if len(tensor.shape) not in (4, 5):
        raise ValueError("Invalid tensor shape, we expect CxDxHxW or BxCxDxHxW. " "Got: {}".format(tensor.shape))

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
    return affine3d(tensor, rotation_matrix[..., :3, :4], mode, padding_mode, align_corners)


def translate(
    tensor: torch.Tensor,
    translation: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    r"""Translate the tensor in pixel units.

    .. image:: _static/img/translate.png

    Args:
        tensor: The image tensor to be warped in shapes of :math:`(B, C, H, W)`.
        translation: tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The translated tensor with shape as input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> translation = torch.tensor([[1., 0.]])
        >>> out = translate(img, translation)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))

    if not isinstance(translation, torch.Tensor):
        raise TypeError("Input translation type is not a torch.Tensor. Got {}".format(type(translation)))

    if len(tensor.shape) not in (3, 4):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape))

    # compute the translation matrix
    translation_matrix: torch.Tensor = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3], mode, padding_mode, align_corners)


def scale(
    tensor: torch.Tensor,
    scale_factor: torch.Tensor,
    center: Union[None, torch.Tensor] = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    r"""Scale the tensor by a factor.

    .. image:: _static/img/scale.png

    Args:
        tensor: The image tensor to be warped in shapes of :math:`(B, C, H, W)`.
        scale_factor: The scale factor apply. The tensor
          must have a shape of (B) or (B, 2), where B is batch size.
          If (B), isotropic scaling will perform.
          If (B, 2), x-y-direction specific scaling will perform.
        center: The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The scaled tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> scale_factor = torch.tensor([[2., 2.]])
        >>> out = scale(img, scale_factor)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))

    if not isinstance(scale_factor, torch.Tensor):
        raise TypeError("Input scale_factor type is not a torch.Tensor. Got {}".format(type(scale_factor)))

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
    return affine(tensor, scaling_matrix[..., :2, :3], mode, padding_mode, align_corners)


def shear(
    tensor: torch.Tensor,
    shear: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Shear the tensor.

    .. image:: _static/img/shear.png

    Args:
        tensor: The image tensor to be skewed with shape of :math:`(B, C, H, W)`.
        shear: tensor containing the angle to shear
          in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains shx shy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The skewed tensor with shape same as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> shear_factor = torch.tensor([[0.5, 0.0]])
        >>> out = shear(img, shear_factor)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))

    if not isinstance(shear, torch.Tensor):
        raise TypeError("Input shear type is not a torch.Tensor. Got {}".format(type(shear)))

    if len(tensor.shape) not in (3, 4):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape))

    # compute the translation matrix
    shear_matrix: torch.Tensor = _compute_shear_matrix(shear)

    # warp using the affine transform
    return affine(tensor, shear_matrix[..., :2, :3], mode, padding_mode, align_corners)


def _side_to_image_size(side_size: int, aspect_ratio: float, side: str = "short") -> Tuple[int, int]:
    if side not in ("short", "long", "vert", "horz"):
        raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")
    if side == "vert":
        return side_size, int(side_size * aspect_ratio)
    if side == "horz":
        return int(side_size / aspect_ratio), side_size
    if (side == "short") ^ (aspect_ratio < 1.0):
        return side_size, int(side_size * aspect_ratio)
    return int(side_size / aspect_ratio), side_size


@perform_keep_shape
def resize(
    input: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    interpolation: str = 'bilinear',
    align_corners: Optional[bool] = None,
    side: str = "short",
    antialias: bool = False,
) -> torch.Tensor:
    r"""Resize the input torch.Tensor to the given size.

    .. image:: _static/img/resize.png

    Args:
        tensor: The image tensor to be skewed with shape of :math:`(..., H, W)`.
            `...` means there can be any number of dimensions.
        size: Desired output size. If size is a sequence like (h, w),
            output size will be matched to this. If size is an int, smaller edge of the image will
            be matched to this number. i.e, if height > width, then image will be rescaled
            to (size * height / width, size)
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            'bicubic' | 'trilinear' | 'area'.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The resized tensor with the shape as the specified size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = resize(img, (6, 8))
        >>> print(out.shape)
        torch.Size([1, 3, 6, 8])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(input)))

    if len(input.shape) < 2:
        raise ValueError('Input tensor must have at least two dimensions. Got {}'.format(len(input.shape)))

    input_size = h, w = input.shape[-2:]
    if isinstance(size, int):
        aspect_ratio = w / h
        size = _side_to_image_size(size, aspect_ratio, side)

    if size == input_size:
        return input

    factors = (h / size[0], w / size[1])

    # We do bluring only for downscaling
    antialias = antialias and (max(factors) > 1)

    if antialias:
        # First, we have to determine sigma
        sigmas = (max(factors[0], 1.0), max(factors[1], 1.0))

        # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
        # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
        # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
        ks = int(2.0 * 2 * sigmas[0] + 1), int(2.0 * 2 * sigmas[1] + 1)
        input = kornia.filters.gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def rescale(
    input: torch.Tensor,
    factor: Union[float, Tuple[float, float]],
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    antialias: bool = False,
) -> torch.Tensor:
    r"""Rescale the input torch.Tensor with the given factor.

    .. image:: _static/img/rescale.png

    Args:
        input: The image tensor to be scale with shape of :math:`(B, C, H, W)`.
        factor: Desired scaling factor in each direction. If scalar, the value is used
            for both the x- and y-direction.
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'trilinear'`` | ``'area'``.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The rescaled tensor with the shape as the specified size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = rescale(img, (2, 3))
        >>> print(out.shape)
        torch.Size([1, 3, 8, 12])
    """
    if isinstance(factor, float):
        factor_vert = factor_horz = factor
    else:
        factor_vert, factor_horz = factor

    height, width = input.size()[-2:]
    size = (int(height * factor_vert), int(width * factor_horz))
    return resize(input, size, interpolation=interpolation, align_corners=align_corners, antialias=antialias)


class Resize(nn.Module):
    r"""Resize the input torch.Tensor to the given size.

    Args:
        size: Desired output size. If size is a sequence like (h, w),
            output size will be matched to this. If size is an int, smaller edge of the image will
            be matched to this number. i.e, if height > width, then image will be rescaled
            to (size * height / width, size)
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            'bicubic' | 'trilinear' | 'area'.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The resized tensor with the shape of the given size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = Resize((6, 8))(img)
        >>> print(out.shape)
        torch.Size([1, 3, 6, 8])
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = 'bilinear',
        align_corners: Optional[bool] = None,
        side: str = "short",
        antialias: bool = False,
    ) -> None:
        super(Resize, self).__init__()
        self.size: Union[int, Tuple[int, int]] = size
        self.interpolation: str = interpolation
        self.align_corners: Optional[bool] = align_corners
        self.side: str = side
        self.antialias: bool = antialias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return resize(
            input,
            self.size,
            self.interpolation,
            align_corners=self.align_corners,
            side=self.side,
            antialias=self.antialias,
        )


class Affine(nn.Module):
    r"""Apply multiple elementary affine transforms simultaneously.

    Args:
        angle: Angle in degrees for counter-clockwise rotation around the center. The tensor
            must have a shape of (B), where B is the batch size.
        translation: Amount of pixels for translation in x- and y-direction. The tensor must
            have a shape of (B, 2), where B is the batch size and the last dimension contains dx and dy.
        scale_factor: Factor for scaling. The tensor must have a shape of (B), where B is the
            batch size.
        shear: Angles in degrees for shearing in x- and y-direction around the center. The
            tensor must have a shape of (B, 2), where B is the batch size and the last dimension contains sx and sy.
        center: Transformation center in pixels. The tensor must have a shape of (B, 2), where
            B is the batch size and the last dimension contains cx and cy. Defaults to the center of image to be
            transformed.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Raises:
        RuntimeError: If not one of ``angle``, ``translation``, ``scale_factor``, or ``shear`` is set.

    Returns:
        The transformed tensor with same shape as input.

    Example:
        >>> img = torch.rand(1, 2, 3, 5)
        >>> angle = 90. * torch.rand(1)
        >>> out = Affine(angle)(img)
        >>> print(out.shape)
        torch.Size([1, 2, 3, 5])
    """

    def __init__(
        self,
        angle: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        scale_factor: Optional[torch.Tensor] = None,
        shear: Optional[torch.Tensor] = None,
        center: Optional[torch.Tensor] = None,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: Optional[bool] = None,
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
        device, dtype = _extract_device_dtype([angle, translation, scale_factor])

        if angle is None:
            angle = torch.zeros(batch_size, device=device, dtype=dtype)
        self.angle = angle

        if translation is None:
            translation = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        self.translation = translation

        if scale_factor is None:
            scale_factor = torch.ones(batch_size, 2, device=device, dtype=dtype)
        self.scale_factor = scale_factor

        self.shear = shear
        self.center = center
        self.mode = mode
        self.padding_mode = padding_mode
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
        return affine(input, matrix[..., :2, :3], self.mode, self.padding_mode, self.align_corners)


class Rescale(nn.Module):
    r"""Rescale the input torch.Tensor with the given factor.

    Args:
        factor: Desired scaling factor in each direction. If scalar, the value is used
            for both the x- and y-direction.
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'trilinear'`` | ``'area'``.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The rescaled tensor with the shape according to the given factor.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = Rescale((2, 3))(img)
        >>> print(out.shape)
        torch.Size([1, 3, 8, 12])
    """

    def __init__(
        self,
        factor: Union[float, Tuple[float, float]],
        interpolation: str = "bilinear",
        align_corners: Optional[bool] = None,
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.factor: Union[float, Tuple[float, float]] = factor
        self.interpolation: str = interpolation
        self.align_corners: Optional[bool] = align_corners
        self.antialias: bool = antialias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rescale(
            input, self.factor, self.interpolation, align_corners=self.align_corners, antialias=self.antialias
        )


class Rotate(nn.Module):
    r"""Rotate the tensor anti-clockwise about the centre.

    Args:
        angle: The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center: The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The rotated tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> angle = torch.tensor([90.])
        >>> out = Rotate(angle)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """

    def __init__(
        self,
        angle: torch.Tensor,
        center: Union[None, torch.Tensor] = None,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: Optional[bool] = None,
    ) -> None:
        super(Rotate, self).__init__()
        self.angle: torch.Tensor = angle
        self.center: Union[None, torch.Tensor] = center
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: Optional[bool] = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rotate(input, self.angle, self.center, self.mode, self.padding_mode, self.align_corners)


class Translate(nn.Module):
    r"""Translate the tensor in pixel units.

    Args:
        translation: tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The translated tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> translation = torch.tensor([[1., 0.]])
        >>> out = Translate(translation)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """

    def __init__(
        self,
        translation: torch.Tensor,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: Optional[bool] = None,
    ) -> None:
        super(Translate, self).__init__()
        self.translation: torch.Tensor = translation
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: Optional[bool] = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return translate(input, self.translation, self.mode, self.padding_mode, self.align_corners)


class Scale(nn.Module):
    r"""Scale the tensor by a factor.

    Args:
        scale_factor: The scale factor apply. The tensor
          must have a shape of (B) or (B, 2), where B is batch size.
          If (B), isotropic scaling will perform.
          If (B, 2), x-y-direction specific scaling will perform.
        center: The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The scaled tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> scale_factor = torch.tensor([[2., 2.]])
        >>> out = Scale(scale_factor)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """

    def __init__(
        self,
        scale_factor: torch.Tensor,
        center: Union[None, torch.Tensor] = None,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: Optional[bool] = None,
    ) -> None:
        super(Scale, self).__init__()
        self.scale_factor: torch.Tensor = scale_factor
        self.center: Union[None, torch.Tensor] = center
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: Optional[bool] = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return scale(input, self.scale_factor, self.center, self.mode, self.padding_mode, self.align_corners)


class Shear(nn.Module):
    r"""Shear the tensor.

    Args:
        shear: tensor containing the angle to shear
          in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains shx shy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The skewed tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> shear_factor = torch.tensor([[0.5, 0.0]])
        >>> out = Shear(shear_factor)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """

    def __init__(
        self, shear: torch.Tensor, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False
    ) -> None:
        super(Shear, self).__init__()
        self.shear: torch.Tensor = shear
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return shear(input, self.shear, self.mode, self.padding_mode, self.align_corners)

import warnings
from typing import Optional, Tuple, Union

import torch

from kornia.core import ImageModule as Module
from kornia.core import Tensor, ones, ones_like, zeros
from kornia.filters import gaussian_blur2d
from kornia.utils import _extract_device_dtype
from kornia.utils.image import perform_keep_shape_image
from kornia.utils.misc import eye_like

from .imgwarp import get_affine_matrix2d, get_projective_transform, get_rotation_matrix2d, warp_affine, warp_affine3d

__all__ = [
    "affine",
    "affine3d",
    "scale",
    "rotate",
    "rotate3d",
    "translate",
    "shear",
    "resize",
    "resize_to_be_divisible",
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


def _compute_tensor_center(tensor: Tensor) -> Tensor:
    """Compute the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    if not 2 <= len(tensor.shape) <= 4:
        raise AssertionError(f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}.")
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: Tensor = torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_tensor_center3d(tensor: Tensor) -> Tensor:
    """Compute the center of tensor plane for (D, H, W), (C, D, H, W) and (B, C, D, H, W)."""
    if not 3 <= len(tensor.shape) <= 5:
        raise AssertionError(f"Must be a 3D tensor as DHW, CDHW and BCDHW. Got {tensor.shape}.")
    depth, height, width = tensor.shape[-3:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center_z: float = float(depth - 1) / 2
    center: Tensor = torch.tensor([center_x, center_y, center_z], device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_rotation_matrix(angle: Tensor, center: Tensor) -> Tensor:
    """Compute a pure affine rotation matrix."""
    scale: Tensor = ones_like(center)
    matrix: Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_rotation_matrix3d(yaw: Tensor, pitch: Tensor, roll: Tensor, center: Tensor) -> Tensor:
    """Compute a pure affine rotation matrix."""
    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 0:
        yaw = yaw.unsqueeze(dim=0)
        pitch = pitch.unsqueeze(dim=0)
        roll = roll.unsqueeze(dim=0)

    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 1:
        yaw = yaw.unsqueeze(dim=1)
        pitch = pitch.unsqueeze(dim=1)
        roll = roll.unsqueeze(dim=1)

    if not (len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 2):
        raise AssertionError(f"Expected yaw, pitch, roll to be (B, 1). Got {yaw.shape}, {pitch.shape}, {roll.shape}.")

    angles: Tensor = torch.cat([yaw, pitch, roll], dim=1)
    scales: Tensor = ones_like(yaw)
    matrix: Tensor = get_projective_transform(center, angles, scales)
    return matrix


def _compute_translation_matrix(translation: Tensor) -> Tensor:
    """Compute affine matrix for translation."""
    matrix: Tensor = eye_like(3, translation, shared_memory=False)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_scaling_matrix(scale: Tensor, center: Tensor) -> Tensor:
    """Compute affine matrix for scaling."""
    angle: Tensor = zeros(scale.shape[:1], device=scale.device, dtype=scale.dtype)
    matrix: Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_shear_matrix(shear: Tensor) -> Tensor:
    """Compute affine matrix for shearing."""
    matrix: Tensor = eye_like(3, shear, shared_memory=False)

    shx, shy = torch.chunk(shear, chunks=2, dim=-1)
    matrix[..., 0, 1:2] += shx
    matrix[..., 1, 0:1] += shy
    return matrix


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166


def affine(
    tensor: Tensor,
    matrix: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
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
    warped: Tensor = warp_affine(tensor, matrix, (height, width), mode, padding_mode, align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


def affine3d(
    tensor: Tensor,
    matrix: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:
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
    warped: Tensor = warp_affine3d(tensor, matrix, (depth, height, width), mode, padding_mode, align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185


def rotate(
    tensor: Tensor,
    angle: Tensor,
    center: Union[None, Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
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
       See a working example `here <https://kornia.github.io/tutorials/nbs/rotate_affine.html>`__.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> angle = torch.tensor([90.])
        >>> out = rotate(img, angle)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input tensor type is not a Tensor. Got {type(tensor)}")

    if not isinstance(angle, Tensor):
        raise TypeError(f"Input angle type is not a Tensor. Got {type(angle)}")

    if center is not None and not isinstance(center, Tensor):
        raise TypeError(f"Input center type is not a Tensor. Got {type(center)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError(f"Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {tensor.shape}")

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    angle = angle.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix: Tensor = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3], mode, padding_mode, align_corners)


def rotate3d(
    tensor: Tensor,
    yaw: Tensor,
    pitch: Tensor,
    roll: Tensor,
    center: Union[None, Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:
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
        Tensor: The rotated tensor with shape as input.
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input tensor type is not a Tensor. Got {type(tensor)}")

    if not isinstance(yaw, Tensor):
        raise TypeError(f"yaw is not a Tensor. Got {type(yaw)}")

    if not isinstance(pitch, Tensor):
        raise TypeError(f"pitch is not a Tensor. Got {type(pitch)}")

    if not isinstance(roll, Tensor):
        raise TypeError(f"roll is not a Tensor. Got {type(roll)}")

    if center is not None and not isinstance(center, Tensor):
        raise TypeError(f"Input center type is not a Tensor. Got {type(center)}")

    if len(tensor.shape) not in (4, 5):
        raise ValueError(f"Invalid tensor shape, we expect CxDxHxW or BxCxDxHxW. Got: {tensor.shape}")

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center3d(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    yaw = yaw.expand(tensor.shape[0])
    pitch = pitch.expand(tensor.shape[0])
    roll = roll.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix: Tensor = _compute_rotation_matrix3d(yaw, pitch, roll, center)

    # warp using the affine transform
    return affine3d(tensor, rotation_matrix[..., :3, :4], mode, padding_mode, align_corners)


def translate(
    tensor: Tensor,
    translation: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
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
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input tensor type is not a Tensor. Got {type(tensor)}")

    if not isinstance(translation, Tensor):
        raise TypeError(f"Input translation type is not a Tensor. Got {type(translation)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError(f"Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {tensor.shape}")

    # compute the translation matrix
    translation_matrix: Tensor = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3], mode, padding_mode, align_corners)


def scale(
    tensor: Tensor,
    scale_factor: Tensor,
    center: Union[None, Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
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
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input tensor type is not a Tensor. Got {type(tensor)}")

    if not isinstance(scale_factor, Tensor):
        raise TypeError(f"Input scale_factor type is not a Tensor. Got {type(scale_factor)}")

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
    scaling_matrix: Tensor = _compute_scaling_matrix(scale_factor, center)

    # warp using the affine transform
    return affine(tensor, scaling_matrix[..., :2, :3], mode, padding_mode, align_corners)


def shear(
    tensor: Tensor,
    shear: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:
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
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input tensor type is not a Tensor. Got {type(tensor)}")

    if not isinstance(shear, Tensor):
        raise TypeError(f"Input shear type is not a Tensor. Got {type(shear)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError(f"Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {tensor.shape}")

    # compute the translation matrix
    shear_matrix: Tensor = _compute_shear_matrix(shear)

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


@perform_keep_shape_image
def resize(
    input: Tensor,
    size: Union[int, Tuple[int, int]],
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    side: str = "short",
    antialias: bool = False,
) -> Tensor:
    r"""Resize the input Tensor to the given size.

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
    if not isinstance(input, Tensor):
        raise TypeError(f"Input tensor type is not a Tensor. Got {type(input)}")

    if len(input.shape) < 2:
        raise ValueError(f"Input tensor must have at least two dimensions. Got {len(input.shape)}")

    input_size = h, w = input.shape[-2:]
    if isinstance(size, int):
        if torch.onnx.is_in_onnx_export():
            warnings.warn("Please pass the size with a tuple when exporting to ONNX to correct the tracing.")
        aspect_ratio = w / h
        size = _side_to_image_size(size, aspect_ratio, side)

    # Skip this dangerous if-else when converting to ONNX.
    if not torch.onnx.is_in_onnx_export():
        if size == input_size:
            return input

    factors = (h / size[0], w / size[1])

    # We do bluring only for downscaling
    antialias = antialias and (max(factors) > 1)

    if antialias:
        # First, we have to determine sigma
        # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
        sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))

        # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
        # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
        # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
        ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

        # Make sure it is odd
        if (ks[0] % 2) == 0:
            ks = ks[0] + 1, ks[1]

        if (ks[1] % 2) == 0:
            ks = ks[0], ks[1] + 1

        input = gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def resize_to_be_divisible(
    input: Tensor,
    divisible_factor: int,
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    side: str = "short",
    antialias: bool = False,
) -> Tensor:
    """Resize the input tensor to be divisible by a certain factor.

    Args:
        input (Tensor): Input tensor to be resized.
        divisible_factor (int): The factor to which the image should be divisible.
        interpolation (str, optional): Interpolation flag. Defaults to "bilinear".
        align_corners (Optional[bool], optional):
            whether to align the corners of the input and output. Defaults to None.
        side (str, optional): Side to resize. Defaults to "short".
        antialias (bool, optional):
            If True, then image will be filtered with Gaussian before downscaling. Defaults to False.

    Returns:
        Tensor: The resized tensor.
    """

    if isinstance(input, Tensor) and len(input.shape) == 4:
        height, width = input.shape[2], input.shape[3]
    if isinstance(input, Tensor) and len(input.shape) == 3:
        height, width = input.shape[1], input.shape[2]

    height = round(height / divisible_factor) * divisible_factor
    width = round(width / divisible_factor) * divisible_factor
    return resize(input, (height, width), interpolation, align_corners, side, antialias)


def rescale(
    input: Tensor,
    factor: Union[float, Tuple[float, float]],
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    r"""Rescale the input Tensor with the given factor.

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


class Resize(Module):
    r"""Resize the input Tensor to the given size.

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

    .. raw:: html

        <gradio-app src="kornia/kornia-resize-antialias"></gradio-app>
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = "bilinear",
        align_corners: Optional[bool] = None,
        side: str = "short",
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.size: Union[int, Tuple[int, int]] = size
        self.interpolation: str = interpolation
        self.align_corners: Optional[bool] = align_corners
        self.side: str = side
        self.antialias: bool = antialias

    def forward(self, input: Tensor) -> Tensor:
        return resize(
            input,
            self.size,
            self.interpolation,
            align_corners=self.align_corners,
            side=self.side,
            antialias=self.antialias,
        )


class Affine(Module):
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
        angle: Optional[Tensor] = None,
        translation: Optional[Tensor] = None,
        scale_factor: Optional[Tensor] = None,
        shear: Optional[Tensor] = None,
        center: Optional[Tensor] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
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
            angle = zeros(batch_size, device=device, dtype=dtype)
        self.angle = angle

        if translation is None:
            translation = zeros(batch_size, 2, device=device, dtype=dtype)
        self.translation = translation

        if scale_factor is None:
            scale_factor = ones(batch_size, 2, device=device, dtype=dtype)
        self.scale_factor = scale_factor

        self.shear = shear
        self.center = center
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input: Tensor) -> Tensor:
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


class Rescale(Module):
    r"""Rescale the input Tensor with the given factor.

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
        align_corners: bool = True,
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.factor: Union[float, Tuple[float, float]] = factor
        self.interpolation: str = interpolation
        self.align_corners: Optional[bool] = align_corners
        self.antialias: bool = antialias

    def forward(self, input: Tensor) -> Tensor:
        return rescale(
            input, self.factor, self.interpolation, align_corners=self.align_corners, antialias=self.antialias
        )


class Rotate(Module):
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
        angle: Tensor,
        center: Union[None, Tensor] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.angle: Tensor = angle
        self.center: Union[None, Tensor] = center
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: Tensor) -> Tensor:
        return rotate(input, self.angle, self.center, self.mode, self.padding_mode, self.align_corners)


class Translate(Module):
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
        self, translation: Tensor, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True
    ) -> None:
        super().__init__()
        self.translation: Tensor = translation
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: Tensor) -> Tensor:
        return translate(input, self.translation, self.mode, self.padding_mode, self.align_corners)


class Scale(Module):
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
        scale_factor: Tensor,
        center: Union[None, Tensor] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.scale_factor: Tensor = scale_factor
        self.center: Union[None, Tensor] = center
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: Tensor) -> Tensor:
        return scale(input, self.scale_factor, self.center, self.mode, self.padding_mode, self.align_corners)


class Shear(Module):
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
        self, shear: Tensor, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True
    ) -> None:
        super().__init__()
        self.shear: Tensor = shear
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: Tensor) -> Tensor:
        return shear(input, self.shear, self.mode, self.padding_mode, self.align_corners)

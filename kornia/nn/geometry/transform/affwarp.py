from typing import Union, Tuple, Optional

import torch
import torch.nn as nn

import kornia

__all__ = [
    "Scale",
    "Rotate",
    "Translate",
    "Shear",
    "Resize",
    "Rescale",
    "Affine",
]


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
        return kornia.geometry.resize(
            input, self.size, self.interpolation, align_corners=self.align_corners, side=self.side)


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
            center = kornia.geometry.transform.affwarp._compute_tensor_center(input).expand(input.size()[0], -1)
        else:
            center = self.center

        matrix = kornia.geometry.transform.affwarp.get_affine_matrix2d(
            self.translation, center, self.scale_factor, -self.angle, sx=sx, sy=sy)
        return kornia.geometry.affine(
            input, matrix[..., :2, :3], align_corners=self.align_corners)


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
        return kornia.geometry.rescale(
            input, self.factor, self.interpolation, align_corners=self.align_corners)


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
        return kornia.geometry.rotate(
            input, self.angle, self.center, align_corners=self.align_corners)


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
        return kornia.geometry.translate(input, self.translation, self.align_corners)


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
        return kornia.geometry.scale(
            input, self.scale_factor, self.center, self.align_corners)


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
        return kornia.geometry.shear(input, self.shear, self.align_corners)

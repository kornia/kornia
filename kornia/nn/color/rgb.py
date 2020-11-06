from typing import Union, cast

import torch
import torch.nn as nn

import kornia


class BgrToRgb(nn.Module):
    r"""Convert image from BGR to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = BgrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(BgrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.rgb.bgr_to_rgb(image)


class RgbToBgr(nn.Module):
    r"""Convert an image from RGB to BGR.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: BGR version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> bgr = RgbToBgr()
        >>> output = bgr(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(RgbToBgr, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.rgb.rgb_to_bgr(image)


class RgbToRgba(nn.Module):
    r"""Convert an image from RGB to RGBA.

    Add an alpha channel to existing RGB image.

    Args:
        alpha_val (float, torch.Tensor): A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        torch.Tensor: RGBA version of the image with shape :math:`(*,4,H,W)`.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 4, H, W)`

    .. note:: The current functionality is NOT supported by Torchscript.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgba = RgbToRgba(1.)
        >>> output = rgba(input)  # 2x4x4x5
    """

    def __init__(self, alpha_val: Union[float, torch.Tensor]) -> None:
        super(RgbToRgba, self).__init__()
        self.alpha_val = alpha_val

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.rgb.rgb_to_rgba(image, self.alpha_val)


class BgrToRgba(nn.Module):
    r"""Convert an image from BGR to RGBA.

    Add an alpha channel to existing RGB image.

    Args:
        alpha_val (float, torch.Tensor): A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        torch.Tensor: RGBA version of the image with shape :math:`(*,4,H,W)`.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 4, H, W)`

    .. note:: The current functionality is NOT supported by Torchscript.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgba = BgrToRgba(1.)
        >>> output = rgba(input)  # 2x4x4x5
    """

    def __init__(self, alpha_val: Union[float, torch.Tensor]) -> None:
        super(BgrToRgba, self).__init__()
        self.alpha_val = alpha_val

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.rgb.rgb_to_rgba(image, self.alpha_val)


class RgbaToRgb(nn.Module):
    r"""Convert an image from RGBA to RGB.

    Remove an alpha channel from RGB image.

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 4, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = RgbaToRgb()
        >>> output = rgba(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(RgbaToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.rgb.rgba_to_rgb(image)


class RgbaToBgr(nn.Module):
    r"""Convert an image from RGBA to BGR.

    Remove an alpha channel from BGR image.

    Returns:
        torch.Tensor: BGR version of the image.

    Shape:
        - image: :math:`(*, 4, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = RgbaToBgr()
        >>> output = rgba(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(RgbaToBgr, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.color.rgb.rgba_to_bgr(image)

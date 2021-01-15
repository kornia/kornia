from typing import Union, cast

import torch
import torch.nn as nn


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to BGR.

    Args:
        image (torch.Tensor): RGB Image to be converted to BGRof of shape :math:`(*,3,H,W)`.

    Returns:
        torch.Tensor: BGR version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_bgr(input) # 2x3x4x5
    """
    return bgr_to_rgb(image)


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to RGB.

    Args:
        image (torch.Tensor): BGR Image to be converted to BGR of shape :math:`(*,3,H,W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    # flip image channels
    out: torch.Tensor = image.flip(-3)
    return out


def rgb_to_rgba(image: torch.Tensor, alpha_val: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Convert an image from RGB to RGBA.

    Args:
        image (torch.Tensor): RGB Image to be converted to RGBA of shape :math:`(*,3,H,W)`.
        alpha_val (float, torch.Tensor): A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        torch.Tensor: RGBA version of the image with shape :math:`(*,4,H,W)`.

    .. note:: The current functionality is NOT supported by Torchscript.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_rgba(input, 1.) # 2x4x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    # add one channel
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)

    a: torch.Tensor = cast(torch.Tensor, alpha_val)

    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))

    return torch.cat([r, g, b, a], dim=-3)


def bgr_to_rgba(image: torch.Tensor, alpha_val: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Convert an image from BGR to RGBA.

    Args:
        image (torch.Tensor): BGR Image to be converted to RGBA of shape :math:`(*,3,H,W)`.
        alpha_val (float, torch.Tensor): A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        torch.Tensor: RGBA version of the image with shape :math:`(*,4,H,W)`.

    .. note:: The current functionality is NOT supported by Torchscript.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgba(input, 1.) # 2x4x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    # convert first to RGB, then add alpha channel
    x_rgb: torch.Tensor = bgr_to_rgb(image)
    return rgb_to_rgba(x_rgb, alpha_val)


def rgba_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from RGBA to RGB.

    Args:
        image (torch.Tensor): RGBA Image to be converted to RGB of shape :math:`(*,4,H,W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> output = rgba_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of (*, 4, H, W).Got {image.shape}")

    # unpack channels
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)

    # compute new channels
    a_one = torch.tensor(1.) - a
    r_new: torch.Tensor = a_one * r + a * r
    g_new: torch.Tensor = a_one * g + a * g
    b_new: torch.Tensor = a_one * b + a * b

    return torch.cat([r, g, b], dim=-3)


def rgba_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from RGBA to BGR.

    Args:
        image (torch.Tensor): RGBA Image to be converted to BGR of shape :math:`(*,4,H,W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> output = rgba_to_bgr(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of (*, 4, H, W).Got {image.shape}")

    # convert to RGB first, then to BGR
    x_rgb: torch.Tensor = rgba_to_rgb(image)
    return rgb_to_bgr(x_rgb)


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
        return bgr_to_rgb(image)


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
        return rgb_to_bgr(image)


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
        return rgb_to_rgba(image, self.alpha_val)


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
        return rgb_to_rgba(image, self.alpha_val)


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
        return rgba_to_rgb(image)


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
        return rgba_to_bgr(image)

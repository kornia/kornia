from typing import cast, Union

import torch
import torch.nn as nn


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to BGR.

    .. image:: _static/img/rgb_to_bgr.png

    Args:
        image: RGB Image to be converted to BGRof of shape :math:`(*,3,H,W)`.

    Returns:
        BGR version of the image with shape of shape :math:`(*,3,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       hello_world_tutorial.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_bgr(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}".format(image.shape))

    return bgr_to_rgb(image)


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to RGB.

    Args:
        image: BGR Image to be converted to BGR of shape :math:`(*,3,H,W)`.

    Returns:
        RGB version of the image with shape of shape :math:`(*,3,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}".format(image.shape))

    # flip image channels
    out: torch.Tensor = image.flip(-3)
    return out


def rgb_to_rgba(image: torch.Tensor, alpha_val: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Convert an image from RGB to RGBA.

    Args:
        image: RGB Image to be converted to RGBA of shape :math:`(*,3,H,W)`.
        alpha_val (float, torch.Tensor): A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        RGBA version of the image with shape :math:`(*,4,H,W)`.

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
        image: BGR Image to be converted to RGBA of shape :math:`(*,3,H,W)`.
        alpha_val: A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        RGBA version of the image with shape :math:`(*,4,H,W)`.

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
        image: RGBA Image to be converted to RGB of shape :math:`(*,4,H,W)`.

    Returns:
        RGB version of the image with shape :math:`(*,3,H,W)`.

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
    a_one = torch.tensor(1.0) - a
    r_new: torch.Tensor = a_one * r + a * r
    g_new: torch.Tensor = a_one * g + a * g
    b_new: torch.Tensor = a_one * b + a * b

    return torch.cat([r, g, b], dim=-3)


def rgba_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from RGBA to BGR.

    Args:
        image: RGBA Image to be converted to BGR of shape :math:`(*,4,H,W)`.

    Returns:
        RGB version of the image with shape :math:`(*,3,H,W)`.

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


def rgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an sRGB image to linear RGB. Used in colorspace conversions.

    .. image:: _static/img/rgb_to_linear_rgb.png

    Args:
        image: sRGB Image to be converted to linear RGB of shape :math:`(*,3,H,W)`.

    Returns:
        linear RGB version of the image with shape of :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_linear_rgb(input) # 2x3x4x5
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}".format(image.shape))

    lin_rgb: torch.Tensor = torch.where(image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)

    return lin_rgb


def linear_rgb_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a linear RGB image to sRGB. Used in colorspace conversions.

    Args:
        image: linear RGB Image to be converted to sRGB of shape :math:`(*,3,H,W)`.

    Returns:
        sRGB version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = linear_rgb_to_rgb(input) # 2x3x4x5
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}".format(image.shape))

    threshold = 0.0031308
    rgb: torch.Tensor = torch.where(
        image > threshold, 1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055, 12.92 * image
    )

    return rgb


class BgrToRgb(nn.Module):
    r"""Convert image from BGR to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = BgrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return bgr_to_rgb(image)


class RgbToBgr(nn.Module):
    r"""Convert an image from RGB to BGR.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        BGR version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> bgr = RgbToBgr()
        >>> output = bgr(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_bgr(image)


class RgbToRgba(nn.Module):
    r"""Convert an image from RGB to RGBA.

    Add an alpha channel to existing RGB image.

    Args:
        alpha_val: A float number for the alpha value or a tensor
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
        alpha_val: A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        RGBA version of the image with shape :math:`(*,4,H,W)`.

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
        RGB version of the image.

    Shape:
        - image: :math:`(*, 4, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = RgbaToRgb()
        >>> output = rgba(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgba_to_rgb(image)


class RgbaToBgr(nn.Module):
    r"""Convert an image from RGBA to BGR.

    Remove an alpha channel from BGR image.

    Returns:
        BGR version of the image.

    Shape:
        - image: :math:`(*, 4, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = RgbaToBgr()
        >>> output = rgba(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgba_to_bgr(image)


class RgbToLinearRgb(nn.Module):
    r"""Convert an image from sRGB to linear RGB.

    Reverses the gamma correction of sRGB to get linear RGB values for colorspace conversions.
    The image data is assumed to be in the range of :math:`[0, 1]`

    Returns:
        Linear RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb_lin = RgbToLinearRgb()
        >>> output = rgb_lin(input)  # 2x3x4x5

    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb

        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

        [3] https://en.wikipedia.org/wiki/SRGB
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_linear_rgb(image)


class LinearRgbToRgb(nn.Module):
    r"""Convert a linear RGB image to sRGB.

    Applies gamma correction to linear RGB values, at the end of colorspace conversions, to get sRGB.

    Returns:
        sRGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> srgb = LinearRgbToRgb()
        >>> output = srgb(input)  # 2x3x4x5

    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb

        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

        [3] https://en.wikipedia.org/wiki/SRGB
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return linear_rgb_to_rgb(image)

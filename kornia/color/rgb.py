from typing import cast

import torch
import torch.nn as nn
from typing import Union


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to BGR.

    See :class:`~kornia.color.RgbToBgr` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to BGR.

    Returns:
        torch.Tensor: BGR version of the image.
    """

    return bgr_to_rgb(image)


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to RGB.

    See :class:`~kornia.color.BgrToRgb` for details.

    Args:
        input (torch.Tensor): BGR Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    # flip image channels
    out: torch.Tensor = image.flip(-3)

    return out


def rgb_to_rgba(image: torch.Tensor, alpha_val: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Convert image from RGB to RGBA.

    See :class:`~kornia.color.RgbToRgba` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to RGBA.

    Returns:
        torch.Tensor: RGBA version of the image.
    """

    if not torch.is_tensor(image):
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
    r"""Convert image from BGR to RGBA.

    See :class:`~kornia.color.BgrToRgba` for details.

    Args:
        image (torch.Tensor): BGR Image to be converted to RGBA.

    Returns:
        torch.Tensor: RGBA version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    x_rgb: torch.Tensor = bgr_to_rgb(image)
    return rgb_to_rgba(x_rgb, alpha_val)


class BgrToRgb(nn.Module):
    r"""Convert image from BGR to RGB.

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): BGR image to be converted to RGB.

    returns:
        torch.Tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.BgrToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    def __init__(self) -> None:
        super(BgrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return bgr_to_rgb(image)


class RgbToBgr(nn.Module):
    r"""Convert image from RGB to BGR.

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to BGR.

    returns:
        torch.Tensor: BGR version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        >>> import torch
        >>> import kornia
        >>> input = torch.rand(2, 3, 4, 5)
        >>> bgr = kornia.color.RgbToBgr()
        >>> output = bgr(input)  # 2x3x4x5

    """

    def __init__(self) -> None:
        super(RgbToBgr, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_bgr(image)


class RgbToRgba(nn.Module):
    r"""Convert image from RGB to RGBA.

    Add an Alpha channel to existing RGB image.

    args:
        image (torch.Tensor): RGB image to be converted to RGBA.
        alpha_val (float, torch.Tensor): A float number for the alpha value.

    returns:
        torch.Tensor: RGBA version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 4, H, W)`

    Examples::

        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgba = kornia.color.RgbToRgba(1.)
        >>> output = rgba(input)  # 2x4x4x5
    """

    def __init__(self, alpha_val: Union[float, torch.Tensor]) -> None:
        super(RgbToRgba, self).__init__()
        self.alpha_val = alpha_val

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_rgba(image, self.alpha_val)


class BgrToRgba(nn.Module):
    r"""Convert image from BGR to RGBA.

    Add an Alpha channel to existing BGR image.

    args:
        image (torch.Tensor): BGR image to be converted to RGBA.
        alpha_val (float, torch.Tensor): A float number for the alpha value.

    returns:
        torch.Tensor: RGBA version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 4, H, W)`

    Examples::

        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgba = kornia.color.BgrToRgba(1.)
        >>> output = rgba(input)  # 2x4x4x5
    """

    def __init__(self, alpha_val: Union[float, torch.Tensor]) -> None:
        super(BgrToRgba, self).__init__()
        self.alpha_val = alpha_val

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_rgba(image, self.alpha_val)

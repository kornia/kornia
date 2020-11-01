import torch
import torch.nn as nn


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

<<<<<<< refs/remotes/kornia/master
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]
=======
    r, g, b = torch.chunk(image, chunks=3, dim=-3)
>>>>>>> [Feat] refactor tests for kornia.color (#759)

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

<<<<<<< refs/remotes/kornia/master
    out: torch.Tensor = torch.stack([y, u, v], -3)
=======
    out: torch.Tensor = torch.cat([y, u, v], -3)
>>>>>>> [Feat] refactor tests for kornia.color (#759)

    return out


def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

<<<<<<< refs/remotes/kornia/master
    y: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]
=======
    y, u, v = torch.chunk(image, chunks=3, dim=-3)
>>>>>>> [Feat] refactor tests for kornia.color (#759)

    r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
    g: torch.Tensor = y + -0.396 * u - 0.581 * v
    b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0

<<<<<<< refs/remotes/kornia/master
    out: torch.Tensor = torch.stack([r, g, b], -3)
=======
    out: torch.Tensor = torch.cat([r, g, b], -3)
>>>>>>> [Feat] refactor tests for kornia.color (#759)

    return out


class RgbToYuv(nn.Module):
    r"""Convert an image from RGB to YUV.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: YUV version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> yuv = RgbToYuv()
        >>> output = yuv(input)  # 2x3x4x5

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def __init__(self) -> None:
        super(RgbToYuv, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rgb_to_yuv(input)


class YuvToRgb(nn.Module):
    r"""Convert an image from YUV to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        torch.Tensor: RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(YuvToRgb, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return yuv_to_rgb(input)

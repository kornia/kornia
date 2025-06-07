# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from typing import ClassVar

import torch

from kornia.core import ImageModule as Module
from kornia.core import Tensor


def _rgb_to_y(r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def rgb_to_ycbcr(image: Tensor) -> Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0, :, :]
    g: Tensor = image[..., 1, :, :]
    b: Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: Tensor = _rgb_to_y(r, g, b)
    cb: Tensor = (b - y) * 0.564 + delta
    cr: Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


def rgb_to_y(image: Tensor) -> Tensor:
    r"""Convert an RGB image to Y.

    Args:
        image: RGB Image to be converted to Y with shape :math:`(*, 3, H, W)`.

    Returns:
        Y version of the image with shape :math:`(*, 1, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_y(input)  # 2x1x4x5

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    y: Tensor = _rgb_to_y(r, g, b)
    return y


def rgb_to_ycbcr420(image: Tensor) -> tuple[Tensor, Tensor]:
    r"""Convert an RGB image to YCbCr 420 (subsampled).

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
        A Tensor containing the CbCr planes with shape :math:`(*, 2, H/2, W/2)`
        
    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_ycbcr420(input)  # (2x1x4x6, 2x2x2x3)

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    ycbcrimage = rgb_to_ycbcr(image)

    return (
        ycbcrimage[..., :1, :, :],
        ycbcrimage[..., 1:3, :, :].unfold(-2, 2, 2).unfold(-2, 2, 2).mean((-1, -2)),
    )


def rgb_to_ycbcr422(image: Tensor) -> tuple[Tensor, Tensor]:
    r"""Convert an RGB image to YCbCr 422 (subsampled).

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
        A Tensor containing the CbCr planes with shape :math:`(*, 2, H, W/2)`
        
    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_ycbcr420(input)  # (2x1x4x6, 2x2x4x3)

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    ycbcrimage = rgb_to_ycbcr(image)

    return (ycbcrimage[..., :1, :, :], ycbcrimage[..., 1:3, :, :].unfold(-1, 2, 2).mean(-1))


def ycbcr_to_rgb(image: Tensor) -> Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: Tensor = image[..., 0, :, :]
    cb: Tensor = image[..., 1, :, :]
    cr: Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: Tensor = cb - delta
    cr_shifted: Tensor = cr - delta

    r: Tensor = y + 1.403 * cr_shifted
    g: Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3).clamp(0, 1)


def ycbcr420_to_rgb(imagey: Tensor, imagecbcr: Tensor) -> Tensor:
    r"""Convert an YCbCr420 image to RGB.

    Input need to be padded to be evenly divisible by 2 horizontal and vertical.

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        imagecbcr: CbCr (chroma) Image planes to be converted to RGB with shape :math:`(*, 2, H/2, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputcbcr = torch.rand(2, 2, 2, 3)
        >>> output = ycbcr420_to_rgb(inputy, inputcbcr)  # 2x3x4x6

    """
    if not isinstance(imagey, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imagey)}")

    if not isinstance(imagecbcr, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imagecbcr)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imagecbcr.shape) < 3 or imagecbcr.shape[-3] != 2:
        raise ValueError(f"Input imagecbcr size must have a shape of (*, 2, H/2, W/2). Got {imagecbcr.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if (
        len(imagecbcr.shape) < 2
        or len(imagey.shape) < 2
        or imagey.shape[-2] / imagecbcr.shape[-2] != 2
        or imagey.shape[-1] / imagecbcr.shape[-1] != 2
    ):
        raise ValueError(
            f"Input imagecbcr H&W must be half the size of the luma plane. Got {imagey.shape} and {imagecbcr.shape}"
        )

    # first upsample
    ycbcr444image = torch.cat(
        [imagey, imagecbcr.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)],
        dim=-3,
    )
    # then convert the ycbcr444 tensor

    return ycbcr_to_rgb(ycbcr444image)


def ycbcr422_to_rgb(imagey: Tensor, imagecbcr: Tensor) -> Tensor:
    r"""Convert an YCbCr422 image to RGB.

    Input need to be padded to be evenly divisible by 2 vertical.

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        imagecbcr: CbCr (chroma) Image planes to be converted to RGB with shape :math:`(*, 2, H, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputcbcr = torch.rand(2, 2, 2, 3)
        >>> output = ycbcr420_to_rgb(inputy, inputcbcr)  # 2x3x4x6

    """
    if not isinstance(imagey, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imagey)}")

    if not isinstance(imagecbcr, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imagecbcr)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imagecbcr.shape) < 3 or imagecbcr.shape[-3] != 2:
        raise ValueError(f"Input imagecbcr size must have a shape of (*, 2, H, W/2). Got {imagecbcr.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if len(imagecbcr.shape) < 2 or len(imagey.shape) < 2 or imagey.shape[-1] / imagecbcr.shape[-1] != 2:
        raise ValueError(
            f"Input imagecbcr W must be half the size of the luma plane. Got {imagey.shape} and {imagecbcr.shape}"
        )

    # first upsample
    ycbcr444image = torch.cat([imagey, imagecbcr.repeat_interleave(2, dim=-1)], dim=-3)
    # then convert the ycbcr444 tensor
    return ycbcr_to_rgb(ycbcr444image)


class RgbToYcbcr(Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_ycbcr(image)


class RgbToYcbcr420(Module):
    r"""Convert an image from RGB to YCbCr 420 (subsampled).

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr420 version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)` and :math:`(*, 2, H/2, W/2)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> ycbcr = RgbToYcbcr420()
        >>> output = ycbcr(input)  # (2x1x4x6, 2x2x2x3)

    """

    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        return rgb_to_ycbcr420(image)


class RgbToYcbcr422(Module):
    r"""Convert an image from RGB to YCbCr 422 (subsampled).

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr422 version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)` and :math:`(*, 2, H, W/2)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> ycbcr = RgbToYcbcr422()
        >>> output = ycbcr(input)  # (2x1x4x6, 2x2x4x3)

    """

    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        return rgb_to_ycbcr422(image)


class YcbcrToRgb(Module):
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: Tensor) -> Tensor:
        return ycbcr_to_rgb(image)


class YCbCr420ToRgb(Module):
    r"""Convert an image from YCbCr to RGB.

    Width and Height must be evenly divisible by 2.

    Returns:
        RGB version of the image.

    Shape:
        - imagey: :math:`(*, 1, H, W)`
        - imagecbcr: :math:`(*, 2, H/2, W/2)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputcbcr = torch.rand(2, 2, 2, 3)
        >>> rgb = YCbCr420ToRgb()
        >>> output = rgb(inputy, inputcbcr)  # 2x3x4x6

    """

    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False

    def forward(self, inputy: Tensor, inputcbcr: Tensor) -> Tensor:
        return ycbcr420_to_rgb(inputy, inputcbcr)


class YCbCr422ToRgb(Module):
    r"""Convert an image from YCbCr to RGB.

    Width must be evenly divisible by 2.

    Returns:
        RGB version of the image.

    Shape:
        - imagey: :math:`(*, 1, H, W)`
        - imagecbcr: :math:`(*, 2, H, W/2)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputcbcr = torch.rand(2, 2, 4, 3)
        >>> rgb = YCbCr422ToRgb()
        >>> output = rgb(inputy, inputcbcr)  # 2x3x4x6

    """

    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False

    def forward(self, inputy: Tensor, inputcbcr: Tensor) -> Tensor:  # skipcq: PYL-R0201
        return ycbcr422_to_rgb(inputy, inputcbcr)

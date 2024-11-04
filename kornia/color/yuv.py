from __future__ import annotations

from typing import ClassVar

import torch

from kornia.core import ImageModule as Module
from kornia.core import Tensor


def rgb_to_yuv(image: Tensor) -> Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of :math:`(0, 1)`. The range of the output is of
    :math:`(0, 1)` to luma and the ranges of U and V are :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`,
    respectively.

    The YUV model adopted here follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0, :, :]
    g: Tensor = image[..., 1, :, :]
    b: Tensor = image[..., 2, :, :]

    y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: Tensor = torch.stack([y, u, v], -3)

    return out


def rgb_to_yuv420(image: Tensor) -> tuple[Tensor, Tensor]:
    r"""Convert an RGB image to YUV 420 (subsampled).

    Input need to be padded to be evenly divisible by 2 horizontal and vertical.

    The image data is assumed to be in the range of :math:`(0, 1)`. The range of the output is of :math:`(0, 1)` to
    luma and the ranges of U and V are :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    The YUV model adopted here follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
        A Tensor containing the UV planes with shape :math:`(*, 2, H/2, W/2)`

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # (2x1x4x6, 2x2x2x3)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)

    return (
        yuvimage[..., :1, :, :],
        yuvimage[..., 1:3, :, :].unfold(-2, 2, 2).unfold(-2, 2, 2).mean((-1, -2)),
    )


def rgb_to_yuv422(image: Tensor) -> tuple[Tensor, Tensor]:
    r"""Convert an RGB image to YUV 422 (subsampled).

    Input need to be padded to be evenly divisible by 2 vertical.

    The image data is assumed to be in the range of :math:`(0, 1)`. The range of the output is of
    :math:`(0, 1)` to luma and the ranges of U and V are :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`,
    respectively.

    The YUV model adopted here follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
       A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
       A Tensor containing the UV planes with shape :math:`(*, 2, H, W/2)`

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # (2x1x4x6, 2x1x4x3)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)

    return (yuvimage[..., :1, :, :], yuvimage[..., 1:3, :, :].unfold(-1, 2, 2).mean(-1))


def yuv_to_rgb(image: Tensor) -> Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of :math:`(0, 1)` for luma (Y). The ranges of U and V are
    :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        image: YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if image.dim() < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: Tensor = image[..., 0, :, :]
    u: Tensor = image[..., 1, :, :]
    v: Tensor = image[..., 2, :, :]

    r: Tensor = y + 1.14 * v  # coefficient for g is 0
    g: Tensor = y + -0.396 * u - 0.581 * v
    b: Tensor = y + 2.029 * u  # coefficient for b is 0

    out: Tensor = torch.stack([r, g, b], -3)

    return out


def yuv420_to_rgb(imagey: Tensor, imageuv: Tensor) -> Tensor:
    r"""Convert an YUV420 image to RGB.

    Input need to be padded to be evenly divisible by 2 horizontal and vertical.

    The image data is assumed to be in the range of :math:`(0, 1)` for luma (Y). The ranges of U and V are
    :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        imageuv: UV (chroma) Image planes to be converted to RGB with shape :math:`(*, 2, H/2, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> output = yuv420_to_rgb(inputy, inputuv)  # 2x3x4x6
    """
    if not isinstance(imagey, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imagey)}")

    if not isinstance(imageuv, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imageuv)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imageuv.shape) < 3 or imageuv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of (*, 2, H/2, W/2). Got {imageuv.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if (
        len(imageuv.shape) < 2
        or len(imagey.shape) < 2
        or imagey.shape[-2] / imageuv.shape[-2] != 2
        or imagey.shape[-1] / imageuv.shape[-1] != 2
    ):
        raise ValueError(
            f"Input imageuv H&W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = torch.cat(
        [imagey, imageuv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)],
        dim=-3,
    )
    # then convert the yuv444 tensor

    return yuv_to_rgb(yuv444image)


def yuv422_to_rgb(imagey: Tensor, imageuv: Tensor) -> Tensor:
    r"""Convert an YUV422 image to RGB.

    Input need to be padded to be evenly divisible by 2 vertical.

    The image data is assumed to be in the range of :math:`(0, 1)` for luma (Y). The ranges of U and V are
    :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        imageuv: UV (luma) Image planes to be converted to RGB with shape :math:`(*, 2, H, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> output = yuv420_to_rgb(inputy, inputuv)  # 2x3x4x5
    """
    if not isinstance(imagey, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imagey)}")

    if not isinstance(imageuv, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(imageuv)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imageuv.shape) < 3 or imageuv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of (*, 2, H, W/2). Got {imageuv.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if len(imageuv.shape) < 2 or len(imagey.shape) < 2 or imagey.shape[-1] / imageuv.shape[-1] != 2:
        raise ValueError(
            f"Input imageuv W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = torch.cat([imagey, imageuv.repeat_interleave(2, dim=-1)], dim=-3)
    # then convert the yuv444 tensor
    return yuv_to_rgb(yuv444image)


class RgbToYuv(Module):
    r"""Convert an image from RGB to YUV.

    The image data is assumed to be in the range of :math:`(0, 1)`.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Returns:
        YUV version of the image.

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

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, input: Tensor) -> Tensor:
        return rgb_to_yuv(input)


class RgbToYuv420(Module):
    r"""Convert an image from RGB to YUV420.

    Width and Height evenly divisible by 2.

    The image data is assumed to be in the range of :math:`(0, 1)`.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Returns:
        YUV420 version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)` and :math:`(*, 2, H/2, W/2)`

    Examples:
        >>> yuvinput = torch.rand(2, 3, 4, 6)
        >>> yuv = RgbToYuv420()
        >>> output = yuv(yuvinput)  # # (2x1x4x6, 2x1x2x3)

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False

    def forward(self, yuvinput: Tensor) -> tuple[Tensor, Tensor]:  # skipcq: PYL-R0201
        return rgb_to_yuv420(yuvinput)


class RgbToYuv422(Module):
    r"""Convert an image from RGB to YUV422.

    Width must be evenly disvisible by 2.

    The image data is assumed to be in the range of :math:`(0, 1)`.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Returns:
        YUV422 version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)` and :math:`(*, 2, H, W/2)`

    Examples:
        >>> yuvinput = torch.rand(2, 3, 4, 6)
        >>> yuv = RgbToYuv422()
        >>> output = yuv(yuvinput)  # # (2x1x4x6, 2x2x4x3)

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False

    def forward(self, yuvinput: Tensor) -> tuple[Tensor, Tensor]:  # skipcq: PYL-R0201
        return rgb_to_yuv422(yuvinput)


class YuvToRgb(Module):
    r"""Convert an image from YUV to RGB.

    The image data is assumed to be in the range of :math:`(0, 1)` for luma (Y). The ranges of U and V are
    :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YuvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, input: Tensor) -> Tensor:
        return yuv_to_rgb(input)


class Yuv420ToRgb(Module):
    r"""Convert an image from YUV to RGB.

    Width and Height must be evenly divisible by 2.

    The image data is assumed to be in the range of :math:`(0, 1)` for luma (Y). The ranges of U and V are
    :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Returns:
        RGB version of the image.

    Shape:
        - imagey: :math:`(*, 1, H, W)`
        - imageuv: :math:`(*, 2, H/2, W/2)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> rgb = Yuv420ToRgb()
        >>> output = rgb(inputy, inputuv)  # 2x3x4x6
    """

    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False

    def forward(self, inputy: Tensor, inputuv: Tensor) -> Tensor:  # skipcq: PYL-R0201
        return yuv420_to_rgb(inputy, inputuv)


class Yuv422ToRgb(Module):
    r"""Convert an image from YUV to RGB.

    Width must be evenly divisible by 2.

    The image data is assumed to be in the range of :math:`(0, 1)` for luma (Y). The ranges of U and V are
    :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Returns:
        RGB version of the image.

    Shape:
        - imagey: :math:`(*, 1, H, W)`
        - imageuv: :math:`(*, 2, H, W/2)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 4, 3)
        >>> rgb = Yuv422ToRgb()
        >>> output = rgb(inputy, inputuv)  # 2x3x4x6
    """

    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False

    def forward(self, inputy: Tensor, inputuv: Tensor) -> Tensor:  # skipcq: PYL-R0201
        return yuv422_to_rgb(inputy, inputuv)

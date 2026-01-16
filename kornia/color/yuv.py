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
from torch import nn
from torch.nn import functional as F
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.exceptions import ShapeError


def _apply_linear_transformation(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 linear color transformation with device-aware optimization.

    Args:
        image: Input image tensor with shape :math:`(*, 3, H, W)`.
        kernel: Transformation matrix with shape :math:`(3, 3)` applied along the channel
            dimension.

    Returns:
        Tensor with the same shape as ``image`` containing the transformed values.
    """
    # Handle Integer inputs by casting to float safely
    if image.is_floating_point():
        image_compute = image
    else:
        image_compute = image.float()

    # Match kernel dtype to the image (propagates float64 if needed)
    kernel_compute = kernel.to(dtype=image_compute.dtype, device=image_compute.device)
    input_shape = image_compute.shape

    # Empirical benchmarks show that einsum is faster on CPU for this specific pattern,
    # while conv2d offers significant speedups on GPU/CUDA.
    # We branch to ensure optimal performance on both devices.
    # BRANCH 1: CPU (Einsum)
    if image_compute.device.type == "cpu":
        out = torch.einsum("oc,...chw->...ohw", kernel, image)
        out = out.contiguous()

    # BRANCH 2: GPU/Accelerators (Conv2d)
    else:
        # Reshape for conv2d: (B*..., C, H, W)
        input_flat = image_compute.reshape(-1, 3, input_shape[-2], input_shape[-1])

        weight = kernel_compute.view(3, 3, 1, 1)
        out_flat = F.conv2d(input_flat, weight)

        # Unflatten back to original shape
        out = out_flat.reshape(input_shape)

    return out


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
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
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])
    
    kernel = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.147, -0.289, 0.436],
            [0.615, -0.515, -0.100],
        ],
        device=image.device,
        dtype=image.dtype,
    )

    return _apply_linear_transformation(image, kernel)


def rgb_to_yuv420(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        A torch.Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
        A torch.Tensor containing the UV planes with shape :math:`(*, 2, H/2, W/2)`

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # (2x1x4x6, 2x2x2x3)

    """
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ShapeError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)

    return (
        yuvimage[..., :1, :, :],
        yuvimage[..., 1:3, :, :].unfold(-2, 2, 2).unfold(-2, 2, 2).mean((-1, -2)),
    )


def rgb_to_yuv422(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
       A torch.Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
       A torch.Tensor containing the UV planes with shape :math:`(*, 2, H, W/2)`

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # (2x1x4x6, 2x1x4x3)

    """
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ShapeError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)

    return (yuvimage[..., :1, :, :], yuvimage[..., 1:3, :, :].unfold(-1, 2, 2).mean(-1))


def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
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
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    kernel = torch.tensor(
        [
            [1.0, 0.0, 1.14],
            [1.0, -0.396, -0.581],
            [1.0, 2.029, 0.0],
        ],
        device=image.device,
        dtype=image.dtype,
    )

    return _apply_linear_transformation(image, kernel)


def yuv420_to_rgb(imagey: torch.Tensor, imageuv: torch.Tensor) -> torch.Tensor:
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
    KORNIA_CHECK_SHAPE(imagey, ["*", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(imageuv, ["*", "2", "H", "W"])

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ShapeError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if (
        len(imageuv.shape) < 2
        or len(imagey.shape) < 2
        or imagey.shape[-2] / imageuv.shape[-2] != 2
        or imagey.shape[-1] / imageuv.shape[-1] != 2
    ):
        raise ShapeError(
            f"Input imageuv H&W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = torch.cat(
        [imagey, imageuv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)],
        dim=-3,
    )
    
    return yuv_to_rgb(yuv444image)


def yuv422_to_rgb(imagey: torch.Tensor, imageuv: torch.Tensor) -> torch.Tensor:
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
    KORNIA_CHECK_SHAPE(imagey, ["*", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(imageuv, ["*", "2", "H", "W"])

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ShapeError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if len(imageuv.shape) < 2 or len(imagey.shape) < 2 or imagey.shape[-1] / imageuv.shape[-1] != 2:
        raise ShapeError(
            f"Input imageuv W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = torch.cat([imagey, imageuv.repeat_interleave(2, dim=-1)], dim=-3)
    
    return yuv_to_rgb(yuv444image)


class RgbToYuv(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rgb_to_yuv(input)


class RgbToYuv420(nn.Module):
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

    def forward(self, yuvinput: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # skipcq: PYL-R0201
        return rgb_to_yuv420(yuvinput)


class RgbToYuv422(nn.Module):
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

    def forward(self, yuvinput: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # skipcq: PYL-R0201
        return rgb_to_yuv422(yuvinput)


class YuvToRgb(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return yuv_to_rgb(input)


class Yuv420ToRgb(nn.Module):
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

    def forward(self, inputy: torch.Tensor, inputuv: torch.Tensor) -> torch.Tensor:  # skipcq: PYL-R0201
        return yuv420_to_rgb(inputy, inputuv)


class Yuv422ToRgb(nn.Module):
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

    def forward(self, inputy: torch.Tensor, inputuv: torch.Tensor) -> torch.Tensor:  # skipcq: PYL-R0201
        return yuv422_to_rgb(inputy, inputuv)

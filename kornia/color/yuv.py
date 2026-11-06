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
import torch.nn.functional as F
from torch import nn

from kornia.color.utils import _apply_linear_transformation
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.exceptions import ShapeError


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    ``yuv = rgb_to_yuv(image)`` converts a ``(*, 3, H, W)`` RGB image in :math:`[0, 1]` to analog YUV. Y is in
    :math:`[0, 1]`, U in :math:`[-0.436, 0.436]` and V in :math:`[-0.615, 0.615]`.

    .. image:: _static/img/rgb_to_yuv.png

    Convention:
        Channels are Y, U, V at axis -3, with the M/PAL coefficients (see
        `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
        items 2.5 and 2.6). U and V are signed and centred at zero. skimage ``rgb2yuv`` uses the same matrix with
        more digits, and OpenCV's float ``COLOR_RGB2YUV`` adds 0.5 to U and V.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.

    Returns:
        YUV image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`yuv_to_rgb`: the inverse.
        - :func:`rgb_to_yuv420`: subsamples U and V by 2 in height and width.
        - :func:`rgb_to_yuv422`: subsamples U and V by 2 in width.
        - :class:`RgbToYuv`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb_to_yuv(input).shape
        torch.Size([2, 3, 4, 5])

    """
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    dtype = image.dtype if image.is_floating_point() else torch.float32
    kernel = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.147, -0.289, 0.436],
            [0.615, -0.515, -0.100],
        ],
        device=image.device,
        dtype=dtype,
    )

    return _apply_linear_transformation(image, kernel)


def rgb_to_yuv420(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert an RGB image to YUV 4:2:0 planes.

    ``y, uv = rgb_to_yuv420(image)`` converts a ``(*, 3, H, W)`` RGB image in :math:`[0, 1]` to the full-resolution Y
    plane and the U and V planes at half the height and width.

    Convention:
        Y, U and V are as in :func:`rgb_to_yuv`. Each UV value is the mean of a 2x2 block of U or V, and the two planes
        are stacked as two channels. Height and width must be even; an odd size raises ``ShapeError``.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.

    Returns:
        Tuple of the Y plane with shape ``(*, 1, H, W)`` and the UV planes with shape ``(*, 2, H/2, W/2)``.

    See Also:
        - :func:`yuv420_to_rgb`: the inverse, which restores the chroma resolution by replication.
        - :func:`rgb_to_yuv422`: halves the chroma in width only.
        - :class:`RgbToYuv420`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> y, uv = rgb_to_yuv420(input)
        >>> y.shape, uv.shape
        (torch.Size([2, 1, 4, 6]), torch.Size([2, 2, 2, 3]))

    """
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ShapeError(f"Input H&W must be evenly divisible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)

    # A 2x2 box mean. avg_pool2d computes it directly; a mean over two unfolded window
    # dims gives the same values but is several times slower, most of all on MPS.
    uv = yuvimage[..., 1:3, :, :]
    leading = list(uv.shape[:-3])
    batch = 1
    for size in leading:
        batch *= size
    height, width = uv.shape[-2], uv.shape[-1]
    uv = F.avg_pool2d(uv.reshape([batch, 2, height, width]), 2)

    return yuvimage[..., :1, :, :], uv.reshape(leading + [2, height // 2, width // 2])


def rgb_to_yuv422(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert an RGB image to YUV 4:2:2 planes.

    ``y, uv = rgb_to_yuv422(image)`` converts a ``(*, 3, H, W)`` RGB image in :math:`[0, 1]` to the full-resolution Y
    plane and the U and V planes at half the width.

    Convention:
        Y, U and V are as in :func:`rgb_to_yuv`. Each UV value is the mean of two horizontal neighbours of U or V, and
        the two planes are stacked as two channels. Width must be even, and an odd width raises ``ShapeError``.
        Height may be odd.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.

    Returns:
        Tuple of the Y plane with shape ``(*, 1, H, W)`` and the UV planes with shape ``(*, 2, H, W/2)``.

    See Also:
        - :func:`yuv422_to_rgb`: the inverse, which restores the chroma width by replication.
        - :func:`rgb_to_yuv420`: halves the chroma in height as well.
        - :class:`RgbToYuv422`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> y, uv = rgb_to_yuv422(input)
        >>> y.shape, uv.shape
        (torch.Size([2, 1, 4, 6]), torch.Size([2, 2, 4, 3]))

    """
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    if image.shape[-1] % 2 == 1:
        raise ShapeError(f"Input W must be evenly divisible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)
    uv = yuvimage[..., 1:3, :, :]

    # The width axis by its positive index: the legacy (TorchScript) ONNX exporter lowers ``unfold(-1, ...)`` with
    # the window and the window-count axes swapped, which gives UV planes of shape (*, 2, H, 2).
    return (yuvimage[..., :1, :, :], uv.unfold(uv.dim() - 1, 2, 2).mean(-1))


def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a YUV image to RGB.

    ``rgb = yuv_to_rgb(image)`` converts a ``(*, 3, H, W)`` YUV image, with Y in :math:`[0, 1]`, U in
    :math:`[-0.436, 0.436]` and V in :math:`[-0.615, 0.615]`, to RGB. It inverts :func:`rgb_to_yuv` exactly, so a round
    trip is limited only by the precision of the dtype.

    Convention:
        Channels are Y, U, V at axis -3, as :func:`rgb_to_yuv` returns them. The result is not clipped.

    Args:
        image: YUV image with shape ``(*, 3, H, W)``.

    Returns:
        RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_yuv`: the inverse.
        - :func:`yuv420_to_rgb`: takes Y and half-resolution UV planes.
        - :func:`yuv422_to_rgb`: takes Y and half-width UV planes.
        - :class:`YuvToRgb`: the module form.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> yuv_to_rgb(input).shape
        torch.Size([2, 3, 4, 5])

    """
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    # The exact inverse of the rounded M/PAL kernel in ``rgb_to_yuv``, rather than a separately
    # rounded copy of the published inverse relations, so a round trip is lossless to the dtype.
    # Over the rationals that inverse is
    #   [[1,      -1/25344,  144439/126720],
    #    [1,  -10001/25344,  -73561/126720],
    #    [1,   51499/25344,     -61/126720]]
    # and every literal below is that value correctly rounded to float64. ``torch.linalg.inv``
    # reproduces it only to 3.3e-16 -- its LU factorization does not even return an exact 1.0 in
    # the first column -- so the constants are rounded from the exact fractions, not from it.
    dtype = image.dtype if image.is_floating_point() else torch.float32
    kernel = torch.tensor(
        [
            [1.0, -3.945707070707071e-05, 1.139827967171717],
            [1.0, -0.39461016414141414, -0.5805003156565657],
            [1.0, 2.0319996843434343, -0.00048137626262626264],
        ],
        device=image.device,
        dtype=dtype,
    )

    return _apply_linear_transformation(image, kernel)


def yuv420_to_rgb(imagey: torch.Tensor, imageuv: torch.Tensor) -> torch.Tensor:
    r"""Convert YUV 4:2:0 planes to RGB.

    ``rgb = yuv420_to_rgb(imagey, imageuv)`` converts a full-resolution Y plane and half-resolution UV planes, as
    :func:`rgb_to_yuv420` returns them, to a ``(*, 3, H, W)`` RGB image.

    Convention:
        UV is upsampled by nearest-neighbor replication, then converted as in :func:`yuv_to_rgb`. Height and width
        must be even, and ``imageuv`` must be exactly half the size of ``imagey``; otherwise ``ShapeError`` is
        raised.

    Args:
        imagey: Y plane with shape ``(*, 1, H, W)``.
        imageuv: UV planes with shape ``(*, 2, H/2, W/2)``.

    Returns:
        RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_yuv420`: the inverse, up to the chroma resolution lost.
        - :func:`yuv422_to_rgb`: takes half-width UV planes.
        - :class:`Yuv420ToRgb`: the module form.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> yuv420_to_rgb(inputy, inputuv).shape
        torch.Size([2, 3, 4, 6])

    """
    KORNIA_CHECK_SHAPE(imagey, ["*", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(imageuv, ["*", "2", "H", "W"])

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ShapeError(f"Input H&W must be evenly divisible by 2. Got {imagey.shape}")

    if (
        len(imageuv.shape) < 2
        or len(imagey.shape) < 2
        or imagey.shape[-2] != 2 * imageuv.shape[-2]
        or imagey.shape[-1] != 2 * imageuv.shape[-1]
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
    r"""Convert YUV 4:2:2 planes to RGB.

    ``rgb = yuv422_to_rgb(imagey, imageuv)`` converts a full-resolution Y plane and half-width UV planes, as
    :func:`rgb_to_yuv422` returns them, to a ``(*, 3, H, W)`` RGB image.

    Convention:
        UV is upsampled along width by nearest-neighbor replication, then converted as in :func:`yuv_to_rgb`. Width
        must be even and ``imageuv`` must be half the width of ``imagey`` with the same height; otherwise
        ``ShapeError`` is raised.

    Args:
        imagey: Y plane with shape ``(*, 1, H, W)``.
        imageuv: UV planes with shape ``(*, 2, H, W/2)``.

    Returns:
        RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_yuv422`: the inverse, up to the chroma resolution lost.
        - :func:`yuv420_to_rgb`: takes half-resolution UV planes.
        - :class:`Yuv422ToRgb`: the module form.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 4, 3)
        >>> yuv422_to_rgb(inputy, inputuv).shape
        torch.Size([2, 3, 4, 6])

    """
    KORNIA_CHECK_SHAPE(imagey, ["*", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(imageuv, ["*", "2", "H", "W"])

    if imagey.shape[-1] % 2 == 1:
        raise ShapeError(f"Input W must be evenly divisible by 2. Got {imagey.shape}")

    if (
        len(imageuv.shape) < 2
        or len(imagey.shape) < 2
        or imagey.shape[-2] != imageuv.shape[-2]
        or imagey.shape[-1] != 2 * imageuv.shape[-1]
    ):
        raise ShapeError(
            f"Input imageuv H must match the luma plane and W must be half its size. "
            f"Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = torch.cat([imagey, imageuv.repeat_interleave(2, dim=-1)], dim=-3)

    return yuv_to_rgb(yuv444image)


class RgbToYuv(nn.Module):
    r"""Convert an image from RGB to YUV (module form).

    ``out = RgbToYuv()(image)`` is :func:`rgb_to_yuv`.

    Convention:
        - See the convention block of :func:`rgb_to_yuv`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb_to_yuv`: the one-call form.
        - :class:`YuvToRgb`: the inverse.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> yuv = RgbToYuv()
        >>> yuv(input).shape
        torch.Size([2, 3, 4, 5])

    References:
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to YUV.

        Args:
            input: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            YUV tensor with shape ``(*, 3, H, W)``.
        """
        return rgb_to_yuv(input)


class RgbToYuv420(nn.Module):
    r"""Convert an image from RGB to YUV 4:2:0 planes (module form).

    ``y, uv = RgbToYuv420()(image)`` is :func:`rgb_to_yuv420`.

    Convention:
        - See the convention block of :func:`rgb_to_yuv420`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 1, H, W)`` and ``(*, 2, H/2, W/2)``

    See Also:
        - :func:`rgb_to_yuv420`: the one-call form.
        - :class:`Yuv420ToRgb`: the inverse.
        - :class:`RgbToYuv422`: halves the chroma in width only.

    Examples:
        >>> yuvinput = torch.rand(2, 3, 4, 6)
        >>> yuv = RgbToYuv420()
        >>> y, uv = yuv(yuvinput)
        >>> y.shape, uv.shape
        (torch.Size([2, 1, 4, 6]), torch.Size([2, 2, 2, 3]))

    References:
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, yuvinput: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # skipcq: PYL-R0201
        """Convert an RGB tensor to YUV420 planes.

        Args:
            yuvinput: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            Tuple of ``(y, uv)`` where ``y`` has shape ``(*, 1, H, W)`` and ``uv`` has
            shape ``(*, 2, H / 2, W / 2)``. The ``1`` channel is luma and the ``2``
            channels are chroma.
        """
        return rgb_to_yuv420(yuvinput)


class RgbToYuv422(nn.Module):
    r"""Convert an image from RGB to YUV 4:2:2 planes (module form).

    ``y, uv = RgbToYuv422()(image)`` is :func:`rgb_to_yuv422`.

    Convention:
        - See the convention block of :func:`rgb_to_yuv422`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 1, H, W)`` and ``(*, 2, H, W/2)``

    See Also:
        - :func:`rgb_to_yuv422`: the one-call form.
        - :class:`Yuv422ToRgb`: the inverse.
        - :class:`RgbToYuv420`: halves the chroma in height as well.

    Examples:
        >>> yuvinput = torch.rand(2, 3, 4, 6)
        >>> yuv = RgbToYuv422()
        >>> y, uv = yuv(yuvinput)
        >>> y.shape, uv.shape
        (torch.Size([2, 1, 4, 6]), torch.Size([2, 2, 4, 3]))

    References:
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, yuvinput: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # skipcq: PYL-R0201
        """Convert an RGB tensor to YUV422 planes.

        Args:
            yuvinput: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            Tuple of ``(y, uv)`` where ``y`` has shape ``(*, 1, H, W)`` and ``uv`` has
            shape ``(*, 2, H, W / 2)``. The ``1`` channel is luma and the ``2`` channels
            are chroma.
        """
        return rgb_to_yuv422(yuvinput)


class YuvToRgb(nn.Module):
    r"""Convert an image from YUV to RGB (module form).

    ``out = YuvToRgb()(image)`` is :func:`yuv_to_rgb`.

    Convention:
        - See the convention block of :func:`yuv_to_rgb`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`yuv_to_rgb`: the one-call form.
        - :class:`RgbToYuv`: the inverse.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YuvToRgb()
        >>> rgb(input).shape
        torch.Size([2, 3, 4, 5])

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Convert a YUV tensor to RGB.

        Args:
            input: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to YUV channels, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return yuv_to_rgb(input)


class Yuv420ToRgb(nn.Module):
    r"""Convert YUV 4:2:0 planes to RGB (module form).

    ``rgb = Yuv420ToRgb()(imagey, imageuv)`` is :func:`yuv420_to_rgb`.

    Convention:
        - See the convention block of :func:`yuv420_to_rgb`.

    Shape:
        - imagey: ``(*, 1, H, W)``
        - imageuv: ``(*, 2, H/2, W/2)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`yuv420_to_rgb`: the one-call form.
        - :class:`RgbToYuv420`: the inverse, up to the chroma resolution lost.
        - :class:`Yuv422ToRgb`: takes half-width UV planes.

    Examples:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> rgb = Yuv420ToRgb()
        >>> rgb(inputy, inputuv).shape
        torch.Size([2, 3, 4, 6])

    """

    # The shape of ``inputy``; ``inputuv`` is the subsampled chroma plane.
    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 1, -1, -1]

    def forward(self, inputy: torch.Tensor, inputuv: torch.Tensor) -> torch.Tensor:  # skipcq: PYL-R0201
        """Convert YUV420 luma/chroma planes to RGB.

        Args:
            inputy: Luma tensor with shape ``(*, 1, H, W)``.
            inputuv: Chroma tensor with shape ``(*, 2, H / 2, W / 2)``.
                For both tensors, ``*`` means any number of leading dimensions (for example,
                batch size), and ``H``/``W`` are the full-resolution height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return yuv420_to_rgb(inputy, inputuv)


class Yuv422ToRgb(nn.Module):
    r"""Convert YUV 4:2:2 planes to RGB (module form).

    ``rgb = Yuv422ToRgb()(imagey, imageuv)`` is :func:`yuv422_to_rgb`.

    Convention:
        - See the convention block of :func:`yuv422_to_rgb`.

    Shape:
        - imagey: ``(*, 1, H, W)``
        - imageuv: ``(*, 2, H, W/2)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`yuv422_to_rgb`: the one-call form.
        - :class:`RgbToYuv422`: the inverse, up to the chroma resolution lost.
        - :class:`Yuv420ToRgb`: takes half-resolution UV planes.

    Examples:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 4, 3)
        >>> rgb = Yuv422ToRgb()
        >>> rgb(inputy, inputuv).shape
        torch.Size([2, 3, 4, 6])

    """

    # The shape of ``inputy``; ``inputuv`` is the subsampled chroma plane.
    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 1, -1, -1]

    def forward(self, inputy: torch.Tensor, inputuv: torch.Tensor) -> torch.Tensor:  # skipcq: PYL-R0201
        """Convert YUV422 luma/chroma planes to RGB.

        Args:
            inputy: Luma tensor with shape ``(*, 1, H, W)``.
            inputuv: Chroma tensor with shape ``(*, 2, H, W / 2)``.
                For both tensors, ``*`` means any number of leading dimensions (for example,
                batch size), and ``H``/``W`` are the full-resolution height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return yuv422_to_rgb(inputy, inputuv)

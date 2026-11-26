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

from dataclasses import dataclass
from enum import Enum

import torch

from kornia.core.check import KORNIA_CHECK_SIZES, CheckMessage


@dataclass(frozen=True)
class ImageSize:
    r"""Hold the height and width of an image, without its channels.

    ``size = ImageSize(height, width)`` stores the two sizes of one image in a frozen data class, read back as
    ``size.height`` and ``size.width``. The positional order is ``(height, width)``, kornia's order for sizes and the
    reverse of :math:`(x, y)` points.

    Args:
        height: image height in pixels, an ``int``, a ``float`` or a one-element tensor.
        width: image width in pixels, an ``int``, a ``float`` or a one-element tensor.

    See Also:
        - :class:`ImageLayout`: holds an ``ImageSize`` together with the channel count and order.
        - :class:`Image`: checks the size against the shape of its data.

    Example:
        >>> size = ImageSize(3, 4)
        >>> size.height
        3
        >>> size.width
        4

    """

    height: int | torch.Tensor
    width: int | torch.Tensor


class ColorSpace(Enum):
    r"""Name the color space of an image: unknown, gray, RGB or BGR.

    ``ColorSpace.RGB`` and its siblings label the channels of an :class:`Image` through :class:`PixelFormat`.
    :class:`Image` checks the label against the channel count: ``GRAY`` needs 1 channel, ``RGB`` and ``BGR`` need 3,
    and ``UNKNOWN`` takes any count. No member has alpha, so a 4-channel image is ``UNKNOWN``.

    See Also:
        - :class:`PixelFormat`: pairs the color space with the bit depth.
        - :class:`Image`: converts between ``GRAY``, ``RGB`` and ``BGR`` with ``to_gray``, ``to_rgb`` and ``to_bgr``.
    """

    UNKNOWN = 0  # for now, in case of multi band images
    GRAY = 1
    RGB = 2
    BGR = 3


@dataclass(frozen=True)
class PixelFormat:
    r"""Hold the color space of an image and the storage size of one channel value.

    ``pixel_format = PixelFormat(color_space, bit_depth)`` describes the values of an :class:`Image`: their
    :class:`ColorSpace` and the bits that store one channel value. The data class checks nothing; :class:`Image`
    raises when either field does not fit its data.

    Args:
        color_space: color space.
        bit_depth: the storage size of one channel value in bits, ``8 * element_size()`` of the data: 8 for
            ``uint8``, 32 for ``float32``.

    See Also:
        - :class:`ColorSpace`: the values of ``color_space``.
        - :class:`ImageLayout`: describes the arrangement of the values instead: size, channel count and order.

    Example:
        >>> pixel_format = PixelFormat(ColorSpace.RGB, 8)
        >>> pixel_format.color_space
        <ColorSpace.RGB: 2>
        >>> pixel_format.bit_depth
        8

    """

    color_space: ColorSpace
    bit_depth: int


class ChannelsOrder(Enum):
    r"""Name where an image keeps its channels: first, ``(C, H, W)``, or last, ``(H, W, C)``.

    ``ChannelsOrder.CHANNELS_FIRST`` is the order of kornia tensors, ``(C, H, W)``, and ``CHANNELS_LAST`` the order of
    NumPy and OpenCV arrays, ``(H, W, C)``. :class:`ImageLayout` carries it, and :class:`Image` checks the shape of its
    data against it.

    See Also:
        - :class:`ImageLayout`: holds the order together with the size and the channel count.
    """

    CHANNELS_FIRST = 0
    CHANNELS_LAST = 1


@dataclass(frozen=True)
class ImageLayout:
    """Hold the size, channel count and channel order of an image.

    ``layout = ImageLayout(ImageSize(H, W), C, ChannelsOrder.CHANNELS_FIRST)`` describes the shape of an
    :class:`Image`'s data. The data class checks nothing; :class:`Image` raises ``ShapeError`` when its data does not
    have this shape.

    Args:
        image_size: image size.
        channels: number of channels.
        channels_order: channels order.

    See Also:
        - :class:`ImageSize`: the ``image_size`` field.
        - :class:`ChannelsOrder`: the ``channels_order`` field.
        - :class:`PixelFormat`: describes the values of the pixels instead of their arrangement.

    Example:
        >>> layout = ImageLayout(ImageSize(3, 4), 3, ChannelsOrder.CHANNELS_LAST)
        >>> layout.image_size
        ImageSize(height=3, width=4)
        >>> layout.channels
        3
        >>> layout.channels_order
        <ChannelsOrder.CHANNELS_LAST: 1>

    """

    image_size: ImageSize
    channels: int
    channels_order: ChannelsOrder


def KORNIA_CHECK_IMAGE_LAYOUT(
    x: torch.Tensor,
    layout: ImageLayout,
    msg: CheckMessage | None = None,
    raises: bool = True,
) -> bool:
    """Check tensor shape matches the expected ImageLayout.

    The sizes are compared as numbers, so an int, float or one-element tensor size is checked the same way, and a
    symbolic size under ``torch.compile`` is related to the tensor's size without being turned into a string.

    Args:
        x: tensor to validate.
        layout: expected image layout.
        msg: custom error message, or a function that returns it; see :func:`~kornia.core.check.KORNIA_CHECK`.
        raises: if True, raise ShapeError on mismatch.

    Returns:
        True if shape matches, False otherwise (when raises=False).

    """
    # Not KORNIA_CHECK_SHAPE: it takes strings and skips every entry that is not all digits, a rule meant for names
    # such as "H", so a tensor or float size would never be compared, and str() of a symbolic size under
    # torch.compile cannot be traced.
    if layout.channels_order == ChannelsOrder.CHANNELS_FIRST:
        sizes = [layout.channels, layout.image_size.height, layout.image_size.width]
    elif layout.channels_order == ChannelsOrder.CHANNELS_LAST:
        sizes = [layout.image_size.height, layout.image_size.width, layout.channels]
    else:
        if raises:
            raise NotImplementedError(f"Layout {layout.channels_order} not implemented.")
        return False

    return KORNIA_CHECK_SIZES(x, sizes, msg, raises)


# TODO: define CompressedImage

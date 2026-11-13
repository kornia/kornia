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

from enum import Enum
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


class CFA(Enum):
    r"""Pattern of the Bayer color filter array of a raw image.

    ``CFA.BG`` and its siblings name the 2x2 cell that tiles the mosaic. As in OpenCV, the name reads pixel (1, 1) and
    its right neighbor, so ``BG`` is the sensor usually called RGGB. The members, with the colors of the top-left cell
    read row by row:

    - ``BG``: R G / G B.
    - ``GB``: G R / B G.
    - ``RG``: B G / G R.
    - ``GR``: G B / R G.

    Rotating or flipping a mosaic of even height and width changes its pattern, and so does cropping an odd number of
    rows or columns from its top or left.

    See Also:
        - :func:`raw_to_rgb`: demosaics a mosaic with this pattern.
        - :func:`rgb_to_raw`: samples an RGB image with this pattern.

    References:
        [1] https://en.wikipedia.org/wiki/Color_filter_array

    """

    BG = 0
    GB = 1
    RG = 2
    GR = 3


def raw_to_rgb(image: torch.Tensor, cfa: CFA) -> torch.Tensor:
    r"""Demosaic a Bayer raw image to RGB.

    ``rgb = raw_to_rgb(image, cfa)`` converts a ``(*, 1, H, W)`` mosaic in :math:`[0, 1]` to a full-resolution
    ``(*, 3, H, W)`` RGB image. It interpolates red and blue bilinearly and green from its four nearest green samples.

    Convention:
        Height and width must be even. The mosaic samples are kept, so ``rgb_to_raw(raw_to_rgb(x, cfa), cfa)``
        returns ``x``.
        Away from the outermost pixel ring the result equals OpenCV's ``COLOR_BAYER_*2RGB`` up to its 8-bit rounding
        (checked against OpenCV 5.0). In that ring OpenCV repeats the values of the ring inside it, and this function
        keeps the samples.

    Args:
        image: raw mosaic with shape ``(*, 1, H, W)``.
        cfa: pattern of the mosaic, see :class:`CFA`.

    Returns:
        RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`raw_to_rgb_2x2_downscaled`: one RGB pixel per 2x2 cell, without interpolation.
        - :func:`rgb_to_raw`: samples an RGB image back into a mosaic.
        - :class:`RawToRgb`: the module form.

    Example:
        >>> rawinput = torch.randn(2, 1, 4, 6)
        >>> rgb = raw_to_rgb(rawinput, CFA.RG) # 2x3x4x6

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if image.dim() < 3 or image.size(-3) != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly divisible by 2. Got {image.shape}")

    imagesize = image.size()

    image = image.reshape(-1, 1, image.shape[-2], image.shape[-1])

    # BG is defined as pel 1,1 being blue, that is the top left is red (RGGB). This matches
    # opencv naming so makes sense to keep
    if cfa == CFA.BG:
        r = image[..., :, ::2, ::2]
        b = image[..., :, 1::2, 1::2]
        rpad = (0, 1, 0, 1)
        bpad = (1, 0, 1, 0)
    elif cfa == CFA.GB:
        r = image[..., :, ::2, 1::2]
        b = image[..., :, 1::2, ::2]
        rpad = (1, 0, 0, 1)
        bpad = (0, 1, 1, 0)
    elif cfa == CFA.RG:
        r = image[..., :, 1::2, 1::2]
        b = image[..., :, ::2, ::2]
        rpad = (1, 0, 1, 0)
        bpad = (0, 1, 0, 1)
    elif cfa == CFA.GR:
        r = image[..., :, 1::2, ::2]
        b = image[..., :, ::2, 1::2]
        rpad = (0, 1, 1, 0)
        bpad = (1, 0, 0, 1)
    else:
        raise ValueError(f"Unsupported CFA Got {cfa}.")

    # upscaling r and b with bi-linear gives reasonable quality
    # Note that depending on where these are sampled we need to F.pad appropriately
    # the bilinear filter will pretty much be based on for example this layout (RG)
    # (which needs to be padded bottom right)
    # +-+-+
    # |B| |
    # | | |
    # +-+-+
    # While in this layout we need to F.pad with additional B samples top left to
    # make sure we interpolate from the correct position
    # +-+-+
    # | | |
    # | |B|
    # +-+-+
    # For an image like this (3x2 blue pixels)
    # +------+
    # |B B B |
    # |      |
    # |B B B |
    # |      |
    # +------+
    # It needs to be expanded to this (4x3 pixels scaled to 7x5 for correct interpolation)
    # +-------+
    # |B B B b|
    # |       |
    # |B B B b|
    # |       |
    # |b b b b|
    # +-------+
    # and we crop the area afterwards. This is since the interpolation will be between first and last pixel
    # evenly spaced between them while the B/R samples will be missing in the corners were they are assumed to exist
    # Further we need to do align_corners to start the interpolation from the middle of the samples in the corners, that
    # way we get to keep the known blue samples across the whole image
    rpadded = F.pad(r, list(rpad), "replicate")
    bpadded = F.pad(b, list(bpad), "replicate")
    # use explicit padding instead of conv2d padding to be able to use reflect which mirror the correct colors
    # for a 2x2 bayer filter
    gpadded = F.pad(image, [1, 1, 1, 1], "reflect")

    r_up = F.interpolate(rpadded, size=(image.shape[-2] + 1, image.shape[-1] + 1), mode="bilinear", align_corners=True)
    b_up = F.interpolate(bpadded, size=(image.shape[-2] + 1, image.shape[-1] + 1), mode="bilinear", align_corners=True)

    # remove the extra padding
    r_up = F.pad(r_up, [-x for x in rpad])
    b_up = F.pad(b_up, [-x for x in bpad])

    # all unknown pixels are the average of the nearby green samples
    kernel = torch.tensor(
        [[[[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]]], dtype=image.dtype, device=image.device
    )

    # This is done on all samples but result for the known green samples is then overwritten by the input
    g_up = F.conv2d(gpadded, kernel)
    # Keep the output's promoted dtype when copying measured samples into an autocast result.
    if g_up.dtype != image.dtype:
        g_up = g_up.to(torch.promote_types(image.dtype, g_up.dtype))

    # overwrite the already known samples which otherwise have values from r/b
    # this depends on the CFA configuration
    if cfa == CFA.BG:
        g_up[:, :, ::2, 1::2] = image[:, :, ::2, 1::2]
        g_up[:, :, 1::2, ::2] = image[:, :, 1::2, ::2]
    elif cfa == CFA.GB:
        g_up[:, :, ::2, ::2] = image[:, :, ::2, ::2]
        g_up[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    elif cfa == CFA.RG:
        g_up[:, :, 1::2, ::2] = image[:, :, 1::2, ::2]
        g_up[:, :, ::2, 1::2] = image[:, :, ::2, 1::2]
    elif cfa == CFA.GR:
        g_up[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
        g_up[:, :, ::2, ::2] = image[:, :, ::2, ::2]
    else:
        raise ValueError(f"Unsupported CFA Got {cfa}.")

    r_up = r_up.view(imagesize)
    g_up = g_up.view(imagesize)
    b_up = b_up.view(imagesize)

    rgb: torch.Tensor = torch.cat([r_up, g_up, b_up], dim=-3)

    return rgb


def rgb_to_raw(image: torch.Tensor, cfa: CFA) -> torch.Tensor:
    r"""Sample an RGB image into a Bayer raw image.

    ``raw = rgb_to_raw(image, cfa)`` keeps one channel at each pixel of a ``(*, 3, H, W)`` RGB image, the one that
    the pattern ``cfa`` puts there, and returns the ``(*, 1, H, W)`` mosaic.

    Convention:
        The inverse conversion requires even height and width.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.
        cfa: pattern of the mosaic, see :class:`CFA`.

    Returns:
        raw mosaic with shape ``(*, 1, H, W)``.

    See Also:
        - :func:`raw_to_rgb`: the inverse, which interpolates the missing samples.
        - :class:`RgbToRaw`: the module form.

    Example:
        >>> rgbinput = torch.rand(2, 3, 4, 6)
        >>> raw = rgb_to_raw(rgbinput, CFA.BG) # 2x1x4x6

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # pick the torch.Tensor with green pixels
    # clone to make sure grad works
    output: torch.Tensor = image[..., 1:2, :, :].clone()

    # overwrite the r/b positions (depending on the cfa configuration) with blue/red pixels
    if cfa == CFA.BG:
        output[..., :, ::2, ::2] = image[..., 0:1, ::2, ::2]  # red
        output[..., :, 1::2, 1::2] = image[..., 2:3, 1::2, 1::2]  # blue
    elif cfa == CFA.GB:
        output[..., :, ::2, 1::2] = image[..., 0:1, ::2, 1::2]  # red
        output[..., :, 1::2, ::2] = image[..., 2:3, 1::2, ::2]  # blue
    elif cfa == CFA.RG:
        output[..., :, 1::2, 1::2] = image[..., 0:1, 1::2, 1::2]  # red
        output[..., :, ::2, ::2] = image[..., 2:3, ::2, ::2]  # blue
    elif cfa == CFA.GR:
        output[..., :, 1::2, ::2] = image[..., 0:1, 1::2, ::2]  # red
        output[..., :, ::2, 1::2] = image[..., 2:3, ::2, 1::2]  # blue
    else:
        raise ValueError(f"Unsupported CFA value. Got {cfa}")

    return output


def raw_to_rgb_2x2_downscaled(image: torch.Tensor, cfa: CFA) -> torch.Tensor:
    r"""Convert a Bayer raw image to RGB at half the height and width.

    ``rgb = raw_to_rgb_2x2_downscaled(image, cfa)`` turns each 2x2 cell of a ``(*, 1, H, W)`` mosaic in
    :math:`[0, 1]` into one RGB pixel: red and blue are the cell's samples and green is the mean of its two green
    samples.

    Convention:
        Height and width must be even.

    Args:
        image: raw mosaic with shape ``(*, 1, H, W)``.
        cfa: pattern of the mosaic, see :class:`CFA`.

    Returns:
        RGB image with shape ``(*, 3, H / 2, W / 2)``.

    See Also:
        - :func:`raw_to_rgb`: keeps the resolution and interpolates.
        - :class:`RawToRgb2x2Downscaled`: the module form.

    Example:
        >>> rawinput = torch.randn(2, 1, 4, 6)
        >>> rgb = raw_to_rgb_2x2_downscaled(rawinput, CFA.RG) # 2x3x2x3

    """
    KORNIA_CHECK(isinstance(image, torch.Tensor), "Input type is not a torch.Tensor")

    KORNIA_CHECK_SHAPE(image, ["*", "1", "H", "W"])

    KORNIA_CHECK(
        image.shape[-2] % 2 == 0 and image.shape[-1] % 2 == 0,
        f"Input H&W must be evenly divisible by 2. Got {image.shape}",
    )

    if cfa == CFA.BG:
        r = image[..., :, ::2, ::2]
        b = image[..., :, 1::2, 1::2]
        g1 = image[..., :, ::2, 1::2]
        g2 = image[..., :, 1::2, ::2]
    elif cfa == CFA.GB:
        r = image[..., :, ::2, 1::2]
        b = image[..., :, 1::2, ::2]
        g1 = image[..., :, ::2, ::2]
        g2 = image[..., :, 1::2, 1::2]
    elif cfa == CFA.RG:
        r = image[..., :, 1::2, 1::2]
        b = image[..., :, ::2, ::2]
        g1 = image[..., :, 1::2, ::2]
        g2 = image[..., :, ::2, 1::2]
    elif cfa == CFA.GR:
        r = image[..., :, 1::2, ::2]
        b = image[..., :, ::2, 1::2]
        g1 = image[..., :, 1::2, 1::2]
        g2 = image[..., :, ::2, ::2]
    else:
        raise ValueError(f"Unsupported CFA Got {cfa}.")

    rgb: torch.Tensor = torch.cat([r, (g1 + g2) / 2, b], dim=-3)

    return rgb


class RawToRgb(nn.Module):
    r"""Demosaic a Bayer raw image to RGB (module form).

    ``conv = RawToRgb(cfa)`` stores the pattern, and ``out = conv(image)`` is :func:`raw_to_rgb` with it.

    Convention:
        - See the convention block of :func:`raw_to_rgb`.

    Shape:
        - image: ``(*, 1, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`raw_to_rgb`: the one-call form.
        - :class:`RawToRgb2x2Downscaled`: half the resolution, without interpolation.
        - :class:`RgbToRaw`: the inverse.

    Example:
        >>> rawinput = torch.rand(2, 1, 4, 6)
        >>> rgb = RawToRgb(CFA.RG)
        >>> output = rgb(rawinput)  # 2x3x4x6

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 1, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def __init__(self, cfa: CFA) -> None:
        super().__init__()
        self.cfa = cfa

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a Bayer raw tensor to RGB.

        Args:
            image: Input tensor with shape ``(*, 1, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``1`` is a single raw Bayer channel, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)`` using this module's CFA pattern.
        """
        return raw_to_rgb(image, cfa=self.cfa)


class RgbToRaw(nn.Module):
    r"""Sample an RGB image into a Bayer raw image (module form).

    ``conv = RgbToRaw(cfa)`` stores the pattern, and ``out = conv(image)`` is :func:`rgb_to_raw` with it.

    Convention:
        - See the convention block of :func:`rgb_to_raw`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 1, H, W)``

    See Also:
        - :func:`rgb_to_raw`: the one-call form.
        - :class:`RawToRgb`: the inverse.

    Example:
        >>> rgbinput = torch.rand(2, 3, 4, 6)
        >>> raw = RgbToRaw(CFA.GB)
        >>> output = raw(rgbinput)  # 2x1x4x6

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 1, -1, -1]

    def __init__(self, cfa: CFA) -> None:
        super().__init__()
        self.cfa = cfa

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to Bayer raw.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            Raw tensor with shape ``(*, 1, H, W)`` using this module's CFA pattern.
        """
        return rgb_to_raw(image, cfa=self.cfa)


class RawToRgb2x2Downscaled(nn.Module):
    r"""Convert a Bayer raw image to RGB at half the height and width (module form).

    ``conv = RawToRgb2x2Downscaled(cfa)`` stores the pattern, and ``out = conv(image)`` is
    :func:`raw_to_rgb_2x2_downscaled` with it.

    Convention:
        - See the convention block of :func:`raw_to_rgb_2x2_downscaled`.

    Shape:
        - image: ``(*, 1, H, W)``
        - output: ``(*, 3, H / 2, W / 2)``

    See Also:
        - :func:`raw_to_rgb_2x2_downscaled`: the one-call form.
        - :class:`RawToRgb`: keeps the resolution and interpolates.

    Example:
        >>> rawinput = torch.rand(2, 1, 4, 6)
        >>> rgb_downscale = RawToRgb2x2Downscaled(CFA.RG)
        >>> output = rgb_downscale(rawinput)  # 2x3x2x3

    """

    def __init__(self, cfa: CFA) -> None:
        super().__init__()
        self.cfa = cfa

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a Bayer raw tensor to downscaled RGB.

        Args:
            image: Input tensor with shape ``(*, 1, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``1`` is a single raw Bayer channel, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H / 2, W / 2)``.
        """
        return raw_to_rgb_2x2_downscaled(image, cfa=self.cfa)

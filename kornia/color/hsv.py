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

import math
from typing import ClassVar

import torch
from torch import nn


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    max_rgb = image.amax(-3)
    min_rgb = image.amin(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    # select the sextant of the first maximal channel, matching torch.max(dim) tie-breaking;
    # branchless selection avoids max/argmax-with-indices and gather, which are ~100x slower
    # than amax/pointwise ops on MPS and block fusion under torch.compile
    r, g, b = torch.unbind(image, dim=-3)
    h = torch.where((r >= g) & (r >= b), h1, torch.where(g >= b, h2, h3))
    h = h / deltac
    h = (h / 6.0) % 1.0
    h = 2.0 * math.pi * h  # we return 0/2pi output

    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    p: torch.Tensor = v * (1.0 - s)
    q: torch.Tensor = v * (1.0 - f * s)
    t: torch.Tensor = v * (1.0 - (1.0 - f) * s)

    hi = hi.long().clamp_(0, 5)

    # branchless per-channel sextant selection, replacing an 18-plane stack + gather: gather blocks
    # pointwise fusion under torch.compile (the stack+gather graph fails to compile on the MPS
    # inductor backend) and materializing an 18-plane buffer costs extra memory traffic in eager.
    # Each where-chain reproduces one row of the original [v,q,p,p,t,v / t,v,v,q,p,p / p,p,t,v,v,q]
    # table indexed by hi, selecting (not recomputing) the same p/q/t/v tensors; the sextant masks
    # are computed once and reused across R/G/B instead of being recomputed per channel.
    m0 = hi == 0
    m1 = hi == 1
    m2 = hi == 2
    m3 = hi == 3
    m4 = hi == 4

    r = torch.where(m0, v, torch.where(m1, q, torch.where(m2, p, torch.where(m3, p, torch.where(m4, t, v)))))
    g = torch.where(m0, t, torch.where(m1, v, torch.where(m2, v, torch.where(m3, q, torch.where(m4, p, p)))))
    b = torch.where(m0, p, torch.where(m1, p, torch.where(m2, t, torch.where(m3, v, torch.where(m4, v, q)))))

    return torch.stack((r, g, b), dim=-3)


class RgbToHsv(nn.Module):
    r"""Convert an image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hsv = RgbToHsv()
        >>> output = hsv(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to HSV.

        Args:
            image: Input tensor with shape :math:`(*, 3, H, W)`.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` is the channel dimension, and ``H``/``W`` are height and width.

        Returns:
            HSV tensor with shape :math:`(*, 3, H, W)`.
        """
        return rgb_to_hsv(image, self.eps)


class HsvToRgb(nn.Module):
    r"""Convert an image from HSV to RGB.

    H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HsvToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an HSV tensor to RGB.

        Args:
            image: Input tensor with shape :math:`(*, 3, H, W)`.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` is the channel dimension, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape :math:`(*, 3, H, W)`.
        """
        return hsv_to_rgb(image)

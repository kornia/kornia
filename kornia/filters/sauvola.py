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

import torch
from torch import nn
from torch.nn import functional as F

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

from .kernels import _check_kernel_size, _unpack_2d_ks

__all__ = ["SauvolaThreshold", "sauvola_threshold"]


def sauvola_threshold(
    input: torch.Tensor,
    window_size: tuple[int, int] | int = 15,
    k: float = 0.2,
    r: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Binarize images using Sauvola's local mean and standard deviation.

    ``mask, threshold = sauvola_threshold(input, window_size, k, r)`` computes
    ``threshold = mean * (1 + k * (std / r - 1))`` in each reflected local window,
    independently for every image and channel. The boolean mask is ``input > threshold``;
    invert it with ``~mask`` to select dark text on a bright page. Unlike a global
    threshold, the local threshold follows uneven illumination across a document.

    Convention:
        - Windows are positive odd sizes, in ``(height, width)`` order. Reflection excludes
          the edge pixel, as in :func:`box_blur`; each image axis must exceed its window radius.
        - Float input is normally in ``[0, 1]`` and defaults to ``r=0.5``; uint8 stays on
          its ``[0, 255]`` scale and defaults to ``r=127.5``. Set ``r`` explicitly for other
          intensity scales. This differs from scikit-image's default ``r=1`` for floating
          images; pass the same ``r`` to both libraries when comparing.
        - The threshold has the input shape and device, and dtype float64 for float64 input,
          float32 otherwise. Moments use float64 accumulation on CPU and for float64 input, float32
          otherwise, with autocast disabled. Thresholds are differentiable for floating inputs; the hard mask is not.
          At zero local variance, the standard deviation's derivative is defined as zero.

    Args:
        input: Finite floating-point or uint8 images with shape ``(B, C, H, W)``.
        window_size: Odd square size, or a pair of odd sizes for a rectangular neighborhood.
        k: Finite nonnegative sensitivity to local contrast. Usually between 0.2 and 0.5.
        r: Positive finite standard-deviation scale. ``None`` selects the scale described above.

    Returns:
        A boolean mask and a local threshold map, both with shape ``(B, C, H, W)``.

    See Also:
        :func:`otsu_threshold`: use a single histogram-based threshold per image channel.
        :func:`box_blur`: compute local means without thresholding.

    References:
        J. Sauvola and M. Pietikainen, Adaptive document image binarization,
        Pattern Recognition 33(2), 225-236 (2000). DOI: 10.1016/S0031-3203(99)00055-2.

    Example:
        >>> image = torch.ones(1, 1, 5, 5)
        >>> mask, threshold = sauvola_threshold(image, window_size=3)
        >>> mask.shape, threshold.shape
        (torch.Size([1, 1, 5, 5]), torch.Size([1, 1, 5, 5]))

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(input.is_floating_point() or input.dtype == torch.uint8, "Expected floating-point or uint8 images.")
    _check_kernel_size(window_size)
    height, width = _unpack_2d_ks(window_size)
    KORNIA_CHECK(input.shape[-2] > height // 2, "Image height must exceed the window radius.")
    KORNIA_CHECK(input.shape[-1] > width // 2, "Image width must exceed the window radius.")
    KORNIA_CHECK(math.isfinite(k) and k >= 0, "k must be finite and nonnegative.")
    if r is None:
        r = 127.5 if input.dtype == torch.uint8 else 0.5
    KORNIA_CHECK(math.isfinite(r) and r > 0, "r must be finite and positive.")

    values = input.to(torch.float64 if input.dtype == torch.float64 else torch.float32)
    if input.shape[0] == 0 or input.shape[1] == 0:
        return input > values, values

    with torch.autocast(device_type=input.device.type, enabled=False):
        # Float64 cumulative sums on CPU avoid losing low local variance when subtracting
        # image-sized sums. Other accelerators keep float32 unless the input is float64.
        accumulation_dtype = torch.float64 if input.device.type == "cpu" else values.dtype
        data = values.to(accumulation_dtype)
        offset = data[..., :1, :1]
        centered = data - offset
        moments = torch.cat((centered, centered.square()), dim=1)
        padded = F.pad(moments, (width // 2, width // 2, height // 2, height // 2), mode="reflect")
        # Leading zeros turn each window sum into four summed-area table reads. The
        # work per output pixel is independent of window area; no patch tensor is built.
        integral = F.pad(padded.cumsum(-1).cumsum(-2), (1, 0, 1, 0))
        sums = integral[..., height:, width:] - integral[..., :-height, width:]
        sums = sums - integral[..., height:, :-width] + integral[..., :-height, :-width]
        mean, mean_square = (sums / (height * width)).chunk(2, dim=1)
        variance = mean_square - mean.square()
        positive = variance > 0
        # Avoid sqrt(0) in the differentiated branch, including on torch 2.5.1.
        std = torch.where(positive, torch.where(positive, variance, 1.0).sqrt(), 0.0)
        threshold = ((mean + offset) * (1.0 + k * (std / r - 1.0))).to(values.dtype)
    return values > threshold, threshold


class SauvolaThreshold(nn.Module):
    r"""Apply a configured Sauvola local threshold to batches of images.

    ``SauvolaThreshold(window_size, k, r)(input)`` returns the same mask and local
    threshold map as :func:`sauvola_threshold`, with the window and contrast parameters
    stored on the module. It has no learnable parameters.

    Args:
        window_size: Positive odd size or ``(height, width)`` window sizes.
        k: Finite nonnegative local-contrast sensitivity.
        r: Positive standard-deviation scale; ``None`` uses the input dtype's default.

    Shape:
        - Input: ``(B, C, H, W)``
        - Outputs: two tensors with shape ``(B, C, H, W)``.

    See Also:
        :func:`sauvola_threshold`: the formula, intensity-scale and boundary conventions.
        :class:`OtsuThreshold`: a global threshold based on the image histogram.

    """

    def __init__(self, window_size: tuple[int, int] | int = 15, k: float = 0.2, r: float | None = None) -> None:
        super().__init__()
        self.window_size = window_size
        self.k = k
        self.r = r

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a bright-pixel mask and a local threshold map for the input images."""
        return sauvola_threshold(input, self.window_size, self.k, self.r)

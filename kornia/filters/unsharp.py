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

import torch
from torch import nn

from .filter import _to_floating
from .gaussian import gaussian_blur2d


def unsharp_mask(
    input: torch.Tensor,
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | torch.Tensor,
    border_type: str = "reflect",
) -> torch.Tensor:
    r"""Sharpen an image by adding back the detail that a Gaussian blur removes.

    ``out = unsharp_mask(input, kernel_size, sigma)`` returns ``2 * input - gaussian_blur2d(input, kernel_size, sigma)``
    for a ``(B, C, H, W)`` image, an image of the same shape.

    .. image:: _static/img/unsharp_mask.png

    Convention:
        - The blur takes ``kernel_size``, ``sigma`` and ``border_type`` as :func:`~kornia.filters.gaussian_blur2d`
          does, and the detail it removes is added back with a fixed gain of 1. See its Convention block for the
          argument order.
        - The output is not clamped: next to an edge it overshoots on both sides, so a step from 0 to 1 comes back
          below 0 and above 1. :ref:`Filtering <filtering-conventions>` maps the call onto scikit-image.
        - An integer or bool image is filtered in float32 and the result is float32 on its own scale, like
          :func:`~kornia.filters.gaussian_blur2d`; a floating image keeps its dtype.
        - Under ``torch.autocast`` the blur runs in the autocast dtype, as in :func:`~kornia.filters.gaussian_blur2d`
          (autocast leaves a float64 image alone), and the result keeps the image's dtype, as for
          :func:`~kornia.enhance.sharpness`.

    Args:
        input: image with shape ``(B, C, H, W)``.
        kernel_size: window ``(kH, kW)`` in pixels, or one integer for both; each a positive odd integer.
        sigma: standard deviation ``(sigma_y, sigma_x)`` in pixels, as a tuple or a tensor.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.

    Returns:
        the sharpened image with shape ``(B, C, H, W)``.

    See Also:
        - :class:`UnsharpMask`: the module form.
        - :func:`gaussian_blur2d`: the blur alone.
        - :func:`~kornia.enhance.sharpness`: blends the image with a blurred copy by a ``factor`` you choose.

    Example:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = unsharp_mask(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """
    input = _to_floating(input)
    data_blur: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma, border_type)
    # Under autocast the blur, a convolution, comes back in the autocast dtype, and torch.lerp, which is not an autocast
    # op and does not promote, raised on the dtype mismatch with the image. Casting the blur back gives the image's
    # dtype, as 2 * image - blur would; the cast is a no-op outside autocast.
    return torch.lerp(data_blur.to(input.dtype), input, weight=2.0)


class UnsharpMask(nn.Module):
    r"""Sharpen an image with a stored Gaussian kernel size and ``sigma`` (module form).

    ``sharpen = UnsharpMask(kernel_size, sigma)`` stores its arguments, and ``out = sharpen(input)`` is
    :func:`unsharp_mask` on a ``(B, C, H, W)`` image.

    Convention:
        See the Convention block on :func:`~kornia.filters.unsharp_mask`.

    Args:
        kernel_size: window ``(kH, kW)`` in pixels, or one integer for both.
        sigma: standard deviation ``(sigma_y, sigma_x)`` in pixels, as a tuple or a tensor.
        border_type: padding mode, as in :func:`unsharp_mask`.

    Shape:
        - Input: ``(B, C, H, W)``
        - Output: ``(B, C, H, W)``

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/unsharp_mask.html>`__.

    See Also:
        - :func:`unsharp_mask`: the one-call form.
        - :class:`GaussianBlur2d`: the blur alone.

    Example:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> sharpen = UnsharpMask((3, 3), (1.5, 1.5))
        >>> output = sharpen(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma: tuple[float, float] | torch.Tensor,
        border_type: str = "reflect",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Sharpen an image by adding back high-frequency detail.

        Unsharp masking first builds a blurred version of the image and then
        uses the difference between the original and blurred images as a detail
        signal. Adding that detail back emphasizes edges and fine texture while
        keeping the tensor layout unchanged.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the image height, and :math:`W` is the
                image width.

        Returns:
            Tensor with shape ``(B, C, H, W)`` containing the sharpened
            image. The spatial scale of sharpening is set by the configured
            Gaussian kernel size and sigma; the amount is fixed.
        """
        return unsharp_mask(input, self.kernel_size, self.sigma, self.border_type)

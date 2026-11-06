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
"""Sepia color filter for RGB image tensors."""

import torch
from torch import nn


def sepia_from_rgb(input: torch.Tensor, rescale: bool = True, eps: float = 1e-6) -> torch.Tensor:
    r"""Apply the sepia filter to an RGB image.

    ``out = sepia(input)`` multiplies the RGB channels of a ``(*, 3, H, W)`` image by a fixed sepia matrix. By default
    it then divides by the image's largest output value.

    Convention:
        - ``rescale`` uses one maximum per image, over channels and pixels, so the tint is preserved.
        - Without ``rescale`` the output is the raw matrix response and can exceed 1.

    Args:
        input: RGB image with shape ``(*, 3, H, W)``.
        rescale: if True, divide by the maximum.
        eps: constant added to the maximum, so a black image does not divide by zero.

    Returns:
        Sepia image with shape ``(*, 3, H, W)``.

    See Also:
        - :class:`Sepia`: the module form, with ``rescale`` and ``eps`` stored.

    Example:
        >>> input = torch.ones(3, 1, 1)
        >>> sepia_from_rgb(input, rescale=False)
        tensor([[[1.3510]],
        <BLANKLINE>
                [[1.2030]],
        <BLANKLINE>
                [[0.9370]]])
    """
    if len(input.shape) < 3 or input.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    r = input[..., 0, :, :]
    g = input[..., 1, :, :]
    b = input[..., 2, :, :]

    r_out = 0.393 * r + 0.769 * g + 0.189 * b
    g_out = 0.349 * r + 0.686 * g + 0.168 * b
    b_out = 0.272 * r + 0.534 * g + 0.131 * b

    sepia_out = torch.stack([r_out, g_out, b_out], dim=-3)

    if rescale:
        max_values = sepia_out.amax(dim=(-3, -2, -1), keepdim=True)
        sepia_out = sepia_out / (max_values + eps)

    return sepia_out


class Sepia(nn.Module):
    r"""Apply the sepia filter with stored settings (module form).

    ``filt = Sepia(rescale, eps)`` stores the settings, and ``out = filt(input)`` is :func:`sepia` on a ``(*, 3, H, W)``
    image.

    Convention:
        - See the convention block of :func:`sepia`.

    Args:
        rescale: if True, divide by the per-image maximum.
        eps: constant added to the maximum.

    See Also:
        - :func:`sepia`: the one-call form.

    Example:
        >>> input = torch.ones(3, 1, 1)
        >>> Sepia(rescale=False)(input)
        tensor([[[1.3510]],
        <BLANKLINE>
                [[1.2030]],
        <BLANKLINE>
                [[0.9370]]])
    """

    def __init__(self, rescale: bool = True, eps: float = 1e-6) -> None:
        """Initialize Sepia.

        Args:
            rescale: if True, divide all output channels by one maximum per image.
            eps: small constant added to the denominator for numerical stability.
        """
        self.rescale = rescale
        self.eps = eps
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of this module."""
        return self.__class__.__name__ + f"(rescale={self.rescale}, eps={self.eps})"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the sepia filter.

        Args:
            input: RGB image tensor with shape ``(*, 3, H, W)``.

        Returns:
            Sepia-filtered tensor with shape ``(*, 3, H, W)``.
        """
        return sepia_from_rgb(input, rescale=self.rescale, eps=self.eps)

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

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


def total_variation(img: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    r"""Compute the total variation of an image.

    ``tv = total_variation(img)`` adds, for each ``(H, W)`` slice of ``img``, the absolute differences between
    vertically and between horizontally adjacent pixels, and returns one value per leading index.

    .. math::

        \text{TV}(I) = \sum |\partial_y I| + \sum |\partial_x I|

    Convention:
        - The channels are not summed: a ``(B, C, H, W)`` batch gives ``(B, C)``.
        - ``reduction='sum'`` is not normalised by the image size. ``'mean'`` averages each term over its own count,
          :math:`(H - 1) W` vertical and :math:`H (W - 1)` horizontal differences, and adds the two means; it is not
          the sum divided by :math:`H W`. An axis with no adjacent pixels contributes zero. There is no ``'none'``.
          :ref:`Losses and metrics <losses-metrics-conventions>` compares both with torchmetrics.
        - Narrow integer inputs are promoted before the differences, to avoid overflow.

    Args:
        img: image with shape ``(*, H, W)``.
        reduction: ``'sum'`` (default) adds the differences, ``'mean'`` averages each term.

    Returns:
        the total variation with shape ``(*,)``.

    See Also:
        - :class:`~kornia.losses.TotalVariation`: the module form, which always sums.
        - :func:`~kornia.losses.inverse_depth_smoothness_loss`: smoothness of a depth map, weighted by image edges.

    Examples:
        >>> total_variation(torch.ones(4, 4))
        tensor(0.)
        >>> total_variation(torch.ones(2, 5, 3, 4, 4)).shape
        torch.Size([2, 5, 3])

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/total_variation_denoising.html>`__.

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation

    """
    # TODO: here torchscript doesn't like KORNIA_CHECK_TYPE
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Not a torch.Tensor type. Got: {type(img)}")

    KORNIA_CHECK_SHAPE(img, ["*", "H", "W"])
    KORNIA_CHECK(reduction in ("mean", "sum"), f"Expected reduction to be one of 'mean'/'sum', but got '{reduction}'.")

    # Adjacent differences and their absolute values can exceed the input integer range.
    if img.dtype in (torch.uint8, torch.int8):
        img = img.to(torch.int16)
    elif img.dtype == torch.int16:
        img = img.to(torch.int32)
    elif img.dtype == torch.int32:
        img = img.to(torch.int64)

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    # Reduce over one flattened dim rather than dim=(-2, -1): the result is identical,
    # and MPS runs a two-trailing-dims reduction several times slower than a one-dim one.
    res1 = pixel_dif1.abs().flatten(-2)
    res2 = pixel_dif2.abs().flatten(-2)

    if reduction == "mean":
        if not img.is_floating_point():
            res1 = res1.float()
            res2 = res2.float()
        # A singleton spatial axis has no differences: sum the empty term to zero instead of averaging it to NaN.
        res1 = res1.mean(dim=-1) if res1.shape[-1] > 0 else res1.sum(dim=-1)
        res2 = res2.mean(dim=-1) if res2.shape[-1] > 0 else res2.sum(dim=-1)
    elif reduction == "sum":
        res1 = res1.sum(dim=-1)
        res2 = res2.sum(dim=-1)
    else:
        raise NotImplementedError("Invalid reduction option.")

    return res1 + res2


class TotalVariation(nn.Module):
    r"""Compute the total variation of an image (module form).

    ``tv = TotalVariation()`` takes no arguments, and ``out = tv(img)`` is :func:`~kornia.losses.total_variation`
    with ``reduction='sum'``.

    Convention:
        See the Convention block of :func:`~kornia.losses.total_variation`.

    Shape:
        - Input: ``(*, H, W)``.
        - Output: ``(*,)``.

    See Also:
        - :func:`~kornia.losses.total_variation`: the one-call form, with ``reduction='mean'`` too.

    Examples:
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    """

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Compute total variation for an image or image batch.

        Args:
            img: Tensor with shape ``(*, H, W)``, where ``*`` represents
                optional leading dimensions such as batch and channels,
                :math:`H` is height, and :math:`W` is width.

        Returns:
            Tensor containing the sum of absolute horizontal and vertical
            finite differences for each leading element. Lower values indicate
            spatially smoother images.
        """
        return total_variation(img)

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

from kornia import metrics


def ssim3d_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    reduction: str = "mean",
    padding: str = "same",
) -> torch.Tensor:
    r"""Compute the DSSIM loss between two volumes from their SSIM map.

    ``loss = ssim3d_loss(img1, img2, window_size)`` maps the SSIM of two ``(B, C, D, H, W)`` volumes to a
    dissimilarity in :math:`[0, 1]`, as :func:`~kornia.losses.ssim_loss` does for images.

    .. math::

      \text{loss}(x, y) = \operatorname{clamp}\left(\frac{1 - \text{SSIM}(x, y)}{2}, 0, 1\right)

    Convention:
        See the Convention block of :func:`~kornia.metrics.ssim3d` for the window, the border and ``max_val``, and
        that of :func:`~kornia.losses.ssim_loss` for the loss.

    Args:
        img1: first volume with shape ``(B, C, D, H, W)``.
        img2: second volume with the same shape.
        window_size: size of the Gaussian window in voxels, a positive odd integer.
        max_val: data range :math:`L` of the volumes, as a Python float.
        eps: term added to the denominator of the SSIM.
        reduction: ``'mean'`` (default) averages the loss, ``'sum'`` adds it, ``'none'`` keeps the SSIM map's shape.
        padding: ``'same'`` or ``'valid'``, as in :func:`~kornia.metrics.ssim3d`.

    Returns:
        the loss, a scalar or, for ``reduction='none'``, with the shape of the SSIM map.

    See Also:
        - :class:`~kornia.losses.SSIM3DLoss`: the module form.
        - :func:`~kornia.metrics.ssim3d`: the SSIM map itself, not the dissimilarity.
        - :func:`~kornia.losses.ssim_loss`: the same loss for images.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5, 5)
        >>> loss = ssim3d_loss(input1, input2, 5)

    """
    # compute the ssim map
    ssim_map: torch.Tensor = metrics.ssim3d(img1, img2, window_size, max_val, eps, padding)

    # compute and reduce the loss
    loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError("Invalid reduction option.")

    return loss


class SSIM3DLoss(nn.Module):
    r"""Compute the DSSIM loss between two volumes from their SSIM map (module form).

    ``criterion = SSIM3DLoss(window_size, ...)`` stores its arguments, and ``loss = criterion(img1, img2)`` is
    :func:`~kornia.losses.ssim3d_loss` on two ``(B, C, D, H, W)`` volumes.

    Convention:
        See the Convention block of :func:`~kornia.losses.ssim3d_loss`.

    Args:
        window_size: size of the Gaussian window in voxels, a positive odd integer.
        max_val: data range :math:`L` of the volumes, as a Python float.
        eps: term added to the denominator of the SSIM.
        reduction: ``'mean'`` (default), ``'sum'`` or ``'none'``, as in :func:`~kornia.losses.ssim3d_loss`.
        padding: ``'same'`` or ``'valid'``, as in :func:`~kornia.metrics.ssim3d`.

    Shape:
        - img1: ``(B, C, D, H, W)``.
        - img2: ``(B, C, D, H, W)``.
        - Output: a scalar or, for ``reduction='none'``, the shape of the SSIM map.

    See Also:
        - :func:`~kornia.losses.ssim3d_loss`: the one-call form.
        - :class:`~kornia.losses.SSIMLoss`: the same loss for images.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5, 5)
        >>> criterion = SSIM3DLoss(5)
        >>> loss = criterion(input1, input2)

    """

    def __init__(
        self, window_size: int, max_val: float = 1.0, eps: float = 1e-12, reduction: str = "mean", padding: str = "same"
    ) -> None:
        super().__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.eps: float = eps
        self.reduction: str = reduction
        self.padding: str = padding

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss between two 3D image or volume tensors.

        Args:
            img1: First volume batch with shape ``(B, C, D, H, W)``, where
                :math:`B` is batch size, :math:`C` is channel count,
                :math:`D` is depth, :math:`H` is height, and :math:`W` is
                width.
            img2: Second volume batch with the same shape as ``img1``.

        Returns:
            SSIM3D loss tensor reduced according to ``self.reduction``. Lower
            values indicate stronger local structural agreement between the two
            volumes.
        """
        return ssim3d_loss(img1, img2, self.window_size, self.max_val, self.eps, self.reduction, self.padding)

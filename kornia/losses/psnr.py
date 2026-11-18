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
from kornia.metrics.psnr import _PSNRMaxVal


def psnr_loss(image: torch.Tensor, target: torch.Tensor, max_val: _PSNRMaxVal) -> torch.Tensor:
    r"""Compute the PSNR loss, the negative PSNR of two images.

    ``loss = psnr_loss(image, target, max_val)`` is ``-psnr(image, target, max_val)``, so minimising it raises
    the PSNR.

    Convention:
        See the Convention block of :func:`~kornia.metrics.psnr`: the MSE is pooled over the batch, and identical
        inputs give ``-inf``.

    Args:
        image: image with shape ``(*)``.
        target: target with the same shape as ``image``.
        max_val: data range :math:`\text{MAX}_I` of the images, as a float, integer or tensor.

    Returns:
        the loss, a scalar for a scalar range or with the shape of a tensor range.

    See Also:
        - :class:`~kornia.losses.PSNRLoss`: the module form.
        - :func:`~kornia.metrics.psnr`: the PSNR itself, with the opposite sign.

    Examples:
        >>> ones = torch.ones(1)
        >>> psnr_loss(ones, 1.2 * ones, 2.) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(-20.0000)

    """
    return -1.0 * metrics.psnr(image, target, max_val)


class PSNRLoss(nn.Module):
    r"""Compute the PSNR loss, the negative PSNR of two images (module form).

    ``criterion = PSNRLoss(max_val)`` stores ``max_val``, and ``loss = criterion(image, target)`` is
    :func:`~kornia.losses.psnr_loss`.

    Convention:
        See the Convention block of :func:`~kornia.losses.psnr_loss`.

    Args:
        max_val: data range :math:`\text{MAX}_I` of the images, as a float, integer or tensor.

    Shape:
        - Image: ``(*)``.
        - Target: ``(*)``, the shape of the image.
        - Output: a scalar for a scalar range, or the shape of a tensor range.

    See Also:
        - :func:`~kornia.losses.psnr_loss`: the one-call form.
        - :func:`~kornia.metrics.psnr`: the PSNR itself, with the opposite sign.

    Examples:
        >>> ones = torch.ones(1)
        >>> criterion = PSNRLoss(2.)
        >>> criterion(ones, 1.2 * ones) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(-20.0000)

    """

    def __init__(self, max_val: _PSNRMaxVal) -> None:
        super().__init__()
        self.max_val: _PSNRMaxVal = max_val

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the negative PSNR loss between two tensors.

        Args:
            image: Predicted image or signal tensor with arbitrary shape.
            target: Reference tensor with the same shape as ``image``.

        Returns:
            Tensor containing the PSNR-based loss computed with
            ``self.max_val``. Lower values correspond to higher peak signal to
            noise ratio.
        """
        return psnr_loss(image, target, self.max_val)

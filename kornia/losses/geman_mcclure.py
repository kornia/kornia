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

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_DEVICE, KORNIA_CHECK_SAME_SHAPE


def geman_mcclure_loss(img1: torch.Tensor, img2: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    r"""Compute the Geman-McClure loss between a prediction and a target.

    ``loss = geman_mcclure_loss(img1, img2)`` penalises each residual :math:`x - y` of ``img1`` (:math:`x`) and
    ``img2`` (:math:`y`) quadratically near 0 and saturates at 2, and its gradient vanishes for large residuals.

    .. math::

        \text{loss}(x, y) = \frac{2 (x - y)^{2}}{(x - y)^{2} + 4}

    Reference:
        [1] https://arxiv.org/pdf/1701.03077.pdf

        [2] Geman and McClure, 1985: Bayesian image analysis:
        An application to single photon emission tomography.

    Convention:
        See the Convention block of :func:`~kornia.losses.charbonnier_loss`; this is Barron's loss with
        :math:`\alpha = -2`.

    Args:
        img1: prediction with shape ``(*)``.
        img2: target with the same shape as ``img1``.
        reduction: ``'none'`` (default) keeps one loss per element, ``'mean'`` averages them, ``'sum'`` adds them.

    Returns:
        the loss with the shape of the inputs for ``reduction='none'``, a scalar otherwise.

    See Also:
        - :class:`~kornia.losses.GemanMcclureLoss`: the module form.
        - :func:`~kornia.losses.cauchy_loss`: grows logarithmically, without bound.
        - :func:`~kornia.losses.welsch_loss`: saturates at 1, faster.

    Example:
        >>> img1 = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 32)
        >>> output = geman_mcclure_loss(img1, img2, reduction="mean")
        >>> output.backward()

    """
    KORNIA_CHECK_IS_TENSOR(img1)

    KORNIA_CHECK_IS_TENSOR(img2)

    KORNIA_CHECK_SAME_SHAPE(img1, img2)

    KORNIA_CHECK_SAME_DEVICE(img1, img2)

    KORNIA_CHECK(
        reduction in ("mean", "sum", "none", None), f"Given type of reduction is not supported. Got: {reduction}"
    )

    # Keep the square, doubled numerator and their backward in float32 for half-precision residuals.
    output_dtype = torch.promote_types(img1.dtype, img2.dtype)
    compute_diff = img1.float() - img2.float() if output_dtype in (torch.float16, torch.bfloat16) else img1 - img2
    diff2 = torch.square(compute_diff)
    loss = 2.0 * diff2 / (diff2 + 4.0)

    # perform reduction
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none" or reduction is None:
        pass
    else:
        raise NotImplementedError("Invalid reduction option.")

    return loss.to(output_dtype) if output_dtype in (torch.float16, torch.bfloat16) else loss


class GemanMcclureLoss(nn.Module):
    r"""Compute the Geman-McClure loss between a prediction and a target (module form).

    ``criterion = GemanMcclureLoss(reduction)`` stores ``reduction``, and ``loss = criterion(img1, img2)`` is
    :func:`~kornia.losses.geman_mcclure_loss`.

    Convention:
        See the Convention block of :func:`~kornia.losses.geman_mcclure_loss`.

    Args:
        reduction: ``'none'`` (default), ``'mean'`` or ``'sum'``, as in :func:`~kornia.losses.geman_mcclure_loss`.

    Shape:
        - img1: prediction ``(*)``.
        - img2: target with the same shape as ``img1``.
        - Output: the shape of the inputs for ``reduction='none'``, a scalar otherwise.

    See Also:
        - :func:`~kornia.losses.geman_mcclure_loss`: the one-call form.
        - :class:`~kornia.losses.CauchyLoss`: grows logarithmically, without bound.
        - :class:`~kornia.losses.WelschLoss`: saturates at 1, faster.

    Example:
        >>> criterion = GemanMcclureLoss(reduction="mean")
        >>> img1 = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 32)
        >>> output = criterion(img1, img2)
        >>> output.backward()

    """

    def __init__(self, reduction: str = "none") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute the Geman-McClure robust regression loss.

        Args:
            img1: Predicted tensor with arbitrary shape.
            img2: Target tensor with the same shape as ``img1``.

        Returns:
            Loss tensor reduced according to ``self.reduction``. The robust
            penalty limits the contribution of large residuals compared with a
            plain squared-error loss.
        """
        return geman_mcclure_loss(img1=img1, img2=img2, reduction=self.reduction)

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

# from torch import Tensor (use torch.Tensor instead)
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_DEVICE, KORNIA_CHECK_SAME_SHAPE


def charbonnier_loss(img1: torch.Tensor, img2: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    r"""Compute the Charbonnier (L1-L2) loss between a prediction and a target.

    ``loss = charbonnier_loss(img1, img2)`` penalises each residual :math:`x - y` of ``img1`` (:math:`x`) and ``img2``
    (:math:`y`) quadratically near 0 and linearly beyond, without bound.

    .. math::

        \text{loss}(x, y) = \sqrt{(x - y)^{2} + 1} - 1

    Reference:
        [1] https://arxiv.org/pdf/1701.03077.pdf
        [2] Charbonnier: https://ieeexplore.ieee.org/document/413553
        [3] L1-L2: https://hal.inria.fr/inria-00074015/document
        [4] https://arxiv.org/pdf/1712.05927.pdf

    Convention:
        - The robust losses are the general loss of Barron [1] on the residual ``img1 - img2`` at scale :math:`c = 1`:
          :math:`\alpha = 1` here, :math:`\alpha = 0` in :func:`~kornia.losses.cauchy_loss`, :math:`\alpha = -2` in
          :func:`~kornia.losses.geman_mcclure_loss` and :math:`\alpha = -\infty` in
          :func:`~kornia.losses.welsch_loss`. Other works use a slightly different Charbonnier form [4].
        - There is no scale argument: the penalty stops being quadratic at residuals of about 1 in the units of the
          data. For a scale :math:`c`, pass ``img1 / c`` and ``img2 / c``.
          :ref:`Losses and metrics <losses-metrics-conventions>` maps the losses onto Barron's code.
        - The losses are symmetric in ``img1`` and ``img2``, which must have the same shape, without broadcasting,
          and the same device.
        - This loss, :func:`~kornia.losses.cauchy_loss`, :func:`~kornia.losses.geman_mcclure_loss` and
          :func:`~kornia.losses.welsch_loss` evaluate float16 and bfloat16 inputs in float32, gradients included, and
          return the input dtype.

    Args:
        img1: prediction with shape ``(*)``.
        img2: target with the same shape as ``img1``.
        reduction: ``'none'`` (default) keeps one loss per element, ``'mean'`` averages them, ``'sum'`` adds them.

    Returns:
        the loss with the shape of the inputs for ``reduction='none'``, a scalar otherwise.

    See Also:
        - :class:`~kornia.losses.CharbonnierLoss`: the module form.
        - :func:`~kornia.losses.cauchy_loss`: grows only logarithmically for large residuals.

    Example:
        >>> img1 = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 32)
        >>> output = charbonnier_loss(img1, img2, reduction="sum")
        >>> output.backward()

    """
    KORNIA_CHECK_IS_TENSOR(img1)

    KORNIA_CHECK_IS_TENSOR(img2)

    KORNIA_CHECK_SAME_SHAPE(img1, img2)

    KORNIA_CHECK_SAME_DEVICE(img1, img2)

    KORNIA_CHECK(
        reduction in ("mean", "sum", "none", None), f"Given type of reduction is not supported. Got: {reduction}"
    )

    # Keep the square and its backward in float32 for half-precision residuals.
    output_dtype = torch.promote_types(img1.dtype, img2.dtype)
    compute_diff = img1.float() - img2.float() if output_dtype in (torch.float16, torch.bfloat16) else img1 - img2
    # Rationalize small residuals to avoid subtracting two nearly equal numbers.
    # Keep the original large-residual branch, including its behavior when the square overflows.
    squared = compute_diff**2
    small = squared < 1.0
    small_squared = torch.where(small, squared, 0.0)
    loss = torch.where(
        small,
        small_squared / ((small_squared + 1.0).sqrt() + 1.0),
        (squared + 1.0).sqrt() - 1.0,
    )

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


class CharbonnierLoss(nn.Module):
    r"""Compute the Charbonnier (L1-L2) loss between a prediction and a target (module form).

    ``criterion = CharbonnierLoss(reduction)`` stores ``reduction``, and ``loss = criterion(img1, img2)`` is
    :func:`~kornia.losses.charbonnier_loss`.

    Convention:
        See the Convention block of :func:`~kornia.losses.charbonnier_loss`.

    Args:
        reduction: ``'none'`` (default), ``'mean'`` or ``'sum'``, as in :func:`~kornia.losses.charbonnier_loss`.

    Shape:
        - img1: prediction ``(*)``.
        - img2: target with the same shape as ``img1``.
        - Output: the shape of the inputs for ``reduction='none'``, a scalar otherwise.

    See Also:
        - :func:`~kornia.losses.charbonnier_loss`: the one-call form.
        - :class:`~kornia.losses.CauchyLoss`: grows only logarithmically for large residuals.

    Example:
        >>> criterion = CharbonnierLoss(reduction="mean")
        >>> img1 = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 32)
        >>> output = criterion(img1, img2)
        >>> output.backward()

    """

    def __init__(self, reduction: str = "none") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute the Charbonnier robust regression loss.

        Args:
            img1: Predicted tensor with arbitrary shape.
            img2: Target tensor with the same shape as ``img1``.

        Returns:
            Loss tensor reduced according to ``self.reduction``. The
            Charbonnier penalty is a smooth approximation of absolute error and
            remains differentiable near zero residual.
        """
        return charbonnier_loss(img1=img1, img2=img2, reduction=self.reduction)

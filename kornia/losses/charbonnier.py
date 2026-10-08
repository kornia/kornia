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
    r"""Criterion that computes the Charbonnier [2] (aka. L1-L2 [3]) loss.

    According to [1], we compute the Charbonnier loss as follows:

    .. math::

        \text{loss}(x, y) = \sqrt{(x - y)^{2} + 1} - 1

    Where:
       - :math:`x` is the prediction.
       - :math:`y` is the target to be regressed to.

    Reference:
        [1] https://arxiv.org/pdf/1701.03077.pdf
        [2] https://ieeexplore.ieee.org/document/413553
        [3] https://hal.inria.fr/inria-00074015/document
        [4] https://arxiv.org/pdf/1712.05927.pdf

    .. note::
        This implementation follows the formulation by Barron [1]. Other works utilize
        a slightly different implementation (see [4]).

    Convention:
        - The robust losses are the general loss of Barron [1] on the residual ``img1 - img2`` at scale
          :math:`c = 1`: :math:`\alpha = 1` here, :math:`\alpha = 0` in :func:`~kornia.losses.cauchy_loss`,
          :math:`\alpha = -2` in :func:`~kornia.losses.geman_mcclure_loss` and :math:`\alpha = -\infty` in
          :func:`~kornia.losses.welsch_loss`.
        - There is no scale argument: :math:`c = 1` is in the units of the data, so the shape of the penalty depends
          on them; for a scale :math:`c`, pass ``img1 / c`` and ``img2 / c``.
          :ref:`Losses and metrics <losses-metrics-conventions>` maps the losses onto Barron's code.
        - The losses are symmetric in ``img1`` and ``img2``, which must have the same shape, without broadcasting,
          and the same device. The default ``reduction='none'`` returns the loss of every element, in the shape of
          the inputs.

    Args:
        img1: the predicted torch.Tensor with shape :math:`(*)`.
        img2: the target torch.Tensor with the same shape as img1.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied (default), ``'mean'``: the sum of the output will be divided
          by the number of elements in the output, ``'sum'``: the output will be
          summed.

    Return:
        the computed loss, with the shape of the inputs for ``reduction='none'`` and a scalar otherwise.

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
    diff = img1 - img2
    compute_diff = diff.float() if diff.dtype in (torch.float16, torch.bfloat16) else diff
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

    return loss.to(diff.dtype) if diff.dtype in (torch.float16, torch.bfloat16) else loss


class CharbonnierLoss(nn.Module):
    r"""Criterion that computes the Charbonnier [2] (aka. L1-L2 [3]) loss.

    According to [1], we compute the Charbonnier loss as follows:

    .. math::

        \text{loss}(x, y) = \sqrt{(x - y)^{2} + 1} - 1

    Where:
       - :math:`x` is the prediction.
       - :math:`y` is the target to be regressed to.

    Reference:
        [1] https://arxiv.org/pdf/1701.03077.pdf
        [2] https://ieeexplore.ieee.org/document/413553
        [3] https://hal.inria.fr/inria-00074015/document
        [4] https://arxiv.org/pdf/1712.05927.pdf

    .. note::
        This implementation follows the formulation by Barron [1]. Other works utilize
        a slightly different implementation (see [4]).

    Convention:
        See the Convention block of :func:`~kornia.losses.charbonnier_loss`.

    Args:
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied (default), ``'mean'``: the sum of the output will be divided
          by the number of elements in the output, ``'sum'``: the output will be
          summed.

    Shape:
        - img1: the predicted torch.Tensor with shape :math:`(*)`.
        - img2: the target torch.Tensor with the same shape as img1.

    Example:
        >>> criterion = CharbonnierLoss(reduction="mean")
        >>> img1 = torch.randn(2, 3, 32, 2107, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 2107)
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

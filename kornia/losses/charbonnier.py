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

from torch import Tensor

from kornia.core import Module
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_DEVICE, KORNIA_CHECK_SAME_SHAPE


def charbonnier_loss(img1: Tensor, img2: Tensor, reduction: str = "none") -> Tensor:
    r"""Criterion that computes the Charbonnier [2] (aka. L1-L2 [3]) loss.

    According to [1], we compute the Charbonnier loss as follows:

    .. math::

        \text{WL}(x, y) = \sqrt{(x - y)^{2} + 1} - 1

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

    Args:
        img1: the predicted tensor with shape :math:`(*)`.
        img2: the target tensor with the same shape as img1.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied (default), ``'mean'``: the sum of the output will be divided
          by the number of elements in the output, ``'sum'``: the output will be
          summed.

    Return:
        a scalar with the computed loss.

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

    # compute loss
    loss = ((img1 - img2) ** 2 + 1.0).sqrt() - 1.0

    # perform reduction
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none" or reduction is None:
        pass
    else:
        raise NotImplementedError("Invalid reduction option.")

    return loss


class CharbonnierLoss(Module):
    r"""Criterion that computes the Charbonnier [2] (aka. L1-L2 [3]) loss.

    According to [1], we compute the Charbonnier loss as follows:

    .. math::

        \text{WL}(x, y) = \sqrt{(x - y)^{2} + 1} - 1

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

    Args:
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied (default), ``'mean'``: the sum of the output will be divided
          by the number of elements in the output, ``'sum'``: the output will be
          summed.

    Shape:
        - img1: the predicted tensor with shape :math:`(*)`.
        - img2: the target tensor with the same shape as img1.

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

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        return charbonnier_loss(img1=img1, img2=img2, reduction=self.reduction)

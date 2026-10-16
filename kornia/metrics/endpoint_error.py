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

import torch
from torch import Tensor, nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE


def aepe(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""Calculate the average endpoint error (AEPE) between 2 flow maps.

    AEPE is the endpoint error between two 2D vectors (e.g., optical flow).
    Given a h x w x 2 optical flow map, the AEPE is:

    .. math::

        \text{AEPE}=\frac{1}{hw}\sum_{i=1, j=1}^{h, w}\sqrt{(I_{i,j,1}-T_{i,j,1})^{2}+(I_{i,j,2}-T_{i,j,2})^{2}}

    Convention:
        - Flow is channel-last, :math:`(*, 2)` such as :math:`(B, H, W, 2)`, in the units of the flow (pixels for
          optical flow). Permute a channel-first :math:`(B, 2, H, W)` flow to :math:`(B, H, W, 2)` first: it raises
          unless :math:`W = 2`, where it is read wrongly. :ref:`Losses and metrics <losses-metrics-conventions>` ports
          RAFT's endpoint error.
        - The endpoint error is the Euclidean distance between the two vectors at every position, symmetric in
          ``input`` and ``target`` and not normalized by the image size. ``reduction='mean'`` (the default) averages
          it over every position of every sample at once and ``'sum'`` adds it up, both into a 0-d tensor;
          ``'none'`` returns the :math:`(*)` map. There is no valid mask: mask sparse ground truth on the ``'none'``
          map. An unknown ``reduction`` raises ``NotImplementedError``.
        - :func:`~kornia.metrics.aepe` and :func:`~kornia.metrics.average_endpoint_error` are the same function.

    Args:
        input: the input flow map with shape :math:`(*, 2)`.
        target: the target flow map with shape :math:`(*, 2)`.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Return:
        the computed AEPE as a 0-d tensor, or the endpoint-error map of shape :math:`(*)` for ``reduction='none'``.

    Examples:
        >>> ones = torch.ones(4, 4, 2)
        >>> aepe(ones, 1.2 * ones)
        tensor(0.2828)

    Reference:
        https://link.springer.com/content/pdf/10.1007/s11263-010-0390-2.pdf

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_IS_TENSOR(target)
    KORNIA_CHECK_SHAPE(input, ["*", "2"])
    KORNIA_CHECK_SHAPE(target, ["*", "2"])
    KORNIA_CHECK(
        input.shape == target.shape, f"input and target shapes must be the same. Got: {input.shape} and {target.shape}"
    )

    epe: Tensor = ((input[..., 0] - target[..., 0]) ** 2 + (input[..., 1] - target[..., 1]) ** 2).sqrt()

    if reduction == "mean":
        epe = epe.mean()
    elif reduction == "sum":
        epe = epe.sum()
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError("Invalid reduction option.")

    return epe


class AEPE(nn.Module):
    r"""Computes the average endpoint error (AEPE) between 2 flow maps.

    EPE is the endpoint error between two 2D vectors (e.g., optical flow).
    Given a h x w x 2 optical flow map, the AEPE is:

    .. math::

        \text{AEPE}=\frac{1}{hw}\sum_{i=1, j=1}^{h, w}\sqrt{(I_{i,j,1}-T_{i,j,1})^{2}+(I_{i,j,2}-T_{i,j,2})^{2}}

    Convention:
        See the Convention block of :func:`~kornia.metrics.aepe`, which this module calls with its ``reduction``.

    Args:
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Shape:
        - input: :math:`(*, 2)`.
        - target :math:`(*, 2)`.
        - output: :math:`()` for ``'mean'`` and ``'sum'``, :math:`(*)` for ``'none'``.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 2)
        >>> input2 = torch.rand(1, 4, 5, 2)
        >>> epe = AEPE(reduction="mean")
        >>> epe = epe(input1, input2)

    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute average endpoint error between vector fields.

        Args:
            input: Predicted vector field with shape :math:`(B, H, W, 2)`.
            target: Target vector field with the same shape as ``input``.

        Returns:
            Endpoint-error tensor reduced according to ``self.reduction``.
            The endpoint error is the Euclidean distance between predicted and
            target vectors at each spatial location.
        """
        return aepe(input, target, self.reduction)


average_endpoint_error = aepe

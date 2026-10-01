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

from typing import Union

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR


def add_weighted(
    src1: torch.Tensor,
    alpha: Union[float, torch.Tensor],
    src2: torch.Tensor,
    beta: Union[float, torch.Tensor],
    gamma: Union[float, torch.Tensor],
) -> torch.Tensor:
    r"""Calculate the weighted sum of two Tensors.

    Convention:
        src1 and src2 must have exactly the same shape; a Python number coefficient applies to
        every element. See :class:`AddWeighted` for the module wrapper.

    .. warning::
        Tensor alpha, beta and gamma must have src1's exact shape: a 0-d or broadcastable
        tensor raises (`#5325 <https://github.com/kornia/kornia/issues/5325>`_).

    .. image:: _static/img/add_weighted.png

    The function calculates the weighted sum of two Tensors as follows:

    .. math::
        out = src1 * alpha + src2 * beta + gamma

    Args:
        src1: torch.Tensor with an arbitrary shape, equal to shape of src2.
        alpha: weight of the src1 elements as Union[float, torch.Tensor].
        src2: torch.Tensor with an arbitrary shape, equal to shape of src1.
        beta: weight of the src2 elements as Union[float, torch.Tensor].
        gamma: scalar added to each sum as Union[float, torch.Tensor].

    Returns:
        Weighted torch.Tensor with shape equal to src1 and src2 shapes.

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = add_weighted(input1, 0.5, input2, 0.5, 1.0)
        >>> output.shape
        torch.Size([1, 1, 5, 5])

    """
    KORNIA_CHECK_IS_TENSOR(src1)
    KORNIA_CHECK_IS_TENSOR(src2)
    KORNIA_CHECK(src1.shape == src2.shape, f"src1 and src2 have different shapes. Got {src1.shape} and {src2.shape}")

    if isinstance(alpha, torch.Tensor):
        KORNIA_CHECK(src1.shape == alpha.shape, "alpha has a different shape than src.")

    if isinstance(beta, torch.Tensor):
        KORNIA_CHECK(src1.shape == beta.shape, "beta has a different shape than src.")

    if isinstance(gamma, torch.Tensor):
        KORNIA_CHECK(src1.shape == gamma.shape, "gamma has a different shape than src.")

    return src1 * alpha + src2 * beta + gamma


class AddWeighted(nn.Module):
    r"""Calculate the weighted sum of two Tensors.

    Convention:
        See :func:`add_weighted`, including its tensor-coefficient defect
        (`#5325 <https://github.com/kornia/kornia/issues/5325>`_).

    The function calculates the weighted sum of two Tensors as follows:

    .. math::
        out = src1 * alpha + src2 * beta + gamma

    Args:
        alpha: weight of the src1 elements as Union[float, torch.Tensor].
        beta: weight of the src2 elements as Union[float, torch.Tensor].
        gamma: scalar added to each sum as Union[float, torch.Tensor].

    Shape:
        - Input1: torch.Tensor with an arbitrary shape, equal to shape of Input2.
        - Input2: torch.Tensor with an arbitrary shape, equal to shape of Input1.
        - Output: Weighted torch.Tensor with shape equal to src1 and src2 shapes.

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = AddWeighted(0.5, 0.5, 1.0)(input1, input2)
        >>> output.shape
        torch.Size([1, 1, 5, 5])

    """

    def __init__(
        self, alpha: Union[float, torch.Tensor], beta: Union[float, torch.Tensor], gamma: Union[float, torch.Tensor]
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, src1: torch.Tensor, src2: torch.Tensor) -> torch.Tensor:
        """Compute a weighted sum of two tensors with a scalar bias term.

        This method is a module wrapper around :func:`add_weighted`, using the
        ``alpha``, ``beta``, and ``gamma`` values defined at initialization.

        Args:
            src1: First input tensor.
            src2: Second input tensor. It must have the same shape as ``src1``.

        Returns:
            A tensor with the same shape as the inputs, computed as
            ``src1 * alpha + src2 * beta + gamma`` with broadcasting applied
            where supported by :func:`add_weighted`.
        """
        return add_weighted(src1, self.alpha, src2, self.beta, self.gamma)

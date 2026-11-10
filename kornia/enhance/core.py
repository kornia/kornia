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


def _broadcasts_to(shape: torch.Size, target: torch.Size) -> bool:
    if len(shape) > len(target):
        return False
    return all(dim in (1, tdim) for dim, tdim in zip(reversed(shape), reversed(target)))


def add_weighted(
    src1: torch.Tensor,
    alpha: Union[float, torch.Tensor],
    src2: torch.Tensor,
    beta: Union[float, torch.Tensor],
    gamma: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Blend two tensors as a weighted sum.

    ``out = add_weighted(src1, alpha, src2, beta, gamma)`` returns ``src1 * alpha + src2 * beta + gamma`` for two
    tensors of one shape, without clamping.

    .. image:: _static/img/add_weighted.png

    Convention:
        A tensor coefficient must broadcast to the shape of ``src1`` without enlarging it: a 0-d scalar, ``(C, 1, 1)``
        or a per-image ``(B, 1, 1, 1)`` weight. A ``(B,)`` weight lines up with the width axis, not the batch: it raises
        unless ``B`` equals ``W`` or 1, and then it weights columns. Use ``(B, 1, 1, 1)`` for one weight per image.

    Args:
        src1: tensor of any shape.
        alpha: weight of ``src1``, a number or a tensor.
        src2: tensor with the shape of ``src1``.
        beta: weight of ``src2``, a number or a tensor.
        gamma: value added to the sum, a number or a tensor.

    Returns:
        Weighted sum with the shape of ``src1``.

    See Also:
        - :class:`AddWeighted`: the module form.

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

    for name, coef in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if isinstance(coef, torch.Tensor):
            KORNIA_CHECK(
                _broadcasts_to(coef.shape, src1.shape),
                f"{name} with shape {tuple(coef.shape)} does not broadcast to the src shape {tuple(src1.shape)}.",
            )

    return src1 * alpha + src2 * beta + gamma


class AddWeighted(nn.Module):
    """Blend two tensors as a weighted sum (module form).

    ``op = AddWeighted(alpha, beta, gamma)`` stores the coefficients and ``out = op(src1, src2)`` is
    :func:`add_weighted` with them. A tensor coefficient is stored as a non-persistent buffer, or as a parameter if it
    is an ``nn.Parameter``.

    Convention:
        - See the convention block of :func:`add_weighted`.

    Args:
        alpha: weight of ``src1``, a number or a tensor.
        beta: weight of ``src2``, a number or a tensor.
        gamma: value added to the sum, a number or a tensor.

    Shape:
        - Input1: tensor of any shape.
        - Input2: tensor with the shape of Input1.
        - Output: same shape as Input1.

    See Also:
        - :func:`add_weighted`: the one-call form.

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
        if isinstance(alpha, torch.Tensor) and not isinstance(alpha, nn.Parameter):
            self.register_buffer("alpha", alpha, persistent=False)
        else:
            self.alpha = alpha
        if isinstance(beta, torch.Tensor) and not isinstance(beta, nn.Parameter):
            self.register_buffer("beta", beta, persistent=False)
        else:
            self.beta = beta
        if isinstance(gamma, torch.Tensor) and not isinstance(gamma, nn.Parameter):
            self.register_buffer("gamma", gamma, persistent=False)
        else:
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

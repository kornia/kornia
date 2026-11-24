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


class Rescale(nn.Module):
    """Multiply the input by a constant factor.

    ``op = Rescale(factor)`` stores ``factor`` and ``out = op(input)`` returns ``input * factor``. A tensor factor is
    stored as a non-persistent buffer, so ``.to()`` moves it, and an ``nn.Parameter`` stays a parameter that optimizers
    and checkpoints include.

    Convention:
        A float factor is applied as a Python number: a float64 input keeps its precision and an integer input promotes
        to the default float dtype.

    Args:
        factor: scaling factor, a ``float`` or a 0-d tensor. An ``int`` raises ``TypeError``.

    Shape:
        - Input: any shape.
        - Output: same shape as the input.

    See Also:
        - :class:`~kornia.geometry.transform.Rescale`: resizes an image by a factor instead of multiplying its values.
        - :func:`adjust_contrast`: multiplies an image by a factor and clamps it to :math:`[0, 1]`.

    Example:
        >>> Rescale(0.5)(torch.ones(1, 1, 2, 2))
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])

    """

    def __init__(self, factor: Union[float, torch.Tensor]) -> None:
        super().__init__()
        if isinstance(factor, float):
            self.factor = factor
        elif isinstance(factor, torch.Tensor) and factor.ndim == 0:
            if isinstance(factor, nn.Parameter):
                self.factor = factor
            else:
                self.register_buffer("factor", factor, persistent=False)
        else:
            raise TypeError(f"Expected factor to be a float or a 0-d torch.Tensor, got {factor}.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Multiply the input tensor by the configured scaling factor.

        Args:
            input: Tensor to be rescaled. In image workflows this is commonly
                shaped as ``(*, C, H, W)``, where ``*`` denotes optional
                leading dimensions (for example batch), ``C`` is channel count,
                and ``H``/``W`` are spatial dimensions.

        Returns:
            A tensor with the same shape as ``input``, where each element is
            multiplied by ``self.factor``.
        """
        # A float factor multiplies as a Python number: float64 keeps its bits, half
        # inputs are scaled in float32 arithmetic, and integer inputs promote to float.
        return input * self.factor

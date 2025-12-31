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

from typing import Any, Union

import torch
from torch import nn

from kornia.utils.image import perform_keep_shape_image


@perform_keep_shape_image
def in_range(
    input: torch.Tensor,
    lower: Union[tuple[Any, ...], torch.Tensor],
    upper: Union[tuple[Any, ...], torch.Tensor],
    return_mask: bool = False,
) -> torch.Tensor:
    r"""Create a mask indicating whether elements of the input torch.tensor are within the specified range.

    .. image:: _static/img/in_range.png

    The formula applied for single-channel torch.tensor is:

    .. math::
        \text{out}(I) = \text{lower}(I) \leq \text{input}(I) \geq \text{upper}(I)

    The formula applied for multi-channel torch.tensor is:

    .. math::
        \text{out}(I) = \bigwedge_{c=0}^{C}
        \left( \text{lower}_c(I) \leq \text{input}_c(I) \geq \text{upper}_c(I) \right)

    torch.where `C` is the number of channels.

    Args:
        input: The input torch.tensor to be filtered in the shape of :math:`(*, *, H, W)`.
        lower: The lower bounds of the filter (inclusive).
        upper: The upper bounds of the filter (inclusive).
        return_mask: If is true, the filtered mask is returned, otherwise the filtered input image.

    Returns:
        A binary mask :math:`(*, 1, H, W)` of input indicating whether elements are within the range
        or filtered input image :math:`(*, *, H, W)`.

    Raises:
        ValueError: If the shape of `lower`, `upper`, and `input` image channels do not match.

    .. note::
        Clarification of `lower` and `upper`:

        - If provided as a tuple, it should have the same number of elements as the channels in the input torch.tensor.
          This bound is then applied uniformly across all batches.

        - When provided as a torch.tensor, it allows for different bounds to be applied to each batch.
          The torch.tensor shape should be (B, C, 1, 1), torch.where B is the batch size and C is
          the number of channels.

        - If the torch.tensor has a 1-D shape, same bound will be applied across all batches.

    Examples:
        >>> rng = torch.manual_seed(1)
        >>> input = torch.rand(1, 3, 3, 3)
        >>> lower = (0.2, 0.3, 0.4)
        >>> upper = (0.8, 0.9, 1.0)
        >>> mask = in_range(input, lower, upper, return_mask=True)
        >>> mask
        tensor([[[[1., 1., 0.],
                  [0., 0., 0.],
                  [0., 1., 1.]]]])
        >>> mask.shape
        torch.Size([1, 1, 3, 3])

    Apply different bounds (`lower` and `upper`) for each batch:

        >>> rng = torch.manual_seed(1)
        >>> input_tensor = torch.rand((2, 3, 3, 3))
        >>> input_shape = input_tensor.shape
        >>> lower = torch.tensor([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]).reshape(input_shape[0], input_shape[1], 1, 1)
        >>> upper = torch.tensor([[0.6, 0.6, 0.6], [0.8, 0.8, 0.8]]).reshape(input_shape[0], input_shape[1], 1, 1)
        >>> mask = in_range(input_tensor, lower, upper, return_mask=True)
        >>> mask
        tensor([[[[0., 0., 1.],
                  [0., 0., 0.],
                  [1., 0., 0.]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0., 0., 0.],
                  [1., 0., 0.],
                  [0., 0., 1.]]]])

    """
    input_shape = input.shape

    if not isinstance(lower, (tuple, torch.Tensor)) or not isinstance(upper, (tuple, torch.Tensor)):
        raise TypeError("Invalid `lower` and `upper` format. Should be tuple or torch.Tensor.")

    if not isinstance(return_mask, bool):
        raise TypeError("Invalid `return_mask` format. Should be boolean.")

    if isinstance(lower, tuple) and isinstance(upper, tuple):
        if len(lower) != input_shape[1] or len(upper) != input_shape[1]:
            raise ValueError("Shape of `lower`, `upper` and `input` image channels must have same shape.")

        lower = (
            torch.tensor(lower, device=input.device, dtype=input.dtype)
            .reshape(1, -1, 1, 1)
            .repeat(input_shape[0], 1, 1, 1)
        )
        upper = (
            torch.tensor(upper, device=input.device, dtype=input.dtype)
            .reshape(1, -1, 1, 1)
            .repeat(input_shape[0], 1, 1, 1)
        )

    elif isinstance(lower, torch.Tensor) and isinstance(upper, torch.Tensor):
        valid_tensor_shape = (input_shape[0], input_shape[1], 1, 1)
        if valid_tensor_shape not in (lower.shape, upper.shape):
            raise ValueError(
                "`lower` and `upper` bounds as Tensors must have compatible shapes with the input (B, C, 1, 1)."
            )
        lower = lower.to(input)
        upper = upper.to(input)

    # Apply lower and upper bounds. Combine masks with logical_and.
    mask = torch.logical_and(input >= lower, input <= upper)
    mask = mask.all(dim=(1), keepdim=True).to(input.dtype)

    if return_mask:
        return mask

    return input * mask


class InRange(nn.Module):
    r"""Create a module for applying lower and upper bounds to input tensors.

    Args:
        input: The input torch.tensor to be filtered.
        lower: The lower bounds of the filter (inclusive).
        upper: The upper bounds of the filter (inclusive).
        return_mask: If is true, the filtered mask is returned, otherwise the filtered input image.

    Returns:
        A binary mask :math:`(*, 1, H, W)` of input indicating whether elements are within the range
        or filtered input image :math:`(*, *, H, W)`.

    .. note::
        View complete documentation in :func:`kornia.filters.in_range`.

    Examples:
        >>> rng = torch.manual_seed(1)
        >>> input = torch.rand(1, 3, 3, 3)
        >>> lower = (0.2, 0.3, 0.4)
        >>> upper = (0.8, 0.9, 1.0)
        >>> mask = InRange(lower, upper, return_mask=True)(input)
        >>> mask
        tensor([[[[1., 1., 0.],
                  [0., 0., 0.],
                  [0., 1., 1.]]]])

    """

    def __init__(
        self,
        lower: Union[tuple[Any, ...], torch.Tensor],
        upper: Union[tuple[Any, ...], torch.Tensor],
        return_mask: bool = False,
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.return_mask = return_mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return in_range(input, self.lower, self.upper, self.return_mask)

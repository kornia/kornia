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

from kornia.image.utils import perform_keep_shape_image


def _as_bchw_bound(name: str, bound: torch.Tensor, input_shape: torch.Size) -> torch.Tensor:
    """Validate a Tensor bound against the ``(B, C, H, W)`` input and return it as a 4-D Tensor.

    A 1-D bound is read per channel: it must have ``C`` elements (or one) and becomes ``(1, C, 1, 1)``. A bound with
    zero to four other dimensions is aligned with the input from the last dimension, and every dimension must be 1 or
    the input's own size.
    """
    if bound.dim() == 1:
        if bound.shape[0] in (1, input_shape[1]):
            return bound.reshape(1, -1, 1, 1)
    elif bound.dim() <= 4:
        shape = (1,) * (4 - bound.dim()) + tuple(bound.shape)
        if all(b in (1, i) for b, i in zip(shape, input_shape)):
            return bound.reshape(shape)
    raise ValueError(
        f"`{name}` as a Tensor must be 0-d, have shape (C,) or (1,), or have two to four dimensions that, aligned "
        f"with the input from the last dimension, are each 1 or equal to the input's. "
        f"Got {tuple(bound.shape)} for an input viewed as (B, C, H, W) = {tuple(input_shape)}."
    )


@perform_keep_shape_image
def in_range(
    input: torch.Tensor,
    lower: Union[tuple[Any, ...], torch.Tensor],
    upper: Union[tuple[Any, ...], torch.Tensor],
    return_mask: bool = False,
) -> torch.Tensor:
    r"""Create a mask indicating whether elements of the input torch.Tensor are within the specified range.

    .. image:: _static/img/in_range.png

    The formula applied for single-channel torch.Tensor is:

    .. math::
        \text{out}(I) = \text{lower}(I) \leq \text{input}(I) \leq \text{upper}(I)

    The formula applied for multi-channel torch.Tensor is:

    .. math::
        \text{out}(I) = \bigwedge_{c=0}^{C-1}
        \left( \text{lower}_c(I) \leq \text{input}_c(I) \leq \text{upper}_c(I) \right)

    where `C` is the number of channels. Both comparisons are inclusive.

    Convention:
        - A NaN channel fails its pixel. The mask has the input's dtype, 1 for a pass. ``lower > upper`` is not
          rejected and selects nothing.
        - ``return_mask=False`` returns ``input * mask``: the channels of a failing pixel become 0, except a NaN or
          infinite one, which becomes NaN.
        - The bounds, laid out as the note below says, are cast to the input's dtype.
        - Known defect: on an integer image a fractional bound truncates toward zero, so a lower bound of ``100.7``
          admits ``100`` (`#5423 <https://github.com/kornia/kornia/issues/5423>`_).

    Args:
        input: The input torch.Tensor to be filtered in the shape of :math:`(*, *, H, W)`.
        lower: The lower bounds of the filter (inclusive).
        upper: The upper bounds of the filter (inclusive).
        return_mask: If is true, the filtered mask is returned, otherwise the filtered input image.

    Returns:
        A binary mask :math:`(*, 1, H, W)` of input indicating whether elements are within the range
        or filtered input image :math:`(*, *, H, W)`.

    Raises:
        TypeError: If `lower` or `upper` is neither a tuple nor a torch.Tensor, if one is a tuple and the other a
            torch.Tensor, or if `return_mask` is not a bool.
        ValueError: If a tuple bound does not have one element per channel, or a torch.Tensor bound does not fit the
            input as described in the note.

    .. note::
        Clarification of `lower` and `upper`:

        - If provided as a tuple, it should have the same number of elements as the channels in the input torch.Tensor.
          This bound is then applied uniformly across all batches.

        - When provided as a torch.Tensor, it allows for different bounds to be applied to each batch.
          The torch.Tensor shape should be (B, C, 1, 1), where B is the batch size and C is
          the number of channels.

        - A 1-D torch.Tensor is read per channel, never per column: its shape is (C,) (or one element, for the
          same bound in every channel) and the same bound is applied across all batches. A 1-D bound of any other
          length, such as (W,) when W is neither C nor 1, raises. When W == C, a (W,) bound is the (C,) bound, not a
          per-column one.

        - Any other torch.Tensor with at most four dimensions is aligned with the input :math:`(B, C, H, W)` from the
          last dimension, and every dimension must be 1 or the input's. For example (1, C, 1, 1) and (C, 1, 1) apply
          the same per-channel bound to all batches, (B, C, H, W), (C, H, W) and (H, W) give a different bound at each
          pixel, and a 0-d torch.Tensor applies one bound everywhere.

        - Each torch.Tensor bound is checked on its own, so a mis-shaped one raises whatever the other bound is.
          The input is read as :math:`(B, C, H, W)`: a 3-D input is :math:`(1, C, H, W)`, a 2-D input is
          :math:`(1, 1, H, W)`, and for more than four dimensions the leading dimensions are flattened into B.

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
        lower = _as_bchw_bound("lower", lower, input_shape).to(input)
        upper = _as_bchw_bound("upper", upper, input_shape).to(input)

    else:
        raise TypeError("Invalid `lower` and `upper` format. Both should be tuples or both torch.Tensor.")

    # Apply lower and upper bounds. Combine masks with logical_and.
    mask = torch.logical_and(input >= lower, input <= upper)
    mask = mask.all(dim=(1), keepdim=True).to(input.dtype)

    if return_mask:
        return mask

    return input * mask


class InRange(nn.Module):
    r"""Create a module for applying lower and upper bounds to input tensors.

    Convention:
        See the Convention block on :func:`~kornia.filters.in_range`.

    Args:
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
        """Select values that fall inside the configured inclusive range.

        The stored ``lower`` and ``upper`` bounds are compared against
        ``input`` channel-wise. Depending on ``self.return_mask``, the module
        either returns the binary in-range mask itself or uses that mask to keep
        only values that satisfy the bounds.

        Args:
            input: Tensor to test against the configured bounds. For images the
                usual shape is :math:`(B, C, H, W)`, where :math:`B` is the
                batch size, :math:`C` is the number of channels, :math:`H` is
                the height, and :math:`W` is the width.

        Returns:
            If ``self.return_mask`` is ``True``, a :math:`(B, 1, H, W)` mask of
            the pixels whose every channel lies within the inclusive range.
            Otherwise, a tensor with values outside the range removed according
            to :func:`in_range`.
        """
        return in_range(input, self.lower, self.upper, self.return_mask)

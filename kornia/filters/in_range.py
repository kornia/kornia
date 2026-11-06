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


def _tuple_bound(bound: tuple[Any, ...], input: torch.Tensor) -> torch.Tensor:
    if input.is_floating_point() or input.is_complex():
        return torch.tensor(bound, device=input.device, dtype=input.dtype)
    # Preserve Python float precision and exact integer bounds. Normalize on CPU before moving to devices
    # such as MPS, which do not support float64.
    dtype = torch.int64 if all(isinstance(value, int) for value in bound) else torch.float64
    return torch.tensor(bound, dtype=dtype)


def _compare_bound(input: torch.Tensor, bound: torch.Tensor, lower: bool) -> torch.Tensor:
    if input.is_floating_point() or input.is_complex():
        bound = bound.to(input)
    elif bound.is_floating_point():
        # Round the bound, not the image: promoting large integer pixels to float can change the comparison.
        bound = bound.ceil() if lower else bound.floor()
        minimum = 0 if input.dtype == torch.bool else torch.iinfo(input.dtype).min
        maximum = 1 if input.dtype == torch.bool else torch.iinfo(input.dtype).max
        below = bound < minimum
        above = bound >= maximum + 1
        # max + 1 is an exact power of two even when max itself rounds up in the bound's floating dtype.
        safe_bound = torch.where((bound >= minimum) & (bound < maximum + 1), bound, 0).to(input)
        below, above = below.to(input.device), above.to(input.device)
        if lower:
            return ((input >= safe_bound) | below) & ~above & ~bound.isnan().to(input.device)
        return ((input <= safe_bound) | above) & ~below & ~bound.isnan().to(input.device)
    else:
        bound = bound.to(device=input.device)
    return input >= bound if lower else input <= bound


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
    r"""Keep the pixels whose channels all lie in an inclusive range.

    ``out = in_range(input, lower, upper)`` multiplies ``input`` by a mask that is 1 where every channel lies between
    its ``lower`` and ``upper`` bounds, and 0 elsewhere. With ``return_mask=True`` it returns the mask instead.

    .. image:: _static/img/in_range.png

    For an input with ``C`` channels the mask at pixel :math:`I` is:

    .. math::
        \text{mask}(I) = \bigwedge_{c=0}^{C-1}
        \left( \text{lower}_c(I) \leq \text{input}_c(I) \leq \text{upper}_c(I) \right)

    Convention:
        - A NaN channel fails its pixel. The mask has the input's dtype, 1 for a pass. For floating input, bounds
          are cast to the input dtype before comparison, so only an interval still reversed after that conversion
          selects nothing.
        - ``return_mask=False`` returns ``input * mask``: the channels of a failing pixel become 0, except a NaN or
          infinite one, which becomes NaN.
        - For integer inputs, fractional lower bounds round up and fractional upper bounds round down. Bounds outside
          the input dtype's range do not wrap or overflow; a range containing no representable input value produces
          an empty mask.

    Args:
        input: image with shape ``(*, C, H, W)``.
        lower: inclusive lower bounds, a tuple with one entry per channel or a tensor; see the note.
        upper: inclusive upper bounds, in the same form as ``lower``.
        return_mask: if ``True``, return the mask; otherwise return the masked image.

    Returns:
        the mask with shape ``(*, 1, H, W)`` if ``return_mask`` is ``True``, otherwise the masked image with the
        shape of ``input``.

    Raises:
        TypeError: If `lower` or `upper` is neither a tuple nor a torch.Tensor, if one is a tuple and the other a
            torch.Tensor, or if `return_mask` is not a bool.
        ValueError: If a tuple bound does not have one element per channel, or a torch.Tensor bound does not fit the
            input as described in the note.

    .. note::
        The forms of ``lower`` and ``upper``:

        - A tuple has one entry per channel and applies to every sample.

        - A tensor of shape ``(B, C, 1, 1)`` gives each sample its own bounds.

        - A 1-D tensor is read per channel, never per column: its shape is ``(C,)``, or one element for the same
          bound in every channel, and it applies to every sample. Any other 1-D length, such as ``(W,)`` when ``W``
          is neither ``C`` nor 1, raises. When ``W == C``, a ``(W,)`` bound is the ``(C,)`` bound, not a
          per-column one.

        - Any other tensor with at most four dimensions is aligned with the input ``(B, C, H, W)`` from the last
          dimension, and every dimension must be 1 or the input's. For example ``(1, C, 1, 1)`` and ``(C, 1, 1)``
          apply the same per-channel bound to all samples, ``(B, C, H, W)``, ``(C, H, W)`` and ``(H, W)`` give a
          different bound at each pixel, and a 0-d tensor applies one bound everywhere.

        - Each tensor bound is checked on its own, so a mis-shaped one raises whatever the other bound is.
          The input is read as ``(B, C, H, W)``: a 3-D input is ``(1, C, H, W)``, a 2-D input is ``(1, 1, H, W)``
          and gives a 2-D output, and for more than four dimensions the leading dimensions are flattened into ``B``.

    See Also:
        - :class:`InRange`: the module form.
        - :func:`otsu_threshold`: picks the threshold from the image histogram instead of taking bounds.

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

        lower = _tuple_bound(lower, input).reshape(1, -1, 1, 1).repeat(input_shape[0], 1, 1, 1)
        upper = _tuple_bound(upper, input).reshape(1, -1, 1, 1).repeat(input_shape[0], 1, 1, 1)

    elif isinstance(lower, torch.Tensor) and isinstance(upper, torch.Tensor):
        lower = _as_bchw_bound("lower", lower, input_shape)
        upper = _as_bchw_bound("upper", upper, input_shape)

    else:
        raise TypeError("Invalid `lower` and `upper` format. Both should be tuples or both torch.Tensor.")

    # Apply lower and upper bounds. Combine masks with logical_and.
    mask = torch.logical_and(_compare_bound(input, lower, lower=True), _compare_bound(input, upper, lower=False))
    mask = mask.all(dim=(1), keepdim=True).to(input.dtype)

    if return_mask:
        return mask

    return input * mask


class InRange(nn.Module):
    r"""Keep the pixels whose channels all lie in stored inclusive bounds (module form).

    ``keep = InRange(lower, upper, return_mask)`` stores its arguments, and ``out = keep(input)`` is :func:`in_range`
    on an image with shape ``(*, C, H, W)``.

    Convention:
        See the Convention block on :func:`~kornia.filters.in_range`.

    Args:
        lower: inclusive lower bounds, as in :func:`in_range`.
        upper: inclusive upper bounds, as in :func:`in_range`.
        return_mask: if ``True``, return the mask; otherwise return the masked image.

    Shape:
        - Input: ``(*, C, H, W)``
        - Output: ``(*, 1, H, W)`` with ``return_mask=True``, otherwise the shape of the input.

    See Also:
        - :func:`in_range`: the one-call form, with the forms the bounds can take.
        - :class:`OtsuThreshold`: picks the threshold from the image histogram.

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
                usual shape is ``(B, C, H, W)``, where :math:`B` is the
                batch size, :math:`C` is the number of channels, :math:`H` is
                the height, and :math:`W` is the width.

        Returns:
            If ``self.return_mask`` is ``True``, a ``(B, 1, H, W)`` mask of
            the pixels whose every channel lies within the inclusive range.
            Otherwise, a tensor with values outside the range removed according
            to :func:`in_range`.
        """
        return in_range(input, self.lower, self.upper, self.return_mask)

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

from enum import IntEnum
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR


class ThresholdType(IntEnum):
    """Select how :func:`threshold` maps each element, with OpenCV's values for the fixed modes.

    Pass a member, or its integer value, as ``type``. Writing ``x`` for an element, ``t`` for ``thresh`` and ``m`` for
    ``maxval``:

    - ``THRESH_BINARY`` (0): ``m`` where ``x > t``, else ``0``.
    - ``THRESH_BINARY_INV`` (1): ``0`` where ``x > t``, else ``m``.
    - ``THRESH_TRUNC`` (2): ``t`` where ``x > t``, else ``x``.
    - ``THRESH_TOZERO`` (3): ``x`` where ``x > t``, else ``0``.
    - ``THRESH_TOZERO_INV`` (4): ``0`` where ``x > t``, else ``x``.
    - ``THRESH_OTSU`` (8): reserved; :func:`threshold` raises ``NotImplementedError`` for it, also when it is OR-ed into
      another mode.

    See Also:
        - :func:`threshold`: applies a mode.
        - :class:`Threshold`: the module form.

    """

    THRESH_BINARY = 0
    THRESH_BINARY_INV = 1
    THRESH_TRUNC = 2
    THRESH_TOZERO = 3
    THRESH_TOZERO_INV = 4

    # OpenCV uses 8 for OTSU, reserved for follow-up PR.
    THRESH_OTSU = 8


# The fixed types ``threshold`` implements. Built once here: iterating the enum class inside ``threshold`` breaks
# ``torch.compile(fullgraph=True)`` on torch 2.9 (Dynamo cannot trace ``EnumMeta.__iter__`` there).
_FIXED_TYPES = frozenset(int(x) for x in ThresholdType if x != ThresholdType.THRESH_OTSU)


def _integer_threshold(input: Tensor, thresh: Union[float, Tensor]) -> tuple[Tensor, Tensor]:
    """Return ``input > thresh`` and the value THRESH_TRUNC writes, for an integer ``input``.

    Casting ``thresh`` to an integer dtype truncates a fraction toward zero and wraps or rejects a value outside the
    dtype's range, so the comparison would use another threshold. An integer is greater than ``thresh`` exactly when
    it is greater than ``floor(thresh)``, which OpenCV also compares with. A threshold below the dtype's range passes
    every element, one at or above its maximum passes none, and ``nan`` passes none, as for a floating-point input.
    """
    low, high = (0, 1) if input.dtype == torch.bool else (torch.iinfo(input.dtype).min, torch.iinfo(input.dtype).max)
    if not isinstance(thresh, Tensor):
        # float64 and int64 hold any Python number a pixel can be compared with. The tensor stays on the CPU, as MPS
        # has no float64; only the results below move to the input's device.
        thresh = torch.tensor(thresh, dtype=torch.int64 if isinstance(thresh, int) else torch.float64)
    if thresh.is_floating_point():
        thresh = thresh.floor()
    below = thresh < low
    inside = (thresh >= low) & (thresh < high)
    safe = torch.where(inside, thresh, 0).to(input.dtype).to(input.device)
    below, inside = below.to(input.device), inside.to(input.device)
    mask = ((input > safe) & inside) | below
    # Where an element passes, the threshold is in range, or below it and the element is clamped to the minimum.
    return mask, torch.where(below, torch.full_like(safe, low), safe)


def threshold(
    input: Tensor,
    thresh: Union[float, Tensor],
    maxval: Union[float, Tensor] = 255.0,
    type: Union[int, ThresholdType] = ThresholdType.THRESH_BINARY,
) -> Tensor:
    """Threshold every element at a fixed level, as ``cv2.threshold`` does.

    ``out = threshold(input, thresh, maxval, type)`` compares each element with ``thresh`` and maps it to ``maxval``,
    ``0``, ``thresh`` or itself, as the :class:`ThresholdType` ``type`` says. It uses OpenCV's mode values, takes
    tensors for ``thresh`` and ``maxval``, and raises for ``THRESH_OTSU``.

    Convention:
        - The comparison is strict: an element passes when ``input > thresh``.
        - ``thresh`` and ``maxval`` broadcast over ``input`` and move to its device. ``maxval`` is converted to the
          dtype of ``input``, and so is ``thresh`` for a floating-point input.
        - On an integer input ``thresh`` is not cast: as in OpenCV, an element passes when it is greater than
          ``floor(thresh)``. A ``thresh`` below the dtype's range passes every element and one at or above its maximum
          passes none. ``THRESH_TRUNC`` then writes ``floor(thresh)``, or the dtype's minimum for a ``thresh`` below its
          range.

    Args:
        input: tensor with shape ``(*, H, W)``.
        thresh: level, a number or a tensor that broadcasts to ``input``.
        maxval: value written by the binary modes, a number or a tensor. The default ``255.0`` turns a :math:`[0, 1]`
            image into 0 and 255; pass ``1.0`` to keep its range.
        type: a :class:`ThresholdType` member or its value.

    Returns:
        Thresholded tensor with the shape, dtype and device of ``input``.

    Raises:
        NotImplementedError: if ``THRESH_OTSU`` is passed.
        ValueError: if ``type`` is not a supported mode.

    See Also:
        - :class:`Threshold`: the module form.
        - :func:`~kornia.filters.otsu_threshold`: finds the level itself with Otsu's method.

    Example:
        >>> x = torch.tensor([[0.2, 0.8]])
        >>> threshold(x, 0.5, 1.0)
        tensor([[0., 1.]])
        >>> threshold(x, 0.5, 1.0, ThresholdType.THRESH_TRUNC)
        tensor([[0.2000, 0.5000]])

    """
    KORNIA_CHECK_IS_TENSOR(input)

    t = int(type)

    # Detect if OTSU flag is present (opencv allows OR-ing)
    if t & int(ThresholdType.THRESH_OTSU):
        raise NotImplementedError("THRESH_OTSU is not implemented yet. Please use a fixed threshold type.")

    KORNIA_CHECK(
        t in _FIXED_TYPES,
        lambda: f"Unsupported threshold type: {type}. Supported: BINARY, BINARY_INV, TRUNC, TOZERO, TOZERO_INV.",
    )

    # Make thresh/maxval tensors on same device/dtype for safe broadcasting
    if input.is_floating_point() or input.is_complex():
        thresh_t = thresh
        if not isinstance(thresh_t, Tensor):
            thresh_t = torch.tensor(thresh_t, device=input.device, dtype=input.dtype)
        else:
            thresh_t = thresh_t.to(device=input.device, dtype=input.dtype)
        mask = input > thresh_t
    else:
        mask, thresh_t = _integer_threshold(input, thresh)

    maxval_t = maxval
    if not isinstance(maxval_t, Tensor):
        maxval_t = torch.tensor(maxval_t, device=input.device, dtype=input.dtype)
    else:
        maxval_t = maxval_t.to(device=input.device, dtype=input.dtype)

    zeros = torch.zeros_like(input)

    if t == int(ThresholdType.THRESH_BINARY):
        return torch.where(mask, maxval_t, zeros)

    if t == int(ThresholdType.THRESH_BINARY_INV):
        return torch.where(mask, zeros, maxval_t)

    if t == int(ThresholdType.THRESH_TRUNC):
        if input.is_floating_point() or input.is_complex():
            return torch.minimum(input, thresh_t)
        return torch.where(mask, thresh_t, input)

    if t == int(ThresholdType.THRESH_TOZERO):
        return torch.where(mask, input, zeros)

    if t == int(ThresholdType.THRESH_TOZERO_INV):
        return torch.where(mask, zeros, input)

    # Should never reach here due to KORNIA_CHECK above
    raise ValueError(f"Unsupported threshold type: {type}")


class Threshold(Module):
    """Threshold every element at a fixed level (module form).

    ``op = Threshold(thresh, maxval, type)`` stores the three values as plain numbers and ``out = op(input)`` is
    :func:`threshold` with them.

    Convention:
        - See the convention block of :func:`threshold`; ``thresh`` and ``maxval`` are numbers here, not tensors.

    Args:
        thresh: level, a number.
        maxval: value written by the binary modes, a number. The default ``255.0`` turns a :math:`[0, 1]` image into 0
            and 255; pass ``1.0`` to keep its range.
        type: a :class:`ThresholdType` member or its value.

    See Also:
        - :func:`threshold`: the one-call form.

    Example:
        >>> Threshold(0.5, 1.0)(torch.tensor([[0.2, 0.8]]))
        tensor([[0., 1.]])

    """

    def __init__(
        self,
        thresh: float,
        maxval: float = 255.0,
        type: Union[int, ThresholdType] = ThresholdType.THRESH_BINARY,
    ) -> None:
        super().__init__()
        self.thresh = float(thresh)
        self.maxval = float(maxval)
        self.type = int(type)

    def forward(self, input: Tensor) -> Tensor:
        """Apply thresholding with this module's configured parameters.

        This method delegates to :func:`threshold` and reuses the threshold
        value, maximum replacement value, and thresholding mode stored on the
        module instance.

        Args:
            input: Input tensor to threshold. Typical image inputs are shaped
                ``(*, C, H, W)``, but any tensor shape supported by
                :func:`threshold` is accepted.

        Returns:
            A tensor with the same shape as ``input`` where values are remapped
            according to ``self.type`` using ``self.thresh`` and ``self.maxval``.
        """
        return threshold(input, self.thresh, self.maxval, self.type)

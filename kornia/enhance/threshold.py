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
    """Threshold types compatible with OpenCV fixed thresholding types.

    Note: THRESH_OTSU is intentionally not supported in this PR.
    """

    THRESH_BINARY = 0
    THRESH_BINARY_INV = 1
    THRESH_TRUNC = 2
    THRESH_TOZERO = 3
    THRESH_TOZERO_INV = 4

    # OpenCV uses 8 for OTSU, reserved for follow-up PR.
    THRESH_OTSU = 8


def threshold(
    input: Tensor,
    thresh: Union[float, Tensor],
    maxval: Union[float, Tensor] = 255.0,
    type: Union[int, ThresholdType] = ThresholdType.THRESH_BINARY,
) -> Tensor:
    """Apply a fixed-level threshold to each element in the input tensor.

    Implements OpenCV-like behavior for the following threshold types:
    - THRESH_BINARY
    - THRESH_BINARY_INV
    - THRESH_TRUNC
    - THRESH_TOZERO
    - THRESH_TOZERO_INV

    Args:
        input: Image tensor of shape (..., H, W). Typically (B, C, H, W).
        thresh: Threshold value (scalar or tensor broadcastable to input).
        maxval: Maximum value used with binary thresholding types.
        type: Threshold type.

    Returns:
        Thresholded tensor with same shape/dtype/device as `input`.

    Raises:
        NotImplementedError: if THRESH_OTSU flag is passed.
        ValueError: if type is not supported.
    """
    KORNIA_CHECK_IS_TENSOR(input)

    t = int(type)

    # Detect if OTSU flag is present (opencv allows OR-ing)
    if t & int(ThresholdType.THRESH_OTSU):
        raise NotImplementedError("THRESH_OTSU is not implemented yet. Please use a fixed threshold type.")

    KORNIA_CHECK(
        t in {int(x) for x in ThresholdType if x != ThresholdType.THRESH_OTSU},
        f"Unsupported threshold type: {type}. Supported: BINARY, BINARY_INV, TRUNC, TOZERO, TOZERO_INV.",
    )

    # Make thresh/maxval tensors on same device/dtype for safe broadcasting
    thresh_t = thresh
    if not isinstance(thresh_t, Tensor):
        thresh_t = torch.tensor(thresh_t, device=input.device, dtype=input.dtype)
    else:
        thresh_t = thresh_t.to(device=input.device, dtype=input.dtype)

    maxval_t = maxval
    if not isinstance(maxval_t, Tensor):
        maxval_t = torch.tensor(maxval_t, device=input.device, dtype=input.dtype)
    else:
        maxval_t = maxval_t.to(device=input.device, dtype=input.dtype)

    mask = input > thresh_t
    zeros = torch.zeros_like(input)

    if t == int(ThresholdType.THRESH_BINARY):
        return torch.where(mask, maxval_t, zeros)

    if t == int(ThresholdType.THRESH_BINARY_INV):
        return torch.where(mask, zeros, maxval_t)

    if t == int(ThresholdType.THRESH_TRUNC):
        return torch.minimum(input, thresh_t)

    if t == int(ThresholdType.THRESH_TOZERO):
        return torch.where(mask, input, zeros)

    if t == int(ThresholdType.THRESH_TOZERO_INV):
        return torch.where(mask, zeros, input)

    # Should never reach here due to KORNIA_CHECK above
    raise ValueError(f"Unsupported threshold type: {type}")


class Threshold(Module):
    """Module wrapper for `kornia.enhance.threshold`."""

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
        return threshold(input, self.thresh, self.maxval, self.type)

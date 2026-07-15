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

from typing import Optional

import torch


def mask_ignore_pixels(
    target: torch.Tensor, ignore_index: Optional[int]
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Replace ignored target labels with a valid class and return their mask.

    Loss functions often need class indices to stay inside the valid class
    range before one-hot encoding or gathering. This helper keeps valid labels
    unchanged, maps ignored positions to class ``0`` temporarily, and returns a
    mask so callers can remove those positions from the final loss.

    Args:
        target: Target label tensor with arbitrary shape.
        ignore_index: Label value that should be ignored. If ``None``, no
            masking is applied.

    Returns:
        Tuple ``(target, target_mask)``. ``target`` is either the original
        tensor or a copy where ignored labels were replaced by zero.
        ``target_mask`` is ``None`` when no ignored pixels are present;
        otherwise it is a boolean tensor with ``True`` at valid positions.
    """
    if ignore_index is None:
        return target, None

    target_mask = target != ignore_index

    # Data-dependent early-out (skip masking when nothing is ignored). Guard it under
    # torch.compile: returning the all-True mask instead of None just makes callers apply
    # a no-op mask, which keeps the result identical while staying fullgraph-safe.
    if not torch.compiler.is_compiling() and target_mask.all():
        return target, None

    # map invalid pixels to a valid class (0)
    # they need to be manually excluded from the loss computation after
    target = target.where(target_mask, target.new_zeros(1))

    return target, target_mask

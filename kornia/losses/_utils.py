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
    if ignore_index is None:
        return target, None

    target_mask = target != ignore_index

    if target_mask.all():
        return target, None

    # map invalid pixels to a valid class (0)
    # they need to be manually excluded from the loss computation after
    target = target.where(target_mask, target.new_zeros(1))

    return target, target_mask

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

import torch

from kornia.core import Tensor, tensor


def normalize_keypoints(kpts: Tensor, size: Tensor) -> Tensor:
    """Normalize tensor of keypoints."""
    if isinstance(size, torch.Size):
        size = tensor(size)[None]
    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) / 2
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts

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

from typing import Any

import torch

from kornia.core import Tensor


def randperm(n: int, ensure_perm: bool = True, **kwargs: Any) -> Tensor:
    """`randomperm` with the ability to ensure the different arrangement generated."""
    perm = torch.randperm(n, **kwargs)
    if ensure_perm:
        while torch.all(torch.eq(perm, torch.arange(n, device=perm.device))):
            perm = torch.randperm(n, **kwargs)
    return perm

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

from typing import Any, Optional

import torch


def randperm(n: int, ensure_perm: bool = True, identity: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    """`randomperm` with the ability to ensure the different arrangement generated.

    ``identity`` is the permutation to reject when ``ensure_perm`` is set; it defaults to ``arange(n)``.
    """
    perm = torch.randperm(n, **kwargs)
    if ensure_perm:
        if identity is None:
            identity = torch.arange(n, device=perm.device)
        while torch.equal(perm, identity):
            perm = torch.randperm(n, **kwargs)
    return perm

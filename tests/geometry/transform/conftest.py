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

from functools import cache

import torch
import torch.nn.functional as F

_UNSUPPORTED_BORDER_PADDING_MSG = "Unsupported Border padding mode"


@cache
def _supports_2d_border_padding_probe(device_type: str) -> bool:
    inp = torch.zeros(1, 1, 2, 2, device=device_type)
    grid = torch.zeros(1, 1, 1, 2, device=device_type)
    try:
        F.grid_sample(inp, grid, padding_mode="border", align_corners=True)
    except RuntimeError as e:
        if _UNSUPPORTED_BORDER_PADDING_MSG in str(e):
            return False
        raise
    return True


def supports_2d_border_padding(device: torch.device) -> bool:
    """Whether this device's 2D ``grid_sample`` supports ``padding_mode='border'``.

    Probed at runtime (and cached per device type): MPS's 2D ``grid_sample`` currently
    raises "Unsupported Border padding mode"; its 3D ``grid_sample`` does support it, so
    this guard is only needed for 2D call sites. Probing (rather than hardcoding on
    ``device.type == "mps"``) means this auto-enables once PyTorch adds support.
    """
    return _supports_2d_border_padding_probe(device.type)

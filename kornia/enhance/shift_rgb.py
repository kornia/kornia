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


import torch

from kornia.core.check import KORNIA_CHECK_IS_COLOR, KORNIA_CHECK_IS_TENSOR


def shift_rgb(image: torch.Tensor, r_shift: torch.Tensor, g_shift: torch.Tensor, b_shift: torch.Tensor) -> torch.Tensor:
    """Shift rgb channels.

    Shift each image's channel by either r_shift for red, g_shift for green and b_shift for blue channels.
    """
    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK_IS_COLOR(image, f"with shape {image.shape}")

    shifts = [r_shift, g_shift, b_shift]

    shifted = (image + torch.stack(shifts, dim=1).view(-1, 3, 1, 1).to(image)).clamp_(min=0, max=1)

    return shifted

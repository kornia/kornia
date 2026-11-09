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

from kornia.core.check import KORNIA_CHECK_IS_COLOR, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.core.utils import _to_device_nonblocking
from kornia.enhance.adjust import _check_floating_image, _clip_unit_fresh


def shift_rgb(image: torch.Tensor, r_shift: torch.Tensor, g_shift: torch.Tensor, b_shift: torch.Tensor) -> torch.Tensor:
    """Add a shift to each color channel of an RGB image.

    ``out = shift_rgb(image, r_shift, g_shift, b_shift)`` adds the red, green and blue shifts to the channels of an RGB
    image ``(B, 3, H, W)`` in :math:`[0, 1]` and clamps the result to :math:`[0, 1]`.

    Convention:
        Ranks below three or a channel count other than three raise ``ImageError``. Other ranks that pass the
        color check but are not four raise ``ShapeError``. The three shifts must share the same shape: all
        ``(B,)`` for per-image shifts or all ``(1,)`` for a shared shift.

    Args:
        image: RGB image with shape ``(B, 3, H, W)``.
        r_shift: red shift with shape ``(B,)``, or ``(1,)`` for all images.
        g_shift: green shift with shape ``(B,)``, or ``(1,)`` for all images.
        b_shift: blue shift with shape ``(B,)``, or ``(1,)`` for all images.

    Returns:
        Shifted image with shape ``(B, 3, H, W)``.

    See Also:
        - :func:`adjust_brightness`: adds one factor to every channel.

    Example:
        >>> image = torch.zeros(1, 3, 2, 2)
        >>> shift_rgb(image, torch.tensor([0.2]), torch.tensor([0.0]), torch.tensor([-0.1]))[0, :, 0, 0]
        tensor([0.2000, 0.0000, 0.0000])

    """
    _check_floating_image(image, "shift_rgb")
    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK_IS_COLOR(image, f"with shape {image.shape}")
    KORNIA_CHECK_SHAPE(image, ["B", "3", "H", "W"])

    shifts = [r_shift, g_shift, b_shift]

    shift = _to_device_nonblocking(torch.stack(shifts, dim=1).view(-1, 3, 1, 1), image.device, image.dtype)
    shifted = image + shift
    # An integer image reaches here only with checks disabled, and integer bounds keep its dtype.
    return _clip_unit_fresh(shifted) if shifted.is_floating_point() else shifted.clamp_(min=0, max=1)

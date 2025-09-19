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

"""Convert an image tensor to an ANSI text string (xterm-256color).

Nice long listing of all 256 colors and their codes.

Taken from https://gist.github.com/klange/1687427
"""

from typing import Tuple, Union

import torch

import kornia
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_IS_IMAGE, KORNIA_CHECK_SHAPE
from kornia.io import ImageLoadType

LEVELS = torch.tensor([0x00, 0x5F, 0x87, 0xAF, 0xD7, 0xFF], dtype=torch.int16)


def rgb2short_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Optimized version to quantize RGB to the nearest xterm-256 color cube ID."""
    device = rgb.device
    levels = LEVELS.to(device)

    # For each channel, find the nearest level in a vectorized way
    idx = torch.bucketize(rgb, levels, right=True) - 1
    idx = idx.clamp(0, len(levels) - 2)
    s = levels[idx]
    b = levels[idx + 1]
    choose_bigger = (rgb - s).abs() >= (rgb - b).abs()
    nearest = torch.where(choose_bigger, b, s)

    indices = torch.searchsorted(levels, nearest)
    r_idx, g_idx, b_idx = indices.unbind(-1)

    return (16 + 36 * r_idx + 6 * g_idx + b_idx).to(torch.int16)


def image_to_string(image: torch.Tensor, max_width: int = 256) -> str:
    """Obtain the closest xterm-256 approximation string from an image tensor.

    The tensor shall be either 0~1 float type or 0~255 long type.

    Args:
        image: an RGB image with shape :math:`3HW`.
        max_width: maximum width of the input image.
    """
    KORNIA_CHECK_IS_IMAGE(image, None, raises=True)
    KORNIA_CHECK_SHAPE(image, ["C", "H", "W"])

    if image.dtype.is_floating_point:
        image_float = image.clamp(0.0, 1.0)
    else:
        image_float = image.float() / 255.0

    if image_float.shape[-1] > max_width:
        new_h = image_float.size(-2) * max_width // image_float.size(-1)
        image_float = torch.nn.functional.interpolate(
            image_float.unsqueeze(0), size=(new_h, max_width), mode="bilinear", align_corners=False
        ).squeeze(0)

    image_int = (image_float * 255.0).round().to(torch.int16)
    H, W = image_int.shape[-2:]

    flat = image_int.permute(1, 2, 0).contiguous().reshape(-1, 3)
    short_ids = _rgb2short_helper(flat).reshape(H, W).cpu()

    lines = ["".join([f"\033[48;5;{s.item()}m  " for s in row]) + "\033[0m" for row in short_ids]
    final_string = "\n".join(lines) + "\n"

    return final_string


def print_image(image: Union[str, Tensor], max_width: int = 96) -> None:
    """Print an image to the terminal.

    .. image:: https://github.com/kornia/data/blob/main/print_image.png?raw=true

    Args:
        image: path to a valid image file or a tensor.
        max_width: maximum width to print to terminal.

    Note:
        Need to use `print_image(...)`.

    """
    if isinstance(image, str):
        img = kornia.io.load_image(image, ImageLoadType.RGB8)
    elif isinstance(image, Tensor):
        img = image
    else:
        raise RuntimeError(f"Expect image type to be either Tensor or str. Got {type(image)}.")
    print(image_to_string(img, max_width))

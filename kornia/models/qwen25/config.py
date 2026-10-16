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

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Qwen25VisionConfig:
    """Configuration for the Qwen2.5-VL vision tower.

    The defaults are the ``vision_config`` of the released
    `Qwen/Qwen2.5-VL-3B-Instruct <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json>`_
    checkpoint. The 7B and 72B releases differ only in ``out_hidden_size`` (3584 and 8192).

    Args:
        depth: Number of transformer blocks.
        hidden_size: Token dimension inside the blocks.
        intermediate_size: Hidden dimension of the gated MLP in each block.
        num_heads: Number of attention heads.
        in_channels: Number of input image channels.
        patch_size: Spatial size of a patch, in pixels.
        temporal_patch_size: Number of frames one patch spans. A still image is
            repeated this many times, as the reference image processor does.
        spatial_merge_size: Side of the square of neighbouring patches the
            merger concatenates into one output token.
        window_size: Side of the attention window, in pixels.
        fullatt_block_indexes: Blocks that attend over the whole image rather
            than within a window.
        out_hidden_size: Dimension of the output tokens.
        rms_norm_eps: Epsilon of the RMS normalizations.
        rope_theta: Base of the rotary position embedding frequencies.
    """

    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    window_size: int = 112
    fullatt_block_indexes: Tuple[int, ...] = (7, 15, 23, 31)
    out_hidden_size: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads}).")
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 4 != 0:
            raise ValueError(f"The head dimension ({head_dim}) must be divisible by 4 for the 2D rotary embedding.")
        merged_patch = self.patch_size * self.spatial_merge_size
        if self.window_size % merged_patch != 0:
            raise ValueError(
                f"window_size ({self.window_size}) must be a multiple of patch_size * spatial_merge_size "
                f"({merged_patch})."
            )

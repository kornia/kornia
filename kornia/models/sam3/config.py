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

"""SAM-3 configuration for model architecture variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Sam3ModelType(Enum):
    """SAM-3 model type variants based on encoder depth/width."""

    tiny = "tiny"
    small = "small"
    base = "base"
    large = "large"


@dataclass
class Sam3Config:
    """Configuration for SAM-3 model architecture.

    Controls encoder/decoder dimensions and depth. Does not handle weight loading.
    """

    model_type: Sam3ModelType | str = Sam3ModelType.base

    # Image encoder configuration
    img_size: int = 1024
    patch_size: int = 16
    in_channels: int = 3
    encoder_embed_dim: int = 768
    encoder_depth: int = 12
    encoder_num_heads: int = 12
    encoder_mlp_ratio: float = 4.0

    # Prompt encoder configuration
    prompt_embed_dim: int = 256
    mask_in_chans: int = 16

    # Mask decoder configuration
    decoder_embed_dim: int = 256
    num_multimask_outputs: int = 3
    decoder_num_heads: int = 8
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256

    def __post_init__(self) -> None:
        """Validate and set architecture parameters based on model type."""
        if isinstance(self.model_type, str):
            self.model_type = Sam3ModelType(self.model_type)

        # Set architecture parameters based on model type
        if self.model_type == Sam3ModelType.tiny:
            self.encoder_embed_dim = 384
            self.encoder_depth = 6
            self.encoder_num_heads = 6
            self.decoder_embed_dim = 128
            self.decoder_num_heads = 4

        elif self.model_type == Sam3ModelType.small:
            self.encoder_embed_dim = 512
            self.encoder_depth = 8
            self.encoder_num_heads = 8
            self.decoder_embed_dim = 192
            self.decoder_num_heads = 6

        elif self.model_type == Sam3ModelType.base:
            self.encoder_embed_dim = 768
            self.encoder_depth = 12
            self.encoder_num_heads = 12
            self.decoder_embed_dim = 256
            self.decoder_num_heads = 8

        elif self.model_type == Sam3ModelType.large:
            self.encoder_embed_dim = 1024
            self.encoder_depth = 24
            self.encoder_num_heads = 16
            self.decoder_embed_dim = 256
            self.decoder_num_heads = 8


__all__ = ["Sam3Config", "Sam3ModelType"]

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
from typing import Optional


@dataclass
class MoonViTConfig:
    """Configuration for MoonViT vision encoder.

    Args:
        image_size: Default input image size (used for initialization, though model supports variable).
        patch_size: Size of image patches.
        num_channels: Number of input channels.
        hidden_size: Hidden dimension size.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Intermediate size in feed-forward network.
        hidden_act: Activation function.
        layer_norm_eps: Epsilon for layer normalization.
        dropout_p: Dropout probability.
        attention_dropout_p: Attention dropout probability.
        rope_theta: Theta value for Rotary Positional Embeddings.
    """

    image_size: int = 384
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    dropout_p: float = 0.0
    attention_dropout_p: float = 0.0
    rope_theta: float = 800000.0


@dataclass
class KimiVLProjectorConfig:
    """Configuration for KimiVL projector.

    Args:
        input_dim: Input dimension (should match vision encoder's hidden_size).
        hidden_dim: Hidden dimension of the MLP.
        output_dim: Output dimension (embedding dim of the LLM).
        dropout_p: Dropout probability.
    """

    input_dim: int = 1152  # Vision encoder output dimension
    hidden_dim: int = 4608  # Hidden dim from official KimiVL weights
    output_dim: int = 2048  # Output dim from official KimiVL weights
    dropout_p: float = 0.0


@dataclass
class KimiVLConfig:
    """Configuration for KimiVL model.

    Args:
        vision_config: Vision encoder configuration.
        projector_config: Projector configuration.
    """

    vision_config: Optional[MoonViTConfig] = None
    projector_config: Optional[KimiVLProjectorConfig] = None

    def __post_init__(self) -> None:
        if self.vision_config is None:
            self.vision_config = MoonViTConfig()
        if self.projector_config is None:
            self.projector_config = KimiVLProjectorConfig()

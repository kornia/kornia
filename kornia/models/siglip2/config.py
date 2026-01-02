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

"""Configuration classes for SigLip2 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SigLip2VisionConfig:
    """Configuration for SigLip2 vision encoder.

    Args:
        image_size: Size of input images.
        patch_size: Size of image patches.
        num_channels: Number of input channels.
        hidden_size: Hidden dimension size.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Intermediate size in feed-forward network.
        hidden_act: Activation function (typically 'gelu').
        layer_norm_eps: Epsilon for layer normalization.
        dropout: Dropout probability.
        attention_dropout: Attention dropout probability.
    """

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0


@dataclass
class SigLip2TextConfig:
    """Configuration for SigLip2 text encoder.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Hidden dimension size.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Intermediate size in feed-forward network.
        max_position_embeddings: Maximum sequence length.
        hidden_act: Activation function (typically 'gelu').
        layer_norm_eps: Epsilon for layer normalization.
        dropout: Dropout probability.
        attention_dropout: Attention dropout probability.
    """

    vocab_size: int = 256000  # SigLip2 uses Gemma tokenizer with 256k vocab
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0


@dataclass
class SigLip2Config:
    """Configuration for SigLip2 model.

    Args:
        vision_config: Vision encoder configuration.
        text_config: Text encoder configuration.
        projection_dim: Dimension of projection layers.
        logit_scale_init_value: Initial value for logit scale (temperature).
    """

    vision_config: Optional[SigLip2VisionConfig] = None
    text_config: Optional[SigLip2TextConfig] = None
    projection_dim: int = 768
    logit_scale_init_value: float = 2.6592

    def __post_init__(self) -> None:
        """Initialize default configs if not provided."""
        if self.vision_config is None:
            self.vision_config = SigLip2VisionConfig()
        if self.text_config is None:
            self.text_config = SigLip2TextConfig()

    @staticmethod
    def from_name(model_name: str) -> SigLip2Config:
        """Create config from model name.

        Supports the following models:
        - google/siglip-base-patch16-224 (V1)
        - google/siglip2-base-patch16-224
        - google/siglip2-base-patch16-256
        - google/siglip2-base-patch16-384
        - google/siglip2-base-patch16-512
        - google/siglip2-large-patch16-256
        - google/siglip2-large-patch16-384
        - google/siglip2-large-patch16-512

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            SigLip2Config instance configured for the specified model.
        """
        # Parse model name
        is_v1 = "google/siglip-base-patch16-224" in model_name
        is_large = "large" in model_name

        # Extract image size from model name
        image_size = 224  # default
        if "224" in model_name:
            image_size = 224
        elif "256" in model_name:
            image_size = 256
        elif "384" in model_name:
            image_size = 384
        elif "512" in model_name:
            image_size = 512

        # Base vs Large configurations
        if is_large:
            hidden_size = 1024
            num_layers = 24
            num_heads = 16
            projection_dim = 1024
        else:  # base
            hidden_size = 768
            num_layers = 12
            num_heads = 12
            projection_dim = 768

        # Intermediate size is typically 4x hidden_size
        intermediate_size = 4 * hidden_size

        # V1 vs V2 vocab size
        vocab_size = 32000 if is_v1 else 256000

        vision_config = SigLip2VisionConfig(
            image_size=image_size,
            patch_size=16,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
        )

        text_config = SigLip2TextConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
        )

        return SigLip2Config(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=projection_dim,
        )

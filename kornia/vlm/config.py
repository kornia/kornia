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

"""Base configuration classes for VLM models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisionEncoderConfig:
    """Base configuration for vision encoders.

    Attributes:
        image_size: Input image size (height and width).
        patch_size: Size of image patches.
        num_channels: Number of input image channels.
        hidden_size: Hidden dimension of the encoder.
        intermediate_size: Intermediate dimension in MLP layers.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        attention_dropout: Dropout rate for attention weights.
        hidden_dropout: Dropout rate for hidden states.
        layer_norm_eps: Epsilon for layer normalization.

    """

    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-6


@dataclass
class LanguageModelConfig:
    """Base configuration for language models.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Hidden dimension of the model.
        intermediate_size: Intermediate dimension in MLP layers.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key-value heads (for GQA).
        head_dim: Dimension of each attention head.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: Epsilon for RMS normalization.
        attention_dropout: Dropout rate for attention.
        hidden_dropout: Dropout rate for hidden states.
        rope_theta: Base for rotary position embeddings.

    """

    vocab_size: int = 257152
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    rope_theta: float = 10000.0


@dataclass
class VLMConfig:
    """Base configuration for Vision-Language Models.

    This configuration combines vision encoder and language model settings
    along with VLM-specific parameters.

    Attributes:
        vision_config: Configuration for the vision encoder.
        text_config: Configuration for the language model.
        projection_dim: Dimension of the multimodal projection.
        image_token_index: Token index representing image in the vocabulary.
        pad_token_id: Padding token index.
        bos_token_id: Beginning of sequence token index.
        eos_token_id: End of sequence token index.

    """

    vision_config: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    text_config: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    projection_dim: Optional[int] = None
    image_token_index: int = 257152
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1

    def __post_init__(self) -> None:
        """Set default projection_dim to text hidden_size if not specified."""
        if self.projection_dim is None:
            self.projection_dim = self.text_config.hidden_size

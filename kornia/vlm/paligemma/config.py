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

"""Configuration classes for PaliGemma models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SigLIPVisionConfig:
    """Configuration for SigLIP Vision Encoder.

    Default values correspond to SigLIP-So400m used in PaliGemma 2.

    Attributes:
        image_size: Input image size.
        patch_size: Size of image patches.
        num_channels: Number of input channels.
        hidden_size: Hidden dimension.
        intermediate_size: MLP intermediate dimension.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        attention_dropout: Dropout for attention weights.
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
    layer_norm_eps: float = 1e-6

    @property
    def num_patches(self) -> int:
        """Number of patches in the image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads


@dataclass
class GemmaConfig:
    """Configuration for Gemma Language Model.

    Default values correspond to Gemma 2B used in PaliGemma 2 3B.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Hidden dimension.
        intermediate_size: MLP intermediate dimension.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key-value heads (for GQA).
        head_dim: Dimension of each attention head.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: Epsilon for RMS normalization.
        rope_theta: Base for rotary position embeddings.
        attention_dropout: Dropout for attention weights.

    """

    vocab_size: int = 257216
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0


@dataclass
class PaliGemma2Config:
    """Configuration for PaliGemma 2 Vision-Language Model.

    Combines SigLIP vision encoder with Gemma language decoder.

    Attributes:
        vision_config: Configuration for the vision encoder.
        text_config: Configuration for the language model.
        projection_dim: Dimension for multimodal projection.
        image_token_index: Token index representing images in vocabulary.
        pad_token_id: Padding token index.
        bos_token_id: Beginning of sequence token index.
        eos_token_id: End of sequence token index.

    """

    vision_config: SigLIPVisionConfig = field(default_factory=SigLIPVisionConfig)
    text_config: GemmaConfig = field(default_factory=GemmaConfig)
    projection_dim: Optional[int] = None
    image_token_index: int = 257152
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1

    def __post_init__(self) -> None:
        """Set projection_dim to text hidden_size if not specified."""
        if self.projection_dim is None:
            self.projection_dim = self.text_config.hidden_size

    @classmethod
    def paligemma2_3b_224(cls) -> "PaliGemma2Config":
        """Create config for PaliGemma 2 3B with 224x224 images."""
        return cls(
            vision_config=SigLIPVisionConfig(image_size=224),
            text_config=GemmaConfig(),
        )

    @classmethod
    def paligemma2_3b_448(cls) -> "PaliGemma2Config":
        """Create config for PaliGemma 2 3B with 448x448 images."""
        return cls(
            vision_config=SigLIPVisionConfig(image_size=448),
            text_config=GemmaConfig(),
        )

    @classmethod
    def paligemma2_3b_896(cls) -> "PaliGemma2Config":
        """Create config for PaliGemma 2 3B with 896x896 images."""
        return cls(
            vision_config=SigLIPVisionConfig(image_size=896),
            text_config=GemmaConfig(),
        )

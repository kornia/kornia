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

from torch import nn


def _activation(hidden_act: str) -> nn.Module:
    """Return the MLP activation that a ``hidden_act`` config value names.

    The names are transformers' ``ACT2FN`` keys: ``"gelu_pytorch_tanh"`` and ``"gelu_new"`` are the tanh approximation
    of GELU, which the released SigLIP and SigLIP 2 checkpoints use, and ``"gelu"`` is the exact GELU.

    Raises:
        ValueError: for any other value.
    """
    if hidden_act in ("gelu_pytorch_tanh", "gelu_new"):
        return nn.GELU(approximate="tanh")
    if hidden_act == "gelu":
        return nn.GELU()
    raise ValueError(
        f"Unsupported hidden_act {hidden_act!r}: expected 'gelu_pytorch_tanh' or 'gelu_new' (tanh GELU) or 'gelu'."
    )


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
        hidden_act: Activation of the MLPs: ``"gelu_pytorch_tanh"`` (default, the checkpoints' value) or
            ``"gelu_new"`` for tanh GELU, ``"gelu"`` for exact GELU. Other values raise ``ValueError`` when the model
            is built.
        layer_norm_eps: Epsilon for layer normalization.
        dropout_p: Dropout probability.
        attention_dropout_p: Attention dropout probability.
    """

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    dropout_p: float = 0.0
    attention_dropout_p: float = 0.0


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
        hidden_act: Activation of the MLPs: ``"gelu_pytorch_tanh"`` (default, the checkpoints' value) or
            ``"gelu_new"`` for tanh GELU, ``"gelu"`` for exact GELU. Other values raise ``ValueError`` when the model
            is built.
        layer_norm_eps: Epsilon for layer normalization.
        dropout_p: Dropout probability.
        attention_dropout_p: Attention dropout probability.
    """

    vocab_size: int = 256000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    dropout_p: float = 0.0
    attention_dropout_p: float = 0.0


# The checkpoints SigLip2Config.from_name supports: name -> (vocab_size, hidden_size, num_hidden_layers,
# num_attention_heads, image_size), from each checkpoint's config.json on the Hugging Face Hub and transformers'
# SiglipTextConfig / SiglipVisionConfig defaults for the fields it omits (vocabulary 32000, hidden size 768, 12 layers,
# 12 heads, image size 224, patch size 16, 64 text positions).
_SUPPORTED_MODELS: dict[str, tuple[int, int, int, int, int]] = {
    "google/siglip-base-patch16-224": (32000, 768, 12, 12, 224),
    "google/siglip2-base-patch16-224": (256000, 768, 12, 12, 224),
    "google/siglip2-base-patch16-256": (256000, 768, 12, 12, 256),
    "google/siglip2-base-patch16-384": (256000, 768, 12, 12, 384),
    "google/siglip2-base-patch16-512": (256000, 768, 12, 12, 512),
    "google/siglip2-large-patch16-256": (256000, 1024, 24, 16, 256),
    "google/siglip2-large-patch16-384": (256000, 1024, 24, 16, 384),
    "google/siglip2-large-patch16-512": (256000, 1024, 24, 16, 512),
}


@dataclass
class SigLip2Config:
    """Configuration for SigLip2 model.

    Args:
        vision_config: Vision encoder configuration.
        text_config: Text encoder configuration.
        projection_dim: Dimension of projection layers.
        logit_scale_init_value: Initial value for logit scale (temperature).
        logit_scale_max: Maximum value for logit_scale to prevent overflow. Default: 100.0
    """

    vision_config: Optional[SigLip2VisionConfig] = None
    text_config: Optional[SigLip2TextConfig] = None
    projection_dim: int = 768
    logit_scale_init_value: float = 2.6592
    logit_scale_max: float = 100.0

    def __post_init__(self) -> None:
        """Initialize default configs if not provided."""
        if self.vision_config is None:
            self.vision_config = SigLip2VisionConfig()
        if self.text_config is None:
            self.text_config = SigLip2TextConfig()

    @staticmethod
    def from_name(model_name: str) -> SigLip2Config:
        """Create the configuration of a supported checkpoint from its name.

        Supports the following models:
        - google/siglip-base-patch16-224 (V1)
        - google/siglip2-base-patch16-224
        - google/siglip2-base-patch16-256
        - google/siglip2-base-patch16-384
        - google/siglip2-base-patch16-512
        - google/siglip2-large-patch16-256
        - google/siglip2-large-patch16-384
        - google/siglip2-large-patch16-512

        Each name gives its checkpoint's ``config.json`` values: patch size 16, 64 text positions, the vocabulary
        (32000 for the SigLIP 1 model, 256000 for SigLIP 2), and the base (hidden size 768, 12 layers, 12 heads) or large
        (1024, 24, 16) architecture.

        Args:
            model_name: HuggingFace model identifier, one of the names above.

        Returns:
            SigLip2Config instance configured for the specified model.

        Raises:
            ValueError: if ``model_name`` is not one of the supported names.
        """
        if model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported SigLip2 model name {model_name!r}. Supported names: {', '.join(_SUPPORTED_MODELS)}."
            )
        vocab_size, hidden_size, num_layers, num_heads, image_size = _SUPPORTED_MODELS[model_name]
        # every supported checkpoint has an MLP of 4 x hidden_size (3072 for base, 4096 for large)
        intermediate_size = 4 * hidden_size

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
            max_position_embeddings=64,
        )

        return SigLip2Config(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=hidden_size,
        )

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
class Qwen3VLVisionConfig:
    """Configuration for the Qwen3-VL vision encoder.

    The vision tower is a ViT with DeepStack fusion: features from multiple
    intermediate transformer layers are merged before projection into the LLM
    embedding space, enabling dynamic-resolution inputs.

    Args:
        patch_size: Spatial patch size used by the patch embedding conv.
        temporal_patch_size: Temporal patch size used for video inputs.
        in_channels: Number of input image channels.
        hidden_size: Transformer hidden dimension.
        num_hidden_layers: Number of transformer blocks in the vision tower.
        num_attention_heads: Attention heads per block.
        intermediate_size: Hidden size of the MLP inside each transformer block.
        hidden_act: Activation function name used inside the MLP.
        layer_norm_eps: Epsilon used by layer normalisation.
        rope_theta: Base frequency for rotary position embeddings.
        spatial_merge_size: Spatial token merge factor between vision and projector.
        deepstack_layer_indices: Transformer layer indices whose features are
            stacked together to form the projector input. ``None`` means the
            default per-size schedule will be used.
    """

    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    spatial_merge_size: int = 2
    deepstack_layer_indices: Optional[tuple[int, ...]] = None


@dataclass
class Qwen3VLProjectorConfig:
    """Configuration for the projector that maps vision features into the LLM space.

    Args:
        input_dim: Per-layer feature dimension produced by the vision encoder.
        num_stack_layers: Number of vision layers concatenated by DeepStack.
        hidden_dim: Hidden dimension of the projector MLP.
        output_dim: Output dimension; must match the LLM embedding size.
        dropout_p: Dropout applied after the projection.
    """

    input_dim: int = 1152
    num_stack_layers: int = 3
    hidden_dim: int = 4608
    output_dim: int = 2048
    dropout_p: float = 0.0


@dataclass
class Qwen3VLTextConfig:
    """Configuration for the Qwen3-VL text decoder.

    Defaults target the dense Qwen3-VL-2B variant. Use :func:`Qwen3VLConfig.from_size`
    to obtain configurations for the 4B and 8B tiers.

    Args:
        vocab_size: Tokenizer vocabulary size.
        hidden_size: Transformer hidden dimension.
        num_hidden_layers: Number of transformer blocks.
        num_attention_heads: Number of query heads per block.
        num_key_value_heads: Number of key/value heads per block (GQA).
        intermediate_size: Hidden size of the MLP inside each transformer block.
        max_position_embeddings: Maximum supported sequence length.
        rms_norm_eps: Epsilon used by RMSNorm.
        rope_theta: Base frequency for rotary position embeddings.
        tie_word_embeddings: Whether the embedding and unembedding share weights.
    """

    vocab_size: int = 151_936
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 5632
    max_position_embeddings: int = 32_768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True


@dataclass
class Qwen3VLConfig:
    """Top-level configuration for the Qwen3-VL family.

    Bundles the vision tower, projector, and text decoder configurations.
    The default values describe the 2B dense model. Use :func:`from_size`
    to obtain a configuration matching one of the published model tiers.

    Args:
        vision_config: Vision encoder configuration.
        projector_config: Projector configuration.
        text_config: Text decoder configuration.
        image_token_id: Token id reserved for image placeholders.
        video_token_id: Token id reserved for video placeholders.
    """

    vision_config: Optional[Qwen3VLVisionConfig] = None
    projector_config: Optional[Qwen3VLProjectorConfig] = None
    text_config: Optional[Qwen3VLTextConfig] = None
    image_token_id: int = 151_655
    video_token_id: int = 151_656

    def __post_init__(self) -> None:
        if self.vision_config is None:
            self.vision_config = Qwen3VLVisionConfig()
        if self.projector_config is None:
            self.projector_config = Qwen3VLProjectorConfig()
        if self.text_config is None:
            self.text_config = Qwen3VLTextConfig()

    @classmethod
    def from_size(cls, size: str) -> Qwen3VLConfig:
        """Return a configuration for one of the supported dense model sizes.

        Args:
            size: One of ``"2b"``, ``"4b"``, or ``"8b"`` (case-insensitive).

        Returns:
            A :class:`Qwen3VLConfig` populated with the corresponding
            vision, projector, and text-decoder defaults.
        """
        key = size.lower()
        if key not in _SIZE_PRESETS:
            valid = ", ".join(sorted(_SIZE_PRESETS))
            raise ValueError(f"Unknown Qwen3-VL size '{size}'. Valid options: {valid}.")
        vision_kwargs, projector_kwargs, text_kwargs = _SIZE_PRESETS[key]
        return cls(
            vision_config=Qwen3VLVisionConfig(**vision_kwargs),
            projector_config=Qwen3VLProjectorConfig(**projector_kwargs),
            text_config=Qwen3VLTextConfig(**text_kwargs),
        )


_SIZE_PRESETS: dict[str, tuple[dict[str, object], dict[str, object], dict[str, object]]] = {
    "2b": (
        {},
        {"output_dim": 2048},
        {"hidden_size": 2048, "num_hidden_layers": 28, "intermediate_size": 5632},
    ),
    "4b": (
        {},
        {"output_dim": 2560},
        {"hidden_size": 2560, "num_hidden_layers": 36, "intermediate_size": 6912},
    ),
    "8b": (
        {},
        {"output_dim": 4096},
        {
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12_288,
        },
    ),
}

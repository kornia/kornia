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
    """Configuration for the Qwen3-VL vision tower.

    Defaults match the dense Qwen3-VL-2B model. The vision tower is a ViT with a
    learned absolute position embedding (bilinearly interpolated to the input
    grid) plus 2D rotary embeddings, ending with a patch-merger that projects
    into the LLM hidden dimension and exposes intermediate DeepStack features.
    """

    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (5, 11, 17)
    initializer_range: float = 0.02


@dataclass
class Qwen3VLTextConfig:
    """Configuration for the Qwen3-VL text decoder.

    Defaults target the dense Qwen3-VL-2B variant. The text stack is not used
    by the vision-only pipeline shipped here, but the dataclass is kept on the
    top-level config so downstream PRs can fill it in without an API change.
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

    Bundles the vision tower and text decoder configurations. The default
    values describe the 2B dense model. Use :func:`from_size` to obtain a
    configuration matching one of the published model tiers.
    """

    vision_config: Optional[Qwen3VLVisionConfig] = None
    text_config: Optional[Qwen3VLTextConfig] = None
    image_token_id: int = 151_655
    video_token_id: int = 151_656

    def __post_init__(self) -> None:
        if self.vision_config is None:
            self.vision_config = Qwen3VLVisionConfig()
        if self.text_config is None:
            self.text_config = Qwen3VLTextConfig()

    @classmethod
    def from_size(cls, size: str) -> Qwen3VLConfig:
        """Return a configuration for one of the supported dense model sizes.

        Valid sizes are ``"2b"``, ``"4b"``, and ``"8b"`` (case-insensitive).
        Only the 2B preset is verified against published weights; the 4B/8B
        presets follow the publicly documented vision/text dimensions.
        """
        key = size.lower()
        if key not in _SIZE_PRESETS:
            valid = ", ".join(sorted(_SIZE_PRESETS))
            raise ValueError(f"Unknown Qwen3-VL size '{size}'. Valid options: {valid}.")
        vision_kwargs, text_kwargs = _SIZE_PRESETS[key]
        return cls(
            vision_config=Qwen3VLVisionConfig(**vision_kwargs),
            text_config=Qwen3VLTextConfig(**text_kwargs),
        )


_SIZE_PRESETS: dict[str, tuple[dict[str, object], dict[str, object]]] = {
    "2b": (
        {"out_hidden_size": 2048},
        {"hidden_size": 2048, "num_hidden_layers": 28, "intermediate_size": 5632},
    ),
    "4b": (
        {"out_hidden_size": 2560},
        {"hidden_size": 2560, "num_hidden_layers": 36, "intermediate_size": 6912},
    ),
    "8b": (
        {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "out_hidden_size": 4096,
            "deepstack_visual_indexes": (8, 16, 24),
        },
        {
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12_288,
        },
    ),
}

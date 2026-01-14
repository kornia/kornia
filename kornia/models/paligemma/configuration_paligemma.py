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

from dataclasses import dataclass, field
from typing import Optional

from kornia.models.siglip2.config import SigLip2VisionConfig


@dataclass
class PaliGemmaConfig:
    """Configuration class for PaliGemma.

    Args:
        vision_config: Configuration for the SigLip2 vision encoder.
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of the embeddings.
        intermediate_size: Dimension of the intermediate (Feed Forward) layer.
        num_hidden_layers: Number of Transformer layers in the text model.
        num_attention_heads: Number of attention heads per transformer layer.
        num_key_value_heads: Number of key/value heads (for multi-query attention).
        head_dim: Dimension of each attention head.
        max_position_embeddings: Maximum number of position embeddings.
        rope_theta: The base period of the RoPE embeddings.
        ignore_index: Index to ignore in the loss function.
        image_token_index: Token index used for image placeholders.
    """

    vision_config: Optional[SigLip2VisionConfig] = field(default_factory=SigLip2VisionConfig)
    vocab_size: int = 257152
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    ignore_index: int = -100
    image_token_index: int = 256000

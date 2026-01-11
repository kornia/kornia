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

from kornia.models.siglip2.config import SigLip2VisionConfig


class PaliGemmaConfig:
    """Configuration class for PaliGemma.

    Args:
        vision_config: Configuration for the SigLip2 vision encoder.
        text_config: Configuration for the Gemma text decoder (to be added).
        ignore_index: Index to ignore in the loss function.
        image_token_index: Token index used for image placeholders.
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of the embeddings.
    """

    def __init__(
        self,
        vision_config: SigLip2VisionConfig = None,
        vocab_size: int = 257152,
        hidden_size: int = 2048,
        intermediate_size: int = 16384,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 1,
        head_dim: int = 256,
        max_position_embeddings: int = 8192,
        **kwargs,
    ):
        self.vision_config = vision_config if vision_config is not None else SigLip2VisionConfig()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
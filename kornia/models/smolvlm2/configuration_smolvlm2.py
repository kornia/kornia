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

"""Configuration classes for the SmolVLM2 vision-language model.

Reference: ``transformers.models.smolvlm.configuration_smolvlm`` (HuggingFace),
checkpoint ``HuggingFaceTB/SmolVLM2-2.2B-Instruct``. SmolVLM2 pairs a SigLIP
vision encoder with a SmolLM2 (Llama-architecture) text decoder, joined by a
pixel-shuffle connector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from kornia.models.siglip2.config import SigLip2VisionConfig


@dataclass
class SmolVLM2TextConfig:
    """Configuration for the SmolVLM2 text decoder (SmolLM2 / Llama architecture).

    Defaults follow the Llama decoder family; the exact values for a given
    checkpoint (e.g. ``HuggingFaceTB/SmolVLM2-2.2B-Instruct``) are provided by
    that checkpoint's ``text_config`` and should be passed explicitly when
    loading pre-trained weights.

    Args:
        vocab_size: Size of the token vocabulary (includes the ``<image>`` token).
        hidden_size: Dimension of the decoder embeddings.
        intermediate_size: Dimension of the SwiGLU feed-forward layer.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of attention (query) heads.
        num_key_value_heads: Number of key/value heads (grouped-query attention).
        head_dim: Dimension of each attention head. If ``None``, uses
            ``hidden_size // num_attention_heads``.
        max_position_embeddings: Maximum sequence length for rotary embeddings.
        rope_theta: Base period of the rotary position embeddings.
        rms_norm_eps: Epsilon for the RMS normalization layers.
        pad_token_id: Padding token index used by the embedding table.
    """

    vocab_size: int = 49280
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: Optional[int] = None
    max_position_embeddings: int = 8192
    rope_theta: float = 130000.0
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 2

    def __post_init__(self) -> None:
        """Fill in ``head_dim`` from ``hidden_size`` and ``num_attention_heads`` when unset."""
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class SmolVLM2Config:
    r"""Configuration for the full SmolVLM2 vision-language model.

    Default values follow the ``HuggingFaceTB/SmolVLM2-2.2B-Instruct``
    checkpoint ``config.json``: a SigLIP ``so400m``-style vision tower
    (``hidden_size=1152``, ``patch_size=14``, ``image_size=384`` giving a
    :math:`27 \\times 27` patch grid) with ``scale_factor=3`` so each image
    contributes :math:`729 / 3^2 = 81` tokens to the decoder.

    Note:
        The checkpoint's vision activation is ``gelu_pytorch_tanh``; Kornia's
        ``SigLip2VisionModel`` currently hard-codes exact ``nn.GELU`` (the same
        approximation difference applies to the PaliGemma model, which shares
        the tower).

    Args:
        vision_config: Configuration for the SigLIP vision encoder used as the
            vision tower.
        text_config: Configuration for the SmolLM2 (Llama) text decoder.
        scale_factor: Pixel-shuffle spatial downsampling factor. The connector
            reduces the number of vision tokens by ``scale_factor ** 2`` and
            widens their channel dimension by the same factor.
        image_token_id: Token id of the ``<image>`` placeholder whose embeddings
            are replaced by projected vision features.
        pad_token_id: Padding token index.
    """

    vision_config: SigLip2VisionConfig = field(
        default_factory=lambda: SigLip2VisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
        )
    )
    text_config: SmolVLM2TextConfig = field(default_factory=SmolVLM2TextConfig)
    scale_factor: int = 3
    image_token_id: int = 49190
    pad_token_id: int = 2

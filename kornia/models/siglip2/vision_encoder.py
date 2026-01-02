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

"""Vision encoder for SigLip2."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import SigLip2Attention
from .config import SigLip2VisionConfig

__all__ = [
    "SigLip2MultiheadAttentionPoolingHead",
    "SigLip2VisionEmbeddings",
    "SigLip2VisionEncoder",
    "SigLip2VisionLayer",
    "SigLip2VisionModel",
]


class SigLip2VisionEmbeddings(nn.Module):
    """Vision embeddings for SigLip2.

    Combines patch embedding and position embeddings.

    Args:
        config: Vision encoder configuration.
    """

    def __init__(self, config: SigLip2VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size

        # extract patches from the image using a convolutional layer
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

        # calculate the number of patches in the image
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # position embeddings - [num_patches, hidden_size]
        self.position_embedding = nn.Parameter(torch.randn(self.num_patches, config.hidden_size))

        # dropout or identity
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width).

        Returns:
            Embedded patches of shape (batch_size, num_patches, hidden_size).
        """
        # extract patches from the image
        embeddings = self.patch_embedding(pixel_values)  # (batch_size, hidden_size, H', W')
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (batch_size, num_patches, hidden_size)

        # add position embeddings to the embeddings
        embeddings = embeddings + self.position_embedding.unsqueeze(0)

        return embeddings


class SigLip2VisionMLP(nn.Module):
    """MLP (feed-forward network) for vision encoder.

    Args:
        config: Vision encoder configuration.
    """

    def __init__(self, config: SigLip2VisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SigLip2VisionLayer(nn.Module):
    """Transformer layer for vision encoder.

    Implements pre-norm architecture with residual connections.

    Args:
        config: Vision encoder configuration.
    """

    def __init__(self, config: SigLip2VisionConfig) -> None:
        super().__init__()
        self.self_attn = SigLip2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.mlp = SigLip2VisionMLP(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SigLip2MultiheadAttentionPoolingHead(nn.Module):
    """Multi-head attention pooling head for vision encoder.

    This implements the pooling mechanism used by HF SigLip vision model.
    Uses a learnable probe token with multi-head attention to pool the sequence.

    Args:
        config: Vision encoder configuration.
    """

    def __init__(self, config: SigLip2VisionConfig) -> None:
        super().__init__()
        # Learnable probe (query token)
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Multi-head attention (using PyTorch's built-in for compatibility)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLip2VisionMLP(config)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # repeat the probe token for the batch size
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)  # (batch_size, 1, hidden_size)

        # multi-head attention: probe as query, hidden_state as key/value
        hidden_state, _ = self.attention(probe, hidden_state, hidden_state)

        # residual connection with layer norm and MLP
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        # return the first (and only) token - [batch_size, hidden_size]
        return hidden_state[:, 0]


class SigLip2VisionEncoder(nn.Module):
    """Vision encoder stack for SigLip2.

    Args:
        config: Vision encoder configuration.
    """

    def __init__(self, config: SigLip2VisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SigLip2VisionLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass through encoder layers.

        Args:
            hidden_states: Input embeddings of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Tuple of (last_hidden_state,) or (last_hidden_state, all_hidden_states).
        """
        all_hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return (hidden_states, tuple(all_hidden_states))

        return (hidden_states,)


class SigLip2VisionModel(nn.Module):
    """Complete vision encoder model for SigLip2.

    Args:
        config: Vision encoder configuration.
    """

    def __init__(self, config: SigLip2VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SigLip2VisionEmbeddings(config)
        self.encoder = SigLip2VisionEncoder(config)

        # post layer norm - [batch_size, hidden_size]
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # head: multi-head attention pooling - [batch_size, hidden_size]
        self.head = SigLip2MultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width).
            attention_mask: Optional attention mask.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Tuple containing:
            - Pooled output (mean pooled)
            - Last hidden state
            - All hidden states (if output_hidden_states=True)
        """
        # Get embeddings
        hidden_states = self.embeddings(pixel_values)

        # Encode
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]

        # apply post layer norm - [batch_size, hidden_size]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # pool: multi-head attention pooling - [batch_size, hidden_size]
        pooled_output = self.head(last_hidden_state)

        if output_hidden_states:
            return (pooled_output, last_hidden_state, encoder_outputs[1])

        # return the pooled output and the last hidden state - [batch_size, hidden_size]
        return (pooled_output, last_hidden_state)

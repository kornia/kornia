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

"""Text encoder for SigLip2."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from kornia.core import Module, Tensor

from .attention import SigLip2Attention
from .config import SigLip2TextConfig

__all__ = ["SigLip2TextEmbeddings", "SigLip2TextEncoder", "SigLip2TextLayer", "SigLip2TextModel"]


class SigLip2TextEmbeddings(Module):
    """Text embeddings for SigLip2.

    Combines token embeddings and position embeddings.

    Args:
        config: Text encoder configuration.
    """

    def __init__(self, config: SigLip2TextConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings (learned)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Note: HF SigLip does NOT use layer_norm or dropout in embeddings
        # Removed to match HF exactly and enable strict weight loading
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, input_ids: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            position_ids: Optional position IDs of shape (batch_size, seq_len).
                If None, uses sequential positions.

        Returns:
            Embedded tokens of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)

        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)

        # Combine embeddings
        embeddings = token_embeddings + position_embeddings

        # Note: HF SigLip does NOT use layer_norm or dropout in embeddings
        # Skip to match HF exactly
        # embeddings = self.layer_norm(embeddings)  # HF doesn't use this
        # embeddings = self.dropout(embeddings)  # HF doesn't use this

        return embeddings


class SigLip2TextMLP(Module):
    """MLP (feed-forward network) for text encoder.

    Args:
        config: Text encoder configuration.
    """

    def __init__(self, config: SigLip2TextConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SigLip2TextLayer(Module):
    """Transformer layer for text encoder.

    Implements pre-norm architecture with residual connections.

    Args:
        config: Text encoder configuration.
    """

    def __init__(self, config: SigLip2TextConfig) -> None:
        super().__init__()
        self.attention = SigLip2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.mlp = SigLip2TextMLP(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
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
        hidden_states = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SigLip2TextEncoder(Module):
    """Text encoder stack for SigLip2.

    Args:
        config: Text encoder configuration.
    """

    def __init__(self, config: SigLip2TextConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SigLip2TextLayer(config) for _ in range(config.num_hidden_layers)])
        # Note: HF SigLip does NOT use layer_norm at encoder level (uses final_layer_norm instead)
        # Keep for compatibility but don't use
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, ...]:
        """Forward pass through encoder layers.

        Args:
            hidden_states: Input embeddings of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Tuple of (last_hidden_state,) or (last_hidden_state, all_hidden_states).
        """
        all_hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # Note: HF SigLip uses final_layer_norm, not encoder.layer_norm
        # We map final_layer_norm -> encoder.layer_norm in weight loading
        # But HF doesn't apply it here - it's applied in the model forward
        # Actually, let me check if HF applies it...
        # For now, skip to match HF structure
        # hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return (hidden_states, tuple(all_hidden_states))

        return (hidden_states,)


class SigLip2TextModel(Module):
    """Complete text encoder model for SigLip2.

    Args:
        config: Text encoder configuration.
    """

    def __init__(self, config: SigLip2TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SigLip2TextEmbeddings(config)
        self.encoder = SigLip2TextEncoder(config)
        # Head layer (HF applies this after pooling)
        self.head = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, ...]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
                Values should be 1 for unmasked tokens and 0 for masked tokens.
            position_ids: Optional position IDs.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Tuple containing:
            - Pooled output (EOS token)
            - Last hidden state
            - All hidden states (if output_hidden_states=True)
        """
        # Get embeddings
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)

        # Encode (attention module handles mask conversion internally)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]

        # Apply final_layer_norm (HF applies it here, not in encoder)
        # We map final_layer_norm -> encoder.layer_norm in weight loading
        last_hidden_state = self.encoder.layer_norm(last_hidden_state)

        # Pool: HF uses last token (assuming "sticky" EOS tokenization)
        # From HF code: "Assuming 'sticky' EOS tokenization, last token is always EOS."
        pooled_output = last_hidden_state[:, -1]

        # Apply head layer (HF applies this after pooling)
        pooled_output = self.head(pooled_output)

        if output_hidden_states:
            return (pooled_output, last_hidden_state, encoder_outputs[1])
        return (pooled_output, last_hidden_state)

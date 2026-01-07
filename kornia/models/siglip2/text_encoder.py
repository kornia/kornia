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

from .attention import SigLip2Attention
from .config import SigLip2TextConfig

__all__ = ["SigLip2TextEmbeddings", "SigLip2TextEncoder", "SigLip2TextLayer", "SigLip2TextModel"]


class SigLip2TextEmbeddings(nn.Module):
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

        # token embeddings - [vocab_size, hidden_size]
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # position embeddings - [max_position_embeddings, hidden_size]
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            position_ids: Optional position IDs of shape (batch_size, seq_len).
                If None, uses sequential positions.

        Returns:
            Embedded tokens of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len = input_ids.shape

        # token embeddings - [batch_size, seq_len, hidden_size]
        token_embeddings = self.token_embedding(input_ids)

        # position embeddings - [batch_size, seq_len, hidden_size]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)

        # combine embeddings - [batch_size, seq_len, hidden_size]
        embeddings = token_embeddings + position_embeddings

        return embeddings


class SigLip2TextMLP(nn.Module):
    """MLP (feed-forward network) for text encoder.

    Args:
        config: Text encoder configuration.
    """

    def __init__(self, config: SigLip2TextConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # linear transformation - [batch_size, seq_len, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # activation function - [batch_size, seq_len, intermediate_size]
        hidden_states = self.activation(hidden_states)
        # linear transformation - [batch_size, seq_len, hidden_size]
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SigLip2TextLayer(nn.Module):
    """Transformer layer for SigLip2 text encoder.

    Implements pre-norm architecture with residual connections.

    Args:
        config: Text encoder configuration.
    """

    def __init__(self, config: SigLip2TextConfig) -> None:
        super().__init__()
        self.self_attn = SigLip2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout_p=config.attention_dropout_p,
        )
        self.mlp = SigLip2TextMLP(config)
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
        # self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # MLP with pre-norm - [batch_size, seq_len, hidden_size]
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # return the hidden states - [batch_size, seq_len, hidden_size]
        return hidden_states


class SigLip2TextEncoder(nn.Module):
    """Text encoder stack for SigLip2.

    Args:
        config: Text encoder configuration.
    """

    def __init__(self, config: SigLip2TextConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SigLip2TextLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass through encoder layers.

        Args:
            hidden_states: Input embeddings of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Tuple of (last_hidden_state,) or (last_hidden_state, all_hidden_states).
        """
        all_hidden_states: list[torch.Tensor] = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return (hidden_states, tuple(all_hidden_states))

        return (hidden_states,)


class SigLip2TextModel(nn.Module):
    """Complete text encoder model for SigLip2.

    Args:
        config: Text encoder configuration.
    """

    def __init__(self, config: SigLip2TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SigLip2TextEmbeddings(config)
        self.encoder = SigLip2TextEncoder(config)
        # final layer norm - [batch_size, hidden_size]
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # head layer - [batch_size, hidden_size]
        self.head = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, ...]:
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
        # get embeddings - [batch_size, seq_len, hidden_size]
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)

        # encode - [batch_size, seq_len, hidden_size]
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]

        # apply final layer norm - [batch_size, hidden_size]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # pool: use the last token - [batch_size, hidden_size]
        pooled_output = last_hidden_state[:, -1]

        # apply head layer - [batch_size, hidden_size]
        pooled_output = self.head(pooled_output)

        if output_hidden_states:
            return (pooled_output, last_hidden_state, encoder_outputs[1])
        return (pooled_output, last_hidden_state)

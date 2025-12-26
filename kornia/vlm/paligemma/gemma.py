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

"""Gemma Language Model implementation.

Gemma is a family of decoder-only language models from Google.
This implementation focuses on Gemma 2B used in PaliGemma 2.

Reference: https://arxiv.org/abs/2403.08295
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Tensor

from ..layers import RMSNorm, SwiGLU
from ..layers.attention import MultiHeadAttention
from .config import GemmaConfig


class GemmaTransformerBlock(nn.Module):
    """Single transformer block for Gemma decoder.

    Pre-norm architecture with self-attention and SwiGLU feedforward.

    Args:
        config: Gemma configuration.
        block_idx: Index of this block (for cache management).

    """

    def __init__(self, config: GemmaConfig, block_idx: int = 0) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.block_idx = block_idx

        self.attn = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dropout=config.attention_dropout,
            bias=False,
            use_rotary=True,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )

        self.ffn = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
        )

        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        cached_kv: Optional[Tuple[Tensor, Tensor]] = None,
        return_weights: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass of the transformer block.

        Args:
            x: Input tensor of shape (B, seq_len, dim).
            mask: Optional causal mask.
            positions: Position indices for RoPE.
            cached_kv: Cached (key, value) from previous step.
            return_weights: Whether to return attention weights.
            use_cache: Whether to return new key-value cache.

        Returns:
            Tuple of output tensor, optional attention weights, optional cache.

        """
        # Self-attention with residual
        residual = x
        x = self.pre_attn_norm(x)

        x, attn_w, new_kv = self.attn(
            x,
            attention_mask=mask,
            position_ids=positions,
            past_key_value=cached_kv,
            output_attentions=return_weights,
            use_cache=use_cache,
        )
        x = residual + x

        # FFN with residual
        residual = x
        x = self.pre_ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_w, new_kv


class GemmaDecoder(nn.Module):
    """Gemma decoder model (without output head).

    Stack of transformer blocks with embedding and final normalization.

    Args:
        config: Gemma configuration.

    """

    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.dim = config.hidden_size

        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [GemmaTransformerBlock(config, block_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer."""
        return self.token_embed

    def set_embeddings(self, value: nn.Embedding) -> None:
        """Set the token embedding layer."""
        self.token_embed = value

    def _make_causal_mask(
        self,
        padding_mask: Optional[Tensor],
        shape: Tuple[int, int],
        cache_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Prepare causal attention mask.

        Args:
            padding_mask: Optional padding mask of shape (B, seq_len).
            shape: (batch_size, seq_len) tuple.
            cache_len: Length of cached key-values.
            dtype: Data type for the mask.
            device: Device for the mask.

        Returns:
            Causal mask of shape (B, 1, seq_len, total_len).

        """
        B, L = shape
        total_len = L + cache_len

        # Create causal mask: lower triangular matrix (including diagonal)
        # Positions can attend to themselves and all previous positions
        # Upper triangle (future positions) is masked with -inf
        #
        # For cached positions: all new positions can attend to all cached positions
        # For new positions: use standard causal mask (lower triangular)
        causal = torch.full((L, total_len), float("-inf"), dtype=dtype, device=device)

        if L > 0:
            # All new positions can attend to all cached positions
            if cache_len > 0:
                causal[:, :cache_len] = 0.0

            # Create lower triangular mask for new positions
            # Position i in new sequence can attend to positions [cache_len, cache_len + i + 1)
            for i in range(L):
                causal[i, cache_len : cache_len + i + 1] = 0.0

        # Expand for batch and heads
        causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, L, total_len)
        causal = causal.expand(B, 1, -1, -1)

        # Apply padding mask if provided
        if padding_mask is not None:
            # padding_mask: (B, total_len) where 1 = valid token, 0 = padding
            # Convert to attention mask: 0 -> -inf, 1 -> 0
            pad_mask = padding_mask[:, None, None, :].to(dtype)
            pad_mask = (1.0 - pad_mask) * float("-inf")
            causal = causal + pad_mask

        return causal

    def forward(
        self,
        token_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        embeds: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        return_weights: bool = False,
        return_intermediates: bool = False,
        use_cache: bool = False,
    ) -> Tuple[
        Tensor, Optional[Tuple[Tensor, ...]], Optional[Tuple[Tensor, ...]], Optional[Tuple[Tuple[Tensor, Tensor], ...]]
    ]:
        """Forward pass of the Gemma decoder.

        Args:
            token_ids: Token IDs of shape (B, seq_len).
            mask: Padding mask of shape (B, seq_len).
            positions: Position indices of shape (B, seq_len).
            embeds: Pre-computed embeddings of shape (B, seq_len, dim).
            kv_cache: Cached key-value states from previous steps.
            return_weights: Whether to return attention weights.
            return_intermediates: Whether to return features from all layers.
            use_cache: Whether to return new cache.

        Returns:
            Tuple of:
                - Final features
                - All layer features (if return_intermediates)
                - All attention weights (if return_weights)
                - New cache (if use_cache)

        """
        # Get embeddings
        if embeds is None:
            embeds = self.token_embed(token_ids)

        # Gemma normalizes embeddings by sqrt(dim)
        x = embeds * (self.dim**0.5)

        B, L = x.shape[:2]
        cache_len = kv_cache[0][0].shape[2] if kv_cache is not None else 0

        # Prepare position IDs
        if positions is None:
            positions = torch.arange(cache_len, cache_len + L, device=x.device)
            positions = positions.unsqueeze(0).expand(B, -1)

        # Prepare causal mask
        causal_mask = self._make_causal_mask(mask, (B, L), cache_len, x.dtype, x.device)

        # Initialize outputs
        all_features: Optional[Tuple[Tensor, ...]] = () if return_intermediates else None
        all_weights: Optional[Tuple[Tensor, ...]] = () if return_weights else None
        new_cache: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = () if use_cache else None

        # Forward through blocks
        for i, block in enumerate(self.blocks):
            if return_intermediates:
                all_features = all_features + (x,)

            cached_kv = kv_cache[i] if kv_cache is not None else None

            x, attn_w, present_kv = block(
                x,
                mask=causal_mask,
                positions=positions,
                cached_kv=cached_kv,
                return_weights=return_weights,
                use_cache=use_cache,
            )

            if use_cache and present_kv is not None:
                new_cache = new_cache + (present_kv,)

            if return_weights and attn_w is not None:
                all_weights = all_weights + (attn_w,)

        # Final normalization
        x = self.final_norm(x)

        if return_intermediates:
            all_features = all_features + (x,)

        return x, all_features, all_weights, new_cache if use_cache else None


class GemmaLM(nn.Module):
    """Gemma model with language modeling head.

    Complete Gemma model for text generation.

    Args:
        config: Gemma configuration.

    """

    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = GemmaDecoder(config)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer."""
        return self.decoder.token_embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the token embedding layer."""
        self.decoder.token_embed = value

    def get_output_projection(self) -> nn.Linear:
        """Get the output projection layer."""
        return self.output_proj

    def set_output_projection(self, value: nn.Linear) -> None:
        """Set the output projection layer."""
        self.output_proj = value

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> Tuple[
        Optional[Tensor],
        Tensor,
        Optional[Tuple[Tensor, ...]],
        Optional[Tuple[Tensor, ...]],
        Optional[Tuple[Tuple[Tensor, Tensor], ...]],
    ]:
        """Forward pass of Gemma LM.

        Args:
            input_ids: Token IDs of shape (B, seq_len).
            attention_mask: Padding mask of shape (B, seq_len).
            position_ids: Position indices of shape (B, seq_len).
            inputs_embeds: Pre-computed embeddings.
            labels: Labels for computing loss.
            past_key_values: Cached key-value states.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states.
            use_cache: Whether to return cache.

        Returns:
            Tuple of (loss, logits, hidden_states, attentions, cache).

        """
        outputs = self.decoder(
            token_ids=input_ids,
            mask=attention_mask,
            positions=position_ids,
            embeds=inputs_embeds,
            kv_cache=past_key_values,
            return_weights=output_attentions,
            return_intermediates=output_hidden_states,
            use_cache=use_cache,
        )

        features = outputs[0]
        logits = self.output_proj(features)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return loss, logits, outputs[1], outputs[2], outputs[3]

    def prepare_for_generation(
        self,
        token_ids: Tensor,
        kv_cache: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> dict:
        """Prepare inputs for generation step.

        Args:
            token_ids: Current token IDs.
            kv_cache: Cached states from previous steps.
            mask: Attention mask.
            **kwargs: Additional arguments.

        Returns:
            Dict of model inputs.

        """
        if kv_cache is not None:
            # Only use the last token
            token_ids = token_ids[:, -1:]

        positions = kwargs.get("position_ids", None)
        if mask is not None and positions is None:
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 1)
            if kv_cache is not None:
                positions = positions[:, -1].unsqueeze(-1)

        return {
            "input_ids": token_ids,
            "past_key_values": kv_cache,
            "attention_mask": mask,
            "position_ids": positions,
            "use_cache": True,
        }


# Aliases for compatibility
GemmaModel = GemmaDecoder
GemmaForCausalLM = GemmaLM

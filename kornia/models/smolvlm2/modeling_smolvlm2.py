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

"""Native PyTorch implementation of the SmolVLM2 vision-language model.

SmolVLM2 combines a SigLIP vision encoder with a SmolLM2 (Llama-architecture)
text decoder. Image features from the vision tower are downsampled by a
pixel-shuffle connector, projected into the text embedding space, and spliced
into the token-embedding sequence at ``<image>`` placeholder positions before
the decoder runs.

References:
    - ``transformers.models.smolvlm.modeling_smolvlm`` (connector, inputs_merger).
    - ``transformers.models.llama.modeling_llama`` (RMSNorm, RoPE, GQA attention,
      SwiGLU MLP) for the SmolLM2 text decoder.

The vision tower reuses Kornia's :class:`~kornia.models.siglip2.vision_encoder.SigLip2VisionModel`
(as :class:`~kornia.models.paligemma.modeling_paligemma.PaliGemma` does).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.models.siglip2.vision_encoder import SigLip2VisionModel

from .configuration_smolvlm2 import SmolVLM2Config, SmolVLM2TextConfig

# ---------------------------------------------------------------------------
# Connector (pixel shuffle + projection)
# ---------------------------------------------------------------------------


class SmolVLM2SimpleMLP(nn.Module):
    """Linear projection from pixel-shuffled vision features to the text space.

    The input width is ``vision_hidden_size * scale_factor ** 2`` because the
    pixel shuffle folds ``scale_factor ** 2`` neighbouring patches into the
    channel dimension.
    """

    def __init__(self, config: SmolVLM2Config) -> None:
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features of shape ``(B, N, vision_hidden * scale_factor**2)`` to text hidden size."""
        return self.proj(x)


class SmolVLM2Connector(nn.Module):
    """Pixel-shuffle connector mapping vision tokens to text-space embeddings.

    Mirrors ``SmolVLMConnector`` in HuggingFace ``transformers``.
    """

    def __init__(self, config: SmolVLM2Config) -> None:
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = SmolVLM2SimpleMLP(config)

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
        r"""Space-to-depth shuffle reducing token count by ``scale_factor ** 2``.

        Args:
            x: Vision tokens with shape :math:`(B, N, D)` where :math:`N` is a
                perfect square (the flattened square patch grid).
            scale_factor: Spatial downsampling factor along each side.

        Returns:
            Tensor with shape :math:`(B, N / s^2, D \\cdot s^2)`.
        """
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        """Downsample vision tokens and project them into the text embedding space."""
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


# ---------------------------------------------------------------------------
# Text decoder (SmolLM2 == Llama architecture)
# ---------------------------------------------------------------------------


class SmolVLM2RMSNorm(nn.Module):
    """Llama-style RMS normalization (no unit offset)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x`` by its root-mean-square and scale."""
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dtype)


class SmolVLM2RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) for the text decoder."""

    def __init__(self, dim: int, base: float = 130000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return ``(cos, sin)`` rotary factors for the given positions.

        Args:
            x: Tensor whose dtype the output is cast to.
            position_ids: Position ids of shape :math:`(B, N)`.

        Returns:
            Tuple of tensors, each with shape :math:`(B, N, \\text{dim})`.
        """
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: ``[x1, x2] -> [-x2, x1]``."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class SmolVLM2TextMLP(nn.Module):
    """SwiGLU feed-forward network used by the SmolLM2 decoder."""

    def __init__(self, config: SmolVLM2TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``down(silu(gate(x)) * up(x))``."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads for grouped-query attention.

    ``(B, num_kv_heads, N, head_dim) -> (B, num_kv_heads * n_rep, N, head_dim)``.
    """
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


class SmolVLM2TextAttention(nn.Module):
    """Grouped-query self-attention with rotary embeddings and SDPA."""

    def __init__(self, config: SmolVLM2TextConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Run grouped-query attention over ``hidden_states`` of shape :math:`(B, N, D)`.

        Args:
            hidden_states: Input tensor of shape :math:`(B, N, D)`.
            cos: Rotary cosine factors of shape :math:`(B, N, \\text{head\\_dim})`.
            sin: Rotary sine factors of shape :math:`(B, N, \\text{head\\_dim})`.
            attention_mask: Optional boolean mask of shape
                :math:`(B, 1, N, N)` where ``True`` means *attend*. It must
                already include causality (see
                :meth:`SmolVLM2TextModel._build_attention_mask`). When ``None``,
                the fast fused causal path of scaled dot-product attention is
                used.

        Returns:
            Tensor of shape :math:`(B, N, D)`.
        """
        bsz, q_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)

        is_causal = attention_mask is None and q_len > 1
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class SmolVLM2TextDecoderLayer(nn.Module):
    """A single SmolLM2 (Llama) decoder layer with pre-normalization."""

    def __init__(self, config: SmolVLM2TextConfig) -> None:
        super().__init__()
        self.self_attn = SmolVLM2TextAttention(config)
        self.mlp = SmolVLM2TextMLP(config)
        self.input_layernorm = SmolVLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SmolVLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply pre-norm self-attention and MLP sub-blocks with residual connections."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SmolVLM2TextModel(nn.Module):
    """SmolLM2 (Llama) text decoder operating on token ids or input embeddings."""

    def __init__(self, config: SmolVLM2TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([SmolVLM2TextDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = SmolVLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SmolVLM2RotaryEmbedding(config.head_dim, base=config.rope_theta)

    def _build_attention_mask(
        self, attention_mask: Optional[torch.Tensor], inputs_embeds: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Combine the causal mask with an optional padding mask.

        Args:
            attention_mask: ``None``, a padding mask of shape :math:`(B, N)`
                (``1`` = attend, ``0`` = padding), or a pre-built boolean mask
                of shape :math:`(B, 1, N, N)` which is passed through unchanged.
            inputs_embeds: Token embeddings of shape :math:`(B, N, D)`, used
                for shape and device.

        Returns:
            ``None`` when no padding mask is given (callers then rely on the
            fused ``is_causal`` path), otherwise a boolean mask of shape
            :math:`(B, 1, N, N)` where ``True`` means *attend*.
        """
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return attention_mask
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must have 2 or 4 dimensions, got {attention_mask.dim()}")
        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        allowed = causal[None, None, :, :] & attention_mask[:, None, None, :].to(torch.bool)
        # Padded query rows would be fully masked and produce NaNs in SDPA;
        # let every position attend to itself (their outputs are ignored anyway).
        idx = torch.arange(seq_len, device=device)
        allowed[:, :, idx, idx] = True
        return allowed

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the final hidden states of shape :math:`(B, N, D)`.

        Exactly one of ``input_ids`` or ``inputs_embeds`` must be provided.
        Attention is always causal; ``attention_mask`` is a padding mask of
        shape :math:`(B, N)` (``1`` = attend) or a pre-built :math:`(B, 1, N, N)`
        boolean mask.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        mask = self._build_attention_mask(attention_mask, inputs_embeds)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask=mask)
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class SmolVLM2Model(nn.Module):
    """SmolVLM2 base model: SigLIP vision tower + connector + SmolLM2 decoder.

    Returns the decoder's last hidden state. Use
    :class:`SmolVLM2ForConditionalGeneration` for token logits.
    """

    def __init__(self, config: SmolVLM2Config) -> None:
        super().__init__()
        self.config = config
        self.image_token_id = config.image_token_id

        self.vision_model = SigLip2VisionModel(config.vision_config)
        self.connector = SmolVLM2Connector(config)
        self.text_model = SmolVLM2TextModel(config.text_config)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        r"""Encode images into text-space embeddings.

        Args:
            pixel_values: Images with shape :math:`(B, 3, H, W)`.

        Returns:
            Projected image tokens with shape
            :math:`(B, N / s^2, \\text{text\\_hidden})`.
        """
        # SigLip2VisionModel returns (pooled_output, last_hidden_state).
        last_hidden_state = self.vision_model(pixel_values)[1]
        return self.connector(last_hidden_state)

    def inputs_merger(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        r"""Replace ``<image>`` token embeddings with projected image features.

        Args:
            input_ids: Token ids of shape :math:`(B, N)`.
            inputs_embeds: Token embeddings of shape :math:`(B, N, D)`.
            image_hidden_states: Projected image tokens of shape
                :math:`(\\text{num\\_images}, \\text{img\\_seq}, D)`.

        Returns:
            Merged embeddings of shape :math:`(B, N, D)`.
        """
        image_mask = input_ids == self.image_token_id
        num_image_tokens = int(image_mask.sum().item())
        if num_image_tokens == 0:
            return inputs_embeds
        flat_image_embeds = image_hidden_states.reshape(-1, image_hidden_states.shape[-1])
        if flat_image_embeds.shape[0] != num_image_tokens:
            raise ValueError(
                f"Number of <image> tokens ({num_image_tokens}) does not match the number of image "
                f"feature vectors ({flat_image_embeds.shape[0]})."
            )
        merged = inputs_embeds.clone()
        merged[image_mask] = flat_image_embeds.to(merged.dtype)
        return merged

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Encode an interleaved image/text batch and run the decoder.

        Args:
            input_ids: Token ids of shape :math:`(B, N)` containing ``<image>``
                placeholders where image features should be inserted.
            pixel_values: Images of shape :math:`(\\text{num\\_images}, 3, H, W)`.
            attention_mask: Optional padding mask of shape :math:`(B, N)`
                (``1`` = attend); causality is applied internally.
            position_ids: Optional position ids of shape :math:`(B, N)`.

        Returns:
            Decoder last hidden state of shape :math:`(B, N, D)`.
        """
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values)
            inputs_embeds = self.inputs_merger(input_ids, inputs_embeds, image_hidden_states)
        return self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


class SmolVLM2ForConditionalGeneration(nn.Module):
    """SmolVLM2 with a language-modeling head producing vocabulary logits."""

    def __init__(self, config: SmolVLM2Config) -> None:
        super().__init__()
        self.config = config
        self.model = SmolVLM2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Return next-token logits of shape :math:`(B, N, \\text{vocab\\_size})`."""
        hidden_states = self.model(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.lm_head(hidden_states)


# Backwards-compatible alias for the previous scaffold symbol.
SmolVLM2 = SmolVLM2ForConditionalGeneration

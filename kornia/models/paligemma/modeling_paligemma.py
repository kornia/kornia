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

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.models.siglip2.vision_encoder import SigLip2VisionModel

from .configuration_paligemma import PaliGemmaConfig


class GemmaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * (1.0 + self.weight)


class GemmaRotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Positional Embedding to query and key states."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaMLP(nn.Module):
    """Multi-Layer Perceptron implementing the GeGLU pattern."""

    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # FIX: Gemma uses 'tanh' approximation for GELU
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GemmaAttention(nn.Module):
    """Multi-headed attention with RoPE and SDPA."""

    def __init__(self, config: PaliGemmaConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=int(self.rope_theta),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            raise ValueError("position_ids cannot be None for GemmaAttention")

        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_key_value_groups)

        # FIX: Ensure causal masking (prevent peeking at future tokens)
        is_causal = True if attention_mask is None else False

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class GemmaDecoderLayer(nn.Module):
    """A single layer of the Gemma Decoder."""

    def __init__(self, config: PaliGemmaConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=1e-6)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class PaliGemma(nn.Module):
    """PaliGemma Model for Vision-Language tasks.

    This model combines a SigLip2 Vision Encoder with a Gemma Language Decoder.
    """

    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.ignore_index
        self.vocab_size = config.vocab_size

        if config.vision_config is None:
            raise ValueError("vision_config cannot be None")
        self.vision_tower = SigLip2VisionModel(config.vision_config)

        self.multi_modal_projector = nn.Linear(config.vision_config.hidden_size, config.hidden_size)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)

        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=1e-6)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            input_ids: Text tokens (batch, input_seq_len)
            pixel_values: Images (batch, channels, height, width)
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.

        Returns:
            logits: Prediction scores (batch, total_seq_len, vocab_size).
        """
        vision_outputs = self.vision_tower(pixel_values)

        if isinstance(vision_outputs, (tuple, list)):
            image_features = vision_outputs[1]
        else:
            image_features = vision_outputs

        if image_features.dim() != 3:
            image_features = image_features.unsqueeze(1)

        image_features = self.multi_modal_projector(image_features)

        inputs_embeds = self.embed_tokens(input_ids)

        # 🔥 FIX: Scale embeddings by sqrt(hidden_size)
        # This is a critical Gemma/PaliGemma specific scaling factor
        inputs_embeds = inputs_embeds * (self.config.hidden_size**0.5)

        # --- Handle Placeholder Token Duplication ---
        num_images = image_features.shape[1]

        # If input has more tokens than images, we assume placeholders are at the start
        if inputs_embeds.shape[1] > num_images:
            inputs_embeds = inputs_embeds[:, num_images:]
        # --------------------------------------------------

        inputs_embeds = torch.cat([image_features, inputs_embeds], dim=1)

        if position_ids is None:
            seq_length = inputs_embeds.shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    @classmethod
    def from_pretrained(cls, model_id: str = "google/paligemma-3b-pt-224", token: Optional[str] = None) -> PaliGemma:
        """Load pretrained weights from Hugging Face."""
        try:
            from transformers import PaliGemmaForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "The 'transformers' library is required to load pretrained weights. "
                "Please install it using: pip install transformers"
            ) from e

        hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, device_map="cpu", token=token, torch_dtype=torch.float32
        )

        if not hasattr(hf_model, "config") or not hasattr(hf_model.config, "vision_config"):
            raise ValueError(f"The model {model_id} does not seem to have a valid PaliGemma configuration.")

        config = PaliGemmaConfig()
        text_conf = hf_model.config.text_config
        vis_conf = hf_model.config.vision_config

        config.vocab_size = text_conf.vocab_size
        config.hidden_size = text_conf.hidden_size
        config.num_hidden_layers = text_conf.num_hidden_layers
        config.num_attention_heads = text_conf.num_attention_heads
        config.intermediate_size = text_conf.intermediate_size
        config.num_key_value_heads = text_conf.num_key_value_heads

        config.vision_config.image_size = vis_conf.image_size
        config.vision_config.patch_size = vis_conf.patch_size
        config.vision_config.hidden_size = vis_conf.hidden_size
        config.vision_config.num_hidden_layers = vis_conf.num_hidden_layers
        config.vision_config.num_attention_heads = vis_conf.num_attention_heads
        config.vision_config.intermediate_size = vis_conf.intermediate_size

        kornia_model = cls(config)
        kornia_sd = kornia_model.state_dict()
        hf_sd = hf_model.state_dict()

        missing_keys: List[str] = []

        for k_key in kornia_sd.keys():
            hf_key = None

            if k_key.startswith("vision_tower."):
                if "vision_tower.head" in k_key:
                    continue
                suffix = k_key.replace("vision_tower.", "")
                if "embeddings.position_embedding" in suffix:
                    hf_key = f"model.vision_tower.vision_model.{suffix}.weight"
                    if hf_key not in hf_sd:
                        hf_key = f"model.vision_tower.vision_model.{suffix}"
                else:
                    hf_key = f"model.vision_tower.vision_model.{suffix}"

            elif k_key.startswith("multi_modal_projector."):
                suffix = k_key.replace("multi_modal_projector.", "")
                hf_key = f"model.multi_modal_projector.linear.{suffix}"

            elif k_key.startswith("embed_tokens.") or k_key.startswith("layers.") or k_key.startswith("norm."):
                hf_key = f"model.language_model.{k_key}"

            elif k_key == "lm_head.weight":
                hf_key = "lm_head.weight"

            if hf_key and hf_key in hf_sd:
                with torch.no_grad():
                    if kornia_sd[k_key].shape == hf_sd[hf_key].shape:
                        kornia_sd[k_key].copy_(hf_sd[hf_key])
                    else:
                        missing_keys.append(
                            f"{k_key} (Shape mismatch: {kornia_sd[k_key].shape} vs {hf_sd[hf_key].shape})"
                        )
            else:
                missing_keys.append(k_key)

        if len(missing_keys) > 0:
            if len(missing_keys) > 20:
                print(f"Warning: {len(missing_keys)} keys were not loaded. This might indicate a mapping issue.")

        return kornia_model

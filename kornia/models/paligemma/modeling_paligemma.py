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

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Note: Ensure sys.path includes the root of kornia repo if running from weird paths
from kornia.models.siglip2.vision_encoder import SigLip2VisionModel

from .configuration_paligemma import PaliGemmaConfig

logger = logging.getLogger(__name__)


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

        # PaliGemma is a decoder-only model, so it should be causal by default.
        # If attention_mask is provided, we use it. If not, we set is_causal=True.
        is_causal = attention_mask is None

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
    """PaliGemma Model for Vision-Language tasks."""

    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.ignore_index
        self.vocab_size = config.vocab_size

        if config.vision_config is None:
            raise ValueError("vision_config cannot be None")

        # Vision Tower
        self.vision_tower = SigLip2VisionModel(config.vision_config)

        # Projector & Embeddings
        self.multi_modal_projector = nn.Linear(config.vision_config.hidden_size, config.hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)

        # Decoder Layers
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
            input_ids: Torch tensor with shape :math:`(B, L)` containing the text token IDs.
            pixel_values: Torch tensor with shape :math:`(B, 3, H, W)` containing the image pixels.
            attention_mask: Optional attention mask with shape :math:`(B, L_{total})`.
            position_ids: Optional position IDs with shape :math:`(B, L_{total})`.

        Returns:
            Logits tensor with shape :math:`(B, L_{total}, V)`, where V is the vocab size.
        """
        # 1. Vision Forward
        vision_outputs = self.vision_tower(pixel_values)
        if isinstance(vision_outputs, (tuple, list)):
            image_features = vision_outputs[1]
        else:
            image_features = vision_outputs

        if image_features.dim() != 3:
            image_features = image_features.unsqueeze(1)

        # 2. Projection
        image_features = self.multi_modal_projector(image_features)

        # 3. Text Embeddings (Scaled)
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds * (self.config.hidden_size**0.5)

        # 4. Robust Token Replacement
        # Detect where image tokens are and replace them with projected image features.
        # This handles prompts like "<image><bos>..." or other variations.
        image_token_mask = input_ids == self.config.image_token_index

        if image_token_mask.any():
            # Ensure the number of image tokens matches the features we have
            num_image_tokens = image_token_mask.sum().item()
            num_features = image_features.shape[0] * image_features.shape[1]
            if num_image_tokens == num_features:
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_token_mask] = image_features.view(-1, image_features.shape[-1])
            else:
                logger.warning(
                    f"Number of image tokens ({num_image_tokens}) does not match "
                    f"number of image features ({num_features}). Skipping replacement."
                )

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
    def from_pretrained(
        cls,
        model_id: str = "google/paligemma-3b-pt-224",
        token: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> PaliGemma:
        """Load pretrained weights from Hugging Face.

        Args:
            model_id: The model ID on Hugging Face Hub.
            token: The HF token.
            device: The device to load the model onto. If None, defaults to CPU.
                    Loading to CPU first might require significant RAM (approx 12GB+ for 3B model).
                    Specifying a GPU device (e.g. "cuda") is recommended for faster loading and less system RAM usage.

        Returns:
            The loaded PaliGemma model.
        """
        try:
            from transformers import PaliGemmaForConditionalGeneration
        except ImportError as e:
            raise ImportError("Transformers library is required.") from e

        # Load HF Model
        # output_loading_info=False to avoid clutter, but we can use it if debugging is needed.
        try:
            hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, device_map=device, token=token, torch_dtype=torch.float32
            )
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                raise RuntimeError(
                    f"Access to model '{model_id}' is gated. Please ensure you have "
                    "accepted the terms on HuggingFace and are logged in using `huggingface-cli login` "
                    "or provide a valid `token`."
                ) from e
            raise e

        config = PaliGemmaConfig()
        text_conf = hf_model.config.text_config
        vis_conf = hf_model.config.vision_config

        # Copy Configs
        config.vocab_size = text_conf.vocab_size
        config.hidden_size = text_conf.hidden_size
        config.num_hidden_layers = text_conf.num_hidden_layers
        config.num_attention_heads = text_conf.num_attention_heads
        config.head_dim = text_conf.head_dim
        config.intermediate_size = text_conf.intermediate_size
        config.num_key_value_heads = text_conf.num_key_value_heads
        config.vision_config.image_size = vis_conf.image_size
        config.vision_config.patch_size = vis_conf.patch_size
        config.vision_config.hidden_size = vis_conf.hidden_size
        config.vision_config.num_hidden_layers = vis_conf.num_hidden_layers
        config.vision_config.num_attention_heads = vis_conf.num_attention_heads
        config.vision_config.intermediate_size = vis_conf.intermediate_size

        kornia_model = cls(config)

        # If device was specified, move kornia model there before loading weights
        if device is not None:
            kornia_model = kornia_model.to(device)

        kornia_sd = kornia_model.state_dict()
        hf_sd = hf_model.state_dict()

        logger.info(f"Loading weights from {model_id}...")

        # ---------------------------------------------------------------------
        # Explicit Weight Mapping
        # ---------------------------------------------------------------------
        # We manually map Kornia keys to HF keys to ensure correctness and avoid
        # accidental mismatches from suffix matching.

        mapping_rules = {
            # --- Vision Tower ---
            "vision_tower.embeddings.patch_embedding.weight": (
                "model.vision_tower.vision_model.embeddings.patch_embedding.weight"
            ),
            "vision_tower.embeddings.patch_embedding.bias": (
                "model.vision_tower.vision_model.embeddings.patch_embedding.bias"
            ),
            "vision_tower.embeddings.position_embedding": (
                "model.vision_tower.vision_model.embeddings.position_embedding.weight"
            ),
            "vision_tower.post_layernorm.weight": "model.vision_tower.vision_model.post_layernorm.weight",
            "vision_tower.post_layernorm.bias": "model.vision_tower.vision_model.post_layernorm.bias",
            # --- Projector ---
            "multi_modal_projector.weight": "model.multi_modal_projector.linear.weight",
            "multi_modal_projector.bias": "model.multi_modal_projector.linear.bias",
            # --- Language Model ---
            "embed_tokens.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        # Map Vision Layers
        for i in range(config.vision_config.num_hidden_layers):
            k_prefix = f"vision_tower.encoder.layers.{i}"
            hf_prefix = f"model.vision_tower.vision_model.encoder.layers.{i}"

            mapping_rules[f"{k_prefix}.self_attn.k_proj.weight"] = f"{hf_prefix}.self_attn.k_proj.weight"
            mapping_rules[f"{k_prefix}.self_attn.v_proj.weight"] = f"{hf_prefix}.self_attn.v_proj.weight"
            mapping_rules[f"{k_prefix}.self_attn.q_proj.weight"] = f"{hf_prefix}.self_attn.q_proj.weight"
            mapping_rules[f"{k_prefix}.self_attn.out_proj.weight"] = f"{hf_prefix}.self_attn.out_proj.weight"
            mapping_rules[f"{k_prefix}.self_attn.k_proj.bias"] = f"{hf_prefix}.self_attn.k_proj.bias"
            mapping_rules[f"{k_prefix}.self_attn.v_proj.bias"] = f"{hf_prefix}.self_attn.v_proj.bias"
            mapping_rules[f"{k_prefix}.self_attn.q_proj.bias"] = f"{hf_prefix}.self_attn.q_proj.bias"
            mapping_rules[f"{k_prefix}.self_attn.out_proj.bias"] = f"{hf_prefix}.self_attn.out_proj.bias"

            mapping_rules[f"{k_prefix}.layer_norm1.weight"] = f"{hf_prefix}.layer_norm1.weight"
            mapping_rules[f"{k_prefix}.layer_norm1.bias"] = f"{hf_prefix}.layer_norm1.bias"
            mapping_rules[f"{k_prefix}.layer_norm2.weight"] = f"{hf_prefix}.layer_norm2.weight"
            mapping_rules[f"{k_prefix}.layer_norm2.bias"] = f"{hf_prefix}.layer_norm2.bias"

            mapping_rules[f"{k_prefix}.mlp.fc1.weight"] = f"{hf_prefix}.mlp.fc1.weight"
            mapping_rules[f"{k_prefix}.mlp.fc1.bias"] = f"{hf_prefix}.mlp.fc1.bias"
            mapping_rules[f"{k_prefix}.mlp.fc2.weight"] = f"{hf_prefix}.mlp.fc2.weight"
            mapping_rules[f"{k_prefix}.mlp.fc2.bias"] = f"{hf_prefix}.mlp.fc2.bias"

        # Map Text Layers
        for i in range(config.num_hidden_layers):
            k_prefix = f"layers.{i}"
            hf_prefix = f"model.layers.{i}"

            mapping_rules[f"{k_prefix}.self_attn.q_proj.weight"] = f"{hf_prefix}.self_attn.q_proj.weight"
            mapping_rules[f"{k_prefix}.self_attn.k_proj.weight"] = f"{hf_prefix}.self_attn.k_proj.weight"
            mapping_rules[f"{k_prefix}.self_attn.v_proj.weight"] = f"{hf_prefix}.self_attn.v_proj.weight"
            mapping_rules[f"{k_prefix}.self_attn.o_proj.weight"] = f"{hf_prefix}.self_attn.o_proj.weight"

            mapping_rules[f"{k_prefix}.mlp.gate_proj.weight"] = f"{hf_prefix}.mlp.gate_proj.weight"
            mapping_rules[f"{k_prefix}.mlp.up_proj.weight"] = f"{hf_prefix}.mlp.up_proj.weight"
            mapping_rules[f"{k_prefix}.mlp.down_proj.weight"] = f"{hf_prefix}.mlp.down_proj.weight"

            mapping_rules[f"{k_prefix}.input_layernorm.weight"] = f"{hf_prefix}.input_layernorm.weight"
            mapping_rules[f"{k_prefix}.post_attention_layernorm.weight"] = (
                f"{hf_prefix}.post_attention_layernorm.weight"
            )

        # Apply Mapping
        missing_keys = []

        for k_key, hf_key in mapping_rules.items():
            if k_key not in kornia_sd:
                # mismatch in manual map vs actual model structure
                continue

            if hf_key not in hf_sd:
                # Key might be missing in HF model (unlikely for matched config)
                missing_keys.append(hf_key)
                continue

            with torch.no_grad():
                if kornia_sd[k_key].shape != hf_sd[hf_key].shape:
                    logger.warning(
                        f"Shape mismatch: {k_key} {kornia_sd[k_key].shape} vs {hf_key} {hf_sd[hf_key].shape}"
                    )
                else:
                    kornia_sd[k_key].copy_(hf_sd[hf_key])

        logger.info("Weights loaded successfully.")
        return kornia_model

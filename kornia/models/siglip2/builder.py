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

"""Builder for SigLip2 models."""

from __future__ import annotations

import json
import logging
from typing import Optional

import torch

from .config import SigLip2Config
from .model import SigLip2Model

logger = logging.getLogger(__name__)

__all__ = ["SigLip2Builder"]


def _map_hf_to_kornia_weight_name(hf_name: str) -> str:
    """Map HuggingFace weight name to Kornia format.

    Our implementation matches HF structure, so most names are identical.
    Only need to handle minor naming differences.

    Args:
        hf_name: HuggingFace weight name.

    Returns:
        Kornia weight name.
    """
    # Remove 'model.' prefix if present (from Siglip2Model wrapper)
    if hf_name.startswith("model."):
        hf_name = hf_name[6:]

    # Handle projection layer naming difference
    if hf_name.startswith("visual_projection"):
        return hf_name.replace("visual_projection", "vision_projection")

    # All other names should match directly
    return hf_name


def _handle_special_components(
    hf_name: str, kornia_name: str, tensor: torch.Tensor, kornia_state_dict: dict[str, torch.Tensor]
) -> bool:
    """Handle special components that need early return.

    Returns True if handled (should continue), False otherwise.
    """
    # Handle vision_model.head components (MultiheadAttention pooling head)
    if "vision_model.head." in hf_name or "text_model.head." in hf_name:
        kornia_state_dict[kornia_name] = tensor
        return True

    # Handle vision_model.post_layernorm
    if "vision_model.post_layernorm" in hf_name:
        kornia_state_dict[kornia_name] = tensor
        return True

    # Handle text_model.final_layer_norm (applied in text_model forward, not encoder)
    if "text_model.final_layer_norm" in hf_name:
        kornia_name = kornia_name.replace("final_layer_norm", "encoder.layer_norm")
        kornia_state_dict[kornia_name] = tensor
        return True

    # Handle logit_bias
    if "logit_bias" in hf_name:
        kornia_state_dict[kornia_name] = tensor
        return True

    return False


def _handle_position_embeddings(kornia_name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Handle position embedding naming and shape differences."""
    # Vision: singular -> plural, and remove .weight (it's a Parameter, not Embedding)
    if "vision_model" in kornia_name and "position_embedding.weight" in kornia_name:
        kornia_name = kornia_name.replace("position_embedding.weight", "position_embeddings")
        # Add batch dimension: [num_patches, hidden_size] -> [1, num_patches, hidden_size]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
    return kornia_name, tensor


def _extract_qkv_component(kornia_name: str, tensor: torch.Tensor) -> tuple[str, str, str] | None:
    """Extract QKV component info from weight name.

    Returns (prefix, component, suffix) if it's a QKV projection, None otherwise.
    """
    if not (
        ".attention.q_proj." in kornia_name
        or ".attention.k_proj." in kornia_name
        or ".attention.v_proj." in kornia_name
    ):
        return None

    parts = kornia_name.split(".")
    if "q_proj" in kornia_name:
        component = "q"
    elif "k_proj" in kornia_name:
        component = "k"
    else:  # v_proj
        component = "v"

    # Find the layer index
    layer_idx = None
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            layer_idx = parts[i + 1]
            break

    if layer_idx is None:
        return None

    # Build prefix
    prefix_parts = []
    for _i, part in enumerate(parts):
        if part == "layers":
            prefix_parts.append(part)
            prefix_parts.append(layer_idx)
            prefix_parts.append("attention")
            break
        prefix_parts.append(part)

    prefix = ".".join(prefix_parts)
    suffix = ".weight" if ".weight" in kornia_name else ".bias"
    return (prefix, component, suffix)


def _combine_qkv_weights(qkv_weights: dict[tuple[str, str, str], torch.Tensor]) -> dict[str, torch.Tensor]:
    """Combine separate Q/K/V projections into fused QKV projection."""
    qkv_dict = {}
    processed = set()

    for (prefix, _component, suffix), _tensor in qkv_weights.items():
        if (prefix, suffix) in processed:
            continue

        q_key = (prefix, "q", suffix)
        k_key = (prefix, "k", suffix)
        v_key = (prefix, "v", suffix)

        if q_key in qkv_weights and k_key in qkv_weights and v_key in qkv_weights:
            q_tensor = qkv_weights[q_key]
            k_tensor = qkv_weights[k_key]
            v_tensor = qkv_weights[v_key]

            # Concatenate along the first dimension (output features)
            qkv_tensor = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

            # Build the final key
            kornia_key = prefix + ".qkv_proj" + suffix
            qkv_dict[kornia_key] = qkv_tensor
            processed.add((prefix, suffix))

    return qkv_dict


def _load_hf_state_dict(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load and map HuggingFace state dict to Kornia format.

    Handles the conversion from HF's separate q/k/v projections to our fused qkv_proj,
    and other naming differences.

    Args:
        hf_state_dict: HuggingFace state dict.

    Returns:
        Kornia state dict.
    """
    kornia_state_dict = {}
    qkv_weights = {}  # (kornia_key_prefix, component) -> tensor

    for hf_name, tensor in hf_state_dict.items():
        # Map basic naming differences first
        kornia_name = _map_hf_to_kornia_weight_name(hf_name)

        # Handle special components (early return cases)
        if _handle_special_components(hf_name, kornia_name, tensor, kornia_state_dict):
            continue

        # Handle position embeddings
        kornia_name, tensor = _handle_position_embeddings(kornia_name, tensor)

        # Handle self_attn -> attention
        if ".self_attn." in kornia_name:
            kornia_name = kornia_name.replace(".self_attn.", ".attention.")

        # Collect q/k/v projections for later fusion
        qkv_info = _extract_qkv_component(kornia_name, tensor)
        if qkv_info is not None:
            prefix, component, suffix = qkv_info
            qkv_weights[(prefix, component, suffix)] = tensor
            continue

        # Handle out_proj -> out_proj (same name)
        if ".attention.out_proj." in kornia_name:
            kornia_state_dict[kornia_name] = tensor
            continue

        # Skip embeddings layer_norm if not in checkpoint
        if "embeddings.layer_norm" in kornia_name and kornia_name not in hf_state_dict:
            continue

        # All other weights map directly (after basic name mapping)
        kornia_state_dict[kornia_name] = tensor

    # Combine q/k/v into qkv_proj
    qkv_dict = _combine_qkv_weights(qkv_weights)
    kornia_state_dict.update(qkv_dict)

    # Handle logit_scale shape difference (checkpoint has [1], model expects [])
    if "logit_scale" in kornia_state_dict:
        logit_scale = kornia_state_dict["logit_scale"]
        if logit_scale.dim() > 0:
            kornia_state_dict["logit_scale"] = logit_scale.squeeze()

    # Handle logit_bias shape difference (checkpoint might have [1], model expects [])
    if "logit_bias" in kornia_state_dict:
        logit_bias = kornia_state_dict["logit_bias"]
        if logit_bias.dim() > 0:
            kornia_state_dict["logit_bias"] = logit_bias.squeeze()

    return kornia_state_dict


def _load_and_update_config(config: SigLip2Config, model_name: str, cache_dir: Optional[str]) -> SigLip2Config:
    """Load and update config from HuggingFace config.json."""
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_name, filename="config.json", cache_dir=cache_dir)
        with open(config_path) as f:
            hf_config = json.load(f)

        # Update vision config
        if hf_config.get("vision_config"):
            vision_hf = hf_config["vision_config"]
            if vision_hf.get("image_size") is not None:
                config.vision_config.image_size = vision_hf["image_size"]
            if vision_hf.get("patch_size") is not None:
                config.vision_config.patch_size = vision_hf["patch_size"]
            if vision_hf.get("hidden_size") is not None:
                config.vision_config.hidden_size = vision_hf["hidden_size"]
            if vision_hf.get("num_hidden_layers") is not None:
                config.vision_config.num_hidden_layers = vision_hf["num_hidden_layers"]
            if vision_hf.get("num_attention_heads") is not None:
                config.vision_config.num_attention_heads = vision_hf["num_attention_heads"]
            if vision_hf.get("intermediate_size") is not None:
                config.vision_config.intermediate_size = vision_hf["intermediate_size"]

        # Update text config
        if hf_config.get("text_config"):
            text_hf = hf_config["text_config"]
            if text_hf.get("vocab_size") is not None:
                config.text_config.vocab_size = text_hf["vocab_size"]
            if text_hf.get("hidden_size") is not None:
                config.text_config.hidden_size = text_hf["hidden_size"]
            if text_hf.get("num_hidden_layers") is not None:
                config.text_config.num_hidden_layers = text_hf["num_hidden_layers"]
            if text_hf.get("num_attention_heads") is not None:
                config.text_config.num_attention_heads = text_hf["num_attention_heads"]
            if text_hf.get("intermediate_size") is not None:
                config.text_config.intermediate_size = text_hf["intermediate_size"]
            if text_hf.get("max_position_embeddings") is not None:
                config.text_config.max_position_embeddings = text_hf["max_position_embeddings"]

        # Update projection dim
        if hf_config.get("projection_dim") is not None:
            config.projection_dim = hf_config["projection_dim"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
        # Config download failed or incomplete - will infer from weights
        pass

    return config


def _download_weights(model_name: str, cache_dir: Optional[str]) -> dict[str, torch.Tensor]:
    """Download model weights from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
    except ImportError as e:
        logger.error(
            "safetensors library is required for loading model weights. Install it with: pip install safetensors"
        )
        raise ImportError(
            "safetensors library is required for loading model weights. Install it with: pip install safetensors"
        ) from e

    try:
        weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors", cache_dir=cache_dir)
        state_dict = {}
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict
    except FileNotFoundError as e:
        logger.error(
            f"Could not find model.safetensors for {model_name}. The model must be available in safetensors format."
        )
        raise FileNotFoundError(
            f"Could not find model.safetensors for {model_name}. The model must be available in safetensors format."
        ) from e


def _infer_max_position_embeddings(config: SigLip2Config, state_dict: dict[str, torch.Tensor]) -> SigLip2Config:
    """Infer max_position_embeddings from checkpoint if not in config."""
    pos_emb_key = "text_model.embeddings.position_embedding.weight"
    if pos_emb_key in state_dict:
        pos_emb_size = state_dict[pos_emb_key].shape[0]
        if config.text_config.max_position_embeddings != pos_emb_size:
            config.text_config.max_position_embeddings = pos_emb_size
    return config


def _set_random_seeds() -> None:
    """Set random seeds for reproducible initialization."""
    torch.manual_seed(42)
    import random

    random.seed(42)
    import numpy as np

    _rng = np.random.default_rng(42)  # Set seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


class SigLip2Builder:
    """Builder for SigLip2 models.

    Provides convenient methods to create SigLip2 models from configs or
    load pretrained weights from HuggingFace.
    """

    @staticmethod
    def from_name(model_name: str) -> SigLip2Model:
        """Build model from model name without loading pretrained weights.

        Supports the same models as from_pretrained().

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            SigLip2Model instance with random initialization.

        Example:
            >>> from kornia.models.siglip2 import SigLip2Builder
            >>> model = SigLip2Builder.from_name("google/siglip2-base-patch16-224")
        """
        config = SigLip2Config.from_name(model_name)
        return SigLip2Model(config)

    @staticmethod
    def from_config(config: SigLip2Config) -> SigLip2Model:
        """Build model from configuration.

        Args:
            config: Model configuration.

        Returns:
            SigLip2Model instance.

        Example:
            >>> from kornia.models.siglip2 import SigLip2Builder, SigLip2Config
            >>> config = SigLip2Config()
            >>> model = SigLip2Builder.from_config(config)
        """
        return SigLip2Model(config)

    @staticmethod
    def from_pretrained(
        model_name: str = "google/siglip2-base-patch16-224",
        cache_dir: Optional[str] = None,
    ) -> SigLip2Model:
        """Load pretrained model from HuggingFace Hub.

        Downloads model weights and config from HuggingFace Hub and loads them
        using pure PyTorch (no transformers dependency).

        Supports the following models:
        - google/siglip-base-patch16-224 (V1)
        - google/siglip2-base-patch16-224
        - google/siglip2-base-patch16-256
        - google/siglip2-base-patch16-384
        - google/siglip2-base-patch16-512
        - google/siglip2-large-patch16-256
        - google/siglip2-large-patch16-384
        - google/siglip2-large-patch16-512

        Args:
            model_name: HuggingFace model identifier. Default: "google/siglip2-base-patch16-224".
            cache_dir: Optional cache directory for model files.

        Returns:
            SigLip2Model instance with pretrained weights.

        .. note::
            This method requires the `huggingface_hub` library to download files.
            Install it with: ``pip install huggingface_hub``
            For safetensors files, also install: ``pip install safetensors``

        Example:
            >>> from kornia.models.siglip2 import SigLip2Builder
            >>> model = SigLip2Builder.from_pretrained("google/siglip2-base-patch16-224")
        """
        # Check for huggingface_hub dependency
        try:
            import huggingface_hub  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "huggingface_hub library is required for downloading pretrained models. "
                "Install it with: pip install huggingface_hub"
            ) from e

        # Create config from model name (PyTorch-only)
        config = SigLip2Config.from_name(model_name)

        # Download and update config from HF
        config = _load_and_update_config(config, model_name, cache_dir)

        # Download model weights
        state_dict = _download_weights(model_name, cache_dir)

        # Infer max_position_embeddings from checkpoint if not in config
        config = _infer_max_position_embeddings(config, state_dict)

        # Set random seeds for reproducible initialization
        _set_random_seeds()

        # Create model and load weights
        model = SigLip2Model(config)
        kornia_state_dict = _load_hf_state_dict(state_dict)
        model.load_state_dict(kornia_state_dict, strict=True)

        return model

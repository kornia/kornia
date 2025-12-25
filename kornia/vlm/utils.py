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

"""Utilities for loading VLM weights from HuggingFace Hub."""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _get_hf_cache_dir(cache_dir: Optional[str] = None) -> Path:
    """Get the HuggingFace cache directory.

    Args:
        cache_dir: Optional custom cache directory.

    Returns:
        Path to the cache directory.

    """
    if cache_dir is not None:
        return Path(cache_dir)

    # Default HuggingFace cache
    return Path.home() / ".cache" / "huggingface" / "hub"


def download_hf_weights(
    model_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Download model weights from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier.
        cache_dir: Optional custom cache directory.
        token: Optional HuggingFace token for gated models.

    Returns:
        Path to the downloaded model directory.

    Raises:
        ImportError: If huggingface_hub is not installed.

    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download weights. Install with: pip install huggingface_hub"
        ) from e

    cache_path = _get_hf_cache_dir(cache_dir)

    logger.info(f"Downloading weights from {model_id}...")
    model_path = snapshot_download(
        model_id,
        cache_dir=str(cache_path),
        token=token,
        allow_patterns=["*.safetensors", "*.json", "*.model"],
    )

    return Path(model_path)


def load_safetensors(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load weights from safetensors file(s).

    Args:
        path: Path to a safetensors file or directory containing safetensors files.

    Returns:
        Dictionary of weight name to tensor.

    Raises:
        ImportError: If safetensors is not installed.

    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError("safetensors is required. Install with: pip install safetensors") from e

    path = Path(path)
    weights = {}

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.safetensors"))

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    return weights


# Weight name mappings from HuggingFace to kornia format
PALIGEMMA_WEIGHT_MAP = {
    # Vision encoder mappings
    "vision_tower.vision_model.embeddings.patch_embedding.weight": "vision_encoder.embeddings.patch_embedding.proj.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias": "vision_encoder.embeddings.patch_embedding.proj.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight": "vision_encoder.embeddings.position_embedding.weight",
    "vision_tower.vision_model.post_layernorm.weight": "vision_encoder.post_layernorm.weight",
    "vision_tower.vision_model.post_layernorm.bias": "vision_encoder.post_layernorm.bias",
    # Projector mappings
    "multi_modal_projector.linear.weight": "projector.linear.weight",
    "multi_modal_projector.linear.bias": "projector.linear.bias",
    # Language model mappings
    "language_model.model.embed_tokens.weight": "language_model.model.embed_tokens.weight",
    "language_model.model.norm.weight": "language_model.model.norm.weight",
    "language_model.lm_head.weight": "language_model.lm_head.weight",
}


def _map_vision_layer_weights(hf_key: str, layer_idx: int) -> str:
    """Map HuggingFace vision layer weight names to kornia format.

    Args:
        hf_key: HuggingFace weight key.
        layer_idx: Layer index.

    Returns:
        Mapped kornia weight key.

    """
    prefix = f"vision_encoder.encoder.layers.{layer_idx}."

    mappings = {
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.q_proj.bias": "self_attn.q_proj.bias",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.k_proj.bias": "self_attn.k_proj.bias",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.v_proj.bias": "self_attn.v_proj.bias",
        "self_attn.out_proj.weight": "self_attn.out_proj.weight",
        "self_attn.out_proj.bias": "self_attn.out_proj.bias",
        "layer_norm1.weight": "layer_norm1.weight",
        "layer_norm1.bias": "layer_norm1.bias",
        "layer_norm2.weight": "layer_norm2.weight",
        "layer_norm2.bias": "layer_norm2.bias",
        "mlp.fc1.weight": "mlp.fc1.weight",
        "mlp.fc1.bias": "mlp.fc1.bias",
        "mlp.fc2.weight": "mlp.fc2.weight",
        "mlp.fc2.bias": "mlp.fc2.bias",
    }

    for hf_suffix, kornia_suffix in mappings.items():
        if hf_key.endswith(hf_suffix):
            return prefix + kornia_suffix

    return None


def _map_language_layer_weights(hf_key: str, layer_idx: int) -> str:
    """Map HuggingFace language layer weight names to kornia format.

    Args:
        hf_key: HuggingFace weight key.
        layer_idx: Layer index.

    Returns:
        Mapped kornia weight key.

    """
    prefix = f"language_model.model.layers.{layer_idx}."

    mappings = {
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    }

    for hf_suffix, kornia_suffix in mappings.items():
        if hf_key.endswith(hf_suffix):
            return prefix + kornia_suffix

    return None


def map_hf_to_kornia_weights(hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map HuggingFace weight names to kornia format.

    Args:
        hf_weights: Dictionary of HuggingFace weights.

    Returns:
        Dictionary of weights with kornia-compatible names.

    """
    kornia_weights = {}

    for hf_key, tensor in hf_weights.items():
        # Check direct mappings first
        if hf_key in PALIGEMMA_WEIGHT_MAP:
            kornia_key = PALIGEMMA_WEIGHT_MAP[hf_key]
            kornia_weights[kornia_key] = tensor
            continue

        # Handle vision encoder layers
        if "vision_tower.vision_model.encoder.layers." in hf_key:
            # Extract layer index
            parts = hf_key.split(".")
            layer_idx = int(parts[4])
            suffix = ".".join(parts[5:])
            kornia_key = _map_vision_layer_weights(suffix, layer_idx)
            if kornia_key:
                kornia_weights[kornia_key] = tensor
                continue

        # Handle language model layers
        if "language_model.model.layers." in hf_key:
            parts = hf_key.split(".")
            layer_idx = int(parts[3])
            suffix = ".".join(parts[4:])
            kornia_key = _map_language_layer_weights(suffix, layer_idx)
            if kornia_key:
                kornia_weights[kornia_key] = tensor
                continue

        # Log unmapped weights
        logger.debug(f"Unmapped weight: {hf_key}")

    return kornia_weights


def load_paligemma_weights(
    model: nn.Module,
    model_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    strict: bool = False,
) -> None:
    """Load PaliGemma weights from HuggingFace Hub.

    Args:
        model: PaliGemma model to load weights into.
        model_id: HuggingFace model identifier.
        cache_dir: Optional custom cache directory.
        token: Optional HuggingFace token for gated models.
        strict: Whether to require all weights to be loaded.

    """
    # Download weights
    model_path = download_hf_weights(model_id, cache_dir=cache_dir, token=token)

    # Load safetensors
    hf_weights = load_safetensors(model_path)

    # Map weight names
    kornia_weights = map_hf_to_kornia_weights(hf_weights)

    # Load into model
    missing, unexpected = [], []
    model_state = model.state_dict()

    for key in model_state.keys():
        if key in kornia_weights:
            if model_state[key].shape == kornia_weights[key].shape:
                model_state[key] = kornia_weights[key]
            else:
                logger.warning(
                    f"Shape mismatch for {key}: expected {model_state[key].shape}, got {kornia_weights[key].shape}"
                )
                missing.append(key)
        else:
            missing.append(key)

    for key in kornia_weights.keys():
        if key not in model_state:
            unexpected.append(key)

    model.load_state_dict(model_state, strict=False)

    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
        logger.debug(f"Missing keys: {missing[:10]}...")

    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
        logger.debug(f"Unexpected keys: {unexpected[:10]}...")

    if strict and (missing or unexpected):
        raise RuntimeError(f"Weight loading failed: {len(missing)} missing, {len(unexpected)} unexpected")

    logger.info(f"Loaded weights from {model_id}")


def load_vision_encoder_weights(
    model: nn.Module,
    model_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    """Load only vision encoder weights from PaliGemma.

    Useful for using SigLIP standalone.

    Args:
        model: SigLIP vision model to load weights into.
        model_id: HuggingFace model identifier.
        cache_dir: Optional custom cache directory.
        token: Optional HuggingFace token for gated models.

    """
    model_path = download_hf_weights(model_id, cache_dir=cache_dir, token=token)
    hf_weights = load_safetensors(model_path)

    # Filter and map only vision weights
    vision_weights = {k: v for k, v in hf_weights.items() if "vision_tower" in k}
    kornia_weights = {}

    for hf_key, tensor in vision_weights.items():
        # Remove vision_tower prefix and map
        if "vision_tower.vision_model.embeddings.patch_embedding" in hf_key:
            kornia_key = hf_key.replace(
                "vision_tower.vision_model.embeddings.patch_embedding", "embeddings.patch_embedding.proj"
            )
            kornia_weights[kornia_key] = tensor
        elif "vision_tower.vision_model.embeddings.position_embedding" in hf_key:
            kornia_key = hf_key.replace(
                "vision_tower.vision_model.embeddings.position_embedding", "embeddings.position_embedding"
            )
            kornia_weights[kornia_key] = tensor
        elif "vision_tower.vision_model.post_layernorm" in hf_key:
            kornia_key = hf_key.replace("vision_tower.vision_model.post_layernorm", "post_layernorm")
            kornia_weights[kornia_key] = tensor
        elif "vision_tower.vision_model.encoder.layers" in hf_key:
            kornia_key = hf_key.replace("vision_tower.vision_model.encoder.layers", "encoder.layers")
            kornia_weights[kornia_key] = tensor

    model.load_state_dict(kornia_weights, strict=False)
    logger.info(f"Loaded vision encoder weights from {model_id}")

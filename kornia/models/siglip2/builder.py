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

        # Remove 'model.' prefix if present (from Siglip2Model wrapper)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                cleaned_state_dict[key[6:]] = value
            else:
                cleaned_state_dict[key] = value
        state_dict = cleaned_state_dict

        # Handle projection layer naming difference (visual_projection -> vision_projection)
        if "visual_projection.weight" in state_dict:
            state_dict["vision_projection.weight"] = state_dict.pop("visual_projection.weight")
        if "visual_projection.bias" in state_dict:
            state_dict["vision_projection.bias"] = state_dict.pop("visual_projection.bias")

        # Handle vision position embedding: HF has position_embedding.weight, we use position_embedding (Parameter)
        if "vision_model.embeddings.position_embedding.weight" in state_dict:
            state_dict["vision_model.embeddings.position_embedding"] = state_dict.pop(
                "vision_model.embeddings.position_embedding.weight"
            )

        # Create model and load weights directly (no transformation needed)
        model = SigLip2Model(config)

        # Handle only shape differences (logit_scale, logit_bias)
        if "logit_scale" in state_dict and state_dict["logit_scale"].dim() > 0:
            state_dict["logit_scale"] = state_dict["logit_scale"].squeeze()
        if "logit_bias" in state_dict and state_dict["logit_bias"].dim() > 0:
            state_dict["logit_bias"] = state_dict["logit_bias"].squeeze()

        model.load_state_dict(state_dict, strict=True)

        return model

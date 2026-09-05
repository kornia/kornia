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

import logging
from typing import Optional

import torch

from kornia.core.download import _hf_cache_file_name, download_file_from_url, hf_url
from kornia.core.safetensors import load_safetensors

from .config import SigLip2Config
from .model import SigLip2Model

_logger = logging.getLogger(__name__)

__all__ = ["SigLip2Builder"]

_WEIGHTS_FILE = "model.safetensors"


def _download_weights(model_name: str, cache_dir: Optional[str]) -> dict[str, torch.Tensor]:
    """Download the checkpoint of a HuggingFace repo and read it into a state dict.

    Args:
        model_name: full ``owner/name`` repository id on the HuggingFace Hub.
        cache_dir: directory to cache the checkpoint in, or ``None`` for torch's
            hub cache -- the same cache every other kornia checkpoint uses.

    Returns:
        The checkpoint's state dict, on the CPU.

    Raises:
        FileNotFoundError: if the downloaded checkpoint is gone by the time it is
            read.
        RuntimeError: if the checkpoint cannot be downloaded.
    """
    path = download_file_from_url(
        hf_url(model_name, _WEIGHTS_FILE),
        # Every repo calls its checkpoint ``model.safetensors``; the cache is one
        # flat directory, so the repo id has to be part of the name.
        file_name=_hf_cache_file_name(model_name, _WEIGHTS_FILE),
        model_dir=cache_dir,
    )
    try:
        return load_safetensors(path)
    except FileNotFoundError as e:
        error_msg = (
            f"Could not find model.safetensors for {model_name}. The model must be available in safetensors format."
        )
        _logger.error(error_msg)
        raise FileNotFoundError(error_msg) from e


def _infer_max_position_embeddings(config: SigLip2Config, state_dict: dict[str, torch.Tensor]) -> SigLip2Config:
    """Infer max_position_embeddings from checkpoint if not in config."""
    pos_emb_key = "text_model.embeddings.position_embedding.weight"
    if pos_emb_key in state_dict:
        pos_emb_size = state_dict[pos_emb_key].shape[0]
        if config.text_config.max_position_embeddings != pos_emb_size:
            config.text_config.max_position_embeddings = pos_emb_size
    return config


class SigLip2Builder:
    """Builder for SigLip2 models.

    Provides convenient methods to create SigLip2 models from configs or
    load pretrained weights from HuggingFace.
    """

    @staticmethod
    def from_name(model_name: str) -> SigLip2Model:
        """Build model from model name without loading pretrained weights.

        Supports the same models as from_pretrained_hf().

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            SigLip2Model instance with random initialization.
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
        """
        return SigLip2Model(config)

    @staticmethod
    def from_pretrained_hf(
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
            cache_dir: Optional cache directory for model files. Defaults to
                torch's hub cache, which is where every other kornia checkpoint
                is cached.

        Returns:
            SigLip2Model instance with pretrained weights.

        .. note::
            The repository must publish its weights as a single
            ``model.safetensors`` file at its root.
        """
        # create config from model name
        config = SigLip2Config.from_name(model_name)

        # download model weights
        state_dict = _download_weights(model_name, cache_dir)

        # infer max_position_embeddings from checkpoint if not in config
        config = _infer_max_position_embeddings(config, state_dict)

        # handle vision position embedding: HF has position_embedding.weight, we use position_embedding
        if "vision_model.embeddings.position_embedding.weight" in state_dict:
            state_dict["vision_model.embeddings.position_embedding"] = state_dict.pop(
                "vision_model.embeddings.position_embedding.weight"
            )

        # create model and load weights directly (no transformation needed)
        model = SigLip2Model(config)
        model.load_state_dict(state_dict, strict=True)

        return model

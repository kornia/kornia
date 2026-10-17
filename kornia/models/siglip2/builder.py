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

from kornia.core.download import download_hf_file
from kornia.core.safetensors import check_safetensors, load_safetensors

from .config import _SUPPORTED_MODELS, SigLip2Config
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
    # validate=: see KimiVL's builder -- a truncated cache entry is re-fetched
    # once rather than returned as a hit on every later call.
    path = download_hf_file(model_name, _WEIGHTS_FILE, model_dir=cache_dir, validate=check_safetensors)
    try:
        return load_safetensors(path)
    except FileNotFoundError as e:
        error_msg = (
            f"Could not find model.safetensors for {model_name} at {path}: the cached checkpoint disappeared "
            "between the download and the read. Delete that cache entry and retry."
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
            SigLip2Model instance with random initialization, configured by ``SigLip2Config.from_name``: a name outside
            its supported list gets the configuration read from the name's parts.

        Raises:
            ValueError: for a name outside the supported list whose architecture its parts cannot give, such as
                ``google/siglip2-so400m-patch14-384`` (see ``SigLip2Config.from_name``).
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

        Another repository, such as a fine-tune or a mirror of one of these, loads when its checkpoint has the
        architecture ``SigLip2Config.from_name`` reads from its name's parts (base unless the name contains
        ``"large"``, patch size 16, the image size in its name).

        Args:
            model_name: HuggingFace model identifier. Default: "google/siglip2-base-patch16-224".
            cache_dir: Optional cache directory for model files. Defaults to
                torch's hub cache, which is where every other kornia checkpoint
                is cached.

        Returns:
            SigLip2Model instance with pretrained weights, in evaluation mode, since it always loads weights (see
            :ref:`pretrained-models-conventions`).

        Raises:
            ValueError: before downloading, for a name outside the supported list whose architecture its parts cannot
                give (see ``SigLip2Config.from_name``).
            RuntimeError: if the checkpoint of a repository outside the supported list does not fit the configuration
                read from its name; the message gives that configuration and lists the supported names.

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
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            if model_name.lower() in _SUPPORTED_MODELS:
                raise
            vision, text = config.vision_config, config.text_config
            raise RuntimeError(
                f"The checkpoint of {model_name!r} does not fit the configuration SigLip2Config.from_name reads from "
                f"this name: hidden size {vision.hidden_size}, {vision.num_hidden_layers} layers, "
                f"{vision.num_attention_heads} heads, patch size {vision.patch_size}, image size {vision.image_size}, "
                f"vocabulary {text.vocab_size}. Supported names: {', '.join(_SUPPORTED_MODELS)}. A repository outside "
                "that list loads only when its checkpoint has the architecture read from its name (see "
                "SigLip2Config.from_name)."
            ) from e

        return model.eval()

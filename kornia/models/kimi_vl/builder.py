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

"""Builder for Kimi-VL models."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from .config import KimiVLConfig, _kimi_vl_a3b_instruct_config
from .model import KimiVLModel

logger = logging.getLogger(__name__)

__all__ = ["KimiVLBuilder"]

# TODO: Publish the converted checkpoint at this Kornia-owned repository before release.
_KIMI_VL_A3B_INSTRUCT_REPO_ID = "kornia/kimi-vl-a3b-instruct-vision"


def _download_weights(model_name: str, cache_dir: Optional[str]) -> dict[str, torch.Tensor]:
    """Download model weights from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
    except ImportError as e:
        error_msg = (
            "safetensors library is required for loading model weights. Install it with: pip install safetensors"
        )
        logger.error(error_msg)
        raise ImportError(error_msg) from e

    try:
        weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors", cache_dir=cache_dir)
        state_dict = {}
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict
    except FileNotFoundError as e:
        error_msg = (
            f"Could not find model.safetensors for {model_name}. The model must be available in safetensors format."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from e


def _load_checkpoint(checkpoint: str) -> dict[str, torch.Tensor]:
    """Load model weights from a local safetensors file."""
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Local checkpoint not found: {checkpoint}")

    try:
        from safetensors import safe_open
    except ImportError as e:
        error_msg = (
            "safetensors library is required for loading model weights. Install it with: pip install safetensors"
        )
        logger.error(error_msg)
        raise ImportError(error_msg) from e

    state_dict = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


class KimiVLBuilder:
    """Builder for Kimi-VL models.

    Provides convenient methods to create Kimi-VL models from configs or
    load pretrained weights.
    """

    @staticmethod
    def from_config(config: KimiVLConfig) -> KimiVLModel:
        """Build model from configuration.

        Args:
            config: Model configuration.

        Returns:
            KimiVLModel instance.
        """
        return KimiVLModel(config)

    @staticmethod
    def from_pretrained_hf(
        cache_dir: Optional[str] = None,
    ) -> KimiVLModel:
        """Load pretrained Kimi-VL-A3B-Instruct weights from Hugging Face Hub.

        Args:
            cache_dir: Optional Hugging Face cache directory.

        Returns:
            KimiVLModel instance with pretrained weights.

        .. note::
            Only Kimi-VL-A3B-Instruct is currently supported.
            This method requires the `huggingface_hub` library to download files.
            Install it with: ``pip install huggingface_hub``
            For safetensors files, also install: ``pip install safetensors``
        """
        # check for huggingface_hub dependency
        try:
            import huggingface_hub  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "huggingface_hub library is required for downloading pretrained models. "
                "Install it with: pip install huggingface_hub"
            ) from e

        # download model weights
        state_dict = _download_weights(_KIMI_VL_A3B_INSTRUCT_REPO_ID, cache_dir)
        return KimiVLBuilder._from_state_dict(state_dict)

    @staticmethod
    def from_checkpoint(checkpoint: str) -> KimiVLModel:
        """Load pretrained Kimi-VL weights from a local safetensors file.

        Args:
            checkpoint: Local safetensors file.

        Returns:
            KimiVLModel instance with pretrained weights.

        .. note::
            This method requires the ``safetensors`` library.
            Install it with: ``pip install safetensors``.
        """
        state_dict = _load_checkpoint(checkpoint)
        return KimiVLBuilder._from_state_dict(state_dict)

    @staticmethod
    def _from_state_dict(state_dict: dict[str, torch.Tensor]) -> KimiVLModel:
        """Build the supported Kimi-VL model and strictly load converted weights.

        Args:
            state_dict: Converted Kimi-VL-A3B-Instruct vision and projector weights.

        Returns:
            KimiVLModel instance with pretrained weights.

        Raises:
            RuntimeError: If the state dictionary is incompatible with the supported model.
        """
        config = _kimi_vl_a3b_instruct_config()
        model = KimiVLBuilder.from_config(config)
        model.load_state_dict(state_dict, strict=True)
        return model

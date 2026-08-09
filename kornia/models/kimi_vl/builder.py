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
from typing import Optional

import torch

from .config import KimiVLConfig, _kimi_vl_a3b_instruct_config
from .model import KimiVLModel

logger = logging.getLogger(__name__)

__all__ = ["KimiVLBuilder"]

_KIMI_VL_A3B_INSTRUCT_REPO_ID = "kornia/kimi-vl-a3b-instruct-vision"


def _download_weights(model_name: str, cache_dir: Optional[str]) -> dict[str, torch.Tensor]:
    """Download model weights from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
    except ImportError as e:
        error_msg = (
            "huggingface_hub and safetensors are required for loading model weights. "
            "Install them with: pip install huggingface_hub safetensors"
        )
        logger.error(error_msg)
        raise ImportError(error_msg) from e

    weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors", cache_dir=cache_dir)
    state_dict = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
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
    def from_pretrained_hf(cache_dir: Optional[str] = None) -> KimiVLModel:
        """Load pretrained Kimi-VL-A3B-Instruct vision weights from Hugging Face Hub.

        Downloads the vision encoder and projector weights of
        `moonshotai/Kimi-VL-A3B-Instruct` from the Kornia-owned safetensors
        checkpoint at https://huggingface.co/kornia/kimi-vl-a3b-instruct-vision.
        The checkpoint values are bitwise-identical to the original release
        (bf16), including the full 64x64 positional-embedding grid, which the
        model interpolates at runtime for other input resolutions.

        Args:
            cache_dir: Optional cache directory for downloaded files.

        Returns:
            KimiVLModel instance with pretrained weights.

        .. note::
            Only Kimi-VL-A3B-Instruct is currently supported. This method
            requires the `huggingface_hub` and `safetensors` libraries:
            ``pip install huggingface_hub safetensors``
        """
        state_dict = _download_weights(_KIMI_VL_A3B_INSTRUCT_REPO_ID, cache_dir)
        model = KimiVLBuilder.from_config(_kimi_vl_a3b_instruct_config())
        model.load_state_dict(state_dict, strict=True)
        return model

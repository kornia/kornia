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

import torch

from .config import KimiVLConfig, _kimi_vl_a3b_instruct_config
from .model import KimiVLModel

__all__ = ["KimiVLBuilder"]

_KIMI_VL_A3B_INSTRUCT_URL = "https://huggingface.co/TomasGuija/kimi-vl-a3b-instruct-vision/resolve/main/model.pt"


def _download_weights(url: str) -> dict[str, torch.Tensor]:
    """Download model weights using PyTorch."""
    return torch.hub.load_state_dict_from_url(url, map_location="cpu")


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
    def from_pretrained_hf() -> KimiVLModel:
        """Load pretrained Kimi-VL-A3B-Instruct weights from Hugging Face Hub.

        Returns:
            KimiVLModel instance with pretrained weights.

        .. note::
            Only Kimi-VL-A3B-Instruct is currently supported.
        """
        state_dict = _download_weights(_KIMI_VL_A3B_INSTRUCT_URL)
        model = KimiVLBuilder.from_config(_kimi_vl_a3b_instruct_config())
        model.load_state_dict(state_dict, strict=True)
        return model

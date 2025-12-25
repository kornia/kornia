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

"""Kornia VLM — Vision-Language Models implemented from scratch with PyTorch.

This subpackage provides research-friendly VLM implementations with easy access
to intermediate representations for experimentation.

Example:
    >>> from kornia.vlm import PaliGemma2
    >>> # Load pretrained model
    >>> model = PaliGemma2.from_hub("google/paligemma2-3b-pt-224")
    >>> # Extract vision features
    >>> features = model.extract_vision_features(images)
    >>> # Full forward pass with intermediate states
    >>> output = model(images, token_ids, return_intermediates=True)

"""

from . import layers, paligemma
from .base import VisionOutput, VLMBase, VLMOutput
from .config import LanguageModelConfig, VisionEncoderConfig, VLMConfig
from .paligemma import (
    GemmaConfig,
    GemmaDecoder,
    GemmaForCausalLM,
    GemmaLM,
    GemmaModel,
    PaliGemma2,
    PaliGemma2Config,
    SigLIPVisionConfig,
    SiglipVisionEncoder,
    SigLIPVisionModel,
)

__all__ = [
    # Base classes
    "VLMBase",
    "VLMConfig",
    "VLMOutput",
    "VisionOutput",
    "VisionEncoderConfig",
    "LanguageModelConfig",
    # PaliGemma
    "PaliGemma2",
    "PaliGemma2Config",
    # Siglip Vision
    "SiglipVisionEncoder",
    "SigLIPVisionConfig",
    "SigLIPVisionModel",  # Alias
    # Gemma Language Model
    "GemmaLM",
    "GemmaDecoder",
    "GemmaConfig",
    "GemmaModel",  # Alias
    "GemmaForCausalLM",  # Alias
    # Submodules
    "layers",
    "paligemma",
]

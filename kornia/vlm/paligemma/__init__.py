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

"""PaliGemma Vision-Language Model implementation."""

from .config import GemmaConfig, PaliGemma2Config, SigLIPVisionConfig
from .gemma import GemmaDecoder, GemmaForCausalLM, GemmaLM, GemmaModel, GemmaTransformerBlock
from .model import PaliGemma2, VisionTextConnector
from .processor import PaliGemmaImageProcessor, PaliGemmaProcessor, PaliGemmaTokenizer
from .siglip import (
    SiglipPatchEmbedder,
    SiglipSelfAttention,
    SiglipTransformerBlock,
    SiglipTransformerStack,
    SiglipVisionEncoder,
    SigLIPVisionModel,
)

__all__ = [
    # Config
    "GemmaConfig",
    "PaliGemma2Config",
    "SigLIPVisionConfig",
    # Siglip Vision Encoder
    "SiglipVisionEncoder",
    "SiglipTransformerStack",
    "SiglipTransformerBlock",
    "SiglipSelfAttention",
    "SiglipPatchEmbedder",
    "SigLIPVisionModel",  # Alias for backward compatibility
    # Gemma Language Model
    "GemmaLM",
    "GemmaDecoder",
    "GemmaTransformerBlock",
    "GemmaForCausalLM",  # Alias for backward compatibility
    "GemmaModel",  # Alias for backward compatibility
    # PaliGemma VLM
    "PaliGemma2",
    "VisionTextConnector",
    # Processor
    "PaliGemmaProcessor",
    "PaliGemmaImageProcessor",
    "PaliGemmaTokenizer",
]

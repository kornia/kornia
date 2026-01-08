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

import torch
from torch import nn

from kornia.models.siglip2.vision_encoder import SigLip2VisionModel

from .configuration_paligemma import PaliGemmaConfig


class PaliGemmaForConditionalGeneration(nn.Module):
    """PaliGemma Model for Vision-Language tasks.

    This model combines a SigLip2 Vision Encoder with a Gemma Language Decoder.
    """

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config

        self.vision_tower = SigLip2VisionModel(config.vision_config)

        self.multi_modal_projector = nn.Linear(config.vision_config.hidden_size, config.hidden_size)

        self.language_model = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor):
        image_features = self.vision_tower(pixel_values)[0]

        image_features = self.multi_modal_projector(image_features)

        return image_features

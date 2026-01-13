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

import pytest
import torch

from kornia.models.paligemma import PaliGemma, PaliGemmaConfig
from kornia.models.paligemma.modeling_paligemma import GemmaAttention, GemmaMLP


class TestPaliGemmaModules:
    @pytest.fixture
    def config(self):
        conf = PaliGemmaConfig()
        conf.hidden_size = 32
        conf.intermediate_size = 64
        conf.num_attention_heads = 4
        conf.head_dim = 8
        conf.vocab_size = 100
        return conf

    def test_mlp(self, config):
        model = GemmaMLP(config)
        x = torch.randn(1, 10, config.hidden_size)
        output = model(x)
        assert output.shape == (1, 10, config.hidden_size)

    def test_attention(self, config):
        model = GemmaAttention(config)
        x = torch.randn(1, 10, config.hidden_size)
        position_ids = torch.arange(10).unsqueeze(0)
        output = model(x, position_ids=position_ids)
        assert output.shape == (1, 10, config.hidden_size)


def test_paligemma_forward_pass() -> None:
    config = PaliGemmaConfig()

    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    config.head_dim = 8
    config.vocab_size = 100
    config.vision_config.image_size = 32
    config.vision_config.patch_size = 16

    model = PaliGemma(config)
    model.eval()

    pixel_values = torch.randn(1, 3, 32, 32)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    logits = model(input_ids=input_ids, pixel_values=pixel_values)

    assert logits.shape == (1, 9, config.vocab_size)

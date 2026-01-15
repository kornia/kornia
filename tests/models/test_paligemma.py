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
        conf.num_hidden_layers = 1
        conf.num_attention_heads = 4
        conf.head_dim = 8
        conf.vocab_size = 100

        conf.vision_config.image_size = 32
        conf.vision_config.patch_size = 16
        conf.vision_config.hidden_size = 32
        conf.vision_config.num_hidden_layers = 1
        conf.vision_config.num_attention_heads = 4
        return conf

    def test_mlp(self, config, device, dtype):
        model = GemmaMLP(config).to(device=device, dtype=dtype)
        x = torch.randn(1, 10, config.hidden_size, device=device, dtype=dtype)
        output = model(x)
        assert output.shape == (1, 10, config.hidden_size)

    def test_attention(self, config, device, dtype):
        model = GemmaAttention(config).to(device=device, dtype=dtype)
        x = torch.randn(1, 10, config.hidden_size, device=device, dtype=dtype)
        position_ids = torch.arange(10, device=device).unsqueeze(0)
        output = model(x, position_ids=position_ids)
        assert output.shape == (1, 10, config.hidden_size)


class TestPaliGemma:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_smoke(self, batch_size, device, dtype):
        config = PaliGemmaConfig()
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 1
        config.num_attention_heads = 4
        config.head_dim = 8
        config.vocab_size = 100

        config.vision_config.image_size = 32
        config.vision_config.patch_size = 16
        config.vision_config.hidden_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4

        model = PaliGemma(config).to(device=device, dtype=dtype)

        pixel_values = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5), device=device)

        logits = model(input_ids=input_ids, pixel_values=pixel_values)

        expected_seq_len = 4 + 5
        assert logits.shape == (batch_size, expected_seq_len, config.vocab_size)

    def test_from_pretrained_interface(self):
        assert hasattr(PaliGemma, "from_pretrained")

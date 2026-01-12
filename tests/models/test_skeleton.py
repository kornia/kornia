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

from kornia.models.paligemma import PaliGemma, PaliGemmaConfig


def test_paligemma_forward_pass() -> None:
    config = PaliGemmaConfig()

    config.hidden_size = 128
    config.intermediate_size = 512
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.head_dim = 32
    config.vocab_size = 1000
    config.vision_config.image_size = 224
    config.vision_config.patch_size = 16

    model = PaliGemma(config)
    model.eval()

    pixel_values = torch.randn(1, 3, 224, 224)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    num_image_tokens = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
    num_text_tokens = input_ids.shape[1]
    expected_seq_len = num_image_tokens + num_text_tokens

    logits = model(input_ids=input_ids, pixel_values=pixel_values)

    assert logits.shape == (1, expected_seq_len, config.vocab_size), (
        f"Shape mismatch! Expected (1, {expected_seq_len}, {config.vocab_size}), got {logits.shape}"
    )

    assert not torch.isnan(logits).any(), "Output contains NaNs"

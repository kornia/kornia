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
from kornia.models.paligemma.modeling_paligemma import GemmaAttention, GemmaMLP, GemmaRMSNorm

from testing.base import BaseTester


class TestPaliGemmaModules(BaseTester):
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
        return conf

    def test_mlp(self, device, config):
        model = GemmaMLP(config).to(device)
        x = torch.randn(1, 10, config.hidden_size, device=device)
        output = model(x)
        assert output.shape == (1, 10, config.hidden_size)

    def test_attention(self, device, config):
        model = GemmaAttention(config).to(device)
        x = torch.randn(1, 10, config.hidden_size, device=device)
        position_ids = torch.arange(10, device=device).unsqueeze(0)
        output = model(x, position_ids=position_ids)
        assert output.shape == (1, 10, config.hidden_size)

    def test_forward(self, device, config):
        model = PaliGemma(config).to(device)
        model.eval()

        pixel_values = torch.randn(1, 3, 32, 32, device=device)
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=device)

        logits = model(input_ids=input_ids, pixel_values=pixel_values)

        assert logits.shape == (1, 9, config.vocab_size)

    def test_text_embeddings_are_scaled(self, device, dtype, config):
        # Gemma multiplies the input embeddings by sqrt(hidden_size), as a tensor in their dtype (GemmaModel.forward's
        # normalizer, modeling_gemma.py v4.51.0), and PaliGemma divides the projected image features by the same
        # factor first (PaliGemmaForConditionalGeneration.get_image_features): text enters scaled, the image unscaled.
        # In float16 and bfloat16 the factor rounded to the dtype gives other products than the Python float.
        model = PaliGemma(config).to(device, dtype).eval()
        captured = {}
        model.layers[0].register_forward_pre_hook(
            lambda module, args, kwargs: captured.__setitem__("decoder_input", args[0]), with_kwargs=True
        )
        pixel_values = torch.randn(2, 3, 32, 32, device=device, dtype=dtype)
        input_ids = torch.randint(1, config.vocab_size, (2, 5), device=device)

        with torch.no_grad():
            model(input_ids=input_ids, pixel_values=pixel_values)
            image = model.multi_modal_projector(model.vision_tower(pixel_values)[1])
            text = model.embed_tokens(input_ids) * torch.tensor(32**0.5, device=device, dtype=dtype)  # hidden_size 32

        decoder_input = captured["decoder_input"]
        assert image.shape[1] == 4  # 32 px at patch 16: 2 x 2 image tokens, before the text
        self.assert_close(decoder_input[:, :4], image, rtol=0.0, atol=0.0)
        self.assert_close(decoder_input[:, 4:], text, rtol=0.0, atol=0.0)

    def test_padding_idx_is_pad_token_id(self, device, config):
        # Gemma's pad_token_id is 0 (configuration_gemma.py v4.51.0). padding_idx = ignore_index = -100 wrapped to
        # vocab_size - 100: row 200 of a 300-token vocabulary, a real token, zeroed and frozen. (The fixture's 100
        # tokens would wrap -100 to row 0 and hide the difference.)
        config.vocab_size = 300
        model = PaliGemma(config).to(device).train()
        embedding = model.embed_tokens

        assert embedding.padding_idx == 0
        assert torch.all(embedding.weight[0] == 0)
        assert not torch.all(embedding.weight[200] == 0)
        input_ids = torch.tensor([[0, 5, 200, 7]], device=device)
        model(input_ids=input_ids, pixel_values=torch.randn(1, 3, 32, 32, device=device)).sum().backward()
        assert embedding.weight.grad[200].abs().sum() > 0
        assert torch.all(embedding.weight.grad[0] == 0)
        # The default config, without building its 257152 x 2048 embedding.
        assert PaliGemmaConfig().pad_token_id == 0

    def test_mlp_uses_tanh_gelu(self, config):
        # Gemma's hidden_act is "gelu_pytorch_tanh" (configuration_gemma.py v4.51.0).
        activation = GemmaMLP(config).act_fn
        assert isinstance(activation, torch.nn.GELU)
        assert activation.approximate == "tanh"

    def test_rms_norm_multiplies_in_float32(self, device):
        # Gemma's RMSNorm multiplies by (1 + weight) in float32 and casts afterwards, "(x * w).to(float16)"
        # (GemmaRMSNorm.forward, modeling_gemma.py v4.51.0). The weight starts at zeros, where 1 + w == 1 hides the
        # order, so it is filled with random values first.
        generator = torch.Generator().manual_seed(0)
        norm = GemmaRMSNorm(64)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(64, generator=generator))
        norm = norm.to(device, torch.float16)
        x = torch.randn(2, 5, 64, generator=generator).to(device, torch.float16)

        out = norm(x)

        x32 = x.float()
        normalized = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert out.dtype == torch.float16
        assert torch.equal(out, (normalized * (1.0 + norm.weight.float())).half())

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
        config.image_token_index = 99

        config.vision_config.image_size = 32
        config.vision_config.patch_size = 16
        config.vision_config.hidden_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4

        model = PaliGemma(config).to(device=device, dtype=dtype)

        pixel_values = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
        
        # We need 4 image tokens (32/16 = 2, 2*2 = 4 patches)
        image_tokens = torch.full((batch_size, 4), config.image_token_index, device=device)
        text_tokens = torch.randint(0, config.vocab_size, (batch_size, 5), device=device)
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)

        logits = model(input_ids=input_ids, pixel_values=pixel_values)

        expected_seq_len = input_ids.shape[1]  # 4 + 5 = 9
        assert logits.shape == (batch_size, expected_seq_len, config.vocab_size)

    def test_from_pretrained_interface(self, device):
        """Test that from_pretrained can be called and handles device placement."""
        # Check standard interface existence
        assert hasattr(PaliGemma, "from_pretrained")

        # Mocking transformers to avoid actual download
        from unittest.mock import MagicMock, patch

        mock_hf_model = MagicMock()

        # Setup specific config attributes that are accessed
        mock_text_config = MagicMock()
        mock_text_config.vocab_size = 100
        mock_text_config.hidden_size = 32
        mock_text_config.num_hidden_layers = 1
        mock_text_config.num_attention_heads = 4
        mock_text_config.head_dim = 8
        mock_text_config.intermediate_size = 64
        mock_text_config.num_key_value_heads = 4

        mock_vision_config = MagicMock()
        mock_vision_config.image_size = 32
        mock_vision_config.patch_size = 16
        mock_vision_config.hidden_size = 32
        mock_vision_config.num_hidden_layers = 1
        mock_vision_config.num_attention_heads = 4
        mock_vision_config.intermediate_size = 64

        mock_hf_model.config.text_config = mock_text_config
        mock_hf_model.config.vision_config = mock_vision_config

        # Setup state dict with matching keys for "Explicit Mapping" coverage
        # We need to simulate the HF keys expected by our new mapping logic
        mock_state_dict = {
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight": torch.randn(32, 3, 16, 16),
            "model.vision_tower.vision_model.embeddings.patch_embedding.bias": torch.randn(32),
            "model.vision_tower.vision_model.embeddings.position_embedding.weight": torch.randn(4, 32),  # 4 patches
            "model.vision_tower.vision_model.post_layernorm.weight": torch.randn(32),
            "model.vision_tower.vision_model.post_layernorm.bias": torch.randn(32),
            "model.multi_modal_projector.linear.weight": torch.randn(32, 32),
            "model.multi_modal_projector.linear.bias": torch.randn(32),
            "model.embed_tokens.weight": torch.randn(100, 32),
            "model.norm.weight": torch.randn(32),
            "lm_head.weight": torch.randn(100, 32),
            # Add one layer param for coverage
            "model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight": torch.randn(32, 32),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        }

        # Ensure shapes match what Kornia expects roughly
        mock_hf_model.state_dict.return_value = mock_state_dict

        with patch("transformers.PaliGemmaForConditionalGeneration.from_pretrained", return_value=mock_hf_model):
            # 1. Test basic load
            model = PaliGemma.from_pretrained("mock-model-id")
            assert isinstance(model, PaliGemma)

            # 2. Test device placement
            # We can't easily check internal device handling with mocks unless we inspect the calls,
            # but we can check the returned model device.
            # (Note: Mocks usually return CPU tensors, so actual move check depends on `torch.tensor` usage in mocks)
            # However, we can check if the API accepts it without error.
            if torch.cuda.is_available():
                model_gpu = PaliGemma.from_pretrained("mock-model-id", device="cuda")
                assert next(model_gpu.parameters()).device.type == "cuda"
            else:
                model_cpu = PaliGemma.from_pretrained("mock-model-id", device="cpu")
                assert next(model_cpu.parameters()).device.type == "cpu"

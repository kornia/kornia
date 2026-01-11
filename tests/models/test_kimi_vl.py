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

from kornia.models.kimi_vl import KimiVLConfig, KimiVLModel
from kornia.models.kimi_vl.config import KimiVLProjectorConfig, MoonViTConfig
from kornia.models.kimi_vl.model import KimiVLProjector
from kornia.models.kimi_vl.moonvit import MoonViT, MoonViTAttention, MoonViTEncoder, MoonViTRotaryEmbedding

from testing.base import BaseTester


@pytest.fixture
def config():
    vision_config = MoonViTConfig(
        image_size=32,
        patch_size=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    projector_config = KimiVLProjectorConfig(
        input_dim=32,  # Matches vision_config.hidden_size
        hidden_dim=64,
        output_dim=64,
    )
    return KimiVLConfig(vision_config=vision_config, projector_config=projector_config)


@pytest.fixture
def model(device, dtype, config):
    return KimiVLModel(config).to(device, dtype)


class TestKimiVLModel(BaseTester):
    def test_smoke(self, model):
        assert model is not None

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_cardinality(self, device, dtype, model, config, batch_size):
        images = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)

        output = model(images)

        # Expected output shape:
        # H_patches = 32/4 = 8
        # W_patches = 32/4 = 8
        # After pixel shuffle (downsample 2): H_new = 4, W_new = 4 -> 16 patches
        # Output dim = 64
        expected_patches = 16
        assert output.shape == (batch_size, expected_patches, config.projector_config.output_dim)

    def test_variable_resolution(self, device, dtype, model, config):
        batch_size = 1
        # 48x48 image -> 12x12 patches -> 6x6 after shuffle -> 36 patches
        images = torch.randn(batch_size, 3, 48, 48, device=device, dtype=dtype)

        output = model(images)
        assert output.shape == (batch_size, 36, config.projector_config.output_dim)

    def test_attention_mask(self, device, dtype, model, config):
        batch_size = 1
        images = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)

        # Create a mask (B, N, N) or (B, 1, N, N)
        # N = 64 (8x8 patches)
        N = 64
        mask = torch.ones(batch_size, N, N, device=device, dtype=torch.bool)

        # Mask out the last token
        mask[:, :, -1] = 0

        output = model(images, attention_mask=mask)
        assert output.shape == (batch_size, 16, config.projector_config.output_dim)

    def test_exception(self, device, dtype, model):
        # Test invalid input shape (missing batch dim)
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            images = torch.randn(3, 32, 32, device=device, dtype=dtype)
            model(images)

    def test_gradcheck(self, device, config):
        # Convert model to float64 for gradcheck
        model = KimiVLModel(config).to(device, torch.float64).train()
        batch_size = 1
        images = torch.randn(batch_size, 3, 32, 32, device=device, dtype=torch.float64, requires_grad=True)

        self.gradcheck(model, images, raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer, model):
        model = model.eval()
        batch_size = 1
        images = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)

        model_optimized = torch_optimizer(model)

        with torch.no_grad():
            expected = model(images)
            actual = model_optimized(images)

        self.assert_close(actual, expected)


class TestKimiVLComponents(BaseTester):
    def test_moonvit(self, device, dtype, config):
        model = MoonViT(config.vision_config).to(device, dtype)
        batch_size = 2
        images = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)

        output = model(images)
        # MoonViT output is (B, N, D)
        # N = (32/4)^2 = 64
        assert output.shape == (batch_size, 64, config.vision_config.hidden_size)

    def test_moonvit_encoder(self, device, dtype, config):
        encoder = MoonViTEncoder(config.vision_config).to(device, dtype)
        batch_size = 2
        seq_len = 64
        hidden_size = config.vision_config.hidden_size

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        # Create dummy cos/sin for RoPE
        head_dim = hidden_size // config.vision_config.num_attention_heads
        cos = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
        sin = torch.randn(seq_len, head_dim, device=device, dtype=dtype)

        output = encoder(x, cos, sin)
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_moonvit_attention(self, device, dtype, config):
        attention = MoonViTAttention(config.vision_config).to(device, dtype)
        batch_size = 2
        seq_len = 64
        hidden_size = config.vision_config.hidden_size

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        # Create dummy cos/sin for RoPE
        head_dim = hidden_size // config.vision_config.num_attention_heads
        cos = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
        sin = torch.randn(seq_len, head_dim, device=device, dtype=dtype)

        output = attention(x, cos, sin)
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_moonvit_rotary_embedding(self, device, dtype, config):
        dim = config.vision_config.hidden_size // config.vision_config.num_attention_heads
        rope = MoonViTRotaryEmbedding(dim).to(device, dtype)

        h, w = 8, 8
        cos, sin = rope(h, w, device)

        seq_len = h * w
        assert cos.shape == (seq_len, dim)
        assert sin.shape == (seq_len, dim)

    def test_projector(self, device, dtype, config):
        model = KimiVLProjector(config.projector_config).to(device, dtype)
        batch_size = 2
        input_features = torch.randn(batch_size, 64, 32, device=device, dtype=dtype)

        output = model(input_features, h=8, w=8)
        assert output.shape == (batch_size, 16, config.projector_config.output_dim)

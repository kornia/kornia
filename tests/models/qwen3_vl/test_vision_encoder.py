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

from kornia.models.qwen3_vl import (
    Qwen3VLAttention,
    Qwen3VLPatchEmbed,
    Qwen3VLRotaryEmbedding,
    Qwen3VLVisionConfig,
    Qwen3VLVisionEncoderOutput,
    Qwen3VLVisionTransformer,
    apply_rotary_pos_emb,
)

from testing.base import BaseTester


def _tiny_vision_config(**overrides):
    base = {
        "patch_size": 4,
        "in_channels": 3,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "layer_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    base.update(overrides)
    return Qwen3VLVisionConfig(**base)


@pytest.fixture
def tiny_config():
    return _tiny_vision_config()


@pytest.fixture
def model(device, dtype, tiny_config):
    return Qwen3VLVisionTransformer(tiny_config).to(device=device, dtype=dtype)


class TestQwen3VLVisionTransformer(BaseTester):
    def test_smoke(self, model):
        assert model is not None
        # Default schedule: layers num/3, 2*num/3, num-1 -> (1, 2, 3) for 4 layers.
        assert model.deepstack_layer_indices == (1, 2, 3)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("hw", [(16, 16), (16, 24)])
    def test_cardinality(self, device, dtype, model, tiny_config, batch_size, hw):
        h, w = hw
        images = torch.randn(batch_size, 3, h, w, device=device, dtype=dtype)
        out = model(images)
        expected_n = (h // tiny_config.patch_size) * (w // tiny_config.patch_size)
        assert isinstance(out, Qwen3VLVisionEncoderOutput)
        assert out.last_hidden_state.shape == (batch_size, expected_n, tiny_config.hidden_size)
        assert out.grid_hw == (h // tiny_config.patch_size, w // tiny_config.patch_size)

    def test_deepstack(self, device, dtype, model, tiny_config):
        images = torch.randn(1, 3, 16, 16, device=device, dtype=dtype)
        out = model(images)
        assert len(out.deepstack_features) == len(model.deepstack_layer_indices)
        for feat in out.deepstack_features:
            assert feat.shape == out.last_hidden_state.shape

    def test_custom_deepstack_indices(self, device, dtype):
        cfg = _tiny_vision_config(deepstack_layer_indices=(0, 2))
        m = Qwen3VLVisionTransformer(cfg).to(device=device, dtype=dtype)
        images = torch.randn(1, 3, 16, 16, device=device, dtype=dtype)
        out = m(images)
        assert m.deepstack_layer_indices == (0, 2)
        assert len(out.deepstack_features) == 2

    def test_exception(self, device, dtype, model):
        with pytest.raises(ValueError, match="divisible by patch_size"):
            model(torch.randn(1, 3, 17, 16, device=device, dtype=dtype))
        with pytest.raises(ValueError, match="Expected 4D input"):
            model(torch.randn(3, 16, 16, device=device, dtype=dtype))

    def test_invalid_deepstack_index_rejected(self):
        cfg = _tiny_vision_config(deepstack_layer_indices=(0, 99))
        with pytest.raises(ValueError, match="out of range"):
            Qwen3VLVisionTransformer(cfg)

    def test_gradcheck(self, device, tiny_config):
        model = Qwen3VLVisionTransformer(tiny_config).to(device=device, dtype=torch.float64).train()
        images = torch.randn(1, 3, 8, 8, device=device, dtype=torch.float64, requires_grad=True)

        def fn(x):
            return model(x).last_hidden_state

        self.gradcheck(fn, images, raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer, model):
        model = model.eval()
        images = torch.randn(1, 3, 16, 16, device=device, dtype=dtype)
        optimized = torch_optimizer(model)
        with torch.no_grad():
            expected = model(images).last_hidden_state
            actual = optimized(images).last_hidden_state
        self.assert_close(actual, expected)


class TestQwen3VLComponents:
    def test_patch_embed_shape(self, device, dtype):
        cfg = _tiny_vision_config()
        embed = Qwen3VLPatchEmbed(cfg).to(device=device, dtype=dtype)
        x = torch.randn(2, 3, 16, 16, device=device, dtype=dtype)
        tokens, h, w = embed(x)
        assert tokens.shape == (2, 16, cfg.hidden_size)
        assert (h, w) == (4, 4)

    def test_rotary_embedding_shape(self, device, dtype):
        head_dim = 8
        rope = Qwen3VLRotaryEmbedding(head_dim=head_dim, theta=10000.0)
        cos, sin = rope(4, 4, device, dtype)
        assert cos.shape == (16, head_dim)
        assert sin.shape == (16, head_dim)
        assert cos.dtype == dtype

    def test_rotary_embedding_rejects_bad_head_dim(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            Qwen3VLRotaryEmbedding(head_dim=6)

    def test_apply_rope_identity(self, device, dtype):
        head_dim = 8
        x = torch.randn(2, 4, 16, head_dim, device=device, dtype=dtype)
        cos = torch.ones(1, 1, 16, head_dim, device=device, dtype=dtype)
        sin = torch.zeros(1, 1, 16, head_dim, device=device, dtype=dtype)
        torch.testing.assert_close(apply_rotary_pos_emb(x, cos, sin), x)

    def test_attention_shape(self, device, dtype):
        cfg = _tiny_vision_config()
        attn = Qwen3VLAttention(cfg).to(device=device, dtype=dtype)
        n = 16
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        x = torch.randn(2, n, cfg.hidden_size, device=device, dtype=dtype)
        cos = torch.randn(n, head_dim, device=device, dtype=dtype)
        sin = torch.randn(n, head_dim, device=device, dtype=dtype)
        out = attn(x, cos, sin)
        assert out.shape == x.shape

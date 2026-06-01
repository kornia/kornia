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
    Qwen3VLImageProcessor,
    Qwen3VLImageProcessorConfig,
    Qwen3VLPatchEmbed,
    Qwen3VLPatchMerger,
    Qwen3VLRotaryEmbedding,
    Qwen3VLVisionConfig,
    Qwen3VLVisionEncoderOutput,
    Qwen3VLVisionModel,
    apply_rotary_pos_emb_vision,
    rotate_half,
)

from testing.base import BaseTester


def _tiny_vision_config(**overrides):
    base = {
        "patch_size": 4,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "hidden_size": 32,
        "depth": 4,
        "num_heads": 4,
        "intermediate_size": 64,
        "layer_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "spatial_merge_size": 2,
        "out_hidden_size": 48,
        "num_position_embeddings": 64,
        "deepstack_visual_indexes": (1, 2, 3),
    }
    base.update(overrides)
    return Qwen3VLVisionConfig(**base)


def _tiny_processor_config():
    return Qwen3VLImageProcessorConfig(
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=8 * 8,
        max_pixels=32 * 32,
    )


def _make_inputs(processor: Qwen3VLImageProcessor, b: int, h: int, w: int, device, dtype):
    img = torch.randn(b, 3, h, w, device=device, dtype=dtype)
    return processor(img)


@pytest.fixture
def tiny_config():
    return _tiny_vision_config()


@pytest.fixture
def model(device, dtype, tiny_config):
    return Qwen3VLVisionModel(tiny_config).to(device=device, dtype=dtype)


@pytest.fixture
def processor(device, dtype):
    return Qwen3VLImageProcessor(_tiny_processor_config()).to(device=device, dtype=dtype)


class TestQwen3VLVisionModel(BaseTester):
    def test_smoke(self, model):
        assert model is not None
        assert model.deepstack_visual_indexes == (1, 2, 3)
        assert len(model.deepstack_merger_list) == 3

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("hw", [(16, 16), (16, 24)])
    def test_cardinality(self, device, dtype, model, processor, tiny_config, batch_size, hw):
        h, w = hw
        out_pre = _make_inputs(processor, batch_size, h, w, device, dtype)
        out = model(out_pre.pixel_values, out_pre.grid_thw)

        merge = tiny_config.spatial_merge_size
        grid_h, grid_w = h // tiny_config.patch_size, w // tiny_config.patch_size
        merged_per_image = (grid_h * grid_w) // (merge * merge)
        expected_merged = batch_size * merged_per_image

        assert isinstance(out, Qwen3VLVisionEncoderOutput)
        assert out.last_hidden_state.shape == (expected_merged, tiny_config.out_hidden_size)
        assert len(out.deepstack_features) == 3
        for f in out.deepstack_features:
            assert f.shape == (expected_merged, tiny_config.out_hidden_size)
        assert out.grid_thw.shape == (batch_size, 3)

    def test_custom_deepstack_indices(self, device, dtype, processor):
        cfg = _tiny_vision_config(deepstack_visual_indexes=(0, 2))
        m = Qwen3VLVisionModel(cfg).to(device=device, dtype=dtype)
        pre = _make_inputs(processor, 1, 16, 16, device, dtype)
        out = m(pre.pixel_values, pre.grid_thw)
        assert m.deepstack_visual_indexes == (0, 2)
        assert len(out.deepstack_features) == 2

    def test_invalid_deepstack_index_rejected(self):
        cfg = _tiny_vision_config(deepstack_visual_indexes=(0, 99))
        with pytest.raises(ValueError, match="out of range"):
            Qwen3VLVisionModel(cfg)

    def test_exception(self, device, dtype, model, processor):
        pre = _make_inputs(processor, 1, 16, 16, device, dtype)
        with pytest.raises(ValueError, match="grid_thw must have shape"):
            model(pre.pixel_values, pre.grid_thw[:, :2])
        with pytest.raises(ValueError, match="integer tensor"):
            model(pre.pixel_values, pre.grid_thw.float())
        with pytest.raises(ValueError, match=r"Expected flat patch tensor"):
            model(pre.pixel_values.unsqueeze(0), pre.grid_thw)

    def test_gradcheck(self, device, tiny_config):
        m = Qwen3VLVisionModel(tiny_config).to(device=device, dtype=torch.float64).train()
        proc = Qwen3VLImageProcessor(_tiny_processor_config()).to(device=device, dtype=torch.float64)
        img = torch.randn(1, 3, 8, 8, device=device, dtype=torch.float64, requires_grad=True)

        def fn(x):
            pre = proc(x)
            return m(pre.pixel_values, pre.grid_thw).last_hidden_state

        self.gradcheck(fn, img, raise_exception=True, fast_mode=True)


class TestQwen3VLComponents:
    def test_patch_embed_shape(self, device, dtype):
        cfg = _tiny_vision_config()
        proc = Qwen3VLImageProcessor(_tiny_processor_config()).to(device=device, dtype=dtype)
        embed = Qwen3VLPatchEmbed(cfg).to(device=device, dtype=dtype)
        pre = proc(torch.randn(2, 3, 16, 16, device=device, dtype=dtype))
        tokens = embed(pre.pixel_values)
        assert tokens.shape == (pre.pixel_values.shape[0], cfg.hidden_size)

    def test_rotary_embedding_shape(self, device, dtype):
        head_dim = 8
        rope = Qwen3VLRotaryEmbedding(head_dim // 2, theta=10000.0).to(device=device)
        out = rope(4)
        assert out.shape == (4, (head_dim // 2) // 2)

    def test_rotary_embedding_rejects_odd_dim(self):
        with pytest.raises(ValueError, match="even"):
            Qwen3VLRotaryEmbedding(dim=3)

    def test_apply_rotary_identity(self, device, dtype):
        head_dim = 8
        x = torch.randn(2, 16, 4, head_dim, device=device, dtype=dtype)
        cos = torch.ones(16, head_dim, device=device, dtype=dtype)
        sin = torch.zeros(16, head_dim, device=device, dtype=dtype)
        q, k = apply_rotary_pos_emb_vision(x, x, cos, sin)
        torch.testing.assert_close(q, x.to(q.dtype))
        torch.testing.assert_close(k, x.to(k.dtype))

    def test_rotate_half(self, device, dtype):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device, dtype=dtype)
        torch.testing.assert_close(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]], device=device, dtype=dtype))

    def test_attention_shape(self, device, dtype):
        cfg = _tiny_vision_config()
        attn = Qwen3VLAttention(cfg).to(device=device, dtype=dtype)
        n = 16
        head_dim = cfg.hidden_size // cfg.num_heads
        x = torch.randn(n, cfg.hidden_size, device=device, dtype=dtype)
        cos = torch.randn(n, head_dim, device=device, dtype=dtype)
        sin = torch.randn(n, head_dim, device=device, dtype=dtype)
        cu = torch.tensor([0, n], dtype=torch.int32, device=device)
        out = attn(x, cu, cos, sin)
        assert out.shape == x.shape

    def test_patch_merger_shape(self, device, dtype):
        cfg = _tiny_vision_config()
        merger = Qwen3VLPatchMerger(cfg, use_postshuffle_norm=False).to(device=device, dtype=dtype)
        n = 16
        x = torch.randn(n, cfg.hidden_size, device=device, dtype=dtype)
        out = merger(x)
        merge_sq = cfg.spatial_merge_size**2
        assert out.shape == (n // merge_sq, cfg.out_hidden_size)

    def test_patch_merger_postshuffle(self, device, dtype):
        cfg = _tiny_vision_config()
        merger = Qwen3VLPatchMerger(cfg, use_postshuffle_norm=True).to(device=device, dtype=dtype)
        n = 16
        x = torch.randn(n, cfg.hidden_size, device=device, dtype=dtype)
        out = merger(x)
        merge_sq = cfg.spatial_merge_size**2
        assert out.shape == (n // merge_sq, cfg.out_hidden_size)

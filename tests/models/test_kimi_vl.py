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

import json
import os

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

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


class TestKimiVLModel(BaseTester):
    def test_smoke(self, device, dtype, config):
        model = KimiVLModel(config).to(device, dtype)
        assert model is not None

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_cardinality(self, device, dtype, config, batch_size):
        model = KimiVLModel(config).to(device, dtype)
        images = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)

        output = model(images)

        # Expected output shape:
        # H_patches = 32/4 = 8
        # W_patches = 32/4 = 8
        # After pixel shuffle (downsample 2): H_new = 4, W_new = 4 -> 16 patches
        # Output dim = 64
        expected_patches = 16
        assert output.shape == (batch_size, expected_patches, config.projector_config.output_dim)

    def test_variable_resolution(self, device, dtype, config):
        model = KimiVLModel(config).to(device, dtype)
        batch_size = 1
        # 48x48 image -> 12x12 patches -> 6x6 after shuffle -> 36 patches
        images = torch.randn(batch_size, 3, 48, 48, device=device, dtype=dtype)

        output = model(images)
        assert output.shape == (batch_size, 36, config.projector_config.output_dim)

    def test_attention_mask(self, device, dtype, config):
        model = KimiVLModel(config).to(device, dtype)
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

    def test_exception(self, device, dtype, config):
        model = KimiVLModel(config).to(device, dtype)
        # Test invalid input shape (missing batch dim)
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            images = torch.randn(3, 32, 32, device=device, dtype=dtype)
            model(images)

    def test_gradcheck(self, device, dtype, config):
        # Convert model to float64 for gradcheck
        model = KimiVLModel(config).to(device, torch.float64).train()
        batch_size = 1
        images = torch.randn(batch_size, 3, 32, 32, device=device, dtype=torch.float64, requires_grad=True)

        self.gradcheck(model, images, raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer, config):
        model = KimiVLModel(config).to(device, dtype).eval()
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


def test_kimi_vl_official_weights():
    """Integration test for loading official KimiVL vision weights.

    To run this test, set the environment variable ``KIMI_VL_WEIGHTS_DIR`` to a directory
    containing the file ``model.safetensors.index.json`` and the referenced shard files.
    If this variable is not set or the files are missing, the test is skipped.
    """

    weights_dir = os.environ.get("KIMI_VL_WEIGHTS_DIR")
    if not weights_dir:
        pytest.skip("KIMI_VL_WEIGHTS_DIR is not set; skipping test_kimi_vl_official_weights")

    index_path = os.path.join(weights_dir, "model.safetensors.index.json")

    if not os.path.exists(index_path):
        pytest.skip(f"Weights index not found at {index_path}; skipping test_kimi_vl_official_weights")

    with open(index_path) as f:
        index = json.load(f)

    vision_keys = [k for k in index["weight_map"].keys() if "vision_tower" in k or "multi_modal_projector" in k]
    shards = {index["weight_map"][k] for k in vision_keys}

    state_dict = {}
    for shard in shards:
        shard_path = os.path.join(weights_dir, shard)
        shard_weights = load_file(shard_path)
        for k in vision_keys:
            if k in shard_weights:
                state_dict[k] = shard_weights[k]

    vision_config = MoonViTConfig(
        image_size=336,
        patch_size=14,
        hidden_size=1152,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=4304,
    )

    projector_config = KimiVLProjectorConfig(input_dim=1152, hidden_dim=4608, output_dim=2048)

    config = KimiVLConfig(vision_config=vision_config, projector_config=projector_config)

    model = KimiVLModel(config)
    model.eval()

    new_state_dict = {}

    def get_w(key):
        return state_dict[f"vision_tower.{key}"]

    new_state_dict["vision_encoder.patch_embed.weight"] = get_w("patch_embed.proj.weight")
    new_state_dict["vision_encoder.patch_embed.bias"] = get_w("patch_embed.proj.bias")

    pos_embed = get_w("patch_embed.pos_emb.weight")
    pos_embed_reshaped = pos_embed.permute(2, 0, 1).unsqueeze(0)
    pos_embed_interp = F.interpolate(pos_embed_reshaped, size=(24, 24), mode="bicubic", align_corners=False)
    pos_embed_final = pos_embed_interp.flatten(2).transpose(1, 2)
    new_state_dict["vision_encoder.pos_embed"] = pos_embed_final

    for i in range(config.vision_config.num_hidden_layers):
        prefix_official = f"encoder.blocks.{i}"
        prefix_kornia = f"vision_encoder.encoder.layers.{i}"

        new_state_dict[f"{prefix_kornia}.norm1.weight"] = get_w(f"{prefix_official}.norm0.weight")
        new_state_dict[f"{prefix_kornia}.norm1.bias"] = get_w(f"{prefix_official}.norm0.bias")
        new_state_dict[f"{prefix_kornia}.norm2.weight"] = get_w(f"{prefix_official}.norm1.weight")
        new_state_dict[f"{prefix_kornia}.norm2.bias"] = get_w(f"{prefix_official}.norm1.bias")

        wqkv = get_w(f"{prefix_official}.wqkv.weight")
        bqkv = get_w(f"{prefix_official}.wqkv.bias")

        hidden_size = config.vision_config.hidden_size
        wq, wk, wv = wqkv.split(hidden_size, dim=0)
        bq, bk, bv = bqkv.split(hidden_size, dim=0)

        new_state_dict[f"{prefix_kornia}.attn.q_proj.weight"] = wq
        new_state_dict[f"{prefix_kornia}.attn.q_proj.bias"] = bq
        new_state_dict[f"{prefix_kornia}.attn.k_proj.weight"] = wk
        new_state_dict[f"{prefix_kornia}.attn.k_proj.bias"] = bk
        new_state_dict[f"{prefix_kornia}.attn.v_proj.weight"] = wv
        new_state_dict[f"{prefix_kornia}.attn.v_proj.bias"] = bv

        new_state_dict[f"{prefix_kornia}.attn.out_proj.weight"] = get_w(f"{prefix_official}.wo.weight")
        new_state_dict[f"{prefix_kornia}.attn.out_proj.bias"] = get_w(f"{prefix_official}.wo.bias")

        new_state_dict[f"{prefix_kornia}.mlp.fc1.weight"] = get_w(f"{prefix_official}.mlp.fc0.weight")
        new_state_dict[f"{prefix_kornia}.mlp.fc1.bias"] = get_w(f"{prefix_official}.mlp.fc0.bias")
        new_state_dict[f"{prefix_kornia}.mlp.fc2.weight"] = get_w(f"{prefix_official}.mlp.fc1.weight")
        new_state_dict[f"{prefix_kornia}.mlp.fc2.bias"] = get_w(f"{prefix_official}.mlp.fc1.bias")

    new_state_dict["vision_encoder.norm.weight"] = get_w("encoder.final_layernorm.weight")
    new_state_dict["vision_encoder.norm.bias"] = get_w("encoder.final_layernorm.bias")

    new_state_dict["projector.pre_norm.weight"] = state_dict["multi_modal_projector.pre_norm.weight"]
    new_state_dict["projector.pre_norm.bias"] = state_dict["multi_modal_projector.pre_norm.bias"]

    new_state_dict["projector.mlp.0.weight"] = state_dict["multi_modal_projector.linear_1.weight"]
    new_state_dict["projector.mlp.0.bias"] = state_dict["multi_modal_projector.linear_1.bias"]
    new_state_dict["projector.mlp.2.weight"] = state_dict["multi_modal_projector.linear_2.weight"]
    new_state_dict["projector.mlp.2.bias"] = state_dict["multi_modal_projector.linear_2.bias"]

    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0

    x = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 144, 2048)

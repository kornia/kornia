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

from unittest.mock import patch

import pytest
import torch

import kornia.models.kimi_vl.builder as kimi_vl_builder
from kornia.core.exceptions import BaseError
from kornia.models.kimi_vl import KimiVLBuilder, KimiVLConfig, KimiVLModel
from kornia.models.kimi_vl.config import KimiVLProjectorConfig, MoonViTConfig
from kornia.models.kimi_vl.model import KimiVLProjector
from kornia.models.kimi_vl.moonvit import (
    MoonViT,
    MoonViTAttention,
    MoonViTEncoder,
    MoonViTRotaryEmbedding,
    apply_rotary_pos_emb,
)

from testing.base import BaseTester


@pytest.fixture
def config():
    vision_config = MoonViTConfig(
        image_size=32,
        patch_size=4,
        init_pos_emb_height=8,
        init_pos_emb_width=8,
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


def _moonvit_attention_inputs(config, batch_size, seq_len, device, dtype):
    """Random ``(B, N, D)`` tokens and the ``(N, d / 2)`` RoPE cosine and sine for :class:`MoonViTAttention`."""
    hidden_size = config.vision_config.hidden_size
    head_dim = hidden_size // config.vision_config.num_attention_heads
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    cos = torch.randn(seq_len, head_dim // 2, device=device, dtype=dtype)
    sin = torch.randn(seq_len, head_dim // 2, device=device, dtype=dtype)
    return x, cos, sin


class TestKimiVLBuilder(BaseTester):
    def test_from_config(self, config):
        model = KimiVLBuilder.from_config(config)

        assert isinstance(model, KimiVLModel)
        assert model.config is config

    def test_from_pretrained_hf(self, config):
        expected = KimiVLBuilder.from_config(config)
        with (
            patch.object(kimi_vl_builder, "_download_weights", return_value=expected.state_dict()) as mock_download,
            patch.object(kimi_vl_builder, "_kimi_vl_a3b_instruct_config", return_value=config),
        ):
            actual = KimiVLBuilder.from_pretrained_hf(cache_dir="cache")

        assert isinstance(actual, KimiVLModel)
        assert actual.config is config
        mock_download.assert_called_once_with(kimi_vl_builder._KIMI_VL_A3B_INSTRUCT_REPO_ID, "cache")
        # ``training`` of the model and of every submodule: ``{False}`` is evaluation mode throughout.
        assert {m.training for m in actual.modules()} == {False}
        assert {m.training for m in KimiVLBuilder.from_config(config).modules()} == {True}

    def test_download_weights_uses_kornias_downloader(self, tmp_path):
        """No optional package: kornia's own downloader, then its own safetensors reader."""
        expected = {"weight": torch.zeros(1)}
        with (
            patch.object(kimi_vl_builder, "download_hf_file", return_value="cached.safetensors") as download,
            patch.object(kimi_vl_builder, "load_safetensors", return_value=expected) as read,
        ):
            state_dict = kimi_vl_builder._download_weights(kimi_vl_builder._KIMI_VL_A3B_INSTRUCT_REPO_ID, str(tmp_path))

        assert state_dict is expected
        read.assert_called_once_with("cached.safetensors")
        download.assert_called_once_with(
            "kornia/kimi-vl-a3b-instruct-vision",
            "model.safetensors",
            model_dir=str(tmp_path),
            validate=kimi_vl_builder.check_safetensors,
        )

    def test_pretrained_config_matches_checkpoint_grid(self):
        config = kimi_vl_builder._kimi_vl_a3b_instruct_config()
        assert config.vision_config.init_pos_emb_height == 64
        assert config.vision_config.init_pos_emb_width == 64


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

    def test_pretrained_image_size(self, device: torch.device, dtype: torch.dtype) -> None:
        pretrained_config = kimi_vl_builder._kimi_vl_a3b_instruct_config()
        image_size = pretrained_config.vision_config.image_size
        patch_size = pretrained_config.vision_config.patch_size
        vision_config = MoonViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            init_pos_emb_height=64,
            init_pos_emb_width=64,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=16,
        )
        projector_config = KimiVLProjectorConfig(input_dim=8, hidden_dim=16, output_dim=8)
        config = KimiVLConfig(vision_config=vision_config, projector_config=projector_config)
        model = KimiVLModel(config).to(device, dtype)

        output = model(torch.randn(1, 3, image_size, image_size, device=device, dtype=dtype))

        output_grid_size = image_size // patch_size // model.projector.downsample_ratio
        assert output.shape == (1, output_grid_size**2, projector_config.output_dim)

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

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_three_dim_mask_is_per_sample(self, device, dtype, model, config, batch_size):
        # A (B, N, N) mask holds one N x N mask per sample, shared by the heads: sample b of a batch equals its own
        # batch-of-one run. The fixture has 4 heads: a mask read per head does not broadcast against B = 2, and with
        # B = 4, the head count, it broadcasts silently to wrong outputs.
        model = model.eval()
        n = 64  # 32 px at patch 4: an 8 x 8 grid
        images = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
        mask = torch.ones(batch_size, n, n, dtype=torch.bool, device=device)
        mask[1, :, n // 2 :] = False  # sample 1 attends to the first half of the keys only
        if batch_size == 4:
            mask[2, :, 1::2] = False  # sample 2: the even keys
            mask[3, :, : n // 4] = False  # sample 3: all but the first quarter

        with torch.no_grad():
            output = model(images, attention_mask=mask)
            for b in range(batch_size):
                self.assert_close(output[b], model(images[b : b + 1], attention_mask=mask[b : b + 1])[0])
            self.assert_close(model(images, attention_mask=mask[:, None]), output)
            batch_of_three = model(
                torch.randn(3, 3, 32, 32, device=device, dtype=dtype),
                attention_mask=torch.ones(3, n, n, dtype=torch.bool, device=device),
            )

        assert batch_of_three.shape == (3, 16, config.projector_config.output_dim)

    def test_exception(self, device, dtype, model):
        # Test invalid input shape (missing batch dim)
        images = torch.randn(3, 32, 32, device=device, dtype=dtype)
        with pytest.raises((RuntimeError, ValueError, IndexError)):
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

    def test_dynamo_fullgraph_three_dim_mask(self, device, dtype, torch_optimizer, config):
        model = MoonViT(config.vision_config).to(device, dtype).eval()
        n = 64
        images = torch.randn(2, 3, 32, 32, device=device, dtype=dtype)
        mask = torch.ones(2, n, n, dtype=torch.bool, device=device)
        mask[1, :, n // 2 :] = False

        model_optimized = torch_optimizer(model, fullgraph=True)

        with torch.no_grad():
            expected = model(images, attention_mask=mask)
            actual = model_optimized(images, attention_mask=mask)

        # Inductor keeps fused half-precision intermediates in float32, so a few outputs differ from eager by one
        # unit in the last place. On the base revision a60a6c932, where a (B, N, N) mask raised, the (B, 1, N, N) form
        # fails the default half tolerance the same way (7 / 4096 outputs in float16, 12 / 4096 in bfloat16).
        self.assert_close(actual, expected, low_tolerance=dtype in (torch.float16, torch.bfloat16))


class TestKimiVLComponents(BaseTester):
    def test_apply_rotary_pos_emb_preserves_float64_precision(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")

        # These values intentionally differ below float32 resolution. Before the
        # dtype-aware promotion, the identity rotation rounded them before casting
        # the result back to float64 and failed both checks below.
        x = torch.tensor(
            [1.000000001, 2.000000001, 3.000000001, 4.000000001],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        ).view(1, 1, 1, 4)
        cos = torch.ones(1, 1, 1, 2, device=device, dtype=torch.float64)
        sin = torch.zeros_like(cos)

        actual = apply_rotary_pos_emb(x, cos, sin)

        self.assert_close(actual, x, rtol=0.0, atol=0.0)
        self.gradcheck(lambda value: apply_rotary_pos_emb(value, cos, sin), x)

    def test_rotary_pos_emb_2d_expected_values(self, device):
        # Source: Moonshot Kimi-VL-A3B-Instruct Rope2DPosEmb at revision
        # 398eede0903cd983a2bfa0cc634e9ac1d843f375. The expected angles below
        # are derived by hand for a 2x2 row-major grid and theta=100:
        # [x * 1, y * 1, x * 0.1, y * 0.1].
        rope = MoonViTRotaryEmbedding(dim=8, theta=100.0)
        cos, sin = rope(h=2, w=2, device=device)
        expected_angles = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.1, 0.0],
                [0.0, 1.0, 0.0, 0.1],
                [1.0, 1.0, 0.1, 0.1],
            ],
            device=device,
        )
        expected_cos = expected_angles.cos()
        expected_sin = expected_angles.sin()

        self.assert_close(cos, expected_cos)
        self.assert_close(sin, expected_sin)

        x = torch.arange(1.0, 33.0, device=device).view(1, 1, 4, 8)
        x_pairs = x.view(1, 1, 4, 4, 2)
        expected = torch.stack(
            (
                x_pairs[..., 0] * expected_cos - x_pairs[..., 1] * expected_sin,
                x_pairs[..., 0] * expected_sin + x_pairs[..., 1] * expected_cos,
            ),
            dim=-1,
        ).flatten(-2)

        actual = apply_rotary_pos_emb(x, cos.view(1, 1, 4, 4), sin.view(1, 1, 4, 4))

        self.assert_close(actual, expected)

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
        cos = torch.randn(seq_len, head_dim // 2, device=device, dtype=dtype)
        sin = torch.randn(seq_len, head_dim // 2, device=device, dtype=dtype)

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
        cos = torch.randn(seq_len, head_dim // 2, device=device, dtype=dtype)
        sin = torch.randn(seq_len, head_dim // 2, device=device, dtype=dtype)

        output = attention(x, cos, sin)
        assert output.shape == (batch_size, seq_len, hidden_size)

    @pytest.mark.parametrize("mask_shape", [(4, 16, 16), (16, 16), (3, 16)])
    def test_moonvit_attention_two_and_three_dim_masks_lead_with_the_batch_size(
        self, device, dtype, config, mask_shape
    ):
        # A 2-D mask is a (B, N) padding mask and a 3-D one a (B, N, N) mask per sample, whatever the batch size, as
        # SigLip2Attention reads them. A leading dimension that is neither B nor 1 raises, naming the 4-D form, rather
        # than broadcast as a per-head (H, N, N) or a shared (N, N) mask: those would be read per sample whenever B
        # happens to equal H or N. Here B = 2 and the fixture has H = 4 heads.
        attention = MoonViTAttention(config.vision_config).to(device, dtype).eval()
        x, cos, sin = _moonvit_attention_inputs(config, 2, 16, device, dtype)
        mask = torch.ones(mask_shape, dtype=torch.bool, device=device)

        with torch.no_grad(), pytest.raises(BaseError, match=r"leading dimension .* pass it as 4-D"):
            attention(x, cos, sin, attention_mask=mask)

    def test_moonvit_attention_four_dim_per_head_mask(self, device, dtype, config):
        # A per-head mask goes in as (1, H, N, N) or (B, H, N, N), which scaled_dot_product_attention broadcasts against
        # the (B, H, N, N) scores: masking only head 0's keys changes head 0's output only.
        attention = MoonViTAttention(config.vision_config).to(device, dtype).eval()
        heads, n = config.vision_config.num_attention_heads, 16
        head_dim = config.vision_config.hidden_size // heads
        x, cos, sin = _moonvit_attention_inputs(config, 2, n, device, dtype)
        mask = torch.ones(1, heads, n, n, dtype=torch.bool, device=device)
        mask[0, 0, :, n // 2 :] = False

        captured = []
        attention.out_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        with torch.no_grad():
            attention(x, cos, sin, attention_mask=mask)
            attention(x, cos, sin)
        masked, unmasked = (c.view(2, n, heads, head_dim) for c in captured)
        assert not torch.allclose(masked[:, :, 0], unmasked[:, :, 0])
        self.assert_close(masked[:, :, 1:], unmasked[:, :, 1:])

    @pytest.mark.parametrize("batch_size", [2, 16])
    def test_moonvit_attention_two_dim_mask_is_a_padding_mask(self, device, dtype, config, batch_size):
        # A (B, N) mask is a padding mask over the keys, read as (B, 1, 1, N) as SigLip2Attention reads it. Its leading
        # dimension is the batch size also when B == N (16 here), where scaled_dot_product_attention would take it for
        # an N x N query x key mask.
        attention = MoonViTAttention(config.vision_config).to(device, dtype).eval()
        n = 16
        x, cos, sin = _moonvit_attention_inputs(config, batch_size, n, device, dtype)
        mask = torch.ones(batch_size, n, dtype=torch.bool, device=device)
        mask[1, n // 2 :] = False  # sample 1 attends to the first half of the keys only

        with torch.no_grad():
            output = attention(x, cos, sin, attention_mask=mask)
            self.assert_close(output, attention(x, cos, sin, attention_mask=mask[:, None, None, :]))
            self.assert_close(output[1:2], attention(x[1:2], cos, sin, attention_mask=mask[1:2]))

    def test_moonvit_attention_mask_with_a_leading_one_is_shared_by_the_batch(self, device, dtype, config):
        # A (1, N) padding mask and a (1, N, N) mask broadcast over the batch; (1, 1, N, N) is the 4-D form of the
        # latter.
        attention = MoonViTAttention(config.vision_config).to(device, dtype).eval()
        n = 16
        x, cos, sin = _moonvit_attention_inputs(config, 2, n, device, dtype)
        keys = torch.ones(1, n, dtype=torch.bool, device=device)
        keys[:, n // 2 :] = False
        pairs = torch.ones(1, n, n, dtype=torch.bool, device=device).tril()

        with torch.no_grad():
            self.assert_close(
                attention(x, cos, sin, attention_mask=keys), attention(x, cos, sin, attention_mask=keys.expand(2, n))
            )
            self.assert_close(
                attention(x, cos, sin, attention_mask=pairs), attention(x, cos, sin, attention_mask=pairs[None])
            )

    def test_moonvit_attention_reads_an_integer_mask_as_boolean(self, device, dtype, config):
        # A 0/1 integer mask, such as a tokenizer's int64 padding mask, is read as the boolean mask; a float mask is
        # added to the scores, as scaled_dot_product_attention reads it.
        attention = MoonViTAttention(config.vision_config).to(device, dtype).eval()
        n = 16
        x, cos, sin = _moonvit_attention_inputs(config, 2, n, device, dtype)
        mask = torch.ones(2, n, dtype=torch.bool, device=device)
        mask[1, n // 2 :] = False
        additive = torch.zeros(2, n, device=device, dtype=dtype).masked_fill(~mask, float("-inf"))

        with torch.no_grad():
            expected = attention(x, cos, sin, attention_mask=mask)
            self.assert_close(attention(x, cos, sin, attention_mask=mask.long()), expected)
            self.assert_close(attention(x, cos, sin, attention_mask=additive), expected)

    def test_moonvit_rotary_embedding(self, device, dtype, config):
        dim = config.vision_config.hidden_size // config.vision_config.num_attention_heads
        rope = MoonViTRotaryEmbedding(dim).to(device, dtype)

        h, w = 8, 8
        cos, sin = rope(h, w, device)

        seq_len = h * w
        assert cos.shape == (seq_len, dim // 2)
        assert sin.shape == (seq_len, dim // 2)

    def test_projector(self, device, dtype, config):
        model = KimiVLProjector(config.projector_config).to(device, dtype)
        batch_size = 2
        input_features = torch.randn(batch_size, 64, 32, device=device, dtype=dtype)

        output = model(input_features, h=8, w=8)
        assert output.shape == (batch_size, 16, config.projector_config.output_dim)

    def test_projector_patch_concatenation_order(self, device):
        # Reference: Moonshot's KimiVLMultiModalProjector at revision
        # 398eede0903cd983a2bfa0cc634e9ac1d843f375. Values 0..15 form a row-major 2x4 grid
        # of two-feature patches. When each 2x2 block is merged, the features from each patch must stay
        # adjacent, with patches ordered top-left, top-right, bottom-left, bottom-right.
        config = KimiVLProjectorConfig(input_dim=2, hidden_dim=8, output_dim=8)
        projector = KimiVLProjector(config).to(device)
        projector.pre_norm = torch.nn.Identity()
        projector.mlp = torch.nn.Identity()
        patches = torch.arange(16.0, device=device).view(1, 8, 2)
        expected = torch.tensor(
            [[[0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0], [4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0]]],
            device=device,
        )

        actual = projector(patches, h=2, w=4)

        self.assert_close(actual, expected, rtol=0.0, atol=0.0)

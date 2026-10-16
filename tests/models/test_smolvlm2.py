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

"""Tests for the SmolVLM2 vision-language model."""

import pytest
import torch

from kornia.models.siglip2.config import SigLip2VisionConfig
from kornia.models.smolvlm2 import (
    SmolVLM2Config,
    SmolVLM2Connector,
    SmolVLM2ForConditionalGeneration,
    SmolVLM2Model,
    SmolVLM2TextConfig,
)
from kornia.models.smolvlm2.modeling_smolvlm2 import SmolVLM2TextModel

from testing.base import BaseTester


@pytest.fixture
def config():
    """Tiny SmolVLM2 config: an 8x8 patch grid so scale_factor=2 gives 16 image tokens."""
    vision_config = SigLip2VisionConfig(
        image_size=64,
        patch_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    text_config = SmolVLM2TextConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    return SmolVLM2Config(vision_config=vision_config, text_config=text_config, scale_factor=2, image_token_id=7)


def _image_tokens_per_image(config):
    grid = config.vision_config.image_size // config.vision_config.patch_size
    return grid * grid // (config.scale_factor**2)


def _make_inputs(config, device, dtype, num_images=1, num_text_tokens=4):
    """Build (input_ids, pixel_values) with <image> blocks followed by text tokens."""
    img_seq = _image_tokens_per_image(config)
    ids = [1] + [config.image_token_id] * (img_seq * num_images) + list(range(10, 10 + num_text_tokens))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    size = config.vision_config.image_size
    pixel_values = torch.randn(num_images, 3, size, size, device=device, dtype=dtype)
    return input_ids, pixel_values


class TestSmolVLM2Connector(BaseTester):
    """Test suite for the pixel-shuffle connector."""

    def test_smoke(self, device, dtype, config):
        connector = SmolVLM2Connector(config).to(device, dtype)
        assert connector is not None

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("scale_factor", [2, 3])
    def test_cardinality(self, device, dtype, config, batch_size, scale_factor):
        """Token count divides by scale_factor**2; channels multiply by scale_factor**2."""
        grid = 6  # divisible by both scale factors
        embed_dim = config.vision_config.hidden_size
        config.scale_factor = scale_factor
        connector = SmolVLM2Connector(config).to(device, dtype)
        x = torch.randn(batch_size, grid * grid, embed_dim, device=device, dtype=dtype)
        shuffled = connector.pixel_shuffle(x, scale_factor)
        assert shuffled.shape == (batch_size, grid * grid // scale_factor**2, embed_dim * scale_factor**2)
        projected = connector(x)
        assert projected.shape == (
            batch_size,
            grid * grid // scale_factor**2,
            config.text_config.hidden_size,
        )

    def test_pixel_shuffle_values(self, device, dtype, config):
        """Pixel shuffle must match the HuggingFace SmolVLM reference implementation.

        Each output token folds a ``scale_factor x scale_factor`` spatial block of
        patches into the channel dimension (space-to-depth).
        """
        connector = SmolVLM2Connector(config).to(device, dtype)
        x = torch.arange(32, device=device, dtype=dtype).reshape(1, 16, 2)
        # Snippet used to generate expected (transformers.models.smolvlm.modeling_smolvlm
        # SmolVLMConnector.pixel_shuffle, requires torch only):
        # x = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)
        # expected = SmolVLMConnector.pixel_shuffle(self=None-like, x, scale_factor=2)  # <-- print and paste below
        expected = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0],
                    [4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0, 24.0, 25.0, 26.0, 27.0],
                    [20.0, 21.0, 22.0, 23.0, 28.0, 29.0, 30.0, 31.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(connector.pixel_shuffle(x, 2), expected)


class TestSmolVLM2TextModel(BaseTester):
    """Test suite for the SmolLM2 (Llama-architecture) text decoder."""

    def test_smoke(self, device, dtype, config):
        model = SmolVLM2TextModel(config.text_config).to(device, dtype)
        assert model is not None

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, device, dtype, config, batch_size):
        model = SmolVLM2TextModel(config.text_config).to(device, dtype).eval()
        input_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, 9), device=device)
        with torch.no_grad():
            out = model(input_ids)
        assert out.shape == (batch_size, 9, config.text_config.hidden_size)

    def test_exception(self, device, dtype, config):
        model = SmolVLM2TextModel(config.text_config).to(device, dtype).eval()
        input_ids = torch.randint(0, config.text_config.vocab_size, (1, 4), device=device)
        embeds = torch.randn(1, 4, config.text_config.hidden_size, device=device, dtype=dtype)
        # neither / both of input_ids and inputs_embeds
        with pytest.raises(ValueError):
            model()
        with pytest.raises(ValueError):
            model(input_ids=input_ids, inputs_embeds=embeds)
        # bad attention mask rank
        with pytest.raises(ValueError):
            model(input_ids=input_ids, attention_mask=torch.ones(1, 1, 4, device=device))

    def test_causality(self, device, dtype, config):
        """Perturbing a future token must not change earlier hidden states."""
        torch.manual_seed(0)
        model = SmolVLM2TextModel(config.text_config).to(device, dtype).eval()
        ids_a = torch.randint(1, config.text_config.vocab_size, (2, 8), device=device)
        ids_b = ids_a.clone()
        ids_b[:, 5] = (ids_b[:, 5] + 1) % config.text_config.vocab_size
        mask = torch.ones(2, 8, dtype=torch.long, device=device)
        mask[0, 6:] = 0
        with torch.no_grad():
            out_a = model(ids_a)
            out_b = model(ids_b)
            out_am = model(ids_a, attention_mask=mask)
            out_bm = model(ids_b, attention_mask=mask)
        # is_causal fast path (no mask) and explicit padding-mask path
        self.assert_close(out_a[:, :5], out_b[:, :5])
        self.assert_close(out_am[:, :5], out_bm[:, :5])
        assert not torch.allclose(out_a[:, 5:], out_b[:, 5:])

    def test_padding_mask_consistency(self, device, dtype, config):
        """An all-ones padding mask must reproduce the no-mask output, without NaNs."""
        torch.manual_seed(0)
        model = SmolVLM2TextModel(config.text_config).to(device, dtype).eval()
        input_ids = torch.randint(1, config.text_config.vocab_size, (2, 8), device=device)
        with torch.no_grad():
            no_mask = model(input_ids)
            ones_mask = model(input_ids, attention_mask=torch.ones(2, 8, dtype=torch.long, device=device))
            right_padded = model(input_ids, attention_mask=torch.tensor([[1] * 8, [1] * 5 + [0] * 3], device=device))
            # left padding fully masks the leading query rows; SDPA must return zeros there, not NaNs
            left_padded = model(input_ids, attention_mask=torch.tensor([[1] * 8, [0] * 3 + [1] * 5], device=device))
        self.assert_close(no_mask, ones_mask)
        assert torch.isfinite(right_padded).all()
        assert torch.isfinite(left_padded).all()
        # valid rows are unaffected by the padding of the other rows
        self.assert_close(right_padded[1, :5], no_mask[1, :5])

    def test_rope_position_zero_identity(self, device, dtype, config):
        """At position 0 the rotary embedding is the identity (cos=1, sin=0)."""
        model = SmolVLM2TextModel(config.text_config).to(device, dtype)
        x = torch.randn(1, 1, config.text_config.hidden_size, device=device, dtype=dtype)
        cos, sin = model.rotary_emb(x, torch.zeros(1, 1, dtype=torch.long, device=device))
        self.assert_close(cos, torch.ones_like(cos))
        self.assert_close(sin, torch.zeros_like(sin))

    def test_rope_autocast_high_position(self, device, config):
        """Autocast must not change the rotary factors: the frequencies are evaluated in float32.

        Position 0 cannot catch this (cos=1, sin=0 for every frequency), so use the
        checkpoint geometry (head_dim=64, rope_theta=130000) at large positions, where a
        half-precision matmul perturbs cos/sin by O(1).
        """
        from kornia.models.smolvlm2.modeling_smolvlm2 import SmolVLM2RotaryEmbedding

        rope = SmolVLM2RotaryEmbedding(64, base=130000.0).to(device)
        x = torch.zeros(1, 1, 64, device=device)
        position_ids = torch.tensor([[1000, 8191]], device=device)
        cos, sin = rope(x, position_ids)
        autocast_dtype = torch.float16 if device.type == "mps" else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            cos_ac, sin_ac = rope(x, position_ids)
        assert cos_ac.dtype == cos.dtype
        assert sin_ac.dtype == sin.dtype
        self.assert_close(cos_ac, cos, rtol=0.0, atol=0.0)
        self.assert_close(sin_ac, sin, rtol=0.0, atol=0.0)

    def test_build_attention_mask(self, device, dtype, config):
        """The mask is exactly ``causal & key_is_valid``: no self-attention entry is added to padded rows."""
        model = SmolVLM2TextModel(config.text_config).to(device, dtype)
        embeds = torch.zeros(3, 4, config.text_config.hidden_size, device=device, dtype=dtype)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]], device=device)
        mask = model._build_attention_mask(attention_mask, embeds)
        assert mask.shape == (3, 1, 4, 4)
        causal = torch.tril(torch.ones(4, 4, dtype=torch.bool, device=device))
        expected = causal[None] & attention_mask[:, None, :].bool()
        assert torch.equal(mask[:, 0], expected)
        # right padding: padded query rows keep the valid keys only; left padding: they are fully masked
        assert not mask[1, 0, 2, 2]
        assert not mask[1, 0, 3, 3]
        assert not mask[2, 0, :2].any()

    def test_rope_reference_values(self, device, dtype):
        """Rotary factors at high positions must match ``transformers``' ``LlamaRotaryEmbedding``.

        Checkpoint geometry (head_dim=64, rope_theta=130000) at positions 1000 and 8191; the
        first four frequencies are pinned (the second half of the last dim repeats the first).
        """
        from kornia.models.smolvlm2.modeling_smolvlm2 import SmolVLM2RotaryEmbedding

        rope = SmolVLM2RotaryEmbedding(64, base=130000.0).to(device)
        x = torch.zeros(1, 2, 64, device=device, dtype=dtype)
        cos, sin = rope(x, torch.tensor([[1000, 8191]], device=device))
        # Generated with transformers 5.20.1 / torch 2.9.1 (CPU float32):
        #   from transformers import LlamaConfig
        #   from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        #   cfg = LlamaConfig(hidden_size=2048, num_attention_heads=32, rope_theta=130000.0,
        #                     max_position_embeddings=8192)
        #   cos, sin = LlamaRotaryEmbedding(config=cfg)(torch.zeros(1, 2, 64), torch.tensor([[1000, 8191]]))
        #   print(cos[0, :, :4].tolist(), sin[0, :, :4].tolist())
        expected_cos = torch.tensor(
            [[0.56237906, 0.55520695, 0.04537873, 0.12597054], [-0.64639050, -0.25162351, -0.99958801, 0.06330082]],
            device=device,
            dtype=dtype,
        )
        expected_sin = torch.tensor(
            [[0.82687956, 0.83171225, 0.99896985, -0.99203396], [-0.76300681, 0.96782517, -0.02870141, 0.99799448]],
            device=device,
            dtype=dtype,
        )
        assert cos.shape == (1, 2, 64)
        self.assert_close(cos[0, :, :4], expected_cos)
        self.assert_close(sin[0, :, :4], expected_sin)
        self.assert_close(cos[0, :, 32:36], expected_cos)
        self.assert_close(sin[0, :, 32:36], expected_sin)

    def test_reference_values(self, device, dtype):
        """Decoder output must match ``transformers``' ``LlamaModel`` with the same weights.

        A tiny GQA decoder with non-unit RMSNorm weights, evaluated at non-zero positions
        (including 1000 and 8191), so RoPE, attention scaling and the norm weights all
        affect the pinned values. The literal is a float32 reference; half-precision runs
        of a decoder at position 8191 cannot reproduce it to a meaningful tolerance.
        """
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("float32 reference literal; half dtypes are covered by the consistency tests")
        text_config = SmolVLM2TextConfig(
            vocab_size=50,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            rope_theta=130000.0,
            pad_token_id=0,
        )
        model = SmolVLM2TextModel(text_config).eval()
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm" in name:  # non-unit RMSNorm weights, as a loaded checkpoint has
                    param.copy_(1.0 + 0.5 * torch.randn(param.shape, generator=generator))
                else:
                    param.copy_(0.2 * torch.randn(param.shape, generator=generator))
        model = model.to(device, dtype)
        input_ids = torch.tensor([[3, 17, 42, 8, 29]], device=device)
        position_ids = torch.tensor([[0, 3, 7, 1000, 8191]], device=device)
        with torch.no_grad():
            out = model(input_ids, position_ids=position_ids)
        # Generated with transformers 5.20.1 / torch 2.9.1 (CPU float32) from the model above:
        #   from transformers import LlamaConfig, LlamaModel
        #   hf = LlamaModel(LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32,
        #       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        #       max_position_embeddings=64, rope_theta=130000.0, rms_norm_eps=1e-5, pad_token_id=0,
        #       attention_bias=False, mlp_bias=False)).eval()
        #   hf.load_state_dict(model.state_dict(), strict=True)
        #   print(hf(input_ids=input_ids, position_ids=position_ids).last_hidden_state[0].tolist())
        # (kornia's output was bit-identical: max_abs_diff = 0.0)
        expected = torch.tensor(
            [
                [
                    -0.0941252,
                    0.3417611,
                    0.0841559,
                    0.4340158,
                    -0.2318656,
                    -0.1717177,
                    0.0553885,
                    0.0354352,
                    0.6823664,
                    0.7307467,
                    2.4059160,
                    1.5216061,
                    0.1641527,
                    -1.4558055,
                    -1.5543094,
                    -1.6000885,
                ],
                [
                    -0.9368023,
                    0.4020294,
                    0.0729017,
                    0.5702167,
                    -1.8820920,
                    0.5415824,
                    0.0546819,
                    -0.1857364,
                    0.2059336,
                    0.0429058,
                    1.1661642,
                    0.2983623,
                    0.7493295,
                    -1.0293766,
                    -0.3516322,
                    -1.8499951,
                ],
                [
                    -0.1160356,
                    0.3339780,
                    0.0619010,
                    -0.7627104,
                    -1.7834356,
                    1.0424188,
                    0.0643094,
                    0.1514155,
                    -0.4440383,
                    1.2530490,
                    1.0009120,
                    1.4765549,
                    -0.1022064,
                    -1.9070032,
                    -1.2815402,
                    -1.3064994,
                ],
                [
                    -1.3200343,
                    0.3595729,
                    0.0334141,
                    0.3876024,
                    -1.4619470,
                    1.3938670,
                    0.0529109,
                    -0.4031243,
                    0.0236761,
                    0.2925563,
                    0.5894080,
                    1.1258978,
                    0.5583172,
                    -1.0668886,
                    -2.0851493,
                    -1.5217695,
                ],
                [
                    -0.7788236,
                    0.3013026,
                    0.0035897,
                    0.3441010,
                    -0.5207199,
                    1.6926521,
                    0.0612411,
                    -0.3969580,
                    -0.3139853,
                    0.6972899,
                    0.4218283,
                    1.7061920,
                    0.7282952,
                    -0.7518105,
                    -1.8844296,
                    -1.8310552,
                ],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out[0], expected)

    def test_gradcheck(self, device, config):
        """Gradients must flow through the decoder to the token embeddings."""
        model = SmolVLM2TextModel(config.text_config).to(device)
        input_ids = torch.randint(0, config.text_config.vocab_size, (1, 4), device=device)
        out = model(input_ids)
        out.sum().backward()
        assert model.embed_tokens.weight.grad is not None
        assert torch.isfinite(model.embed_tokens.weight.grad).all()


class TestSmolVLM2Model(BaseTester):
    """Test suite for the full SmolVLM2 model."""

    def test_smoke(self, device, dtype, config):
        model = SmolVLM2Model(config).to(device, dtype)
        assert model is not None

    @pytest.mark.parametrize("num_images", [1, 2])
    def test_cardinality(self, device, dtype, config, num_images):
        model = SmolVLM2ForConditionalGeneration(config).to(device, dtype).eval()
        input_ids, pixel_values = _make_inputs(config, device, dtype, num_images=num_images)
        with torch.no_grad():
            logits = model(input_ids, pixel_values=pixel_values)
        assert logits.shape == (1, input_ids.shape[1], config.text_config.vocab_size)
        assert torch.isfinite(logits).all()

    def test_exception(self, device, dtype, config):
        """Mismatched <image> token count vs image features must raise."""
        model = SmolVLM2Model(config).to(device, dtype).eval()
        input_ids, pixel_values = _make_inputs(config, device, dtype, num_images=1)
        # two images worth of pixels but one image worth of <image> tokens
        bad_pixels = torch.cat([pixel_values, pixel_values], dim=0)
        with pytest.raises(ValueError):
            model(input_ids, pixel_values=bad_pixels)

    def test_get_image_features(self, device, dtype, config):
        model = SmolVLM2Model(config).to(device, dtype).eval()
        size = config.vision_config.image_size
        pixel_values = torch.randn(2, 3, size, size, device=device, dtype=dtype)
        with torch.no_grad():
            features = model.get_image_features(pixel_values)
        assert features.shape == (2, _image_tokens_per_image(config), config.text_config.hidden_size)

    def test_inputs_merger(self, device, dtype, config):
        """Image features must land exactly at the <image> positions, text embeddings elsewhere."""
        model = SmolVLM2Model(config).to(device, dtype).eval()
        img_seq = _image_tokens_per_image(config)
        input_ids = torch.tensor([[1] + [config.image_token_id] * img_seq + [10, 11]], device=device)
        inputs_embeds = model.text_model.embed_tokens(input_ids)
        image_hidden_states = torch.randn(1, img_seq, config.text_config.hidden_size, device=device, dtype=dtype)
        merged = model.inputs_merger(input_ids, inputs_embeds, image_hidden_states)
        image_mask = input_ids == config.image_token_id
        self.assert_close(merged[image_mask], image_hidden_states.reshape(-1, config.text_config.hidden_size))
        self.assert_close(merged[~image_mask], inputs_embeds[~image_mask])

    def test_text_only(self, device, dtype, config):
        """The model must run without pixel_values (text-only path)."""
        model = SmolVLM2Model(config).to(device, dtype).eval()
        input_ids = torch.tensor([[1, 10, 11, 12]], device=device)
        with torch.no_grad():
            out = model(input_ids)
        assert out.shape == (1, 4, config.text_config.hidden_size)

    def test_dynamo(self, device, dtype, torch_optimizer, config):
        """Test torch.compile compatibility: compiled outputs must match eager."""
        torch.manual_seed(0)
        model = SmolVLM2ForConditionalGeneration(config).to(device, dtype).eval()
        input_ids, pixel_values = _make_inputs(config, device, dtype)
        model_optimized = torch_optimizer(model)
        with torch.no_grad():
            expected = model(input_ids, pixel_values=pixel_values)
            actual = model_optimized(input_ids, pixel_values=pixel_values)
        self.assert_close(actual, expected)

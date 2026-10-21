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

"""Tests for SigLip2 model."""

import math
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import kornia
import kornia.models.siglip2.builder as siglip2_builder
from kornia.core.exceptions import BaseError, ShapeError
from kornia.models.siglip2 import SigLip2Builder, SigLip2Config, SigLip2Model, SigLip2Result
from kornia.models.siglip2.attention import SigLip2Attention
from kornia.models.siglip2.config import SigLip2TextConfig, SigLip2VisionConfig
from kornia.models.siglip2.preprocessor import SigLip2ImagePreprocessor
from kornia.models.siglip2.text_encoder import SigLip2TextEmbeddings, SigLip2TextEncoder, SigLip2TextModel
from kornia.models.siglip2.vision_encoder import (
    SigLip2MultiheadAttentionPoolingHead,
    SigLip2VisionEmbeddings,
    SigLip2VisionEncoder,
    SigLip2VisionModel,
)

from testing.base import BaseTester, supports_reflect_padding


@pytest.fixture
def config():
    """Fixture for SigLip2Config."""
    return SigLip2Config(
        vision_config=SigLip2VisionConfig(
            image_size=32,
            patch_size=4,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
        ),
        text_config=SigLip2TextConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
        ),
        projection_dim=16,
    )


@pytest.fixture
def model(device, dtype, config):
    """Fixture for SigLip2Model."""
    return SigLip2Model(config).to(device, dtype).eval()


def _create_input_ids(batch_size, seq_len, config, device):
    """Create input_ids with smaller range to avoid memory issues with large vocab."""
    return torch.randint(0, min(100, config.text_config.vocab_size), (batch_size, seq_len), device=device)


def _from_name_or_error(model_name):
    """``SigLip2Config.from_name(model_name)``, or the type of the error it raises, to compare two names' readings."""
    try:
        return SigLip2Config.from_name(model_name)
    except ValueError as error:
        return type(error)


def _tiny_vision_config():
    """32 px images at patch 4: an 8 x 8 grid of 64 patches."""
    return SigLip2VisionConfig(
        image_size=32,
        patch_size=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )


def _patch_rows_4_to_7_masked(device):
    """A ``(1, 64)`` mask of the 8 x 8 grid that drops patches 32..63, i.e. patch rows 4..7 (pixel rows 16..31)."""
    mask = torch.ones(1, 64, dtype=torch.bool, device=device)
    mask[:, 32:] = False
    return mask


# (vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, image_size) of each supported
# checkpoint: its hub config.json (read 2026-10-16), with the defaults of transformers v4.51.0 SiglipTextConfig and
# SiglipVisionConfig (configuration_siglip.py: vocab 32000, hidden 768, intermediate 3072, 12 layers, 12 heads, image
# 224, patch 16, text max_position_embeddings 64) for the fields it omits. No config.json sets a patch size or
# max_position_embeddings, so every name has patch 16 and 64 text positions.
_HUB_CONFIGS = {
    "google/siglip-base-patch16-224": (32000, 768, 3072, 12, 12, 224),
    "google/siglip2-base-patch16-224": (256000, 768, 3072, 12, 12, 224),
    "google/siglip2-base-patch16-256": (256000, 768, 3072, 12, 12, 256),
    "google/siglip2-base-patch16-384": (256000, 768, 3072, 12, 12, 384),
    "google/siglip2-base-patch16-512": (256000, 768, 3072, 12, 12, 512),
    "google/siglip2-large-patch16-256": (256000, 1024, 4096, 24, 16, 256),
    "google/siglip2-large-patch16-384": (256000, 1024, 4096, 24, 16, 384),
    "google/siglip2-large-patch16-512": (256000, 1024, 4096, 24, 16, 512),
}


class TestSigLip2Builder:
    """The download path: kornia's own downloader and safetensors reader, no optional package."""

    def test_download_weights_uses_kornias_downloader(self, tmp_path):
        expected = {"weight": torch.zeros(1)}
        with (
            patch.object(siglip2_builder, "download_hf_file", return_value="cached.safetensors") as download,
            patch.object(siglip2_builder, "load_safetensors", return_value=expected) as read,
        ):
            state_dict = siglip2_builder._download_weights("google/siglip2-base-patch16-224", str(tmp_path))

        assert state_dict is expected
        read.assert_called_once_with("cached.safetensors")
        # The repository reaches the downloader whole, so its cache name carries
        # the owner; every HF repo publishes a ``model.safetensors``.
        download.assert_called_once_with(
            "google/siglip2-base-patch16-224",
            "model.safetensors",
            model_dir=str(tmp_path),
            validate=siglip2_builder.check_safetensors,
        )

    def test_the_variant_reaches_the_downloader(self, tmp_path):
        """Two variants must not resolve to one download, and so to one cache entry."""
        repos = []
        with (
            patch.object(siglip2_builder, "download_hf_file", return_value="cached.safetensors") as download,
            patch.object(siglip2_builder, "load_safetensors", return_value={}),
        ):
            for model_name in ("google/siglip2-base-patch16-224", "google/siglip2-base-patch16-256"):
                siglip2_builder._download_weights(model_name, None)
                repos.append(download.call_args.args[0])

        assert repos == ["google/siglip2-base-patch16-224", "google/siglip2-base-patch16-256"]

    def test_from_pretrained_hf_returns_eval_mode(self, monkeypatch, config):
        monkeypatch.setattr(SigLip2Config, "from_name", staticmethod(lambda model_name: config))
        state_dict = SigLip2Model(config).state_dict()
        monkeypatch.setattr(siglip2_builder, "_download_weights", lambda model_name, cache_dir: state_dict)

        # ``training`` of the model and of every submodule: ``{False}`` is evaluation mode throughout.
        weighted = SigLip2Builder.from_pretrained_hf("google/siglip2-base-patch16-224")
        assert {m.training for m in weighted.modules()} == {False}
        named = SigLip2Builder.from_name("google/siglip2-base-patch16-224")
        assert {m.training for m in named.modules()} == {True}
        assert {m.training for m in SigLip2Builder.from_config(config).modules()} == {True}

    def test_a_missing_checkpoint_is_reported_with_the_model_name(self, tmp_path):
        with (
            patch.object(siglip2_builder, "download_hf_file", return_value="cached.safetensors"),
            patch.object(siglip2_builder, "load_safetensors", side_effect=FileNotFoundError("gone")),
            pytest.raises(FileNotFoundError, match=r"Could not find model\.safetensors for google/nope"),
        ):
            siglip2_builder._download_weights("google/nope", None)

    @pytest.mark.parametrize("model_name", sorted(_HUB_CONFIGS))
    def test_from_name_table(self, model_name):
        # Config only: building the model would allocate the full random network (about 375M parameters for base).
        vocab_size, hidden_size, intermediate_size, num_layers, num_heads, image_size = _HUB_CONFIGS[model_name]
        config = SigLip2Config.from_name(model_name)

        for sub_config in (config.vision_config, config.text_config):
            assert sub_config.hidden_size == hidden_size
            assert sub_config.intermediate_size == intermediate_size
            assert sub_config.num_hidden_layers == num_layers
            assert sub_config.num_attention_heads == num_heads
        assert config.vision_config.patch_size == 16
        assert config.vision_config.image_size == image_size
        assert config.text_config.vocab_size == vocab_size
        assert config.text_config.max_position_embeddings == 64
        assert config.projection_dim == hidden_size

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            # (vocab_size, hidden_size, num_hidden_layers, num_attention_heads, image_size, patch_size, text positions)
            ("someone/siglip2-base-patch16-224-finetuned", (256000, 768, 12, 12, 224, 16, 512)),
            ("someone/siglip2-large-patch16-384-mirror", (256000, 1024, 24, 16, 384, 16, 512)),
            ("someone/my-siglip", (256000, 768, 12, 12, 224, 16, 512)),
            # an image size counts only as a whole number, of the repository part after the last "/" first
            ("org2560/siglip2-base-patch16-384", (256000, 768, 12, 12, 384, 16, 512)),
            ("org-256/siglip2-base-patch16-384", (256000, 768, 12, 12, 384, 16, 512)),
            ("x/siglip2-base-patch16-ckpt2560", (256000, 768, 12, 12, 224, 16, 512)),
            ("x/siglip2-base-patch16-224-ckpt2560", (256000, 768, 12, 12, 224, 16, 512)),
            # the owner part counts when the repository part has no size
            ("siglip384/finetune", (256000, 768, 12, 12, 384, 16, 512)),
            # names are read in any case, and what the owner part says about the architecture does not count
            ("someone/SigLIP2-Large-Patch16-256", (256000, 1024, 24, 16, 256, 16, 512)),
            ("giantlab/siglip2-base-patch16-224-ft", (256000, 768, 12, 12, 224, 16, 512)),
            ("patch14ai/siglip2-base-patch16-224", (256000, 768, 12, 12, 224, 16, 512)),
            # a fine-tune or mirror of the supported SigLIP 1 checkpoint has its 32000-token vocabulary
            ("someone/siglip-base-patch16-224-finetuned", (32000, 768, 12, 12, 224, 16, 512)),
            ("someone/SigLIP-Base-Patch16-224", (32000, 768, 12, 12, 224, 16, 512)),
        ],
    )
    def test_from_name_reads_an_unlisted_name_from_its_parts(self, model_name, expected):
        # A fine-tune or mirror of a supported checkpoint under another name: base unless a word of the repository
        # part is "large", the image size the name contains (224 if none), patch 16, the SigLIP 2 vocabulary, 512 text
        # positions (which from_pretrained_hf adapts to the checkpoint). Config only: no model is built.
        config = SigLip2Config.from_name(model_name)
        vision, text = config.vision_config, config.text_config
        actual = (
            text.vocab_size,
            vision.hidden_size,
            vision.num_hidden_layers,
            vision.num_attention_heads,
            vision.image_size,
            vision.patch_size,
            text.max_position_embeddings,
        )
        assert actual == expected
        assert (text.hidden_size, text.num_hidden_layers, text.num_attention_heads) == expected[1:4]

    @pytest.mark.parametrize(
        "model_name",
        [
            # #6004: so400m has hidden size 1152 and patch 14, giant is wider still, and the SigLIP 1 checkpoints have
            # the 32000-token vocabulary; the name parser would build a base patch-16 SigLIP 2 model for each.
            "google/siglip2-so400m-patch14-384",
            "google/siglip2-giant-opt-patch16-384",
            "google/siglip-large-patch16-256",
            "google/siglip-base-patch16-384",
            "google/siglip2-base-patch32-256",
            "google/siglip2-base-patch16-naflex",
            # the same architectures under another owner
            "someone/siglip2-so400m-patch14-384-finetuned",
            "someone/siglip2-giant-opt-patch16-256-mirror",
            "someone/siglip2-base-patch32-256-mirror",
            "someone/siglip2-base-patch16-naflex-ft",
            # SigLIP 1 checkpoints other than the supported base 224 one, under another owner
            "someone/siglip-large-patch16-256",
            "someone/siglip-base-patch16-384",
            "someone/siglip-base-patch16-256-multilingual",
            "Someone/SigLIP-Large-Patch16-256-mirror",
        ],
    )
    def test_from_name_rejects_a_name_it_cannot_read(self, model_name):
        # A google/ checkpoint outside the table, or a repository part with the word so400m, giant, naflex or a patch
        # size other than 16, names an architecture the parts reading cannot build: it raises rather than return a
        # model of the wrong shape. Neither call builds a model.
        with pytest.raises(ValueError, match=r"cannot build .* Supported names: google/siglip-base-patch16-224, "):
            SigLip2Config.from_name(model_name)
        with pytest.raises(ValueError, match="cannot build"):
            SigLip2Builder.from_name(model_name)

    @pytest.mark.parametrize(
        "repository",
        [
            "siglip2-base-patch16-224-finetuned",
            "siglip2-large-patch16-384-mirror",
            "my-siglip",
            "siglip-base-patch16-224-finetuned",
            "siglip2-so400m-patch14-384-finetuned",
            "siglip2-base-patch32-256-mirror",
            "siglip-large-patch16-256",
        ],
    )
    @pytest.mark.parametrize("owner", ["LargeCorp", "large-models", "giantlab", "so400m-fans", "naflex", "patch14ai"])
    def test_from_name_reads_the_architecture_from_the_repository_part_only(self, repository, owner):
        # An owner that says large, giant, so400m, naflex or a patch size gets the reading of a neutral owner, for a
        # name that reads and for one that raises alike ("LargeCorp/siglip2-base-..." was the large architecture).
        assert _from_name_or_error(f"{owner}/{repository}") == _from_name_or_error(f"someone/{repository}")

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            # (hidden_size, image_size), and patch size 16 and the SigLIP 2 vocabulary throughout: a marker inside a
            # longer word of the repository part is not the marker
            ("someone/siglip2-base-patch16-224-enlarged", (768, 224)),
            ("someone/siglip2-base-patch16-224-giantdata", (768, 224)),
            ("someone/siglip2-base-patch16-224-naflexible", (768, 224)),
            ("someone/siglip2-base-patch16-224-so400mx", (768, 224)),
            ("someone/dispatch2-siglip2-base-patch16-224", (768, 224)),
            ("someone/notsiglip-large-patch16-256", (1024, 256)),
            # words are separated by any character that is not a letter or a digit
            ("someone/siglip2_large_patch16_384", (1024, 384)),
            ("someone/SigLIP2.Large.Patch16.512", (1024, 512)),
        ],
    )
    def test_from_name_reads_whole_words(self, model_name, expected):
        config = SigLip2Config.from_name(model_name)
        assert (config.vision_config.hidden_size, config.vision_config.image_size) == expected
        assert (config.vision_config.patch_size, config.text_config.vocab_size) == (16, 256000)

    @pytest.mark.parametrize(
        "model_name", ["someone/siglip2_so400m_patch14_384", "someone/siglip2-base.patch32", "someone/siglip_large_ft"]
    )
    def test_from_name_rejects_whole_words_after_any_separator(self, model_name):
        with pytest.raises(ValueError, match="cannot build"):
            SigLip2Config.from_name(model_name)

    @pytest.mark.parametrize(
        "model_name",
        [
            "someone/siglip2-base-patch16-224-finetuned",
            "someone/siglip2-large-patch16-384-mirror",
            "someone/siglip-base-patch16-224-finetuned",
            "org-256/siglip2-finetune",
            "someone/siglip2-so400m-patch14-384",
            "someone/siglip-large-patch16-256",
        ],
    )
    def test_from_name_reads_an_unlisted_name_in_any_case(self, model_name):
        expected = _from_name_or_error(model_name)
        assert _from_name_or_error(model_name.upper()) == expected
        assert _from_name_or_error(model_name.swapcase().replace("SOMEONE", "Someone")) == expected

    @pytest.mark.parametrize("model_name", ["Google/SigLIP2-Large-Patch16-384", "GOOGLE/siglip-base-patch16-224"])
    def test_from_name_table_ignores_case(self, model_name):
        # A supported checkpoint named in another case is that checkpoint, with its 64 text positions.
        assert SigLip2Config.from_name(model_name) == SigLip2Config.from_name(model_name.lower())

    def test_from_pretrained_hf_loads_an_unlisted_name_that_fits(self, monkeypatch, config):
        # The tiny config stands in for the configuration read from the name, and its own weights for a fine-tune's.
        monkeypatch.setattr(SigLip2Config, "from_name", staticmethod(lambda model_name: config))
        state_dict = SigLip2Model(config).state_dict()
        monkeypatch.setattr(siglip2_builder, "_download_weights", lambda model_name, cache_dir: state_dict)

        model = SigLip2Builder.from_pretrained_hf("someone/siglip2-base-patch16-224-finetuned")

        assert {m.training for m in model.modules()} == {False}
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, state_dict[name])

    def test_from_pretrained_hf_names_the_supported_list_when_an_unlisted_checkpoint_does_not_fit(
        self, monkeypatch, config
    ):
        # Weights of a wider model (hidden size 48 instead of 32) stand in for a large checkpoint under a name that
        # reads as base: load_state_dict's shape mismatch becomes an error naming the supported checkpoints.
        monkeypatch.setattr(SigLip2Config, "from_name", staticmethod(lambda model_name: config))
        wider = SigLip2Config(
            vision_config=replace(config.vision_config, hidden_size=48),
            text_config=replace(config.text_config, hidden_size=48),
            projection_dim=config.projection_dim,
        )
        state_dict = SigLip2Model(wider).state_dict()
        monkeypatch.setattr(siglip2_builder, "_download_weights", lambda model_name, cache_dir: state_dict)

        with pytest.raises(RuntimeError, match=r"Supported names: google/siglip-base-patch16-224, ") as error:
            SigLip2Builder.from_pretrained_hf("someone/my-siglip")
        # The message reports the configuration actually read from the name, here the tiny stand-in.
        assert "hidden size 32, 2 layers, 4 heads, patch size 4, image size 32, vocabulary 100" in str(error.value)
        assert "size mismatch" in str(error.value.__cause__)

    @pytest.mark.parametrize(
        ("model_name", "downloaded"),
        [
            # a listed name in another case is downloaded, and so cached, under its listed spelling, not as a second
            # copy (the Hub redirects the case variant to the same repository)
            ("Google/SigLIP2-Base-Patch16-224", "google/siglip2-base-patch16-224"),
            ("GOOGLE/siglip-base-patch16-224", "google/siglip-base-patch16-224"),
            # an unlisted name keeps the caller's spelling: kornia does not know its repository's
            ("Someone/SigLIP2-Base-Patch16-224-ft", "Someone/SigLIP2-Base-Patch16-224-ft"),
        ],
    )
    def test_from_pretrained_hf_downloads_a_listed_name_under_its_listed_spelling(
        self, monkeypatch, config, model_name, downloaded
    ):
        monkeypatch.setattr(SigLip2Config, "from_name", staticmethod(lambda model_name: config))
        state_dict = SigLip2Model(config).state_dict()
        repositories = []

        def download(model_name, cache_dir):
            repositories.append(model_name)
            return state_dict

        monkeypatch.setattr(siglip2_builder, "_download_weights", download)
        SigLip2Builder.from_pretrained_hf(model_name)
        assert repositories == [downloaded]

    @pytest.mark.parametrize("model_name", ["google/siglip2-base-patch16-224", "Google/SigLIP2-Base-Patch16-224"])
    def test_from_pretrained_hf_passes_on_a_listed_checkpoints_load_error(self, monkeypatch, config, model_name):
        # A listed name's checkpoint that does not fit is not a naming question: load_state_dict's own error surfaces.
        monkeypatch.setattr(SigLip2Config, "from_name", staticmethod(lambda model_name: config))
        wider = SigLip2Config(
            vision_config=replace(config.vision_config, hidden_size=48),
            text_config=replace(config.text_config, hidden_size=48),
            projection_dim=config.projection_dim,
        )
        state_dict = SigLip2Model(wider).state_dict()
        monkeypatch.setattr(siglip2_builder, "_download_weights", lambda model_name, cache_dir: state_dict)

        with pytest.raises(RuntimeError, match="size mismatch") as error:
            SigLip2Builder.from_pretrained_hf(model_name)
        assert "outside that list" not in str(error.value)
        assert error.value.__cause__ is None


class TestSigLip2Model(BaseTester):
    """Test suite for SigLip2 model."""

    def test_smoke(self, device, dtype, config):
        """Test basic model instantiation."""
        model = SigLip2Model(config).to(device, dtype)
        assert model is not None

    def test_identity_projections(self, device, dtype, config):
        """Test the default identity projection path with a lightweight model."""
        config = replace(config, projection_dim=config.vision_config.hidden_size)
        model = SigLip2Model(config).to(device, dtype).eval()
        image_size = config.vision_config.image_size
        pixel_values = torch.randn(1, 3, image_size, image_size, device=device, dtype=dtype)
        input_ids = _create_input_ids(1, 10, config, device)

        with torch.no_grad():
            output = model(pixel_values=pixel_values, input_ids=input_ids)

        assert isinstance(model.vision_projection, torch.nn.Identity)
        assert isinstance(model.text_projection, torch.nn.Identity)
        assert output.image_embeds.shape == (1, config.projection_dim)
        assert output.text_embeds.shape == (1, config.projection_dim)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_cardinality(self, device, dtype, model, config, batch_size):
        """Test output shapes with different inputs and batch sizes."""
        image_size = config.vision_config.image_size
        pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
        seq_len = 10
        input_ids = _create_input_ids(batch_size, seq_len, config, device)

        with torch.no_grad():
            # Image only
            output: SigLip2Result = model(pixel_values=pixel_values)
            assert output.image_embeds is not None
            assert output.image_embeds.shape == (batch_size, config.projection_dim)
            assert output.text_embeds is None

            # Text only
            output: SigLip2Result = model(input_ids=input_ids)
            assert output.text_embeds is not None
            assert output.text_embeds.shape == (batch_size, config.projection_dim)
            assert output.image_embeds is None

            # Joint
            output: SigLip2Result = model(pixel_values=pixel_values, input_ids=input_ids)
            assert output.image_embeds.shape == (batch_size, config.projection_dim)
            assert output.text_embeds.shape == (batch_size, config.projection_dim)
            assert output.logits_per_image.shape == (batch_size, batch_size)
            assert output.logits_per_text.shape == (batch_size, batch_size)

    def test_exception(self, device, dtype, model, config):
        """Test exception handling."""
        # Test invalid pixel_values shape (wrong number of dimensions)
        image_size = config.vision_config.image_size
        invalid_pixel_values = torch.randn(3, image_size, image_size, device=device, dtype=dtype)
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            model.get_image_features(invalid_pixel_values)

        # Test invalid attention mask shape
        input_ids = _create_input_ids(2, 10, config, device)
        invalid_attention_mask = torch.ones(2, 5, device=device)  # Wrong sequence length
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            model.get_text_features(input_ids, attention_mask=invalid_attention_mask)

        # Test input_ids with wrong number of dimensions
        invalid_input_ids = torch.randint(0, 100, (10,), device=device)  # Missing batch dimension
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            model.get_text_features(invalid_input_ids)

    def test_get_image_features(self, device, dtype, model, config):
        """Test get_image_features method."""

        batch_size = 2
        image_size = config.vision_config.image_size
        pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)

        with torch.no_grad():
            features = model.get_image_features(pixel_values)

        assert features.shape == (batch_size, config.projection_dim)
        # Check normalization
        norms = features.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms))

    def test_get_text_features(self, device, dtype, model, config):
        """Test get_text_features method."""

        batch_size = 2
        seq_len = 10
        input_ids = _create_input_ids(batch_size, seq_len, config, device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with torch.no_grad():
            features = model.get_text_features(input_ids, attention_mask=attention_mask)

        assert features.shape == (batch_size, config.projection_dim)
        assert torch.isfinite(features).all()
        # Check normalization
        norms = features.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms))

    def test_attention_mask_handling(self, device, dtype, model, config):
        """Test attention mask handling in text encoder."""

        batch_size = 2
        seq_len = 10
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        input_ids[0, :5] = torch.arange(1, 6, device=device)
        input_ids[1, :8] = torch.arange(11, 19, device=device)

        # Create attention mask with different lengths
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        attention_mask[0, 5:] = 0  # First sequence has 5 tokens
        attention_mask[1, 8:] = 0  # Second sequence has 8 tokens

        with torch.no_grad():
            features = model.get_text_features(input_ids, attention_mask=attention_mask)

        assert features.shape == (batch_size, config.projection_dim)
        assert torch.isfinite(features).all()
        assert not torch.allclose(features[0], features[1])

    def test_return_loss(self, device, dtype, model, config):
        """Test forward pass with return_loss=True and verify logit_scale clamping."""
        batch_size = 2
        image_size = config.vision_config.image_size
        pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
        seq_len = 10
        input_ids = _create_input_ids(batch_size, seq_len, config, device)

        with torch.no_grad():
            output = model(pixel_values=pixel_values, input_ids=input_ids, return_loss=True)

        assert output.loss is not None
        assert output.loss.item() >= 0.0  # Loss should be non-negative

        # Test logit_scale clamping with extreme values
        with torch.no_grad():
            # Test max clamping
            model.logit_scale.data.fill_(100.0)
            output_max = model(pixel_values=pixel_values, input_ids=input_ids)
            assert torch.isfinite(output_max.logits_per_image).all(), "Max clamp: logits contain non-finite values"
            expected_max = torch.tensor(config.logit_scale_max, device=device, dtype=dtype).log().exp()
            self.assert_close(output_max.logit_scale, expected_max)

            # Test min clamping
            model.logit_scale.data.fill_(-10.0)
            output_min = model(pixel_values=pixel_values, input_ids=input_ids)
            assert output_min.logit_scale.item() >= 1.0, f"Min clamp failed: {output_min.logit_scale.item()} < 1.0"

    def test_return_loss_is_the_pairwise_sigmoid_loss(self, device, dtype, model, config, monkeypatch):
        batch_size = 2
        image_size = config.vision_config.image_size
        pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
        input_ids = _create_input_ids(batch_size, 10, config, device)

        # transformers' SiglipModel loss (modeling_siglip.py v4.51.0, SiglipModel.forward, after big_vision's
        # siglip.py): every pair enters, the diagonal labelled +1 and the rest -1,
        # -logsigmoid((2 I - 1) * logits_per_text).sum(-1).mean().
        with torch.no_grad():
            random_init = model(pixel_values=pixel_values, input_ids=input_ids, return_loss=True)

        # A literal: the unit vectors e1 and e2 as both embeddings, logit scale 10 and bias 5, give logits 15 on the
        # diagonal and 5 off it. Each row's loss is log1p(exp(-15)) + log1p(exp(5)) = 5.0067156543913915 (by hand, in
        # float64); a loss of the diagonal alone is log1p(exp(-15)) = 3.1e-07.
        unit_vectors = torch.eye(batch_size, config.projection_dim, device=device, dtype=dtype)
        monkeypatch.setattr(model, "get_image_features", lambda *args, **kwargs: unit_vectors)
        monkeypatch.setattr(model, "get_text_features", lambda *args, **kwargs: unit_vectors)
        with torch.no_grad():
            model.logit_scale.fill_(math.log(10.0))
            model.logit_bias.fill_(5.0)
            output = model(pixel_values=pixel_values, input_ids=input_ids, return_loss=True)
        self.assert_close(output.loss, torch.tensor(5.0067156543913915, device=device, dtype=dtype))

        # The same formula on the model's own logits at random init.
        labels = 2 * torch.eye(batch_size, device=device, dtype=dtype) - 1
        expected = -torch.nn.functional.logsigmoid(labels * random_init.logits_per_text).sum(-1).mean()
        self.assert_close(random_init.loss, expected)

    def test_return_loss_needs_as_many_images_as_texts(self, device, dtype, model, config):
        # The loss labels pair i with pair i; a 1 x 1 identity would broadcast +1 to every pair of unequal batches.
        image_size = config.vision_config.image_size
        pixel_values = torch.randn(3, 3, image_size, image_size, device=device, dtype=dtype)
        input_ids = _create_input_ids(1, 10, config, device)

        with torch.no_grad():
            for images, texts in ((pixel_values, input_ids), (pixel_values[:1], input_ids.expand(3, -1))):
                with pytest.raises(BaseError, match="as many images as texts"):
                    model(pixel_values=images, input_ids=texts, return_loss=True)
            # Without the loss, unequal batches give rectangular logits.
            output = model(pixel_values=pixel_values, input_ids=input_ids)

        assert output.logits_per_image.shape == (3, 1)
        assert output.logits_per_text.shape == (1, 3)
        assert output.loss is None

    def test_pixel_attention_mask_reaches_the_image_embeddings_and_the_loss(self, device, dtype, model, config):
        # transformers' Siglip2Model.forward takes a pixel_attention_mask for the vision tower (modeling_siglip2.py
        # v4.51.0). Here it reaches get_image_features, so the pixels of masked patches change neither the image
        # embeddings nor the loss.
        pixel_values = torch.randn(2, 3, 32, 32, device=device, dtype=dtype)
        changed = pixel_values.clone()
        changed[..., 16:, :] = torch.randn(2, 3, 16, 32, device=device, dtype=dtype)
        mask = _patch_rows_4_to_7_masked(device).expand(2, -1)
        input_ids = _create_input_ids(2, 10, config, device)

        with torch.no_grad():
            output = model(pixel_values, input_ids, return_loss=True, pixel_attention_mask=mask)
            output_changed = model(changed, input_ids, return_loss=True, pixel_attention_mask=mask)
            features = model.get_image_features(pixel_values, attention_mask=mask)

        self.assert_close(output.image_embeds, features)
        self.assert_close(output_changed.image_embeds, output.image_embeds)
        self.assert_close(output_changed.loss, output.loss)

    def test_gradcheck(self, device, dtype, config):
        """Test gradient computation correctness."""
        # Convert model to float64 for gradcheck
        model = SigLip2Model(config).to(device, torch.float64).train()
        batch_size = 1
        image_size = config.vision_config.image_size
        pixel_values = torch.randn(
            batch_size, 3, image_size, image_size, device=device, dtype=torch.float64, requires_grad=True
        )
        seq_len = 5
        input_ids = _create_input_ids(batch_size, seq_len, config, device).to(torch.int64)

        # Only check gradients for pixel_values (input_ids are indices, not differentiable)
        def func(pixel_vals):
            # Use input_ids as closure variable, not as gradcheck input
            return model.get_image_features(pixel_vals) + model.get_text_features(input_ids)

        self.gradcheck(func, pixel_values, raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer, model, config):
        """Test torch.compile compatibility."""
        batch_size = 1
        image_size = config.vision_config.image_size
        pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
        seq_len = 10
        input_ids = _create_input_ids(batch_size, seq_len, config, device)

        model_optimized = torch_optimizer(model)

        with torch.no_grad():
            expected = model(pixel_values=pixel_values, input_ids=input_ids)
            actual = model_optimized(pixel_values=pixel_values, input_ids=input_ids)

        self.assert_close(actual.image_embeds, expected.image_embeds)
        self.assert_close(actual.text_embeds, expected.text_embeds)

    def test_dynamo_return_loss_check_is_dynamic(self, device, dtype, torch_optimizer, model, config):
        # The equal-batches check of return_loss formats the batch sizes only when it fails: an f-string evaluated on
        # every call made Dynamo specialize the batch size, so every new batch compiled a graph of its own.
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()
        compiled = torch_optimizer(model, backend=counter, fullgraph=True, dynamic=True)
        # no batch equals another axis (3 channels, 4 heads, 8 head dims, 10 tokens), so duck sizing cannot tie it
        for batch_size in (5, 6, 7):
            pixel_values = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
            input_ids = _create_input_ids(batch_size, 10, config, device)
            with torch.no_grad():
                actual = compiled(pixel_values=pixel_values, input_ids=input_ids, return_loss=True)
                expected = model(pixel_values=pixel_values, input_ids=input_ids, return_loss=True)
            self.assert_close(actual.loss, expected.loss, low_tolerance=True)
        assert counter.frame_count == 1

    def test_dynamo_fullgraph_masked_vision_model(self, device, dtype, torch_optimizer):
        """The vision attention mask reaches the encoder and the pooling head inside one compiled graph."""
        model = SigLip2VisionModel(_tiny_vision_config()).to(device, dtype).eval()
        pixel_values = torch.randn(1, 3, 32, 32, device=device, dtype=dtype)
        mask = _patch_rows_4_to_7_masked(device)

        model_optimized = torch_optimizer(model, fullgraph=True)

        with torch.no_grad():
            expected = model(pixel_values, attention_mask=mask)
            actual = model_optimized(pixel_values, attention_mask=mask)

        # Inductor keeps fused half-precision intermediates in float32, so a few outputs differ from eager by one
        # unit in the last place. The last hidden state of this masked model fails the default half tolerance the same
        # way on the base revision a60a6c932, whose encoder code this change does not touch.
        low_tolerance = dtype in (torch.float16, torch.bfloat16)
        self.assert_close(actual[0], expected[0], low_tolerance=low_tolerance)
        self.assert_close(actual[1], expected[1], low_tolerance=low_tolerance)


class TestSigLip2Components(BaseTester):
    """Test suite for SigLip2 individual components."""

    def test_vision_embeddings(self, device, dtype):
        """Test SigLip2VisionEmbeddings."""
        config = SigLip2VisionConfig(image_size=32, patch_size=4, hidden_size=32)
        embeddings = SigLip2VisionEmbeddings(config).to(device, dtype)

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device, dtype=dtype)

        with torch.no_grad():
            output = embeddings(pixel_values)

        num_patches = (config.image_size // config.patch_size) ** 2
        assert output.shape == (batch_size, num_patches, config.hidden_size)

    @pytest.mark.parametrize(
        "size",
        [(16, 64), (8, 128), (33, 39), (32, 39), (39, 32), (40, 40)],
        ids=["2x8-grid", "1x16-grid", "33x39", "32x39", "39x32", "40x40"],
    )
    def test_vision_embeddings_reject_other_sizes(self, device, dtype, size):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        config = SigLip2VisionConfig(
            image_size=32,
            patch_size=8,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
        )
        embeddings = SigLip2VisionEmbeddings(config).to(device, dtype)
        model = SigLip2VisionModel(config).to(device, dtype)

        with torch.no_grad():
            assert embeddings(torch.rand(1, 3, 32, 32, device=device, dtype=dtype)).shape == (1, 16, 32)
            # The position table is the (32 // 8) ** 2 = 16 cells of a 4 x 4 grid. The patch grids 2 x 8 of 16 x 64,
            # 1 x 16 of 8 x 128 and 4 x 4 of 33 x 39, 32 x 39 and 39 x 32 (remainders dropped) also have 16 cells;
            # 40 x 40 gives 5 x 5. 32 x 39 and 39 x 32 have one side right, so the check must test both.
            pixel_values = torch.rand(1, 3, *size, device=device, dtype=dtype)
            with pytest.raises(ValueError, match="32"):
                embeddings(pixel_values)
            # SigLip2VisionModel, and through it SigLip2Model and PaliGemma, embed the image first.
            with pytest.raises(ValueError, match="32"):
                model(pixel_values)

    def test_dynamo_fullgraph_vision_embeddings(self, device, dtype, torch_optimizer):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        # The size check reads static shapes: it must compile without a graph break.
        embeddings = SigLip2VisionEmbeddings(_tiny_vision_config()).to(device, dtype).eval()
        pixel_values = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)

        embeddings_optimized = torch_optimizer(embeddings, fullgraph=True)

        with torch.no_grad():
            self.assert_close(embeddings_optimized(pixel_values), embeddings(pixel_values))

    def test_vision_encoder(self, device, dtype):
        """Test SigLip2VisionEncoder."""
        config = SigLip2VisionConfig(
            image_size=32,
            patch_size=4,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
        )
        encoder = SigLip2VisionEncoder(config).to(device, dtype)
        embeddings = SigLip2VisionEmbeddings(config).to(device, dtype)

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device, dtype=dtype)

        with torch.no_grad():
            # Encoder expects embeddings, not raw pixel values
            hidden_states = embeddings(pixel_values)
            output = encoder(hidden_states)

        num_patches = (config.image_size // config.patch_size) ** 2
        assert output[0].shape == (batch_size, num_patches, config.hidden_size)

    def test_vision_model(self, device, dtype):
        """Test SigLip2VisionModel."""
        config = SigLip2VisionConfig(
            image_size=32,
            patch_size=4,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
        )
        model = SigLip2VisionModel(config).to(device, dtype)

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device, dtype=dtype)

        with torch.no_grad():
            pooled_output, last_hidden_state = model(pixel_values)

        assert pooled_output.shape == (batch_size, config.hidden_size)
        num_patches = (config.image_size // config.patch_size) ** 2
        assert last_hidden_state.shape == (batch_size, num_patches, config.hidden_size)

    def test_text_embeddings(self, device, dtype):
        """Test SigLip2TextEmbeddings."""
        config = SigLip2TextConfig(vocab_size=100, hidden_size=32, max_position_embeddings=32)
        embeddings = SigLip2TextEmbeddings(config).to(device, dtype)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = embeddings(input_ids)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_text_encoder(self, device, dtype):
        """Test SigLip2TextEncoder."""
        config = SigLip2TextConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
        )
        encoder = SigLip2TextEncoder(config).to(device, dtype)
        embeddings = SigLip2TextEmbeddings(config).to(device, dtype)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with torch.no_grad():
            # Encoder expects hidden_states (embeddings), not input_ids
            hidden_states = embeddings(input_ids)
            output = encoder(hidden_states, attention_mask=attention_mask)

        assert output[0].shape == (batch_size, seq_len, config.hidden_size)

    def test_text_model(self, device, dtype):
        """Test SigLip2TextModel."""
        config = SigLip2TextConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
        )
        model = SigLip2TextModel(config).to(device, dtype)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with torch.no_grad():
            pooled_output, last_hidden_state = model(input_ids=input_ids, attention_mask=attention_mask)

        assert pooled_output.shape == (batch_size, config.hidden_size)
        assert last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

    def test_attention(self, device, dtype):
        """Test SigLip2Attention."""
        hidden_size = 32
        num_heads = 4
        attention = SigLip2Attention(hidden_size=hidden_size, num_heads=num_heads).to(device, dtype)

        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        with torch.no_grad():
            output = attention(hidden_states, attention_mask=attention_mask)
            output_without_mask = attention(hidden_states)

        # Attention returns a single tensor, not a tuple
        assert output.shape == (batch_size, seq_len, hidden_size)
        # An all-ones mask must be a no-op. Passing any ``attn_mask`` disqualifies the CUDA flash
        # backend, so the two calls can run different SDPA kernels and agree only up to kernel-level
        # rounding, which is the same order as the default half-precision tolerance.
        tol = {"rtol": 5e-2, "atol": 5e-2} if dtype in (torch.float16, torch.bfloat16) else {}
        self.assert_close(output, output_without_mask, **tol)

    @pytest.mark.parametrize("mask_shape", [(2, 1), (2, 10, 1), (2, 1, 10, 1), (2, 1, 1, 1), (2, 3, 10)])
    def test_attention_mask_covers_every_token(self, device, dtype, mask_shape):
        # A key dimension of 1 broadcast one value over all tokens: on the CPU and in CUDA float64 such a mask was the
        # same as no mask, and in CUDA float32 and half it raised SDPA's "last dimension must be contiguous". A query
        # dimension that is neither 1 nor the sequence length raised SDPA's broadcasting error.
        attention = SigLip2Attention(hidden_size=32, num_heads=4).to(device, dtype).eval()
        hidden_states = torch.randn(2, 10, 32, device=device, dtype=dtype)
        mask = torch.ones(mask_shape, dtype=torch.bool, device=device)

        with torch.no_grad(), pytest.raises(ShapeError, match=r"10 tokens"):
            attention(hidden_states, attention_mask=mask)

    def test_attention_three_dim_mask_broadcasts_over_the_queries(self, device, dtype):
        # A (B, 1, N) mask is a per-sample key mask shared by every query, the 3-D form of a (B, N) padding mask.
        attention = SigLip2Attention(hidden_size=32, num_heads=4).to(device, dtype).eval()
        hidden_states = torch.randn(2, 10, 32, device=device, dtype=dtype)
        mask = torch.ones(2, 10, dtype=torch.bool, device=device)
        mask[1, 5:] = False

        with torch.no_grad():
            self.assert_close(
                attention(hidden_states, attention_mask=mask[:, None]), attention(hidden_states, attention_mask=mask)
            )

    @pytest.mark.parametrize(
        ("batch_size", "input_size"),
        [
            (1, (32, 32)),
            (1, (64, 64)),
            (2, (48, 64)),
            (4, (16, 16)),
        ],
    )
    def test_image_preprocessor(self, device, dtype, batch_size, input_size):
        """Test SigLip2ImagePreprocessor with different configurations."""
        image_size = (32, 32)
        preprocessor = SigLip2ImagePreprocessor(image_size=image_size).to(device, dtype)

        # Test with batch of images (4D tensor)
        images = torch.randint(0, 255, (batch_size, 3, *input_size), device=device, dtype=dtype)
        with torch.no_grad():
            output = preprocessor(images)
        assert output.shape == (batch_size, 3, image_size[0], image_size[1])

    def test_image_preprocessor_single_image(self, device, dtype):
        """Test SigLip2ImagePreprocessor with single image (3D tensor)."""
        image_size = (32, 32)
        preprocessor = SigLip2ImagePreprocessor(image_size=image_size).to(device, dtype)

        # Test with single image (3D tensor) - preprocessor adds batch dimension
        image = torch.randint(0, 255, (3, 64, 64), device=device, dtype=dtype)
        with torch.no_grad():
            output = preprocessor(image)
        assert output.shape == (1, 3, image_size[0], image_size[1])

    def test_pooling_head_ignores_masked_patches(self, device, dtype):
        # transformers' Siglip2VisionTransformer passes the patch mask to its pooling head as well
        # (Siglip2MultiheadAttentionPoolingHead.forward(hidden_state, attention_mask), modeling_siglip2.py v4.51.0), so
        # the pixels of masked patches cannot reach the pooled embedding.
        model = SigLip2VisionModel(_tiny_vision_config()).to(device, dtype).eval()
        pixel_values = torch.randn(1, 3, 32, 32, device=device, dtype=dtype)
        changed = pixel_values.clone()
        changed[..., 16:, :] = torch.randn(1, 3, 16, 32, device=device, dtype=dtype)
        mask = _patch_rows_4_to_7_masked(device)

        with torch.no_grad():
            pooled, _ = model(pixel_values, attention_mask=mask)
            pooled_changed, _ = model(changed, attention_mask=mask)
            # The same mask broadcast over the queries, as (B, N, N) and as (B, 1, N, N).
            pooled_3d, _ = model(pixel_values, attention_mask=mask[:, None, :].expand(1, 64, 64))
            pooled_4d, _ = model(pixel_values, attention_mask=mask[:, None, None, :].expand(1, 1, 64, 64))
            # The usual 3-D padding mask, the outer product, also masks the padded queries: every key of a padded
            # patch's row is masked, so a key counts as padding only when no query may attend to it.
            pooled_outer, _ = model(pixel_values, attention_mask=mask[:, :, None] & mask[:, None, :])

        self.assert_close(pooled_changed, pooled)
        self.assert_close(pooled_3d, pooled)
        self.assert_close(pooled_4d, pooled)
        self.assert_close(pooled_outer, pooled)

    def test_pooling_head_broadcasts_a_single_mask_over_the_batch(self, device, dtype):
        config = _tiny_vision_config()
        head = SigLip2MultiheadAttentionPoolingHead(config).to(device, dtype).eval()
        hidden_state = torch.randn(3, 64, config.hidden_size, device=device, dtype=dtype)
        mask = _patch_rows_4_to_7_masked(device)  # (1, 64)

        with torch.no_grad():
            pooled = head(hidden_state, attention_mask=mask)
            per_sample = torch.cat([head(hidden_state[b : b + 1], attention_mask=mask) for b in range(3)])

        self.assert_close(pooled, per_sample)

    def test_pooling_head_all_masked_sample_pools_uniformly(self, device, dtype):
        # transformers' pooling head masks with _prepare_4d_attention_mask, which adds torch.finfo(dtype).min to the
        # scores of masked keys (modeling_attn_mask_utils.py v4.51.0, AttentionMaskConverter._expand_mask): a sample
        # with every key masked gets equal scores, so uniform weights, where -inf would give nan. In float16 that holds
        # only while the scores stay within about +-16 of 0 (finfo.min + score rounds to a multiple of 32); this tiny
        # random head's scores are far smaller.
        config = _tiny_vision_config()
        head = SigLip2MultiheadAttentionPoolingHead(config).to(device, dtype).eval()
        hidden_state = torch.randn(2, 64, config.hidden_size, device=device, dtype=dtype)
        mask = torch.ones(2, 64, dtype=torch.bool, device=device)
        mask[0, 32:] = False  # sample 0 partly masked
        mask[1] = False  # sample 1 entirely masked

        with torch.no_grad():
            pooled = head(hidden_state, attention_mask=mask)
            pooled_alone = head(hidden_state[:1], attention_mask=mask[:1])
            # Uniform weights: every head averages the value projections of all 64 tokens.
            hidden_size = config.hidden_size
            attention = head.attention
            values = torch.nn.functional.linear(
                hidden_state[1:], attention.in_proj_weight[2 * hidden_size :], attention.in_proj_bias[2 * hidden_size :]
            )
            attended = attention.out_proj(values.mean(1, keepdim=True))
            uniform = (attended + head.mlp(head.layernorm(attended)))[:, 0]

        assert torch.isfinite(pooled).all()
        self.assert_close(pooled[1:], uniform)
        self.assert_close(pooled[:1], pooled_alone)

    @pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
    def test_pooling_head_all_masked_sample_pools_uniformly_under_autocast(self, device, dtype, autocast_dtype):
        # Under autocast the hidden state stays float32 while nn.MultiheadAttention computes the scores in the
        # autocast dtype, where float32's finfo.min rounds to -inf: the fill has to be finite in the autocast dtype,
        # or a sample with every key masked pools to nan.
        if dtype != torch.float32:
            pytest.skip("autocast casts float32 inputs only")
        config = _tiny_vision_config()
        head = SigLip2MultiheadAttentionPoolingHead(config).to(device, dtype).eval()
        hidden_state = torch.randn(2, 64, config.hidden_size, device=device, dtype=dtype)
        mask = torch.ones(2, 64, dtype=torch.bool, device=device)
        mask[0, 32:] = False  # sample 0 partly masked
        mask[1] = False  # sample 1 entirely masked

        with torch.no_grad():
            expected = head(hidden_state, attention_mask=mask)
            with torch.autocast(device.type, dtype=autocast_dtype):
                pooled = head(hidden_state, attention_mask=mask)

        assert torch.isfinite(pooled).all()
        # the autocast run rounds the projections to the autocast dtype
        self.assert_close(pooled.to(dtype), expected, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize(
        ("param_dtype", "autocast_dtype"), [(torch.bfloat16, torch.float16), (torch.float64, torch.float16)]
    )
    def test_pooling_head_all_masked_sample_pools_uniformly_in_the_score_dtype(
        self, device, dtype, param_dtype, autocast_dtype
    ):
        # The scores are computed in the autocast dtype for a float16, bfloat16 or float32 input and in float64 for a
        # float64 one, which autocast leaves alone. A bfloat16 mask filled with float16's minimum held -65536, which
        # overflows to -inf in float16 (nan); a float64 mask filled with it kept the score differences (not uniform).
        if dtype != torch.float32:
            pytest.skip("one run per device: the dtypes are the parameters")
        if param_dtype == torch.float64 and device.type == "mps":
            pytest.skip("MPS does not support float64")
        if param_dtype == torch.bfloat16 and device.type == "cpu":
            pytest.skip("CPU float16 autocast rejects the bfloat16 parameters of layer_norm")
        config = _tiny_vision_config()
        head = SigLip2MultiheadAttentionPoolingHead(config).to(device, param_dtype).eval()
        hidden_state = torch.randn(1, 64, config.hidden_size, device=device, dtype=param_dtype)
        mask = torch.zeros(1, 64, dtype=torch.bool, device=device)
        captured = {}
        forward = head.attention.forward
        head.attention.forward = lambda *args, **kwargs: captured.setdefault("out", forward(*args, **kwargs))

        with torch.no_grad(), torch.autocast(device.type, dtype=autocast_dtype):
            pooled = head(hidden_state, attention_mask=mask)

        assert torch.isfinite(pooled).all()
        weights = captured["out"][1].float()
        self.assert_close(weights, torch.full_like(weights, 1 / 64), rtol=0.0, atol=0.0)

    def test_dynamo_fullgraph_pooling_head_under_autocast(self, device, dtype, torch_optimizer):
        # The autocast fill is chosen inside the compiled graph too. torch.amp.is_autocast_available cannot be traced
        # before torch 2.6 (a graph break, so fullgraph raised on torch 2.5.1), so the head asks it in eager mode only.
        if dtype != torch.float32:
            pytest.skip("autocast casts float32 inputs only")
        autocast_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        config = _tiny_vision_config()
        head = SigLip2MultiheadAttentionPoolingHead(config).to(device, dtype).eval()
        hidden_state = torch.randn(2, 64, config.hidden_size, device=device, dtype=dtype)
        mask = torch.ones(2, 64, dtype=torch.bool, device=device)
        mask[1] = False  # sample 1 entirely masked: nan with the float32 fill under autocast

        compiled = torch_optimizer(head, fullgraph=True)
        with torch.no_grad(), torch.autocast(device.type, dtype=autocast_dtype):
            pooled = compiled(hidden_state, attention_mask=mask)
            expected = head(hidden_state, attention_mask=mask)

        assert torch.isfinite(pooled).all()
        self.assert_close(pooled, expected, low_tolerance=True)

    def test_pooling_head_masked_on_the_meta_device(self):
        # torch.is_autocast_enabled raises for a device type without autocast, such as meta, which shape inference and
        # deferred initialization use: the head asks torch.amp.is_autocast_available first in eager mode.
        config = _tiny_vision_config()
        head = SigLip2MultiheadAttentionPoolingHead(config).to("meta").eval()
        hidden_state = torch.empty(2, 64, config.hidden_size, device="meta")
        mask = torch.ones(2, 64, dtype=torch.bool, device="meta")

        assert head(hidden_state, attention_mask=mask).shape == (2, config.hidden_size)

    def test_activation_follows_hidden_act(self):
        # transformers' SiglipTextConfig and SiglipVisionConfig default hidden_act is "gelu_pytorch_tanh"
        # (configuration_siglip.py v4.51.0), and no released config.json overrides it.
        assert SigLip2VisionConfig().hidden_act == "gelu_pytorch_tanh"
        assert SigLip2TextConfig().hidden_act == "gelu_pytorch_tanh"
        vision_config = _tiny_vision_config()
        text_config = SigLip2TextConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
        )

        def approximations(hidden_act):
            vision = SigLip2VisionModel(replace(vision_config, hidden_act=hidden_act))
            text = SigLip2TextModel(replace(text_config, hidden_act=hidden_act))
            # Every encoder MLP and the pooling head's MLP of the vision tower, then the text encoder MLPs.
            mlps = [layer.mlp for layer in vision.encoder.layers] + [vision.head.mlp]
            mlps += [layer.mlp for layer in text.encoder.layers]
            return [mlp.activation.approximate for mlp in mlps]

        assert approximations(vision_config.hidden_act) == ["tanh"] * 5
        assert approximations("gelu_pytorch_tanh") == ["tanh"] * 5
        assert approximations("gelu_new") == ["tanh"] * 5
        assert approximations("gelu") == ["none"] * 5
        with pytest.raises(ValueError, match="relu6"):
            SigLip2VisionModel(replace(vision_config, hidden_act="relu6"))
        with pytest.raises(ValueError, match="relu6"):
            SigLip2TextModel(replace(text_config, hidden_act="relu6"))

    def test_preprocessor_interpolation(self, device, dtype):
        # The SigLIP 2 checkpoints' preprocessor_config.json says "resample": 2 (bilinear), the SigLIP 1
        # google/siglip-base-patch16-224 says 3 (bicubic); both rescale by 1/255 and normalise with mean = std = 0.5.
        if not supports_reflect_padding(device, dtype):
            pytest.skip("the antialiased resize blurs with reflection_pad2d, unavailable for this device/dtype")
        images = torch.randint(0, 255, (2, 3, 300, 400), device=device).to(dtype)

        def expected(interpolation):
            resized = kornia.geometry.resize(
                images * (1.0 / 255.0), (224, 224), interpolation=interpolation, align_corners=False, antialias=True
            )
            return (resized - 0.5) / 0.5

        with torch.no_grad():
            self.assert_close(SigLip2ImagePreprocessor()(images), expected("bilinear"))
            self.assert_close(SigLip2ImagePreprocessor.from_config(224)(images), expected("bilinear"))
            try:
                torch.nn.functional.interpolate(images[:1, :1, :2, :2], size=(3, 3), mode="bicubic")
            except NotImplementedError:
                # torch 2.5.1 has no MPS upsample_bicubic2d; the bilinear default is checked above
                pytest.skip("bicubic interpolation is unavailable for this device/dtype")
            self.assert_close(SigLip2ImagePreprocessor(interpolation="bicubic")(images), expected("bicubic"))
            bicubic = SigLip2ImagePreprocessor.from_config(224, interpolation="bicubic")
            self.assert_close(bicubic(images), expected("bicubic"))

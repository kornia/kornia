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

"""Configuration classes for SigLip2 model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from torch import nn


def _activation(hidden_act: str) -> nn.Module:
    """Return the MLP activation that a ``hidden_act`` config value names.

    The names are transformers' ``ACT2FN`` keys: ``"gelu_pytorch_tanh"`` and ``"gelu_new"`` are the tanh approximation
    of GELU, which the released SigLIP and SigLIP 2 checkpoints use, and ``"gelu"`` is the exact GELU.

    Raises:
        ValueError: for any other value.
    """
    if hidden_act in ("gelu_pytorch_tanh", "gelu_new"):
        return nn.GELU(approximate="tanh")
    if hidden_act == "gelu":
        return nn.GELU()
    raise ValueError(
        f"Unsupported hidden_act {hidden_act!r}: expected 'gelu_pytorch_tanh' or 'gelu_new' (tanh GELU) or 'gelu'."
    )


@dataclass
class SigLip2VisionConfig:
    """Configuration for SigLip2 vision encoder.

    Args:
        image_size: Size of input images.
        patch_size: Size of image patches.
        num_channels: Number of input channels.
        hidden_size: Hidden dimension size.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Intermediate size in feed-forward network.
        hidden_act: Activation of the MLPs: ``"gelu_pytorch_tanh"`` (default, the checkpoints' value) or
            ``"gelu_new"`` for tanh GELU, ``"gelu"`` for exact GELU. Other values raise ``ValueError`` when the model
            is built.
        layer_norm_eps: Epsilon for layer normalization.
        dropout_p: Dropout probability.
        attention_dropout_p: Attention dropout probability.
    """

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    dropout_p: float = 0.0
    attention_dropout_p: float = 0.0


@dataclass
class SigLip2TextConfig:
    """Configuration for SigLip2 text encoder.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Hidden dimension size.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Intermediate size in feed-forward network.
        max_position_embeddings: Maximum sequence length.
        hidden_act: Activation of the MLPs: ``"gelu_pytorch_tanh"`` (default, the checkpoints' value) or
            ``"gelu_new"`` for tanh GELU, ``"gelu"`` for exact GELU. Other values raise ``ValueError`` when the model
            is built.
        layer_norm_eps: Epsilon for layer normalization.
        dropout_p: Dropout probability.
        attention_dropout_p: Attention dropout probability.
    """

    vocab_size: int = 256000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    dropout_p: float = 0.0
    attention_dropout_p: float = 0.0


# The checkpoints SigLip2Config.from_name supports: name -> (vocab_size, hidden_size, num_hidden_layers,
# num_attention_heads, image_size), from each checkpoint's config.json on the Hugging Face Hub and transformers'
# SiglipTextConfig / SiglipVisionConfig defaults for the fields it omits (vocabulary 32000, hidden size 768, 12 layers,
# 12 heads, image size 224, patch size 16, 64 text positions).
_SUPPORTED_MODELS: dict[str, tuple[int, int, int, int, int]] = {
    "google/siglip-base-patch16-224": (32000, 768, 12, 12, 224),
    "google/siglip2-base-patch16-224": (256000, 768, 12, 12, 224),
    "google/siglip2-base-patch16-256": (256000, 768, 12, 12, 256),
    "google/siglip2-base-patch16-384": (256000, 768, 12, 12, 384),
    "google/siglip2-base-patch16-512": (256000, 768, 12, 12, 512),
    "google/siglip2-large-patch16-256": (256000, 1024, 24, 16, 256),
    "google/siglip2-large-patch16-384": (256000, 1024, 24, 16, 384),
    "google/siglip2-large-patch16-512": (256000, 1024, 24, 16, 512),
}


def _listed_name(model_name: str) -> Optional[str]:
    """The ``_SUPPORTED_MODELS`` key that ``model_name`` names in any case, else ``None``."""
    key = model_name.lower()
    return key if key in _SUPPORTED_MODELS else None


def _repository_words(model_name: str) -> list[str]:
    """The lowercase words of the repository part of ``model_name``, after its last ``/``, in order.

    A word is a run of letters and digits, so any other character separates words: ``"SigLIP2_Large.ft"`` is
    ``["siglip2", "large", "ft"]``. The architecture of a name outside ``_SUPPORTED_MODELS`` is read from these words
    alone, each compared whole: ``"LargeCorp/siglip2-base"`` is not large, ``"x/siglip2-base-enlarged"`` is not large
    either, and ``"x/dispatch2-siglip2-base-patch16"`` does not say patch size 2.
    """
    return re.findall(r"[0-9a-z]+", model_name.lower().rsplit("/", 1)[-1])


def _says_siglip1(words: list[str]) -> bool:
    """Whether repository ``words`` name a SigLIP 1 checkpoint: ``siglip`` (not ``siglip2``) then base or large."""
    return any(word == "siglip" and size in ("base", "large") for word, size in zip(words, words[1:]))


def _says_siglip1_base_224(words: list[str]) -> bool:
    """Whether repository ``words`` contain ``siglip base patch16 224``: a fine-tune or mirror of that checkpoint."""
    return any(words[i : i + 4] == ["siglip", "base", "patch16", "224"] for i in range(len(words) - 3))


def _read_name_parts(model_name: str) -> tuple[int, int, int, int, int]:
    """Read (vocab_size, hidden_size, num_hidden_layers, num_attention_heads, image_size) from a name's parts.

    Used by ``SigLip2Config.from_name`` for a name outside ``_SUPPORTED_MODELS`` that ``_unreadable_name`` accepts, in
    any case. The architecture comes from the words of the repository part (see ``_repository_words``): large if one
    of them is ``large``, base otherwise, and the SigLIP 1 vocabulary only for a fine-tune or mirror of
    ``google/siglip-base-patch16-224``. The image size is the first of 224, 256, 384 and 512 that is a whole number of
    the repository part, else of the owner part before the last ``/``, else 224.
    """
    # Whole numbers only ("x/siglip2-ckpt2560" is 224), and the repository's before the owner's:
    # "org-256/siglip2-base-patch16-384" is 384, "siglip384/finetune" is 384.
    image_size = 224
    for part in reversed(model_name.rsplit("/", 1)):
        numbers = set(re.split(r"[^0-9]+", part))
        size = next((size for size in (224, 256, 384, 512) if str(size) in numbers), None)
        if size is not None:
            image_size = size
            break
    words = _repository_words(model_name)
    vocab_size = 32000 if _says_siglip1_base_224(words) else 256000
    if "large" in words:
        return vocab_size, 1024, 24, 16, image_size
    return vocab_size, 768, 12, 12, image_size


def _unreadable_name(model_name: str) -> Optional[str]:
    """Why ``_read_name_parts`` cannot give the architecture of a name outside ``_SUPPORTED_MODELS``, else ``None``.

    The parts reading builds a base or large SigLIP 2 model at patch size 16. A ``google/`` name outside the table is
    a released checkpoint whose configuration was not checked (SigLIP 1 has the 32000-token vocabulary, so400m a
    hidden size of 1152), as is a SigLIP 1 repository (the words ``siglip base`` or ``siglip large``) other than a
    fine-tune or mirror of ``google/siglip-base-patch16-224``. A repository part with the word so400m, giant or naflex,
    or a word ``patch<n>`` with n other than 16, names an architecture the reading cannot build. Only the words of the
    repository part count (see ``_repository_words``).
    """
    if model_name.lower().startswith("google/"):
        return "it is a google/ checkpoint outside the supported list"
    words = _repository_words(model_name)
    for marker in ("so400m", "giant", "naflex"):
        if marker in words:
            return f"the name says {marker!r}, an architecture other than base or large at patch size 16"
    for word in words:
        patch = re.fullmatch(r"patch(\d+)", word)
        if patch is not None and patch.group(1) != "16":
            return f"the name says patch size {patch.group(1)}, not 16"
    if _says_siglip1(words) and not _says_siglip1_base_224(words):
        return "it names a SigLIP 1 checkpoint other than google/siglip-base-patch16-224"
    return None


@dataclass
class SigLip2Config:
    """Configuration for SigLip2 model.

    Args:
        vision_config: Vision encoder configuration.
        text_config: Text encoder configuration.
        projection_dim: Dimension of projection layers.
        logit_scale_init_value: Initial value for logit scale (temperature).
        logit_scale_max: Maximum value for logit_scale to prevent overflow. Default: 100.0
    """

    vision_config: Optional[SigLip2VisionConfig] = None
    text_config: Optional[SigLip2TextConfig] = None
    projection_dim: int = 768
    logit_scale_init_value: float = 2.6592
    logit_scale_max: float = 100.0

    def __post_init__(self) -> None:
        """Initialize default configs if not provided."""
        if self.vision_config is None:
            self.vision_config = SigLip2VisionConfig()
        if self.text_config is None:
            self.text_config = SigLip2TextConfig()

    @staticmethod
    def from_name(model_name: str) -> SigLip2Config:
        """Create the configuration of a checkpoint from its name.

        Supports the following models:
        - google/siglip-base-patch16-224 (V1)
        - google/siglip2-base-patch16-224
        - google/siglip2-base-patch16-256
        - google/siglip2-base-patch16-384
        - google/siglip2-base-patch16-512
        - google/siglip2-large-patch16-256
        - google/siglip2-large-patch16-384
        - google/siglip2-large-patch16-512

        Each of these names gives its checkpoint's ``config.json`` values: patch size 16, 64 text positions, the
        vocabulary (32000 for the SigLIP 1 model, 256000 for SigLIP 2), and the base (hidden size 768, 12 layers, 12
        heads) or large (1024, 24, 16) architecture.

        Names are read in any case. Any other name, such as a fine-tune or a mirror of one of these checkpoints, is
        read from its parts. Its architecture comes from the words of the repository part after the last ``/``, a word
        being a run of letters and digits compared whole: the large architecture if one of them is ``large`` and the
        base one otherwise, patch size 16, and the SigLIP 2 vocabulary (SigLIP 1's for a fine-tune or mirror of
        ``google/siglip-base-patch16-224``, whose words contain ``siglip base patch16 224``). The owner part does not
        count, so ``LargeCorp/siglip2-base-patch16-224`` is base, and neither does part of a word, so
        ``someone/siglip2-base-patch16-224-enlarged`` is base too. The image size is the first of 224, 256, 384 and 512
        that is a whole number in the repository part, else in the owner part, else 224, and the text has 512
        positions. That configuration fits only a checkpoint with that architecture:
        ``SigLip2Builder.from_pretrained_hf`` adapts the text positions to the checkpoint and raises ``RuntimeError``
        when its other shapes differ.

        A name whose parts say an architecture this reading cannot build raises: a ``google/`` name outside the list
        (another released checkpoint, such as ``google/siglip2-so400m-patch14-384`` or the SigLIP 1
        ``google/siglip-large-patch16-256``), a SigLIP 1 repository (the words ``siglip base`` or ``siglip large``)
        other than a fine-tune or mirror of ``google/siglip-base-patch16-224`` (such as
        ``someone/siglip-large-patch16-256``), or a repository part with the word ``so400m``, ``giant`` or ``naflex``,
        or ``patch<n>`` for a patch size other than 16.

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            SigLip2Config instance configured for the specified model.

        Raises:
            ValueError: for a name outside the list whose architecture its parts cannot give.
        """
        text_kwargs: dict[str, int] = {}
        listed = _listed_name(model_name)
        if listed is not None:
            vocab_size, hidden_size, num_layers, num_heads, image_size = _SUPPORTED_MODELS[listed]
            text_kwargs["max_position_embeddings"] = 64
        else:
            reason = _unreadable_name(model_name)
            if reason is not None:
                raise ValueError(
                    f"SigLip2Config.from_name cannot build {model_name!r}: {reason}. Supported names: "
                    f"{', '.join(_SUPPORTED_MODELS)}. Another name is read as a fine-tune or mirror of one of these, "
                    "with the base or large architecture at patch size 16."
                )
            vocab_size, hidden_size, num_layers, num_heads, image_size = _read_name_parts(model_name)
        # every supported checkpoint has an MLP of 4 x hidden_size (3072 for base, 4096 for large)
        intermediate_size = 4 * hidden_size

        vision_config = SigLip2VisionConfig(
            image_size=image_size,
            patch_size=16,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
        )

        text_config = SigLip2TextConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            **text_kwargs,
        )

        return SigLip2Config(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=hidden_size,
        )

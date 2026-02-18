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

from __future__ import annotations

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """Vision encoder stub representing the image tower used in SmolVLM2.

    In the final implementation this module will align with the reference
    vision backbone (e.g. SigLIP or equivalent) and support loading
    pretrained weights from Hugging Face.
    """

    def __init__(self, hidden_dim: int = 768, output_dim: int = 1152) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, hidden_dim, kernel_size=14, stride=14)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass of the vision encoder.

        Args:
            images: (B, C, H, W)

        Returns:
            features: (B, NumPatches, VisionDim)
        """
        # (B, C, H, W) -> (B, D, H/14, W/14)
        x = self.conv(images)
        # Flatten spatial dimensions: (B, D, N)
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class MultimodalProjector(nn.Module):
    """Projector aligning vision features with the language model space.

    This is a lightweight structural placeholder mirroring the role of the
    multimodal projector in SmolVLM2. The architecture will be refined to
    match the reference implementation in subsequent updates.
    """

    def __init__(self, vision_dim: int = 1152, text_dim: int = 2048) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, text_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(text_dim, text_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multimodal projector.

        Args:
            image_features: (B, NumPatches, VisionDim)

        Returns:
            projected_features: (B, NumPatches, TextDim)
        """
        return self.fc2(self.act(self.fc1(image_features)))


class LanguageModel(nn.Module):
    """Decoder-only language model stub.

    This module represents the transformer-based decoder used in SmolVLM2.
    Attention layers, positional encoding, and full decoder blocks will be
    introduced in future updates to match the reference architecture and
    enable pretrained weight loading.
    """

    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass of the language model.

        Args:
            inputs_embeds: (B, SeqLen, HiddenDim)

        Returns:
            logits: (B, SeqLen, VocabSize)
        """
        return self.head(inputs_embeds)


class SmolVLM2(nn.Module):
    """SmolVLM2 architecture scaffold.

    This module defines the high-level composition of the SmolVLM2
    vision-language model:

        vision encoder → multimodal projector → language model

    The current implementation provides lightweight structural components
    intended to mirror the real architecture and will be extended to match
    the reference implementation and support loading official pretrained
    weights from Hugging Face.

    Note:
        This is not a functional pretrained model yet. Transformer decoder
        blocks, positional handling, and weight-loading utilities will be
        added in subsequent incremental updates.
    """

    def __init__(
        self,
        vision_dim: int = 1152,
        text_dim: int = 2048,
        vocab_size: int = 49152,
    ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.projector = MultimodalProjector(vision_dim=vision_dim, text_dim=text_dim)
        self.language_model = LanguageModel(vocab_size=vocab_size, hidden_dim=text_dim)

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the VLM.

        1. Encode images.
        2. Project image embeddings to text space.
        3. Embed text tokens.
        4. Fuse/Concatenate embeddings.
        5. Pass through Language Model.

        Args:
            images: (B, 3, H, W)
            text_tokens: (B, TextSeqLen)

        Returns:
            logits: (B, TotalSeqLen, VocabSize)
        """
        # 1. Vision Encoding
        # (B, NumPatches, VisionDim)
        image_embeds = self.vision_encoder(images)

        # 2. Projection
        # (B, NumPatches, TextDim)
        projected_images = self.projector(image_embeds)

        # 3. Text Embedding
        # (B, TextSeqLen, TextDim)
        text_embeds = self.language_model.embed(text_tokens)

        # 4. Fusion (Simple concatenation for now)
        # In a real scenario, this would handle masking and specific token replacement
        # (B, NumPatches + TextSeqLen, TextDim)
        inputs_embeds = torch.cat([projected_images, text_embeds], dim=1)

        # 5. Language Modeling
        logits = self.language_model(inputs_embeds)

        return logits
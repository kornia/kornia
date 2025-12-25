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

"""Base classes for Vision-Language Models."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from kornia.core import Tensor


@dataclass
class VisionOutput:
    """Output from a vision encoder model.

    This dataclass provides access to all intermediate representations from the vision encoder,
    which is useful for research and feature extraction.

    Attributes:
        features: Final layer output features of shape (B, num_patches, hidden_dim).
        layer_features: Optional tuple of all layer outputs, each of shape (B, num_patches, hidden_dim).
            Only returned when return_intermediates=True.
        attention_weights: Optional tuple of attention weights from each layer,
            each of shape (B, num_heads, num_patches, num_patches).
            Only returned when return_attention_weights=True.
        pooled: Optional pooled output of shape (B, hidden_dim).
            Typically the [CLS] token or mean-pooled features.

    """

    features: Tensor
    layer_features: Optional[Tuple[Tensor, ...]] = None
    attention_weights: Optional[Tuple[Tensor, ...]] = None
    pooled: Optional[Tensor] = None


@dataclass
class VLMOutput:
    """Output from a Vision-Language Model.

    This dataclass provides access to all intermediate representations from both the
    vision encoder and language decoder, enabling research and interpretability studies.

    Attributes:
        logits: Language model output logits of shape (B, seq_len, vocab_size).
        loss: Optional loss value when labels are provided.
        vision_features: Optional tuple of vision encoder layer outputs.
            Only returned when return_intermediates=True.
        text_features: Optional tuple of language decoder layer outputs.
            Only returned when return_intermediates=True.
        vision_attention: Optional tuple of vision attention weights.
            Only returned when return_attention_weights=True.
        text_attention: Optional tuple of language attention weights.
            Only returned when return_attention_weights=True.
        projected: Vision features after projection to text embedding space,
            of shape (B, num_image_tokens, text_hidden_dim).
        kv_cache: Optional cached key-value states for efficient generation.

    """

    logits: Tensor
    loss: Optional[Tensor] = None
    vision_features: Optional[Tuple[Tensor, ...]] = None
    text_features: Optional[Tuple[Tensor, ...]] = None
    vision_attention: Optional[Tuple[Tensor, ...]] = None
    text_attention: Optional[Tuple[Tensor, ...]] = None
    projected: Optional[Tensor] = None
    kv_cache: Optional[Tuple[Tuple[Tensor, ...], ...]] = None


class VLMBase(nn.Module):
    """Abstract base class for Vision-Language Models.

    This class defines the interface that all VLM implementations should follow,
    providing a consistent API for model loading, inference, and generation.

    Subclasses must implement:
        - forward(): Full forward pass through the model
        - extract_vision_features(): Extract vision encoder features
        - generate(): Autoregressive text generation
        - from_hub(): Load pretrained weights

    Attributes:
        vision_tower: The vision encoder module (e.g., SigLIP, ViT).
        text_decoder: The language decoder module (e.g., Gemma, LLaMA).
        connector: The multimodal projection layer.

    Example:
        >>> model = PaliGemma2.from_hub("google/paligemma2-3b-pt-224")
        >>> # Extract vision features
        >>> vision_output = model.extract_vision_features(images)
        >>> # Full forward pass with intermediate states
        >>> output = model(images, token_ids, return_intermediates=True)
        >>> # Generate text
        >>> generated = model.generate(images, prompt="Describe this image:")

    """

    vision_tower: nn.Module
    text_decoder: nn.Module
    connector: nn.Module

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        token_ids: Tensor,
        mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_intermediates: bool = False,
        return_attention_weights: bool = False,
    ) -> VLMOutput:
        """Forward pass through the Vision-Language Model.

        Args:
            images: Input images of shape (B, C, H, W).
            token_ids: Tokenized text input of shape (B, seq_len).
            mask: Optional attention mask of shape (B, seq_len).
            labels: Optional labels for computing loss, shape (B, seq_len).
            return_intermediates: Whether to return all hidden states.
            return_attention_weights: Whether to return attention weights.

        Returns:
            VLMOutput containing logits and optional intermediate representations.

        """
        raise NotImplementedError

    @abstractmethod
    def extract_vision_features(
        self,
        images: Tensor,
        return_intermediates: bool = False,
        return_attention_weights: bool = False,
    ) -> VisionOutput:
        """Extract features from the vision encoder only.

        This method provides direct access to vision encoder outputs without
        going through the full VLM pipeline, useful for feature extraction
        and research.

        Args:
            images: Input images of shape (B, C, H, W).
            return_intermediates: Whether to return all layer hidden states.
            return_attention_weights: Whether to return attention weights.

        Returns:
            VisionOutput with features and optional intermediates.

        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        images: Tensor,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        sample: bool = False,
    ) -> str:
        """Generate text given an image and prompt.

        Args:
            images: Input image of shape (B, C, H, W) or (C, H, W).
            prompt: Text prompt to condition generation.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            sample: Whether to use sampling (vs greedy decoding).

        Returns:
            Generated text string.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_hub(
        cls,
        model_id: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ) -> "VLMBase":
        """Load a pretrained model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier (e.g., "google/paligemma2-3b-pt-224").
            device: Device to load the model on.
            dtype: Data type for model parameters.
            cache_dir: Directory to cache downloaded weights.

        Returns:
            Instantiated model with pretrained weights.

        """
        raise NotImplementedError

    def get_image_embeddings(
        self,
        images: Tensor,
        return_intermediates: bool = False,
        return_attention_weights: bool = False,
    ) -> VisionOutput:
        """Alias for extract_vision_features() for convenience.

        Args:
            images: Input images of shape (B, C, H, W).
            return_intermediates: Whether to return all layer hidden states.
            return_attention_weights: Whether to return attention weights.

        Returns:
            VisionOutput with vision encoder features.

        """
        return self.extract_vision_features(images, return_intermediates, return_attention_weights)

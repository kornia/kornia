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

"""PaliGemma 2 Vision-Language Model implementation.

PaliGemma combines a SigLIP vision encoder with a Gemma language decoder
for multimodal understanding and generation tasks.

Reference: https://arxiv.org/abs/2407.07726
"""

from typing import Optional, Tuple

import torch
from torch import nn

from kornia.core import Tensor

from ..base import VisionOutput, VLMBase, VLMOutput
from .config import PaliGemma2Config
from .gemma import GemmaLM
from .siglip import SiglipVisionEncoder


class VisionTextConnector(nn.Module):
    """Linear projection connecting vision and text embedding spaces.

    Projects vision encoder outputs to match the language model's hidden dimension.

    Args:
        vision_dim: Hidden dimension of vision encoder.
        text_dim: Hidden dimension of language model.

    """

    def __init__(self, vision_dim: int, text_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(vision_dim, text_dim, bias=True)

    def forward(self, visual_features: Tensor) -> Tensor:
        """Project vision features to text space.

        Args:
            visual_features: Vision features of shape (B, n_patches, vision_dim).

        Returns:
            Projected features of shape (B, n_patches, text_dim).

        """
        return self.proj(visual_features)


class PaliGemma2(VLMBase):
    """PaliGemma 2 Vision-Language Model.

    Combines Siglip vision encoder with Gemma language decoder for multimodal
    understanding and generation. Designed for research with easy access to
    intermediate representations.

    Args:
        config: PaliGemma 2 configuration.

    Example:
        >>> config = PaliGemma2Config.paligemma2_3b_224()
        >>> model = PaliGemma2(config)
        >>> images = torch.randn(1, 3, 224, 224)
        >>> # Extract vision features
        >>> vision_output = model.extract_vision_features(images)
        >>> print(vision_output.features.shape)
        >>> # Full forward pass
        >>> token_ids = torch.randint(0, 1000, (1, 10))
        >>> output = model(images, token_ids)

    """

    def __init__(self, config: PaliGemma2Config) -> None:
        super().__init__()
        self.config = config

        # Vision encoder (Siglip)
        self.vision_tower = SiglipVisionEncoder(config.vision_config)

        # Multimodal connector
        self.connector = VisionTextConnector(
            vision_dim=config.vision_config.hidden_size,
            text_dim=config.text_config.hidden_size,
        )

        # Language model (Gemma)
        self.text_decoder = GemmaLM(config.text_config)

        # Special token indices
        self.image_token_idx = config.image_token_index
        self.pad_token_idx = config.pad_token_id

    def extract_vision_features(
        self,
        images: Tensor,
        return_intermediates: bool = False,
        return_attention_weights: bool = False,
    ) -> VisionOutput:
        """Extract features from the vision encoder.

        This method provides direct access to vision encoder outputs without
        going through the full VLM pipeline.

        Args:
            images: Input images of shape (B, C, H, W).
            return_intermediates: Whether to return all layer features.
            return_attention_weights: Whether to return attention weights.

        Returns:
            VisionOutput with vision encoder features.

        Example:
            >>> images = torch.randn(2, 3, 224, 224)
            >>> features = model.extract_vision_features(images, return_intermediates=True)
            >>> features.features.shape
            torch.Size([2, 256, 1152])
            >>> len(features.layer_features)  # All 27 layers + embedding
            28

        """
        return self.vision_tower(
            images,
            return_attention_weights=return_attention_weights,
            return_intermediates=return_intermediates,
        )

    def _fuse_image_text_embeddings(
        self,
        visual_embeds: Tensor,
        text_embeds: Tensor,
        token_ids: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Fuse image features with text embeddings.

        Replaces image token positions in the text embeddings with projected
        vision features.

        Args:
            visual_embeds: Projected vision features (B, n_patches, dim).
            text_embeds: Text embeddings (B, seq_len, dim).
            token_ids: Token IDs (B, seq_len).
            mask: Attention mask (B, seq_len).

        Returns:
            Tuple of (fused_embeds, mask, positions).

        """
        B, n_patches, dim = visual_embeds.shape
        _, L, _ = text_embeds.shape

        # Find image token positions
        is_image_token = token_ids == self.image_token_idx

        # Count expected vs actual image tokens
        n_image_tokens = is_image_token.sum(dim=1)

        # Create output embeddings
        fused = text_embeds.clone()

        # Replace image tokens with image features
        for batch_idx in range(B):
            image_positions = is_image_token[batch_idx].nonzero(as_tuple=True)[0]
            if len(image_positions) > 0:
                # Take only as many image features as there are image tokens
                n_to_insert = min(len(image_positions), n_patches)
                fused[batch_idx, image_positions[:n_to_insert]] = visual_embeds[batch_idx, :n_to_insert]

        # Create position IDs - sequential indices starting from 0
        # Transformers uses sequential position IDs for all tokens
        # Padding is handled by attention mask, not position IDs
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(B, -1)

        return fused, mask, positions

    def forward(
        self,
        images: Optional[Tensor] = None,
        token_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_intermediates: bool = False,
        return_attention_weights: bool = False,
        use_cache: bool = False,
        pixel_values: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> VLMOutput:
        """Forward pass through the Vision-Language Model.

        Args:
            images: Input images of shape (B, C, H, W). Can use pixel_values instead.
            token_ids: Tokenized text input of shape (B, seq_len). Can use input_ids instead.
            mask: Optional attention mask of shape (B, seq_len). Can use attention_mask instead.
            labels: Optional labels for computing loss, shape (B, seq_len).
            return_intermediates: Whether to return all layer features.
            return_attention_weights: Whether to return attention weights.
            use_cache: Whether to return key-value cache.
            pixel_values: Transformers-compatible alias for images.
            input_ids: Transformers-compatible alias for token_ids.
            attention_mask: Transformers-compatible alias for mask.

        Returns:
            VLMOutput containing logits and optional intermediate representations.

        """
        # Support transformers-compatible argument names
        if images is None and pixel_values is not None:
            images = pixel_values
        if token_ids is None and input_ids is not None:
            token_ids = input_ids
        if mask is None and attention_mask is not None:
            mask = attention_mask

        if images is None:
            raise ValueError("Either images or pixel_values must be provided")
        if token_ids is None:
            raise ValueError("Either token_ids or input_ids must be provided")

        B = images.shape[0]

        if mask is None:
            mask = torch.ones(token_ids.shape, dtype=torch.long, device=token_ids.device)

        # Encode images
        vision_out = self.vision_tower(
            images,
            return_attention_weights=return_attention_weights,
            return_intermediates=return_intermediates,
        )

        # Project vision features to text space
        visual_features = vision_out.features
        projected = self.connector(visual_features)

        # Get text embeddings
        text_embeds = self.text_decoder.get_input_embeddings()(token_ids)

        # Fuse image and text embeddings
        fused_embeds, mask, positions = self._fuse_image_text_embeddings(projected, text_embeds, token_ids, mask)

        # Forward through language model
        lm_out = self.text_decoder(
            input_ids=None,
            attention_mask=mask,
            position_ids=positions,
            inputs_embeds=fused_embeds,
            labels=labels,
            output_attentions=return_attention_weights,
            output_hidden_states=return_intermediates,
            use_cache=use_cache,
        )

        loss, logits, text_features, text_attn, kv_cache = lm_out

        return VLMOutput(
            logits=logits,
            loss=loss,
            vision_features=vision_out.layer_features,
            text_features=text_features,
            vision_attention=vision_out.attention_weights,
            text_attention=text_attn,
            projected=projected,
            kv_cache=kv_cache,
        )

    @torch.no_grad()
    def generate(
        self,
        images: Tensor,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        sample: bool = False,
        tokenizer=None,
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
            tokenizer: Tokenizer for encoding/decoding text.

        Returns:
            Generated text string.

        Raises:
            ValueError: If tokenizer is not provided.

        """
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for generation")

        # Ensure batch dimension
        if images.dim() == 3:
            images = images.unsqueeze(0)

        device = images.device
        B = images.shape[0]

        # Encode prompt
        encoded = tokenizer(prompt, return_tensors="pt", padding=True)
        token_ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)

        # Encode images
        vision_out = self.vision_tower(images)
        visual_features = vision_out.features
        projected = self.connector(visual_features)

        # Get initial embeddings
        text_embeds = self.text_decoder.get_input_embeddings()(token_ids)

        # Fuse image and text
        fused_embeds, mask, positions = self._fuse_image_text_embeddings(projected, text_embeds, token_ids, mask)

        # Generate
        kv_cache = None
        generated_tokens = []

        for _ in range(max_tokens):
            if kv_cache is None:
                # First step: use fused embeddings
                outputs = self.text_decoder(
                    input_ids=None,
                    inputs_embeds=fused_embeds,
                    attention_mask=mask,
                    position_ids=positions,
                    use_cache=True,
                )
            else:
                # Subsequent steps: use only new token
                outputs = self.text_decoder(
                    input_ids=next_tokens,
                    attention_mask=mask,
                    past_key_values=kv_cache,
                    use_cache=True,
                )

            _, logits, _, _, kv_cache = outputs
            next_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Sample or greedy
            if sample:
                # Top-k filtering
                if top_k > 0:
                    cutoff = torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[next_logits < cutoff] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumsum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = 0
                    remove_idx = remove.scatter(1, sorted_idx, remove)
                    next_logits[remove_idx] = float("-inf")

                probs = torch.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated_tokens.append(next_tokens)

            # Update attention mask
            mask = torch.cat([mask, torch.ones((B, 1), device=device)], dim=-1)

            # Check for EOS
            if (next_tokens == self.config.eos_token_id).all():
                break

        # Decode generated tokens
        generated_ids = torch.cat(generated_tokens, dim=-1)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text

    @classmethod
    def from_hub(
        cls,
        model_id: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ) -> "PaliGemma2":
        """Load a pretrained model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier (e.g., "google/paligemma2-3b-pt-224").
            device: Device to load the model on.
            dtype: Data type for model parameters.
            cache_dir: Directory to cache downloaded weights.

        Returns:
            Instantiated model with pretrained weights.

        Example:
            >>> model = PaliGemma2.from_hub("google/paligemma2-3b-pt-224")
            >>> model = model.to("cuda")

        """
        from ..utils import load_paligemma_weights

        # Determine config based on model_id
        if "224" in model_id:
            config = PaliGemma2Config.paligemma2_3b_224()
        elif "448" in model_id:
            config = PaliGemma2Config.paligemma2_3b_448()
        elif "896" in model_id:
            config = PaliGemma2Config.paligemma2_3b_896()
        else:
            config = PaliGemma2Config.paligemma2_3b_224()

        # Create model
        model = cls(config)

        # Load weights
        load_paligemma_weights(model, model_id, cache_dir=cache_dir)

        # Move to device and dtype
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        return model

    @classmethod
    def from_config(cls, config: PaliGemma2Config) -> "PaliGemma2":
        """Create a model from configuration (random weights).

        Args:
            config: PaliGemma 2 configuration.

        Returns:
            Instantiated model with random weights.

        """
        return cls(config)

    # Keep backward compatibility alias
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ) -> "PaliGemma2":
        """Alias for from_hub() for compatibility."""
        return cls.from_hub(model_id, device, dtype, cache_dir)

    def encode_image(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> VisionOutput:
        """Alias for extract_vision_features() for compatibility."""
        return self.extract_vision_features(pixel_values, output_hidden_states, output_attentions)

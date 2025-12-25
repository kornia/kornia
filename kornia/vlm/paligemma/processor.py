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

"""Preprocessing utilities for PaliGemma models."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from kornia.core import Tensor


@dataclass
class ProcessorOutput:
    """Output from the PaliGemma processor.

    Attributes:
        pixel_values: Preprocessed images of shape (B, C, H, W).
        input_ids: Tokenized text of shape (B, seq_len).
        attention_mask: Attention mask of shape (B, seq_len).

    """

    pixel_values: Tensor
    input_ids: Tensor
    attention_mask: Tensor


class PaliGemmaImageProcessor:
    """Image preprocessor for PaliGemma models.

    Handles resizing, normalization, and conversion to tensor format.

    Args:
        image_size: Target image size.
        mean: Normalization mean (RGB).
        std: Normalization std (RGB).

    Example:
        >>> processor = PaliGemmaImageProcessor(image_size=224)
        >>> images = torch.randn(2, 3, 256, 256)
        >>> processed = processor(images)
        >>> processed.shape
        torch.Size([2, 3, 224, 224])

    """

    def __init__(
        self,
        image_size: int = 224,
        mean: tuple = (0.5, 0.5, 0.5),
        std: tuple = (0.5, 0.5, 0.5),
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(
        self,
        images: Union[Tensor, List[Tensor]],
        return_tensors: str = "pt",
    ) -> Tensor:
        """Preprocess images for PaliGemma.

        Args:
            images: Input images as tensor(s) of shape (C, H, W) or (B, C, H, W).
                Values should be in [0, 1] range.
            return_tensors: Output format ("pt" for PyTorch).

        Returns:
            Preprocessed images of shape (B, C, H, W).

        """
        # Handle list of images
        if isinstance(images, list):
            images = torch.stack(images)

        # Ensure batch dimension
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Resize to target size
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Normalize
        mean = torch.tensor(self.mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std

        return images


class PaliGemmaTokenizer:
    """Simple tokenizer wrapper for PaliGemma.

    Wraps a SentencePiece tokenizer and handles image token insertion.

    Args:
        tokenizer_path: Path to the SentencePiece model file.
        image_token: Special token for image placeholders.
        image_token_id: Token ID for the image token.
        num_image_tokens: Number of image tokens to insert.

    """

    def __init__(
        self,
        tokenizer_path: Optional[Union[str, Path]] = None,
        image_token: str = "<image>",
        image_token_id: int = 257152,
        num_image_tokens: int = 256,
    ) -> None:
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens

        self._tokenizer = None
        if tokenizer_path is not None:
            self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, path: Union[str, Path]) -> None:
        """Load SentencePiece tokenizer.

        Args:
            path: Path to the tokenizer model.

        """
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError("sentencepiece is required. Install with: pip install sentencepiece") from e

        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.Load(str(path))

    @classmethod
    def from_pretrained(cls, model_id: str, cache_dir: Optional[str] = None) -> "PaliGemmaTokenizer":
        """Load tokenizer from HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier.
            cache_dir: Optional cache directory.

        Returns:
            Loaded tokenizer.

        """
        from ..utils import download_hf_weights

        model_path = download_hf_weights(model_id, cache_dir=cache_dir)

        # Find tokenizer file
        tokenizer_file = model_path / "tokenizer.model"
        if not tokenizer_file.exists():
            # Try alternative name
            tokenizer_file = next(model_path.glob("*.model"), None)

        if tokenizer_file is None:
            raise FileNotFoundError(f"No tokenizer found in {model_path}")

        return cls(tokenizer_path=tokenizer_file)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text.
            add_bos: Whether to add BOS token.
            add_eos: Whether to add EOS token.

        Returns:
            List of token IDs.

        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        tokens = self._tokenizer.Encode(text)

        if add_bos:
            tokens = [self._tokenizer.bos_id()] + tokens
        if add_eos:
            tokens = tokens + [self._tokenizer.eos_id()]

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            Decoded text.

        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        if isinstance(token_ids, Tensor):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            # Filter out special tokens
            bos_id = self._tokenizer.bos_id()
            eos_id = self._tokenizer.eos_id()
            token_ids = [t for t in token_ids if t not in (bos_id, eos_id, self.image_token_id)]

        return self._tokenizer.Decode(token_ids)

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        max_length: Optional[int] = None,
        add_image_tokens: bool = True,
    ) -> dict:
        """Tokenize text with optional image token insertion.

        Args:
            text: Input text or list of texts.
            return_tensors: Output format ("pt" for PyTorch).
            padding: Whether to pad to same length.
            max_length: Maximum sequence length.
            add_image_tokens: Whether to prepend image tokens.

        Returns:
            Dictionary with input_ids and attention_mask.

        """
        if isinstance(text, str):
            text = [text]

        all_input_ids = []
        for t in text:
            tokens = self.encode(t)

            if add_image_tokens:
                # Prepend image tokens
                image_tokens = [self.image_token_id] * self.num_image_tokens
                tokens = image_tokens + tokens

            if max_length is not None:
                tokens = tokens[:max_length]

            all_input_ids.append(tokens)

        # Pad if needed
        if padding:
            max_len = max(len(ids) for ids in all_input_ids)
            attention_masks = []

            for i, ids in enumerate(all_input_ids):
                pad_len = max_len - len(ids)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
                all_input_ids[i] = ids + [0] * pad_len  # Pad with 0

        else:
            attention_masks = [[1] * len(ids) for ids in all_input_ids]

        if return_tensors == "pt":
            input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        else:
            input_ids = all_input_ids
            attention_mask = attention_masks

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class PaliGemmaProcessor:
    """Combined processor for PaliGemma models.

    Handles both image preprocessing and text tokenization.

    Args:
        image_size: Target image size.
        tokenizer_path: Path to tokenizer model.

    Example:
        >>> processor = PaliGemmaProcessor(image_size=224)
        >>> images = torch.randn(1, 3, 256, 256)
        >>> output = processor(images=images, text="Describe this image")

    """

    def __init__(
        self,
        image_size: int = 224,
        tokenizer_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.image_processor = PaliGemmaImageProcessor(image_size=image_size)
        self.tokenizer = PaliGemmaTokenizer(tokenizer_path=tokenizer_path)

        # Number of image tokens based on image size and patch size
        num_patches = (image_size // 14) ** 2
        self.tokenizer.num_image_tokens = num_patches

    @classmethod
    def from_pretrained(cls, model_id: str, cache_dir: Optional[str] = None) -> "PaliGemmaProcessor":
        """Load processor from HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier.
            cache_dir: Optional cache directory.

        Returns:
            Loaded processor.

        """
        from ..utils import download_hf_weights

        # Determine image size from model_id
        if "224" in model_id:
            image_size = 224
        elif "448" in model_id:
            image_size = 448
        elif "896" in model_id:
            image_size = 896
        else:
            image_size = 224

        model_path = download_hf_weights(model_id, cache_dir=cache_dir)

        # Find tokenizer file
        tokenizer_file = model_path / "tokenizer.model"
        if not tokenizer_file.exists():
            tokenizer_file = next(model_path.glob("*.model"), None)

        processor = cls(image_size=image_size, tokenizer_path=tokenizer_file)
        return processor

    def __call__(
        self,
        images: Optional[Union[Tensor, List[Tensor]]] = None,
        text: Optional[Union[str, List[str]]] = None,
        return_tensors: str = "pt",
        padding: bool = True,
        max_length: Optional[int] = None,
    ) -> ProcessorOutput:
        """Process images and text for PaliGemma.

        Args:
            images: Input images.
            text: Input text.
            return_tensors: Output format.
            padding: Whether to pad sequences.
            max_length: Maximum sequence length.

        Returns:
            ProcessorOutput with pixel_values, input_ids, attention_mask.

        """
        pixel_values = None
        if images is not None:
            pixel_values = self.image_processor(images, return_tensors=return_tensors)

        input_ids = None
        attention_mask = None
        if text is not None:
            encoded = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_length,
                add_image_tokens=(images is not None),
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        return ProcessorOutput(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

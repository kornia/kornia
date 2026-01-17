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

"""Weight loader for converting HuggingFace Qwen2.5-VL weights to kornia format."""

from __future__ import annotations

import re

import torch


class Qwen25WeightLoader:
    """Load and convert HuggingFace Qwen2.5-VL weights to kornia format.

    This class handles downloading pretrained weights from HuggingFace and
    converting the weight naming convention to match kornia's architecture.

    Args:
        model_id: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct").

    Example:
        >>> loader = Qwen25WeightLoader("Qwen/Qwen2.5-VL-3B-Instruct")
        >>> vision_weights = loader.load_weights("vision_encoder")
        >>> model.load_state_dict(vision_weights)
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        self.model_id = model_id
        self.hf_to_kornia_map = self._build_key_mapping()

    def _build_key_mapping(self) -> dict[str, str]:
        """Build HuggingFace to kornia key mapping for vision encoder.

        Returns:
            Dictionary mapping HF keys to kornia keys. Uses {i} as placeholder
            for layer indices that will be replaced during conversion.
        """
        return {
            # Patch embedder (conv layer - no bias)
            "visual.patch_embed.proj.weight": "patch_embed.conv.weight",
            # Note: No bias, no LayerNorm in patch_embed
            # Vision blocks - attention
            "visual.blocks.{i}.norm1.weight": "blocks.{i}.norm1.weight",
            # No norm1.bias - LayerNorm doesn't use bias
            "visual.blocks.{i}.attn.qkv.weight": "blocks.{i}.attn.qkv.weight",
            "visual.blocks.{i}.attn.qkv.bias": "blocks.{i}.attn.qkv.bias",
            "visual.blocks.{i}.attn.proj.weight": "blocks.{i}.attn.proj.weight",
            "visual.blocks.{i}.attn.proj.bias": "blocks.{i}.attn.proj.bias",
            # Vision blocks - Gated MLP (SwiGLU)
            "visual.blocks.{i}.norm2.weight": "blocks.{i}.norm2.weight",
            # No norm2.bias - LayerNorm doesn't use bias
            "visual.blocks.{i}.mlp.gate_proj.weight": "blocks.{i}.mlp.gate_proj.weight",
            "visual.blocks.{i}.mlp.gate_proj.bias": "blocks.{i}.mlp.gate_proj.bias",
            "visual.blocks.{i}.mlp.up_proj.weight": "blocks.{i}.mlp.up_proj.weight",
            "visual.blocks.{i}.mlp.up_proj.bias": "blocks.{i}.mlp.up_proj.bias",
            "visual.blocks.{i}.mlp.down_proj.weight": "blocks.{i}.mlp.down_proj.weight",
            "visual.blocks.{i}.mlp.down_proj.bias": "blocks.{i}.mlp.down_proj.bias",
            # Final merger (2-layer MLP with LayerNorm - no bias in LN)
            "visual.merger.ln_q.weight": "merger.ln_q.weight",
            # No merger.ln_q.bias - LayerNorm doesn't use bias
            "visual.merger.mlp.0.weight": "merger.mlp.0.weight",
            "visual.merger.mlp.0.bias": "merger.mlp.0.bias",
            "visual.merger.mlp.2.weight": "merger.mlp.2.weight",
            "visual.merger.mlp.2.bias": "merger.mlp.2.bias",
        }

    def _download_hf_weights(self) -> dict[str, torch.Tensor]:
        """Download weights from HuggingFace Hub.

        Returns:
            State dictionary with HuggingFace key names.

        Raises:
            ImportError: If required libraries are not installed.
        """
        try:
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
        except ImportError as e:
            raise ImportError(
                "huggingface_hub and safetensors libraries are required. "
                "Install with: pip install huggingface_hub safetensors"
            ) from e

        # Download checkpoint files dynamically
        # First, try to get the index file to discover shard count
        state_dict = {}
        shard_files = []

        try:
            # Try to download model.safetensors.index.json to discover shards
            import json

            from huggingface_hub import hf_hub_download

            index_path = hf_hub_download(
                repo_id=self.model_id,
                filename="model.safetensors.index.json",
            )
            with open(index_path) as f:
                index = json.load(f)
                # Get unique shard filenames
                shard_files = list(set(index["weight_map"].values()))
        except Exception:
            # Fallback: assume 2 shards (for Qwen2.5-VL-3B)
            shard_files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

        for filename in shard_files:
            try:
                # Download the checkpoint file
                checkpoint_path = hf_hub_download(
                    repo_id=self.model_id,
                    filename=filename,
                )

                # Load weights from safetensors (filter for vision encoder only)
                # Load to GPU if available to reduce system RAM usage
                device = "cuda" if torch.cuda.is_available() else "cpu"
                with safe_open(checkpoint_path, framework="pt", device=device) as f:
                    for key in f.keys():
                        # Only load vision encoder to reduce memory (7GB -> 1.6GB)
                        if key.startswith("visual."):
                            state_dict[key] = f.get_tensor(key)

            except Exception as e:
                print(f"Note: Could not load {filename}: {e}")
                # If we can't load both shards, that's okay - we might have what we need
                continue

        if not state_dict:
            raise RuntimeError(
                f"Failed to load any weights from {self.model_id}. Please check your internet connection and model_id."
            )

        return state_dict

    def _convert_key(self, hf_key: str, pattern: str, kornia_pattern: str) -> str:
        """Convert a single HuggingFace key to kornia format.

        Args:
            hf_key: Original HuggingFace key.
            pattern: HF pattern with {i} placeholders.
            kornia_pattern: Kornia pattern with {i} placeholders.

        Returns:
            Converted kornia key.
        """
        # Replace {i} with regex pattern to extract indices
        # Use fullmatch to avoid partial matches
        regex_pattern = pattern.replace(".", r"\.").replace("{i}", r"(\d+)")
        match = re.fullmatch(regex_pattern, hf_key)

        if not match:
            # Return None if no match instead of returning invalid pattern
            return None

        # Replace each {i} in kornia pattern with extracted index
        result = kornia_pattern
        for idx in match.groups():
            result = result.replace("{i}", idx, 1)

        return result

    def _belongs_to_component(self, key: str, component: str) -> bool:
        """Check if a key belongs to the specified component.

        Args:
            key: Weight key name.
            component: Component name ("vision_encoder", "projector", "decoder", or "all").

        Returns:
            True if key belongs to component.
        """
        if component == "all":
            return True
        elif component == "vision_encoder":
            return key.startswith(("patch_embed", "blocks", "rotary_pos_emb", "merger"))
        elif component == "projector":
            return key.startswith("projector")
        elif component == "decoder":
            return key.startswith("decoder")
        return False

    def load_weights(self, component: str = "vision_encoder") -> dict[str, torch.Tensor]:
        """Load and convert weights for specified component.

        Args:
            component: Which component to load weights for. Options:
                - "vision_encoder": Vision encoder only (default)
                - "projector": Projector only
                - "decoder": Decoder only
                - "all": All components

        Returns:
            State dictionary with kornia key names.

        Example:
            >>> loader = Qwen25WeightLoader()
            >>> vision_weights = loader.load_weights("vision_encoder")
            >>> print(f"Loaded {len(vision_weights)} parameters")
        """
        # Download HF weights
        hf_state_dict = self._download_hf_weights()

        # Convert keys
        kornia_state_dict = {}
        matched_count = 0
        vision_count = 0

        for hf_key, tensor in hf_state_dict.items():
            if "visual" in hf_key:
                vision_count += 1
            
            # Try to match against each pattern
            converted = False
            for hf_pattern, kornia_pattern in self.hf_to_kornia_map.items():
                # Check if key matches pattern (use fullmatch for exact matching)
                regex_pattern = hf_pattern.replace(".", r"\.").replace("{i}", r"\d+")
                if re.fullmatch(regex_pattern, hf_key):
                    kornia_key = self._convert_key(hf_key, hf_pattern, kornia_pattern)

                    # Skip if conversion failed
                    if kornia_key is None:
                        continue

                    # Filter by component
                    if self._belongs_to_component(kornia_key, component):
                        kornia_state_dict[kornia_key] = tensor
                        converted = True
                        matched_count += 1
                        break

        return kornia_state_dict

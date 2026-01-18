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

"""ONNX Vision Encoder Inference Pipeline

Loads and chains the 34 ONNX components to provide a single inference interface.
"""

from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort


class Qwen2VLVisionEncoderONNX:
    """ONNX inference pipeline for Qwen2.5-VL Vision Encoder.

    Loads 34 separate ONNX components and chains them together:
    image → patch_embed → 32 transformer blocks → merger → features

    Example:
        >>> encoder = Qwen2VLVisionEncoderONNX("onnx_model/components_full")
        >>> image = np.random.randn(1, 3, 448, 448).astype(np.float32)
        >>> features = encoder(image)
        >>> print(features.shape)  # (1, 256, 2048)
    """
    
    def __init__(self, components_dir: Union[str, Path], providers: list = None):
        """Initialize the ONNX vision encoder pipeline.

        Args:
            components_dir: Directory containing the 34 ONNX component files
            providers: ONNX Runtime execution providers (e.g., ['CUDAExecutionProvider'])
                      If None, uses default providers (CPU)
        """
        self.components_dir = Path(components_dir)
        self.providers = providers
        
        if not self.components_dir.exists():
            raise FileNotFoundError(f"Components directory not found: {components_dir}")

        print("Loading ONNX Vision Encoder components...")

        # Load patch embedding
        self.patch_embed = self._load_component("patch_embed.onnx")

        # Load all 32 transformer blocks
        self.blocks = []
        for i in range(32):
            block = self._load_component(f"block_{i:02d}.onnx")
            self.blocks.append(block)

        # Load merger
        self.merger = self._load_component("merger.onnx")
        
        print(f"Loaded 34 components from {components_dir}")
    
    def _load_component(self, filename: str) -> ort.InferenceSession:
        """Load a single ONNX component."""
        path = self.components_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Component not found: {path}")
        if self.providers:
            return ort.InferenceSession(str(path), providers=self.providers)
        return ort.InferenceSession(str(path))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run inference on an input image.

        Args:
            image: Input image tensor of shape (B, 3, H, W) as float32.
                   Works with any H and W (will be padded to multiples of 14 if needed).
        
        Returns:
            Feature tensor of shape (B, num_tokens, 2048)
            where num_tokens depends on input size
        """
        # Validate input
        if image.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {image.shape}")
        if image.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[1]}")
        
        # Compute grid dimensions from image size
        # The model pads internally to multiples of 14, so we match that
        H, W = image.shape[2:4]
        grid_h = (H + 13) // 14  # Rounds up to nearest 14
        grid_w = (W + 13) // 14
        
        # 1. Patch embedding
        x = self.patch_embed.run(None, {"image": image})[0]

        # 2. Transformer blocks (32 layers)
        for block in self.blocks:
            x = block.run(None, {"input": x})[0]
        
        # 3. Merger (spatial compression) - pass grid dimensions
        grid_h_tensor = np.array(grid_h, dtype=np.int64)
        grid_w_tensor = np.array(grid_w, dtype=np.int64)
        output = self.merger.run(
            None, 
            {
                "input": x,
                "grid_h": grid_h_tensor,
                "grid_w": grid_w_tensor,
            }
        )[0]
        
        return output

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Alias for __call__ for explicit inference."""
        return self(image)


def main():
    """Example usage and validation."""
    import argparse

    parser = argparse.ArgumentParser(description="ONNX Vision Encoder Inference")
    parser.add_argument(
        "--components-dir", type=str, default="onnx_model/components_full", help="Directory containing ONNX components"
    )
    parser.add_argument("--validate", action="store_true", help="Validate against PyTorch model")
    args = parser.parse_args()

    # Load ONNX pipeline
    encoder = Qwen2VLVisionEncoderONNX(args.components_dir)

    # Create test input
    print("\nRunning inference on test image (448x448)...")
    test_image = np.random.randn(1, 3, 448, 448).astype(np.float32)

    # Run inference
    features = encoder(test_image)
    
    print(f"\nInference successful!")
    print(f"  Input shape: {test_image.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Output mean: {features.mean():.6f}")
    print(f"  Output std: {features.std():.6f}")

    # Optional: Validate against PyTorch
    if args.validate:
        print("\nValidating against PyTorch model...")
        import os

        import torch

        os.environ["HF_HOME"] = "onnx_model/hf_cache"

        from kornia.models.qwen25 import Qwen2VLVisionTransformer

        model_pytorch = Qwen2VLVisionTransformer.from_pretrained()
        model_pytorch.eval().cpu()

        with torch.no_grad():
            pytorch_output = model_pytorch(torch.from_numpy(test_image))

        pytorch_output_np = pytorch_output.numpy()

        diff = np.abs(features - pytorch_output_np)
        max_diff = diff.max()
        mean_diff = diff.mean()

        print(f"\n  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-3:
            print("\n  Validation PASSED!")
        else:
            print(f"\n  Validation failed (max diff: {max_diff:.6f})")


if __name__ == "__main__":
    main()

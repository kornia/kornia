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

"""Export Qwen2.5-VL Vision Encoder to ONNX format.

Due to a PyTorch limitation with torch.onnx.export and load_state_dict,
the model is exported as separate components that can be chained together.

Usage:
    python -m kornia.models.qwen25.export_onnx_components --output-dir ./onnx_components

The exported components can be loaded using Qwen2VLVisionEncoderONNX:
    from kornia.models.qwen25 import Qwen2VLVisionEncoderONNX
    encoder = Qwen2VLVisionEncoderONNX("./onnx_components")
    features = encoder(image)
"""

import argparse
from pathlib import Path
from typing import Union

import torch
from torch import nn


class _MergerWrapper(nn.Module):
    """Wrapper to provide fixed grid dimensions to merger for ONNX export."""

    def __init__(self, merger: nn.Module, grid_h: int = 32, grid_w: int = 32):
        super().__init__()
        self.merger = merger
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merger(x, self.grid_h, self.grid_w)


def export_vision_encoder_components(
    output_dir: Union[str, Path],
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    opset_version: int = 17,
    input_size: tuple = (448, 448),
) -> None:
    """Export Qwen2.5-VL Vision Encoder as separate ONNX components.

    Args:
        output_dir: Directory to save ONNX components.
        model_id: HuggingFace model ID for pretrained weights.
        opset_version: ONNX opset version.
        input_size: Input image size (H, W). Must be divisible by 28.

    The exported components are:
        - patch_embed.onnx: Converts image to patch embeddings
        - block_00.onnx to block_31.onnx: 32 transformer blocks
        - merger.onnx: Spatial compression layer
    """
    from .qwen2_vl import Qwen2VLVisionTransformer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained model from {model_id}...")
    model = Qwen2VLVisionTransformer.from_pretrained(model_id)
    model.eval()
    model = model.cpu()

    h, w = input_size
    grid_h, grid_w = h // 14, w // 14

    print(f"\nExporting to: {output_dir}")
    print("=" * 60)

    # 1. Export Patch Embedding
    print("\n[1/34] Exporting patch_embed...")
    dummy_img = torch.randn(1, 3, h, w)
    torch.onnx.export(
        model.patch_embed,
        dummy_img,
        str(output_dir / "patch_embed.onnx"),
        opset_version=opset_version,
        export_params=True,
        input_names=["image"],
        output_names=["patches"],
    )
    size = (output_dir / "patch_embed.onnx").stat().st_size / 1024 / 1024
    print(f"  ✓ patch_embed.onnx: {size:.2f} MB")

    # 2. Export Transformer Blocks
    print("\n[2-33/34] Exporting transformer blocks...")
    seq_len = grid_h * grid_w
    dummy_patches = torch.randn(1, seq_len, model.embed_dim)

    total_block_size = 0
    for i in range(len(model.blocks)):
        torch.onnx.export(
            model.blocks[i],
            dummy_patches,
            str(output_dir / f"block_{i:02d}.onnx"),
            opset_version=opset_version,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
        )
        size = (output_dir / f"block_{i:02d}.onnx").stat().st_size / 1024 / 1024
        total_block_size += size

        if (i + 1) % 8 == 0 or i == len(model.blocks) - 1:
            print(f"  ✓ Blocks {max(0, i - 7):02d}-{i:02d} done")

    print(f"  Total blocks: {total_block_size:.2f} MB")

    # 3. Export Merger
    print("\n[34/34] Exporting merger...")
    merger_wrapped = _MergerWrapper(model.merger, grid_h, grid_w)
    torch.onnx.export(
        merger_wrapped,
        dummy_patches,
        str(output_dir / "merger.onnx"),
        opset_version=opset_version,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
    )
    merger_size = (output_dir / "merger.onnx").stat().st_size / 1024 / 1024
    print(f"  ✓ merger.onnx: {merger_size:.2f} MB")

    # Summary
    patch_size = (output_dir / "patch_embed.onnx").stat().st_size / 1024 / 1024
    total_size = patch_size + total_block_size + merger_size

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print("\n✓ Exported 34 components")
    print(f"Total size: {total_size:.2f} MB (~{total_size / 1024:.2f} GB)")
    print(f"Location: {output_dir}")
    print("\nUsage:")
    print("  from kornia.models.qwen25 import Qwen2VLVisionEncoderONNX")
    print(f'  encoder = Qwen2VLVisionEncoderONNX("{output_dir}")')
    print("  features = encoder(image)")


def main() -> None:
    """CLI entry point for exporting ONNX components."""
    parser = argparse.ArgumentParser(description="Export Qwen2.5-VL Vision Encoder to ONNX components")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_components",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=448,
        help="Input image height (must be divisible by 28)",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=448,
        help="Input image width (must be divisible by 28)",
    )
    args = parser.parse_args()

    export_vision_encoder_components(
        output_dir=args.output_dir,
        model_id=args.model_id,
        opset_version=args.opset,
        input_size=(args.input_height, args.input_width),
    )


if __name__ == "__main__":
    main()

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
    # Dynamic export (default - supports any input size)
    python -m kornia.models.qwen25.export_onnx_components --output-dir ./onnx_components

    # Fixed size export (optimized for specific size, use --fixed)
    python -m kornia.models.qwen25.export_onnx_components --output-dir ./onnx_fixed --fixed

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


class _MergerWrapperFixed(nn.Module):
    """Wrapper with fixed grid dimensions for ONNX export."""

    def __init__(self, merger: nn.Module, grid_h: int, grid_w: int):
        super().__init__()
        self.merger = merger
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merger(x, self.grid_h, self.grid_w)


class _MergerWrapperDynamic(nn.Module):
    """ONNX-compatible merger wrapper with grid dimensions as inputs.

    Accepts grid_h and grid_w as 0-dim tensor inputs, enabling dynamic
    aspect ratio support. Uses torch operations that ONNX can trace.
    """

    def __init__(self, merger: nn.Module):
        super().__init__()
        self.ln_q = merger.ln_q
        self.mlp = merger.mlp

    def forward(self, x: torch.Tensor, grid_h: torch.Tensor, grid_w: torch.Tensor) -> torch.Tensor:
        """Forward with grid dimensions as inputs.

        Args:
            x: Patches tensor (B, seq_len, 1280)
            grid_h: Grid height as 0-dim tensor
            grid_w: Grid width as 0-dim tensor

        Returns:
            Features tensor (B, seq_len/4, 2048)
        """
        x = self.ln_q(x)

        B = x.shape[0]
        C = x.shape[2]

        # Convert grid dims for reshape - use view with computed total
        gh = grid_h.reshape(())  # Ensure 0-dim
        gw = grid_w.reshape(())

        # Reshape to grid: (B, seq_len, C) -> (B, grid_h, grid_w, C)
        x = x.view(B, gh, gw, C)

        # Merge 2x2: (B, H, W, C) -> (B, H/2, 2, W/2, 2, C)
        H_half = gh // 2
        W_half = gw // 2
        x = x.view(B, H_half, 2, W_half, 2, C)

        # Permute to group 2x2
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        # Flatten 2x2: (B, H/2, W/2, 4*C)
        x = x.view(B, H_half * W_half, C * 4)

        # MLP
        x = self.mlp(x)

        return x


def export_vision_encoder_components(
    output_dir: Union[str, Path],
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    opset_version: int = 17,
    input_size: tuple = (448, 448),
    dynamic: bool = True,
) -> None:
    """Export Qwen2.5-VL Vision Encoder as separate ONNX components.

    Args:
        output_dir: Directory to save ONNX components.
        model_id: HuggingFace model ID for pretrained weights.
        opset_version: ONNX opset version.
        input_size: Input image size (H, W) for example input. Must be divisible by 28.
        dynamic: If True, export with dynamic axes for variable input sizes.

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

    export_mode = "DYNAMIC" if dynamic else f"FIXED ({h}x{w})"
    print(f"\nExport mode: {export_mode}")
    print(f"Exporting to: {output_dir}")
    print("=" * 60)

    # Dynamic axes configuration
    if dynamic:
        patch_embed_dynamic = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "patches": {0: "batch", 1: "seq_len"},
        }
        block_dynamic = {
            "input": {0: "batch", 1: "seq_len"},
            "output": {0: "batch", 1: "seq_len"},
        }
    else:
        patch_embed_dynamic = None
        block_dynamic = None

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
        dynamic_axes=patch_embed_dynamic,
    )
    size = (output_dir / "patch_embed.onnx").stat().st_size / 1024 / 1024
    print(f"  patch_embed.onnx: {size:.2f} MB")

    # 2. Export Transformer Blocks
    print("\n[2-33/34] Exporting transformer blocks...")
    seq_len = grid_h * grid_w
    embed_dim = 1280  # Qwen2.5-VL vision encoder embedding dimension
    dummy_patches = torch.randn(1, seq_len, embed_dim)

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
            dynamic_axes=block_dynamic,
        )
        size = (output_dir / f"block_{i:02d}.onnx").stat().st_size / 1024 / 1024
        total_block_size += size

        if (i + 1) % 8 == 0 or i == len(model.blocks) - 1:
            print(f"  Blocks {max(0, i - 7):02d}-{i:02d} done")

    print(f"  Total blocks: {total_block_size:.2f} MB")

    # 3. Export Merger
    print("\n[34/34] Exporting merger...")
    if dynamic:
        # Use dynamic wrapper that accepts grid dims as inputs
        merger_wrapped = _MergerWrapperDynamic(model.merger)
        # Create dummy grid dimensions as scalar tensors
        dummy_grid_h = torch.tensor(grid_h, dtype=torch.int64)
        dummy_grid_w = torch.tensor(grid_w, dtype=torch.int64)

        torch.onnx.export(
            merger_wrapped,
            (dummy_patches, dummy_grid_h, dummy_grid_w),  # 3 inputs
            str(output_dir / "merger.onnx"),
            opset_version=opset_version,
            export_params=True,
            input_names=["input", "grid_h", "grid_w"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 1: "seq_len"},
                "output": {0: "batch", 1: "out_seq_len"},
                # grid_h and grid_w are scalars, no dynamic axes
            },
        )
    else:
        # Use fixed wrapper
        merger_wrapped = _MergerWrapperFixed(model.merger, grid_h, grid_w)
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
    print(f"  merger.onnx: {merger_size:.2f} MB")

    # Summary
    patch_size = (output_dir / "patch_embed.onnx").stat().st_size / 1024 / 1024
    total_size = patch_size + total_block_size + merger_size

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nExported 34 components ({export_mode})")
    print(f"Total size: {total_size:.2f} MB (~{total_size / 1024:.2f} GB)")
    print(f"Location: {output_dir}")

    if dynamic:
        print("\nDynamic export: Supports ANY input size!")
        print("   Note: Input H and W must be divisible by 28.")
    else:
        print(f"\nFixed export: Only supports {h}x{w} input size.")
        print("   Use --dynamic flag for variable input sizes.")

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
        help="Input image height (used as example for dynamic export)",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=448,
        help="Input image width (used as example for dynamic export)",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Export with fixed input size (optimized, but only supports one size)",
    )
    args = parser.parse_args()

    export_vision_encoder_components(
        output_dir=args.output_dir,
        model_id=args.model_id,
        opset_version=args.opset,
        input_size=(args.input_height, args.input_width),
        dynamic=not args.fixed,
    )


if __name__ == "__main__":
    main()

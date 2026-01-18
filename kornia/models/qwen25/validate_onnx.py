"""Validate ONNX export against PyTorch using random images (simpler)"""
import os
from pathlib import Path
import numpy as np
import torch
from kornia.models.qwen25 import Qwen2VLVisionEncoderONNX, Qwen2VLVisionTransformer

# Set HF cache
cache_dir = Path(__file__).parent / "onnx_model" / "hf_cache"
os.environ["HF_HOME"] = str(cache_dir)


def main() -> None:
    """Run ONNX vs PyTorch validation across multiple input sizes."""
    print("=" * 60)
    print("ONNX vs PyTorch Validation")
    print("=" * 60)

    # Load PyTorch model
    print("\n1. Loading PyTorch model...")
    pytorch_model = Qwen2VLVisionTransformer.from_pretrained()
    pytorch_model.eval().cpu()
    print("   PyTorch model loaded")

    # Load ONNX model
    print("\n2. Loading ONNX model...")
    onnx_encoder = Qwen2VLVisionEncoderONNX("onnx_components")
    print("   ONNX encoder loaded")

    # Test with different sizes (simulating different video frames)
    print("\n3. Comparing outputs for multiple sizes:")
    print("-" * 60)

    test_sizes = [
        (224, 224),
        (336, 224),
        (224, 336),
        (280, 280),
    ]

    results: list[float] = []
    for h, w in test_sizes:
        # Create random input (simulating normalized image)
        np.random.seed(42)  # For reproducibility
        img = np.random.randn(1, 3, h, w).astype(np.float32) * 0.5

        # PyTorch inference
        with torch.no_grad():
            img_torch = torch.from_numpy(img).float()
            pytorch_out = pytorch_model(img_torch)
            pytorch_out = pytorch_out.numpy()

        # ONNX inference
        onnx_out = onnx_encoder(img)

        # Compare
        max_diff = np.abs(pytorch_out - onnx_out).max()
        mean_diff = np.abs(pytorch_out - onnx_out).mean()

        status = "PASS" if max_diff < 0.001 else "WARN" if max_diff < 0.01 else "FAIL"
        print(f"{h}×{w}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f} {status}")
        results.append(max_diff)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_max_diff = np.mean(results)
    if all(r < 0.001 for r in results):
        print("\nVALIDATION PASSED!")
        print(f"   Max difference: {max(results):.6f}")
        print(f"   Avg difference: {avg_max_diff:.6f}")
        print("\n   ONNX export is numerically equivalent to PyTorch!")
    else:
        print(f"\nSome differences detected. Max: {max(results):.6f}")


if __name__ == "__main__":
    main()

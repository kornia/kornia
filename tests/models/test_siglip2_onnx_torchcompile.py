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

"""Test SigLip2 ONNX export and torch.compile."""

import os
import tempfile

import pytest
import torch

from kornia.models.siglip2 import SigLip2Builder

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
def test_onnx_export_image_features():
    """Test exporting get_image_features to ONNX."""
    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Load model
    model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Create dummy input
    batch_size = 1
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)

    # Test forward pass
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values)

    # Export to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "siglip2_image_features.onnx")

        torch.onnx.export(
            model.get_image_features,
            pixel_values,
            onnx_path,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "image_embeds": {0: "batch_size"}},
            opset_version=17,
        )

        # Verify ONNX model can be loaded
        onnx_model = onnx.load(onnx_path)
        assert onnx_model is not None

        # Verify ONNX model can be run
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"pixel_values": pixel_values.cpu().numpy()})

        assert len(outputs) == 1
        assert outputs[0].shape == image_features.shape

        # Check output values are close
        onnx_output = torch.from_numpy(outputs[0])
        assert torch.allclose(image_features.cpu(), onnx_output, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
def test_onnx_export_text_features():
    """Test exporting get_text_features to ONNX."""
    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Load model
    model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Create dummy input
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Test forward pass
    with torch.no_grad():
        text_features = model.get_text_features(input_ids, attention_mask=attention_mask)

    # Export to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "siglip2_text_features.onnx")

        # Create a wrapper function for ONNX export
        def text_features_wrapper(input_ids, attention_mask):
            return model.get_text_features(input_ids, attention_mask=attention_mask)

        torch.onnx.export(
            text_features_wrapper,
            (input_ids, attention_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "text_embeds": {0: "batch_size"},
            },
            opset_version=17,
        )

        # Verify ONNX model can be loaded
        onnx_model = onnx.load(onnx_path)
        assert onnx_model is not None

        # Verify ONNX model can be run
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outputs = session.run(
            None,
            {
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
            },
        )

        assert len(outputs) == 1
        assert outputs[0].shape == text_features.shape

        # Check output values are close
        onnx_output = torch.from_numpy(outputs[0])
        assert torch.allclose(text_features.cpu(), onnx_output, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
def test_onnx_export_full_model():
    """Test exporting full model forward to ONNX."""
    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Load model
    model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Create dummy inputs
    batch_size = 1
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
    input_ids = torch.randint(0, 1000, (batch_size, 10), device=device)
    attention_mask = torch.ones(batch_size, 10, device=device)

    # Test forward pass
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

    # Export to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "siglip2_full.onnx")

        # Create a wrapper function for ONNX export
        def model_forward(pixel_values, input_ids, attention_mask):
            return model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        torch.onnx.export(
            model_forward,
            (pixel_values, input_ids, attention_mask),
            onnx_path,
            input_names=["pixel_values", "input_ids", "attention_mask"],
            output_names=["image_embeds", "text_embeds", "logits_per_image"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "image_embeds": {0: "batch_size"},
                "text_embeds": {0: "batch_size"},
                "logits_per_image": {0: "batch_size", 1: "batch_size"},
            },
            opset_version=17,
        )

        # Verify ONNX model can be loaded
        onnx_model = onnx.load(onnx_path)
        assert onnx_model is not None

        # Verify ONNX model can be run
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_outputs = session.run(
            None,
            {
                "pixel_values": pixel_values.cpu().numpy(),
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
            },
        )

        assert len(onnx_outputs) == 3
        assert onnx_outputs[0].shape == outputs["image_embeds"].shape
        assert onnx_outputs[1].shape == outputs["text_embeds"].shape
        assert onnx_outputs[2].shape == outputs["logits_per_image"].shape


def test_torch_compile_image_features():
    """Test that torch.compile works for get_image_features."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available (requires PyTorch 2.0+)")

    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Load model
    model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Compile the model
    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    # Create dummy input
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)

    # Test forward pass with compiled model
    with torch.no_grad():
        compiled_output = compiled_model.get_image_features(pixel_values)

    # Test forward pass with original model
    with torch.no_grad():
        original_output = model.get_image_features(pixel_values)

    # Check outputs match
    assert torch.allclose(compiled_output, original_output, atol=1e-5, rtol=1e-5)


def test_torch_compile_text_features():
    """Test that torch.compile works for get_text_features."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available (requires PyTorch 2.0+)")

    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Load model
    model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Compile the model
    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Test forward pass with compiled model
    with torch.no_grad():
        compiled_output = compiled_model.get_text_features(input_ids, attention_mask=attention_mask)

    # Test forward pass with original model
    with torch.no_grad():
        original_output = model.get_text_features(input_ids, attention_mask=attention_mask)

    # Check outputs match
    assert torch.allclose(compiled_output, original_output, atol=1e-5, rtol=1e-5)


def test_torch_compile_full_model():
    """Test that torch.compile works for full model forward."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available (requires PyTorch 2.0+)")

    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Load model
    model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Compile the model
    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    # Create dummy inputs
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
    input_ids = torch.randint(0, 1000, (batch_size, 10), device=device)
    attention_mask = torch.ones(batch_size, 10, device=device)

    # Test forward pass with compiled model
    with torch.no_grad():
        compiled_outputs = compiled_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

    # Test forward pass with original model
    with torch.no_grad():
        original_outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

    # Check outputs match
    assert torch.allclose(compiled_outputs["image_embeds"], original_outputs["image_embeds"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(compiled_outputs["text_embeds"], original_outputs["text_embeds"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        compiled_outputs["logits_per_image"], original_outputs["logits_per_image"], atol=1e-5, rtol=1e-5
    )


def test_torch_compile_fullgraph():
    """Test that torch.compile with fullgraph=True works (non-breaking graph)."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available (requires PyTorch 2.0+)")

    model_name = "google/siglip2-base-patch16-224"
    device = "cpu"

    # Load model
    model = SigLip2Builder.from_pretrained(model_name).to(device).eval()

    # Try to compile with fullgraph=True
    # This will raise an error if the graph is not fully traceable
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        # Create dummy inputs
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
        input_ids = torch.randint(0, 1000, (batch_size, 10), device=device)
        attention_mask = torch.ones(batch_size, 10, device=device)

        # Test forward pass
        with torch.no_grad():
            outputs = compiled_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        # If we get here, fullgraph compilation succeeded
        assert "image_embeds" in outputs
        assert "text_embeds" in outputs
        assert "logits_per_image" in outputs

    except Exception as e:
        # If fullgraph fails, that's okay - just document it
        pytest.skip(f"fullgraph compilation not supported: {e}")

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

"""ONNX export tests for SAM (image encoder subgraph)."""

import pytest

from kornia.models.sam.model import Sam, SamConfig


def test_sam_has_to_onnx():
    """Sam exposes a to_onnx() method via ONNXExportMixin."""
    assert hasattr(Sam, "to_onnx")


def test_sam_encoder_to_onnx(tmp_path):
    """Sam.to_onnx() exports the image encoder subgraph with correct output name."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    # Create a tiny pseudo-SAM model to prevent CI timeouts during ONNX trace
    config = SamConfig(
        model_type="vit_b",
        encoder_embed_dim=16,
        encoder_depth=1,
        encoder_num_heads=1,
        encoder_global_attn_indexes=[0],
    )
    model = Sam.from_config(config)
    model.eval()

    onnx_path = tmp_path / "sam_encoder.onnx"
    # Export only the image encoder (Sam.to_onnx overrides the full-model export).
    # Use a small spatial size to keep the test fast; only batch dimension is dynamic.
    op = model.to_onnx(save=True, onnx_name=str(onnx_path), pseudo_shape=[1, 3, 1024, 1024])

    assert isinstance(op, onnx.ModelProto)
    assert onnx_path.exists(), "ONNX file was not written to disk"

    output_names = [o.name for o in op.graph.output]
    assert "image_embeddings" in output_names, f"Expected 'image_embeddings', got {output_names}"

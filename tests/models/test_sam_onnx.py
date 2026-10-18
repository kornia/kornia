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
import torch

from kornia.core._compat import torch_version_ge, torch_version_lt
from kornia.models.sam.architecture.common import LayerNorm
from kornia.models.sam.architecture.image_encoder import ImageEncoderViT
from kornia.models.sam.model import Sam, SamConfig
from kornia.models.tiny_vit import TinyViT

from testing.base import assert_close

# Every test in this module must stay device-free: the mark deselects the whole file on non-CPU devices.
pytestmark = pytest.mark.device_agnostic


def test_sam_has_to_onnx():
    """Sam exposes a to_onnx() method via ONNXExportMixin."""
    assert hasattr(Sam, "to_onnx")


@pytest.mark.timeout(120)
@pytest.mark.xfail(
    torch_version_ge(2, 14) and torch_version_lt(2, 15, 0),
    reason=(
        "torch 2.14's dynamo ONNX exporter cannot translate aten.mul.Scalar for this graph in the CPU "
        "build (the CUDA build exports it): DispatchError 'No decompositions registered for the "
        "real-valued input'; not yet reported upstream"
    ),
    raises=torch.onnx.OnnxExporterError,
    # Strict on the CPU builds, where an upstream fix inside 2.14.x must announce itself as an XPASS failure rather
    # than sit silently until the 2.15 bound expires. A CUDA build of 2.14.0 exports this graph (2.14.0+cu130 on
    # Linux), so there the marker is not strict.
    strict=torch.version.cuda is None,
)
def test_sam_encoder_to_onnx(tmp_path):
    """Sam.to_onnx() exports the image encoder subgraph with correct output name."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    # Create a tiny pseudo-SAM model to prevent CI timeouts during ONNX trace
    config = SamConfig(
        model_type=None,
        encoder_embed_dim=16,
        encoder_depth=1,
        encoder_num_heads=1,
        encoder_global_attn_indexes=[0],
    )
    model = Sam.from_config(config)
    model.eval()
    # The tiny encoder, not ViT-B (89.7M encoder parameters): 680,128, mostly the neck's 3 x 3 convolution.
    assert sum(p.numel() for p in model.image_encoder.parameters()) < 1e6

    onnx_path = tmp_path / "sam_encoder.onnx"
    # Export only the image encoder (Sam.to_onnx overrides the full-model export).
    # Use a small spatial size to keep the test fast; only batch dimension is dynamic.
    op = model.to_onnx(save=True, onnx_name=str(onnx_path), pseudo_shape=[1, 3, 1024, 1024])

    assert isinstance(op, onnx.ModelProto)
    assert onnx_path.exists(), "ONNX file was not written to disk"

    output_names = [o.name for o in op.graph.output]
    assert "image_embeddings" in output_names, f"Expected 'image_embeddings', got {output_names}"


def _dims(value_info):
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


def _check_onnx_against_eager(op, encoder, batches, size):
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(0)
    for batch in batches:
        images = torch.rand(batch, 3, size, size, generator=generator)
        (embeddings,) = session.run(None, {session.get_inputs()[0].name: images.numpy()})
        with torch.no_grad():
            expected = encoder(images)
        assert_close(torch.from_numpy(embeddings), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.timeout(120)
def test_vit_encoder_legacy_onnx_matches_eager_at_every_batch():
    """On the legacy exporter, a ViT image encoder runs at batches other than the traced one."""
    pytest.importorskip("onnx")
    torch.manual_seed(0)
    model = Sam.from_config(
        SamConfig(
            model_type=None, encoder_embed_dim=16, encoder_depth=1, encoder_num_heads=1, encoder_global_attn_indexes=[0]
        )
    )
    # A 64 px encoder with a windowed and a global block: the 1024 px one holds a 4096 x 4096 attention per image.
    model.image_encoder = ImageEncoderViT(
        img_size=64,
        embed_dim=16,
        depth=2,
        num_heads=1,
        norm_layer=LayerNorm,
        use_rel_pos=True,
        global_attn_indexes=(1,),
        window_size=14,
        out_chans=256,
    )
    model.eval()

    op = model.to_onnx(save=False, pseudo_shape=[1, 3, 64, 64], dynamo=False)

    assert _dims(op.graph.input[0]) == ["batch", 3, 64, 64]
    _check_onnx_against_eager(op, model.image_encoder, (1, 2, 3), 64)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("export_kwargs", [{}, {"dynamo": False}], ids=["default", "legacy"])
def test_tiny_vit_encoder_onnx_runs_at_the_batches_it_declares(export_kwargs):
    """A MobileSAM (TinyViT) encoder: a dynamic batch on the dynamo exporter, the pseudo batch on the legacy one.

    The legacy TorchScript exporter (the default before torch 2.9) keeps the traced batch in TinyViT's reshapes, so
    its graph is traced at the pseudo shape's batch and declares that batch, rather than a dynamic batch that runs at
    the traced size only.
    """
    pytest.importorskip("onnx")
    legacy = not export_kwargs.get("dynamo", torch_version_ge(2, 9, 0))
    if not legacy:
        pytest.importorskip("onnxscript")
    torch.manual_seed(0)
    model = Sam.from_name("mobile_sam")
    # The MobileSAM encoder layout at 64 px instead of 1024 px.
    model.image_encoder = TinyViT.from_config("5m", img_size=64, mobile_sam=True)
    model.eval()

    op = model.to_onnx(save=False, pseudo_shape=[1, 3, 64, 64], **export_kwargs)

    if legacy:
        _check_onnx_against_eager(op, model.image_encoder, (1,), 64)
        assert _dims(op.graph.input[0]) == [1, 3, 64, 64]
        assert _dims(op.graph.output[0])[0] == 1
    else:
        _check_onnx_against_eager(op, model.image_encoder, (1, 2, 3), 64)
        assert isinstance(_dims(op.graph.input[0])[0], str)

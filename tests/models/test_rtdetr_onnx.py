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

"""ONNX export tests for RT-DETR."""

import pytest

from kornia.models.rt_detr.model import RTDETR, RTDETRConfig


def test_rtdetr_to_onnx(tmp_path):
    """RT-DETR exports to ONNX with correct dual output names."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    config = RTDETRConfig.from_name("rtdetr_r18vd", num_classes=10)
    model = RTDETR.from_config(config)
    model.eval()

    onnx_path = tmp_path / "rtdetr.onnx"
    op = model.to_onnx(save=True, onnx_name=str(onnx_path), pseudo_shape=[1, 3, 640, 640])

    assert isinstance(op, onnx.ModelProto)
    assert onnx_path.exists(), "ONNX file was not written to disk"

    # Verify the output nodes are named correctly
    output_names = [o.name for o in op.graph.output]
    assert "pred_logits" in output_names, f"Expected 'pred_logits' in outputs, got {output_names}"
    assert "pred_boxes" in output_names, f"Expected 'pred_boxes' in outputs, got {output_names}"

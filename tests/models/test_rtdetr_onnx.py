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
import torch

from kornia.models.rt_detr.model import RTDETR, RTDETRConfig
from kornia.models.rt_detr.post_processor import DETRPostProcessor

from testing.base import assert_close

# Every test in this module must stay device-free: the mark deselects the whole file on non-CPU devices.
pytestmark = pytest.mark.device_agnostic


@pytest.mark.timeout(120)
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


def _post_processor() -> DETRPostProcessor:
    return DETRPostProcessor(confidence_threshold=0.0, num_classes=2, num_top_queries=1, confidence_filtering=False)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("sizes_dtype", [torch.float32, torch.int64])
def test_postprocessor_onnx_per_image_sizes(tmp_path, sizes_dtype):
    """The exported post-processor scales each image of a batch by its own size, as eager does."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    logits = torch.tensor([[[5.0, -5.0]]] * 2)
    # one normalised (cx, cy, w, h) box per image
    boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]] * 2)
    # (H, W) of each image
    sizes = torch.tensor([[100, 200], [400, 800]], dtype=sizes_dtype)

    onnx_path = tmp_path / "detr_post_processor.onnx"
    torch.onnx.export(
        _post_processor(),
        (logits, boxes, sizes),
        str(onnx_path),
        input_names=["logits", "boxes", "sizes"],
        output_names=["detections"],
        dynamic_axes={"logits": {0: "N"}, "boxes": {0: "N"}, "sizes": {0: "N"}, "detections": {0: "N"}},
        opset_version=18,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    (detections,) = session.run(None, {"logits": logits.numpy(), "boxes": boxes.numpy(), "sizes": sizes.numpy()})
    # xywh, x = (cx - w / 2) * W, y = (cy - h / 2) * H, w * W, h * H, with each image's own (H, W)
    expected = torch.tensor([[[75.0, 37.5, 50.0, 25.0]], [[300.0, 150.0, 200.0, 100.0]]])
    assert_close(torch.from_numpy(detections)[..., 2:], expected)
    assert_close(torch.from_numpy(detections), _post_processor()(logits, boxes, sizes))

    # the batch axis is dynamic: image 1 alone
    (detections,) = session.run(
        None, {"logits": logits[1:].numpy(), "boxes": boxes[1:].numpy(), "sizes": sizes[1:].numpy()}
    )
    assert_close(torch.from_numpy(detections)[..., 2:], expected[1:])


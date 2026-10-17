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
from torch import nn

from kornia.contrib.object_detection import ObjectDetector
from kornia.models.processors import ResizePreProcessor
from kornia.models.rt_detr.model import RTDETR, RTDETRConfig
from kornia.models.rt_detr.post_processor import DETRPostProcessor

from testing.base import DYNAMO_UNAVAILABLE_REASON, assert_close, dynamo_is_available

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


class _FixedDetections(nn.Module):
    """Stand-in for RT-DETR: one fixed query per image, a box in normalised (cx, cy, w, h)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor([[[5.0, -5.0]]]))
        self.register_buffer("boxes", torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = images.shape[0]
        return self.logits.expand(batch, -1, -1), self.boxes.expand(batch, -1, -1)


@pytest.mark.timeout(120)
def test_resize_preprocessor_onnx_dynamic_sizes():
    """With a dynamic H and W, the exported detector reads each input's size instead of the trace's."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    detector = ObjectDetector(_FixedDetections(), ResizePreProcessor(32, 32), _post_processor())
    exported = detector.to_onnx(image_size=None, include_pre_and_post_processor=True, save=False)
    session = ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])

    generator = torch.Generator().manual_seed(0)
    for height, width in [(77, 128), (96, 100)]:
        images = torch.rand(1, 3, height, width, generator=generator)
        (detections,) = session.run(None, {"input": images.numpy()})
        detections = torch.from_numpy(detections)

        assert_close(detections, detector(images))
        # xywh of the normalised box (0.5, 0.5, 0.25, 0.25) in this image's pixels
        expected = torch.tensor([[[0.375 * width, 0.375 * height, 0.25 * width, 0.25 * height]]])
        assert_close(detections[..., 2:], expected)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "dynamo",
    [False, pytest.param(True, marks=pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON))],
)
def test_resnet_d_onnx_odd_sizes(tmp_path, dynamo):
    """At an odd feature map the exported ResNet-D matches eager in old and new ONNX Runtime alike.

    Runtimes disagree on a partial ``ceil_mode`` window of an ``AveragePool`` with ``count_include_pad=1``: ONNX
    Runtime up to 1.28 divides it by the whole 2 x 2 window, 1.29 and later by the pixels inside it, as eager does.
    With ``count_include_pad=0`` both divide by the pixels inside it. The graph is checked as well as the runtime, as
    the runtime that runs this test may be one that hides the difference.
    """
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from kornia.models.rt_detr.architecture.resnet_d import ResNetD

    class _Features(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = ResNetD.from_config(18)

        def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tuple(self.backbone(images))

    model = _Features().eval()
    # the stem takes 92 x 96 to 23 x 24: the first stride-2 shortcut pool has a partial window at the bottom
    images = torch.rand(1, 3, 92, 96, generator=torch.Generator().manual_seed(0))
    onnx_path = tmp_path / "resnet_d.onnx"
    if dynamo:
        torch.onnx.export(model, (images,), dynamo=True, opset_version=18).save(str(onnx_path))
    else:
        torch.onnx.export(model, (images,), str(onnx_path), dynamo=False, opset_version=17, input_names=["images"])

    pools = [node for node in onnx.load(str(onnx_path)).graph.node if node.op_type == "AveragePool"]
    assert len(pools) == 3
    for node in pools:
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert attributes.get("count_include_pad", 0) == 0

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: images.numpy()})
    with torch.no_grad():
        expected = model(images)
    for actual, fmap in zip(outputs, expected):
        assert_close(torch.from_numpy(actual), fmap, atol=1e-4, rtol=1e-4)

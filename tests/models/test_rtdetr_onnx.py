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

import importlib

import pytest
import torch
from torch import nn

from kornia.contrib.object_detection import ObjectDetector
from kornia.core._compat import torch_version_ge, torch_version_lt
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


# ``to_onnx`` without ``dynamo=``: the dynamo exporter from torch 2.9, the TorchScript exporter before it.
_EXPORTERS = [pytest.param({}, id="default"), pytest.param({"dynamo": False}, id="legacy")]


def _importorskip_exporter(export_kwargs: dict) -> None:
    pytest.importorskip("onnx")
    if export_kwargs.get("dynamo", torch_version_ge(2, 9, 0)):
        pytest.importorskip("onnxscript")


def _small_rtdetr() -> RTDETR:
    """A randomly initialised RT-DETR-R18 with 10 queries, which runs on 96 x 128 images (252 anchors)."""
    torch.manual_seed(0)
    return RTDETR.from_config(RTDETRConfig("resnet18d", num_classes=4, head_num_queries=10)).eval()


def _images(shapes: list[tuple[int, int, int]]) -> list[torch.Tensor]:
    # Seed 38 keeps the queries of ``_small_rtdetr`` apart on these shapes, see ``_eager_with_distinct_queries``.
    generator = torch.Generator().manual_seed(38)
    return [torch.rand(batch, 3, height, width, generator=generator) for batch, height, width in shapes]


def _eager_with_distinct_queries(model: RTDETR, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's eager ``(logits, boxes)``, after checking that its query selection has no near tie.

    RT-DETR keeps the queries with the highest encoder scores, in score order. Where two scores nearly tie, a
    difference of 1e-6 between ONNX Runtime and eager swaps two queries, and the outputs then differ by whole boxes
    although the export is right. A random init on random images ties easily, so the inputs are drawn with a seed
    whose scores stay at least 1e-4 apart; a change to the model's initialisation that breaks this fails here.
    """
    scores: list[torch.Tensor] = []
    handle = model.decoder.enc_score_head.register_forward_hook(lambda m, i, out: scores.append(out.max(-1).values))
    try:
        with torch.no_grad():
            logits, boxes = model(images)
    finally:
        handle.remove()
    kept = scores[0].topk(model.decoder.num_queries + 1, dim=1).values
    gap = (kept[:, :-1] - kept[:, 1:]).min().item()
    assert gap > 1e-4, f"near tie between kept encoder scores ({gap:.1e}): pick another seed in _images"
    return logits, boxes


def _check_onnx_against_eager(op, model: RTDETR, images: torch.Tensor) -> None:
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(op.SerializeToString(), providers=["CPUExecutionProvider"])
    logits, boxes = session.run(None, {session.get_inputs()[0].name: images.numpy()})
    expected_logits, expected_boxes = _eager_with_distinct_queries(model, images)
    assert_close(torch.from_numpy(logits), expected_logits, rtol=1e-4, atol=1e-4)
    assert_close(torch.from_numpy(boxes), expected_boxes, rtol=1e-4, atol=1e-4)


def _dims(value_info) -> list:
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("export_kwargs", _EXPORTERS)
def test_rtdetr_onnx_matches_eager_at_other_batches(export_kwargs):
    """``RTDETR.to_onnx`` keeps the batch dynamic: batches 1 and 3 match eager on a graph traced at 2, H != W."""
    _importorskip_exporter(export_kwargs)
    model = _small_rtdetr()
    op = model.to_onnx(save=False, pseudo_shape=[1, 3, 96, 128], **export_kwargs)

    assert [o.name for o in op.graph.output] == ["pred_logits", "pred_boxes"]
    for images in _images([(1, 96, 128), (3, 96, 128)]):
        _check_onnx_against_eager(op, model, images)


@pytest.mark.timeout(120)
def test_rtdetr_legacy_onnx_ties_the_callers_output_names_to_the_input_batch():
    """``RTDETR.to_onnx`` gives a caller's ``output_names`` the input's dynamic batch, as the declared names get it."""
    _importorskip_exporter({"dynamo": False})
    op = _small_rtdetr().to_onnx(save=False, pseudo_shape=[1, 3, 96, 128], dynamo=False, output_names=["a", "b"])

    assert [o.name for o in op.graph.output] == ["a", "b"]
    assert [_dims(output)[0] for output in op.graph.output] == ["batch", "batch"]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("export_kwargs", _EXPORTERS)
def test_detector_model_only_onnx_names_both_outputs(export_kwargs):
    """Without pre- and post-processing, the detector exports RT-DETR's two outputs by name, at a dynamic size."""
    _importorskip_exporter(export_kwargs)
    model = _small_rtdetr()
    detector = ObjectDetector(model, ResizePreProcessor(96, 96), _post_processor())
    op = detector.to_onnx(image_size=None, include_pre_and_post_processor=False, save=False, **export_kwargs)

    assert [o.name for o in op.graph.output] == ["pred_logits", "pred_boxes"]
    # each output's batch is the input's dynamic batch
    batch = _dims(op.graph.input[0])[0]
    assert isinstance(batch, str)
    assert [_dims(output)[0] for output in op.graph.output] == [batch, batch]
    # two sizes, multiples of 32 with H != W, at batches other than the traced 2
    for images in _images([(1, 96, 128), (3, 96, 128), (1, 128, 96), (3, 128, 96)]):
        _check_onnx_against_eager(op, model, images)


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
    """With a dynamic batch, H and W, the exported detector reads each input's size instead of the trace's."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    detector = ObjectDetector(_FixedDetections(), ResizePreProcessor(32, 32), _post_processor())
    exported = detector.to_onnx(image_size=None, include_pre_and_post_processor=True, save=False)
    session = ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])

    generator = torch.Generator().manual_seed(0)
    # A batch of 2 came out as one image's detections when the pre-processor looped over the batch.
    for batch, height, width in [(1, 77, 128), (1, 96, 100), (2, 77, 128)]:
        images = torch.rand(batch, 3, height, width, generator=generator)
        (detections,) = session.run(None, {"input": images.numpy()})
        detections = torch.from_numpy(detections)

        assert_close(detections, detector(images))
        # xywh of the normalised box (0.5, 0.5, 0.25, 0.25) in this image's pixels
        expected = torch.tensor([[[0.375 * width, 0.375 * height, 0.25 * width, 0.25 * height]]]).expand(batch, -1, -1)
        assert_close(detections[..., 2:], expected)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("output_names", "expected_names"),
    [(["a", "b"], ["a", "b"]), (None, ["pred_logits", "pred_boxes"])],
    ids=["callers_names", "none"],
)
def test_detector_model_only_legacy_onnx_ties_the_callers_output_names_to_the_input_batch(output_names, expected_names):
    """A caller's ``output_names`` win over ``ONNX_OUTPUT_NAMES``, and each still gets the input's dynamic batch.

    ``output_names=None``, the default of ``torch.onnx.export``, keeps the declared names.
    """
    _importorskip_exporter({"dynamo": False})
    detector = ObjectDetector(_small_rtdetr(), ResizePreProcessor(96, 96), _post_processor())
    op = detector.to_onnx(
        image_size=None, include_pre_and_post_processor=False, save=False, dynamo=False, output_names=output_names
    )

    assert [o.name for o in op.graph.output] == expected_names
    batch = _dims(op.graph.input[0])[0]
    assert [_dims(output)[0] for output in op.graph.output] == [batch, batch]


@pytest.mark.timeout(120)
def test_detector_model_only_onnx_without_declared_names_keeps_output():
    """A model that declares no ``ONNX_OUTPUT_NAMES`` keeps the mixin's single ``output`` name."""
    _importorskip_exporter({})
    detector = ObjectDetector(_FixedDetections(), ResizePreProcessor(32, 32), _post_processor())
    assert not hasattr(detector.model, "ONNX_OUTPUT_NAMES")

    op = detector.to_onnx(image_size=None, include_pre_and_post_processor=False, save=False)

    assert op.graph.output[0].name == "output"


class _ScoredDetections(nn.Module):
    """One query per image, scored by the image's mean so that the export keeps its input; a fixed box."""

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = images.mean((1, 2, 3))
        boxes = images.new_tensor([0.5, 0.5, 0.25, 0.25]).expand(images.shape[0], 1, 4)
        return torch.stack([score, -score], -1)[:, None], boxes


@pytest.mark.timeout(120)
def test_detector_with_per_image_detections_exports_one_image():
    """A post-processor returning a list per image is exported for a batch of one, not for any batch's first image."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    post_processor = DETRPostProcessor(confidence_threshold=0.5, num_classes=2, num_top_queries=1)
    detector = ObjectDetector(_ScoredDetections(), ResizePreProcessor(32, 32), post_processor)
    images = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    assert isinstance(detector(images), list)

    exported = detector.to_onnx(image_size=32, save=False)
    assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1

    session = ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])
    (detections,) = session.run(None, {"input": images.numpy()})
    assert_close(torch.from_numpy(detections), detector(images)[0])


class _PerImageDetections(nn.Module):
    """A caller's own post-processor, not a ``DETRPostProcessor``: one ``(Q, 6)`` tensor per image."""

    def forward(self, logits: torch.Tensor, boxes: torch.Tensor, sizes: torch.Tensor) -> list[torch.Tensor]:
        scores, labels = logits.max(-1, keepdim=True)
        detections = torch.cat([labels.to(boxes), scores, boxes], -1)
        return [detections[i] for i in range(detections.shape[0])]


@pytest.mark.timeout(120)
def test_detector_with_a_custom_per_image_post_processor_exports_one_image():
    """The output decides, not the post-processor's class: any per-image list is exported for a batch of one."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    detector = ObjectDetector(_ScoredDetections(), ResizePreProcessor(32, 32), _PerImageDetections())
    images = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(0))

    exported = detector.to_onnx(image_size=32, save=False)
    assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1

    session = ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])
    (detections,) = session.run(None, {"input": images.numpy()})
    assert_close(torch.from_numpy(detections), detector(images)[0])


class _BatchedDetections(nn.Module):
    """``_PerImageDetections`` as one ``(B, Q, 6)`` tensor, which follows any batch."""

    def forward(self, logits: torch.Tensor, boxes: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        scores, labels = logits.max(-1, keepdim=True)
        return torch.cat([labels.to(boxes), scores, boxes], -1)


_DYNAMIC_BATCH = [
    pytest.param({"dynamo": False, "dynamic_axes": {"input": {0: "batch"}}}, id="torchscript"),
    pytest.param(
        {"dynamo": True, "dynamic_shapes": ({0: "batch"},)},
        # conftest.py deselects IDs containing "dynamo" unless KORNIA_TEST_OPTIMIZER is set.
        id="torch_export",
        marks=[
            pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON),
            pytest.mark.skipif(torch_version_lt(2, 9, 0), reason="to_onnx uses dynamic_shapes from torch 2.9"),
        ],
    ),
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("override", _DYNAMIC_BATCH)
def test_detector_with_per_image_detections_rejects_a_dynamic_batch_override_6111(override):
    """A caller's dynamic batch cannot reach a per-image list export, which would return one image's detections."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    detector = ObjectDetector(_ScoredDetections(), ResizePreProcessor(32, 32), _PerImageDetections())
    with pytest.raises(ValueError, match="confidence_filtering=False"):
        detector.to_onnx(image_size=32, save=False, **override)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("override", _DYNAMIC_BATCH)
def test_detector_with_batched_detections_accepts_a_dynamic_batch_override_6111(override):
    """A detector whose output is one tensor exports the caller's dynamic batch and runs at another batch size."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    detector = ObjectDetector(_ScoredDetections(), ResizePreProcessor(32, 32), _BatchedDetections())
    exported = detector.to_onnx(image_size=32, save=False, **override)
    assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param

    images = torch.rand(3, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    session = ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])
    (detections,) = session.run(None, {"input": images.numpy()})
    assert_close(torch.from_numpy(detections), detector(images))


@pytest.mark.timeout(120)
@pytest.mark.parametrize("override", [pytest.param({}, id="default"), *_DYNAMIC_BATCH])
def test_detector_model_only_onnx_of_a_per_image_detector_keeps_a_dynamic_batch_6111(override):
    """Without the post-processor, the model's tensor outputs follow any batch, whatever the post-processor returns."""
    _importorskip_exporter(override)
    ort = pytest.importorskip("onnxruntime")

    detector = ObjectDetector(_ScoredDetections(), ResizePreProcessor(32, 32), _PerImageDetections())
    exported = detector.to_onnx(image_size=32, include_pre_and_post_processor=False, save=False, **override)
    assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param

    images = torch.rand(3, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    session = ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])
    scores, boxes = session.run(None, {"input": images.numpy()})
    expected_scores, expected_boxes = detector.model(images)
    assert_close(torch.from_numpy(scores), expected_scores)
    assert_close(torch.from_numpy(boxes), expected_boxes)


class _NormalisedDetections(_ScoredDetections):
    """``_ScoredDetections`` behind a batch norm, whose statistics a forward pass in training mode would update."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(3)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(self.norm(images))


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("post_processor", "override"),
    [
        pytest.param(_BatchedDetections, {"dynamo": False}, id="batched_torchscript"),
        pytest.param(_PerImageDetections, {"dynamo": False}, id="per_image_torchscript"),
        pytest.param(
            _PerImageDetections,
            {"dynamo": False, "dynamic_axes": {"input": {0: "batch"}}},
            id="per_image_torchscript_batch_override",
        ),
    ],
)
def test_detector_rechecks_tensor_output_at_the_tracing_batch_6161(post_processor, override, monkeypatch):
    """Only a positive list reading can be reused without checking the actual tracing batch."""
    pytest.importorskip("onnx")
    mixin = importlib.import_module("kornia.core.mixin.onnx")
    probe = mixin._returns_per_image_outputs
    calls = []

    def _counting(module, inputs):
        calls.append(type(module).__name__)
        return probe(module, inputs)

    monkeypatch.setattr(
        importlib.import_module("kornia.contrib.object_detection"), "_returns_per_image_outputs", _counting
    )
    monkeypatch.setattr(mixin, "_returns_per_image_outputs", _counting)
    detector = ObjectDetector(_ScoredDetections(), ResizePreProcessor(32, 32), post_processor())

    if "dynamic_axes" in override:
        with pytest.raises(ValueError, match="confidence_filtering=False"):
            detector.to_onnx(image_size=32, save=False, **override)
    else:
        detector.to_onnx(image_size=32, save=False, **override)

    assert calls == ["ObjectDetector"] * (2 if post_processor is _BatchedDetections else 1)
    assert "_onnx_per_image_output" not in detector.__dict__


@pytest.mark.timeout(120)
@pytest.mark.parametrize("per_image", [True, False], ids=["per_image", "batched"])
def test_detector_output_probe_runs_in_eval_mode_on_the_models_dtype(per_image):
    """The forward pass that reads the output kind keeps statistics and each training flag, in the model's dtype.

    A batched detector exports a dynamic batch, whose check after the export (#6111) probes the tracing input too.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    model = _NormalisedDetections()
    post_processor = _PerImageDetections() if per_image else _BatchedDetections()
    detector = ObjectDetector(model, ResizePreProcessor(32, 32), post_processor).double().train()
    detector.pre_processor.eval()  # a mixed state, which ``train()`` would not restore
    running_mean = model.norm.running_mean.clone()

    exported = detector.to_onnx(image_size=32, save=False)

    batch = exported.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch.dim_value == 1 if per_image else batch.dim_param
    assert torch.equal(model.norm.running_mean, running_mean)
    assert model.norm.num_batches_tracked.item() == 0
    assert detector.training
    assert model.norm.training
    assert not detector.pre_processor.training


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "dynamo",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON),
                # torch 2.5.1 and 2.6.0 fail on the stem's batch norm (2.5.1 on a bare ``BatchNorm2d`` too) with
                # ``_native_batch_norm_legit_no_training``; 2.9.1 passes, and 2.9 makes dynamo the default exporter.
                pytest.mark.skipif(
                    torch_version_lt(2, 9, 0),
                    reason="the dynamo ONNX exporter before torch 2.9 cannot translate BatchNorm",
                ),
            ],
        ),
    ],
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


@pytest.mark.timeout(120)
def test_detector_batch_dependent_output_rejects_dynamic_export_6197(tmp_path):
    pytest.importorskip("onnx")

    class BatchDependentDetections(_BatchedDetections):
        def forward(self, logits, boxes, sizes):
            outputs = super().forward(logits, boxes, sizes)
            return outputs if outputs.shape[0] == 1 else list(outputs.unbind(0))

    detector = ObjectDetector(_ScoredDetections(), ResizePreProcessor(32, 32), BatchDependentDetections())
    path = tmp_path / "detector.onnx"
    with pytest.raises(ValueError, match="returns one output per image"):
        detector.to_onnx(image_size=32, onnx_name=str(path), dynamo=False)
    assert not path.exists()

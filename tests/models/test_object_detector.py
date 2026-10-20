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

import warnings
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import kornia
from kornia.core._compat import torch_version_ge, torch_version_lt
from kornia.models.rt_detr import RTDETR, DETRPostProcessor, RTDETRConfig

from testing.base import BaseTester


class TestObjectDetector(BaseTester):
    @pytest.mark.parametrize("num_detections", [0, 1, 2])
    @pytest.mark.parametrize("as_list", [False, True])
    @pytest.mark.parametrize("method", ["visualize", "save"])
    def test_supplied_detections(self, device, dtype, num_detections, as_list, method, monkeypatch, tmp_path):
        detector = kornia.contrib.object_detection.ObjectDetector(
            torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()
        )
        forward = Mock(side_effect=AssertionError("Supplied detections must not run inference"))
        monkeypatch.setattr(detector, "forward", forward)
        images = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
        detections = torch.tensor([[[0, 0.9, 1, 1, 2, 2], [1, 0.8, 4, 4, 2, 2]]], device=device, dtype=dtype).expand(
            2, -1, -1
        )[:, :num_detections]
        expected = images.clone()
        if num_detections >= 1:
            expected[:, :, 1, 1:4] = expected[:, :, 3, 1:4] = 0
            expected[:, :, 1:4, 1] = expected[:, :, 1:4, 3] = 0
        if num_detections == 2:
            expected[:, :, 4, 4:7] = expected[:, :, 6, 4:7] = 0
            expected[:, :, 4:7, 4] = expected[:, :, 4:7, 6] = 0

        detections = list(detections) if as_list else detections
        if method == "save":
            detector.save(images, detections, directory=str(tmp_path))
            files = sorted(tmp_path.glob("*.png"))
            assert len(files) == len(images)
            output = [
                kornia.io.load_image(str(path), kornia.io.ImageLoadType.UNCHANGED).to(images) / 255 for path in files
            ]
        else:
            output = detector.visualize(images, detections)

        self.assert_close(torch.stack(output), expected)
        self.assert_close(images, torch.ones_like(images))
        forward.assert_not_called()

    def test_visualize_without_detections(self, device, dtype, monkeypatch):
        detector = kornia.contrib.object_detection.ObjectDetector(
            torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()
        )
        images = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
        forward = Mock(return_value=torch.empty(2, 0, 6, device=device, dtype=dtype))
        monkeypatch.setattr(detector, "forward", forward)

        self.assert_close(torch.stack(detector.visualize(images)), images)
        forward.assert_called_once_with(images)

    def test_visualize_keeps_a_border_box_four_sided(self, device):
        detector = kornia.contrib.object_detection.ObjectDetector(
            torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()
        )
        h, w = 6, 9
        # (class, score, x, y, w, h): a detection covering the whole image, so x + w = W and y + h = H
        detection = torch.tensor([[0.0, 0.9, 0.0, 0.0, float(w), float(h)]], device=device)
        out = detector.visualize(torch.ones(1, 3, h, w, device=device), [detection])[0]
        # visualize draws in the default colour, black, on the white image
        assert (out[:, 0] == 0).all()
        assert (out[:, h - 1] == 0).all()
        assert (out[:, :, 0] == 0).all()
        assert (out[:, :, w - 1] == 0).all()

    def test_smoke(self, device, dtype):
        batch_size = 2
        confidence = 0.3
        config = RTDETRConfig("resnet18d", 10, head_num_queries=10)
        model = RTDETR.from_config(config).to(device, dtype).eval()
        pre_processor = kornia.models.processors.ResizePreProcessor(32, 32)
        post_processor = DETRPostProcessor(confidence, num_top_queries=3).to(device, dtype).eval()
        detector = kornia.contrib.object_detection.ObjectDetector(model, pre_processor, post_processor)

        sizes = torch.randint(5, 10, (batch_size, 2)) * 32
        imgs = [torch.randn(3, h, w, device=device, dtype=dtype) for h, w in sizes]
        pre_processor_out = pre_processor(imgs)
        detections = detector(imgs)

        assert pre_processor_out[0].shape[-1] == 32
        assert pre_processor_out[0].shape[-2] == 32
        assert len(detections) == batch_size
        for dets in detections:
            assert dets.shape[1] == 6, dets.shape
            assert torch.all(dets[:, 0].int() == dets[:, 0])
            assert torch.all(dets[:, 1] >= 0.3)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(("confidence_filtering", "expected"), [(False, False), (None, True), (True, True)])
    def test_builder_honours_confidence_filtering_false(self, confidence_filtering, expected):
        detector = kornia.contrib.object_detection.RTDETRDetectorBuilder.build(
            "rtdetr_r18vd", pretrained=False, confidence_filtering=confidence_filtering
        )
        assert detector.post_processor.confidence_filtering is expected

        # Without filtering the post-processor returns the (B, Q, 6) tensor, with it a list of (D, 6) tensors.
        out = detector.post_processor(torch.zeros(1, 300, 80), torch.zeros(1, 300, 4), torch.tensor([[640, 640]]))
        assert isinstance(out, list) is expected

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    @pytest.mark.skipif(torch_version_lt(2, 0, 0), reason="Unsupported ONNX opset version: 16")
    @pytest.mark.xfail(
        torch_version_ge(2, 14) and torch_version_lt(2, 15, 0),
        reason=(
            "torch 2.14's dynamo ONNX exporter cannot translate aten.mul.Scalar for this graph: "
            "DispatchError 'No ONNX function found for aten.mul.Scalar'. Same exporter gap as "
            "test_sam_onnx.py::test_sam_encoder_to_onnx; not yet reported upstream"
        ),
        raises=torch.onnx.OnnxExporterError,
        # Strict, as in test_sam_onnx.py: an upstream fix inside 2.14.x must announce itself as
        # an XPASS rather than sit silent until the 2.15 bound expires. Both variants fail today.
        strict=True,
    )
    @pytest.mark.parametrize("variant", ["resnet50d", "hgnetv2_l"])
    def test_onnx(self, device, dtype, tmp_path: Path, variant: str):
        config = RTDETRConfig(variant, 1)
        model = RTDETR.from_config(config).to(device=device, dtype=dtype).eval()
        pre_processor = kornia.models.processors.ResizePreProcessor(640, 640)
        post_processor = DETRPostProcessor(0.3, num_top_queries=3)
        detector = kornia.contrib.object_detection.ObjectDetector(model, pre_processor, post_processor)

        data = torch.rand(1, 3, 400, 640, device=device, dtype=dtype)

        model_path = tmp_path / "rtdetr.onnx"

        dynamic_axes = {"images": {0: "N"}}
        # Suppress onnxscript deprecation warnings (Python 3.15 compatibility)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="onnxscript.converter")
            torch.onnx.export(
                detector,
                data,
                model_path,
                input_names=["images"],
                output_names=["detections"],
                dynamic_axes=dynamic_axes,
                opset_version=17,
            )

        assert model_path.is_file()

    def test_output_check_keeps_an_image_module_cache(self):
        # ObjectDetector.to_onnx first runs the detector once to tell a per-image list from one tensor. That run, like
        # the export's own, leaves the show()/save() cache of an image module inside the detector as it was.
        from kornia.contrib.object_detection import _returns_detections_per_image

        class _Pre(kornia.core.ImageModule):
            def forward(self, x):
                return x, torch.tensor([[x.shape[-2], x.shape[-1]]] * x.shape[0])

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 6, 1)

            def forward(self, x):
                y = self.conv(x).mean((-2, -1))
                return y[:, None, :2], y[:, None, 2:]

        class _Post(torch.nn.Module):
            def forward(self, logits, boxes, sizes):
                return torch.cat([logits, boxes], -1)

        pre = _Pre()
        detector = kornia.contrib.object_detection.ObjectDetector(_Model(), pre, _Post())
        pre(torch.rand(1, 3, 8, 8))
        cache = pre.__dict__["_output_image"]
        assert not _returns_detections_per_image(detector, torch.rand(2, 3, 16, 16))
        assert pre.__dict__["_output_image"] is cache

    def test_output_check_gives_back_the_random_state(self):
        # The output check runs the detector in eval mode; a model that still draws there must not consume the
        # global random state, wherever the check runs, before the export or after it.
        from kornia.contrib.object_detection import _returns_detections_per_image

        class _Pre(torch.nn.Module):
            def forward(self, x):
                return x, x.new_full((x.shape[0], 2), float(x.shape[-1]))

        class _DrawingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 6, 1)

            def forward(self, x):
                y = self.conv(x + 0.0 * torch.rand_like(x)).mean((-2, -1))
                return y[:, None, :2], y[:, None, 2:]

        class _Post(torch.nn.Module):
            def forward(self, logits, boxes, sizes):
                return torch.cat([logits, boxes], -1)

        detector = kornia.contrib.object_detection.ObjectDetector(_DrawingModel(), _Pre(), _Post())
        images = torch.rand(2, 3, 16, 16)
        state = torch.get_rng_state()
        assert not _returns_detections_per_image(detector, images)
        assert torch.equal(torch.get_rng_state(), state)

    @pytest.mark.parametrize("exporter", [{}, {"dynamo": False}], ids=["default-exporter", "legacy-exporter"])
    def test_export_gives_back_the_random_state_6140(self, exporter):
        # ObjectDetector.to_onnx draws an input for its output check before the export draws its dummy. Both give back
        # what they drew, so the draws after an export are the ones the same calls make without it.
        pytest.importorskip("onnx")
        if not exporter and not torch_version_lt(2, 9, 0):
            pytest.importorskip("onnxscript")  # the default exporter from torch 2.9 is the dynamo one, which needs it

        class _Pre(kornia.core.ImageModule):
            def forward(self, x):
                return x, x.new_full((x.shape[0], 2), float(x.shape[-1]))

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 6, 1)

            def forward(self, x):
                y = self.conv(x).mean((-2, -1))
                return y[:, None, :2], y[:, None, 2:]

        class _Post(torch.nn.Module):
            def forward(self, logits, boxes, sizes):
                return torch.cat([logits, boxes], -1)

        def _draws_after(export):
            torch.manual_seed(0)
            detector = kornia.contrib.object_detection.ObjectDetector(_Model(), _Pre(), _Post())
            detector(torch.rand(1, 3, 16, 16))
            if export:
                detector.to_onnx(image_size=16, save=False, **exporter)
            return torch.rand(8)

        self.assert_close(_draws_after(True), _draws_after(False))

    def test_results_from_detections_misspelt_format_5907(self, device, dtype):
        detections = torch.zeros(1, 6, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="'xywhh' is not a valid BoundingBoxDataFormat") as excinfo:
            kornia.contrib.object_detection.results_from_detections(detections, format="xywhh")
        assert isinstance(excinfo.value, KeyError)

    def test_results_from_detections(self, device, dtype):
        # label_id, confidence, data
        detections = torch.tensor(
            [
                [0, 0.9, 0.0, 0.0, 1.0, 1.0],
                [1, 0.8, 0.0, 0.0, 1.0, 1.0],
                [2, 0.7, 0.0, 0.0, 1.0, 1.0],
                [3, 0.6, 0.0, 0.0, 1.0, 1.0],
                [4, 0.5, 0.0, 0.0, 1.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )

        detector_results: list = kornia.contrib.object_detection.results_from_detections(detections, format="xywh")

        assert len(detector_results) == 5
        for j, det in enumerate(detector_results):
            for i in range(4):
                assert det.bbox.data[i] == float(detections[j, i + 2])

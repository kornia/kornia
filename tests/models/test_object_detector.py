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

from unittest.mock import Mock

import pytest
import torch

import kornia
from kornia.core._compat import torch_version_lt
from kornia.models.rt_detr import RTDETR, DETRPostProcessor, RTDETRConfig

from testing.base import BaseTester, onnx_exporter_params

_EXPORTERS = onnx_exporter_params("default-exporter", "legacy-exporter")


class TestObjectDetector(BaseTester):
    @pytest.mark.parametrize("num_detections", [0, 2])
    @pytest.mark.parametrize("method", ["visualize", "save"])
    def test_supplied_detections(self, device, dtype, num_detections, method, monkeypatch, tmp_path):
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
        if num_detections == 2:
            expected[:, :, 1, 1:4] = expected[:, :, 3, 1:4] = 0
            expected[:, :, 1:4, 1] = expected[:, :, 1:4, 3] = 0
            expected[:, :, 4, 4:7] = expected[:, :, 6, 4:7] = 0
            expected[:, :, 4:7, 4] = expected[:, :, 4:7, 6] = 0

        # visualize and save iterate a (B, D, 6) tensor and a list of (D, 6) tensors alike; the list is what a
        # confidence-filtering post-processor returns.
        detections = list(detections)
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
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("variant", ["resnet50d", "hgnetv2_l"])
    def test_onnx(self, variant):
        """The detector's model-only export of the ResNet-50 and HGNetV2 backbones matches eager in ONNX Runtime."""
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        if not torch_version_lt(2, 9, 0):
            pytest.importorskip("onnxscript")  # the default exporter from torch 2.9 is the dynamo one, which needs it

        torch.manual_seed(0)
        # Two decoder layers and 4 queries keep the export short; the backbone is the part only this test exports.
        config = RTDETRConfig(variant, 1, head_num_queries=4, head_num_decoder_layers=2)
        model = RTDETR.from_config(config).eval()
        pre_processor = kornia.models.processors.ResizePreProcessor(640, 640)
        post_processor = DETRPostProcessor(0.3, num_top_queries=3)
        detector = kornia.contrib.object_detection.ObjectDetector(model, pre_processor, post_processor)
        exported = detector.to_onnx(image_size=None, include_pre_and_post_processor=False, save=False)
        session = ort.InferenceSession(exported.SerializeToString(), providers=["CPUExecutionProvider"])

        # A batch and a size other than the traced (1, 3, 352, 352). Seed 84 keeps the encoder scores of the 4 kept
        # queries and the next one at least 2e-4 apart for both backbones.
        images = torch.rand(2, 3, 96, 128, generator=torch.Generator().manual_seed(84))
        scores: list[torch.Tensor] = []
        hook = model.decoder.enc_score_head.register_forward_hook(lambda m, i, out: scores.append(out.max(-1).values))
        with torch.no_grad():
            expected = model(images)
        hook.remove()
        # RT-DETR keeps the queries of the highest encoder scores; a near tie there would let a rounding difference
        # between ONNX Runtime and eager swap two whole boxes.
        kept = scores[0].topk(config.head_num_queries + 1, dim=1).values
        assert (kept[:, :-1] - kept[:, 1:]).min() > 1e-4, "near tie between kept encoder scores: pick another seed"

        actual = session.run(None, {session.get_inputs()[0].name: images.numpy()})
        assert [o.name for o in session.get_outputs()] == ["pred_logits", "pred_boxes"]
        for got, want in zip(actual, expected):
            self.assert_close(torch.from_numpy(got), want, rtol=1e-4, atol=1e-4)

    def test_output_check_keeps_an_image_module_cache(self):
        # ObjectDetector.to_onnx first runs the detector once to tell a per-image list from one tensor. That run, like
        # the export's own, leaves the show()/save() cache of an image module inside the detector as it was.
        from kornia.core.mixin.onnx import _returns_per_image_outputs

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
        assert not _returns_per_image_outputs(detector, torch.rand(2, 3, 16, 16))
        assert pre.__dict__["_output_image"] is cache

    def test_output_check_gives_back_the_random_state(self):
        # The output check runs the detector in eval mode; a model that still draws there must not consume the
        # global random state, wherever the check runs, before the export or after it.
        from kornia.core.mixin.onnx import _returns_per_image_outputs

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
        assert not _returns_per_image_outputs(detector, images)
        assert torch.equal(torch.get_rng_state(), state)

    @pytest.mark.parametrize("exporter", _EXPORTERS)
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

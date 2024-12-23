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

from pathlib import Path

import pytest
import torch

import kornia
from kornia.contrib.models.rt_detr import RTDETR, DETRPostProcessor, RTDETRConfig
from kornia.utils._compat import torch_version_lt

from testing.base import BaseTester


class TestObjectDetector(BaseTester):
    def test_smoke(self, device, dtype):
        batch_size = 3
        confidence = 0.3
        config = RTDETRConfig("resnet50d", 10, head_num_queries=10)
        model = RTDETR.from_config(config).to(device, dtype).eval()
        pre_processor = kornia.models.utils.ResizePreProcessor(32, 32)
        post_processor = DETRPostProcessor(confidence, num_top_queries=3).to(device, dtype).eval()
        detector = kornia.models.detection.ObjectDetector(model, pre_processor, post_processor)

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

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_lt(2, 0, 0), reason="Unsupported ONNX opset version: 16")
    @pytest.mark.parametrize("variant", ("resnet50d", "hgnetv2_l"))
    def test_onnx(self, device, dtype, tmp_path: Path, variant: str):
        config = RTDETRConfig(variant, 1)
        model = RTDETR.from_config(config).to(device=device, dtype=dtype).eval()
        pre_processor = kornia.models.utils.ResizePreProcessor(640, 640)
        post_processor = DETRPostProcessor(0.3, num_top_queries=3)
        detector = kornia.models.detection.ObjectDetector(model, pre_processor, post_processor)

        data = torch.rand(3, 400, 640, device=device, dtype=dtype)

        model_path = tmp_path / "rtdetr.onnx"

        dynamic_axes = {"images": {0: "N"}}
        torch.onnx.export(
            detector,
            [data],
            model_path,
            input_names=["images"],
            output_names=["detections"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )

        assert model_path.is_file()

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

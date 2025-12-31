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

import pytest
import torch

import kornia
from kornia.contrib.face_detection import FaceKeypoint

from testing.base import BaseTester


@pytest.fixture(scope="function")
def face_detector_model(device, dtype):
    """Fixture to instantiate FaceDetector once and reuse across tests."""
    return kornia.contrib.FaceDetector().to(device, dtype)


class TestFaceDetection(BaseTester):
    @pytest.mark.slow
    def test_smoke(self, device, dtype):
        assert kornia.contrib.FaceDetector().to(device, dtype) is not None

    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_valid(self, batch_size, device, dtype):
        torch.manual_seed(44)
        img = torch.rand(batch_size, 3, 320, 320, device=device, dtype=dtype)
        face_detection = kornia.contrib.FaceDetector().to(device, dtype)
        dets = face_detection(img)
        assert isinstance(dets, list)
        assert len(dets) == batch_size  # same as the number of images
        assert isinstance(dets[0], torch.Tensor)
        assert dets[0].shape[0] >= 0  # number of detections
        assert dets[0].shape[1] == 15  # dims of each detection

    @pytest.mark.slow
    def test_jit(self, device, dtype):
        op = kornia.contrib.FaceDetector().to(device, dtype)
        op_jit = torch.jit.script(op)
        assert op_jit is not None

    @pytest.mark.slow
    def test_results(self, device, dtype):
        data = torch.tensor(
            [0.0, 0.0, 100.0, 200.0, 10.0, 10.0, 20.0, 10.0, 10.0, 50.0, 100.0, 50.0, 150.0, 10.0, 0.99],
            device=device,
            dtype=dtype,
        )
        res = kornia.contrib.FaceDetectorResult(data)
        assert res.xmin == 0.0
        assert res.ymin == 0.0
        assert res.xmax == 100.0
        assert res.ymax == 200.0
        assert res.score == 0.99
        assert res.width == 100.0
        assert res.height == 200.0
        assert res.top_left.tolist() == [0.0, 0.0]
        assert res.top_right.tolist() == [100.0, 0.0]
        assert res.bottom_right.tolist() == [100.0, 200.0]
        assert res.bottom_left.tolist() == [0.0, 200.0]
        assert res.get_keypoint(FaceKeypoint.EYE_LEFT).tolist() == [10.0, 10.0]
        assert res.get_keypoint(FaceKeypoint.EYE_RIGHT).tolist() == [20.0, 10.0]
        assert res.get_keypoint(FaceKeypoint.NOSE).tolist() == [10.0, 50.0]
        assert res.get_keypoint(FaceKeypoint.MOUTH_LEFT).tolist() == [100.0, 50.0]
        assert res.get_keypoint(FaceKeypoint.MOUTH_RIGHT).tolist() == [150.0, 10.0]

    @pytest.mark.slow
    def test_results_raise(self, device, dtype):
        data = torch.zeros(14, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            _ = kornia.contrib.FaceDetectorResult(data)

    def test_exception_invalid_input_shape(self, face_detector_model, device, dtype):
        """Test that FaceDetector raises appropriate error for invalid input."""
        img = torch.rand(1, 4, 320, 320, device=device, dtype=dtype)  # Wrong channels

        with pytest.raises(RuntimeError) as errinfo:
            face_detector_model(img)
        assert "expected" in str(errinfo).lower()

    @pytest.mark.slow
    def test_dynamo(self, device, dtype, torch_optimizer):
        """Test FaceDetector with torch dynamo optimizer."""
        torch.manual_seed(44)
        img = torch.rand(1, 3, 320, 320, device=device, dtype=dtype)

        op = kornia.contrib.FaceDetector().to(device, dtype)
        op_optimized = torch_optimizer(op)

        # Compare outputs
        result_original = op(img)
        result_optimized = op_optimized(img)

        assert len(result_original) == len(result_optimized)

    @pytest.mark.slow
    def test_model_onnx(self, device, dtype, tmp_path):
        torch.manual_seed(44)
        img = torch.rand(1, 3, 320, 320, device=device, dtype=dtype)
        face_detection = kornia.contrib.FaceDetector().to(device, dtype)
        model = face_detection.model

        model_path = tmp_path / "facedetector_model.onnx"

        dynamic_axes = {"images": {0: "B"}}
        torch.onnx.export(
            model,
            img,
            model_path,
            input_names=["images"],
            output_names=["loc", "conf", "iou"],
            dynamic_axes=dynamic_axes,
        )

        assert model_path.is_file()

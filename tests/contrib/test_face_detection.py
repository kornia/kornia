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

import pytest
import torch

import kornia
from kornia.contrib.face_detection import FaceKeypoint
from kornia.models.yunet import YuNet

from testing.base import BaseTester


class TestFaceDetection(BaseTester):
    @pytest.fixture
    def face_detector(self, device, dtype):
        return kornia.contrib.FaceDetector().to(device, dtype)

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

    @pytest.mark.slow
    def test_model_onnx(self, device, dtype, tmp_path):
        torch.manual_seed(44)
        img = torch.rand(1, 3, 320, 320, device=device, dtype=dtype)
        face_detection = kornia.contrib.FaceDetector().to(device, dtype)
        model = face_detection.model

        model_path = tmp_path / "facedetector_model.onnx"

        dynamic_axes = {"images": {0: "B"}}
        # Suppress onnxscript deprecation warnings (Python 3.15 compatibility)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="onnxscript.converter")
            torch.onnx.export(
                model,
                img,
                model_path,
                input_names=["images"],
                output_names=["loc", "conf", "iou"],
                dynamic_axes=dynamic_axes,
            )
        assert model_path.is_file()

    @pytest.mark.device_agnostic
    def test_yunet_mode(self, monkeypatch):
        # ``training`` of the model and of every submodule: ``{False}`` is evaluation mode throughout.
        assert {m.training for m in YuNet("test", pretrained=False).modules()} == {True}

        state_dict = YuNet("test", pretrained=False).state_dict()
        monkeypatch.setattr("kornia.models.yunet.model.load_state_dict_from_url", lambda *args, **kwargs: state_dict)
        assert {m.training for m in YuNet("test", pretrained=True).modules()} == {False}

    @pytest.mark.parametrize("phase", ["test", "train"])
    def test_yunet_empty_batch_6266(self, device, dtype, phase):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        model = YuNet(phase, pretrained=False).to(device, dtype).eval()
        out = model(torch.rand(0, 3, 64, 64, device=device, dtype=dtype))
        assert {k: v.shape for k, v in out.items()} == {"loc": (0, 235, 14), "conf": (0, 235, 2), "iou": (0, 235, 1)}

    def test_empty_batch_6266(self, device, dtype, monkeypatch):
        """An empty batch gives one detection tensor per image: none."""
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        state_dict = YuNet("test", pretrained=False).state_dict()
        monkeypatch.setattr("kornia.models.yunet.model.load_state_dict_from_url", lambda *args, **kwargs: state_dict)
        face_detector = kornia.contrib.FaceDetector().to(device, dtype)
        assert face_detector(torch.rand(0, 3, 64, 64, device=device, dtype=dtype)) == []

    def test_exception(self, face_detector, device, dtype):
        img = torch.rand(1, 4, 320, 320, device=device, dtype=dtype)

        with pytest.raises(ValueError, match=r"\(\*, 3, H, W\)"):
            face_detector(img)

    def test_preprocess_checks_the_channels(self, face_detector, device, dtype):
        # ``preprocess`` flips RGB to BGR with ``rgb_to_bgr``, which checks for ``(*, 3, H, W)``. A plain ``flip(-3)``
        # would accept these and leave the error to YuNet, or flip the height of a channels-last image.
        for shape in ((1, 4, 64, 96), (1, 1, 64, 96), (1, 64, 96, 3)):
            with pytest.raises(ValueError, match=r"\(\*, 3, H, W\)"):
                face_detector.preprocess(torch.rand(*shape, device=device, dtype=dtype) * 255)

    def test_yunet_receives_bgr_0_255(self, face_detector, device, dtype):
        # The YuNet weights were trained on BGR, float 0-255, with no mean or std: the DALI loader converts to BGR and
        # resizes to float (ShiqiYu/libfacedetection.train@74f3aa7 src/data.py:76, 90), and evaluation reads
        # ``cv2.imread`` images (tasks/task1/datasets.py:53, detect.py:100-106).
        # ``FaceDetector`` takes RGB in [0, 255] and only flips the channels.
        captured = []
        handle = face_detector.model.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
        img = torch.rand(1, 3, 64, 96, device=device, dtype=dtype) * 255
        try:
            face_detector(img)
        finally:
            handle.remove()

        assert len(captured) == 1
        assert torch.equal(captured[0], img.flip(-3))
        assert torch.equal(face_detector.preprocess(img), img.flip(-3))

    def test_result_checks_the_last_dimension(self, device, dtype):
        # A row holds 15 values: the box (4), five (x, y) landmarks (10) and the score (1).
        kornia.contrib.FaceDetectorResult(torch.rand(15, device=device, dtype=dtype))
        kornia.contrib.FaceDetectorResult(torch.rand(3, 15, device=device, dtype=dtype))
        with pytest.raises(ValueError, match="15"):
            kornia.contrib.FaceDetectorResult(torch.rand(20, 14, device=device, dtype=dtype))

    @pytest.mark.slow
    def test_dynamo(self, face_detector, device, dtype, torch_optimizer):
        torch.manual_seed(44)
        img = torch.rand(1, 3, 320, 320, device=device, dtype=dtype)

        op_compiled = torch_optimizer(face_detector)

        out_ref = face_detector(img)
        out_opt = op_compiled(img)

        assert len(out_ref) == len(out_opt)

        for r, o in zip(out_ref, out_opt):
            torch.testing.assert_close(r, o, rtol=1e-4, atol=1e-4)

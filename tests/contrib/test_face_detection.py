import pytest
import torch

import kornia
from kornia.contrib.face_detection import FaceKeypoint

from testing.base import BaseTester


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

import torch
import kornia
import pytest
from torch.testing import assert_allclose

from typing import Type


class _TestParams:
    height = 4
    width = 6
    fx = 1
    fy = 2
    cx = width / 2
    cy = height / 2


@pytest.fixture(params=[2, 4])
def batch_size(request):
    return request.param


class TestStereoCamera:
    @staticmethod
    def _get_real_left_camera(batch_size, device, dtype):
        cam = torch.tensor([[996.40068207, 0., 375.02582169, 0.],
                            [0., 996.40068207, 240.26374817, 0.],
                            [0., 0., 1., 0.]], device=device, dtype=dtype)
        return cam.expand(batch_size, -1, -1)

    @staticmethod
    def _get_real_right_camera(batch_size, device, dtype):
        cam = torch.tensor([[996.40068207, 0., 375.02582169, -5430.17323447],
                            [0., 996.40068207, 240.26374817, 0.],
                            [0., 0., 1., 0.]], device=device, dtype=dtype)
        return cam.expand(batch_size, -1, -1)

    @staticmethod
    def _create_rectified_camera(params: Type[_TestParams], batch_size, device, dtype, tx_fx=None):
        intrinsics = torch.zeros((3, 4), device=device, dtype=dtype)
        intrinsics[..., 0, 0] = params.fx
        intrinsics[..., 1, 1] = params.fy
        intrinsics[..., 0, 2] = params.cx
        intrinsics[..., 1, 2] = params.cy

        if tx_fx:
            intrinsics[..., 0, 3] = tx_fx

        return intrinsics.expand(batch_size, -1, -1)

    def _create_left_camera(self, batch_size, device, dtype):
        return self._create_rectified_camera(_TestParams, batch_size, device, dtype)

    def _create_right_camera(self, batch_size, device, dtype, tx_fx):
        return self._create_rectified_camera(_TestParams, batch_size, device, dtype, tx_fx=tx_fx)

    def _create_stereo_camera(self, batch_size, device, dtype, tx_fx):
        left_rectified_camera = self._create_left_camera(batch_size, device, dtype)
        right_rectified_camera = self._create_right_camera(batch_size, device, dtype, tx_fx)
        return left_rectified_camera, right_rectified_camera

    def _get_real_stereo_camera(self, batch_size, device, dtype):
        return (self._get_real_left_camera(batch_size, device, dtype),
                self._get_real_right_camera(batch_size, device, dtype))

    def test_stereo_camera_attributes_smoke(self, batch_size, device, dtype):
        tx_fx = -10
        left_rectified_camera, right_rectified_camera = self._create_stereo_camera(batch_size, device, dtype, tx_fx)

        stereo_camera = kornia.StereoCamera(left_rectified_camera, right_rectified_camera)

        def _assert_all(x, y):
            assert torch.all(torch.eq(x, y))

        _assert_all(stereo_camera.fx, _TestParams.fx)
        _assert_all(stereo_camera.fy, _TestParams.fy)
        _assert_all(stereo_camera.cx, _TestParams.cx)
        _assert_all(stereo_camera.cy, _TestParams.cy)
        _assert_all(stereo_camera.tx, - tx_fx / _TestParams.fx)

        assert stereo_camera.Q.shape == (batch_size, 4, 4)

    def test_stereo_camera_attributes_real(self, batch_size, device, dtype):
        left_rectified_camera, right_rectified_camera = self._get_real_stereo_camera(batch_size, device, dtype)

        stereo_camera = kornia.StereoCamera(left_rectified_camera, right_rectified_camera)
        assert_allclose(stereo_camera.fx, left_rectified_camera[..., 0, 0])
        assert_allclose(stereo_camera.fy, left_rectified_camera[..., 1, 1])
        assert_allclose(stereo_camera.cx, left_rectified_camera[..., 0, 2])
        assert_allclose(stereo_camera.cy, left_rectified_camera[..., 1, 2])
        assert_allclose(stereo_camera.tx, - right_rectified_camera[..., 0, 3] / right_rectified_camera[..., 0, 0])

        assert stereo_camera.Q.shape == (batch_size, 4, 4)

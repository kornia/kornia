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


class _RealTestData:
    """
    From the kitti 2015 dataset.
    See here for more info:
    https://github.com/yanii/kitti-pcl/blob/master/KITTI_README.TXT
    """
    @property
    def height(self):
        return 375

    @property
    def width(self):
        return 1242

    @staticmethod
    def _get_real_left_camera(batch_size, device, dtype):
        """
        From data_scene_flow_calib/testing/calib_cam_to_cam/000000.txt:P_rect_02
        Args:
            batch_size:
            device:
            dtype:

        Returns:

        """
        cam = torch.tensor([7.215377e+02, 0.000000e+00, 6.095593e+02,
                            4.485728e+01, 0.000000e+00, 7.215377e+02,
                            1.728540e+02, 2.163791e-01, 0.000000e+00,
                            0.000000e+00, 1.000000e+00, 2.745884e-03], device=device, dtype=dtype).reshape(3, 4)
        return cam.expand(batch_size, -1, -1)

    @staticmethod
    def _get_real_right_camera(batch_size, device, dtype):
        """
        From data_scene_flow_calib/testing/calib_cam_to_cam/000000.txt:P_rect_02
        Args:
            batch_size:
            device:
            dtype:

        Returns:

        """
        cam = torch.tensor([7.215377e+02, 0.000000e+0, 6.095593e+02,
                            -3.395242e+02, 0.000000e+00, 7.215377e+02,
                            1.728540e+02, 2.199936e+00, 0.000000e+00,
                            0.000000e+00, 1.000000e+00, 2.729905e-03], device=device, dtype=dtype).reshape(3, 4)
        return cam.expand(batch_size, -1, -1)

    @staticmethod
    def _get_real_stereo_camera(batch_size, device, dtype):
        return (_RealTestData._get_real_left_camera(batch_size, device, dtype),
                _RealTestData._get_real_right_camera(batch_size, device, dtype))


class _SmokeTestData:
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

    @staticmethod
    def _create_left_camera(batch_size, device, dtype):
        return _SmokeTestData._create_rectified_camera(_TestParams, batch_size, device, dtype)

    @staticmethod
    def _create_right_camera(batch_size, device, dtype, tx_fx):
        return _SmokeTestData._create_rectified_camera(_TestParams, batch_size, device, dtype, tx_fx=tx_fx)

    @staticmethod
    def _create_stereo_camera(batch_size, device, dtype, tx_fx):
        left_rectified_camera = _SmokeTestData._create_left_camera(batch_size, device, dtype)
        right_rectified_camera = _SmokeTestData._create_right_camera(batch_size, device, dtype, tx_fx)
        return left_rectified_camera, right_rectified_camera


@pytest.fixture(params=[2, 4])
def batch_size(request):
    return request.param


class TestStereoCamera:
    @staticmethod
    def _create_disparity_tensor(batch_size, height, width, max_disparity, device):
        size = (batch_size, height, width)
        return torch.randint(size=size, low=0, high=max_disparity, device=device, dtype=torch.float32)

    def test_stereo_camera_attributes_smoke(self, batch_size, device, dtype):
        tx_fx = -10
        left_rectified_camera, right_rectified_camera = _SmokeTestData._create_stereo_camera(batch_size, device, dtype,
                                                                                             tx_fx)

        stereo_camera = kornia.StereoCamera(left_rectified_camera, right_rectified_camera)

        def _assert_all(x, y):
            assert torch.all(torch.eq(x, y))

        _assert_all(stereo_camera.fx, _TestParams.fx)
        _assert_all(stereo_camera.fy, _TestParams.fy)
        _assert_all(stereo_camera.cx, _TestParams.cx)
        _assert_all(stereo_camera.cy, _TestParams.cy)
        _assert_all(stereo_camera.tx, tx_fx / _TestParams.fx)

        assert stereo_camera.Q.shape == (batch_size, 4, 4)

    def test_stereo_camera_attributes_real(self, batch_size, device, dtype):
        left_rectified_camera, right_rectified_camera = _RealTestData._get_real_stereo_camera(batch_size, device, dtype)

        stereo_camera = kornia.StereoCamera(left_rectified_camera, right_rectified_camera)
        assert_allclose(stereo_camera.fx, left_rectified_camera[..., 0, 0])
        assert_allclose(stereo_camera.fy, left_rectified_camera[..., 1, 1])
        assert_allclose(stereo_camera.cx, left_rectified_camera[..., 0, 2])
        assert_allclose(stereo_camera.cy, left_rectified_camera[..., 1, 2])
        assert_allclose(stereo_camera.tx, right_rectified_camera[..., 0, 3] / right_rectified_camera[..., 0, 0])

        assert stereo_camera.Q.shape == (batch_size, 4, 4)

    def test_reproject_disparity_to_3D_smoke(self, batch_size, device, dtype):
        tx_fx = -10
        left_rectified_camera, right_rectified_camera = _SmokeTestData._create_stereo_camera(batch_size, device, dtype,
                                                                                             tx_fx)
        disparity_tensor = self._create_disparity_tensor(batch_size, _TestParams.height, _TestParams.width,
                                                         max_disparity=2, device=device)
        stereo_camera = kornia.StereoCamera(left_rectified_camera, right_rectified_camera)
        xyz = stereo_camera.reproject_disparity_to_3D(disparity_tensor)

        assert xyz.shape == (batch_size, _TestParams.height, _TestParams.width, 3)
        assert xyz.dtype == torch.float32
        assert xyz.device == device

    def test_reproject_disparity_to_3D_real(self, batch_size, device, dtype):
        height, width = _RealTestData().height, _RealTestData().width
        disparity_tensor = self._create_disparity_tensor(batch_size, height, width, max_disparity=80,
                                                         device=device)
        left_rectified_camera, right_rectified_camera = _RealTestData._get_real_stereo_camera(batch_size, device, dtype)
        stereo_camera = kornia.StereoCamera(left_rectified_camera, right_rectified_camera)

        xyz = stereo_camera.reproject_disparity_to_3D(disparity_tensor)

        assert xyz.shape == (batch_size, height, width, 3)
        assert xyz.device == device
        assert xyz.dtype == dtype

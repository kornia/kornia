from typing import Type

import pytest
import torch
from torch.testing import assert_allclose

from kornia.geometry.camera.stereo import StereoCamera


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    return request.param


class _TestParams:
    """Collection of test parameters for smoke test."""

    height = 4
    width = 6
    fx = 1
    fy = 2
    cx = width / 2
    cy = height / 2


class _RealTestData:
    """Collection of data from a real stereo setup."""

    @property
    def height(self):
        return 375

    @property
    def width(self):
        return 1242

    @staticmethod
    def _get_real_left_camera(batch_size, device, dtype):
        cam = torch.tensor(
            [
                9.9640068207290187e02,
                0.0,
                3.7502582168579102e02,
                0.0,
                0.0,
                9.9640068207290187e02,
                2.4026374816894531e02,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            device=device,
            dtype=dtype,
        ).reshape(3, 4)
        return cam.expand(batch_size, -1, -1)

    @staticmethod
    def _get_real_right_camera(batch_size, device, dtype):
        cam = torch.tensor(
            [
                9.9640068207290187e02,
                0.0,
                3.7502582168579102e02,
                -5.4301732344712009e03,
                0.0,
                9.9640068207290187e02,
                2.4026374816894531e02,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            device=device,
            dtype=dtype,
        ).reshape(3, 4)
        return cam.expand(batch_size, -1, -1)

    @staticmethod
    def _get_real_stereo_camera(batch_size, device, dtype):
        return (
            _RealTestData._get_real_left_camera(batch_size, device, dtype),
            _RealTestData._get_real_right_camera(batch_size, device, dtype),
        )

    @staticmethod
    def _get_real_disparity(batch_size, device, dtype):
        # First 10 cols of 1 row in a real disparity map.
        disp = torch.tensor(
            [
                [
                    [
                        [67.5039],
                        [67.5078],
                        [67.5117],
                        [67.5156],
                        [67.5195],
                        [67.5234],
                        [67.5273],
                        [67.5312],
                        [67.5352],
                        [67.5391],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).permute(0, 2, 3, 1)
        return disp.expand(batch_size, -1, -1, -1)

    @staticmethod
    def _get_real_point_cloud(batch_size, device, dtype):
        # First 10 cols of 1 row in the ground truth point cloud computed from above disparity map.
        pc = torch.tensor(
            [
                [
                    [[-30.2769, -19.3972, 80.4424]],
                    [[-30.1945, -19.3961, 80.4377]],
                    [[-30.1120, -19.3950, 80.4330]],
                    [[-30.0295, -19.3938, 80.4284]],
                    [[-29.9471, -19.3927, 80.4237]],
                    [[-29.8646, -19.3916, 80.4191]],
                    [[-29.7822, -19.3905, 80.4144]],
                    [[-29.6998, -19.3893, 80.4098]],
                    [[-29.6174, -19.3882, 80.4051]],
                    [[-29.5350, -19.3871, 80.4005]],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        return pc.expand(batch_size, -1, -1, -1)


class _SmokeTestData:
    """Collection of smoke test data."""

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


class TestStereoCamera:
    """Test class for :class:`~kornia.geometry.camera.stereo.StereoCamera`"""

    @staticmethod
    def _create_disparity_tensor(batch_size, height, width, max_disparity, device, dtype):
        size = (batch_size, height, width, 1)
        return torch.randint(size=size, low=0, high=max_disparity, device=device, dtype=dtype)

    @staticmethod
    def test_stereo_camera_attributes_smoke(batch_size, device, dtype):
        """Test proper setup of the class for smoke data."""
        tx_fx = -10
        left_rectified_camera, right_rectified_camera = _SmokeTestData._create_stereo_camera(
            batch_size, device, dtype, tx_fx
        )

        stereo_camera = StereoCamera(left_rectified_camera, right_rectified_camera)

        def _assert_all(x, y):
            assert torch.all(torch.eq(x, y))

        _assert_all(stereo_camera.fx, _TestParams.fx)
        _assert_all(stereo_camera.fy, _TestParams.fy)
        _assert_all(stereo_camera.cx_left, _TestParams.cx)
        _assert_all(stereo_camera.cy, _TestParams.cy)
        _assert_all(stereo_camera.tx, -tx_fx / _TestParams.fx)

        assert stereo_camera.Q.shape == (batch_size, 4, 4)
        assert stereo_camera.Q.dtype in (torch.float16, torch.float32, torch.float64)

    @staticmethod
    def test_stereo_camera_attributes_real(batch_size, device, dtype):
        """Test proper setup of the class for real data."""
        left_rectified_camera, right_rectified_camera = _RealTestData._get_real_stereo_camera(batch_size, device, dtype)

        stereo_camera = StereoCamera(left_rectified_camera, right_rectified_camera)
        assert_allclose(stereo_camera.fx, left_rectified_camera[..., 0, 0])
        assert_allclose(stereo_camera.fy, left_rectified_camera[..., 1, 1])
        assert_allclose(stereo_camera.cx_left, left_rectified_camera[..., 0, 2])
        assert_allclose(stereo_camera.cy, left_rectified_camera[..., 1, 2])
        assert_allclose(stereo_camera.tx, -right_rectified_camera[..., 0, 3] / right_rectified_camera[..., 0, 0])

        assert stereo_camera.Q.shape == (batch_size, 4, 4)
        assert stereo_camera.Q.dtype in (torch.float16, torch.float32, torch.float64)

    def test_reproject_disparity_to_3D_smoke(self, batch_size, device, dtype):
        """Test reprojecting of disparity to 3D for smoke data."""
        tx_fx = -10
        left_rectified_camera, right_rectified_camera = _SmokeTestData._create_stereo_camera(
            batch_size, device, dtype, tx_fx
        )
        disparity_tensor = self._create_disparity_tensor(
            batch_size, _TestParams.height, _TestParams.width, max_disparity=2, device=device, dtype=dtype
        )
        stereo_camera = StereoCamera(left_rectified_camera, right_rectified_camera)
        xyz = stereo_camera.reproject_disparity_to_3D(disparity_tensor)

        assert xyz.shape == (batch_size, _TestParams.height, _TestParams.width, 3)
        assert xyz.dtype in (torch.float16, torch.float32, torch.float64)
        assert xyz.device == device

    @staticmethod
    def test_reproject_disparity_to_3D_real(batch_size, device, dtype):
        """Test reprojecting of disparity to 3D for known outcome."""
        disparity_tensor = _RealTestData._get_real_disparity(batch_size, device, dtype)
        xyz_gt = _RealTestData._get_real_point_cloud(batch_size, device, dtype)

        left_rectified_camera, right_rectified_camera = _RealTestData._get_real_stereo_camera(batch_size, device, dtype)
        stereo_camera = StereoCamera(left_rectified_camera, right_rectified_camera)

        xyz = stereo_camera.reproject_disparity_to_3D(disparity_tensor)

        assert_allclose(xyz, xyz_gt)

    def test_reproject_disparity_to_3D_simple(self, batch_size, device, dtype):
        """Test reprojecting of disparity to 3D for real data."""
        height, width = _RealTestData().height, _RealTestData().width
        max_disparity = 80
        disparity_tensor = self._create_disparity_tensor(
            batch_size, height, width, max_disparity=max_disparity, device=device, dtype=dtype
        )
        left_rectified_camera, right_rectified_camera = _RealTestData._get_real_stereo_camera(batch_size, device, dtype)
        stereo_camera = StereoCamera(left_rectified_camera, right_rectified_camera)

        xyz = stereo_camera.reproject_disparity_to_3D(disparity_tensor)

        assert xyz.shape == (batch_size, height, width, 3)
        assert xyz.dtype in (torch.float16, torch.float32, torch.float64)
        assert xyz.dtype == dtype

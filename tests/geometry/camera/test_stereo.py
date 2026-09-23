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

from kornia.geometry.camera import StereoCamera
from kornia.geometry.camera.stereo import StereoException, reproject_disparity_to_3D

from testing.base import BaseTester


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


class _RealTestData(BaseTester):
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
        )
        # The literal is already (B, rows=1, cols=10, 1). It used to be permuted to
        # (B, 10, 1, 1) -- ten rows of one column -- which contradicts the comment above
        # and the ground truth below, and made this test pass only because the pixel
        # indices were swapped inside reproject_disparity_to_3D (#4269).
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
        ).permute(0, 2, 1, 3)
        # Same ten points, laid out as the one row of ten columns the comment describes.

        return pc.expand(batch_size, -1, -1, -1)


class _SmokeTestData:
    """Collection of smoke test data."""

    @staticmethod
    def _create_rectified_camera(params, batch_size, device, dtype, tx_fx=None):
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


class TestStereoCamera(BaseTester):
    """Test class for :class:`~kornia.geometry.camera.stereo.StereoCamera`"""

    @pytest.mark.parametrize("camera_index", [0, 1])
    @pytest.mark.parametrize("shape", [(3, 3), (4, 4), (3, 5)])
    def test_exception_invalid_camera_shape(self, batch_size, device, dtype, camera_index, shape):
        cameras = list(_SmokeTestData._create_stereo_camera(batch_size, device, dtype, tx_fx=-10))
        cameras[camera_index] = torch.nn.functional.pad(cameras[camera_index], (0, shape[1] - 4, 0, shape[0] - 3))
        camera_name = "rectified_left_camera" if camera_index == 0 else "rectified_right_camera"

        with pytest.raises(StereoException, match=f"Expected each '{camera_name}' to be of shape"):
            StereoCamera(*cameras)

    @pytest.mark.parametrize("disparity_shape", [(1, 1, 3, 5), (2, 3, 5, 2)])
    @pytest.mark.parametrize("entrypoint", ["method", "function"])
    def test_reproject_disparity_layout_error_4374(self, disparity_shape, entrypoint, device, dtype):
        left, right = _SmokeTestData._create_stereo_camera(disparity_shape[0], device, dtype, tx_fx=-10)
        camera = StereoCamera(left, right)
        disparity = torch.ones(disparity_shape, device=device, dtype=dtype)

        with pytest.raises(StereoException) as exc_info:
            if entrypoint == "method":
                camera.reproject_disparity_to_3D(disparity)
            else:
                reproject_disparity_to_3D(disparity, camera.Q)

        assert str(exc_info.value).splitlines()[0] == (
            "Expected 'disparity_tensor' to have channels-last shape (B, H, W, 1) "
            "with a single channel in the last dimension. "
            f"Got {disparity.shape}."
        )

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

    def test_empty_batch_4281(self, device, dtype):
        # Regression for kornia#4281: an empty stereo rig is vacuously valid.
        left_rectified_camera = torch.zeros(0, 3, 4, device=device, dtype=dtype)
        right_rectified_camera = torch.zeros(0, 3, 4, device=device, dtype=dtype)

        stereo_camera = StereoCamera(left_rectified_camera, right_rectified_camera)

        assert stereo_camera.batch_size == 0
        assert stereo_camera.Q.shape == (0, 4, 4)
        assert stereo_camera.Q.dtype == dtype
        assert stereo_camera.Q.device == device

    def test_stereo_camera_attributes_real(self, batch_size, device, dtype):
        """Test proper setup of the class for real data."""
        left_rectified_camera, right_rectified_camera = _RealTestData._get_real_stereo_camera(batch_size, device, dtype)

        stereo_camera = StereoCamera(left_rectified_camera, right_rectified_camera)
        self.assert_close(stereo_camera.fx, left_rectified_camera[..., 0, 0])
        self.assert_close(stereo_camera.fy, left_rectified_camera[..., 1, 1])
        self.assert_close(stereo_camera.cx_left, left_rectified_camera[..., 0, 2])
        self.assert_close(stereo_camera.cy, left_rectified_camera[..., 1, 2])
        self.assert_close(stereo_camera.tx, -right_rectified_camera[..., 0, 3] / right_rectified_camera[..., 0, 0])
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

    def test_reproject_disparity_to_3D_real(self, batch_size, device, dtype):
        """Test reprojecting of disparity to 3D for known outcome."""
        disparity_tensor = _RealTestData._get_real_disparity(batch_size, device, dtype)
        xyz_gt = _RealTestData._get_real_point_cloud(batch_size, device, dtype)

        left_rectified_camera, right_rectified_camera = _RealTestData._get_real_stereo_camera(batch_size, device, dtype)
        stereo_camera = StereoCamera(left_rectified_camera, right_rectified_camera)

        xyz = stereo_camera.reproject_disparity_to_3D(disparity_tensor)

        self.assert_close(xyz, xyz_gt)

    def test_reproject_disparity_to_3D_uses_the_column_for_x(self, batch_size, device, dtype):
        """X must come from the column and Y from the row, as in cv2.reprojectImageTo3D.

        The meshgrid was unbound as ``v, u``, but create_meshgrid returns ``(x, y)``, so the
        row fed X and the column fed Y: every pixel got the value belonging to its transpose.
        A square rig with fx == fy and cx == cy hides it everywhere except off the diagonal,
        so this uses an asymmetric 3x5 rig with cx != cy (#4269).
        """
        fx, fy, cx, cy, tx = 100.0, 100.0, 4.0, 3.0, 0.5
        left = torch.zeros(batch_size, 3, 4, device=device, dtype=dtype)
        left[:, 0, 0], left[:, 1, 1], left[:, 2, 2] = fx, fy, 1.0
        left[:, 0, 2], left[:, 1, 2] = cx, cy
        right = left.clone()
        right[:, 0, 3] = -tx * fx
        camera = StereoCamera(left, right)

        rows, cols, disparity = 3, 5, 10.0
        points = camera.reproject_disparity_to_3D(
            torch.full((batch_size, rows, cols, 1), disparity, device=device, dtype=dtype)
        )

        depth = fx * tx / disparity
        expected = torch.stack(
            [
                (torch.arange(cols, device=device, dtype=dtype) - cx).view(1, 1, cols).expand(1, rows, cols)
                * depth
                / fx,
                (torch.arange(rows, device=device, dtype=dtype) - cy).view(1, rows, 1).expand(1, rows, cols)
                * depth
                / fy,
                torch.full((1, rows, cols), depth, device=device, dtype=dtype),
            ],
            dim=-1,
        ).expand(batch_size, -1, -1, -1)
        self.assert_close(points, expected)

        # The axis dependence, stated directly: X varies across a row, Y does not.
        assert not torch.allclose(points[0, 0, :, 0], points[0, 0, :1, 0].expand(cols))
        self.assert_close(points[0, 0, :, 1], points[0, 0, :1, 1].expand(cols))

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

    @staticmethod
    def _asymmetric_stereo(device, dtype, tx_fx=-50.0, fy=100.0, cx_right=4.0, batch=1):
        """Build a rectified pair with fx = 100, cx = 4, cy = 3 and a caller-chosen fy, tx * fx and right cx.

        cx = 4 != cy = 3 and the pins below use H = 3 != W = 5, so a row/column swap moves X and Y by different
        amounts; fy and tx * fx are parameters so a pin can vary one of them at a time. The right camera's last
        column is -tx * fx, so the default -50 is a baseline of tx = 0.5.
        """
        left = torch.tensor(
            [[[100.0, 0.0, 4.0, 0.0], [0.0, fy, 3.0, 0.0], [0.0, 0.0, 1.0, 0.0]]] * batch, device=device, dtype=dtype
        )
        right = torch.tensor(
            [[[100.0, 0.0, cx_right, tx_fx], [0.0, fy, 3.0, 0.0], [0.0, 0.0, 1.0, 0.0]]] * batch,
            device=device,
            dtype=dtype,
        )
        return StereoCamera(left, right)

    def test_convention_reproject_disparity_takes_bhw1(self, device, dtype):
        # The disparity map is channels-last (B, H, W, 1) for the method and the module-level function, and the
        # cloud is (B, H, W, 3); (B, 1, H, W) and (B, H, W) are rejected with a message naming the layout
        # (#4374). With fx = 100 and tx = 0.5, a disparity of 10 puts every point at Z = fx * tx / d = 5.
        cam = self._asymmetric_stereo(device, dtype)
        disparity = torch.full((1, 3, 5, 1), 10.0, device=device, dtype=dtype)
        points = cam.reproject_disparity_to_3D(disparity)
        assert points.shape == (1, 3, 5, 3)
        self.assert_close(points[0, 0, 0, 2], torch.tensor(5.0, device=device, dtype=dtype))
        with pytest.raises(StereoException, match=r"channels-last shape \(B, H, W, 1\)"):
            cam.reproject_disparity_to_3D(torch.full((1, 1, 3, 5), 10.0, device=device, dtype=dtype))
        with pytest.raises(StereoException, match="to have 4 dimensions"):
            cam.reproject_disparity_to_3D(torch.full((1, 3, 5), 10.0, device=device, dtype=dtype))

    def test_convention_reproject_method_and_free_function_are_byte_identical(self, device, dtype):
        # The method forwards the cached self.Q to the module-level function. The cloud is not all zero, so the
        # equality is not the trivial one.
        cam = self._asymmetric_stereo(device, dtype)
        disparity = torch.full((1, 3, 5, 1), 10.0, device=device, dtype=dtype)
        assert torch.equal(cam.reproject_disparity_to_3D(disparity), reproject_disparity_to_3D(disparity, cam.Q))
        assert cam.Q is cam._Q_matrix
        assert cam.reproject_disparity_to_3D(disparity).abs().max().item() > 0.1

    def test_convention_tx_is_the_right_translation_over_fx_and_fixes_q(self, device, dtype):
        # tx = -P_right[0, 3] / fx, and Q is built from it with Q[0, 0] = fy * (-tx), Q[1, 1] = fx * (-tx): the
        # row that scales the first output coordinate carries fy, so the fy = 50 arm fails a reading with fx
        # there. Q[3, 3] = fy * (cx_left - cx_right) is always 0 (#4270, below).
        cam = self._asymmetric_stereo(device, dtype)
        self.assert_close(cam.tx, torch.tensor([0.5], device=device, dtype=dtype))
        self.assert_close(cam.tx, -cam.rectified_right_camera[..., 0, 3] / cam.fx)
        symmetric_q = torch.tensor(
            [[[-50.0, 0.0, 0.0, 200.0], [0.0, -50.0, 0.0, 150.0], [0.0, 0.0, 0.0, -5000.0], [0.0, 0.0, -100.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(cam.Q, symmetric_q)
        assert cam.Q[0, 3, 3].item() == 0.0
        asymmetric_q = torch.tensor(
            [[[-25.0, 0.0, 0.0, 100.0], [0.0, -50.0, 0.0, 150.0], [0.0, 0.0, 0.0, -2500.0], [0.0, 0.0, -50.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(self._asymmetric_stereo(device, dtype, fy=50.0).Q, asymmetric_q)
        assert (symmetric_q - asymmetric_q).abs().max().item() > 1.0

    def test_wart_reproject_disparity_zero_disparity_returns_the_numerator_4267(self, device, dtype):
        # Wart pin for #4267: zero disparity (a point at infinity) gives W = 0, and the masked homogeneous divide
        # returns the numerator, a finite point -- behind the camera on a real rig -- so Q and -Q disagree
        # there. Delete or invert when #4267 settles one singular-input policy.
        disparity = torch.zeros(1, 1, 1, 1, device=device, dtype=dtype)
        self.assert_close(
            self._asymmetric_stereo(device, dtype).reproject_disparity_to_3D(disparity),
            torch.tensor([[[[200.0, 150.0, -5000.0]]]], device=device, dtype=dtype),
        )
        q = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 1.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor([[[[1.0, 2.0, 3.0]]]], device=device, dtype=dtype)
        self.assert_close(reproject_disparity_to_3D(disparity, q), expected)
        self.assert_close(reproject_disparity_to_3D(disparity, -q), -expected)

    def test_convention_reproject_disparity_uses_the_column_as_u_4269(self, device, dtype):
        # u is the column index and v the row index, as in cv2.reprojectImageTo3D: X = (u - cx) Z / fx,
        # Y = (v - cy) Z / fy, Z = fx * tx / d (repaired in #4269). fx = 100 != fy = 50 and H != W, so a reading
        # that swapped the indices or the focal lengths cannot reproduce the map.
        points = self._asymmetric_stereo(device, dtype, fy=50.0).reproject_disparity_to_3D(
            torch.full((1, 3, 5, 1), 10.0, device=device, dtype=dtype)
        )
        rows = torch.arange(3.0, device=device, dtype=dtype).view(3, 1).expand(3, 5)
        columns = torch.arange(5.0, device=device, dtype=dtype).view(1, 5).expand(3, 5)
        depth = torch.full((3, 5), 5.0, device=device, dtype=dtype)
        opencv = torch.stack([(columns - 4.0) * 0.05, (rows - 3.0) * 0.1, depth], dim=-1)[None]
        self.assert_close(points, opencv)
        # X depends only on the column and Y only on the row.
        assert bool((points[0, :, :, 0] == points[0, :1, :, 0]).all())
        assert bool((points[0, :, :, 1] == points[0, :, :1, 1]).all())

    def test_wart_stereo_rejects_differing_principal_points_4270(self, device, dtype):
        # Wart pin for #4270: Q[3, 3] = fy * (cx_left - cx_right) exists for a differing cx, but the constructor
        # rejects one, so Q[3, 3] is always 0. Delete when #4270 is repaired.
        with pytest.raises(StereoException, match="same parameters except for the last column"):
            self._asymmetric_stereo(device, dtype, cx_right=5.0)
        cam = self._asymmetric_stereo(device, dtype)
        assert torch.equal(cam.cx_left, cam.cx_right)
        assert cam.Q[0, 3, 3].item() == 0.0

    @pytest.mark.parametrize("bad_index", [0, 1])
    def test_convention_stereo_rejects_a_batch_with_one_positive_tx_fx_4270(self, bad_index, device, dtype):
        # The right camera's last column is -tx * fx; a batch in which any rig has it positive (cameras swapped)
        # is rejected (#4270), whichever position the bad rig holds.
        left = torch.tensor(
            [[[100.0, 0.0, 4.0, 0.0], [0.0, 100.0, 3.0, 0.0], [0.0, 0.0, 1.0, 0.0]]] * 2, device=device, dtype=dtype
        )
        right = torch.tensor(
            [[[100.0, 0.0, 4.0, -50.0], [0.0, 100.0, 3.0, 0.0], [0.0, 0.0, 1.0, 0.0]]] * 2, device=device, dtype=dtype
        )
        mixed_right = right.clone()
        mixed_right[bad_index, 0, 3] = 50.0

        with pytest.raises(StereoException, match="to be negative"):
            StereoCamera(left, mixed_right)
        with pytest.raises(StereoException, match="to be negative"):
            self._asymmetric_stereo(device, dtype, tx_fx=50.0)
        # the all-negative batch is still accepted, with the baseline read back unchanged
        self.assert_close(StereoCamera(left, right).tx, torch.tensor([0.5, 0.5], device=device, dtype=dtype))

    @pytest.mark.parametrize("zero", [0.0, -0.0])
    def test_convention_stereo_rejects_a_zero_baseline_4270(self, zero, device, dtype):
        # tx * fx == 0 (no baseline) is rejected (#4270) for a single rig and for one rig of a batch; -0.0 too.
        with pytest.raises(StereoException, match="non-zero stereo baseline"):
            self._asymmetric_stereo(device, dtype, tx_fx=zero)

        good = self._asymmetric_stereo(device, dtype, batch=2)
        right = good.rectified_right_camera.clone()
        right[1, 0, 3] = zero
        with pytest.raises(StereoException, match="non-zero stereo baseline"):
            StereoCamera(good.rectified_left_camera, right)

        # a real baseline on the same fixture still reprojects away from the origin
        disparity = torch.full((1, 3, 5, 1), 10.0, device=device, dtype=dtype)
        assert self._asymmetric_stereo(device, dtype).reproject_disparity_to_3D(disparity).abs().max().item() > 0.1

    def test_convention_stereo_rejects_a_four_by_four_pair_4270(self, device, dtype):
        cam = self._asymmetric_stereo(device, dtype)
        bottom = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        with pytest.raises(StereoException, match=r"to be of shape \(3, 4\)"):
            StereoCamera(
                torch.cat([cam.rectified_left_camera, bottom], dim=-2),
                torch.cat([cam.rectified_right_camera, bottom], dim=-2),
            )
        assert cam.Q.shape == (1, 4, 4)
        with pytest.raises(StereoException, match="to have 3 dimensions"):
            StereoCamera(cam.rectified_left_camera[0], cam.rectified_right_camera[0])

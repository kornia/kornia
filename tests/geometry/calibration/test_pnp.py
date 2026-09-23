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
from kornia.core.exceptions import BaseError, ShapeError
from kornia.geometry.calibration.pnp import _mean_isotropic_scale_normalize

from testing.base import BaseTester


class TestSolvePnpDlt(BaseTester):
    @staticmethod
    def _get_samples(shape, low, high, device, dtype):
        """Return a tensor having the given shape and whose values are in the range [low, high)"""
        return ((high - low) * torch.rand(shape, device=device, dtype=dtype)) + low

    @staticmethod
    def _project_to_image(world_points, world_to_cam_4x4, repeated_intrinsics):
        r"""Projects points in the world coordinate system to the image coordinate system.

        Since cam_points will have shape (B, N, 3), repeated_intrinsics should have shape (B, N, 3, 3) so that
        kornia.geometry.project_points can be used.
        """
        cam_points = kornia.geometry.transform_points(world_to_cam_4x4, world_points)
        return kornia.geometry.project_points(cam_points, repeated_intrinsics)

    @staticmethod
    def _get_world_points_and_img_points(cam_points, world_to_cam_4x4, repeated_intrinsics):
        r"""Calculates world_points and img_points.

        Since cam_points will have shape (B, N, 3), repeated_intrinsics should have shape (B, N, 3, 3) so that
        kornia.geometry.project_points can be used.
        """
        cam_to_world_4x4 = kornia.geometry.inverse_transformation(world_to_cam_4x4)
        world_points = kornia.geometry.transform_points(cam_to_world_4x4, cam_points)
        img_points = kornia.geometry.project_points(cam_points, repeated_intrinsics)

        return world_points, img_points

    def _get_test_data(self, num_points, device, dtype):
        """Creates some test data.

        Batch size is fixed to 2 for all tests.
        """
        batch_size = 2
        torch.manual_seed(84)

        tau = 2 * 3.141592653589793
        axis_angle_1 = self._get_samples(shape=(1, 3), low=-tau, high=tau, dtype=dtype, device=device)
        axis_angle_2 = self._get_samples(shape=(1, 3), low=-tau, high=tau, dtype=dtype, device=device)
        rotation_1 = kornia.geometry.axis_angle_to_rotation_matrix(axis_angle_1)
        rotation_2 = kornia.geometry.axis_angle_to_rotation_matrix(axis_angle_2)

        translation_1 = self._get_samples(shape=(3,), low=-100, high=100, dtype=dtype, device=device)
        translation_2 = self._get_samples(shape=(3,), low=-100, high=100, dtype=dtype, device=device)

        temp = torch.eye(4, dtype=dtype, device=device)
        world_to_cam_mats = temp.unsqueeze(0).repeat(batch_size, 1, 1)
        world_to_cam_mats[0, :3, :3] = torch.squeeze(rotation_1)
        world_to_cam_mats[0, :3, 3] = translation_1
        world_to_cam_mats[1, :3, :3] = torch.squeeze(rotation_2)
        world_to_cam_mats[1, :3, 3] = translation_2

        intrinsic_1 = torch.tensor(
            [[500.0, 0.0, 250.0], [0.0, 500.0, 250.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device
        )

        intrinsic_2 = torch.tensor(
            [[1000.0, 0.0, 550.0], [0.0, 750.0, 200.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device
        )

        intrinsics = torch.stack([intrinsic_1, intrinsic_2], dim=0)

        cam_points_xy = self._get_samples(
            shape=(batch_size, num_points, 2), low=-100, high=100, dtype=dtype, device=device
        )
        cam_points_z = self._get_samples(
            shape=(batch_size, num_points, 1), low=0.5, high=100, dtype=dtype, device=device
        )
        cam_points = torch.cat([cam_points_xy, cam_points_z], dim=-1)

        repeated_intrinsics = intrinsics.unsqueeze(1).repeat(1, num_points, 1, 1)
        world_points, img_points = self._get_world_points_and_img_points(
            cam_points, world_to_cam_mats, repeated_intrinsics
        )
        world_to_cam_3x4 = world_to_cam_mats[:, :3, :]

        return intrinsics, world_to_cam_3x4, world_points, img_points

    @pytest.mark.parametrize("num_points", (6, 20))
    def test_smoke(self, num_points, device, dtype):
        intrinsics, _, world_points, img_points = self._get_test_data(num_points, device, dtype)
        batch_size = world_points.shape[0]

        pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)
        assert pred_world_to_cam.shape == (batch_size, 3, 4)

    @pytest.mark.parametrize("num_points", (6,))
    def test_gradcheck(self, num_points, device):
        intrinsics, _, world_points, img_points = self._get_test_data(num_points, device, torch.float64)
        self.gradcheck(kornia.geometry.solve_pnp_dlt, (world_points, img_points, intrinsics))

    @pytest.mark.parametrize("num_points", (8,))
    def test_gradcheck_weights(self, num_points, device):
        intrinsics, _, world_points, img_points = self._get_test_data(num_points, device, torch.float64)
        weights = torch.rand(*world_points.shape[:2], device=device, dtype=torch.float64).abs()
        self.gradcheck(kornia.geometry.solve_pnp_dlt, (world_points, img_points, intrinsics, weights))

    @pytest.mark.parametrize("num_points", (6, 20))
    def test_pred_world_to_cam(self, num_points, device, dtype):
        intrinsics, gt_world_to_cam, world_points, img_points = self._get_test_data(num_points, device, dtype)
        pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)
        self.assert_close(pred_world_to_cam, gt_world_to_cam, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("num_points", (16, 20))
    def test_pred_world_to_cam_weighted(self, num_points, device, dtype):
        intrinsics, gt_world_to_cam, world_points, img_points = self._get_test_data(num_points, device, dtype)
        weights = torch.ones(*world_points.shape[:2], device=device, dtype=dtype)
        pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics, weights)
        self.assert_close(pred_world_to_cam, gt_world_to_cam, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("num_points", (25,))
    def test_pred_world_to_cam_weighted_rand(self, num_points, device, dtype):
        torch.manual_seed(0)
        intrinsics, gt_world_to_cam, world_points, img_points = self._get_test_data(num_points, device, dtype)
        weights = torch.ones(*world_points.shape[:2], device=device, dtype=dtype)
        weights[0, 0:2] = 1e-9
        world_points[0, 0:1] *= 0.1
        world_points[0, 1:2] += 100
        pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics, weights)
        self.assert_close(pred_world_to_cam, gt_world_to_cam, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("num_points", (6, 20))
    def test_project(self, num_points, device, dtype):
        intrinsics, _, world_points, img_points = self._get_test_data(num_points, device, dtype)

        pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)

        pred_world_to_cam_4x4 = kornia.core.ops.eye_like(4, pred_world_to_cam)
        pred_world_to_cam_4x4[:, :3, :] = pred_world_to_cam

        repeated_intrinsics = intrinsics.unsqueeze(1).repeat(1, num_points, 1, 1)
        pred_img_points = self._project_to_image(world_points, pred_world_to_cam_4x4, repeated_intrinsics)

        self.assert_close(pred_img_points, img_points, atol=1e-3, rtol=1e-3)

    @staticmethod
    def _convention_world_points(device, dtype):
        # Six non-coplanar, non-collinear points spread over x, y and z with mixed signs. A planar or collinear set
        # trips solve_pnp_dlt's own singular-value guard, and a
        # symmetric set would hide a transposed [R|t].
        return torch.tensor(
            [
                [
                    [5.0, -5.0, 10.0],
                    [0.0, 0.0, 11.5],
                    [2.5, 3.0, 16.0],
                    [9.0, -2.0, 13.0],
                    [-4.0, 5.0, 12.0],
                    [-5.0, 5.0, 11.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )

    def test_convention_returns_the_world_to_camera_extrinsics(self, device, dtype):
        # The (B, 3, 4) [R | t] maps world points into the camera frame (PinholeCamera.extrinsics, OpenCV
        # solvePnP). Camera-frame world points recover [I | 0]; a camera with cam = world + (1, 0, 0) recovers
        # t = (+1, 0, 0), where a camera-to-world reading would give (-1, 0, 0).
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("solve_pnp_dlt accepts float32 and float64 only (pinned below)")
        world_points = self._convention_world_points(device, dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        identity = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], device=device, dtype=dtype
        )
        recovered = kornia.geometry.solve_pnp_dlt(world_points, kornia.geometry.project_points(world_points, K), K)
        assert recovered.shape == (1, 3, 4)
        self.assert_close(recovered, identity, atol=1e-4, rtol=1e-4)
        shift = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        shifted = kornia.geometry.solve_pnp_dlt(
            world_points, kornia.geometry.project_points(world_points + shift, K), K
        )
        expected = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], device=device, dtype=dtype
        )
        self.assert_close(shifted, expected, atol=1e-4, rtol=1e-4)

    def test_convention_validation_errors_name_the_argument(self, device, dtype):
        # solve_pnp_dlt accepts float32 and float64 only, needs N >= 6 points and a float svd_eps; each check
        # raises a kornia BaseError that names its argument.
        world_points = self._convention_world_points(device, dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        img_points = world_points[..., :2] / world_points[..., 2:]
        if dtype not in (torch.float32, torch.float64):
            with pytest.raises(BaseError, match="world_points must be float32 or float64"):
                kornia.geometry.solve_pnp_dlt(world_points, img_points, K)
            return
        with pytest.raises(BaseError, match="world_points must hold at least 6 points"):
            kornia.geometry.solve_pnp_dlt(world_points[:, :5], img_points[:, :5], K)
        with pytest.raises(BaseError, match="svd_eps must be a float, got int"):
            kornia.geometry.solve_pnp_dlt(world_points, img_points, K, svd_eps=1)

    def test_convention_rejects_4x4_intrinsics(self, device, dtype):
        # intrinsics is the (B, 3, 3) K: the (B, 4, 4) matrix a PinholeCamera stores is rejected rather than
        # truncated, while its upper-left block solves.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("solve_pnp_dlt accepts float32 and float64 only")
        world_points = self._convention_world_points(device, dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        img_points = kornia.geometry.project_points(world_points, K)
        K_4x4 = torch.eye(4, device=device, dtype=dtype)[None].clone()
        K_4x4[:, :3, :3] = K
        with pytest.raises(ShapeError, match="expected 3, got 4"):
            kornia.geometry.solve_pnp_dlt(world_points, img_points, K_4x4)
        assert kornia.geometry.solve_pnp_dlt(world_points, img_points, K_4x4[:, :3, :3]).shape == (1, 3, 4)

    def test_convention_planar_world_points_raise(self, device, dtype):
        # A coplanar point set (the six points flattened onto z = 5) raises kornia's singular-value
        # AssertionError instead of returning a wrong pose.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("solve_pnp_dlt accepts float32 and float64 only")
        world_points = self._convention_world_points(device, dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        planar = torch.stack(
            [world_points[0, :, 0], world_points[0, :, 1], torch.full_like(world_points[0, :, 0], 5.0)], -1
        )[None]
        with pytest.raises(AssertionError, match="last singular value"):
            kornia.geometry.solve_pnp_dlt(planar, kornia.geometry.project_points(planar, K), K)

    def test_wart_zero_weight_point_still_enters_the_degeneracy_check_4799(self, device, dtype):
        # kornia#4799: the same coplanar set plus two off-plane points recovers [I | 0]; with those two points at
        # weight 0 it still passes the degeneracy check, which ignores the weights, and the rank-deficient system
        # returns a wrong pose. A fix that checks the weighted points raises here instead.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("solve_pnp_dlt accepts float32 and float64 only")
        world_points = self._convention_world_points(device, dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        planar = torch.stack(
            [world_points[0, :, 0], world_points[0, :, 1], torch.full_like(world_points[0, :, 0], 5.0)], -1
        )[None]
        off_plane = torch.tensor([[[0.0, 0.0, 8.0], [1.0, -1.0, 11.0]]], device=device, dtype=dtype)
        points = torch.cat([planar, off_plane], 1)
        img_points = kornia.geometry.project_points(points, K)
        weights = torch.ones(1, points.shape[1], device=device, dtype=dtype)
        pose = kornia.geometry.solve_pnp_dlt(points, img_points, K, weights=weights)
        assert pose[0, :, 3].abs().max() < 1e-2
        weights[0, -2:] = 0.0
        pose = kornia.geometry.solve_pnp_dlt(points, img_points, K, weights=weights)
        assert pose[0, :, 3].abs().max() > 1.0


class TestNormalization(BaseTester):
    @pytest.mark.parametrize("dimension", (2, 3, 5))
    def test_smoke(self, dimension, device, dtype):
        batch_size = 10
        num_points = 100
        points = torch.rand((batch_size, num_points, dimension), device=device, dtype=dtype)
        points_norm, transform = _mean_isotropic_scale_normalize(points)

        assert points_norm.shape == (batch_size, num_points, dimension)
        assert transform.shape == (batch_size, dimension + 1, dimension + 1)

    @pytest.mark.parametrize("dimension", (2, 3, 5))
    def test_gradcheck(self, dimension, device):
        batch_size = 3
        num_points = 5
        points = torch.rand((batch_size, num_points, dimension), device=device, dtype=torch.float64)

        self.gradcheck(_mean_isotropic_scale_normalize, (points,))

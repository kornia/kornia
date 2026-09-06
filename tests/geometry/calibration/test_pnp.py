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
from kornia.core.exceptions import ShapeError
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
        img_points = kornia.geometry.project_points(cam_points, repeated_intrinsics)

        return img_points

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
        # Six non-coplanar, non-collinear points spread over x, y and z with mixed signs -- the audit's set
        # (full_audit.py, WPTS). A planar or collinear set trips solve_pnp_dlt's own singular-value guard, and a
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
        # Convention pin (audit labels 5b-pnp-01, 5b-pnp-02, 5b-pnp-03): solve_pnp_dlt returns a (B, 3, 4)
        # [R | t] that maps WORLD points INTO the camera frame -- the same direction as PinholeCamera.extrinsics
        # and OpenCV's solvePnP rvec/tvec, not the camera pose in the world. Two cases: world points that are
        # already camera-frame points recover [I | 0], and a camera translated so that cam = world + (1, 0, 0)
        # recovers t = (+1, 0, 0). A cam-to-world reading would give t = (-1, 0, 0) on the second case, which is
        # why the identity case alone is not enough.
        # Snippet used to generate expected: solve_pnp_dlt(W, project_points(W + [1., 0., 0.], K), K) executed
        # 2026-09-06 on the batch-5b worktree (torch 2.14.0, cpu float64) -> [[[1., -0., -0., 1.], [0., 1., -0.,
        # 0.], [0., 0., 1., 0.]]], max abs error vs [I | (1, 0, 0)] 6.14e-15; the identity case gives [I | 0] to
        # 8.01e-14.
        if dtype != torch.float64:
            pytest.skip("float64-only pin: float32 recovers [R|t] to 1.06e-05 on mps, outside the float32 atol")
        world_points = self._convention_world_points(device, dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        identity = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], device=device, dtype=dtype
        )
        recovered = kornia.geometry.solve_pnp_dlt(world_points, kornia.geometry.project_points(world_points, K), K)
        assert recovered.shape == (1, 3, 4)
        self.assert_close(recovered, identity)
        shift = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        shifted = kornia.geometry.solve_pnp_dlt(
            world_points, kornia.geometry.project_points(world_points + shift, K), K
        )
        expected = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], device=device, dtype=dtype
        )
        self.assert_close(shifted, expected)

    def test_convention_world_to_camera_translation_sign(self, device, dtype):
        # Convention pin (audit label 5b-pnp-03), the tolerance-free half of the pin above so that the
        # frame-direction claim is covered on every device and not only where float64 exists (mps has none).
        # A camera translated so that cam = world + (1, 0, 0) gives a world-to-camera [R | t] with t = (+1, 0, 0);
        # the cam-to-world reading is t = (-1, 0, 0). The assertion is a sign test, not a value test, so it needs
        # no tolerance and survives the float32 solve on mps, which recovers t only to about 1e-05.
        # Snippet used to generate expected: solve_pnp_dlt(W, project_points(W + [1., 0., 0.], K), K)[0, :, 3]
        # executed 2026-09-06 on the batch-5b worktree (torch 2.14.0) -> cpu float32
        # [1.0000005960464478, 3.45e-06, 5.88e-06]; mps float32 [1.0000009536743164, -2.21e-06, -1.06e-05].
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("solve_pnp_dlt rejects half precision before it solves anything (BaseError)")
        world_points = self._convention_world_points(device, dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        shift = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        translation = kornia.geometry.solve_pnp_dlt(
            world_points, kornia.geometry.project_points(world_points + shift, K), K
        )[0, :, 3]
        assert translation[0] > 0.5
        assert abs(translation[1]) < 0.5
        assert abs(translation[2]) < 0.5

    def test_convention_rejects_4x4_intrinsics(self, device, dtype):
        # Convention pin (no audit label -- the audit probed solve_pnp_dlt's point-count and degeneracy guards,
        # not its intrinsics shape; the executed snippet below is the evidence): ``intrinsics`` is the (B, 3, 3)
        # K, and the (B, 4, 4) intrinsics matrix that a PinholeCamera stores is rejected by the shape check
        # rather than silently truncated to its upper-left block. The positive control is the same call with
        # that upper-left block passed on its own, which solves: so the rejection is about the shape and not
        # about the camera.
        # Snippet used to generate expected: solve_pnp_dlt(W, project_points(W, K), eye(4)[None] with K in the
        # upper-left 3x3) executed 2026-09-06 on the batch-5b worktree (torch 2.14.0) -> ShapeError("Shape
        # mismatch at dimension 1: expected 3, got 4. | Expected shape: ['B', '3', '3'] | Actual shape:
        # [1, 4, 4]") on cpu float32 and float64. In float16 and bfloat16 the earlier dtype validation fires
        # first with a bare BaseError("Validation condition failed"), so those cells are skipped.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("solve_pnp_dlt rejects half precision before the intrinsics shape check (BaseError)")
        world_points = self._convention_world_points(device, dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        img_points = kornia.geometry.project_points(world_points, K)
        K_4x4 = torch.eye(4, device=device, dtype=dtype)[None].clone()
        K_4x4[:, :3, :3] = K
        with pytest.raises(ShapeError, match="expected 3, got 4"):
            kornia.geometry.solve_pnp_dlt(world_points, img_points, K_4x4)
        assert kornia.geometry.solve_pnp_dlt(world_points, img_points, K_4x4[:, :3, :3]).shape == (1, 3, 4)

    def test_convention_planar_world_points_raise(self, device, dtype):
        # Convention pin (audit labels 5b-pnp-04, 5b-pnp-05): the DLT needs a non-degenerate configuration, and
        # the function enforces it -- a coplanar point set (the same six points flattened onto z = 5) raises
        # AssertionError naming the last singular value, rather than returning a silently wrong pose. This is a
        # documented, validated contract, so it is a convention and not a wart.
        # Snippet used to generate expected: solve_pnp_dlt(planar, project_points(planar, K), K) executed
        # 2026-09-06 on the batch-5b worktree (torch 2.14.0, cpu float32 and float64) -> AssertionError("The last
        # singular value of one/more of the elements of the batch is smaller than 0.0001. ..."). In float16 and
        # bfloat16 the earlier dtype validation fires first, so those cells are skipped.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("solve_pnp_dlt rejects half precision before the degeneracy check (BaseError)")
        world_points = self._convention_world_points(device, dtype)
        K = torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        planar = torch.stack(
            [world_points[0, :, 0], world_points[0, :, 1], torch.full_like(world_points[0, :, 0], 5.0)], -1
        )[None]
        with pytest.raises(AssertionError, match="last singular value"):
            kornia.geometry.solve_pnp_dlt(planar, kornia.geometry.project_points(planar, K), K)


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

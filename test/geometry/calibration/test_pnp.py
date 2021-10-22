import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.geometry.calibration.pnp import _mean_isotropic_scale_normalize
from kornia.testing import assert_close, tensor_to_gradcheck_var


class TestSolvePnpDlt:
    @staticmethod
    def _get_samples(shape, low, high, device, dtype):
        """Return a tensor having the given shape and whose values are in the range [low, high)"""
        return ((high - low) * torch.rand(shape, device=device, dtype=dtype)) + low

    @staticmethod
    def _project_to_image(world_points, world_to_cam_4x4, repeated_intrinsics):
        r"""Projects points in the world coordinate system to the image coordinate system.

        Since cam_points will have shape (B, N, 3), repeated_intrinsics should have
        shape (B, N, 3, 3) so that kornia.geometry.project_points can be used.
        """
        cam_points = kornia.geometry.transform_points(world_to_cam_4x4, world_points)
        img_points = kornia.geometry.project_points(cam_points, repeated_intrinsics)

        return img_points

    @staticmethod
    def _get_world_points_and_img_points(cam_points, world_to_cam_4x4, repeated_intrinsics):
        r"""Calculates world_points and img_points.

        Since cam_points will have shape (B, N, 3), repeated_intrinsics should have
        shape (B, N, 3, 3) so that kornia.geometry.project_points can be used.
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
        angle_axis_1 = self._get_samples(shape=(1, 3), low=-tau, high=tau, dtype=dtype, device=device)
        angle_axis_2 = self._get_samples(shape=(1, 3), low=-tau, high=tau, dtype=dtype, device=device)
        rotation_1 = kornia.geometry.angle_axis_to_rotation_matrix(angle_axis_1)
        rotation_2 = kornia.geometry.angle_axis_to_rotation_matrix(angle_axis_2)

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

    @pytest.mark.parametrize("num_points", (6, 20,))
    def test_smoke(self, num_points, device, dtype):

        intrinsics, _, world_points, img_points = self._get_test_data(num_points, device, dtype)
        batch_size = world_points.shape[0]

        pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)
        assert pred_world_to_cam.shape == (batch_size, 3, 4)

    @pytest.mark.parametrize("num_points", (6,))
    def test_gradcheck(self, num_points, device, dtype):

        intrinsics, _, world_points, img_points = self._get_test_data(num_points, device, dtype)

        world_points = tensor_to_gradcheck_var(world_points)
        img_points = tensor_to_gradcheck_var(img_points)
        intrinsics = tensor_to_gradcheck_var(intrinsics)

        assert gradcheck(kornia.geometry.solve_pnp_dlt, (world_points, img_points, intrinsics), raise_exception=True)

    @pytest.mark.parametrize("num_points", (6, 20,))
    def test_pred_world_to_cam(self, num_points, device, dtype):

        intrinsics, gt_world_to_cam, world_points, img_points = self._get_test_data(num_points, device, dtype)

        pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)
        assert_close(pred_world_to_cam, gt_world_to_cam, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("num_points", (6, 20,))
    def test_project(self, num_points, device, dtype):

        intrinsics, _, world_points, img_points = self._get_test_data(num_points, device, dtype)

        pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)

        pred_world_to_cam_4x4 = kornia.eye_like(4, pred_world_to_cam)
        pred_world_to_cam_4x4[:, :3, :] = pred_world_to_cam

        repeated_intrinsics = intrinsics.unsqueeze(1).repeat(1, num_points, 1, 1)
        pred_img_points = self._project_to_image(world_points, pred_world_to_cam_4x4, repeated_intrinsics)

        assert_close(pred_img_points, img_points, atol=1e-3, rtol=1e-3)


class TestNormalization:
    @pytest.mark.parametrize("dimension", (2, 3, 5))
    def test_smoke(self, dimension, device, dtype):

        batch_size = 10
        num_points = 100
        points = torch.rand((batch_size, num_points, dimension), device=device, dtype=dtype)
        points_norm, transform = _mean_isotropic_scale_normalize(points)

        assert points_norm.shape == (batch_size, num_points, dimension)
        assert transform.shape == (batch_size, dimension + 1, dimension + 1)

    @pytest.mark.parametrize("dimension", (2, 3, 5))
    def test_gradcheck(self, dimension, device, dtype):

        batch_size = 3
        num_points = 5
        points = torch.rand((batch_size, num_points, dimension), device=device, dtype=dtype)
        points = tensor_to_gradcheck_var(points)

        assert gradcheck(_mean_isotropic_scale_normalize, (points,), raise_exception=True)

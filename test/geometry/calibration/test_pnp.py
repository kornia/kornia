import torch
import pytest
from torch.autograd import gradcheck

import kornia
from kornia.testing import assert_close, tensor_to_gradcheck_var
from kornia.geometry.conversions import convert_points_to_homogeneous


class TestSolvePnpDlt:

    def _get_samples(self, shape, low, high, device, dtype):
        """Return a tensor having the given shape and whose values are in the range [low, high)"""
        return ((high - low) * torch.rand(shape, device=device, dtype=dtype)) + low

    def _project_to_image(self, world_points, world_to_cam_4x4, intrinsic):
        world_points_h = convert_points_to_homogeneous(world_points)
        cam_points = (torch.matmul(world_to_cam_4x4, world_points_h.T).T)[:, :3]

        temp = torch.matmul(intrinsic, cam_points.T).T
        img_points = temp[:, :2] / temp[:, 2:3]

        return img_points

    def _get_test_data(self, num_points, device, dtype):
        intrinsic = torch.tensor([
            [500.0, 0.0, 250.0],
            [0.0, 500.0, 250.0],
            [0.0, 0.0, 1.0],
        ], dtype=dtype, device=device)

        tau = 2 * 3.141592653589793
        torch.manual_seed(84)

        angle_axis = self._get_samples(
            shape=(1, 3), low=-tau, high=tau, dtype=dtype, device=device,
        )
        translation = self._get_samples(
            shape=(3,), low=-100, high=100, dtype=dtype, device=device,
        )
        world_points = self._get_samples(
            shape=(num_points, 3), low=-100, high=100, dtype=dtype, device=device,
        )

        rotation = kornia.angle_axis_to_rotation_matrix(angle_axis)

        world_to_cam_4x4 = torch.eye(4, dtype=dtype, device=device)
        world_to_cam_4x4[:3, :3] = rotation
        world_to_cam_4x4[:3, 3] = translation

        img_points = self._project_to_image(world_points, world_to_cam_4x4, intrinsic)
        world_to_cam_3x4 = world_to_cam_4x4[:3, :]

        return intrinsic, world_to_cam_3x4, world_points, img_points

    @pytest.mark.parametrize("num_points", (6, 20, 200))
    def test_smoke(self, num_points, device, dtype):
        intrinsic, gt_world_to_cam, world_points, img_points = \
            self._get_test_data(num_points, device, dtype)

        pred_world_to_cam = kornia.solve_pnp_dlt(world_points, img_points, intrinsic)
        assert pred_world_to_cam.shape == (3, 4)

    @pytest.mark.parametrize("num_points", (6, 20, 200))
    def test_gradcheck(self, num_points, device, dtype):
        intrinsic, gt_world_to_cam, world_points, img_points = \
            self._get_test_data(num_points, device, dtype)

        world_points = tensor_to_gradcheck_var(world_points)
        img_points = tensor_to_gradcheck_var(img_points)
        intrinsic = tensor_to_gradcheck_var(intrinsic)

        assert gradcheck(
            kornia.solve_pnp_dlt, (world_points, img_points, intrinsic),
            raise_exception=True, atol=1e-4
        )

    @pytest.mark.parametrize("num_points", (6, 20, 200))
    def test_pred_world_to_cam(self, num_points, device, dtype):
        intrinsic, gt_world_to_cam, world_points, img_points = \
            self._get_test_data(num_points, device, dtype)

        pred_world_to_cam = kornia.solve_pnp_dlt(world_points, img_points, intrinsic)
        assert_close(pred_world_to_cam, gt_world_to_cam, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("num_points", (6, 20, 200))
    def test_project(self, num_points, device, dtype):
        intrinsic, gt_world_to_cam, world_points, img_points = \
            self._get_test_data(num_points, device, dtype)

        pred_world_to_cam = kornia.solve_pnp_dlt(world_points, img_points, intrinsic)

        pred_world_to_cam_4x4 = torch.eye(4, dtype=dtype, device=device)
        pred_world_to_cam_4x4[:3, :] = pred_world_to_cam

        pred_img_points = self._project_to_image(
            world_points, pred_world_to_cam_4x4, intrinsic
        )

        # Different tolerances for dtype torch.float32
        if dtype == torch.float32:
            atol, rtol = 1e-1, 1e-1
        else:
            atol, rtol = 1e-4, 1e-4

        assert_close(pred_img_points, img_points, atol=atol, rtol=rtol)

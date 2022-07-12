import math

import torch

import kornia
from kornia.geometry.nerf.rays import (
    RandomRaySampler,
    UniformRaySampler,
    calc_ray_t_vals,
    cameras_for_ids,
    sample_lengths,
    sample_ray_points,
)
from kornia.testing import assert_close


def create_camera_dimensions(device, dtype):
    n_cams1 = 3
    n_cams2 = 2
    heights: torch.Tensor = torch.cat(
        (
            torch.tensor([200] * n_cams1, device=device, dtype=dtype),
            torch.tensor([100] * n_cams2, device=device, dtype=dtype),
        )
    )
    widths: torch.Tensor = torch.cat(
        (
            torch.tensor([300] * n_cams1, device=device, dtype=dtype),
            torch.tensor([400] * n_cams2, device=device, dtype=dtype),
        )
    )
    num_rays: torch.Tensor = torch.cat(
        (
            torch.tensor([10] * n_cams1, device=device, dtype=dtype),
            torch.tensor([15] * n_cams2, device=device, dtype=dtype),
        )
    )
    return heights, widths, num_rays


class TestRaySampler_2DPoints:
    def test_dimensions_random_sampler(self, device, dtype):
        heights, widths, num_rays = create_camera_dimensions(device, dtype)
        sampler = RandomRaySampler(1, 1)
        points_2d_camera = sampler.sample_points_2d(heights, widths, num_rays)
        assert len(points_2d_camera) == 2
        assert points_2d_camera[10].points_2d.shape == (3, 10, 2)
        assert points_2d_camera[15].points_2d.shape == (2, 15, 2)

    def test_dimensions_uniform_sampler(self, device, dtype):
        heights, widths, _ = create_camera_dimensions(device, dtype)
        sampler = UniformRaySampler(1, 1)
        points_2d_camera = sampler.sample_points_2d(heights, widths)
        assert len(points_2d_camera) == 2
        assert points_2d_camera[60000].points_2d.shape == (3, 60000, 2)
        assert points_2d_camera[40000].points_2d.shape == (2, 40000, 2)


def create_intrinsics(fxs, fys, cxs, cys, device, dtype):
    intrinsics_batch = []
    for fx, fy, cx, cy in zip(fxs, fys, cxs, cys):
        intrinsics = torch.eye(4, device=device, dtype=dtype)
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        intrinsics_batch.append(intrinsics)
    return torch.stack(intrinsics_batch)


def create_extrinsics_with_rotation(alphas, betas, gammas, txs, tys, tzs, device, dtype):
    extrinsics_batch = []
    for alpha, beta, gamma, tx, ty, tz in zip(alphas, betas, gammas, txs, tys, tzs):
        Rx = torch.eye(3, device=device, dtype=dtype)
        Rx[1, 1] = math.cos(alpha)
        Rx[1, 2] = math.sin(alpha)
        Rx[2, 1] = -Rx[1, 2]
        Rx[2, 2] = Rx[1, 1]

        Ry = torch.eye(3, device=device, dtype=dtype)
        Ry[0, 0] = math.cos(beta)
        Ry[0, 2] = -math.sin(beta)
        Ry[2, 0] = -Ry[0, 2]
        Ry[2, 2] = Ry[0, 0]

        Rz = torch.eye(3, device=device, dtype=dtype)
        Rz[0, 0] = math.cos(gamma)
        Rz[0, 1] = math.sin(gamma)
        Rz[1, 0] = -Rz[0, 1]
        Rz[1, 1] = Rz[0, 0]

        Ryz = torch.matmul(Ry, Rz)
        R = torch.matmul(Rx, Ryz)

        extrinsics = torch.eye(4, device=device, dtype=dtype)
        extrinsics[..., 0, -1] = tx
        extrinsics[..., 1, -1] = ty
        extrinsics[..., 2, -1] = tz
        extrinsics[:3, :3] = R

        extrinsics_batch.append(extrinsics)
    return torch.stack(extrinsics_batch)


def create_four_cameras(device, dtype) -> kornia.geometry.camera.PinholeCamera:
    height = torch.tensor([5, 4, 4, 4], device=device, dtype=dtype)
    width = torch.tensor([9, 7, 7, 7], device=device, dtype=dtype)

    fx = width.tolist()
    fy = height.tolist()

    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    tx = [0.0, 0.0, 0.0, 0.0]
    ty = [0.0, 0.0, 0.0, 0.0]
    tz = [11.0, 11.0, 11.0, 11.0]

    pi = torch.pi
    alpha = [pi / 2.0, pi / 2.0, pi / 2.0, 0.0]
    beta = [0.0, 0.0, 0.0, pi]
    gamma = [-pi / 2.0, 0.0, pi / 2.0, 0.0]

    intrinsics = create_intrinsics(fx, fy, cx, cy, device=device, dtype=dtype)
    extrinsics = create_extrinsics_with_rotation(alpha, beta, gamma, tx, ty, tz, device=device, dtype=dtype)

    cameras = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)
    return cameras


class TestRaySampler_3DPoints:
    def test_dimensions_uniform_sampler(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        uniform_sampler_four_cameras = UniformRaySampler(1, 1)
        points_2d_four_cameras = uniform_sampler_four_cameras.sample_points_2d(cameras.height, cameras.width)
        cameras_28 = cameras_for_ids(cameras, points_2d_four_cameras[28].camera_ids)
        points_3d_28 = cameras_28.unproject(points_2d_four_cameras[28].points_2d, 1)
        cameras_40 = cameras_for_ids(cameras, points_2d_four_cameras[45].camera_ids)
        points_3d_40 = cameras_40.unproject(points_2d_four_cameras[45].points_2d, 1)
        assert points_3d_28.shape == (3, 28, 3)
        assert points_3d_40.shape == (1, 45, 3)

    def test_dimensions_ray_params(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        uniform_sampler_four_cameras = UniformRaySampler(1, 2)
        uniform_sampler_four_cameras.calc_ray_params(cameras)
        lengths = sample_lengths(uniform_sampler_four_cameras.origins.shape[0], 10, irregular=True)
        assert uniform_sampler_four_cameras.origins.shape == (3 * 28 + 45, 3)
        assert uniform_sampler_four_cameras.directions.shape == (3 * 28 + 45, 3)
        assert uniform_sampler_four_cameras.camera_ids.shape == (3 * 28 + 45,)
        assert uniform_sampler_four_cameras.points_2d.shape == (3 * 28 + 45, 2)
        assert lengths.shape == (3 * 28 + 45, 10)

    def test_dimensions_sample_ray_points(self, device, dtype):
        cameras = create_four_cameras(device, torch.float32)
        uniform_sampler_four_cameras = UniformRaySampler(1, 2)
        uniform_sampler_four_cameras.calc_ray_params(cameras)
        lengths = sample_lengths(uniform_sampler_four_cameras.origins.shape[0], 10, irregular=True)
        points_3d = sample_ray_points(
            uniform_sampler_four_cameras.origins, uniform_sampler_four_cameras.directions, lengths
        )
        assert points_3d.shape == (3 * 28 + 45, 10, 3)

    def test_t_vals(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        uniform_sampler_four_cameras = UniformRaySampler(2, 3.5)
        uniform_sampler_four_cameras.calc_ray_params(cameras)
        lengths = sample_lengths(uniform_sampler_four_cameras.origins.shape[0], 10, irregular=False)
        points_3d = sample_ray_points(
            uniform_sampler_four_cameras.origins, uniform_sampler_four_cameras.directions, lengths
        )
        t_vals = calc_ray_t_vals(points_3d)
        assert t_vals.shape == (3 * 28 + 45, 10)
        assert_close(t_vals[22, -1], 1.5)  # Testing middle ray

    def test_dimensions_ray_params_in_ndc(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        uniform_sampler_four_cameras = UniformRaySampler(2, 3.5)
        uniform_sampler_four_cameras.calc_ray_params(cameras)
        origins_ndc, directions_ndc = uniform_sampler_four_cameras.transform_ray_params_world_to_ndc(cameras)
        assert origins_ndc.shape == (3 * 28 + 45, 3)
        assert directions_ndc.shape == (3 * 28 + 45, 3)

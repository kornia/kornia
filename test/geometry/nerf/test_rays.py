import math

import torch

import kornia
from kornia.geometry.nerf.rays import RandomRaySampler  # , UniformRaySampler


class TestRandom2DSampler:
    def test_dimensions(self, device, dtype):
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
        sampler = RandomRaySampler(1, 1, 1)
        points_2d_camera = sampler.sample_points_2d(heights, widths, num_rays)
        assert len(points_2d_camera) == 2
        assert points_2d_camera[10].points_2d.shape == (3, 10, 2)
        assert points_2d_camera[15].points_2d.shape == (2, 15, 2)


class TestUniform2DSampler:
    pass


class TestRays:
    def _create_intrinsics(self, fxs, fys, cxs, cys, device, dtype):
        intrinsics_batch = []
        for fx, fy, cx, cy in zip(fxs, fys, cxs, cys):
            intrinsics = torch.eye(4, device=device, dtype=dtype)
            intrinsics[0, 0] = fx
            intrinsics[1, 1] = fy
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            intrinsics_batch.append(intrinsics)
        return torch.stack(intrinsics_batch)

    def _create_extrinsics_with_rotation(self, alphas, betas, gammas, txs, tys, tzs, device, dtype):
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

    def _create_four_cameras(self, device, dtype) -> kornia.geometry.camera.PinholeCamera:
        height = torch.tensor([4, 4, 4, 4])
        width = torch.tensor([6, 6, 6, 6])

        fx = [1.0, 1.0, 1.0, 1.0]
        fy = [1.0, 1.0, 1.0, 1.0]
        cx = [width[0] / 2.0, width[1] / 2.0, width[2] / 2.0, width[3] / 2.0]
        cy = [height[0] / 2.0, height[1] / 2.0, height[2] / 2.0, height[3] / 2.0]

        tx = [1.0, 1.0, 1.0, 1.0]
        ty = [0.0, 0.0, 0.0, 0.0]
        tz = [0.0, 0.0, 0.0, 0.0]
        pi = torch.pi
        alpha = [0.0, 0.0, 0.0, 0.0]
        beta = [0.0, 0.0, 0.0, pi / 2.0]
        gamma = [pi, -pi / 2.0, 0.0, 0.0]

        intrinsics = self._create_intrinsics(fx, fy, cx, cy, device=device, dtype=dtype)
        extrinsics = self._create_extrinsics_with_rotation(alpha, beta, gamma, tx, ty, tz, device=device, dtype=dtype)

        cameras = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, height, width)
        return cameras

    def test_rays(self, device, dtype):
        pass
        # num_rays: torch.Tensor = torch.ones(4, device=device, dtype=torch.int) * 5

        # cameras = self._create_four_cameras(device=device, dtype=dtype)

        # ray_sampler = RandomRaySampler()

        # rays = Rays(
        #     cameras, ray_sampler, num_rays, 1.0, 2.0, 10
        # )  # FIXME: num_rays should be a property of (random) sampler, and not sent here to Rays
        # print(rays)

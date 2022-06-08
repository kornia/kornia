import torch

import kornia
from kornia.geometry.nerf.rays import RandomRaySampler, Rays  # , RaySampler, UniformRaySampler


def test_ray_sampler(self, device, dtype):
    n_cams1 = 3
    n_cams2 = 2
    heights: torch.Tensor = torch.cat((torch.tensor([200] * n_cams1), torch.tensor([100] * n_cams2)))
    widths: torch.Tensor = torch.cat((torch.tensor([300] * n_cams1), torch.tensor([400] * n_cams2)))
    num_rays: torch.Tensor = torch.cat((torch.tensor([10] * n_cams1), torch.tensor([15] * n_cams2)))
    sampler = RandomRaySampler()
    sampler.sample_points(heights, widths, num_rays)
    assert len(sampler.points_2d_camera) == 2
    assert sampler.points_2d_camera[10].points_2d.shape == (3, 10, 2)
    assert sampler.points_2d_camera[15].points_2d.shape == (2, 15, 2)


def test_rays(self, device, dtype):
    batch_size = 5
    n = 2  # Points per batch
    height, width = 4, 6
    fx, fy, cx, cy = 1, 2, width / 2, height / 2
    alpha, beta, gamma = 0.0, 0.0, 0.4
    tx, ty, tz = 0, 0, 3

    intrinsics = self._create_intrinsics(batch_size, fx, fy, cx, cy, device=device, dtype=dtype)
    extrinsics = self._create_extrinsics_with_rotation(
        batch_size, alpha, beta, gamma, tx, ty, tz, device=device, dtype=dtype
    )

    heights = torch.ones(batch_size, device=device, dtype=dtype) * height
    widths = torch.ones(batch_size, device=device, dtype=dtype) * width
    num_rays: torch.Tensor = torch.ones(batch_size, device=device, dtype=torch.int) * n

    cameras = kornia.geometry.camera.PinholeCamera(intrinsics, extrinsics, heights, widths)

    ray_sampler = RandomRaySampler()

    rays = Rays(
        cameras, ray_sampler, num_rays
    )  # FIXME: num_rays should be a property of (random) sampler, and not sent here to Rays
    print(rays)

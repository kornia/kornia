import torch

from kornia.geometry.nerf.rays import RandomRaySampler  # , RaySampler, UniformRaySampler


def test_ray_sampler():
    n_cams = 5
    heights: torch.Tensor = torch.Tensor([200] * n_cams)
    widths: torch.Tensor = torch.Tensor([300] * n_cams)
    num_rays: torch.Tensor = torch.Tensor([10] * n_cams)
    sampler = RandomRaySampler(heights, widths, num_rays)
    sampler.sample_points()

import torch

from kornia.geometry.nerf.rays import RandomRaySampler  # , RaySampler, UniformRaySampler


def test_ray_sampler():
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

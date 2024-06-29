from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from kornia.core import Tensor


@torch.no_grad()
def sample_keypoints(
    scoremap: Tensor, num_samples: Optional[int] = 10_000, return_scoremap: bool = True, increase_coverage: bool = True
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    device = scoremap.device
    dtype = scoremap.dtype
    B, H, W = scoremap.shape
    if increase_coverage:
        weights = (-(torch.linspace(-2, 2, steps=51, device=device, dtype=dtype) ** 2)).exp()[None, None]
        # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
        local_density_x = F.conv2d((scoremap[:, None] + 1e-6) * 10000, weights[..., None, :], padding=(0, 51 // 2))
        local_density = F.conv2d(local_density_x, weights[..., None], padding=(51 // 2, 0))[:, 0]
        scoremap = scoremap * (local_density + 1e-8) ** (-1 / 2)
    grid = get_grid(B, H, W, device=device).reshape(B, H * W, 2)
    inds = torch.topk(scoremap.reshape(B, H * W), k=num_samples).indices  # type: ignore
    kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_samples, 2))  # type: ignore
    if return_scoremap:
        return kps, torch.gather(scoremap.reshape(B, H * W), dim=1, index=inds)
    return kps


def get_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    x1_n_ = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n_[2], x1_n_[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


def dedode_denormalize_pixel_coordinates(flow: torch.Tensor, h: int, w: int) -> torch.Tensor:
    flow = torch.stack(
        (
            w * (flow[..., 0] + 1) / 2,
            h * (flow[..., 1] + 1) / 2,
        ),
        dim=-1,
    )
    return flow

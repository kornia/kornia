from typing import Tuple

import torch

from kornia.core import Tensor


@torch.no_grad()
def warp_kpts(
    kpts0: Tensor, depth0: Tensor, depth1: Tensor, T_0to1: Tensor, K0: Tensor, K1: Tensor
) -> Tuple[Tensor, Tensor]:
    """Warp kpts0 from I0 to I1 with depth, K and Rt Also check covisibility and depth consistency. Depth is
    consistent if relative error < 0.2 (hard-coded).

    Args:
        kpts0: [N, L, 2] - <x, y>,
        depth0: [N, H, W],
        depth1: [N, H, W],
        T_0to1: [N, 3, 4],
        K0: [N, 3, 3],
        K1: [N, 3, 3],
    Returns:
        calculable_mask: [N, L]
        warped_keypoints0: [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w - 1) * (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0

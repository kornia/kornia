from __future__ import annotations

import torch

import kornia.geometry.epipolar as epi
from kornia.core import Device, Dtype, Tensor, tensor
from testing.laf import create_random_homography


def create_rectified_fundamental_matrix(batch_size: int, dtype: Dtype = None, device: Device = None) -> Tensor:
    """Create a batch of rectified fundamental matrices of shape Bx3x3."""
    F_rect = tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype).view(1, 3, 3)
    F_repeat = F_rect.expand(batch_size, 3, 3)
    return F_repeat


def create_random_fundamental_matrix(
    batch_size: int, std_val: float = 1e-3, dtype: Dtype = None, device: Device = None
) -> Tensor:
    """Create a batch of random fundamental matrices of shape Bx3x3."""
    F_rect = create_rectified_fundamental_matrix(batch_size, dtype, device)
    H_left = create_random_homography(F_rect, 3, std_val)
    H_right = create_random_homography(F_rect, 3, std_val)
    return H_left.permute(0, 2, 1) @ F_rect @ H_right


def generate_two_view_random_scene(
    device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    num_views: int = 2
    num_points: int = 30

    scene: dict[str, torch.Tensor] = epi.generate_scene(num_views, num_points)

    # internal parameters (same K)
    K1 = scene["K"].to(device, dtype)
    K2 = K1.clone()

    # rotation
    R1 = scene["R"][0:1].to(device, dtype)
    R2 = scene["R"][1:2].to(device, dtype)

    # translation
    t1 = scene["t"][0:1].to(device, dtype)
    t2 = scene["t"][1:2].to(device, dtype)

    # projection matrix, P = K(R|t)
    P1 = scene["P"][0:1].to(device, dtype)
    P2 = scene["P"][1:2].to(device, dtype)

    # fundamental matrix
    F_mat = epi.fundamental_from_projections(P1[..., :3, :], P2[..., :3, :])

    F_mat = epi.normalize_transformation(F_mat)

    # points 3d
    X = scene["points3d"].to(device, dtype)

    # projected points
    x1 = scene["points2d"][0:1].to(device, dtype)
    x2 = scene["points2d"][1:2].to(device, dtype)

    return {
        "K1": K1,
        "K2": K2,
        "R1": R1,
        "R2": R2,
        "t1": t1,
        "t2": t2,
        "P1": P1,
        "P2": P2,
        "F": F_mat,
        "X": X,
        "x1": x1,
        "x2": x2,
    }

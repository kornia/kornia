from typing import Dict

import torch

import kornia.geometry.epipolar as epi


def generate_two_view_random_scene(
    device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    num_views: int = 2
    num_points: int = 30

    scene: Dict[str, torch.Tensor] = epi.generate_scene(num_views, num_points)

    # internal parameters (same K)
    K1 = scene['K'].to(device, dtype)
    K2 = K1.clone()

    # rotation
    R1 = scene['R'][0:1].to(device, dtype)
    R2 = scene['R'][1:2].to(device, dtype)

    # translation
    t1 = scene['t'][0:1].to(device, dtype)
    t2 = scene['t'][1:2].to(device, dtype)

    # projection matrix, P = K(R|t)
    P1 = scene['P'][0:1].to(device, dtype)
    P2 = scene['P'][1:2].to(device, dtype)

    # fundamental matrix
    F_mat = epi.fundamental_from_projections(P1[..., :3, :], P2[..., :3, :])

    F_mat = epi.normalize_transformation(F_mat)

    # points 3d
    X = scene['points3d'].to(device, dtype)

    # projected points
    x1 = scene['points2d'][0:1].to(device, dtype)
    x2 = scene['points2d'][1:2].to(device, dtype)

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

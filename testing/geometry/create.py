# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch

import kornia.geometry.epipolar as epi
from kornia.core.ops import eye_like


def create_random_homography(data: torch.Tensor, eye_size: int, std_val: float = 1e-3) -> torch.Tensor:
    """Create a batch of random homographies of shape Bx3x3."""
    std = torch.zeros(data.shape[0], eye_size, eye_size, device=data.device, dtype=data.dtype)
    eye = eye_like(eye_size, data)
    return eye + std.uniform_(-std_val, std_val)


def create_rectified_fundamental_matrix(
    batch_size: int, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """Create a batch of rectified fundamental matrices of shape Bx3x3."""
    F_rect = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype).view(
        1, 3, 3
    )
    return F_rect.expand(batch_size, 3, 3)


def create_random_fundamental_matrix(
    batch_size: int,
    std_val: float = 1e-3,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Create a batch of random fundamental matrices of shape Bx3x3."""
    F_rect = create_rectified_fundamental_matrix(batch_size, dtype, device)
    H_left = create_random_homography(F_rect, 3, std_val)
    H_right = create_random_homography(F_rect, 3, std_val)
    return H_left.permute(0, 2, 1) @ F_rect @ H_right


def generate_two_view_random_scene(
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    if device is None:
        device = torch.device("cpu")
    num_views: int = 2
    num_points: int = 30

    with torch.random.fork_rng():
        torch.manual_seed(4886)
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


def create_dominant_plane_scene(
    num_points: int,
    inlier_ratio: float,
    plane_fraction: float,
    seed: int,
    zoom: float = 1.0,
    skew: float = 0.0,
    noise: float = 0.5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two views of a scene where ``plane_fraction`` of the inliers lie on one plane, DEGENSAC's regime.

    Camera 1 has a focal length of 800 px and principal point (320, 240). Camera 2 is rotated 8 degrees about y and
    3 about x, translated by (-1, 0.1, 0.15), with focal length ``800 * zoom`` and skew ``800 * zoom * skew``. Plane
    points lie on z = 6 + 0.2 x; off-plane points at depths 2.5 to 10.5. Inliers get Gaussian noise of ``noise`` px in
    both images; outliers are uniform in 640 x 480. Generated in float64 on the CPU from ``seed``, then shuffled.

    Returns:
        ``kp1``, ``kp2`` ``(N, 2)``; ``labels`` ``(N,)``: 0 plane, 1 off-plane inlier, 2 outlier; and the noise-free
        off-plane points of both images ``(M, 2)``, to measure whether an estimate explains the off-plane geometry.
    """
    f64 = torch.float64
    generator = torch.Generator().manual_seed(seed)

    def rotation(axis: str, degrees: float) -> torch.Tensor:
        c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        if axis == "y":
            return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=f64)
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=f64)

    K1 = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=f64)
    K2 = torch.tensor(
        [[800.0 * zoom, 800.0 * zoom * skew, 320.0], [0.0, 800.0 * zoom, 240.0], [0.0, 0.0, 1.0]], dtype=f64
    )
    R = rotation("y", 8.0) @ rotation("x", 3.0)
    t = torch.tensor([-1.0, 0.1, 0.15], dtype=f64)

    def project(K: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        x = X @ K.T
        return x[:, :2] / x[:, 2:]

    num_inliers = int(num_points * inlier_ratio)
    num_plane = int(num_inliers * plane_fraction)
    num_off = num_inliers - num_plane
    xy = (torch.rand(num_plane, 2, generator=generator, dtype=f64) - 0.5) * torch.tensor([4.0, 3.0], dtype=f64)
    plane = torch.cat([xy, 6.0 + 0.2 * xy[:, :1]], 1)
    depth = 2.5 + 8.0 * torch.rand(num_off, 1, generator=generator, dtype=f64)
    ray = (torch.rand(num_off, 2, generator=generator, dtype=f64) - 0.5) * torch.tensor([0.6, 0.45], dtype=f64)
    off = torch.cat([ray * depth, depth], 1) + torch.tensor([0.3, 0.0, 0.0], dtype=f64)
    X = torch.cat([plane, off])
    clean1, clean2 = project(K1, X), project(K2, X @ R.T + t)
    kp1 = clean1 + noise * torch.randn(clean1.shape, generator=generator, dtype=f64)
    kp2 = clean2 + noise * torch.randn(clean2.shape, generator=generator, dtype=f64)
    num_outliers = num_points - num_inliers
    size = torch.tensor([640.0, 480.0], dtype=f64)
    kp1 = torch.cat([kp1, torch.rand(num_outliers, 2, generator=generator, dtype=f64) * size])
    kp2 = torch.cat([kp2, torch.rand(num_outliers, 2, generator=generator, dtype=f64) * size])
    labels = torch.cat([torch.zeros(num_plane), torch.ones(num_off), torch.full((num_outliers,), 2.0)]).long()
    order = torch.randperm(num_points, generator=generator)
    return (
        kp1[order].to(device, dtype),
        kp2[order].to(device, dtype),
        labels[order].to(device),
        clean1[num_plane:].to(device, dtype),
        clean2[num_plane:].to(device, dtype),
    )

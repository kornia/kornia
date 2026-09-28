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

from typing import Dict

import torch

from kornia.geometry.conversions import axis_angle_to_rotation_matrix
from kornia.geometry.epipolar import projection_from_KRt


def two_view_scene(device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Asymmetric two-view scene built in ``dtype`` from literals.

    ``K1 != K2``, ``fx != fy``, ``cx != cy``, a rotation about a non-axis direction and a translation off every
    axis, so swapping the images, transposing a matrix or negating ``t`` changes every quantity derived from it.
    Twelve points at depth 4 to 6 in the first camera, no two sharing a coordinate. ``x1``/``x2`` are exact pixel
    projections through ``P1 = K1 [I | 0]`` and ``P2 = K2 [R | t]``; ``(R, t)`` is world-to-camera for camera 2.
    """
    K1 = torch.tensor([[[800.0, 0.0, 320.0], [0.0, 760.0, 200.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
    K2 = torch.tensor([[[700.0, 0.0, 300.0], [0.0, 740.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
    R = axis_angle_to_rotation_matrix(torch.tensor([[0.1, -0.2, 0.05]], device=device, dtype=dtype))
    t = torch.tensor([[[0.5], [0.1], [0.02]]], device=device, dtype=dtype)
    X = torch.tensor(
        [
            [
                [-0.0075, 0.5364, 4.4163],
                [-0.7359, -0.3852, 5.8596],
                [-0.0198, 0.7929, 5.4462],
                [0.2646, -0.3022, 5.4847],
                [-0.9553, -0.6623, 5.0526],
                [0.0370, 0.3953, 4.4873],
                [-0.6779, -0.4355, 5.1692],
                [0.8304, -0.2058, 4.0663],
                [-0.1612, 0.1058, 4.2774],
                [-0.9277, -0.6295, 4.4845],
                [-0.3898, 0.8640, 5.6309],
                [-0.4603, -0.6986, 5.5863],
            ]
        ],
        device=device,
        dtype=dtype,
    )
    eye = torch.eye(3, device=device, dtype=dtype)[None]
    P1 = projection_from_KRt(K1, eye, torch.zeros(1, 3, 1, device=device, dtype=dtype))
    P2 = projection_from_KRt(K2, R, t)

    def proj(P: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        Xh = torch.cat([X, torch.ones_like(X[..., :1])], -1)
        x = (P @ Xh.transpose(1, 2)).transpose(1, 2)
        return x[..., :2] / x[..., 2:]

    return {"K1": K1, "K2": K2, "R": R, "t": t, "X": X, "P1": P1, "P2": P2, "x1": proj(P1, X), "x2": proj(P2, X)}


def calibrated_two_view_scene(
    n: int, noise: float, seed: int, dtype: torch.dtype = torch.float64
) -> Dict[str, torch.Tensor]:
    """Seeded two-view scene in normalized camera coordinates, for essential-matrix tests.

    ``n`` points at depths around 4 in front of both cameras, a small rotation and a unit translation off every axis;
    ``x2`` carries Gaussian noise of standard deviation ``noise`` (calibrated units). ``E = [t]_x R`` is scaled to unit
    Frobenius norm, so that ``x2^T E x1 = 0`` for the noise-free projections.
    """
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, generator=generator, dtype=dtype) * torch.tensor([1.0, 1.0, 0.5], dtype=dtype)
    X = X + torch.tensor([0.0, 0.0, 4.0], dtype=dtype)
    R = axis_angle_to_rotation_matrix(torch.tensor([[0.05, -0.1, 0.03]], dtype=dtype))[0]
    t = torch.nn.functional.normalize(torch.tensor([1.0, 0.2, 0.1], dtype=dtype), dim=0)
    Y = X @ R.T + t
    x1, x2 = X[:, :2] / X[:, 2:], Y[:, :2] / Y[:, 2:]
    x2 = x2 + noise * torch.randn(n, 2, generator=generator, dtype=dtype)
    zero = torch.zeros((), dtype=dtype)
    tx = torch.stack(
        [torch.stack([zero, -t[2], t[1]]), torch.stack([t[2], zero, -t[0]]), torch.stack([-t[1], t[0], zero])]
    )
    E = tx @ R
    return {"x1": x1, "x2": x2, "E": E / E.norm(), "R": R, "t": t}

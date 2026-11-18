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


# A result computed in float32 and rounded once to a half dtype is within half a unit in the last place of the float64
# result of the same rounded inputs, so within 0.5 eps of its largest entry; 1/64 eps covers the float32 arithmetic.
ONE_ROUNDING_EPS = 0.5 + 2**-6


def max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the largest error of any batch element, relative to that element's largest ``expected`` entry.

    ``actual`` may be on any device and in any dtype; ``expected`` is a float64 CPU tensor of the same shape.
    """
    error = (actual.cpu().double() - expected).abs().flatten(1).amax(-1)
    return (error / expected.abs().flatten(1).amax(-1)).max().item()


def pixel_two_view_batch(batch: int, num_points: int, seed: int) -> Dict[str, torch.Tensor]:
    """Seeded batch of two-view geometries in pixels, built on the CPU in float64.

    Each element has a rotation with axis-angle components of standard deviation 0.2 rad, a unit translation,
    ``E = [t]_x R``, two cameras with focal lengths of 900 to 1100 px and principal points within 20 px of (640, 360),
    ``F = K2^{-T} E K1^{-1}``, and ``num_points`` first-image points ``x1`` spread uniformly over a 1280 x 720 image.
    For accuracy tests that round these inputs to a lower-precision dtype.
    """
    f64 = torch.float64
    generator = torch.Generator().manual_seed(seed)
    R = axis_angle_to_rotation_matrix(0.2 * torch.randn(batch, 3, generator=generator, dtype=f64))
    t = torch.nn.functional.normalize(torch.randn(batch, 3, generator=generator, dtype=f64), dim=-1)
    zero = torch.zeros(batch, dtype=f64)
    tx = torch.stack(
        [
            torch.stack([zero, -t[:, 2], t[:, 1]], -1),
            torch.stack([t[:, 2], zero, -t[:, 0]], -1),
            torch.stack([-t[:, 1], t[:, 0], zero], -1),
        ],
        -2,
    )
    E = tx @ R

    def camera() -> torch.Tensor:
        focal = 900.0 + 200.0 * torch.rand(batch, 2, generator=generator, dtype=f64)
        center = torch.tensor([640.0, 360.0], dtype=f64) + 40.0 * torch.rand(batch, 2, generator=generator, dtype=f64)
        center = center - 20.0
        K = torch.zeros(batch, 3, 3, dtype=f64)
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = focal[:, 0], focal[:, 1], *center.unbind(-1), 1.0
        return K

    K1, K2 = camera(), camera()
    F = torch.linalg.inv(K2).mT @ E @ torch.linalg.inv(K1)
    image = torch.tensor([1280.0, 720.0], dtype=f64)
    x1 = torch.rand(batch, num_points, 2, generator=generator, dtype=f64) * image
    return {"E": E, "F": F, "K1": K1, "K2": K2, "x1": x1}


def affine_two_view_scene(device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Two affine cameras and twelve correspondences, built in ``dtype`` from literals.

    The cameras ``x = M X + t`` (``M1 = [[1.2, 0.1, -0.3], [0.05, -1.1, 0.4]]``, ``t1 = [320, 240]``,
    ``M2 = [[0.9, 0.5, 0.2], [-0.3, -0.8, 0.7]]``, ``t2 = [300, 260]``) see twelve non-coplanar points spread over
    about 190 units. ``x1``/``x2`` are their exact pixel projections. ``F`` is the affine fundamental matrix of the
    cameras, with ``[a, b] M2 + [c, d] M1 = 0`` and ``e = -([a, b] t2 + [c, d] t1)`` for
    ``F = [[0, 0, a], [0, 0, b], [c, d, e]]``, scaled so that ``F[2, 2] = 1``. ``noise`` holds fixed offsets
    ``[dx1, dy1, dx2, dy2]`` and ``F_gold`` the minimizer of the summed squared geometric distances of the noisy
    correspondences, found by a general-purpose optimizer (``scipy.optimize.minimize`` on
    ``sum(r**2) / (a**2 + b**2 + c**2 + d**2)`` with ``r = a x2 + b y2 + c x1 + d y1 + e``). Generated by::

        X = np.array([[-80, 35, 12], [64, -71, 40], [-23, 90, -55], [47, 18, 73], [-95, -42, -8], [12, 58, 27],
                      [81, -9, -66], [-56, -77, 51], [30, 44, -90], [-7, -60, 15], [93, 71, 5], [-38, 6, 84]], float)
        x1, x2 = X @ M1.T + t1, X @ M2.T + t2
        n = np.linalg.svd(np.c_[M2.T, M1.T])[2][-1]  # [a, b, c, d]
        F = np.zeros((3, 3))
        F[0, 2], F[1, 2], F[2, 0], F[2, 1], F[2, 2] = *n, -(n[:2] @ t2 + n[2:] @ t1)
        F /= F[2, 2]
        x1n, x2n = x1 + noise[:, :2], x2 + noise[:, 2:]
        cost = lambda p: np.sum((p[0] * x2n[:, 0] + p[1] * x2n[:, 1] + p[2] * x1n[:, 0] + p[3] * x1n[:, 1] + p[4])
                                ** 2) / (p[:4] ** 2).sum()
        p = minimize(cost, np.array([F[0, 2], F[1, 2], F[2, 0], F[2, 1], 1.0]) * 1.03, method="Nelder-Mead",
                     options=dict(xatol=1e-14, fatol=1e-16, maxiter=200000, maxfev=200000)).x
        p = minimize(cost, p, method="BFGS", options=dict(gtol=1e-14)).x
        F_gold = np.zeros((3, 3))
        F_gold[0, 2], F_gold[1, 2], F_gold[2, 0], F_gold[2, 1], F_gold[2, 2] = p / p[4]
    """
    x1 = torch.tensor(
        [
            [
                [223.9, 202.3], [377.7, 337.3], [317.9, 117.85], [356.3, 251.75], [204.2, 278.25], [332.1, 187.6],
                [436.1, 227.55], [229.8, 342.3], [387.4, 157.1], [301.1, 311.65], [437.2, 168.55], [249.8, 265.1],
            ]
        ],
        device=device,
        dtype=dtype,
    )  # fmt: skip
    x2 = torch.tensor(
        [
            [
                [247.9, 264.4], [330.1, 325.6], [313.3, 156.4], [365.9, 282.6], [191.9, 316.5], [345.2, 228.9],
                [355.2, 196.7], [221.3, 374.1], [331.0, 152.8], [266.7, 320.6], [420.2, 178.8], [285.6, 325.4],
            ]
        ],
        device=device,
        dtype=dtype,
    )  # fmt: skip
    noise = torch.tensor(
        [
            [
                [0.31, -0.12, -0.25, 0.4], [-0.44, 0.07, 0.18, -0.29], [0.05, 0.36, -0.41, 0.12],
                [0.22, -0.33, 0.09, 0.27], [-0.15, 0.24, 0.38, -0.06], [0.42, 0.11, -0.2, -0.37],
                [-0.3, -0.27, 0.14, 0.33], [0.08, 0.45, -0.1, -0.21], [-0.36, -0.04, 0.29, 0.16],
                [0.19, -0.4, -0.32, 0.03], [-0.07, 0.2, 0.44, -0.39], [0.35, -0.18, -0.03, 0.25],
            ]
        ],
        device=device,
        dtype=dtype,
    )  # fmt: skip
    F = torch.tensor(
        [
            [
                [0.0, 0.0, 0.005968045112781947],
                [0.0, 0.0, -0.010385338345864652],
                [-0.007471804511278188, 0.009586466165413523, 1.0],
            ]
        ],
        device=device,
        dtype=dtype,
    )
    F_gold = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0059939486403256526],
                [0.0, 0.0, -0.010334260027148275],
                [-0.007497428822287553, 0.009532615627438538, 1.0],
            ]
        ],
        device=device,
        dtype=dtype,
    )
    return {"x1": x1, "x2": x2, "noise": noise, "F": F, "F_gold": F_gold}

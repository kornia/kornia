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

"""Module including useful metrics for Structure from Motion."""

import math

import torch
from torch import Tensor, ones_like

from kornia.core.check import KORNIA_CHECK_IS_TENSOR
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar.fundamental import _epipolar_design_rows
from kornia.geometry.linalg import point_line_distance


def _sampson_epipolar_distance_manual_impl_(
    pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool = True, eps: float = 1e-8
) -> Tensor:
    """Return Sampson distance for correspondences given the fundamental matrix.

    Args:
        pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed Sampson distance with shape :math:`(*, N)`.

    """
    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")

    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    # Extract coords; support 2D (w=1) and 3D homogeneous
    x = pts1[..., :, 0]
    y = pts1[..., :, 1]
    u = pts2[..., :, 0]
    v = pts2[..., :, 1]
    # homogeneous weights with correct dtype/shape
    w1 = pts1[..., :, 2] if pts1.shape[-1] == 3 else ones_like(x)
    w2 = pts2[..., :, 2] if pts2.shape[-1] == 3 else ones_like(u)

    # Grab F entries and add a length-1 axis to broadcast across N
    f00 = Fm[..., 0, 0][..., None]
    f01 = Fm[..., 0, 1][..., None]
    f02 = Fm[..., 0, 2][..., None]
    f10 = Fm[..., 1, 0][..., None]
    f11 = Fm[..., 1, 1][..., None]
    f12 = Fm[..., 1, 2][..., None]
    f20 = Fm[..., 2, 0][..., None]
    f21 = Fm[..., 2, 1][..., None]
    f22 = Fm[..., 2, 2][..., None]

    # Fx = F @ [x,y,w1]
    Fx0 = f00 * x + f01 * y + f02 * w1
    Fx1 = f10 * x + f11 * y + f12 * w1
    Fx2 = f20 * x + f21 * y + f22 * w1

    # (F^T x')_{1:2} for x' = [u,v,w2]
    # (first two coordinates only)
    Ft0 = f00 * u + f10 * v + f20 * w2  # (F^T x')_0
    Ft1 = f01 * u + f11 * v + f21 * w2  # (F^T x')_1

    # Numerator: (x'^T F x)^2 = (u*Fx0 + v*Fx1 + w2*Fx2)^2
    num = (u * Fx0 + v * Fx1 + w2 * Fx2) ** 2
    # Denominator: ||(F x)_{1:2}||^2 + ||(F^T x')_{1:2}||^2
    den = Fx0 * Fx0 + Fx1 * Fx1 + Ft0 * Ft0 + Ft1 * Ft1 + eps
    out: Tensor = num / den
    if squared:
        return out
    return (out + eps).sqrt()


def _sampson_epipolar_distance_matmul_impl_(
    pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool = True, eps: float = 1e-8
) -> Tensor:
    """Return Sampson distance for correspondences given the fundamental matrix.

    Args:
        pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed Sampson distance with shape :math:`(*, N)`.

    """
    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)

    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)

    # From Hartley and Zisserman, Sampson error (11.9)
    # sam =  (x'^T F x) ** 2 / (  (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2)) )

    # line1_in_2 = (F @ pts1.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)
    # line2_in_1 = (F.transpose(dim0=-2, dim1=-1) @ pts2.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)

    # Instead we can just transpose F once and switch the order of multiplication
    F_t: Tensor = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2: Tensor = pts1 @ F_t
    line2_in_1: Tensor = pts2 @ Fm

    # numerator = (x'^T F x) ** 2
    numerator: Tensor = (pts2 * line1_in_2).sum(dim=-1).pow(2)

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator: Tensor = line1_in_2[..., :2].norm(2, dim=-1).pow(2) + line2_in_1[..., :2].norm(2, dim=-1).pow(2)
    out: Tensor = numerator / denominator
    if squared:
        return out
    return (out + eps).sqrt()


# Shared-point Sampson distances take the GEMM path on CPU from two models on. On CUDA both paths are launch-bound
# below about 2**16 model-point pairs, where the GEMM path's extra kernels cost ~40 us, and it wins above
# (2048 models x 1000 points: 718 -> 261 us; RTX 4090, torch 2.14).
_SAMPSON_SHARED_MIN_PAIRS_ACCELERATOR = 2**16


def _shares_points(pts1: Tensor, pts2: Tensor) -> bool:
    """Whether ``pts1`` and ``pts2`` are one set of correspondences: leading dimensions of size 1, equal counts."""
    return math.prod(pts1.shape[:-2]) == 1 and math.prod(pts2.shape[:-2]) == 1 and pts1.shape[-2] == pts2.shape[-2]


def _sampson_errors(F: Tensor, x1: Tensor, x2: Tensor, eps: float) -> Tensor:
    """Squared Sampson distances ``(M, N)`` of fundamental matrices ``(M, 3, 3)`` on homogeneous points ``(N, 3)``.

    The line ``F x1`` and the first two components of ``F^T x2`` of every model come from ``(3M, 3) @ (3, N)`` and
    ``(2M, 3) @ (3, N)`` products; the residual ``x2 . (F x1)`` and the squared gradient norm are then formed in the
    order of the manual implementation, and ``eps`` is added to the denominator. Expanding either into monomials of
    the coordinates, as :func:`_sampson_quadratic_basis` does, cancels near the epipoles: on pixel-scale float32 points
    0.01-1000 px from both epipoles an expanded denominator returned negative and infinite distances, and for
    ``F = [[1, 0, -1000], [0, 1, -1000], [-1000, -1000, 2e6]]``, ``x1 = (1000.1, 1000.2)``, ``x2 = (1000.3, 1000.4)``
    an expanded residual gives 0.0521 where this gives the manual path's 0.0319 (float64: 0.0403).
    """
    m, n = F.shape[0], x1.shape[0]
    lines = (F.reshape(3 * m, 3) @ x1.T).view(m, 3, n)
    lines_t = (F.mT[:, :2].reshape(2 * m, 3) @ x2.T).view(m, 2, n)
    residual = x2[:, 0] * lines[:, 0] + x2[:, 1] * lines[:, 1] + x2[:, 2] * lines[:, 2]
    denominator = lines[:, 0].square() + lines[:, 1].square() + lines_t[:, 0].square() + lines_t[:, 1].square() + eps
    return residual.square() / denominator


def _sampson_epipolar_distance_shared_impl_(
    pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool, eps: float, denominator_eps: float
) -> Tensor:
    """Sampson distances of many fundamental matrices on one set of correspondences, by :func:`_sampson_errors`.

    ``pts1`` and ``pts2`` have leading dimensions of size 1; the result has the broadcast shape ``(*, N)``. Half
    precision is computed in float32 and returned in the input dtype. ``denominator_eps`` is ``eps``, or 0 where the
    CUDA matmul implementation would have run (#4881).
    """
    num_points = pts1.shape[-2]
    dtype = torch.promote_types(torch.promote_types(pts1.dtype, pts2.dtype), Fm.dtype)
    work = torch.promote_types(dtype, torch.float32)
    x1 = pts1.reshape(num_points, pts1.shape[-1]).to(work)
    x2 = pts2.reshape(num_points, pts2.shape[-1]).to(work)
    if x1.shape[-1] == 2:
        x1 = convert_points_to_homogeneous(x1)
    if x2.shape[-1] == 2:
        x2 = convert_points_to_homogeneous(x2)
    out = _sampson_errors(Fm.reshape(-1, 3, 3).to(work), x1, x2, denominator_eps)
    if not squared:
        out = (out + eps).sqrt()
    shape = torch.broadcast_shapes(pts1.shape[:-2], pts2.shape[:-2], Fm.shape[:-2])
    return out.reshape(*shape, num_points).to(dtype)


def _sampson_quadratic_basis(x1: Tensor, x2: Tensor) -> Tensor:
    """Per-correspondence monomials ``(27, 2N)`` of the Sampson distance, for RANSAC's sampling loop only.

    ``[vec F, vec Q1, vec Q2] @ basis`` gives the epipolar residuals and the squared gradient norms of every model in
    one product, with ``Q1 = F[:2]^T F[:2]`` and ``Q2 = F[:, :2] F[:, :2]^T``: ``x2^T F x1`` is linear in
    ``vec(x2 x1^T)``, and ``|F[:2] x1|^2 + |F[:, :2]^T x2|^2 = x1^T Q1 x1 + x2^T Q2 x2`` in ``vec(x1 x1^T)`` and
    ``vec(x2 x2^T)``. It writes 2N values per model where :func:`_sampson_errors` writes 5N, which is RANSAC's reason
    to keep it (ALIKED matches, N ~ 700: 5.6 against 7.0 ms per pair on CPU, 10.0 against 11.4 ms on CUDA). The
    expansion cancels near the epipoles, which is harmless only on Hartley-normalized homogeneous points ``(N, 3)``
    and unit-norm models: on 58.3M float32 inlier decisions at 0.5-4 px from 90 PhotoTourism pairs it disagreed with
    float64 24 times and the line form 30 times.
    """
    n = x1.shape[0]
    basis = x1.new_zeros(27, 2 * n)
    basis[:9, :n] = _epipolar_design_rows(x1, x2).T
    basis[9:18, n:] = _epipolar_design_rows(x1, x1).T
    basis[18:, n:] = _epipolar_design_rows(x2, x2).T
    return basis


def _sampson_from_quadratic_basis(F: Tensor, basis: Tensor) -> Tensor:
    """Squared Sampson distances ``(M, N)`` of fundamental matrices ``(M, 3, 3)`` from :func:`_sampson_quadratic_basis`.

    A cancelled denominator can come out negative. ``clamp_min(0)`` lifts such a distance to 0, which keeps the inlier
    decisions of the unguarded ratio (a negative value is below every threshold) without the MSAC contribution above
    1, and passes NaN through, so a non-finite correspondence stays an outlier. A floor on the denominator instead
    turned 1849 of 2.56M near-epipole decisions into false inliers. A value guard for a no-grad loop, not a gradient
    guard (#4229).
    """
    n = basis.shape[1] // 2
    q1 = F[:, :2, :].mT @ F[:, :2, :]
    q2 = F[:, :, :2] @ F[:, :, :2].mT
    out = torch.cat([F.flatten(1), q1.flatten(1), q2.flatten(1)], 1) @ basis
    return (out[:, :n].square() / out[:, n:]).clamp_min(0)


def sampson_epipolar_distance(
    pts1: Tensor,
    pts2: Tensor,
    Fm: Tensor,
    squared: bool = True,
    eps: float = 1e-8,
    use_matmul_at_less_than_points: int = 10000,
) -> Tensor:
    r"""Return Sampson distance for correspondences given the fundamental matrix.

    Convention:
        - ``pts1`` are first-image points, ``pts2`` second-image points, and ``Fm`` follows
          :math:`x_2^\top F x_1 = 0`, as returned by :func:`find_fundamental`
          (see :ref:`two-view geometry <two-view-conventions>`).
        - Returns squared pixel distances by default; a 3-vector point is used as given, so it must have
          :math:`w = 1`.
        - One set of correspondences scored against several matrices (points with leading dimensions of size 1)
          is computed from two matrix products for all of them, in at least float32; the result matches the
          per-matrix computation to roundoff.
        - Known defects: ``eps`` is added to the denominator, so the value depends on the scale of ``Fm``;
          ``squared=False`` returns :math:`\sqrt{d^2 + \epsilon}`, which is not zero for an exact match; and on
          CUDA the matmul path omits ``eps`` from the denominator, so CUDA and CPU results differ
          (`#4881 <https://github.com/kornia/kornia/issues/4881>`_).

    Args:
        pts1: points in the first image with shape :math:`(*, N, 2)` or :math:`(*, N, 3)`.
        pts2: points in the second image with shape :math:`(*, N, 2)` or :math:`(*, N, 3)`.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.
        squared: if True (default), the squared distance is returned, else its square root.
        eps: Small constant added to the denominator and, for ``squared=False``, inside the square root.
        use_matmul_at_less_than_points: If ``Fm`` is on CUDA and the number of points is less than this value,
            use the matmul implementation.

    Returns:
        the computed Sampson distance with shape :math:`(*, N)`.

    """
    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")
    if (len(Fm.shape) < 3) or Fm.shape[-2:] != (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")
    num_points = pts1.shape[-2]
    matmul = Fm.device.type == "cuda" and num_points < use_matmul_at_less_than_points
    num_models = math.prod(Fm.shape[:-2])
    if (
        num_models >= 2
        and _shares_points(pts1, pts2)
        and (Fm.device.type == "cpu" or num_models * num_points >= _SAMPSON_SHARED_MIN_PAIRS_ACCELERATOR)
    ):
        # The CUDA matmul implementation leaves eps out of the denominator (#4881); the shared path follows it there.
        return _sampson_epipolar_distance_shared_impl_(pts1, pts2, Fm, squared, eps, 0.0 if matmul else eps)
    if matmul:
        return _sampson_epipolar_distance_matmul_impl_(pts1, pts2, Fm, squared, eps)
    return _sampson_epipolar_distance_manual_impl_(pts1, pts2, Fm, squared, eps)


def _symmetrical_epipolar_distance_manual_impl_(
    pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool = True, eps: float = 1e-8
) -> Tensor:
    """Return symmetric epipolar distance for correspondences given the fundamental matrix (CPU-optimized)."""
    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")

    if (len(Fm.shape) < 3) or Fm.shape[-2:] != (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    # Extract coords; support 2D (w=1) and 3D homogeneous
    x = pts1[..., :, 0]
    y = pts1[..., :, 1]
    u = pts2[..., :, 0]
    v = pts2[..., :, 1]

    # homogeneous weights with correct dtype/shape
    w1 = pts1[..., :, 2] if pts1.shape[-1] == 3 else ones_like(x)
    w2 = pts2[..., :, 2] if pts2.shape[-1] == 3 else ones_like(u)

    # Grab F entries and add a length-1 axis to broadcast across N
    f00 = Fm[..., 0, 0][..., None]
    f01 = Fm[..., 0, 1][..., None]
    f02 = Fm[..., 0, 2][..., None]
    f10 = Fm[..., 1, 0][..., None]
    f11 = Fm[..., 1, 1][..., None]
    f12 = Fm[..., 1, 2][..., None]
    f20 = Fm[..., 2, 0][..., None]
    f21 = Fm[..., 2, 1][..., None]
    f22 = Fm[..., 2, 2][..., None]

    # Fx = F @ [x, y, w1]^T  (compute components explicitly)
    Fx0 = f00 * x + f01 * y + f02 * w1
    Fx1 = f10 * x + f11 * y + f12 * w1
    Fx2 = f20 * x + f21 * y + f22 * w1

    # (F^T x')_{0:1} for x' = [u, v, w2]
    Ft0 = f00 * u + f10 * v + f20 * w2  # (F^T x')_0
    Ft1 = f01 * u + f11 * v + f21 * w2  # (F^T x')_1

    # Numerator: (x'^T F x)^2 = (u*Fx0 + v*Fx1 + w2*Fx2)^2
    num = (u * Fx0 + v * Fx1 + w2 * Fx2).pow(2)

    # denominator_inv = 1/|| (F x)_{1:2} ||^2 + 1/|| (F^T x')_{1:2} ||^2
    inv1 = 1.0 / (Fx0.pow(2) + Fx1.pow(2) + eps)
    inv2 = 1.0 / (Ft0.pow(2) + Ft1.pow(2) + eps)
    den_inv = inv1 + inv2

    out: Tensor = num * den_inv
    if squared:
        return out
    return (out + eps).sqrt()


def _symmetrical_epipolar_distance_matmul_impl_(
    pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool = True, eps: float = 1e-8
) -> Tensor:
    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)

    F_t: Tensor = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2: Tensor = pts1 @ F_t
    line2_in_1: Tensor = pts2 @ Fm

    numerator: Tensor = (pts2 * line1_in_2).sum(dim=-1).pow(2)

    denominator_inv: Tensor = 1.0 / (line1_in_2[..., :2].norm(2, dim=-1).pow(2) + eps) + 1.0 / (
        line2_in_1[..., :2].norm(2, dim=-1).pow(2) + eps
    )
    out: Tensor = numerator * denominator_inv
    if squared:
        return out
    return (out + eps).sqrt()


def symmetrical_epipolar_distance(
    pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool = True, eps: float = 1e-8
) -> Tensor:
    r"""Return symmetrical epipolar distance for correspondences given the fundamental matrix.

    Convention:
        - Argument order and units as :func:`sampson_epipolar_distance`; the value is the sum of the two
          squared point-to-epiline distances.
        - Known defects: ``eps`` makes the value depend on the scale of ``Fm``, and ``squared=False`` returns
          :math:`\sqrt{d^2 + \epsilon}`, as in :func:`sampson_epipolar_distance`
          (`#4881 <https://github.com/kornia/kornia/issues/4881>`_).

    Args:
       pts1: points in the first image with shape :math:`(*, N, 2)` or :math:`(*, N, 3)`.
       pts2: points in the second image with shape :math:`(*, N, 2)` or :math:`(*, N, 3)`.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.
       squared: if True (default), the squared distance is returned, else its square root.
       eps: Small constant added to the denominators and, for ``squared=False``, inside the square root.

    Returns:
        the computed Symmetrical distance with shape :math:`(*, N)`.

    """
    if Fm.device.type == "cuda":
        num_points = pts1.shape[-2]
        if num_points < 10000:
            return _symmetrical_epipolar_distance_matmul_impl_(pts1, pts2, Fm, squared, eps)
    return _symmetrical_epipolar_distance_manual_impl_(pts1, pts2, Fm, squared, eps)


def left_to_right_epipolar_distance(pts1: Tensor, pts2: Tensor, Fm: Tensor) -> Tensor:
    r"""Return one-sided epipolar distance for correspondences given the fundamental matrix.

    Convention:
        - Argument order as :func:`sampson_epipolar_distance`; returns the unsquared pixel distance of each
          ``pts2`` to the epipolar line of its ``pts1`` in the second image.
        - Known defects: ``point_line_distance`` adds ``eps`` to the line norm, so the value depends slightly on
          the scale of ``Fm`` (`#4881 <https://github.com/kornia/kornia/issues/4881>`_).

    Args:
       pts1: points in the first image with shape :math:`(*, N, 2)` or :math:`(*, N, 3)`.
       pts2: points in the second image with shape :math:`(*, N, 2)` or :math:`(*, N, 3)`.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to
         avoid ambiguity with torch.nn.functional.

    Returns:
        the one-sided distance with shape :math:`(*, N)`.

    """
    KORNIA_CHECK_IS_TENSOR(pts1)
    KORNIA_CHECK_IS_TENSOR(pts2)
    KORNIA_CHECK_IS_TENSOR(Fm)

    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)

    F_t: Tensor = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2: Tensor = pts1 @ F_t

    return point_line_distance(pts2, line1_in_2)


def right_to_left_epipolar_distance(pts1: Tensor, pts2: Tensor, Fm: Tensor) -> Tensor:
    r"""Return one-sided epipolar distance for correspondences given the fundamental matrix.

    Convention:
        - Argument order as :func:`sampson_epipolar_distance`; returns the unsquared pixel distance of each
          ``pts1`` to the epipolar line of its ``pts2`` in the first image.
        - Known defects: ``point_line_distance`` adds ``eps`` to the line norm, so the value depends slightly on
          the scale of ``Fm`` (`#4881 <https://github.com/kornia/kornia/issues/4881>`_).

    Args:
       pts1: points in the first image with shape :math:`(*, N, 2)` or :math:`(*, N, 3)`.
       pts2: points in the second image with shape :math:`(*, N, 2)` or :math:`(*, N, 3)`.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to
         avoid ambiguity with torch.nn.functional.

    Returns:
        the one-sided distance with shape :math:`(*, N)`.

    """
    KORNIA_CHECK_IS_TENSOR(pts1)
    KORNIA_CHECK_IS_TENSOR(pts2)
    KORNIA_CHECK_IS_TENSOR(Fm)

    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)

    line2_in_1: Tensor = pts2 @ Fm

    return point_line_distance(pts1, line2_in_1)

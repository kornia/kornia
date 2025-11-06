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

import torch
from torch import Tensor, ones_like

from kornia.core.check import KORNIA_CHECK_IS_TENSOR
from kornia.geometry.conversions import convert_points_to_homogeneous
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
    # Extract coordinates; support 2D (assume w=1) and 3D homogeneous
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


@torch.jit.script
def sampson_epipolar_distance(
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
    if Fm.device.type == "cuda":
        num_points = pts1.shape[-2]
        if num_points < 10000:
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
    """Return symmetrical epipolar distance for correspondences given the fundamental matrix.

    Args:
       pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
             converted automatically.
       pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
             converted automatically.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.
       squared: if True (default), the squared distance is returned.
       eps: Small constant for safe sqrt.

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

    This method measures the distance from points in the right images to the epilines
    of the corresponding points in the left images as they reflect in the right images.

    Args:
       pts1: correspondences from the left images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       pts2: correspondences from the right images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to
         avoid ambiguity with torch.nn.functional.

    Returns:
        the computed Symmetrical distance with shape :math:`(*, N)`.

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

    This method measures the distance from points in the left images to the epilines
    of the corresponding points in the right images as they reflect in the left images.

    Args:
       pts1: correspondences from the left images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       pts2: correspondences from the right images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to
         avoid ambiguity with torch.nn.functional.

    Returns:
        the computed Symmetrical distance with shape :math:`(*, N)`.

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

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

from typing import Optional

import torch
import torch.nn.functional as F

from kornia.core.utils import is_exporting


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/distortion_model.hpp#L75
def tilt_projection(taux: torch.Tensor, tauy: torch.Tensor, return_inverse: bool = False) -> torch.Tensor:
    r"""Estimate the tilt projection matrix or the inverse tilt projection matrix.

    Convention:
        - the rotation is ``R = Ry(tauy) @ Rx(taux)`` and ``Pz`` is built from the third column of ``R``. Both
          branches return exactly ``eye(3)`` when ``taux`` and ``tauy`` are zero, which is the case for a
          ``dist`` vector whose 13th and 14th entries are zero.
        - ``return_inverse=True`` returns the inverse of ``Pz @ R``. That is the branch
          :func:`~kornia.geometry.calibration.undistort_points` applies, and it is what reproduces OpenCV's
          ``undistortPoints`` on this repository's own reference values.
        - ``return_inverse=False`` returns ``Pz @ R.T``, so the two branches are not inverses of each other.

    .. warning::
        OpenCV's ``computeTiltProjectionMatrix``, which this implementation cites, returns ``Pz @ R`` for the
        forward branch, and so does the tilt step written out at the top of the ``kornia.geometry.calibration``
        documentation page; the branch here returns ``Pz @ R.T``. Tracked as
        `#4276 <https://github.com/kornia/kornia/issues/4276>`_. The consequence is silent while both angles
        are zero and otherwise makes
        :func:`~kornia.geometry.calibration.distort_points` and
        :func:`~kornia.geometry.calibration.undistort_points` stop being inverses. The behaviour is documented
        as it is and pinned by the ``test_convention_*`` / ``test_wart_*`` tests in
        ``tests/geometry/calibration/test_distort.py``, including the strict ``xfail``
        ``test_convention_tilt_projection_branches_are_inverses_4276``.

    Args:
        taux: Rotation angle in radians around the :math:`x`-axis with shape :math:`(*, 1)`.
        tauy: Rotation angle in radians around the :math:`y`-axis with shape :math:`(*, 1)`.
        return_inverse: False to obtain the tilt projection matrix. True for the inverse matrix.

    Returns:
        torch.Tensor: Tilt projection matrix, or the inverse tilt projection matrix when ``return_inverse`` is
        True, with shape :math:`(*, 3, 3)`.

    """
    if taux.shape != tauy.shape:
        raise ValueError(f"Shape of taux {taux.shape} and tauy {tauy.shape} do not match.")

    ndim: int = taux.dim()
    taux = taux.reshape(-1)
    tauy = tauy.reshape(-1)

    cTx = torch.cos(taux)
    sTx = torch.sin(taux)
    cTy = torch.cos(tauy)
    sTy = torch.sin(tauy)
    zero = torch.zeros_like(cTx)
    one = torch.ones_like(cTx)

    Rx = torch.stack([one, zero, zero, zero, cTx, sTx, zero, -sTx, cTx], -1).reshape(-1, 3, 3)
    Ry = torch.stack([cTy, zero, -sTy, zero, one, zero, sTy, zero, cTy], -1).reshape(-1, 3, 3)
    R = Ry @ Rx

    if return_inverse:
        invR22 = 1 / R[..., 2, 2]
        invPz = torch.stack(
            [invR22, zero, R[..., 0, 2] * invR22, zero, invR22, R[..., 1, 2] * invR22, zero, zero, one], -1
        ).reshape(-1, 3, 3)

        inv_tilt = R.transpose(-1, -2) @ invPz
        if ndim == 0:
            inv_tilt = torch.squeeze(inv_tilt)

        return inv_tilt

    Pz = torch.stack(
        [R[..., 2, 2], zero, -R[..., 0, 2], zero, R[..., 2, 2], -R[..., 1, 2], zero, zero, one], -1
    ).reshape(-1, 3, 3)

    tilt = Pz @ R.transpose(-1, -2)
    if ndim == 0:
        tilt = torch.squeeze(tilt)

    return tilt


def distort_points(
    points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, new_K: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Distortion of a set of 2D points based on the lens distortion model.

    Radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Convention:
        - ``points`` are **pixel** coordinates and so is the result.
          Those pixels are measured on the integer-centre grid described in the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
          :func:`~kornia.geometry.camera.distort_points_affine` and
          :func:`~kornia.geometry.camera.distort_points_kannala_brandt` are the counterparts that take a point
          on the normalized :math:`z = 1` plane and a flat parameter vector instead of ``K`` and ``dist``.
        - ``dist`` is OpenCV's coefficient vector in the order listed under ``Args``. The lengths 4, 5, 8, 12
          and 14 are accepted and every other length raises :class:`ValueError`; an accepted shorter vector is
          zero-padded to 14 internally, so a 4-element vector and its 14-element zero padding give the same
          answer.
        - ``new_K`` and ``K`` play opposite roles: ``new_K`` maps the incoming pixel onto the normalized
          plane and ``K`` maps the distorted normalized point back to pixels. ``new_K`` defaults to ``K``.
        - :func:`~kornia.geometry.calibration.undistort_points` is the inverse map and takes the same
          coefficient layout, with the two intrinsics in the mirrored roles.

    .. warning::
        With a non-zero :math:`\tau_x` or :math:`\tau_y` this function applies
        :func:`~kornia.geometry.calibration.tilt_projection`\'s forward branch, ``Pz @ R.T``, while
        :func:`~kornia.geometry.calibration.undistort_points` applies the inverse branch, the inverse of
        ``Pz @ R``. The two are therefore not inverses of each other there, and this function disagrees with
        the inverse of OpenCV's ``undistortPoints``. Tracked as
        `#4276 <https://github.com/kornia/kornia/issues/4276>`_ and pinned by
        ``test_wart_tilt_projection_forward_is_pz_times_r_transpose_4276`` in
        ``tests/geometry/calibration/test_distort.py`` and
        ``test_wart_distort_undistort_round_trip_breaks_with_tilt_4276`` in
        ``tests/geometry/calibration/test_undistort.py``.

    .. warning::
        ``torch.compile(fullgraph=True)`` fails on this function because the tilt test reads the coefficient
        values on the host; ONNX export is already routed around it by ``is_exporting()``. Tracked as
        `#4286 <https://github.com/kornia/kornia/issues/4286>`_.

    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.
        new_K: Intrinsic camera matrix of the distorted image. By default, it is the same as K but you may additionally
            scale and shift the result by using a different matrix. Shape: :math:`(*, 3, 3)`. Default: None.

    Returns:
        Distorted 2D points with shape :math:`(*, N, 2)`.

    Example:
        >>> points = torch.rand(1, 1, 2)
        >>> K = torch.eye(3)[None]
        >>> dist_coeff = torch.rand(1, 4)
        >>> points_dist = distort_points(points, K, dist_coeff)

    """
    if points.dim() < 2 and points.shape[-1] != 2:
        raise ValueError(f"points shape is invalid. Got {points.shape}.")

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K matrix shape is invalid. Got {K.shape}.")

    if new_K is None:
        new_K = K
    elif new_K.shape[-2:] != (3, 3):
        raise ValueError(f"new_K matrix shape is invalid. Got {new_K.shape}.")

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}")

    # Adding torch.zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = F.pad(dist, [0, 14 - dist.shape[-1]])

    # Convert 2D points from pixels to normalized camera coordinates
    new_cx: torch.Tensor = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy: torch.Tensor = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx: torch.Tensor = new_K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy: torch.Tensor = new_K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x: torch.Tensor = (points[..., 0] - new_cx) / new_fx  # (BxN - Bx1)/Bx1 -> BxN or (N,)
    y: torch.Tensor = (points[..., 1] - new_cy) / new_fy  # (BxN - Bx1)/Bx1 -> BxN or (N,)

    # Distort points
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    rad_poly = (1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r4 + dist[..., 4:5] * r6) / (
        1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r4 + dist[..., 7:8] * r6
    )
    xd = (
        x * rad_poly
        + 2 * dist[..., 2:3] * x * y
        + dist[..., 3:4] * (r2 + 2 * x * x)
        + dist[..., 8:9] * r2
        + dist[..., 9:10] * r4
    )
    yd = (
        y * rad_poly
        + dist[..., 2:3] * (r2 + 2 * y * y)
        + 2 * dist[..., 3:4] * x * y
        + dist[..., 10:11] * r2
        + dist[..., 11:12] * r4
    )

    # Compensate for tilt distortion. The zero test reads the data, which graph capture cannot do, so the
    # exported graph always applies the tilt (an identity when both tau coefficients are zero).
    if is_exporting() or torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        tilt = tilt_projection(dist[..., 12], dist[..., 13])

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        points_untilt = torch.stack([xd, yd, torch.ones_like(xd)], -1) @ tilt.transpose(-2, -1)
        xd = points_untilt[..., 0] / points_untilt[..., 2]
        yd = points_untilt[..., 1] / points_untilt[..., 2]

    # Convert points from normalized camera coordinates to pixel coordinates
    cx: torch.Tensor = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy: torch.Tensor = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx: torch.Tensor = K[..., 0:1, 0]  # focal in x (Bx1)
    fy: torch.Tensor = K[..., 1:2, 1]  # focal in y (Bx1)

    x = fx * xd + cx
    y = fy * yd + cy

    return torch.stack([x, y], -1)

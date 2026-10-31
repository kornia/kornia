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

"""Add OpenCV's Brown-Conrady lens distortion to pixel coordinates, and build its sensor-tilt matrix."""

from typing import Optional

import torch
import torch.nn.functional as F


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/distortion_model.hpp#L75
def tilt_projection(taux: torch.Tensor, tauy: torch.Tensor, return_inverse: bool = False) -> torch.Tensor:
    r"""Build the 3x3 sensor-tilt matrix of OpenCV's distortion model, or its inverse.

    ``T = tilt_projection(taux, tauy)`` takes the tilt angles in radians, the last two of the 14 coefficients of
    OpenCV's distortion vector, and returns the projective ``(*, 3, 3)`` matrix that
    :func:`~kornia.geometry.calibration.distort_points` applies to the distorted normalized point ``(x, y, 1)``
    before dividing by the third component; ``return_inverse=True`` returns its inverse, which
    :func:`~kornia.geometry.calibration.undistort_points` applies first. Those two functions call it themselves:
    call it directly only to inspect or reuse the tilt step.

    Convention:
        - the rotation is ``R = Ry(tauy) @ Rx(taux)`` and ``Pz`` is built from the third column of ``R``;
          ``return_inverse=False`` returns ``Pz @ R`` (OpenCV's tilt projection) and ``return_inverse=True`` its
          inverse, the branch :func:`~kornia.geometry.calibration.undistort_points` applies. Zero angles give
          ``eye(3)``.
        - scalar angles return ``(3, 3)``; otherwise a trailing singleton angle axis, if present, is
          consumed and the remaining dimensions are the batch shape.

    Args:
        taux: Rotation angle in radians around the :math:`x`-axis with any shape, matching the other angle.
        tauy: Rotation angle in radians around the :math:`y`-axis with any shape, matching the other angle.
        return_inverse: False to obtain the tilt projection matrix. True for the inverse matrix.

    Returns:
        torch.Tensor: Tilt projection matrix, or the inverse tilt projection matrix when ``return_inverse`` is
        True, with shape ``(3, 3)`` for scalar angles or ``(*, 3, 3)`` for non-scalar angles, where a
        trailing singleton angle-component axis is omitted from :math:`*`.

    Raises:
        ValueError: if ``taux`` and ``tauy`` differ in shape.

    See Also:
        - :func:`~kornia.geometry.calibration.distort_points`: applies this matrix when given 14 coefficients.
        - :func:`~kornia.geometry.calibration.undistort_points`: applies its inverse when given 14 coefficients.

    """
    if taux.shape != tauy.shape:
        raise ValueError(f"Shape of taux {taux.shape} and tauy {tauy.shape} do not match.")

    if taux.dim() > 0 and taux.shape[-1] == 1:
        taux = taux.squeeze(-1)
        tauy = tauy.squeeze(-1)

    cTx = torch.cos(taux)
    sTx = torch.sin(taux)
    cTy = torch.cos(tauy)
    sTy = torch.sin(tauy)
    zero = torch.zeros_like(cTx)
    one = torch.ones_like(cTx)

    Rx = torch.stack(
        [
            torch.stack([one, zero, zero], -1),
            torch.stack([zero, cTx, sTx], -1),
            torch.stack([zero, -sTx, cTx], -1),
        ],
        -2,
    )
    Ry = torch.stack(
        [
            torch.stack([cTy, zero, -sTy], -1),
            torch.stack([zero, one, zero], -1),
            torch.stack([sTy, zero, cTy], -1),
        ],
        -2,
    )
    R = Ry @ Rx

    if return_inverse:
        invR22 = 1 / R[..., 2, 2]
        invPz = torch.stack(
            [
                torch.stack([invR22, zero, R[..., 0, 2] * invR22], -1),
                torch.stack([zero, invR22, R[..., 1, 2] * invR22], -1),
                torch.stack([zero, zero, one], -1),
            ],
            -2,
        )

        return R.transpose(-1, -2) @ invPz

    Pz = torch.stack(
        [
            torch.stack([R[..., 2, 2], zero, -R[..., 0, 2]], -1),
            torch.stack([zero, R[..., 2, 2], -R[..., 1, 2]], -1),
            torch.stack([zero, zero, one], -1),
        ],
        -2,
    )

    return Pz @ R


def distort_points(
    points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, new_K: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Distort pixel coordinates with OpenCV's Brown-Conrady model of 4 to 14 coefficients.

    ``distorted = distort_points(points, K, dist)`` takes ideal pinhole pixels ``(*, N, 2)``, the intrinsics
    ``(*, 3, 3)`` and OpenCV's coefficient vector ``(*, n)``, ``n`` one of 4, 5, 8, 12 or 14, and returns the pixels
    ``(*, N, 2)`` where a lens with that distortion images the same rays: the values ``cv2.projectPoints`` gives for
    the points' rays. To remove distortion from measured pixels, use
    :func:`~kornia.geometry.calibration.undistort_points`; from a whole image,
    :func:`~kornia.geometry.calibration.undistort_image`.

    Radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Convention:
        - ``points`` are **pixel** coordinates in ``(u, v)`` order and so is the result (integer pixel centres;
          see :class:`~kornia.geometry.camera.pinhole.PinholeCamera`).
          :func:`~kornia.geometry.camera.distort_points_affine` and
          :func:`~kornia.geometry.camera.distort_points_kannala_brandt` take a point on the normalized
          :math:`z = 1` plane and a flat parameter vector instead.
        - ``dist`` is OpenCV's coefficient vector in the order listed under ``Args``. The lengths 4, 5, 8, 12
          and 14 are accepted, zero-padded to 14 internally; any other length raises :class:`ValueError`.
        - ``new_K`` maps the incoming pixel onto the normalized plane and ``K`` maps the distorted point back
          to pixels; ``new_K`` defaults to ``K``. :func:`~kornia.geometry.calibration.undistort_points` is the
          inverse map, with the two intrinsics in the mirrored roles.
        - matching leading dimensions of ``points``, ``K`` and ``dist`` are preserved, and the leading dimensions
          ``*`` of the three broadcast by torch's rules: ``(N, 2)`` points with ``(B, 3, 3)`` cameras give
          ``(B, N, 2)``.
        - the skew entry ``K[0, 1]`` is ignored, as in
          :func:`~kornia.geometry.conversions.normalize_points_with_intrinsics`.
        - the trailing size of ``points`` is not checked today: ``(N, 3)`` points are truncated to their first two
          columns, and a single ``(2,)`` point returns ``(1, 2)``
          (`#6467 <https://github.com/kornia/kornia/issues/6467>`_).

    Args:
        points: Input image points with shape ``(*, N, 2)``.
        K: Intrinsic camera matrix with shape ``(*, 3, 3)``.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape ``(*, n)``.
        new_K: Intrinsic camera matrix used to map the incoming ``points`` from pixels onto the normalized
            :math:`z = 1` plane -- the opposite of what the argument order suggests, since ``K`` is the one
            that maps the distorted normalized point back to pixels. By default it is the same as ``K``, in
            which case both steps use the same camera; a different matrix rescales and shifts the **input**,
            not the result. Shape: ``(*, 3, 3)``. Default: None.

    Returns:
        Distorted 2D points with shape ``(*, N, 2)``.

    See Also:
        - :func:`~kornia.geometry.calibration.undistort_points`: the inverse, by fixed-point iteration, with ``K``
          and ``new_K`` in mirrored roles.
        - :func:`~kornia.geometry.calibration.undistort_image`: removes the distortion from an image by sampling
          it at the positions this function returns.
        - :func:`~kornia.geometry.calibration.tilt_projection`: the sensor-tilt matrix of the last two
          coefficients.
        - :func:`~kornia.geometry.camera.distort_points_kannala_brandt`: the fisheye model, from normalized points
          and a flat 8-vector.
        - :func:`~kornia.geometry.camera.distort_points_affine`: the intrinsics alone, from normalized points and a
          flat ``[fx, fy, cx, cy]``.

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

    has_tilt = dist.shape[-1] == 14
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

    # Zero tilt is the identity but has nonzero angle derivatives. Branch on the model's
    # coefficient count, not its values, so calibration can learn tilt from a zero initialization.
    if has_tilt:
        tilt = tilt_projection(dist[..., 12:13], dist[..., 13:14])

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

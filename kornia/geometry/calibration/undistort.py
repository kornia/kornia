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

"""Remove OpenCV's Brown-Conrady lens distortion from pixel coordinates and from images."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import _at_least_float32
from kornia.geometry.grid import create_meshgrid
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import remap

from .distort import distort_points, tilt_projection


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L384
def undistort_points(
    points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, new_K: Optional[torch.Tensor] = None, num_iters: int = 5
) -> torch.Tensor:
    r"""Remove Brown-Conrady lens distortion from pixel coordinates by fixed-point iteration.

    ``points = undistort_points(distorted, K, dist)`` takes measured pixels ``(*, N, 2)``, the intrinsics
    ``(*, 3, 3)`` and OpenCV's coefficient vector ``(*, n)``, ``n`` one of 4, 5, 8, 12 or 14, and returns the
    ideal pinhole pixels ``(*, N, 2)`` after ``num_iters`` fixed-point steps; within the convergence region, with
    the default 5 steps, the result matches ``cv2.undistortPoints(points, K, dist, P=P)`` with
    ``P = K if new_K is None else new_K`` and no skew in ``P``: OpenCV applies ``P[0, 1]``, which this function
    ignores, and without ``P`` it returns normalized coordinates. For normalized
    coordinates, as :func:`~kornia.geometry.epipolar.find_essential` takes, pass the result to
    :func:`~kornia.geometry.conversions.normalize_points_with_intrinsics`; for a whole image, use
    :func:`~kornia.geometry.calibration.undistort_image`.

    Radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Convention:
        - ``points`` are **pixel** coordinates in ``(u, v)`` order and so is the result (integer pixel centres;
          see :class:`~kornia.geometry.camera.pinhole.PinholeCamera`).
        - ``dist`` follows the layout of :func:`~kornia.geometry.calibration.distort_points`, the forward map
          this function inverts; ``K`` maps the incoming pixel onto the normalized plane and ``new_K`` maps the
          undistorted point back to pixels.
        - the inverse is a fixed-point iteration of ``num_iters`` steps, so the round trip through
          :func:`~kornia.geometry.calibration.distort_points` closes only to the accuracy the iteration reaches;
          within its convergence region more steps improve it down to the dtype's rounding.
        - matching leading dimensions of ``points``, ``K`` and ``dist`` are preserved, and the leading dimensions
          broadcast as in :func:`~kornia.geometry.calibration.distort_points`, the sensor-tilt step of 14
          coefficients included.
        - the skew entries ``K[0, 1]`` and ``new_K[0, 1]`` are ignored, as in
          :func:`~kornia.geometry.conversions.normalize_points_with_intrinsics`.
        - float16 and bfloat16 inputs are iterated in float32 and the result is returned in their dtype.

    .. warning::
        The iteration has no convergence test and no valid-radius guard: outside the convergence region it can
        enter a two-point cycle, so the answer depends on the parity of ``num_iters``.
        `#4285 <https://github.com/kornia/kornia/issues/4285>`_.

    Args:
        points: Input image points with shape ``(*, N, 2)``.
        K: Intrinsic camera matrix with shape ``(*, 3, 3)``.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape ``(*, n)``.
        new_K: Intrinsic camera matrix used to map the undistorted normalized point back to pixels, while ``K``
            is the one that maps the incoming distorted ``points`` from pixels onto the normalized
            :math:`z = 1` plane -- the mirror image of the two roles in
            :func:`~kornia.geometry.calibration.distort_points`. By default it is the same as ``K``, in which
            case both steps use the same camera; a different matrix rescales and shifts the **result**.
            Shape: ``(*, 3, 3)``. Default: None.
        num_iters: Number of undistortion iterations. Default: 5.

    Returns:
        Undistorted 2D points with shape ``(*, N, 2)``.

    See Also:
        - :func:`~kornia.geometry.calibration.distort_points`: the forward map, in closed form, with ``K`` and
          ``new_K`` in mirrored roles.
        - :func:`~kornia.geometry.calibration.undistort_image`: undistorts every pixel of an image instead of a
          point list.
        - :func:`~kornia.geometry.calibration.tilt_projection`: the inverse sensor-tilt matrix applied first with
          14 coefficients.
        - :func:`~kornia.geometry.camera.undistort_points_kannala_brandt`: the fisheye inverse, from pixels to
          normalized points.
        - :func:`~kornia.geometry.camera.undistort_points_affine`: the intrinsics alone, from pixels to normalized
          points, in closed form.
        - :class:`~kornia.sensors.camera.distortion_model.BrownConradyTransform`: the same inverse on ``Vector2``
          pixels, to normalized points.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 4, 2)
        >>> K = torch.eye(3)[None]
        >>> dist = torch.rand(1, 4)
        >>> undistort_points(x, K, dist)
        tensor([[[-0.1513, -0.1165],
                 [ 0.0711,  0.1100],
                 [-0.0697,  0.0228],
                 [-0.1843, -0.1606]]])

    """
    KORNIA_CHECK_SHAPE(points, ["*", "N", "2"])
    KORNIA_CHECK_SHAPE(K, ["*", "3", "3"])

    if points.dim() < 2 and points.shape[-1] != 2:
        raise ValueError(f"points shape is invalid. Got {points.shape}.")

    if new_K is None:
        new_K = K
    else:
        KORNIA_CHECK_SHAPE(new_K, ["*", "3", "3"])

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}")

    # Half-precision inputs are iterated in float32: each step would otherwise round the normalized coordinates and
    # the polynomials to the input dtype, and the pixel result is rounded once at the end.
    out_dtype = torch.promote_types(
        torch.promote_types(points.dtype, K.dtype), torch.promote_types(dist.dtype, new_K.dtype)
    )
    points, K, new_K, dist = (_at_least_float32(t) for t in (points, K, new_K, dist))

    has_tilt = dist.shape[-1] == 14
    # Adding torch.zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = F.pad(dist, [0, 14 - dist.shape[-1]])

    # Convert 2D points from pixels to normalized camera coordinates
    cx: torch.Tensor = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy: torch.Tensor = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx: torch.Tensor = K[..., 0:1, 0]  # focal in x (Bx1)
    fy: torch.Tensor = K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x: torch.Tensor = (points[..., 0] - cx) / fx  # (BxN - Bx1)/Bx1 -> BxN
    y: torch.Tensor = (points[..., 1] - cy) / fy  # (BxN - Bx1)/Bx1 -> BxN

    # Zero tilt is the identity but has nonzero angle derivatives. Branch on the model's
    # coefficient count, not its values, so calibration can learn tilt from a zero initialization.
    if has_tilt:
        inv_tilt = tilt_projection(dist[..., 12:13], dist[..., 13:14], True)
        xy = torch.stack([x, y], dim=-1)
        # transform_points pairs a batch of transforms with the leading axis of the points, so give it one transform
        # per point set: the tilt then broadcasts against the points like the other coefficients.
        # (TorchScript has no torch.broadcast_shapes, so broadcast 1 x 1 corners for the batch shape.)
        batch_shape = list(torch.broadcast_tensors(xy[..., :1, :1], inv_tilt[..., :1, :1])[0].shape[:-2])
        inv_tilt = inv_tilt.expand(batch_shape + [3, 3])
        xy = xy.expand(batch_shape + list(xy.shape[-2:]))

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        x, y = transform_points(inv_tilt, xy).unbind(-1)

    # Iteratively undistort points
    x0, y0 = x, y
    for _ in range(num_iters):
        r2 = x * x + y * y

        inv_rad_poly = (1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2**3) / (
            1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2**3
        )
        deltaX = (
            2 * dist[..., 2:3] * x * y
            + dist[..., 3:4] * (r2 + 2 * x * x)
            + dist[..., 8:9] * r2
            + dist[..., 9:10] * r2 * r2
        )
        deltaY = (
            dist[..., 2:3] * (r2 + 2 * y * y)
            + 2 * dist[..., 3:4] * x * y
            + dist[..., 10:11] * r2
            + dist[..., 11:12] * r2 * r2
        )

        x = (x0 - deltaX) * inv_rad_poly
        y = (y0 - deltaY) * inv_rad_poly

    # Convert points from normalized camera coordinates to pixel coordinates
    new_cx: torch.Tensor = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy: torch.Tensor = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx: torch.Tensor = new_K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy: torch.Tensor = new_K[..., 1:2, 1]  # focal in y (Bx1)
    x = new_fx * x + new_cx
    y = new_fy * y + new_cy
    out = torch.stack([x, y], -1)
    # Only a half-precision result is cast back: integer inputs return the float32 result of the arithmetic.
    return out.to(out_dtype) if out_dtype in (torch.float16, torch.bfloat16) else out


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L287
def undistort_image(image: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    r"""Undistort an image with OpenCV's Brown-Conrady coefficients, keeping its camera matrix K.

    ``out = undistort_image(image, K, dist)`` takes images ``(*, C, H, W)``, the intrinsics ``(*, 3, 3)`` and
    OpenCV's coefficient vector ``(*, n)``, ``n`` one of 4, 5, 8, 12 or 14, and returns the image a camera with the
    same ``K`` and no distortion would take, ``(*, C, H, W)``: each output pixel samples the input bilinearly at its
    :func:`~kornia.geometry.calibration.distort_points` position, with zeros outside the input. There is no
    ``new_K``, so the result keeps the input's size and ``K``, as ``cv2.undistort`` does by default. For point
    coordinates, use :func:`~kornia.geometry.calibration.undistort_points`.

    Radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Convention:
        - the leading dimensions of ``image`` (in front of ``C, H, W``), ``K`` (in front of :math:`3 \times 3`)
          and ``dist`` (in front of its ``n`` coefficients) must match exactly, so an unbatched ``(C, H, W)``
          image takes a ``(3, 3)`` ``K`` and an ``(n,)`` ``dist``; the one exception is the legacy unbatched
          call, a ``(1, C, H, W)`` image with a ``(3, 3)`` ``K`` and an ``(n,)`` ``dist``.
        - an image that is not floating point raises :class:`ValueError`.
        - the sampling map applies :func:`~kornia.geometry.calibration.distort_points` to the integer pixel
          centres (see :class:`~kornia.geometry.camera.pinhole.PinholeCamera`) and is resampled bilinearly with
          a fixed ``align_corners=True``, so all-zero coefficients return the input only to rounding.
        - a float16 or bfloat16 image builds the map, distorts it and samples in float32, with ``K`` and ``dist``
          cast to float32, and returns the image dtype; in half precision the pixel coordinates themselves round to
          half pixels (float16, from 512) or whole pixels (bfloat16, from 128).

    Args:
        image: Input image with shape ``(*, C, H, W)``.
        K: Intrinsic camera matrix with shape ``(*, 3, 3)``.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape ``(*, n)``.

    Returns:
        Undistorted image with shape ``(*, C, H, W)``.

    See Also:
        - :func:`~kornia.geometry.calibration.undistort_points`: undistorts a list of pixel coordinates instead of
          an image.
        - :func:`~kornia.geometry.calibration.distort_points`: the map whose positions the image is sampled at.
        - :func:`~kornia.geometry.transform.remap`: the sampling step, to resample with other maps or options.

    Example:
        >>> img = torch.rand(1, 3, 5, 5)
        >>> K = torch.eye(3)[None]
        >>> dist_coeff = torch.rand(1, 4)
        >>> out = undistort_image(img, K, dist_coeff)
        >>> out.shape
        torch.Size([1, 3, 5, 5])

    """
    if len(image.shape) < 3:
        raise ValueError(f"Image shape is invalid. Got: {image.shape}.")

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K matrix shape is invalid. Got {K.shape}.")

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}.")

    if not image.is_floating_point():
        raise ValueError(f"Invalid input image data type. Input should be float. Got {image.dtype}.")

    # Input with image shape (1, C, H, W), K shape (3, 3), dist shape (4)
    # allowed to avoid a breaking change.
    if (image.shape[:-3] != K.shape[:-2] or image.shape[:-3] != dist.shape[:-1]) and not all(
        (image.shape[:-3] == (1,), K.shape[:-2] == (), dist.shape[:-1] == ())
    ):
        raise ValueError(
            "Input shape is invalid. Input batch dimensions should match. "
            f"Got {image.shape[:-3]}, {K.shape[:-2]}, {dist.shape[:-1]}."
        )

    channels, rows, cols = image.shape[-3:]
    B = image.numel() // (channels * rows * cols)

    # Pixel coordinates reach the image size, where float16 rounds them to half pixels from 512 and bfloat16 to whole
    # pixels from 128, so a half image builds and distorts its map and samples in float32, as the warps do.
    work_dtype = torch.float32 if image.dtype in (torch.float16, torch.bfloat16) else image.dtype
    if work_dtype != image.dtype:
        K, dist = K.to(work_dtype), dist.to(work_dtype)

    # Create point coordinates for each pixel of the image
    xy_grid: torch.Tensor = create_meshgrid(rows, cols, False, image.device, work_dtype)
    pts = xy_grid.reshape(-1, 2)  # (rows*cols)x2 matrix of pixel coordinates

    # Distort points and define maps
    ptsd: torch.Tensor = distort_points(pts, K, dist)  # Bx(rows*cols)x2
    mapx: torch.Tensor = ptsd[..., 0].reshape(B, rows, cols)  # B x rows x cols, float
    mapy: torch.Tensor = ptsd[..., 1].reshape(B, rows, cols)  # B x rows x cols, float

    # Remap image to undistort
    out = remap(image.reshape(B, channels, rows, cols).to(work_dtype), mapx, mapy, align_corners=True)

    return out.to(image.dtype).view_as(image)

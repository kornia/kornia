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


import torch

from kornia.geometry.calibration import distort_points, undistort_points
from kornia.geometry.camera.distortion_kannala_brandt import (
    distort_points_kannala_brandt,
    undistort_points_kannala_brandt,
)
from kornia.geometry.vector import Vector2


class AffineTransform:
    """Map image-plane ``Vector2`` points to pixels with ``[fx, fy, cx, cy]`` alone, with no lens distortion.

    ``uv = AffineTransform().distort(params, points)`` takes ``params`` that start with ``[fx, fy, cx, cy]`` and a
    :class:`~kornia.geometry.vector.Vector2` of image-plane points, and returns the ``Vector2`` of pixels
    ``(fx * x + cx, fy * y + cy)``. :meth:`undistort` inverts it in closed form.
    :class:`~kornia.sensors.camera.PinholeModel` feeds it normalized ``z = 1`` points and
    :class:`~kornia.sensors.camera.Orthographic` the camera-frame ``(x, y)``.

    See Also:
        - :class:`BrownConradyTransform`: adds OpenCV's radial and tangential lens distortion before the intrinsics.
        - :class:`KannalaBrandtK3Transform`: adds a fisheye distortion of the ray angle before the intrinsics.
        - :func:`~kornia.geometry.camera.distort_points_affine`: the same map on plain tensors.
        - :func:`~kornia.geometry.camera.undistort_points_affine`: the same inverse on plain tensors.

    """

    def distort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Distort one or more Vector2 points using the affine transform.

        Convention:
            - ``points`` is on the **normalized** :math:`z = 1` plane and the result is in **pixels**:
              ``u = fx * x + cx`` and ``v = fy * y + cy``, with ``params`` laid out as ``[fx, fy, cx, cy]``;
              the ``Vector2`` counterpart of :func:`~kornia.geometry.camera.distort_points_affine`.
            - :meth:`undistort` is the closed-form inverse.

        Args:
            params: torch.Tensor representing the affine transform parameters.
            points: Vector2 representing the points to distort.

        Returns:
            Vector2 representing the distorted points.

        Example:
            >>> params = torch.Tensor([1., 2., 3., 4.])
            >>> points = Vector2.from_coords(1., 2.)
            >>> AffineTransform().distort(params, points)
            x: 4.0
            y: 8.0

        """
        fx, fy, cx, cy = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        u = points.x * fx + cx
        v = points.y * fy + cy
        return Vector2.from_coords(u, v)

    def undistort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Undistort one or more Vector2 points using the affine transform.

        Convention:
            - ``points`` is in **pixels** and the result is on the normalized :math:`z = 1` plane:
              ``x = (u - cx) / fx`` and ``y = (v - cy) / fy``, the closed-form inverse of :meth:`distort`
              with the same ``params`` layout -- see the Convention block there.

        Args:
            params: torch.Tensor representing the affine transform parameters.
            points: Vector2 representing the points to undistort.

        Returns:
            Vector2 representing the undistorted points.

        Example:
            >>> params = torch.Tensor([1., 2., 3., 4.])
            >>> points = Vector2.from_coords(1., 2.)
            >>> AffineTransform().undistort(params, points)
            x: -2.0
            y: -1.0

        """
        fx, fy, cx, cy = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        x = (points.x - cx) / fx
        y = (points.y - cy) / fy
        return Vector2.from_coords(x, y)


class BrownConradyTransform:
    """Map normalized ``z = 1`` ``Vector2`` points to pixels with OpenCV's Brown-Conrady lens distortion, and back.

    ``uv = BrownConradyTransform().distort(params, points)`` takes the 12-value vector
    ``[fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]`` and a :class:`~kornia.geometry.vector.Vector2` of normalized
    points, and returns the ``Vector2`` of distorted pixels. :meth:`undistort` maps distorted pixels back to normalized
    points by fixed-point iteration. It is the distortion of :class:`~kornia.sensors.camera.BrownConradyModel`.

    See Also:
        - :class:`KannalaBrandtK3Transform`: a fisheye distortion of the ray angle instead of the normalized radius.
        - :class:`AffineTransform`: the intrinsics alone, with no distortion.
        - :func:`~kornia.geometry.calibration.distort_points`: the same distortion on pixel tensors, with a ``3x3``
          ``K`` and every OpenCV coefficient layout; :meth:`distort` calls it with the identity as ``new_K``.
        - :func:`~kornia.geometry.calibration.undistort_points`: its inverse on pixel tensors; :meth:`undistort`
          runs its five default fixed-point steps.

    """

    def distort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Apply Brown-Conrady lens distortion to ideal normalized points.

        Args:
            params: Camera and distortion parameters ordered as
                ``(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)``.
                The first four values define the camera intrinsics and the
                trailing eight values follow the Brown-Conrady coefficient
                layout used by :func:`kornia.geometry.calibration.distort_points`.
            points: Ideal two-dimensional points on the normalized
                :math:`z = 1` image plane. Leading dimensions may represent a batch.

        Returns:
            Distorted two-dimensional points in pixel coordinates.

        """
        fx, fy, cx, cy = (
            params[..., 0],
            params[..., 1],
            params[..., 2],
            params[..., 3],
        )
        zero = torch.zeros_like(fx)
        one = torch.ones_like(fx)

        K = torch.stack(
            (
                torch.stack((fx, zero, cx), dim=-1),
                torch.stack((zero, fy, cy), dim=-1),
                torch.stack((zero, zero, one), dim=-1),
            ),
            dim=-2,
        )

        identity = torch.eye(3, device=params.device, dtype=params.dtype)

        point_data = points.data
        squeeze_point_dim = point_data.ndim == params.ndim
        if squeeze_point_dim:
            point_data = point_data.unsqueeze(-2)

        distorted = distort_points(
            point_data,
            K,
            params[..., 4:],
            new_K=identity,
        )

        if squeeze_point_dim:
            distorted = distorted.squeeze(-2)

        return Vector2(distorted)

    def undistort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Remove Brown-Conrady lens distortion from observed points.

        Args:
            params: Camera and distortion parameters ordered as
                ``(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)``.
                The layout is identical to :meth:`distort`.
            points: Distorted two-dimensional points in pixel coordinates.
                Leading dimensions may represent a batch.

        Returns:
            Undistorted two-dimensional points on the normalized
            :math:`z = 1` image plane.

        """
        fx, fy, cx, cy = (
            params[..., 0],
            params[..., 1],
            params[..., 2],
            params[..., 3],
        )
        zero = torch.zeros_like(fx)
        one = torch.ones_like(fx)

        K = torch.stack(
            (
                torch.stack((fx, zero, cx), dim=-1),
                torch.stack((zero, fy, cy), dim=-1),
                torch.stack((zero, zero, one), dim=-1),
            ),
            dim=-2,
        )

        identity = torch.eye(3, device=params.device, dtype=params.dtype)

        point_data = points.data
        squeeze_point_dim = point_data.ndim == params.ndim
        if squeeze_point_dim:
            point_data = point_data.unsqueeze(-2)

        undistorted = undistort_points(
            point_data,
            K,
            params[..., 4:],
            new_K=identity,
        )

        if squeeze_point_dim:
            undistorted = undistorted.squeeze(-2)

        return Vector2(undistorted)


class KannalaBrandtK3Transform:
    """Map normalized ``z = 1`` ``Vector2`` points to fisheye pixels with the Kannala-Brandt K3 model, and back.

    ``uv = KannalaBrandtK3Transform().distort(params, points)`` takes the ``(..., 8)`` vector
    ``[fx, fy, cx, cy, k0, k1, k2, k3]`` and a :class:`~kornia.geometry.vector.Vector2` of normalized points, and
    returns the ``Vector2`` of fisheye pixels; :meth:`undistort` inverts it iteratively. It is the distortion of
    :class:`~kornia.sensors.camera.KannalaBrandtK3`.

    See Also:
        - :class:`BrownConradyTransform`: OpenCV's distortion of the normalized radius, with tangential terms.
        - :class:`AffineTransform`: the intrinsics alone, with no distortion.
        - :func:`~kornia.geometry.camera.distort_points_kannala_brandt`: the same map on plain tensors.
        - :func:`~kornia.geometry.camera.undistort_points_kannala_brandt`: its inverse on plain tensors.

    """

    def distort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Distort normalized points to fisheye pixels with the Kannala-Brandt K3 model.

        Args:
            params: ``(..., 8)`` vector ``[fx, fy, cx, cy, k0, k1, k2, k3]``; another last dimension raises
                ``ShapeError``.
            points: ``Vector2`` of points on the normalized :math:`z = 1` plane.

        Returns:
            ``Vector2`` of the distorted pixels.

        """
        return Vector2(distort_points_kannala_brandt(points.data, params))

    def undistort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Undistort fisheye pixels to normalized points with the Kannala-Brandt K3 model.

        Args:
            params: ``(..., 8)`` vector ``[fx, fy, cx, cy, k0, k1, k2, k3]``, as in :meth:`distort`.
            points: ``Vector2`` of distorted pixels.

        Returns:
            ``Vector2`` of points on the normalized :math:`z = 1` plane.

        """
        return Vector2(undistort_points_kannala_brandt(points.data, params))

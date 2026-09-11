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
    """Apply an affine transformation to a set of 2D points.

    This class handles the scaling and shifting of coordinates, typically used
    to map normalized coordinates to pixel coordinates.
    """

    def distort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Distort one or more Vector2 points using the affine transform.

        Convention:
            - ``points`` is on the **normalized** :math:`z = 1` plane and the result is in **pixels**:
              ``u = fx * x + cx`` and ``v = fy * y + cy``, with ``params`` laid out as ``[fx, fy, cx, cy]``.
              Those pixels are on the integer-centre grid described in the Convention block on
              :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
            - it computes the same values as :func:`~kornia.geometry.camera.distort_points_affine`, which
              takes plain tensors where this method takes ``Vector2``; the two camera type systems are kept
              separate by design, which is recorded in
              `#4274 <https://github.com/kornia/kornia/issues/4274>`_.
            - :meth:`undistort` is the closed-form inverse, one subtraction and one division per axis with no
              iteration.

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
    """Implement the Brown-Conrady model for lens distortion and undistortion.

    The model accounts for radial distortion (due to lens shape) and tangential
    distortion (due to lens misalignment). It is commonly used to transform
    points between ideal pinhole projections and distorted image coordinates.

    The transform expects a 12-value parameter vector ordered as
    :math:`(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)`. The first four
    values are camera intrinsics and the trailing eight values are the
    Brown-Conrady distortion coefficients.

    :meth:`distort` maps normalized :math:`z = 1` image-plane coordinates to
    distorted pixel coordinates, while :meth:`undistort` maps distorted pixel
    coordinates back to normalized :math:`z = 1` coordinates.
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
    """Apply the Kannala-Brandt (K3) distortion model.

    This model is specifically designed for fisheye lenses with significant
    radial distortion, using a polynomial approximation for the projection.

    """

    def distort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Apply Kannala-Brandt K3 fisheye distortion to normalized points.

        Args:
            params: Fisheye distortion coefficients for the K3 polynomial model.
            points: Ideal two-dimensional normalized points before fisheye
                distortion is applied.

        Returns:
            Distorted two-dimensional points following the K3 fisheye model.

        """
        return Vector2(distort_points_kannala_brandt(points.data, params))

    def undistort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Remove Kannala-Brandt K3 fisheye distortion from observed points.

        Args:
            params: Fisheye distortion coefficients matching the K3 model used
                for distortion.
            points: Distorted two-dimensional fisheye points.

        Returns:
            Undistorted normalized points that approximate ideal pinhole
            coordinates.

        """
        return Vector2(undistort_points_kannala_brandt(points.data, params))

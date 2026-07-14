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

from kornia.geometry.vector import Vector2


class AffineTransform:
    """Apply an affine transformation to a set of 2D points.

    This class handles the scaling and shifting of coordinates, typically used
    to map normalized coordinates to pixel coordinates.
    """

    def distort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Distort one or more Vector2 points using the affine transform.

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

    Args:
        params: A tensor containing the distortion coefficients
            (usually k1, k2, p1, p2, k3).
        points: A :class:`Vector2` representing the 2D coordinates to be transformed.
    """

    def distort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Apply Brown-Conrady lens distortion to ideal normalized points.

        Args:
            params: Distortion parameter tensor, typically containing radial
                coefficients such as ``k1``, ``k2``, ``k3`` and tangential
                coefficients such as ``p1`` and ``p2``.
            points: Ideal two-dimensional normalized points before lens
                distortion. Leading dimensions may represent a batch.

        Returns:
            Distorted two-dimensional points in the same coordinate convention.

        Raises:
            NotImplementedError: The Brown-Conrady transform interface is
                declared here, but the concrete computation is not implemented.
        """
        raise NotImplementedError

    def undistort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Remove Brown-Conrady lens distortion from observed points.

        Args:
            params: Distortion parameter tensor matching the coefficients used
                by :meth:`distort`.
            points: Distorted two-dimensional points, usually measured in the
                normalized image plane.

        Returns:
            Undistorted two-dimensional points that approximate the ideal
            pinhole projection.

        Raises:
            NotImplementedError: The Brown-Conrady inverse transform interface
                is declared here, but the concrete computation is not
                implemented.
        """
        raise NotImplementedError


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

        Raises:
            NotImplementedError: The K3 distortion interface is declared here,
                but the concrete computation is not implemented.
        """
        raise NotImplementedError

    def undistort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        """Remove Kannala-Brandt K3 fisheye distortion from observed points.

        Args:
            params: Fisheye distortion coefficients matching the K3 model used
                for distortion.
            points: Distorted two-dimensional fisheye points.

        Returns:
            Undistorted normalized points that approximate ideal pinhole
            coordinates.

        Raises:
            NotImplementedError: The K3 inverse distortion interface is
                declared here, but the concrete computation is not implemented.
        """
        raise NotImplementedError

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
    """Apply an affine transformation to a set of 2D points.

    This class handles the scaling and shifting of coordinates, typically used
    to map normalized coordinates to pixel coordinates.
    """

    def distort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError

    def undistort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError


class KannalaBrandtK3Transform:
    """Apply the Kannala-Brandt (K3) distortion model.

    This model is specifically designed for fisheye lenses with significant 
    radial distortion, using a polynomial approximation for the projection.
    """

    def distort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError

    def undistort(self, params: torch.Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError

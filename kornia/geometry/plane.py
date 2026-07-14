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

# kornia.geometry.plane module inspired by Eigen::geometry::Hyperplane
# https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h

from typing import Optional

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE
from kornia.core.tensor_wrapper import _unwrap, _wrap
from kornia.core.utils import _torch_svd_cast
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.vector import Scalar, Vector3

__all__ = ["Hyperplane", "fit_plane"]


def normalized(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm_sq = (v * v).sum(dim=-1, keepdim=True) + eps
    return v * norm_sq.rsqrt()


class Hyperplane(nn.Module):
    """Represent a hyperplane in n-dimensional space.

    Args:
        n: The normal vector of the hyperplane.
        d: The scalar distance from the origin.
    """

    def __init__(self, n: Vector3, d: Scalar) -> None:
        super().__init__()
        KORNIA_CHECK_TYPE(n, Vector3)
        KORNIA_CHECK_TYPE(d, Scalar)
        # TODO: fix checkers
        # KORNIA_CHECK_SHAPE(n, ["B", "*"])
        # KORNIA_CHECK_SHAPE(d, ["B"])
        self._n = n
        self._d = d

    def __str__(self) -> str:
        return f"Normal: {self.normal}\nOffset: {self.offset}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def normal(self) -> Vector3:
        """Return the vector perpendicular to the hyperplane.

        Returns:
            :class:`~kornia.geometry.vector.Vector3` storing the normal
            direction. For a 3D plane this is the vector :math:`n` in
            :math:`n^T x + d = 0`.
        """
        return self._n

    @property
    def offset(self) -> Scalar:
        """Return the scalar offset in the implicit plane equation.

        Returns:
            :class:`~kornia.geometry.vector.Scalar` containing the ``d`` term
            in :math:`n^T x + d = 0`. The value controls where the plane sits
            relative to the origin for the stored normal direction.
        """
        return self._d

    def abs_distance(self, p: Vector3) -> Scalar:
        """Compute unsigned distances from points to the hyperplane.

        Args:
            p: Point or batch of points wrapped as
                :class:`~kornia.geometry.vector.Vector3`. The last coordinate
                dimension represents ``(x, y, z)``.

        Returns:
            :class:`~kornia.geometry.vector.Scalar` with non-negative distance
            values. Leading batch dimensions follow the broadcasted point and
            plane inputs.
        """
        return Scalar(self.signed_distance(p).abs())

    # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L145
    # TODO: tests
    def signed_distance(self, p: Vector3) -> Scalar:
        """Compute signed distances from points to the hyperplane.

        The sign is determined by the stored normal vector. Points in the
        normal direction have positive values; points on the opposite side have
        negative values; points on the plane evaluate to zero.

        Args:
            p: Point or batch of points as
                :class:`~kornia.geometry.vector.Vector3`, or a compatible
                tensor-like vector accepted by the dot-product routine.

        Returns:
            :class:`~kornia.geometry.vector.Scalar` containing signed distance
            values for each input point.
        """
        KORNIA_CHECK(isinstance(p, Vector3 | torch.Tensor))
        return self.normal.dot(p) + self.offset

    # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L154
    # TODO: tests
    def projection(self, p: Vector3) -> Vector3:
        """Project points onto the hyperplane along the normal direction.

        Args:
            p: Point or batch of points wrapped as
                :class:`~kornia.geometry.vector.Vector3`.

        Returns:
            :class:`~kornia.geometry.vector.Vector3` containing the closest
            point on the hyperplane for each input point, preserving leading
            batch dimensions.
        """
        dist = self.signed_distance(p)
        if len(dist.shape) != len(self.normal):
            # non batched plane project a batch of points
            dist = dist[..., None]  # Nx1
        # TODO: TypeError: bad operand type for unary -: 'Scalar'
        return p - dist.data * self.normal
        # TODO: make that Vector can subtract Scalar
        # return p - self.signed_distance(p) * self.normal

    @classmethod
    def from_vector(self, n: Vector3, e: Vector3) -> "Hyperplane":
        """Create a hyperplane from a normal and one point on the plane.

        Args:
            n: Normal vector :math:`n` defining the plane orientation.
            e: Point :math:`e` that lies on the target plane.

        Returns:
            :class:`Hyperplane` whose offset is chosen so that
            :math:`n^T e + d = 0`.
        """
        normal: Vector3 = n
        offset = -normal.dot(e)
        return Hyperplane(normal, Scalar(offset))

    @classmethod
    def through(cls, p0: torch.Tensor, p1: torch.Tensor, p2: Optional[torch.Tensor] = None) -> "Hyperplane":
        """Construct a line-like 2D hyperplane or a 3D plane through points.

        Args:
            p0: First point tensor, shaped ``(..., 2)`` for the 2D case or
                ``(..., 3)`` for the 3D case.
            p1: Second point tensor with the same shape as ``p0``.
            p2: Optional third point tensor. If omitted, the method builds the
                2D line representation from ``p0`` and ``p1``. If provided, it
                builds the 3D plane passing through all three points.

        Returns:
            :class:`Hyperplane` with a normal and offset determined by the
            provided point set.
        """
        # 2d case
        if p2 is None:
            # TODO: improve tests
            KORNIA_CHECK_SHAPE(p0, ["*", "2"])
            KORNIA_CHECK(p0.shape == p1.shape)
            # TODO: implement `.unitOrthonormal`
            normal2d = normalized(p1 - p0)
            offset2d = -batched_dot_product(p0, normal2d)
            return Hyperplane(_wrap(normal2d, Vector3), _wrap(offset2d, Scalar))
        # 3d case
        KORNIA_CHECK_SHAPE(p0, ["*", "3"])
        KORNIA_CHECK(p0.shape == p1.shape)
        KORNIA_CHECK(p1.shape == p2.shape)
        v0, v1 = (p2 - p0), (p1 - p0)
        normal = torch.linalg.cross(v0, v1, dim=-1)
        norm = normal.norm(-1)

        # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L108
        def compute_normal_svd(v0: torch.Tensor, v1: torch.Tensor) -> "Vector3":
            # NOTE: for reason torch.TensorWrapper does not stack well
            m = torch.stack((_unwrap(v0), _unwrap(v1)), -2)  # Bx2x3
            _, _, V = _torch_svd_cast(m)  # kornia solution lies in the last row
            return _wrap(V[..., :, -1], Vector3)  # Bx3

        normal_mask = norm <= v0.norm(-1) * v1.norm(-1) * 1e-6
        normal = torch.where(normal_mask, compute_normal_svd(v0, v1).data, normal / (norm + 1e-6))
        offset = -batched_dot_product(p0, normal)

        return Hyperplane(_wrap(normal, Vector3), _wrap(offset, Scalar))


# TODO: factor to avoid duplicated from line.py
# https://github.com/strasdat/Sophus/blob/23.04-beta/cpp/sophus/geometry/fit_plane.h
def fit_plane(points: Vector3) -> Hyperplane:
    """Fit a plane from a set of points using SVD.

    Args:
        points: tensor containing a batch of sets of n-dimensional points. The  expected
            shape of the tensor is :math:`(N, D)`.

    Return:
        The computed hyperplane object.

    """
    # TODO: fix to support more type check here
    # KORNIA_CHECK_SHAPE(points, ["N", "D"])
    if points.shape[-1] != 3:
        raise TypeError("vector must be (*, 3)")

    mean = points.mean(-2, True)
    points_centered = points - mean

    # NOTE: not optimal for 2d points, but for now works for other dimensions
    _, _, V = _torch_svd_cast(points_centered)

    # the first left eigenvector is the direction on the fited line
    direction = V[..., :, -1]  # BxD
    origin = mean[..., 0, :]  # BxD

    return Hyperplane.from_vector(Vector3(direction), Vector3(origin))

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
from kornia.core.exceptions import BaseError
from kornia.core.tensor_wrapper import _unwrap, _wrap
from kornia.core.utils import _torch_svd_cast
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.vector import Scalar, Vector3

__all__ = ["Hyperplane", "fit_plane"]


class Hyperplane(nn.Module):
    r"""Represent a plane in 3D space by its normal :math:`n` and offset :math:`d`.

    Convention:
        - The plane is :math:`n \cdot x + d = 0`, and :meth:`signed_distance` is :math:`n \cdot x + d`: positive on
          the side the normal points to, and ``d`` at the origin. :meth:`from_vector` sets :math:`d = -n \cdot e`.
          :func:`~kornia.geometry.depth.depth_from_plane_equation` takes :math:`n \cdot X = d` instead: pass it
          ``-offset``.
        - The normal must be unit for :meth:`signed_distance`, :meth:`abs_distance` and :meth:`projection` to give
          Euclidean distances and the closest point. :meth:`through` and :func:`fit_plane` return a unit normal; the
          constructor and :meth:`from_vector` do not normalise it.
        - :meth:`through` points the normal along :math:`(p_2 - p_0) \times (p_1 - p_0)`, as Eigen's
          ``Hyperplane::Through`` does: the opposite of the right-hand normal of the loop
          :math:`p_0 \to p_1 \to p_2`. Swapping two points flips it.
        - Known defects: ``normal`` and ``offset`` are not registered module state, so ``state_dict()`` is empty and
          ``.to()`` neither moves nor casts them (`#4923 <https://github.com/kornia/kornia/issues/4923>`_); in
          float16 a small triangle whose cross product underflows gets the opposite normal from :meth:`through`
          (`#5064 <https://github.com/kornia/kornia/issues/5064>`_).

    Args:
        n: The normal vector :math:`n`, a :class:`~kornia.geometry.vector.Vector3`.
        d: The offset :math:`d`, a :class:`~kornia.geometry.vector.Scalar`.
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
        return p - dist.data[..., None] * self.normal
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
        """Construct the 3D plane through three points.

        Only the three-point form is supported: :class:`Hyperplane` stores its normal as a
        :class:`~kornia.geometry.vector.Vector3`, so it cannot represent a 2D line, and calling
        ``through`` with two points raises.

        Args:
            p0: First point tensor, shaped ``(..., 3)``.
            p1: Second point tensor with the same shape as ``p0``.
            p2: Third point tensor with the same shape as ``p0``. It is required; the default of
                ``None`` only exists so that a two-point call fails with a clear error.

        Returns:
            :class:`Hyperplane` passing through the three points.

        Raises:
            BaseError: if ``p2`` is omitted, or if the points are not ``(..., 3)`` tensors of the
                same shape.
        """
        if p2 is None:
            # Raised directly rather than through KORNIA_CHECK, so it still fires with checks disabled.
            raise BaseError(
                "Hyperplane.through requires three points p0, p1 and p2 of shape (..., 3); "
                "the two-point (2D line) form is not supported."
            )

        p0_data = _unwrap(p0)
        p1_data = _unwrap(p1)
        p2_data = _unwrap(p2)

        KORNIA_CHECK_SHAPE(p0_data, ["*", "3"])
        KORNIA_CHECK(p0_data.shape == p1_data.shape)
        KORNIA_CHECK(p1_data.shape == p2_data.shape)

        v0, v1 = (p2_data - p0_data), (p1_data - p0_data)
        normal = torch.linalg.cross(v0, v1, dim=-1)

        norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        v0_norm = torch.linalg.vector_norm(v0, dim=-1, keepdim=True)
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True)

        # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L108
        def compute_normal_svd(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
            m = torch.stack((v0, v1), -2)  # Bx2x3
            _, _, V = _torch_svd_cast(m)  # kornia solution lies in the last row
            return V[..., :, -1]  # Bx3

        eps = torch.finfo(p0_data.dtype).eps if p0_data.is_floating_point() else 1e-6
        normal_mask = norm <= v0_norm * v1_norm * eps
        norm_safe = torch.where(normal_mask, torch.ones_like(norm), norm)
        normal = torch.where(normal_mask, compute_normal_svd(v0, v1), normal / norm_safe)
        offset = -batched_dot_product(p0_data, normal)

        return Hyperplane(_wrap(normal, Vector3), _wrap(offset, Scalar))


# TODO: factor to avoid duplicated from line.py
# https://github.com/strasdat/Sophus/blob/23.04-beta/cpp/sophus/geometry/fit_plane.h
def fit_plane(points: Vector3) -> Hyperplane:
    """Fit a plane from a set of points using SVD.

    Convention:
        Returns a :class:`Hyperplane` (see its conventions) through the centroid of the points, with a unit normal
        of unspecified sign. Each batch row is fitted on its own.

    Args:
        points: a tensor or a :class:`~kornia.geometry.vector.Vector3` of 3D points, of shape :math:`(N, 3)` or
            :math:`(B, N, 3)`. Another number of coordinates raises ``TypeError``.

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

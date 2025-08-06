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

# kornia.geometry.line module inspired by Eigen::geometry::ParametrizedLine
# https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/ParametrizedLine.h
from typing import Iterator, Optional, Tuple, Union

import torch

from kornia.core import Module, Parameter, Tensor, normalize, where
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.geometry.linalg import batched_dot_product, squared_norm
from kornia.geometry.plane import Hyperplane
from kornia.utils.helpers import _torch_svd_cast

__all__ = ["ParametrizedLine", "fit_line"]


class ParametrizedLine(Module):
    """Class that describes a parametrize line.

    A parametrized line is defined by an origin point :math:`o` and a unit
    direction vector :math:`d` such that the line corresponds to the set

    .. math::

        l(t) = o + t * d
    """

    def __init__(self, origin: Tensor, direction: Tensor) -> None:
        """Initialize a parametrized line of direction and origin.

        Args:
            origin: any point on the line of any dimension.
            direction: the normalized vector direction of any dimension.

        Example:
            >>> o = torch.tensor([0.0, 0.0])
            >>> d = torch.tensor([1.0, 1.0])
            >>> l = ParametrizedLine(o, d)

        """
        super().__init__()
        self._origin = Parameter(origin)
        self._direction = Parameter(direction)

    def __str__(self) -> str:
        return f"Origin: {self.origin}\nDirection: {self.direction}"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, idx: int) -> Tensor:
        return self.origin if idx == 0 else self.direction

    def __iter__(self) -> Iterator[Tensor]:
        yield from (self.origin, self.direction)

    @property
    def origin(self) -> Tensor:
        """Return the line origin point."""
        return self._origin

    @property
    def direction(self) -> Tensor:
        """Return the line direction vector."""
        return self._direction

    def dim(self) -> int:
        """Return the dimension in which the line holds."""
        return self.direction.shape[-1]

    @classmethod
    def through(cls, p0: Tensor, p1: Tensor) -> "ParametrizedLine":
        """Construct a parametrized line going from a point :math:`p0` to :math:`p1`.

        Args:
            p0: tensor with first point :math:`(B, D)` where `D` is the point dimension.
            p1: tensor with second point :math:`(B, D)` where `D` is the point dimension.

        Example:
            >>> p0 = torch.tensor([0.0, 0.0])
            >>> p1 = torch.tensor([1.0, 1.0])
            >>> l = ParametrizedLine.through(p0, p1)

        """
        return ParametrizedLine(p0, normalize((p1 - p0), p=2, dim=-1))

    def point_at(self, t: Union[float, Tensor]) -> Tensor:
        """Get the point at :math:`t` along this line.

        Args:
            t: step along the line.

        Return:
            tensor with the point.

        Example:
            >>> p0 = torch.tensor([0.0, 0.0])
            >>> p1 = torch.tensor([1.0, 1.0])
            >>> l = ParametrizedLine.through(p0, p1)
            >>> p2 = l.point_at(0.1)

        """
        return self.origin + self.direction * t

    def projection(self, point: Tensor) -> Tensor:
        """Return the projection of a point onto the line.

        Args:
            point: the point to be projected.

        """
        return self.origin + (self.direction @ (point - self.origin)) * self.direction

    # TODO: improve order and speed
    def squared_distance(self, point: Tensor) -> Tensor:
        """Return the squared distance of a point to its projection onte the line.

        Args:
            point: the point to calculate the distance onto the line.

        """
        diff: Tensor = point - self.origin
        return squared_norm(diff - (self.direction @ diff) * self.direction)

    # TODO: improve order and speed
    def distance(self, point: Tensor) -> Tensor:
        """Return the distance of a point to its projections onto the line.

        Args:
            point: the point to calculate the distance into the line.

        """
        return self.squared_distance(point).sqrt()

    # TODO(edgar) implement the following:
    # - intersection
    # - intersection_parameter
    # - intersection_point

    # TODO: add tests, and possibly return a mask
    def intersect(self, plane: Hyperplane, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
        """Return the intersection point between the line and a given plane.

        Args:
            plane: the plane to compute the intersection point.
            eps: epsilon for numerical stability.

        Return:
            - the lambda value used to compute the look at point.
            - the intersected point.

        """
        dot_prod = batched_dot_product(plane.normal.data, self.direction.data)
        dot_prod_mask = dot_prod.abs() >= eps

        # TODO: add check for dot product
        res_lambda = where(
            dot_prod_mask,
            -(plane.offset + batched_dot_product(plane.normal.data, self.origin.data)) / dot_prod,
            torch.empty_like(dot_prod),
        )

        res_point = self.point_at(res_lambda)
        return res_lambda, res_point


def _fit_line_ols_2d(points: Tensor) -> ParametrizedLine:
    x = points[..., 0]
    y = points[..., 1]
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    dx = x - x_mean
    dy = y - y_mean

    denom = (dx * dx).sum(dim=-1, keepdim=True)  # (B, 1)
    slope = torch.where(denom > 1e-8, (dx * dy).sum(dim=-1, keepdim=True) / denom, torch.zeros_like(denom))

    # For vertical lines, fallback to [0,1] direction
    direction = torch.where(
        denom > 1e-8,
        torch.cat([torch.ones_like(slope), slope], dim=-1),
        torch.tensor([0.0, 1.0], device=points.device).expand(points.shape[0], 2),
    )

    direction = direction / direction.norm(dim=-1, keepdim=True)
    origin = torch.cat([x_mean, y_mean], dim=-1)
    return ParametrizedLine(origin, direction)


def _fit_line_weighted_ols_2d(points: Tensor, weights: Tensor) -> ParametrizedLine:
    x = points[..., 0]  # (B, N)
    y = points[..., 1]  # (B, N)

    w_sum = weights.sum(dim=-1, keepdim=True)  # (B, 1)
    x_mean = (weights * x).sum(dim=-1, keepdim=True) / w_sum  # (B, 1)
    y_mean = (weights * y).sum(dim=-1, keepdim=True) / w_sum  # (B, 1)

    dx = x - x_mean  # (B, N)
    dy = y - y_mean  # (B, N)

    weighted_dx2 = weights * dx * dx
    weighted_dxdy = weights * dx * dy

    denom = weighted_dx2.sum(dim=-1, keepdim=True)  # (B, 1)
    slope = weighted_dxdy.sum(dim=-1, keepdim=True) / denom  # (B, 1)

    # Replace NaNs or infs from division by zero
    slope = torch.where(torch.isfinite(slope), slope, torch.zeros_like(slope))

    # direction = normalize([1, slope]) or [0,1] if vertical
    is_vertical = denom <= 1e-8
    direction = torch.cat([torch.ones_like(slope), slope], dim=-1)  # (B, 2)
    direction[is_vertical.squeeze(-1)] = torch.tensor([0.0, 1.0], device=points.device)

    direction = direction / direction.norm(dim=-1, keepdim=True)
    origin = torch.cat([x_mean, y_mean], dim=-1)

    return ParametrizedLine(origin, direction)


def fit_line(points: Tensor, weights: Optional[Tensor] = None) -> ParametrizedLine:
    """Fit a line from a set of points.

    Args:
        points: tensor containing a batch of sets of n-dimensional points. The expected
            shape of the tensor is :math:`(B, N, D)`.
        weights: weights to use to solve the equations system. The expected
            shape of the tensor is :math:`(B, N)`.

    Return:
        A tensor containing the direction of the fitted line of shape :math:`(B, D)`.

    Example:
        >>> points = torch.rand(2, 10, 3)
        >>> weights = torch.ones(2, 10)
        >>> line = fit_line(points, weights)
        >>> line.direction.shape
        torch.Size([2, 3])
    """
    KORNIA_CHECK_IS_TENSOR(points, "points must be a tensor")
    KORNIA_CHECK_SHAPE(points, ["B", "N", "D"])

    B, N, D = points.shape

    # Fast path: use OLS for unweighted 2D case
    if D == 2:
        if weights is not None:
            KORNIA_CHECK_IS_TENSOR(weights, "weights must be a tensor")
            KORNIA_CHECK_SHAPE(weights, ["B", "N"])
            KORNIA_CHECK(points.shape[0] == weights.shape[0])
            return _fit_line_weighted_ols_2d(points, weights)
        else:
            return _fit_line_ols_2d(points)

    mean = points.mean(-2, True)
    A = points - mean

    if weights is not None:
        KORNIA_CHECK_IS_TENSOR(weights, "weights must be a tensor")
        KORNIA_CHECK_SHAPE(weights, ["B", "N"])
        KORNIA_CHECK(points.shape[0] == weights.shape[0])
        A = A.transpose(-2, -1) @ torch.diag_embed(weights) @ A
    else:
        A = A.transpose(-2, -1) @ A

    # NOTE: not optimal for 2d points, but for now works for other dimensions
    _, _, V = _torch_svd_cast(A)
    V = V.transpose(-2, -1)

    # the first left eigenvector is the direction on the fitted line
    direction = V[..., 0, :]  # BxD
    origin = mean[..., 0, :]  # BxD

    return ParametrizedLine(origin, direction)

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
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE, are_checks_enabled
from kornia.core.exceptions import ValueCheckError
from kornia.core.utils import _torch_svd_cast, is_compiling, register_module_state
from kornia.geometry.conversions import _normalize_last_dim
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.plane import Hyperplane
from kornia.geometry.vector import Scalar

__all__ = ["ParametrizedLine", "fit_line"]


class ParametrizedLine(nn.Module):
    """Class that describes a parametrize line.

    A parametrized line is defined by an origin point :math:`o` and a unit
    direction vector :math:`d` such that the line corresponds to the set

    .. math::

        l(t) = o + t * d
    """

    def __init__(self, origin: torch.Tensor, direction: torch.Tensor) -> None:
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
        register_module_state(self, "_origin", origin)
        register_module_state(self, "_direction", direction)

    def __str__(self) -> str:
        return f"Origin: {self.origin}\nDirection: {self.direction}"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.origin if idx == 0 else self.direction

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from (self.origin, self.direction)

    @property
    def origin(self) -> torch.Tensor:
        """Return the line origin point."""
        return self._origin

    @property
    def direction(self) -> torch.Tensor:
        """Return the line direction vector."""
        return self._direction

    def dim(self) -> int:
        """Return the dimension in which the line holds."""
        return self.direction.shape[-1]

    @classmethod
    def through(cls, p0: torch.Tensor, p1: torch.Tensor) -> "ParametrizedLine":
        """Construct a parametrized line going from a point :math:`p0` to :math:`p1`.

        Args:
            p0: tensor with first point :math:`(B, D)` where `D` is the point dimension.
            p1: tensor with second point :math:`(B, D)` where `D` is the point dimension.

        Raises:
            ValueCheckError: if ``p0`` and ``p1`` coincide, so the line has no direction.

        Example:
            >>> p0 = torch.tensor([0.0, 0.0])
            >>> p1 = torch.tensor([1.0, 1.0])
            >>> l = ParametrizedLine.through(p0, p1)

        """
        direction = p1 - p0
        if not torch.jit.is_scripting() and are_checks_enabled() and not is_compiling():
            if not bool((direction.abs().amax(dim=-1) > 0).all()):
                raise ValueCheckError("ParametrizedLine.through requires two distinct points; p0 and p1 coincide.")
        return ParametrizedLine(p0, _normalize_last_dim(direction, 1e-12))

    def point_at(self, t: Union[float, torch.Tensor, Scalar]) -> torch.Tensor:
        """Get the point at :math:`t` along this line.

        Args:
            t: step along the line: a number or a 0-d tensor for every row, or one step per row of shape
                :math:`(B,)` or :math:`(B, 1)` for a line of shape :math:`(B, D)`.

        Return:
            tensor with the point.

        Example:
            >>> p0 = torch.tensor([0.0, 0.0])
            >>> p1 = torch.tensor([1.0, 1.0])
            >>> l = ParametrizedLine.through(p0, p1)
            >>> p2 = l.point_at(0.1)

        """
        if isinstance(t, Scalar):
            t = t.data
        if isinstance(t, torch.Tensor) and t.ndim > 0 and t.ndim == self.direction.ndim - 1:
            t = t[..., None]
        return self.origin + self.direction * t

    def projection(self, point: torch.Tensor) -> torch.Tensor:
        """Return the projection of a point onto the line.

        Args:
            point: the point to be projected.

        """
        return self.origin + batched_dot_product(self.direction, point - self.origin)[..., None] * self.direction

    def squared_distance(self, point: torch.Tensor) -> torch.Tensor:
        """Return the squared distance of a point to its projection onto the line.

        Args:
            point: the point to calculate the distance onto the line.
        """
        perp = self._perpendicular(point)
        return torch.sum(perp * perp, dim=-1)

    def distance(self, point: torch.Tensor) -> torch.Tensor:
        """Return the distance of a point to its projections onto the line.

        Args:
            point: the point to calculate the distance onto the line.
        """
        return torch.linalg.vector_norm(self._perpendicular(point), dim=-1)

    def _perpendicular(self, point: torch.Tensor) -> torch.Tensor:
        # The component of ``point - origin`` orthogonal to the line, row by row. Its squared norm is a sum of
        # squares, so it cannot go negative. ``||d||^2 - (d . u)^2`` cancels for points near the line.
        d = point - self.origin
        return d - torch.sum(d * self.direction, dim=-1, keepdim=True) * self.direction

    # TODO(edgar) implement the following:
    # - intersection
    # - intersection_parameter
    # - intersection_point

    # TODO: add tests, and possibly return a mask
    def intersect(self, plane: Hyperplane, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the intersection point between the line and a given plane.

        Args:
            plane: the plane to compute the intersection point.
            eps: epsilon for numerical stability.

        Return:
            - the lambda value used to compute the look at point.
            - the intersected point.

        Note:
            If the line is parallel to the plane (``|normal . direction| < eps``) there is no unique
            intersection; the function returns lambda ``0`` and the line origin as the point.

        """
        dot_prod = batched_dot_product(plane.normal.data, self.direction)
        dot_prod_mask = dot_prod.abs() >= eps

        # TODO: add check for dot product
        res_lambda = torch.where(
            dot_prod_mask,
            -(plane.offset.data + batched_dot_product(plane.normal.data, self.origin)) / dot_prod,
            torch.zeros_like(dot_prod),
        )

        res_point = self.point_at(res_lambda)
        return res_lambda, res_point


def _fit_line_ols_2d(points: torch.Tensor) -> ParametrizedLine:
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
        torch.tensor([0.0, 1.0], device=points.device, dtype=points.dtype).expand(points.shape[0], 2),
    )

    direction = direction / direction.norm(dim=-1, keepdim=True)
    origin = torch.cat([x_mean, y_mean], dim=-1)
    return ParametrizedLine(origin, direction)


def _fit_line_weighted_ols_2d(points: torch.Tensor, weights: torch.Tensor) -> ParametrizedLine:
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

    # direction = F.normalize([1, slope]) or [0,1] if vertical
    is_vertical = denom <= 1e-8
    direction = torch.cat([torch.ones_like(slope), slope], dim=-1)  # (B, 2)
    replacement = torch.tensor([0.0, 1.0], device=points.device, dtype=points.dtype)
    direction[is_vertical.squeeze(-1)] = replacement

    direction = direction / direction.norm(dim=-1, keepdim=True)
    origin = torch.cat([x_mean, y_mean], dim=-1)

    return ParametrizedLine(origin, direction)


def _reject_degenerate_line(points: torch.Tensor, weights: Optional[torch.Tensor]) -> None:
    """Raise ``ValueCheckError`` when the point set cannot determine a line.

    A line needs at least two points that are not all identical. Comparing every point with
    the first one is exact at any scale; comparing with the mean is not, because its rounding
    leaves a nonzero residual for identical points such as three copies of (0.1, 0.7). With
    weights, the weighted mean is undefined unless the weight sum of every row is positive,
    and a point with weight 0 does not count, so the same rule applies to the points with
    positive weight, compared with the point of largest weight.

    The value checks are skipped under ``torch.compile``/export, where they would be a
    data-dependent branch, and by ``disable_checks()``, like every kornia value check.
    """
    if torch.jit.is_scripting() or is_compiling() or not are_checks_enabled():
        return
    n = points.shape[-2]
    if n < 2:
        raise ValueCheckError(f"fit_line requires at least two points to determine a line; got a set of {n} point(s).")
    if not bool((points != points[..., :1, :]).flatten(-2).any(-1).all()):
        raise ValueCheckError("fit_line requires at least two distinct points; the given points are all identical.")
    # Weights of the wrong type or shape are left to the type and shape checks in fit_line.
    if isinstance(weights, torch.Tensor) and weights.shape == points.shape[:2]:
        if not bool((weights.sum(-1) > 0).all()):
            raise ValueCheckError(
                "fit_line requires a positive sum of weights; the given weights do not sum to a positive value."
            )
        # A point with weight 0 does not count: some point with positive weight must differ from the point of
        # largest weight, which is positive once the weight sum is.
        ref = points.gather(-2, weights.argmax(-1)[:, None, None].expand(-1, 1, points.shape[-1]))
        if not bool(((points != ref).any(-1) & (weights > 0)).any(-1).all()):
            raise ValueCheckError(
                "fit_line requires at least two distinct points with positive weight; the points with positive "
                "weight are all identical."
            )


def fit_line(points: torch.Tensor, weights: Optional[torch.Tensor] = None) -> ParametrizedLine:
    """Fit a line from a set of points.

    Args:
        points: tensor containing a batch of sets of n-dimensional points. The expected
            shape of the tensor is :math:`(B, N, D)`.
        weights: weights to use to solve the equations system. The expected
            shape of the tensor is :math:`(B, N)`.

    Return:
        A tensor containing the direction of the fitted line of shape :math:`(B, D)`.

    Raises:
        ValueCheckError: if the points do not determine a line — fewer than two points,
            all points identical, or (with weights) a zero weight sum or all points with
            positive weight identical.

    Example:
        >>> points = torch.rand(2, 10, 3)
        >>> weights = torch.ones(2, 10)
        >>> line = fit_line(points, weights)
        >>> line.direction.shape
        torch.Size([2, 3])
    """
    KORNIA_CHECK_IS_TENSOR(points, "points must be a tensor")
    KORNIA_CHECK_SHAPE(points, ["B", "N", "D"])

    _B, _N, D = points.shape

    _reject_degenerate_line(points, weights)

    # Fast path: use OLS for unweighted 2D case
    if D == 2:
        if weights is not None:
            KORNIA_CHECK_IS_TENSOR(weights, "weights must be a tensor")
            KORNIA_CHECK_SHAPE(weights, ["B", "N"])
            KORNIA_CHECK(points.shape[0] == weights.shape[0])
            return _fit_line_weighted_ols_2d(points, weights)
        return _fit_line_ols_2d(points)

    if weights is not None:
        KORNIA_CHECK_IS_TENSOR(weights, "weights must be a tensor")
        KORNIA_CHECK_SHAPE(weights, ["B", "N"])
        KORNIA_CHECK(points.shape[0] == weights.shape[0])
        # Weighted total least squares: centre on the weighted centroid, as the D = 2 branch does. A row whose
        # weights sum to 0 keeps the unweighted mean instead of 0 / 0, whose NaN would make the SVD raise.
        w_sum = weights.sum(-1)[..., None, None]
        has_weight = w_sum != 0
        w_sum = torch.where(has_weight, w_sum, torch.ones_like(w_sum))
        mean = torch.where(
            has_weight, (weights[..., None] * points).sum(-2, keepdim=True) / w_sum, points.mean(-2, True)
        )
        A = points - mean
        A = A.transpose(-2, -1) @ torch.diag_embed(weights) @ A
    else:
        mean = points.mean(-2, True)
        A = points - mean
        A = A.transpose(-2, -1) @ A

    # NOTE: not optimal for 2d points, but for now works for other dimensions
    _, _, V = _torch_svd_cast(A)
    V = V.transpose(-2, -1)

    # the first left eigenvector is the direction on the fitted line
    direction = V[..., 0, :]  # BxD
    origin = mean[..., 0, :]  # BxD

    return ParametrizedLine(origin, direction)

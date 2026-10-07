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
import math
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
    r"""Class that describes a parametrized line.

    A parametrized line is defined by an origin point :math:`o` and a
    direction vector :math:`d` such that the line corresponds to the set

    .. math::

        l(t) = o + t * d

    Convention:
        - The constructor does not normalise or check ``direction``, so :meth:`point_at` steps ``t`` in units of its
          length. :meth:`through` and :func:`fit_line` return a unit direction; after :meth:`through`, ``t`` is the
          Euclidean distance from ``p0``.
        - :meth:`projection`, :meth:`squared_distance` and :meth:`distance` require a unit ``direction``.
        - :meth:`intersect` returns ``(lambda, point)`` with ``point = point_at(lambda)``: ``lambda`` is in units of
          the stored direction, and the plane's normal need not be unit.
    """

    def __init__(self, origin: torch.Tensor, direction: torch.Tensor) -> None:
        """Initialize a parametrized line of direction and origin.

        Args:
            origin: any point on the line of any dimension.
            direction: the direction vector of the line, of the same dimension.

        Example:
            >>> o = torch.tensor([0.0, 0.0])
            >>> d = torch.tensor([0.6, 0.8])
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
        if (
            not torch.jit.is_scripting()
            and are_checks_enabled()
            and not is_compiling()
            and not bool((direction.abs().amax(dim=-1) > 0).all())
        ):
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
        """Return the distance of a point to its projection onto the line.

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
            eps: absolute threshold on ``|normal . direction|`` below which the line counts as parallel to the plane.

        Return:
            - the lambda value used to compute the look at point.
            - the intersected point.

        Note:
            If the line is parallel to the plane (``|normal . direction| < eps``) there is no unique
            intersection; the function returns lambda ``0`` and the line origin as the point.
            Within this fallback branch, lambda has zero derivatives and the point differentiates as the origin.

        """
        dot_prod = batched_dot_product(plane.normal.data, self.direction)
        dot_prod_mask = dot_prod.abs() >= eps

        # torch.where differentiates both branches: dividing by zero in a parallel row gives NaN gradients
        # even though its selected lambda is zero. Substitute a safe denominator before the division.
        dot_prod_safe = torch.where(dot_prod_mask, dot_prod, torch.ones_like(dot_prod))
        res_lambda = torch.where(
            dot_prod_mask,
            -(plane.offset.data + batched_dot_product(plane.normal.data, self.origin)) / dot_prod_safe,
            torch.zeros_like(dot_prod),
        )

        res_point = self.point_at(res_lambda)
        return res_lambda, res_point


def _tls_direction_2d(dx: torch.Tensor, dy: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    """Return the unit direction of the total least squares line through centred 2-D points (#5040).

    The direction is the principal axis of the (weighted) scatter of ``(dx, dy)``, in closed form:
    ``theta = 0.5 * atan2(2 * sxy, sxx - syy)`` lies in ``[-pi / 2, pi / 2]`` and the direction is
    ``(cos(theta), sin(theta))``, so its x component is non-negative. ``theta`` does not change when the points are
    scaled, so they are first divided by their largest magnitude: the second moments then cannot overflow (float16
    does past 65504) or underflow, whatever the unit of the coordinates.
    """
    scale = torch.maximum(dx.abs().amax(dim=-1, keepdim=True), dy.abs().amax(dim=-1, keepdim=True))  # (B, 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    dx = dx / scale
    dy = dy / scale
    wdx = dx if weights is None else weights * dx
    wdy = dy if weights is None else weights * dy
    sxx = (wdx * dx).sum(dim=-1, keepdim=True)
    syy = (wdy * dy).sum(dim=-1, keepdim=True)
    sxy = (wdx * dy).sum(dim=-1, keepdim=True)

    # theta and the direction in float32 for half-precision input: compiled half-precision kernels keep theta in
    # float32 but round pi / 2 to the input dtype, which gave an exactly vertical line x = -4.8e-4.
    dtype = sxy.dtype
    if dtype in (torch.float16, torch.bfloat16):
        sxx, syy, sxy = sxx.float(), syy.float(), sxy.float()
    # 0.5 * atan2(2 * sxy, sxx - syy), computed with both signs flipped. An exactly vertical set has sxy = +0: eager
    # atan2(+0, negative) is +pi, but the ONNX export's atan2 returns -pi for either zero. atan2(-0, negative) is -pi
    # in both, so theta is +pi / 2 in eager, compiled and exported graphs alike.
    theta = -0.5 * torch.atan2(-2 * sxy, sxx - syy)  # (B, 1)
    # cos(theta) as sin(pi / 2 - |theta|): float32 atan2 can return the float nearest to pi, which is larger than pi,
    # and the cosine of half of it is -4.4e-8, whereas pi / 2 - |theta| is never negative.
    return torch.cat([torch.sin(math.pi / 2 - theta.abs()), theta.sin()], dim=-1).to(dtype)


def _fit_line_tls_2d(points: torch.Tensor) -> ParametrizedLine:
    """Fit a 2-D line by total least squares (#5040).

    The points are centred relative to the first one, so a coordinate that equals the first point's is exactly 0
    after centring, whatever the rounding of the mean. An exactly vertical set then has ``sxy = +0`` and gets the
    direction ``(0, 1)``; the float32 rounding of ``pi / 2`` can leave an x component of 1.2e-7.
    """
    x0 = points[..., :1, 0]  # (B, 1)
    y0 = points[..., :1, 1]  # (B, 1)
    x = points[..., 0] - x0  # (B, N)
    y = points[..., 1] - y0  # (B, N)
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)

    direction = _tls_direction_2d(x - x_mean, y - y_mean, None)
    origin = torch.cat([x0 + x_mean, y0 + y_mean], dim=-1)
    return ParametrizedLine(origin, direction)


def _fit_line_weighted_tls_2d(points: torch.Tensor, weights: torch.Tensor) -> ParametrizedLine:
    """Fit a 2-D line by weighted total least squares: weighted centroid and weighted moments (#5040).

    Centred relative to the first point, like :func:`_fit_line_tls_2d`.
    """
    out_dtype = torch.promote_types(points.dtype, weights.dtype)
    compute_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    points, weights = points.to(compute_dtype), weights.to(compute_dtype)
    # Uniform weight scaling preserves the centroid and direction. Normalise before the sums,
    # and accumulate half-precision inputs in float32, as in the D >= 3 branch.
    weight_scale = weights.abs().amax(dim=-1, keepdim=True)
    weights = weights / torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))

    x0 = points[..., :1, 0]  # (B, 1)
    y0 = points[..., :1, 1]  # (B, 1)
    x = points[..., 0] - x0  # (B, N)
    y = points[..., 1] - y0  # (B, N)
    w_sum = weights.sum(dim=-1, keepdim=True)  # (B, 1)
    x_mean = (weights * x).sum(dim=-1, keepdim=True) / w_sum  # (B, 1)
    y_mean = (weights * y).sum(dim=-1, keepdim=True) / w_sum  # (B, 1)

    direction = _tls_direction_2d(x - x_mean, y - y_mean, weights)
    origin = torch.cat([x0 + x_mean, y0 + y_mean], dim=-1)
    return ParametrizedLine(origin.to(out_dtype), direction.to(out_dtype))


def _reject_degenerate_line(points: torch.Tensor, weights: Optional[torch.Tensor]) -> None:
    """Raise ``ValueCheckError`` when the point set cannot determine a line.

    A line needs at least two points that are not all identical. Comparing every point with
    the first one is exact at any scale; comparing with the mean is not, because its rounding
    leaves a nonzero residual for identical points such as three copies of (0.1, 0.7). With
    weights, the weighted mean is undefined unless the weight sum of every row is positive,
    and a point with weight 0 does not count, so the same rule applies to the points with
    positive weight, compared with the point of largest weight. Every weight must be
    non-negative: a negative weight can make the weighted scatter indefinite, and then the
    2-D and the D >= 3 branches disagree.

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
        if bool((weights < 0).any()):
            raise ValueCheckError(
                "fit_line requires non-negative weights; a negative weight can make the weighted scatter indefinite."
            )


def fit_line(points: torch.Tensor, weights: Optional[torch.Tensor] = None) -> ParametrizedLine:
    r"""Fit a line from a set of points by total least squares.

    Convention:
        - Returns a :class:`ParametrizedLine` (see its conventions) through the centroid of the points, weighted
          when ``weights`` are given, whose unit direction minimises the (weighted) sum of squared perpendicular
          distances to the points, for every dimensionality. Each batch row is fitted on its own.
        - For 2-D points the direction is computed in closed form and has a non-negative x component, so an exactly
          vertical line gets the direction ``(0, 1)`` up to rounding. For :math:`D \ge 3` it is the principal
          direction of the scatter matrix, whose sign is not specified.

    Args:
        points: tensor containing a batch of sets of n-dimensional points. The expected
            shape of the tensor is :math:`(B, N, D)`.
        weights: per-point weights, used in the centroid and in the fit. The expected
            shape of the tensor is :math:`(B, N)`.

    Return:
        The fitted line, with origin and direction of shape :math:`(B, D)`.

    Raises:
        ValueCheckError: if the points do not determine a line — fewer than two points,
            all points identical, or (with weights) a zero weight sum, a negative weight, or all
            points with positive weight identical.

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

    # Fast path: closed-form total least squares for the 2D case
    if D == 2:
        if weights is not None:
            KORNIA_CHECK_IS_TENSOR(weights, "weights must be a tensor")
            KORNIA_CHECK_SHAPE(weights, ["B", "N"])
            KORNIA_CHECK(points.shape[0] == weights.shape[0])
            return _fit_line_weighted_tls_2d(points, weights)
        return _fit_line_tls_2d(points)

    # The scatter matrix is quadratic in the coordinates. Form it from centred, unit-scale
    # offsets in a compute dtype before SVD; casting an already-overflowed half matrix cannot recover it.
    if weights is not None:
        KORNIA_CHECK_IS_TENSOR(weights, "weights must be a tensor")
        KORNIA_CHECK_SHAPE(weights, ["B", "N"])
        KORNIA_CHECK(points.shape[0] == weights.shape[0])
    # The output keeps the promoted dtype of points and weights, as in the D = 2 branch.
    out_dtype = points.dtype if weights is None else torch.promote_types(points.dtype, weights.dtype)
    compute_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    work_points = points.to(compute_dtype)

    if weights is not None:
        work_weights = weights.to(compute_dtype)
        weight_scale = work_weights.abs().amax(dim=-1, keepdim=True)
        work_weights = work_weights / torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
        # Weighted total least squares: centre on the weighted centroid, as the D = 2 branch does. A row whose
        # weights sum to 0 keeps the unweighted mean instead of 0 / 0, whose NaN would make the SVD raise.
        w_sum = work_weights.sum(-1)[..., None, None]
        has_weight = w_sum != 0
        w_sum = torch.where(has_weight, w_sum, torch.ones_like(w_sum))
        mean = torch.where(
            has_weight,
            (work_weights[..., None] * work_points).sum(-2, keepdim=True) / w_sum,
            work_points.mean(-2, True),
        )
    else:
        mean = work_points.mean(-2, True)

    A = work_points - mean
    scale = A.abs().amax(dim=(-2, -1), keepdim=True)
    A = A / torch.where(scale > 0, scale, torch.ones_like(scale))
    A = A.transpose(-2, -1) @ torch.diag_embed(work_weights) @ A if weights is not None else A.transpose(-2, -1) @ A

    # NOTE: not optimal for 2d points, but for now works for other dimensions
    _, _, V = _torch_svd_cast(A)
    V = V.transpose(-2, -1)

    # the first left eigenvector is the direction on the fitted line
    direction = V[..., 0, :].to(out_dtype)  # BxD
    origin = mean[..., 0, :].to(out_dtype)  # BxD

    return ParametrizedLine(origin, direction)

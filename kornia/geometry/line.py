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

r"""Lines in any dimension as origin plus direction: distances, plane intersection and fitting.

A line is :math:`l(t) = o + t d`, with an origin ``o`` and a direction ``d`` of shape ``(D,)`` for one line or
``(B, D)`` for ``B`` lines, in any dimension ``D``. The constructor stores the direction as given; the distances and
the projection need it to be unit, which :meth:`~kornia.geometry.line.ParametrizedLine.through` and :func:`fit_line`
return. ``kornia.geometry.ray.Ray`` is the same class as :class:`ParametrizedLine`.

Which function to use:

- A line from an origin and a direction, or through two points, with the points along it, the projection and distance
  of points, and the intersection with a 3D plane: :class:`ParametrizedLine`.
- The line that best fits each of ``B`` sets of points, by total least squares, optionally weighted: :func:`fit_line`.
- A plane in 3D, and the plane that best fits 3D points: :class:`~kornia.geometry.plane.Hyperplane` and
  :func:`~kornia.geometry.plane.fit_plane`.
- The distance from 2D points to lines given as coefficients :math:`ax + by + c = 0`, without a line object:
  :func:`~kornia.geometry.linalg.point_line_distance`.
"""

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
    r"""Represent a line o + t d in any dimension, with distances and plane intersection.

    ``line = ParametrizedLine(origin, direction)`` stores an origin and a direction of shape ``(D,)``, or ``(B, D)``
    for ``B`` lines, in any dimension ``D``, as given; ``ParametrizedLine.through(p0, p1)`` builds the line from
    ``p0`` towards ``p1`` with a unit direction. ``line.point_at(t)`` returns the points :math:`o + t d`; for a unit
    direction, ``line.projection(p)`` returns the closest points on the line ``(..., D)`` and ``line.distance(p)``
    the distances ``(...)`` of points ``(..., D)``; ``line.intersect(plane)`` returns where a 3D line meets a
    :class:`~kornia.geometry.plane.Hyperplane`. To fit the line to many points, use :func:`fit_line`; for 2D lines
    given as coefficients :math:`ax + by + c = 0`, :func:`~kornia.geometry.linalg.point_line_distance`.

    The line is the set

    .. math::

        l(t) = o + t * d

    Convention:
        - The constructor does not normalise or check ``direction``, so :meth:`point_at` steps ``t`` in units of its
          length. :meth:`through` and :func:`fit_line` return a unit direction; after :meth:`through`, ``t`` is the
          Euclidean distance from ``p0``.
        - :meth:`projection`, :meth:`squared_distance` and :meth:`distance` require a unit ``direction``.
        - Points broadcast against the lines as torch broadcasts ``(..., D)`` against ``(B, D)``: the axis before
          the coordinates pairs with the line batch. One line, or a batch of one, takes points of any leading shape;
          a batch of ``B > 1`` takes ``(B, D)``, one point per line, a single ``(D,)`` point, or ``(N, B, D)`` for
          ``N`` points per line. ``(B, N, D)`` is not ``N`` points per line: it raises unless ``N`` is ``B`` or 1.
          With ``N = B``, line ``j`` meets ``point[:, j]``; with ``N = 1``, every line meets every point and the
          result is ``(B, B)``.
        - :meth:`intersect` returns ``(lambda, point)`` with ``point = point_at(lambda)``: ``lambda`` is in units of
          the stored direction, and the plane's normal need not be unit. It takes a 3D line. A batch of ``B`` lines
          meets one plane, a batch of ``B`` planes or a batch of one, and a batch of one line meets a batch of
          planes of any size. One unbatched line meets one plane or a batch of one, but with a batch of ``M > 1``
          planes it raises ``RuntimeError``, except for ``M = 3``, where it silently returns one wrong point
          (`#6483 <https://github.com/kornia/kornia/issues/6483>`_); build the line from ``origin[None]`` and
          ``direction[None]`` instead.
        - ``origin`` and ``direction`` are registered module state, so ``state_dict()`` and ``.to()`` include them.
          Outside graph capture, a floating-point tensor that takes no part in autograd (no ``grad_fn`` and
          ``requires_grad=False``) becomes an ``nn.Parameter``, so a line built from plain tensors, including the
          result of :meth:`through` and :func:`fit_line` on such tensors, returns points and distances that require
          grad. A caller's ``nn.Parameter`` is kept; any other floating-point tensor, such as a leaf that requires
          grad or a tensor with autograd history, is stored as a buffer and keeps its autograd link; an integer or
          bool tensor raises ``TypeCheckError``. ``copy.deepcopy`` of a line whose state carries autograd history
          raises ``RuntimeError`` (`#4923 <https://github.com/kornia/kornia/issues/4923>`_).

    See Also:
        - :func:`fit_line`: fits the line to a set of points by total least squares, instead of taking an origin
          and a direction.
        - :class:`~kornia.geometry.plane.Hyperplane`: a plane in 3D, :math:`n \cdot x + d = 0`, the operand of
          :meth:`intersect`.
        - :func:`~kornia.geometry.linalg.point_line_distance`: the distance from 2D points to lines given as
          coefficients :math:`(a, b, c)`, without a line object.

    Example:
        >>> line = ParametrizedLine.through(torch.tensor([0.0, 0.0]), torch.tensor([3.0, 4.0]))
        >>> line.point_at(10.0)
        tensor([6., 8.], grad_fn=<AddBackward0>)
        >>> line.distance(torch.tensor([[4.0, 3.0], [-3.0, -4.0]]))
        tensor([1.4000, 0.0000], grad_fn=<LinalgVectorNormBackward0>)

    """

    def __init__(self, origin: torch.Tensor, direction: torch.Tensor) -> None:
        """Initialize a parametrized line of direction and origin.

        Args:
            origin: a point on the line, ``(D,)`` or ``(B, D)``, in any dimension ``D``.
            direction: the direction vector of the line, of the same shape; stored as given, neither normalised nor
                checked against ``origin``.

        Raises:
            TypeCheckError: if ``origin`` or ``direction`` is an integer or bool tensor.

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
        """Return the dimension ``D`` of the space the line lies in, the size of the last axis of the direction."""
        return self.direction.shape[-1]

    @classmethod
    def through(cls, p0: torch.Tensor, p1: torch.Tensor) -> "ParametrizedLine":
        """Construct a parametrized line going from a point :math:`p0` to :math:`p1`.

        The origin is ``p0`` and the direction the unit vector from ``p0`` towards ``p1``, so ``point_at(t)`` is the
        point at Euclidean distance ``t`` from ``p0``.

        Args:
            p0: tensor with first point ``(B, D)`` or ``(D,)``, where ``D`` is the point dimension.
            p1: tensor with second point of the same shape.

        Raises:
            ValueCheckError: if ``p0`` and ``p1`` coincide in any row, so the line has no direction. Like every kornia
                value check, it is skipped under ``torch.compile`` and by ``disable_checks()``.

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
            t: step along the line, in units of the direction's length: a number or a 0-d tensor for every row, or
                one step per row of shape ``(B,)`` or ``(B, 1)`` for a line of shape ``(B, D)``. For ``N`` points on
                one line of shape ``(D,)``, pass ``(N, 1)``: a ``(1,)`` step gives one ``(D,)`` point, an ``(N,)``
                step raises for ``N`` other than 1 and ``D``, and with ``N = D`` it multiplies each coordinate of the
                direction by its own value (`#6483 <https://github.com/kornia/kornia/issues/6483>`_).

        Return:
            the points :math:`o + t d`, ``(D,)`` or ``(B, D)``, or ``(N, D)`` for an ``(N, 1)`` step.

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

        The projection is the closest point on the line only for a unit direction.

        Args:
            point: the point to be projected, ``(..., D)``, broadcast against the lines as the class conventions
                say.

        Return:
            the projected points, of the broadcast shape ``(..., D)``.

        """
        return self.origin + batched_dot_product(self.direction, point - self.origin)[..., None] * self.direction

    def squared_distance(self, point: torch.Tensor) -> torch.Tensor:
        """Return the squared distance of a point to its projection onto the line.

        It is the squared Euclidean distance to the line only for a unit direction.

        Args:
            point: the point to calculate the distance onto the line, ``(..., D)``.

        Return:
            the squared distances, of the broadcast leading shape ``(...)``.
        """
        perp = self._perpendicular(point)
        return torch.sum(perp * perp, dim=-1)

    def distance(self, point: torch.Tensor) -> torch.Tensor:
        """Return the distance of a point to its projection onto the line.

        It is the Euclidean distance to the line only for a unit direction.

        Args:
            point: the point to calculate the distance onto the line, ``(..., D)``.

        Return:
            the distances, of the broadcast leading shape ``(...)``.
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
            - lambda, the line parameter of the intersection in units of the stored direction: ``()`` for one line
              and one plane, ``(B,)`` for the broadcast batch, and ``(1,)`` for one unbatched line with a batch of
              one plane.
            - the intersection point ``point_at(lambda)``: ``(3,)``, or ``(B, 3)`` for a batch of lines; see the
              class conventions for one unbatched line with a batch of planes.

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
    dtype = points.dtype
    # Promote before subtracting or accumulating: finite half coordinates can have an overflowing difference,
    # and even unit-scale second moments overflow float16 for a sufficiently large point set.
    if dtype in (torch.float16, torch.bfloat16):
        points = points.float()
    x0 = points[..., :1, 0]  # (B, 1)
    y0 = points[..., :1, 1]  # (B, 1)
    x = points[..., 0] - x0  # (B, N)
    y = points[..., 1] - y0  # (B, N)
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)

    direction = _tls_direction_2d(x - x_mean, y - y_mean, None)
    origin = torch.cat([x0 + x_mean, y0 + y_mean], dim=-1)
    return ParametrizedLine(origin.to(dtype), direction.to(dtype))


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
    r"""Fit a line to each of B point sets in any dimension by weighted total least squares.

    ``line = fit_line(points, weights)`` takes ``B`` sets of ``N`` points ``(B, N, D)`` in any dimension ``D``, and
    optional non-negative per-point weights ``(B, N)``, and returns a :class:`ParametrizedLine` whose origin and unit
    direction are ``(B, D)``: for each set, the line through the centroid that minimises the sum of squared
    perpendicular distances, both weighted when ``weights`` are given. A single set ``(N, D)`` raises ``ShapeError``:
    pass ``points[None]``. Every point with a positive weight pulls on the fit, so down-weight outliers or remove them
    first. For a plane through 3D points, use :func:`~kornia.geometry.plane.fit_plane` instead.

    Convention:
        - Returns a :class:`ParametrizedLine` (see its conventions) through the centroid of the points, weighted
          when ``weights`` are given, whose unit direction minimises the (weighted) sum of squared perpendicular
          distances to the points, for every dimensionality. Each batch row is fitted on its own.
        - For 2-D points the direction is computed in closed form and has a non-negative x component, so an exactly
          vertical line gets the direction ``(0, 1)`` up to rounding. For :math:`D \ge 3` it is the principal
          direction of the scatter matrix, whose sign is not specified.
        - The output has the promoted dtype of ``points`` and ``weights``; float16 and bfloat16 inputs are fitted in
          float32 and the line is cast back.

    Args:
        points: tensor containing a batch of sets of n-dimensional points. The expected
            shape of the tensor is ``(B, N, D)``.
        weights: per-point weights, used in the centroid and in the fit. The expected
            shape of the tensor is ``(B, N)``.

    Return:
        The fitted line, with origin and direction of shape ``(B, D)``.

    Raises:
        TypeCheckError: if ``points`` or ``weights`` is not a tensor.
        ShapeError: if ``points`` is not 3-D, or ``weights`` is not 2-D.
        BaseError: if ``weights`` has another batch size than ``points``.
        RuntimeError: if ``weights`` has another number of points than ``points``, from torch.
        ValueCheckError: if the points do not determine a line — fewer than two points,
            all points identical, or (with weights) a zero weight sum, a negative weight, or all
            points with positive weight identical. Like every kornia value check, these are skipped under
            ``torch.compile`` and by ``disable_checks()``.

    See Also:
        - :class:`ParametrizedLine`: the returned line, which can also be built from an origin and a direction or
          through two points.
        - :func:`~kornia.geometry.plane.fit_plane`: the same total least squares for a plane through 3D points,
          unweighted, and for a set ``(N, 3)`` as well as a batch.

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
    # When weighted, scale the columns directly instead of allocating an N-by-N diagonal weight matrix.
    A = (A.transpose(-2, -1) * work_weights[..., None, :]) @ A if weights is not None else A.transpose(-2, -1) @ A

    # NOTE: not optimal for 2d points, but for now works for other dimensions
    _, _, V = _torch_svd_cast(A)
    V = V.transpose(-2, -1)

    # the first left eigenvector is the direction on the fitted line
    direction = V[..., 0, :].to(out_dtype)  # BxD
    origin = mean[..., 0, :].to(out_dtype)  # BxD

    return ParametrizedLine(origin, direction)

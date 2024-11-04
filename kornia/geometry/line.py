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
        """Initializes a parametrized line of direction and origin.

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
        """Constructs a parametrized line going from a point :math:`p0` to :math:`p1`.

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
        """The point at :math:`t` along this line.

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


def fit_line(points: Tensor, weights: Optional[Tensor] = None) -> ParametrizedLine:
    """Fit a line from a set of points.

    Args:
        points: tensor containing a batch of sets of n-dimensional points. The  expected
            shape of the tensor is :math:`(B, N, D)`.
        weights: weights to use to solve the equations system. The  expected
            shape of the tensor is :math:`(B, N)`.

    Return:
        A tensor containing the direction of the fited line of shape :math:`(B, D)`.

    Example:
        >>> points = torch.rand(2, 10, 3)
        >>> weights = torch.ones(2, 10)
        >>> line = fit_line(points, weights)
        >>> line.direction.shape
        torch.Size([2, 3])
    """
    KORNIA_CHECK_IS_TENSOR(points, "points must be a tensor")
    KORNIA_CHECK_SHAPE(points, ["B", "N", "D"])

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

    # the first left eigenvector is the direction on the fited line
    direction = V[..., 0, :]  # BxD
    origin = mean[..., 0, :]  # BxD

    return ParametrizedLine(origin, direction)

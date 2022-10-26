# kornia.geometry.plane module inspired by Eigen::geometry::Hyperplane
# https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h

from typing import Optional

from kornia.core import Module, Parameter, Tensor, normalize
from kornia.testing import KORNIA_CHECK_SHAPE

__all__ = ["Hyperplane"]


class Hyperplane(Module):
    def __init__(self, n: Tensor, d: Tensor) -> None:
        super().__init__()
        KORNIA_CHECK_SHAPE(n, ["B", "*"])
        KORNIA_CHECK_SHAPE(d, ["B", "*"])
        self._n = Parameter(n)
        self._d = Parameter(d)

    @property
    def normal(self) -> Tensor:
        return self._n

    @property
    def offset(self) -> Tensor:
        return self._d

    @classmethod
    def through(cls, p0: Tensor, p1: Tensor, p2: Optional[Tensor] = None) -> "Hyperplane":
        # 2d case
        if p2 is None:
            KORNIA_CHECK_SHAPE(p0, ["B", "2"])
            KORNIA_CHECK_SHAPE(p1, ["B", "2"])
            normal = normalize((p1 - p2), p=2, dim=-1)
            offset = -(p0 @ normal)
            return Hyperplane(normal, offset)
        # 3d case
        KORNIA_CHECK_SHAPE(p0, ["B", "3"])
        KORNIA_CHECK_SHAPE(p1, ["B", "3"])
        KORNIA_CHECK_SHAPE(p2, ["B", "3"])
        v0, v1 = (p2 - p1), (p1 - p0)
        normal = v0.cross(v1)
        norm = normal.norm(-1)
        # TODO: use where
        # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L108
        # general case
        normal = normal / norm
        offset = -(p0 @ normal)
        return Hyperplane(normal, offset)

    # TODO: implement me
    # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L125
    @classmethod
    def from_line(cls, line) -> "Hyperplane":
        pass

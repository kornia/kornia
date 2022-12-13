# kornia.geometry.plane module inspired by Eigen::geometry::Hyperplane
# https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h

from typing import Optional

from kornia.core import Module, Tensor, normalize
from kornia.geometry.linalg import batched_dot_product
from kornia.testing import KORNIA_CHECK_SHAPE
from kornia.utils.helpers import _torch_svd_cast

__all__ = ["Hyperplane", "fit_plane"]


# NOTE: in the near future the constructor will change
# class Hyperplane(Module):
#     def __init__(self, n: _VectorType, d: _VectorType) -> None:
#         super().__init__()
#         self._n = n
#         self._d = d


class Hyperplane(Module):
    def __init__(self, n: Tensor, d: Tensor) -> None:
        super().__init__()
        # KORNIA_CHECK_SHAPE(n, ["B", "*"])  # FIXME: resolve shape bugs. @edgarriba
        # KORNIA_CHECK_SHAPE(d, ["B"])  # FIXME: resolve shape bugs. @edgarriba
        self._n = n
        self._d = d

    def __str__(self) -> str:
        return f"Normal: {self.normal}\nOffset: {self.offset}"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, idx) -> Tensor:
        return self.normal if idx == 0 else self.offset

    def __iter__(self):
        yield from (self.normal, self.offset)

    @property
    def normal(self) -> Tensor:
        return self._n

    @property
    def offset(self) -> Tensor:
        return self._d

    @classmethod
    def from_vector(self, n: Tensor, e: Tensor) -> "Hyperplane":
        normal = n
        offset = -batched_dot_product(normal, e)
        return Hyperplane(normal, offset)

    @classmethod
    def through(cls, p0: Tensor, p1: Tensor, p2: Optional[Tensor] = None) -> "Hyperplane":
        # 2d case
        if p2 is None:
            KORNIA_CHECK_SHAPE(p0, ["B", "2"])
            KORNIA_CHECK_SHAPE(p1, ["B", "2"])
            # TODO: implement `.unitOrthonormal`
            normal = normalize((p1 - p0), p=2, dim=-1)
            offset = -batched_dot_product(p0, normal)
            return Hyperplane(normal, offset)
        # 3d case
        # KORNIA_CHECK_SHAPE(p0, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
        # KORNIA_CHECK_SHAPE(p1, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
        # KORNIA_CHECK_SHAPE(p2, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
        v0, v1 = (p2 - p1), (p1 - p0)
        normal = v0.cross(v1)
        norm = normal.norm(-1)
        # TODO: use where
        # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L108
        # general case
        normal = normal / (norm + 1e-6)
        offset = -batched_dot_product(p0, normal)
        return Hyperplane(normal, offset)


# TODO: factor to avoid duplicated from line.py
# https://github.com/strasdat/Sophus/blob/23.04-beta/cpp/sophus/geometry/fit_plane.h
def fit_plane(points: Tensor) -> Hyperplane:
    """Fit a plane from a set of points using SVD.

    Args:
        points: tensor containing a batch of sets of n-dimensional points. The  expected
            shape of the tensor is :math:`(B, N, D)`.

    Return:
        The computed hyperplane object.
    """
    KORNIA_CHECK_SHAPE(points, ["B", "N", "D"])

    mean = points.mean(-2, True)
    points_centered = points - mean

    # NOTE: not optimal for 2d points, but for now works for other dimensions
    _, _, V = _torch_svd_cast(points_centered)
    V = V.transpose(-2, -1)

    # the first left eigenvector is the direction on the fited line
    direction = V[..., 0, :]  # BxD
    origin = mean[..., 0, :]  # BxD

    return Hyperplane.from_vector(direction, origin)

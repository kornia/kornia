# kornia.geometry.plane module inspired by Eigen::geometry::Hyperplane
# https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h

from typing import Optional

from kornia.core import Module, Tensor, stack, where
from kornia.core.tensor_wrapper import unwrap, wrap
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.vector import Scalar, Vector3
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE
from kornia.utils.helpers import _torch_svd_cast

__all__ = ["Hyperplane", "fit_plane"]


def normalized(v, eps=1e-6):
    return v / batched_dot_product(v, v).add(eps).sqrt()


class Hyperplane(Module):
    def __init__(self, n: Vector3, d: Scalar) -> None:
        super().__init__()
        KORNIA_CHECK_TYPE(n, Vector3)
        KORNIA_CHECK_TYPE(d, Scalar)
        # TODO: fix checkers
        # KORNIA_CHECK_SHAPE(n, ["B", "*"])
        # KORNIA_CHECK_SHAPE(d, ["B"])
        self._n: Vector3 = n
        self._d: Scalar = d

    def __str__(self) -> str:
        return f"Normal: {self.normal}\nOffset: {self.offset}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def normal(self) -> Vector3:
        return self._n

    @property
    def offset(self) -> Scalar:
        return self._d

    def abs_distance(self, p: Vector3) -> Scalar:
        return Scalar(self.signed_distance(p).abs())

    # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L145
    # TODO: tests
    def signed_distance(self, p: Vector3) -> Scalar:
        KORNIA_CHECK(isinstance(p, (Vector3, Tensor)))
        return self.normal.dot(p) + self.offset

    # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L154
    # TODO: tests
    def projection(self, p: Vector3) -> Vector3:
        return Vector3(p - (self.signed_distance(p).view(-1, 1) * self.normal).data)
        # TODO: make that Vector can subtract Scalar
        # return p - self.signed_distance(p) * self.normal

    @classmethod
    def from_vector(self, n: Vector3, e: Vector3) -> "Hyperplane":
        normal: Vector3 = n
        # TODO: implement as below
        # offset: Scalar = -normal.dot(e)
        offset: Scalar = -batched_dot_product(normal, e)
        return Hyperplane(normal, Scalar(offset))

    @classmethod
    def through(cls, p0: Tensor, p1: Tensor, p2: Optional[Tensor] = None) -> "Hyperplane":
        # 2d case
        if p2 is None:
            # TODO: improve tests
            KORNIA_CHECK_SHAPE(p0, ["*", "2"])
            KORNIA_CHECK(p0.shape == p1.shape)
            # TODO: implement `.unitOrthonormal`
            normal: Vector3 = normalized(p1 - p0)
            offset: Tensor = -batched_dot_product(p0, normal)
            return Hyperplane(normal, Scalar(offset))
        # 3d case
        KORNIA_CHECK_SHAPE(p0, ["*", "3"])
        KORNIA_CHECK(p0.shape == p1.shape)
        KORNIA_CHECK(p1.shape == p2.shape)
        v0, v1 = (p2 - p0), (p1 - p0)
        normal: Vector3 = v0.cross(v1)
        norm = normal.norm(-1)

        # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Hyperplane.h#L108
        def compute_normal_svd(v0, v1):
            # NOTE: for reason TensorWrapper does not stack well
            m = stack((unwrap(v0), unwrap(v1)), -2)  # Bx2x3
            _, _, V = _torch_svd_cast(m)  # kornia solution lies in the last row
            return wrap(V[..., -1, :], Vector3)  # Bx3

        normal_mask = norm <= v0.norm(-1) * v1.norm(-1) * 1e-6
        normal = where(normal_mask, compute_normal_svd(v0, v1), normal / (norm + 1e-6))
        offset = -batched_dot_product(p0, normal)
        # NOTE: make sure  offset is tensor
        return Hyperplane(normal, Scalar(offset))


# TODO: factor to avoid duplicated from line.py
# https://github.com/strasdat/Sophus/blob/23.04-beta/cpp/sophus/geometry/fit_plane.h
def fit_plane(points: Vector3) -> Hyperplane:
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
    U, S, V = _torch_svd_cast(points_centered)

    # the first left eigenvector is the direction on the fited line
    direction = V[..., -1, :]  # BxD
    origin = mean[..., 0, :]  # BxD

    return Hyperplane.from_vector(Vector3(direction), Vector3(origin))

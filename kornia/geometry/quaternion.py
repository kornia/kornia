from math import pi
from typing import Tuple

from kornia.core import Module, Parameter, Tensor, as_tensor, concatenate, cos, rand, sin, sqrt, stack
from kornia.testing import KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE

# NOTE: discuss whether is more appropiated to inherit from Tensor


class Quaternion(Module):
    def __init__(self, data: Tensor) -> None:
        super().__init__()
        # NOTE: discuss whether we want to support more dimensions
        KORNIA_CHECK_SHAPE(data, ["B", "4"])
        self._data = Parameter(data)

    def __repr__(self) -> str:
        return f"real: {self.real} \nvec: {self.vec}"

    def __getitem__(self, idx):
        return self.data[idx]

    def __neg__(self) -> 'Quaternion':
        return Quaternion(-self.data)

    def __add__(self, right: 'Quaternion') -> 'Quaternion':
        KORNIA_CHECK_TYPE(right, Quaternion)
        return Quaternion(self.data + right.data)

    def __sub__(self, right: 'Quaternion') -> 'Quaternion':
        KORNIA_CHECK_TYPE(right, Quaternion)
        return Quaternion(self.data - right.data)

    def __mul__(self, right: 'Quaternion') -> 'Quaternion':
        KORNIA_CHECK_TYPE(right, Quaternion)
        # NOTE: borrowed from sophus sympy
        # https://github.com/strasdat/Sophus/blob/785fef35b7d9e0fc67b4964a69124277b7434a44/sympy/sophus/quaternion.py#L19
        new_real = self.real * right.real - self.vec @ right.vec.T
        new_vec = self.real * right.vec + right.real * self.vec + self.vec.cross(right.vec)
        return Quaternion(concatenate((new_real, new_vec), -1))

    def __div__(self, right: 'Quaternion') -> 'Quaternion':
        if isinstance(right, Tensor):
            return Quaternion(self.data / right)
        KORNIA_CHECK_TYPE(right, Quaternion)
        return self * right.inv()

    def __truediv__(self, right: 'Quaternion') -> 'Quaternion':
        return self.__div__(right)

    @property
    def data(self) -> Tensor:
        return self._data

    @property
    def real(self) -> Tensor:
        return self.w

    @property
    def vec(self) -> Tensor:
        return self.data[..., 1:]

    @property
    def v(self) -> Tensor:
        return self.vec

    @property
    def q(self) -> Tensor:
        return self.data

    @property
    def scalar(self) -> Tensor:
        return self.real

    @property
    def w(self) -> Tensor:
        return self.data[..., 0:1]

    @property
    def x(self) -> Tensor:
        return self.data[..., 1:2]

    @property
    def y(self) -> Tensor:
        return self.data[..., 2:3]

    @property
    def z(self) -> Tensor:
        return self.data[..., 3:4]

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    def matrix(self):
        from kornia.geometry.conversions import QuaternionCoeffOrder, quaternion_to_rotation_matrix

        return quaternion_to_rotation_matrix(self.data, order=QuaternionCoeffOrder.WXYZ)

    @classmethod
    def identity(cls, batch_size: int) -> Tensor:
        data: Tensor = as_tensor([1.0, 0.0, 0.0, 0.0])
        data = data.repeat(batch_size, 1)
        return cls(data)

    @classmethod
    def from_coeffs(cls, scalar: float, x: float, y: float, z: float) -> 'Quaternion':
        return cls(as_tensor([[scalar, x, y, z]]))

    @classmethod
    def random(cls, batch_size: int) -> 'Quaternion':
        """Generate a random unit quaternion.

        Uniformly distributed across the rotation space As per: http://planning.cs.uiuc.edu/node198.html
        """
        r1, r2, r3 = rand(3, batch_size)
        q1 = sqrt(1.0 - r1) * (sin(2 * pi * r2))
        q2 = sqrt(1.0 - r1) * (cos(2 * pi * r2))
        q3 = sqrt(r1) * (sin(2 * pi * r3))
        q4 = sqrt(r1) * (cos(2 * pi * r3))
        return cls(stack((q1, q2, q3, q4), -1))

    def norm(self) -> Tensor:
        return self.data.norm(p=2, dim=-1)

    def conj(self) -> 'Quaternion':
        return Quaternion(concatenate((self.real, -self.vec), -1))

    def inv(self) -> 'Quaternion':
        return self.conj() / self.squared_norm()

    def squared_norm(self) -> Tensor:
        return self._batched_squared_norm(self.vec) + self.real**2

    def _batched_squared_norm(self, x):
        return (x[..., None, :] @ x[..., :, None])[..., 0]

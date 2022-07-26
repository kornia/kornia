from math import pi
from typing import Tuple, Union

from kornia.core import Module, Parameter, Tensor, as_tensor, concatenate, rand, stack
from kornia.geometry.conversions import (
    QuaternionCoeffOrder,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from kornia.testing import KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE


class Quaternion(Module):
    def __init__(self, data: Tensor) -> None:
        super().__init__()
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
        new_real = self.real * right.real - self._batched_squared_norm(self.vec, right.vec)
        new_vec = self.real * right.vec + right.real * self.vec + self.vec.cross(right.vec)
        return Quaternion(concatenate((new_real, new_vec), -1))

    def __div__(self, right: Union[Tensor, 'Quaternion']) -> 'Quaternion':
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
    def coeffs(self) -> Tensor:
        return self._data

    @property
    def real(self) -> Tensor:
        return self.w

    @property
    def vec(self) -> Tensor:
        return self.data[..., 1:]

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
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def polar_angle(self) -> Tensor:
        return (self.scalar / self.norm()).acos()

    def matrix(self) -> Tensor:
        return quaternion_to_rotation_matrix(self.data, order=QuaternionCoeffOrder.WXYZ)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> 'Quaternion':
        return cls(rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.WXYZ))

    @classmethod
    def identity(cls, batch_size: int) -> 'Quaternion':
        data: Tensor = as_tensor([1.0, 0.0, 0.0, 0.0])
        data = data.repeat(batch_size, 1)
        return cls(data)

    @classmethod
    def from_coeffs(cls, w: float, x: float, y: float, z: float) -> 'Quaternion':
        return cls(as_tensor([[w, x, y, z]]))

    @classmethod
    def random(cls, batch_size: int) -> 'Quaternion':
        """Generate a random unit quaternion.

        Uniformly distributed across the rotation space As per: http://planning.cs.uiuc.edu/node198.html
        """
        r1, r2, r3 = rand(3, batch_size)
        q1 = (1.0 - r1).sqrt() * ((2 * pi * r2).sin())
        q2 = (1.0 - r1).sqrt() * ((2 * pi * r2).cos())
        q3 = r1.sqrt() * (2 * pi * r3).sin()
        q4 = r1.sqrt() * (2 * pi * r3).cos()
        return cls(stack((q1, q2, q3, q4), -1))

    def norm(self) -> Tensor:
        return self.data.norm(p=2, dim=-1)

    def normalize(self) -> 'Quaternion':
        return Quaternion(normalize_quaternion(self.data))

    def conj(self) -> 'Quaternion':
        return Quaternion(concatenate((self.real, -self.vec), -1))

    def inv(self) -> 'Quaternion':
        return self.conj() / self.squared_norm()

    def squared_norm(self) -> Tensor:
        return self._batched_squared_norm(self.vec) + self.real**2

    def _batched_squared_norm(self, x, y=None):
        if y is None:
            y = x
        return (x[..., None, :] @ y[..., :, None])[..., 0]

    # TODO: implement me
    def slerp(self):
        raise NotImplementedError

from typing import List, Optional

from kornia.core import Module, Parameter, Tensor, as_tensor, normalize, rand, stack
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.linalg import batched_squared_norm as _squared_norm
from kornia.testing import KORNIA_CHECK_SHAPE

__all__ = ["Vec3", "UnitVector"]


# TODO: model _Scalar
class _ScalarType:
    pass


# NOTE: not very clear if this is a module or custom Tensor
# tensor subclass doesn't like to override methods
class _VectorType:
    def __init__(self, data: Tensor) -> None:
        super().__init__()
        # TODO: we should support (N or BxN)
        # KORNIA_CHECK_SHAPE(data, ["B", "N"])
        self._data = data

    def __new(self, data):
        return type(self)(data)

    def __getattr__(self, name: str):
        """Direct access to the backend methods."""
        if name in ["dot"]:
            return
        return getattr(self.data, name)

    def __neg__(self):
        return self.__new(-self.data)

    def __add__(self, right):
        return self.__new(self.data + right.data)

    def __sub__(self, right):
        return self.__new(self.data - right.data)

    def __mul__(self, right):
        return self.__new(self.data * right.data)

    def __div__(self, right):
        return self.__new(self.data / right.data)

    def __truediv__(self, right):
        return self.__div__(right)

    @property
    def data(self) -> Tensor:
        return self._data

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self) -> str:
        return f"{self.data}"

    @classmethod
    def random(cls, shape: List[int], device=None, dtype=None) -> "_VectorType":
        return cls(rand(shape, device=device, dtype=dtype))

    def normalized(self) -> "_VectorType":
        vec_norm = normalize(self.data, p=2, dim=-1)
        return _VectorType(vec_norm)

    def squared_norm(self) -> "_VectorType":
        return _VectorType(_squared_norm(self.data))

    def dot(self, right: "_VectorType") -> "_VectorType":
        return _VectorType(batched_dot_product(self.data, right.data))

    def cross(self, right):
        return type(self)(self.data.cross(right.data))


class Vec3(_VectorType):
    def __init__(self, vector: Tensor) -> None:
        super().__init__(vector)

    @property
    def x(self) -> Tensor:
        return self.data[..., 0]

    @property
    def y(self) -> Tensor:
        return self.data[..., 1]

    @property
    def z(self) -> Tensor:
        return self.data[..., 2]

    @classmethod
    def random(cls, shape: Optional[List[int]] = None, device=None, dtype=None) -> "Vec3":
        if shape is None:
            shape = []
        return cls(rand(shape + [3], device=device, dtype=dtype))

    @classmethod
    def from_coords(cls, x: float, y: float, z: float) -> "Vec3":
        return Vec3(as_tensor([x, y, z]))

    @classmethod
    def from_coords_tensor(cls, x, y, z) -> "Vec3":
        return Vec3(stack((x, y, z), -1))


# TODO: adapt to _VevtorType
class UnitVector(Module):
    def __init__(self, vector: Tensor) -> None:
        super().__init__()
        KORNIA_CHECK_SHAPE(vector, ["B", "N"])
        self._vector = Parameter(vector)

    @property
    def vector(self) -> Tensor:
        return self._vector

    @classmethod
    def from_unit_vector(cls, v: Tensor) -> "UnitVector":
        # TODO: add checks https://github.com/strasdat/Sophus/blob/23.04-beta/cpp/sophus/geometry/ray.h#L59
        return UnitVector(_VectorType(v))

    @classmethod
    def from_vector(cls, v: Tensor) -> "UnitVector":
        """From a vector and normalize."""
        return UnitVector(_VectorType(v).normalized())

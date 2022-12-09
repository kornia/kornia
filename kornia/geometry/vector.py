from typing import List, Union

from kornia.core import Module, Parameter, Tensor, normalize, rand
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.linalg import batched_squared_norm as _squared_norm
from kornia.testing import KORNIA_CHECK_SHAPE

__all__ = ["UnitVector"]


# TODO: model _Scalar
class _ScalarType:
    pass


# NOTE: not very clear if this is a module or custom Tensor
# tensor subclass doesn't like to override methods
class _VectorType(Module):
    def __init__(self, vector: Tensor) -> None:
        super().__init__()
        # KORNIA_CHECK_SHAPE(vector, ["B", "N"])  # FIXME: resolve shape bugs. @edgarriba
        self._vector = Parameter(vector)

    @property
    def data(self) -> Tensor:
        return self._vector

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
        vec_norm = normalize(self._vector, p=2, dim=-1)
        return _VectorType(vec_norm)

    def squared_norm(self) -> "_VectorType":
        return _VectorType(_squared_norm(self.data))

    def dot(self, right: "_VectorType") -> "_VectorType":
        if len(right.shape) == 1:
            return _VectorType(self.data.dot(right.data))
        return _VectorType(batched_dot_product(self.data, right.data))


class UnitVector(Module):
    def __init__(self, vector: Union[Tensor, _VectorType]) -> None:
        super().__init__()
        KORNIA_CHECK_SHAPE(vector, ["B", "N"])
        if isinstance(vector, Tensor):
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

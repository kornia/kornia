from typing import List

from kornia.core import Module, Parameter, Tensor, normalize, rand
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
        KORNIA_CHECK_SHAPE(vector, ["B", "N"])
        self._vector = Parameter(vector)

    @property
    def data(self) -> Tensor:
        return self._vector

    @property
    def device(self):
        return self._vector.device

    @property
    def dtype(self):
        return self._vector.dtype

    @property
    def shape(self):
        return self._vector.shape

    def __repr__(self) -> str:
        return f"vec: {tuple(self._vector.shape)} {self._vector}"

    @classmethod
    def random(cls, shape: List[int], device=None, dtype=None) -> "_VectorType":
        return cls(rand(shape, device=device, dtype=dtype))

    def normalized(self) -> "_VectorType":
        vec_norm = normalize(self._vector, p=2, dim=-1)
        return _VectorType(vec_norm)

    # TODO: implement me
    def squared_norm(self) -> "_VectorType":
        raise NotImplementedError("Not implemented")

    # TODO: implement batched_dot
    def dot(self) -> "_VectorType":
        raise NotImplementedError("Not implemented")


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

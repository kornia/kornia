from typing import Optional, Tuple, Union

from kornia.core import Tensor, as_tensor, normalize, rand, stack
from kornia.core.tensor_wrapper import TensorWrapper
from kornia.geometry.linalg import batched_dot_product, batched_squared_norm
from kornia.testing import KORNIA_CHECK

__all__ = ["Vector3", "Scalar"]


# TODO: implement more functionality to validate
class Scalar(TensorWrapper):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data)


class Vector3(TensorWrapper):
    def __init__(self, vector: Tensor) -> None:
        super().__init__(vector)
        KORNIA_CHECK(vector.shape[-1] == 3)

    @property
    def x(self) -> Tensor:
        return self.data[..., 0]

    @property
    def y(self) -> Tensor:
        return self.data[..., 1]

    @property
    def z(self) -> Tensor:
        return self.data[..., 2]

    def normalized(self) -> "Vector3":
        return Vector3(normalize(self.data, p=2, dim=-1))

    def dot(self, right: "Vector3") -> Scalar:
        return Scalar(batched_dot_product(self.data, right.data))

    def squared_norm(self) -> Scalar:
        return Scalar(batched_squared_norm(self.data))

    @classmethod
    def random(cls, shape: Optional[Tuple[int, ...]] = None, device=None, dtype=None) -> "Vector3":
        if shape is None:
            shape = ()
        return cls(rand(shape + (3,), device=device, dtype=dtype))

    @classmethod
    def from_coords(cls, x: Union[float, Tensor], y: Union[float, Tensor], z: Union[float, Tensor]) -> "Vector3":
        if not (isinstance(x, Tensor) and isinstance(y, Tensor) and isinstance(z, Tensor)):
            return Vec3(as_tensor((x, y, z)))
        return Vec3(stack((x, y, z), -1))


Vec3 = Vector3

# TODO: adapt to TensorWrapper

# class UnitVector(Module):
#     def __init__(self, vector: Tensor) -> None:
#         super().__init__()
#         KORNIA_CHECK_SHAPE(vector, ["B", "N"])
#         self._vector = Parameter(vector)
#
#     @property
#     def vector(self) -> Tensor:
#         return self._vector
#
#     @classmethod
#     def from_unit_vector(cls, v: Tensor) -> "UnitVector":
#         # TODO: add checks https://github.com/strasdat/Sophus/blob/23.04-beta/cpp/sophus/geometry/ray.h#L59
#         return UnitVector(_VectorType(v))
#
#     @classmethod
#     def from_vector(cls, v: Tensor) -> "UnitVector":
#         """From a vector and normalize."""
#         return UnitVector(_VectorType(v).normalized())
#

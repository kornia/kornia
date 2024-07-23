from typing import Optional, Tuple, Union, cast

from kornia.core import Device, Dtype, Tensor, as_tensor, normalize, rand, stack
from kornia.core.check import KORNIA_CHECK
from kornia.core.tensor_wrapper import TensorWrapper, wrap  # type: ignore[attr-defined]
from kornia.geometry.linalg import batched_dot_product, batched_squared_norm

__all__ = ["Vector3", "Vector2", "Scalar"]


# TODO: implement more functionality to validate
class Scalar(TensorWrapper):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data)


class Vector3(TensorWrapper):
    def __init__(self, vector: Tensor) -> None:
        super().__init__(vector)
        KORNIA_CHECK(vector.shape[-1] == 3)

    def __repr__(self) -> str:
        return f"x: {self.x}\ny: {self.y}\nz: {self.z}"

    def __getitem__(self, idx: Union[slice, int, Tensor]) -> "Vector3":
        return Vector3(self.data[idx, ...])

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
    def random(
        cls, shape: Optional[Tuple[int, ...]] = None, device: Optional[Device] = None, dtype: Dtype = None
    ) -> "Vector3":
        if shape is None:
            shape = ()
        return cls(rand((*shape, 3), device=device, dtype=dtype))

    # TODO: polish overload
    # @overload
    # @classmethod
    # def from_coords(
    #     cls, x: Tensor, y: Tensor, z: Tensor, device=None, dtype=None
    # ) -> "Vector3":
    #     KORNIA_CHECK(isinstance(x, Tensor))
    #     KORNIA_CHECK(type(x) == type(y) == type(z))
    #     return wrap(as_tensor((x, y, z), device=device, dtype=dtype), Vector3)

    # TODO: polish overload
    # @overload
    # @classmethod
    # def from_coords(
    #     cls, x: float, y: float, z: float, device=None, dtype=None
    # ) -> "Vector3":
    #     KORNIA_CHECK(isinstance(x, float))
    #     KORNIA_CHECK(type(x) == type(y) == type(z))
    #     return wrap(as_tensor((x, y, z), device=device, dtype=dtype), Vector3)

    @classmethod
    def from_coords(
        cls,
        x: Union[float, Tensor],
        y: Union[float, Tensor],
        z: Union[float, Tensor],
        device: Optional[Device] = None,
        dtype: Dtype = None,
    ) -> "Vector3":
        KORNIA_CHECK(type(x) is type(y) is type(z))
        KORNIA_CHECK(isinstance(x, (Tensor, float)))
        if isinstance(x, float):
            return wrap(as_tensor((x, y, z), device=device, dtype=dtype), Vector3)
        # TODO: this is totally insane ...
        tensors: Tuple[Tensor, ...] = (x, cast(Tensor, y), cast(Tensor, z))
        return wrap(stack(tensors, -1), Vector3)


class Vector2(TensorWrapper):
    def __init__(self, vector: Tensor) -> None:
        super().__init__(vector)
        KORNIA_CHECK(vector.shape[-1] == 2)

    def __repr__(self) -> str:
        return f"x: {self.x}\ny: {self.y}"

    def __getitem__(self, idx: Union[slice, int, Tensor]) -> "Vector2":
        return Vector2(self.data[idx, ...])

    @property
    def x(self) -> Tensor:
        return self.data[..., 0]

    @property
    def y(self) -> Tensor:
        return self.data[..., 1]

    def normalized(self) -> "Vector2":
        return Vector2(normalize(self.data, p=2, dim=-1))

    def dot(self, right: "Vector2") -> Scalar:
        return Scalar(batched_dot_product(self.data, right.data))

    def squared_norm(self) -> Scalar:
        return Scalar(batched_squared_norm(self.data))

    @classmethod
    def random(cls, shape: Optional[Tuple[int, ...]] = None, device: Device = None, dtype: Dtype = None) -> "Vector2":
        if shape is None:
            shape = ()
        return cls(rand((*shape, 2), device=device, dtype=dtype))

    @classmethod
    def from_coords(
        cls, x: Union[float, Tensor], y: Union[float, Tensor], device: Device = None, dtype: Dtype = None
    ) -> "Vector2":
        KORNIA_CHECK(type(x) is type(y))
        KORNIA_CHECK(isinstance(x, (Tensor, float)))
        if isinstance(x, float):
            return wrap(as_tensor((x, y), device=device, dtype=dtype), Vector2)
        # TODO: this is totally insane ...
        tensors: Tuple[Tensor, ...] = (x, cast(Tensor, y))
        return wrap(stack(tensors, -1), Vector2)


Vec3 = Vector3
Vec2 = Vector2

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

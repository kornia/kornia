# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK
from kornia.core.tensor_wrapper import TensorWrapper, wrap  # type: ignore[attr-defined]
from kornia.geometry.linalg import batched_dot_product, batched_squared_norm

__all__ = ["Scalar", "Vector2", "Vector3"]


# TODO: implement more functionality to validate
class Scalar(TensorWrapper):
    """Wrap a tensor representing a scalar value."""
    def __init__(self, data: torch.Tensor) -> None:
        super().__init__(data)


class Vector3(TensorWrapper):
    """Wrap a tensor representing a 3D vector."""
    def __init__(self, vector: torch.Tensor) -> None:
        super().__init__(vector)
        KORNIA_CHECK(vector.shape[-1] == 3)

    def __repr__(self) -> str:
        return f"x: {self.x}\ny: {self.y}\nz: {self.z}"

    def __getitem__(self, idx: Union[slice, int, torch.Tensor]) -> "Vector3":
        return Vector3(self.data[idx, ...])

    @property
    def x(self) -> torch.Tensor:
        return self.data[..., 0]

    @property
    def y(self) -> torch.Tensor:
        return self.data[..., 1]

    @property
    def z(self) -> torch.Tensor:
        return self.data[..., 2]

    def normalized(self) -> "Vector3":
        return Vector3(F.normalize(self.data, p=2, dim=-1))

    def dot(self, right: "Vector3") -> Scalar:
        return Scalar(batched_dot_product(self.data, right.data))

    def squared_norm(self) -> Scalar:
        return Scalar(batched_squared_norm(self.data))

    @classmethod
    def random(
        cls,
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Vector3":
        if shape is None:
            shape = ()
        return cls(torch.rand((*shape, 3), device=device, dtype=dtype))

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
        x: Union[float, torch.Tensor],
        y: Union[float, torch.Tensor],
        z: Union[float, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Vector3":
        KORNIA_CHECK(type(x) is type(y) is type(z))
        KORNIA_CHECK(isinstance(x, (torch.Tensor, float)))
        if isinstance(x, float):
            return wrap(torch.as_tensor((x, y, z), device=device, dtype=dtype), Vector3)
        # TODO: this is totally insane ...
        tensors: Tuple[torch.Tensor, ...] = (x, cast(torch.Tensor, y), cast(torch.Tensor, z))
        return wrap(torch.stack(tensors, -1), Vector3)


class Vector2(TensorWrapper):
    """Wrap a tensor representing a 2D vector."""
    def __init__(self, vector: torch.Tensor) -> None:
        super().__init__(vector)
        KORNIA_CHECK(vector.shape[-1] == 2)

    def __repr__(self) -> str:
        return f"x: {self.x}\ny: {self.y}"

    def __getitem__(self, idx: Union[slice, int, torch.Tensor]) -> "Vector2":
        return Vector2(self.data[idx, ...])

    @property
    def x(self) -> torch.Tensor:
        return self.data[..., 0]

    @property
    def y(self) -> torch.Tensor:
        return self.data[..., 1]

    def normalized(self) -> "Vector2":
        return Vector2(F.normalize(self.data, p=2, dim=-1))

    def dot(self, right: "Vector2") -> Scalar:
        return Scalar(batched_dot_product(self.data, right.data))

    def squared_norm(self) -> Scalar:
        return Scalar(batched_squared_norm(self.data))

    @classmethod
    def random(
        cls,
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Vector2":
        if shape is None:
            shape = ()
        return cls(torch.rand((*shape, 2), device=device, dtype=dtype))

    @classmethod
    def from_coords(
        cls,
        x: Union[float, torch.Tensor],
        y: Union[float, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Vector2":
        KORNIA_CHECK(type(x) is type(y))
        KORNIA_CHECK(isinstance(x, (torch.Tensor, float)))
        if isinstance(x, float):
            return wrap(torch.as_tensor((x, y), device=device, dtype=dtype), Vector2)
        # TODO: this is totally insane ...
        tensors: Tuple[torch.Tensor, ...] = (x, cast(torch.Tensor, y))
        return wrap(torch.stack(tensors, -1), Vector2)


Vec3 = Vector3
Vec2 = Vector2

# TODO: adapt to TensorWrapper

# class UnitVector(Module):
#     def __init__(self, vector: torch.Tensor) -> None:
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

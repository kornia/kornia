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
from kornia.core.tensor_wrapper import TensorWrapper, _wrap  # type: ignore[attr-defined]
from kornia.geometry.linalg import batched_dot_product, batched_squared_norm

__all__ = ["Scalar", "Vector2", "Vector3"]


# TODO: implement more functionality to validate
class Scalar(TensorWrapper):
    """Wrap a tensor of scalars of any shape, such as the per-vector result of :meth:`Vector3.dot`.

    The tensor is wrapped without a copy, and the call-path type defect of :class:`Vector3` applies to it
    (`#5022 <https://github.com/kornia/kornia/issues/5022>`_).
    """

    def __init__(self, data: torch.Tensor) -> None:
        super().__init__(data)


class Vector3(TensorWrapper):
    r"""Wrap a tensor of 3D vectors, shape :math:`(*, 3)`.

    Convention:
        - The tensor is wrapped without a copy; any leading shape and dtype are accepted. :attr:`x`, :attr:`y` and
          :attr:`z` are plain tensors of the leading shape :math:`(*)`, and :meth:`dot` and :meth:`squared_norm`
          return a :class:`Scalar` of that shape (for :meth:`dot`, the two operands' broadcast leading shape).
        - :meth:`random` draws vectors uniformly in the unit cube from torch's global generator, so every vector
          lies in the first octant: it is not a random direction.
        - Known defect: the returned type depends on the call path (``copy.deepcopy(v)`` and ``v.clone()`` are
          plain tensors, while a torch function rewraps its result as a ``Vector3``, so ``torch.linalg.norm(v,
          dim=-1)`` raises unless its result happens to end in 3), and a tuple index such as ``v[..., 0]`` raises
          (`#5022 <https://github.com/kornia/kornia/issues/5022>`_).
    """

    def __init__(self, vector: torch.Tensor) -> None:
        super().__init__(vector)
        KORNIA_CHECK(vector.shape[-1] == 3)

    def __repr__(self) -> str:
        return f"x: {self.x}\ny: {self.y}\nz: {self.z}"

    def __getitem__(self, idx: Union[slice, int, torch.Tensor]) -> "Vector3":
        return Vector3(self.data[idx, ...])

    @property
    def x(self) -> torch.Tensor:
        """Return the x-coordinate stored in the last tensor dimension."""
        return self.data[..., 0]

    @property
    def y(self) -> torch.Tensor:
        """Return the y-coordinate stored in the last tensor dimension."""
        return self.data[..., 1]

    @property
    def z(self) -> torch.Tensor:
        """Return the z-coordinate stored in the last tensor dimension."""
        return self.data[..., 2]

    def normalized(self) -> "Vector3":
        """Return a copy with each vector divided by its Euclidean norm.

        Returns:
            New :class:`Vector3` of the same shape. The norm is floored at ``1e-12``, so a shorter vector is scaled
            by ``1e12`` instead of normalized (`#3952 <https://github.com/kornia/kornia/issues/3952>`_) and a zero
            vector stays zero, except in ``float16``, where the floor underflows and a zero vector gives NaN
            (`#5062 <https://github.com/kornia/kornia/issues/5062>`_).
        """
        return Vector3(F.normalize(self.data, p=2, dim=-1))

    def dot(self, right: "Vector3") -> Scalar:
        """Compute dot products with another 3D vector wrapper.

        Args:
            right: Right-hand :class:`Vector3` operand with compatible leading
                dimensions.

        Returns:
            :class:`Scalar` containing :math:`x_1 x_2 + y_1 y_2 + z_1 z_2` for
            each leading element.
        """
        return Scalar(batched_dot_product(self.data, right.data))

    def squared_norm(self) -> Scalar:
        """Compute squared Euclidean lengths of the wrapped vectors.

        Returns:
            :class:`Scalar` containing :math:`x^2 + y^2 + z^2` for each
            vector.
        """
        return Scalar(batched_squared_norm(self.data))

    @classmethod
    def random(
        cls,
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Vector3":
        """Create random 3D vectors with optional leading dimensions.

        See the Convention block on :class:`Vector3`.

        Args:
            shape: Optional leading dimensions before the final coordinate
                dimension. For example, ``shape=(B, N)`` creates
                :math:`(B, N, 3)` vectors.
            device: Target device for the generated tensor.
            dtype: Target dtype for the generated tensor.

        Returns:
            :class:`Vector3` wrapping a random tensor with shape
            ``(*shape, 3)``.
        """
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
        """Construct a 3D vector wrapper from x, y, and z coordinates.

        All coordinate inputs must share the same Python type (all floats or
        all tensors). For tensor inputs, coordinates are stacked along the last
        dimension to produce ``(..., 3)`` output.

        Args:
            x: X-coordinate value or tensor.
            y: Y-coordinate value or tensor.
            z: Z-coordinate value or tensor.
            device: Device used when scalar inputs are converted to tensor.
            dtype: Dtype used when scalar inputs are converted to tensor.

        Returns:
            :class:`Vector3` containing the assembled coordinates.
        """
        KORNIA_CHECK(type(x) is type(y) is type(z))
        KORNIA_CHECK(isinstance(x, torch.Tensor | float))
        if isinstance(x, float):
            return _wrap(torch.as_tensor((x, y, z), device=device, dtype=dtype), Vector3)
        # TODO: this is totally insane ...
        tensors: Tuple[torch.Tensor, ...] = (x, cast(torch.Tensor, y), cast(torch.Tensor, z))
        return _wrap(torch.stack(tensors, -1), Vector3)


class Vector2(TensorWrapper):
    r"""Wrap a tensor of 2D vectors, shape :math:`(*, 2)`.

    See the Convention block on :class:`Vector3`, which applies to ``(x, y)`` vectors; :meth:`random` fills the
    unit square.
    """

    def __init__(self, vector: torch.Tensor) -> None:
        super().__init__(vector)
        KORNIA_CHECK(vector.shape[-1] == 2)

    def __repr__(self) -> str:
        return f"x: {self.x}\ny: {self.y}"

    def __getitem__(self, idx: Union[slice, int, torch.Tensor]) -> "Vector2":
        return Vector2(self.data[idx, ...])

    @property
    def x(self) -> torch.Tensor:
        """Return the x-coordinate stored in the last tensor dimension."""
        return self.data[..., 0]

    @property
    def y(self) -> torch.Tensor:
        """Return the y-coordinate stored in the last tensor dimension."""
        return self.data[..., 1]

    def normalized(self) -> "Vector2":
        """Return a copy with each vector divided by its Euclidean norm.

        Returns:
            New :class:`Vector2` of the same shape, with the norm floored as in :meth:`Vector3.normalized`.
        """
        return Vector2(F.normalize(self.data, p=2, dim=-1))

    def dot(self, right: "Vector2") -> Scalar:
        """Compute dot products with another 2D vector wrapper.

        Args:
            right: Right-hand :class:`Vector2` operand with compatible leading
                dimensions.

        Returns:
            :class:`Scalar` containing :math:`x_1 x_2 + y_1 y_2` for each
            leading element.
        """
        return Scalar(batched_dot_product(self.data, right.data))

    def squared_norm(self) -> Scalar:
        """Compute squared Euclidean lengths of the wrapped vectors.

        Returns:
            :class:`Scalar` containing :math:`x^2 + y^2` for each vector.
        """
        return Scalar(batched_squared_norm(self.data))

    @classmethod
    def random(
        cls,
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Vector2":
        """Create random 2D vectors with optional leading dimensions.

        See the Convention block on :class:`Vector3`.

        Args:
            shape: Optional leading dimensions before the final coordinate
                dimension. For example, ``shape=(B, N)`` creates
                :math:`(B, N, 2)` vectors.
            device: Target device for the generated tensor.
            dtype: Target dtype for the generated tensor.

        Returns:
            :class:`Vector2` wrapping a random tensor with shape
            ``(*shape, 2)``.
        """
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
        """Construct a 2D vector wrapper from x and y coordinates.

        Both coordinates must share the same Python type (both floats or both
        tensors). For tensor inputs, coordinates are stacked along the last
        dimension to produce ``(..., 2)`` output.

        Args:
            x: X-coordinate value or tensor.
            y: Y-coordinate value or tensor.
            device: Device used when scalar inputs are converted to tensor.
            dtype: Dtype used when scalar inputs are converted to tensor.

        Returns:
            :class:`Vector2` containing the assembled coordinates.
        """
        KORNIA_CHECK(type(x) is type(y))
        KORNIA_CHECK(isinstance(x, torch.Tensor | float))
        if isinstance(x, float):
            return _wrap(torch.as_tensor((x, y), device=device, dtype=dtype), Vector2)
        # TODO: this is totally insane ...
        tensors: Tuple[torch.Tensor, ...] = (x, cast(torch.Tensor, y))
        return _wrap(torch.stack(tensors, -1), Vector2)


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

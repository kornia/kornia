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

# kornia.geometry.so2 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so2.py
from __future__ import annotations

from typing import Optional, Union, overload

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR
from kornia.core.exceptions import TypeCheckError
from kornia.geometry.liegroup._utils import (
    check_so2_matrix_shape,
    check_so2_t_shape,
    check_so2_theta_shape,
    check_so2_z_shape,
)
from kornia.geometry.vector import Vector2


class So2(nn.Module):
    r"""Base class to represent the So2 group.

    The SO(2) is the group of all rotations about the origin of two-dimensional Euclidean space
    :math:`R^2` under the operation of composition.
    See more: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group

    We internally represent the rotation by a torch.complex number.

    Example:
        >>> real = torch.tensor([1.0])
        >>> imag = torch.tensor([2.0])
        >>> So2(torch.complex(real, imag))
        Parameter containing:
        tensor([1.+2.j], requires_grad=True)

    """

    def __init__(self, z: torch.Tensor) -> None:
        """Construct the base class.

        Internally represented by torch.complex number `z`.

        Args:
            z: Complex number with the shape of :math:`(B, 1)` or :math:`(B)`.

        Example:
            >>> real = torch.tensor(1.0)
            >>> imag = torch.tensor(2.0)
            >>> So2(torch.complex(real, imag)).z
            Parameter containing:
            tensor(1.+2.j, requires_grad=True)

        """
        super().__init__()
        KORNIA_CHECK_IS_TENSOR(z)
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_so2_z_shape(z)
        self._z = z

    def __repr__(self) -> str:
        return f"{self.z}"

    def __getitem__(self, idx: int | slice) -> So2:
        return So2(self._z[idx])

    @overload
    def __mul__(self, right: So2) -> So2: ...

    @overload
    def __mul__(self, right: torch.Tensor | Vector2) -> torch.Tensor | Vector2: ...

    def __mul__(self, right: So2 | torch.Tensor | Vector2) -> So2 | torch.Tensor | Vector2:
        """Perform a left-multiplication either rotation concatenation or point-transform.

        Args:
            right: the other So2 transformation.

        Return:
            The resulting So2 transformation.

        """
        z = self.z
        if isinstance(right, So2):
            return So2(z * right.z)
        elif isinstance(right, (Vector2, torch.Tensor)):
            # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
            if isinstance(right, torch.Tensor):
                check_so2_t_shape(right)
            x = right.data[..., 0]
            y = right.data[..., 1]

            if not z.is_complex():
                # Handling case where z was converted to real by module.to(dtype)
                # This is a fallback to avoid crash, but So2 should ideally keep complex z
                real = z
                imag = torch.zeros_like(z)
            else:
                real = z.real
                imag = z.imag

            out = torch.stack((real * x - imag * y, imag * x + real * y), -1)
            if isinstance(right, torch.Tensor):
                return out
            else:
                return Vector2(out)
        else:
            raise TypeCheckError(f"Not So2 or torch.Tensor type. Got: {type(right)}")

    @property
    def z(self) -> torch.Tensor:
        """Return the underlying data with shape :math:`(B, 1)`."""
        return self._z

    @staticmethod
    def exp(theta: torch.Tensor) -> So2:
        """Convert elements of lie algebra to elements of lie group.

        Args:
            theta: angle in radians of shape :math:`(B, 1)` or :math:`(B)`.

        Example:
            >>> v = torch.tensor([3.1415/2])
            >>> s = So2.exp(v)
            >>> s
            tensor([4.6329e-05+1.j], grad_fn=<So2Backward>)

        """
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_so2_theta_shape(theta)
        return So2(torch.complex(torch.cos(theta), torch.sin(theta)))

    def log(self) -> torch.Tensor:
        """Convert elements of lie group to elements of lie algebra.

        Example:
            >>> real = torch.tensor([1.0])
            >>> imag = torch.tensor([3.0])
            >>> So2(torch.complex(real, imag)).log()
            tensor([1.2490], grad_fn=<Atan2Backward0>)

        """
        if not self.z.is_complex():
            return torch.zeros_like(self.z)
        return self.z.imag.atan2(self.z.real)

    @staticmethod
    def hat(theta: torch.Tensor) -> torch.Tensor:
        """Convert elements from vector space to lie algebra. Returns matrix of shape :math:`(B, 2, 2)`.

        Args:
            theta: angle in radians of shape :math:`(B)`.

        Example:
            >>> theta = torch.tensor(3.1415/2)
            >>> So2.hat(theta)
            tensor([[0.0000, -1.5708],
                    [1.5708,  0.0000]])

        """
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_so2_theta_shape(theta)
        z = torch.zeros_like(theta)
        row0 = torch.stack((z, -theta), -1)
        row1 = torch.stack((theta, z), -1)
        return torch.stack((row0, row1), -2)

    @staticmethod
    def vee(omega: torch.Tensor) -> torch.Tensor:
        """Convert elements from lie algebra to vector space. Returns vector of shape :math:`(B,)`.

        Args:
            omega: 2x2-matrix representing lie algebra.

        Example:
            >>> v = torch.ones(3)
            >>> omega = So2.hat(v)
            >>> So2.vee(omega)
            tensor([1., 1., 1.])

        """
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_so2_matrix_shape(omega)
        return omega[..., 1, 0]

    def matrix(self) -> torch.Tensor:
        """Convert the torch.complex number to a rotation matrix of shape :math:`(B, 2, 2)`.

        Example:
            >>> s = So2.identity()
            >>> m = s.matrix()
            >>> m
            tensor([[1., -0.],
                    [0., 1.]], grad_fn=<StackBackward0>)

        """
        if not self.z.is_complex():
            real = self.z
            imag = torch.zeros_like(self.z)
        else:
            real = self.z.real
            imag = self.z.imag
        row0 = torch.stack((real, -imag), -1)
        row1 = torch.stack((imag, real), -1)
        return torch.stack((row0, row1), -2)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> So2:
        """Create So2 from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B, 2, 2)`.

        Example:
            >>> m = torch.eye(2)
            >>> s = So2.from_matrix(m)
            >>> s.z
            tensor(1.+0.j, grad_fn=<ComplexBackward0>)

        """
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_so2_matrix_shape(matrix)
        # check_so2_matrix(matrix)
        z = torch.complex(matrix[..., 0, 0], matrix[..., 1, 0])
        return cls(z)

    @classmethod
    def identity(
        cls,
        batch_size: Optional[int] = None,
        device: Union[None, str, torch.device] = None,
        dtype: Union[None, torch.dtype] = None,
    ) -> So2:
        """Create a So2 group representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> s = So2.identity(batch_size=2)
            >>> s
            tensor([1.+0.j, 1.+0.j], grad_fn=<So2Backward>)

        """
        real_data = torch.tensor(1.0, device=device, dtype=dtype)
        imag_data = torch.tensor(0.0, device=device, dtype=dtype)
        if batch_size is not None:
            KORNIA_CHECK(batch_size >= 1, msg="batch_size must be positive")
            real_data = real_data.repeat(batch_size)
            imag_data = imag_data.repeat(batch_size)
        return cls(torch.complex(real_data, imag_data))

    def inverse(self) -> So2:
        """Return the inverse transformation.

        Example:
            >>> s = So2.identity()
            >>> s.inverse().z
            tensor(1.+0.j, grad_fn=<DivBackward0>)

        """
        return So2(1 / self.z)

    @classmethod
    def random(
        cls,
        batch_size: Optional[int] = None,
        device: Union[None, str, torch.device] = None,
        dtype: Union[None, torch.dtype] = None,
    ) -> So2:
        """Create a So2 group representing a random rotation.

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> s = So2.random()
            >>> s = So2.random(batch_size=3)

        """
        import math

        rand_shape = (batch_size,) if batch_size is not None else ()
        theta = torch.rand(rand_shape, device=device, dtype=dtype) * 2 * math.pi - math.pi
        return cls.exp(theta)

    def adjoint(self) -> torch.Tensor:
        """Return the adjoint matrix of shape :math:`(B, 1, 1)`.

        Example:
            >>> s = So2.identity()
            >>> s.adjoint()
            tensor([[1.]])

        """
        if self.z.dim() == 0:
            return torch.ones((1, 1), device=self.z.device, dtype=self.z.real.dtype)
        batch_size = self.z.shape[0]
        return torch.ones((batch_size, 1, 1), device=self.z.device, dtype=self.z.real.dtype)

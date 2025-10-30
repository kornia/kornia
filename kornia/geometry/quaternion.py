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

# kornia.geometry.quaternion module inspired by Eigen, Sophus-sympy, and PyQuaternion.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/quaternion.py
# https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
# https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Quaternion.h
from math import pi
from typing import Any, Optional, Tuple, Union

import torch

from kornia.core import Device, Dtype, Module, Parameter, Tensor, concatenate, rand, stack, tensor, where
from kornia.core.check import KORNIA_CHECK_TYPE
from kornia.geometry.conversions import (
    axis_angle_to_quaternion,
    euler_from_quaternion,
    normalize_quaternion,
    quaternion_from_euler,
    quaternion_to_axis_angle,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from kornia.geometry.linalg import batched_dot_product


class Quaternion(Module):
    r"""Base class to represent a Quaternion.

    A quaternion is a four dimensional vector representation of a rotation transformation in 3d.
    See more: https://en.wikipedia.org/wiki/Quaternion

    The general definition of a quaternion is given by:

    .. math::

        Q = a + b \cdot \mathbf{i} + c \cdot \mathbf{j} + d \cdot \mathbf{k}

    Thus, we represent a rotation quaternion as a contiguous tensor structure to
    perform rigid bodies transformations:

    .. math::

        Q = \begin{bmatrix} q_w & q_x & q_y & q_z \end{bmatrix}

    Example:
        >>> q = Quaternion.identity(batch_size=4)
        >>> q.data
        tensor([[1., 0., 0., 0.],
                [1., 0., 0., 0.],
                [1., 0., 0., 0.],
                [1., 0., 0., 0.]])
        >>> q.real
        tensor([1., 1., 1., 1.])
        >>> q.vec
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])

    """

    _data: Union[Tensor, Parameter]

    def __init__(self, data: Union[Tensor, Parameter]) -> None:
        """Construct a quaternion from tensor or parameter data.

        Args:
            data: tensor or parameter containing the quaternion data with the shape of :math:`(B, 4)`.

        Example:
            >>> # Create with tensor (no gradients tracked by default)
            >>> data = torch.tensor([1., 0., 0., 0.])
            >>> q1 = Quaternion(data)
            >>> # Create with parameter (gradients tracked)
            >>> param_data = torch.nn.Parameter(torch.tensor([1., 0., 0., 0.]))
            >>> q2 = Quaternion(param_data)

        """
        super().__init__()
        if not isinstance(data, (Tensor, Parameter)):
            raise TypeError(f"Expected Tensor or Parameter, got {type(data)}")
        # KORNIA_CHECK_SHAPE(data, ["B", "4"])  # FIXME: resolve shape bugs. @edgarriba
        self._data = data

    def to(self, *args: Any, **kwargs: Any) -> "Quaternion":
        """Move and/or cast the quaternion data.

        Args:
            *args: Arguments to pass to tensor.to()
            **kwargs: Keyword arguments to pass to tensor.to()

        Returns:
            A new Quaternion with converted data.
        """
        return Quaternion(self._data.to(*args, **kwargs))

    def _to_scalar_quaternion(self, value: Union[Tensor, float]) -> "Quaternion":
        """Convert a scalar, tensor, or numeric value to a scalar quaternion.

        A scalar quaternion has the form [real, 0, 0, 0] where real is the input value.

        Args:
            value: The scalar, tensor, or numeric value to convert.

        Returns:
            A Quaternion object representing the scalar quaternion.
        """
        if isinstance(value, (int, float)):
            value = torch.tensor(value, device=self.data.device, dtype=self.data.dtype)
        elif isinstance(value, torch.Tensor):
            value = value.to(device=self.data.device, dtype=self.data.dtype)

        # Broadcast value to match the shape of self.real
        try:
            target_shape = torch.broadcast_shapes(self.real.shape, value.shape)
        except RuntimeError as e:
            raise ValueError(f"Cannot broadcast shapes {self.real.shape} and {value.shape}") from e

        broadcasted = self.real.expand(target_shape) + value.expand(target_shape)
        # Create scalar quaternion: [value, 0, 0, 0]
        # Expand value to match the broadcasted shape, then add quaternion dimension
        if value.dim() == 0:  # scalar
            # Expand to match the broadcasted shape
            expanded_value = value.expand_as(broadcasted)
        else:
            # Use broadcasting to get the right shape
            expanded_value = torch.broadcast_to(value, broadcasted.shape)

        # Create zeros for the imaginary part
        zeros = torch.zeros_like(expanded_value).unsqueeze(-1).expand(*expanded_value.shape, 3)

        # Stack real and imaginary parts: [real, 0, 0, 0]
        scalar_quat_data = torch.cat([expanded_value.unsqueeze(-1), zeros], dim=-1)

        return Quaternion(scalar_quat_data)

    def __repr__(self) -> str:
        return f"{self.data}"

    def __getitem__(self, idx: Union[int, slice]) -> "Quaternion":
        return Quaternion(self.data[idx])

    def __neg__(self) -> "Quaternion":
        """Inverts the sign of the quaternion data.

        Example:
            >>> q = Quaternion.identity()
            >>> -q.data
            tensor([-1., -0., -0., -0.])

        """
        return Quaternion(-self.data)

    def __add__(self, right: Union["Quaternion", Tensor, float]) -> "Quaternion":
        """Add a given quaternion, scalar, or tensor.

        Args:
            right: the quaternion, scalar, or tensor to add.

        Example:
            >>> q1 = Quaternion.identity()
            >>> q2 = Quaternion(tensor([2., 0., 1., 1.]))
            >>> q3 = q1 + q2
            >>> q3.data
            tensor([3., 0., 1., 1.])

        """
        if isinstance(right, Quaternion):
            return Quaternion(self.data + right.data)
        else:
            right_quat = self._to_scalar_quaternion(right)
            return Quaternion(self.data + right_quat.data)

    def __sub__(self, right: Union["Quaternion", Tensor, float]) -> "Quaternion":
        """Subtract a given quaternion, scalar, or tensor.

        Args:
            right: the quaternion, scalar, or tensor to subtract.

        Example:
            >>> q1 = Quaternion(tensor([2., 0., 1., 1.]))
            >>> q2 = Quaternion.identity()
            >>> q3 = q1 - q2
            >>> q3.data
            tensor([1., 0., 1., 1.])

        """
        if isinstance(right, Quaternion):
            return Quaternion(self.data - right.data)
        else:
            right_quat = self._to_scalar_quaternion(right)
            # For scalar operations, ensure we return a tensor to preserve gradients
            result_data = self.data - right_quat.data
            if isinstance(result_data, Parameter):
                result_data = result_data.data  # Convert to tensor to preserve gradients
            return Quaternion(result_data)

    def __mul__(self, right: Union["Quaternion", Tensor, float]) -> "Quaternion":
        # If right is a Quaternion, do quaternion multiplication
        if isinstance(right, Quaternion):
            new_real = self.real * right.real - batched_dot_product(self.vec, right.vec)
            new_vec = (
                self.real[..., None] * right.vec
                + right.real[..., None] * self.vec
                + torch.linalg.cross(self.vec, right.vec, dim=-1)
            )
            return Quaternion(concatenate((new_real[..., None], new_vec), -1))

        # If right is a scalar/tensor, convert to scalar quaternion and multiply
        else:
            right_quat = self._to_scalar_quaternion(right)
            new_real = self.real * right_quat.real - batched_dot_product(self.vec, right_quat.vec)
            new_vec = (
                self.real[..., None] * right_quat.vec
                + right_quat.real[..., None] * self.vec
                + torch.linalg.cross(self.vec, right_quat.vec, dim=-1)
            )
            return Quaternion(concatenate((new_real[..., None], new_vec), -1))

    def __rmul__(self, left: Union[Tensor, float]) -> "Quaternion":
        """Right multiplication (left * self) where left is a scalar or tensor."""
        left_quat = self._to_scalar_quaternion(left)
        new_real = left_quat.real * self.real - batched_dot_product(left_quat.vec, self.vec)
        new_vec = (
            left_quat.real[..., None] * self.vec
            + self.real[..., None] * left_quat.vec
            + torch.linalg.cross(left_quat.vec, self.vec, dim=-1)
        )
        return Quaternion(concatenate((new_real[..., None], new_vec), -1))

    def __div__(self, right: Union[Tensor, "Quaternion", float]) -> "Quaternion":
        if isinstance(right, Quaternion):
            return self * right.inv()
        else:
            # For scalars/tensors, just divide the quaternion data directly
            if isinstance(right, (int, float)):
                right_tensor = torch.tensor(right, device=self.data.device, dtype=self.data.dtype)
            else:
                right_tensor = right.to(device=self.data.device, dtype=self.data.dtype)

            # For division by scalar, expand to [right, right, right, right] for element-wise division
            if right_tensor.dim() == 0:  # scalar
                divisor = right_tensor.expand_as(self.data[..., 0]).unsqueeze(-1).expand_as(self.data)
            else:
                # Broadcast the tensor to match the quaternion dimensions
                divisor = right_tensor.unsqueeze(-1).expand_as(self.data)

            # For scalar operations, ensure we return a tensor to preserve gradients
            result_data = self.data / divisor
            if isinstance(result_data, Parameter):
                result_data = result_data.data  # Convert to tensor to preserve gradients
            return Quaternion(result_data)

    def __truediv__(self, right: Union[Tensor, "Quaternion", float]) -> "Quaternion":
        return self.__div__(right)

    def __radd__(self, left: Union[Tensor, float]) -> "Quaternion":
        """Right addition (left + self) where left is a scalar or tensor."""
        left_quat = self._to_scalar_quaternion(left)
        return left_quat + self

    def __rsub__(self, left: Union[Tensor, float]) -> "Quaternion":
        """Right subtraction (left - self) where left is a scalar or tensor."""
        left_quat = self._to_scalar_quaternion(left)
        return left_quat - self

    def __rtruediv__(self, left: Union[Tensor, float]) -> "Quaternion":
        """Right division (left / self) where left is a scalar or tensor."""
        left_quat = self._to_scalar_quaternion(left)
        return left_quat / self

    def __rdiv__(self, left: Union[Tensor, float]) -> "Quaternion":
        """Right division (left / self) where left is a scalar or tensor."""
        return self.__rtruediv__(left)

    def __pow__(self, t: float) -> "Quaternion":
        """Return the power of a quaternion raised to exponent t.

        Args:
            t: raised exponent.

        Example:
            >>> q = Quaternion(tensor([1., .5, 0., 0.]))
            >>> q_pow = q**2

        """
        theta = self.polar_angle[..., None]
        vec_norm = self.vec.norm(dim=-1, keepdim=True)
        n = where(vec_norm != 0, self.vec / vec_norm, self.vec * 0)
        w = (t * theta).cos()
        xyz = (t * theta).sin() * n
        return Quaternion(concatenate((w, xyz), -1))

    @property
    def data(self) -> Tensor:
        """Return the underlying data with shape :math:`(B, 4)`."""
        return self._data

    @property
    def coeffs(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return a tuple with the underlying coefficients in WXYZ order."""
        return self.w, self.x, self.y, self.z

    @property
    def real(self) -> Tensor:
        """Return the real part with shape :math:`(B,)`.

        Alias for
        :func: `~kornia.geometry.quaternion.Quaternion.w`
        """
        return self.w

    @property
    def vec(self) -> Tensor:
        """Return the vector with the imaginary part with shape :math:`(B, 3)`."""
        return self.data[..., 1:]

    @property
    def q(self) -> Tensor:
        """Return the underlying data with shape :math:`(B, 4)`.

        Alias for :func:`~kornia.geometry.quaternion.Quaternion.data`
        """
        return self.data

    @property
    def scalar(self) -> Tensor:
        """Return a scalar with the real with shape :math:`(B,)`.

        Alias for
        :func: `~kornia.geometry.quaternion.Quaternion.w`
        """
        return self.real

    @property
    def w(self) -> Tensor:
        """Return the :math:`q_w` with shape :math:`(B,)`."""
        return self.data[..., 0]

    @property
    def x(self) -> Tensor:
        """Return the :math:`q_x` with shape :math:`(B,)`."""
        return self.data[..., 1]

    @property
    def y(self) -> Tensor:
        """Return the :math:`q_y` with shape :math:`(B,)`."""
        return self.data[..., 2]

    @property
    def z(self) -> Tensor:
        """Return the :math:`q_z` with shape :math:`(B,)`."""
        return self.data[..., 3]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying data with shape :math:`(B, 4)`."""
        return tuple(self.data.shape)

    @property
    def polar_angle(self) -> Tensor:
        """Return the polar angle with shape :math:`(B,1)`.

        Example:
            >>> q = Quaternion.identity()
            >>> q.polar_angle
            tensor(0.)

        """
        return (self.scalar / self.norm()).acos()

    def matrix(self) -> Tensor:
        """Convert the quaternion to a rotation matrix of shape :math:`(B, 3, 3)`.

        Example:
            >>> q = Quaternion.identity()
            >>> m = q.matrix()
            >>> m
            tensor([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]])

        """
        return quaternion_to_rotation_matrix(self.data)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> "Quaternion":
        """Create a quaternion from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B, 3, 3)`.

        Example:
            >>> m = torch.eye(3)[None]
            >>> q = Quaternion.from_matrix(m)
            >>> q.data
            tensor([[1., 0., 0., 0.]])

        """
        return cls(rotation_matrix_to_quaternion(matrix))

    @classmethod
    def from_euler(cls, roll: Tensor, pitch: Tensor, yaw: Tensor) -> "Quaternion":
        """Create a quaternion from Euler angles.

        Args:
            roll: the roll euler angle.
            pitch: the pitch euler angle.
            yaw: the yaw euler angle.

        Example:
            >>> roll, pitch, yaw = tensor(0), tensor(1), tensor(0)
            >>> q = Quaternion.from_euler(roll, pitch, yaw)
            >>> q.data
            tensor([0.8776, 0.0000, 0.4794, 0.0000])

        """
        w, x, y, z = quaternion_from_euler(roll=roll, pitch=pitch, yaw=yaw)
        q = stack((w, x, y, z), -1)
        return cls(q)

    def to_euler(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert the quaternion to a triple of Euler angles (roll, pitch, yaw).

        Example:
            >>> q = Quaternion(tensor([2., 0., 1., 1.]))
            >>> roll, pitch, yaw = q.to_euler()
            >>> roll
            tensor(2.0344)
            >>> pitch
            tensor(1.5708)
            >>> yaw
            tensor(2.2143)

        """
        return euler_from_quaternion(self.w, self.x, self.y, self.z)

    @classmethod
    def from_axis_angle(cls, axis_angle: Tensor) -> "Quaternion":
        """Create a quaternion from axis-angle representation.

        Args:
            axis_angle: rotation vector of shape :math:`(B, 3)`.

        Example:
            >>> axis_angle = torch.tensor([[1., 0., 0.]])
            >>> q = Quaternion.from_axis_angle(axis_angle)
            >>> q.data
            tensor([[0.8776, 0.4794, 0.0000, 0.0000]])

        """
        return cls(axis_angle_to_quaternion(axis_angle))

    def to_axis_angle(self) -> Tensor:
        """Convert the quaternion to an axis-angle representation.

        Example:
            >>> q = Quaternion.identity()
            >>> axis_angle = q.to_axis_angle()
            >>> axis_angle
            tensor([0., 0., 0.])

        """
        return quaternion_to_axis_angle(self.data)

    @classmethod
    def identity(
        cls, batch_size: Optional[int] = None, device: Optional[Device] = None, dtype: Dtype = None
    ) -> "Quaternion":
        """Create a quaternion representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> q = Quaternion.identity()
            >>> q.data
            tensor([1., 0., 0., 0.])

        """
        data = tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        if batch_size is not None:
            data = data.repeat(batch_size, 1)
        return cls(data)

    @classmethod
    def from_coeffs(cls, w: float, x: float, y: float, z: float) -> "Quaternion":
        """Create a quaternion from the data coefficients.

        Args:
            w: a float representing the :math:`q_w` component.
            x: a float representing the :math:`q_x` component.
            y: a float representing the :math:`q_y` component.
            z: a float representing the :math:`q_z` component.

        Example:
            >>> q = Quaternion.from_coeffs(1., 0., 0., 0.)
            >>> q.data
            tensor([1., 0., 0., 0.])

        """
        return cls(tensor([w, x, y, z]))

    # TODO: update signature
    # def random(cls, shape: Optional[List] = None, device = None, dtype = None) -> 'Quaternion':
    @classmethod
    def random(
        cls, batch_size: Optional[int] = None, device: Optional[Device] = None, dtype: Dtype = None
    ) -> "Quaternion":
        """Create a random unit quaternion of shape :math:`(B, 4)`.

        Uniformly distributed across the rotation space as per: http://planning.cs.uiuc.edu/node198.html

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> q = Quaternion.random()
            >>> q = Quaternion.random(batch_size=2)

        """
        rand_shape = (batch_size,) if batch_size is not None else ()

        r1, r2, r3 = rand((3, *rand_shape), device=device, dtype=dtype)
        q1 = (1.0 - r1).sqrt() * ((2 * pi * r2).sin())
        q2 = (1.0 - r1).sqrt() * ((2 * pi * r2).cos())
        q3 = r1.sqrt() * (2 * pi * r3).sin()
        q4 = r1.sqrt() * (2 * pi * r3).cos()
        return cls(stack((q1, q2, q3, q4), -1))

    def slerp(self, q1: "Quaternion", t: float) -> "Quaternion":
        """Return a unit quaternion spherically interpolated between quaternions self.q and q1.

        See more: https://en.wikipedia.org/wiki/Slerp

        Args:
            q1: second quaternion to be interpolated between.
            t: interpolation ratio, range [0-1]

        Example:
            >>> q0 = Quaternion.identity()
            >>> q1 = Quaternion(torch.tensor([1., .5, 0., 0.]))
            >>> q2 = q0.slerp(q1, .3)

        """
        KORNIA_CHECK_TYPE(q1, Quaternion)
        q0 = self.normalize()
        q1 = q1.normalize()
        return q0 * (q0.inv() * q1) ** t

    def norm(self, keepdim: bool = False) -> Tensor:
        """Compute the norm (magnitude) of the quaternion.

        Args:
            keepdim: whether to retain the last dimension.

        Returns:
            The norm of the quaternion(s) as a tensor.

        Example:
            >>> q = Quaternion.identity()
            >>> q.norm()
            tensor(1.)

        """
        # p==2, dim|axis==-1, keepdim
        return self.data.norm(2, -1, keepdim)

    def normalize(self) -> "Quaternion":
        """Return a normalized (unit) quaternion.

        Returns:
            The normalized quaternion.

        Example:
            >>> q = Quaternion(tensor([2., 1., 0., 0.]))
            >>> q_norm = q.normalize()

        """
        return Quaternion(normalize_quaternion(self.data))

    def conj(self) -> "Quaternion":
        """Compute the conjugate of the quaternion.

        Returns:
            The conjugate quaternion, with the vector part negated.

        Example:
            >>> q = Quaternion(tensor([1., 2., 3., 4.]))
            >>> q_conj = q.conj()

        """
        return Quaternion(concatenate((self.real[..., None], -self.vec), -1))

    def inv(self) -> "Quaternion":
        """Compute the inverse of the quaternion.

        Returns:
            The inverse quaternion.

        Example:
            >>> q = Quaternion.identity()
            >>> q_inv = q.inv()

        """
        return self.conj() / self.squared_norm()

    def squared_norm(self) -> Tensor:
        """Compute the squared norm (magnitude) of the quaternion.

        Returns:
            The squared norm of the quaternion(s) as a tensor.

        Example:
            >>> q = Quaternion.identity()
            >>> q.squared_norm()
            tensor(1.)

        """
        return batched_dot_product(self.vec, self.vec) + self.real**2


def average_quaternions(Q: "Quaternion", w: Optional[torch.Tensor] = None) -> "Quaternion":
    """Compute (weighted) average of multiple quaternions.

    Args:
        Q (Quaternion): quaternion object containing data of shape (M, 4).
        w (torch.Tensor, optional): Weights of shape (M,). If None, uniform weights are used.


    Returns:
        Quaternion: averaged quaternion (shape (4,)), wrapped back in the Quaternion class.
    """
    data = Q.data
    KORNIA_CHECK_TYPE(Q, Quaternion)

    M = data.shape[0]
    if w is None:
        A = (data.T @ data) / M
    else:
        w = w.to(data.device, dtype=data.dtype)
        if w.numel() != M:
            raise ValueError(f"weights length {w.numel()} must match number of quaternions {M}")
        w = w / w.sum()
        A = data.T @ torch.diag(w) @ data

    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    q_avg = eigenvectors[:, torch.argmax(eigenvalues)]
    q_avg = q_avg / q_avg.norm()

    return Quaternion(q_avg.unsqueeze(0))

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
from torch import nn

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


class Quaternion(nn.Module):
    r"""Base class to represent a Quaternion.

    A quaternion is a four dimensional vector representation of a rotation transformation in 3d.
    See more: https://en.wikipedia.org/wiki/Quaternion

    The general definition of a quaternion is given by:

    .. math::

        Q = a + b \cdot \mathbf{i} + c \cdot \mathbf{j} + d \cdot \mathbf{k}

    Thus, we represent a rotation quaternion as a contiguous torch.Tensor structure to
    perform rigid bodies transformations:

    .. math::

        Q = \begin{bmatrix} q_w & q_x & q_y & q_z \end{bmatrix}

    Convention:
        - ``data`` is ``(w, x, y, z)``, real part first, with shape :math:`(*, 4)`; only the last axis is checked.
          Other libraries' orders, such as scipy's default ``(x, y, z, w)``, are mapped on
          :ref:`Rotations and rigid motions <rotation-conventions>`.
        - ``*`` between quaternions is the Hamilton product, so ``(q1 * q2).matrix()`` is
          ``q1.matrix() @ q2.matrix()``, and ``q`` and ``-q`` are the same rotation. A float or tensor operand of
          ``+``, ``-``, ``*`` or ``/`` is the real quaternion ``[s, 0, 0, 0]``, and a tensor holds one such scalar per
          quaternion of the batch.
        - Nothing normalises the stored data or the results of ``*``, ``**`` and ``inv()``. ``matrix()``,
          ``to_axis_angle()``, ``polar_angle`` and ``slerp`` read only the direction of ``q``, so any positive scale
          gives the same result.
        - Known defects: ``to_euler()`` of a non-unit ``q`` returns wrong angles
          (`#3953 <https://github.com/kornia/kornia/issues/3953>`_); the gradient of ``polar_angle`` is NaN at the
          identity (`#4927 <https://github.com/kornia/kornia/issues/4927>`_); data given as a plain tensor is not
          registered with ``nn.Module``, so an enclosing module's ``state_dict()``, ``load_state_dict()`` and
          ``.to()`` skip it, while an ``nn.Parameter`` is saved, restored and moved
          (`#4923 <https://github.com/kornia/kornia/issues/4923>`_).

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

    _data: Union[torch.Tensor, nn.Parameter]

    def __init__(self, data: Union[torch.Tensor, nn.Parameter]) -> None:
        """Construct a quaternion from torch.Tensor or parameter data.

        Args:
            data: torch.Tensor or parameter containing the quaternion data with the shape of :math:`(*, 4)`.

        Example:
            >>> # Create with torch.tensor(no gradients tracked by default)
            >>> data = torch.tensor([1., 0., 0., 0.])
            >>> q1 = Quaternion(data)
            >>> # Create with parameter (gradients tracked)
            >>> param_data = torch.nn.Parameter(torch.tensor([1., 0., 0., 0.]))
            >>> q2 = Quaternion(param_data)

        """
        super().__init__()

        if not isinstance(data, (torch.Tensor, nn.Parameter)):
            raise TypeError(f"Expected torch.Tensor or nn.Parameter, got {type(data)}")
        # KORNIA_CHECK_SHAPE(data, ["B", "4"])   # FIXME: resolve shape bugs. @edgarriba

        if data.ndim == 0 or data.shape[-1] != 4:
            raise ValueError(f"Quaternion input must have last dimension == 4. Got shape {tuple(data.shape)}")
        self._data = data

    def to(self, *args: Any, **kwargs: Any) -> "Quaternion":
        """Move and/or cast the quaternion data.

        Args:
            *args: Arguments to pass to torch.Tensor.to()
            **kwargs: Keyword arguments to pass to torch.Tensor.to()

        Returns:
            A new Quaternion with converted data.
        """
        return Quaternion(self._data.to(*args, **kwargs))

    def _to_scalar_quaternion(self, value: Union[torch.Tensor, float]) -> "Quaternion":
        """Convert a scalar, torch.Tensor, or numeric value to a scalar quaternion.

        A scalar quaternion has the form [real, 0, 0, 0] where real is the input value.

        Args:
            value: The scalar, torch.Tensor, or numeric value to convert.

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
            >>> (-q).data
            tensor([-1., -0., -0., -0.])

        """
        return Quaternion(-self.data)

    def __add__(self, right: Union["Quaternion", torch.Tensor, float]) -> "Quaternion":
        """Add a given quaternion, scalar, or torch.Tensor.

        Args:
            right: the quaternion, scalar, or torch.Tensor to add.

        Example:
            >>> q1 = Quaternion.identity()
            >>> q2 = Quaternion(torch.tensor([2., 0., 1., 1.]))
            >>> q3 = q1 + q2
            >>> q3.data
            tensor([3., 0., 1., 1.])

        """
        if isinstance(right, Quaternion):
            return Quaternion(self.data + right.data)
        right_quat = self._to_scalar_quaternion(right)
        return Quaternion(self.data + right_quat.data)

    def __sub__(self, right: Union["Quaternion", torch.Tensor, float]) -> "Quaternion":
        """Subtract a given quaternion, scalar, or torch.Tensor.

        Args:
            right: the quaternion, scalar, or torch.Tensor to subtract.

        Example:
            >>> q1 = Quaternion(torch.tensor([2., 0., 1., 1.]))
            >>> q2 = Quaternion.identity()
            >>> q3 = q1 - q2
            >>> q3.data
            tensor([1., 0., 1., 1.])

        """
        if isinstance(right, Quaternion):
            return Quaternion(self.data - right.data)
        right_quat = self._to_scalar_quaternion(right)
        # For scalar operations, ensure we return a torch.Tensor to preserve gradients
        result_data = self.data - right_quat.data
        if isinstance(result_data, nn.Parameter):
            result_data = result_data.data  # Convert to torch.Tensor to preserve gradients
        return Quaternion(result_data)

    def __mul__(self, right: Union["Quaternion", torch.Tensor, float]) -> "Quaternion":
        # If right is a Quaternion, do quaternion multiplication
        if isinstance(right, Quaternion):
            new_real = self.real * right.real - batched_dot_product(self.vec, right.vec)
            new_vec = (
                self.real[..., None] * right.vec
                + right.real[..., None] * self.vec
                + torch.linalg.cross(self.vec, right.vec, dim=-1)
            )
            return Quaternion(torch.cat((new_real[..., None], new_vec), -1))

        # If right is a scalar/torch.Tensor, convert to scalar quaternion and multiply
        right_quat = self._to_scalar_quaternion(right)
        new_real = self.real * right_quat.real - batched_dot_product(self.vec, right_quat.vec)
        new_vec = (
            self.real[..., None] * right_quat.vec
            + right_quat.real[..., None] * self.vec
            + torch.linalg.cross(self.vec, right_quat.vec, dim=-1)
        )
        return Quaternion(torch.cat((new_real[..., None], new_vec), -1))

    def __rmul__(self, left: Union[torch.Tensor, float]) -> "Quaternion":
        """Right multiplication (left * self) where left is a scalar or torch.Tensor."""
        left_quat = self._to_scalar_quaternion(left)
        new_real = left_quat.real * self.real - batched_dot_product(left_quat.vec, self.vec)
        new_vec = (
            left_quat.real[..., None] * self.vec
            + self.real[..., None] * left_quat.vec
            + torch.linalg.cross(left_quat.vec, self.vec, dim=-1)
        )
        return Quaternion(torch.cat((new_real[..., None], new_vec), -1))

    def __div__(self, right: Union[torch.Tensor, "Quaternion", float]) -> "Quaternion":
        if isinstance(right, Quaternion):
            return self * right.inv()
        # For scalars/tensors, just divide the quaternion data directly
        if isinstance(right, (int, float)):
            right_tensor = torch.tensor(right, device=self.data.device, dtype=self.data.dtype)
        else:
            right_tensor = right.to(device=self.data.device, dtype=self.data.dtype)

        # For division by scalar, expand to [right, right, right, right] for element-wise division
        if right_tensor.dim() == 0:  # scalar
            divisor = right_tensor.expand_as(self.data[..., 0]).unsqueeze(-1).expand_as(self.data)
        else:
            # Broadcast the torch.Tensor to match the quaternion dimensions
            divisor = right_tensor.unsqueeze(-1).expand_as(self.data)

        # For scalar operations, ensure we return a torch.Tensor to preserve gradients
        result_data = self.data / divisor
        if isinstance(result_data, nn.Parameter):
            result_data = result_data.data  # Convert to torch.Tensor to preserve gradients
        return Quaternion(result_data)

    def __truediv__(self, right: Union[torch.Tensor, "Quaternion", float]) -> "Quaternion":
        return self.__div__(right)

    def __radd__(self, left: Union[torch.Tensor, float]) -> "Quaternion":
        """Right addition (left + self) where left is a scalar or torch.Tensor."""
        left_quat = self._to_scalar_quaternion(left)
        return left_quat + self

    def __rsub__(self, left: Union[torch.Tensor, float]) -> "Quaternion":
        """Right subtraction (left - self) where left is a scalar or torch.Tensor."""
        left_quat = self._to_scalar_quaternion(left)
        return left_quat - self

    def __rtruediv__(self, left: Union[torch.Tensor, float]) -> "Quaternion":
        """Right division (left / self) where left is a scalar or torch.Tensor."""
        left_quat = self._to_scalar_quaternion(left)
        return left_quat / self

    def __rdiv__(self, left: Union[torch.Tensor, float]) -> "Quaternion":
        """Right division (left / self) where left is a scalar or torch.Tensor."""
        return self.__rtruediv__(left)

    def __pow__(self, t: float) -> "Quaternion":
        r"""Return the power of a quaternion raised to exponent t.

        For :math:`q = \|q\| (\cos\theta + n \sin\theta)` this is
        :math:`q^t = \|q\|^t (\cos t\theta + n \sin t\theta)`, so ``q**2 == q * q`` and ``q**-1 == q.inv()``.

        Args:
            t: raised exponent.

        Example:
            >>> q = Quaternion(torch.tensor([1., .5, 0., 0.]))
            >>> q_pow = q**2

        """
        w = self.scalar[..., None]
        vec_norm = self.vec.norm(dim=-1, keepdim=True)
        theta = torch.atan2(vec_norm, w)
        # On the real axis (|v| = 0) take sin(t * theta) / |v| from its limit t * cos(t * theta) / w, exact when
        # theta = 0 or t is an integer, and keep both arms' denominators nonzero so values and gradients stay finite.
        is_real = vec_norm == 0
        safe_vec_norm = torch.where(is_real, torch.ones_like(vec_norm), vec_norm)
        safe_w = torch.where(w == 0, torch.ones_like(w), w)
        sin_ratio = torch.where(is_real, t * (t * theta).cos() / safe_w, (t * theta).sin() / safe_vec_norm)
        scale = self.norm(keepdim=True) ** t
        return Quaternion(torch.cat((scale * (t * theta).cos(), scale * sin_ratio * self.vec), -1))

    @property
    def data(self) -> torch.Tensor:
        """Return the underlying data with shape :math:`(B, 4)`."""
        return self._data

    @property
    def coeffs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a tuple with the underlying coefficients in WXYZ order."""
        return self.w, self.x, self.y, self.z

    @property
    def real(self) -> torch.Tensor:
        """Return the real part with shape :math:`(B,)`.

        Alias for :attr:`~kornia.geometry.quaternion.Quaternion.w`.
        """
        return self.w

    @property
    def vec(self) -> torch.Tensor:
        """Return the vector with the imaginary part with shape :math:`(B, 3)`."""
        return self.data[..., 1:]

    @property
    def q(self) -> torch.Tensor:
        """Return the underlying data with shape :math:`(B, 4)`.

        Alias for :func:`~kornia.geometry.quaternion.Quaternion.data`
        """
        return self.data

    @property
    def scalar(self) -> torch.Tensor:
        """Return a scalar with the real with shape :math:`(B,)`.

        Alias for :attr:`~kornia.geometry.quaternion.Quaternion.w`.
        """
        return self.real

    @property
    def w(self) -> torch.Tensor:
        """Return the :math:`q_w` with shape :math:`(B,)`."""
        return self.data[..., 0]

    @property
    def x(self) -> torch.Tensor:
        """Return the :math:`q_x` with shape :math:`(B,)`."""
        return self.data[..., 1]

    @property
    def y(self) -> torch.Tensor:
        """Return the :math:`q_y` with shape :math:`(B,)`."""
        return self.data[..., 2]

    @property
    def z(self) -> torch.Tensor:
        """Return the :math:`q_z` with shape :math:`(B,)`."""
        return self.data[..., 3]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying data with shape :math:`(B, 4)`."""
        return tuple(self.data.shape)

    @property
    def polar_angle(self) -> torch.Tensor:
        r"""Return the polar angle :math:`\arccos(w / |q|)` in :math:`[0, \pi]`, with shape :math:`(B,)`.

        ``q`` rotates by twice this angle about the axis ``vec``.

        Example:
            >>> q = Quaternion.identity()
            >>> q.polar_angle
            tensor(0.)

        """
        return (self.scalar / self.norm()).acos()

    def matrix(self) -> torch.Tensor:
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
    def from_matrix(cls, matrix: torch.Tensor) -> "Quaternion":
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
    def from_euler(cls, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> "Quaternion":
        """Create a quaternion from Euler angles.

        Args:
            roll: the roll euler angle.
            pitch: the pitch euler angle.
            yaw: the yaw euler angle.

        Example:
            >>> roll, pitch, yaw = torch.tensor(0), torch.tensor(1), torch.tensor(0)
            >>> q = Quaternion.from_euler(roll, pitch, yaw)
            >>> q.data
            tensor([0.8776, 0.0000, 0.4794, 0.0000])

        """
        w, x, y, z = quaternion_from_euler(roll=roll, pitch=pitch, yaw=yaw)
        q = torch.stack((w, x, y, z), -1)
        return cls(q)

    def to_euler(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert the quaternion to a triple of Euler angles (roll, pitch, yaw).

        Example:
            >>> q = Quaternion.from_euler(torch.tensor(0.3), torch.tensor(0.2), torch.tensor(0.1))
            >>> roll, pitch, yaw = q.to_euler()
            >>> roll
            tensor(0.3000)
            >>> pitch
            tensor(0.2000)
            >>> yaw
            tensor(0.1000)

        """
        return euler_from_quaternion(self.w, self.x, self.y, self.z)

    @classmethod
    def from_axis_angle(cls, axis_angle: torch.Tensor) -> "Quaternion":
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

    def to_axis_angle(self) -> torch.Tensor:
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
        cls,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
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
        data = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
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
        return cls(torch.tensor([w, x, y, z]))

    # TODO: update signature
    # def random(cls, shape: Optional[List] = None, device = None, dtype = None) -> 'Quaternion':
    @classmethod
    def random(
        cls,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
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

        r1, r2, r3 = torch.rand((3, *rand_shape), device=device, dtype=dtype)
        q1 = (1.0 - r1).sqrt() * ((2 * pi * r2).sin())
        q2 = (1.0 - r1).sqrt() * ((2 * pi * r2).cos())
        q3 = r1.sqrt() * (2 * pi * r3).sin()
        q4 = r1.sqrt() * (2 * pi * r3).cos()
        return cls(torch.stack((q1, q2, q3, q4), -1))

    def slerp(self, q1: "Quaternion", t: float) -> "Quaternion":
        """Return a unit quaternion spherically interpolated between quaternions self.q and q1.

        The interpolation follows the shorter arc between the two rotations, whatever the signs of the stored
        quaternions: ``q1`` and ``-q1`` give the same result, and at ``t = 1`` the output is ``q1`` or ``-q1``.
        The exception is a half turn whose relative quaternion ``self.inv() * q1`` has a real part of exactly zero:
        both arcs are then equally short and the path is not unique. The arc taken follows the sign of the vector
        part of ``self.inv() * q1``, so ``q1`` and ``-q1`` take opposite arcs, and the result is not continuous in
        ``q1`` there.

        See more: https://en.wikipedia.org/wiki/Slerp

        Args:
            q1: second quaternion to be interpolated between.
            t: interpolation ratio, ``0`` at ``self`` and ``1`` at ``q1``. It is not validated: values outside
                ``[0, 1]`` extrapolate along the same arc. A per-batch ratio has shape :math:`(B, 1)`.

        Example:
            >>> q0 = Quaternion.identity()
            >>> q1 = Quaternion(torch.tensor([1., .5, 0., 0.]))
            >>> q2 = q0.slerp(q1, .3)

        """
        KORNIA_CHECK_TYPE(q1, Quaternion)
        q0 = self.normalize()
        q1 = q1.normalize()
        # q0 * exp(t * log(q0^-1 q1)): the principal log of the relative rotation selects the shorter arc, and both
        # conversions keep a finite gradient at the identity (q0 == q1).
        rel = quaternion_to_axis_angle((q0.inv() * q1).data)
        return q0 * Quaternion(axis_angle_to_quaternion(t * rel))

    def norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm (magnitude) of the quaternion.

        Args:
            keepdim: whether to retain the last dimension.

        Returns:
            The norm of the quaternion(s) as a torch.Tensor.

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
            >>> q = Quaternion(torch.tensor([2., 1., 0., 0.]))
            >>> q_norm = q.normalize()

        """
        return Quaternion(normalize_quaternion(self.data))

    def conj(self) -> "Quaternion":
        """Compute the conjugate of the quaternion.

        Returns:
            The conjugate quaternion, with the vector part negated.

        Example:
            >>> q = Quaternion(torch.tensor([1., 2., 3., 4.]))
            >>> q_conj = q.conj()

        """
        return Quaternion(torch.cat((self.real[..., None], -self.vec), -1))

    def inv(self) -> "Quaternion":
        """Compute the inverse of the quaternion.

        Returns:
            The inverse quaternion.

        Example:
            >>> q = Quaternion.identity()
            >>> q_inv = q.inv()

        """
        return self.conj() / self.squared_norm()

    def squared_norm(self) -> torch.Tensor:
        """Compute the squared norm (magnitude) of the quaternion.

        Returns:
            The squared norm of the quaternion(s) as a torch.Tensor.

        Example:
            >>> q = Quaternion.identity()
            >>> q.squared_norm()
            tensor(1.)

        """
        return batched_dot_product(self.vec, self.vec) + self.real**2


def average_quaternions(Q: "Quaternion", w: Optional[torch.Tensor] = None) -> "Quaternion":
    r"""Compute (weighted) average of multiple quaternions.

    Convention:
        - The chordal mean of scipy's ``Rotation.mean``: the eigenvector of
          :math:`\sum_i w_i q_i q_i^\top / \sum_i w_i` with the largest eigenvalue. ``q_i`` and ``-q_i`` count the
          same, and the sign of the result is arbitrary.
        - Only the ratios of ``w`` matter.
        - Known defect: the members are not normalised, so a member of norm ``n`` counts with an extra weight
          ``n**2``, and negative weights are not rejected
          (`#4974 <https://github.com/kornia/kornia/issues/4974>`_).

    Args:
        Q (Quaternion): quaternion object containing data of shape (M, 4).
        w (torch.Tensor, optional): Weights of shape (M,). If None, uniform weights are used.


    Returns:
        Quaternion: averaged quaternion of shape (1, 4), wrapped back in the Quaternion class.
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

    orig_dtype = A.dtype
    if A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    # Use float32 eigenvalues for argmax to avoid half-precision rounding
    # changing which eigenvector is selected when eigenvalues are close.
    max_idx = torch.argmax(eigenvalues)
    eigenvectors = eigenvectors.to(orig_dtype)
    q_avg = eigenvectors[:, max_idx]
    q_avg = q_avg / q_avg.norm()

    return Quaternion(q_avg.unsqueeze(0))

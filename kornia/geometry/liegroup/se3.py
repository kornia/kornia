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

# kornia.geometry.se3 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se3.py
from __future__ import annotations

from typing import Optional, Union, overload

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_DEVICES, KORNIA_CHECK_SHAPE
from kornia.core.exceptions import DeviceError, TypeCheckError
from kornia.core.utils import register_module_state
from kornia.geometry.liegroup.so3 import So3, _check_points_layout, _so3_rotate, _so3_small_angle_coefficients
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.vector import Vector3


def _check_se3_r_t_shape(r: Quaternion | So3, t: torch.Tensor) -> None:
    q = r.q if isinstance(r, So3) else r
    r_shape = q.data.shape
    r_batch = r_shape[:-1]
    t_batch = t.shape[:-1]
    if len(r_batch) == 0 and len(t_batch) == 0:
        return
    if len(r_batch) > 0 and len(t_batch) > 0:
        try:
            torch.broadcast_shapes(r_batch, t_batch)
            return
        except RuntimeError:
            raise ValueError(f"Incompatible rotation shape {r_shape} and translation shape {t.shape}.") from None
    if (len(r_batch) == 0 and t_batch == (1,)) or (r_batch == (1,) and len(t_batch) == 0):
        return
    raise ValueError(
        f"Incompatible rotation shape {r_shape} and translation shape {t.shape}: both must be batched or unbatched."
    )


def _check_se3_r_t_dtype_device(r: Quaternion | So3, t: torch.Tensor) -> None:
    """Reject a translation the rotation's methods cannot combine with: another dtype or another device."""
    q = r.q if isinstance(r, So3) else r
    q_data = q.data
    if t.is_complex():
        raise TypeCheckError(
            f"Se3 translation must be a real floating-point tensor; got {t.dtype}.",
            actual_type=t.dtype,
        )
    if not t.is_floating_point():
        # register_module_state keeps an existing nn.Parameter as given, so it cannot reject an integer one
        raise TypeCheckError(
            f"Se3 stores 'translation' as module state, which needs a floating-point tensor; got {t.dtype}. "
            "Cast it first, for example with `.float()`.",
            actual_type=t.dtype,
        )
    if t.dtype != q_data.dtype:
        raise TypeCheckError(
            f"Se3 translation must have the dtype of its rotation, {q_data.dtype}; got {t.dtype}. "
            "Cast one of them first, for example with `.to(dtype)`.",
            actual_type=t.dtype,
            expected_type=q_data.dtype,
        )
    if t.device != q_data.device:
        raise DeviceError(
            f"Se3 translation must be on the device of its rotation, {q_data.device}; got {t.device}.",
            actual_devices=[q_data.device, t.device],
            expected_device=q_data.device,
        )


class Se3(nn.Module):
    r"""Base class to represent the Se3 group.

    The SE(3) is the group of rigid body transformations about the origin of three-dimensional Euclidean
    space :math:`R^3` under the operation of composition.
    See more: https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf

    Convention:
        - The tangent vector is :math:`[\upsilon, \omega]`, translation first
          (:ref:`Rotations and rigid motions <rotation-conventions>` maps it onto Sophus and GTSAM): ``exp`` rotates by
          ``So3.exp(omega)`` and translates by :math:`V(\omega) \upsilon`, ``hat`` is
          :math:`[[\hat\omega, \upsilon], [0, 0]]`, and ``log`` is principal, with :math:`|\omega| \le \pi`.
        - ``matrix()`` is the 4x4 :math:`[[R, t], [0, 1]]`. ``a * b`` is ``a.matrix() @ b.matrix()``, so ``b`` acts
          first, and ``g * p`` is :math:`R p + t`. ``adjoint()`` is :math:`[[R, \hat t R], [0, R]]`.
        - The rotation is an :class:`~kornia.geometry.liegroup.So3`, whose storage and point-shape conventions
          apply: a non-unit quaternion stands for the rotation of its direction. ``from_matrix`` ignores the bottom
          row.
        - ``t`` is always a tensor registered as module state, whichever constructor built the pose; a ``Vector3``
          passed to the constructor is unwrapped. ``g * p`` returns a ``Vector3`` only when ``p`` is one.
        - ``state_dict()``, ``load_state_dict()`` and module device/dtype conversions include both the rotation
          and translation. Rotation state follows :class:`~kornia.geometry.quaternion.Quaternion`'s
          buffer/parameter convention.
        - ``a * b`` and ``g * p`` compute in float32 for float16/bfloat16 inputs and return the input dtype,
          rounding :math:`R p + t` once; ``matrix()`` and ``inverse()`` take the rotation from
          :class:`~kornia.geometry.liegroup.So3`, which does the same.

    Example:
        >>> q = Quaternion.identity()
        >>> s = Se3(q, torch.ones(3))
        >>> s.r
        tensor([1., 0., 0., 0.])
        >>> s.t
        Parameter containing:
        tensor([1., 1., 1.], requires_grad=True)

    """

    def __init__(self, rotation: Quaternion | So3, translation: Vector3 | torch.Tensor) -> None:
        """Construct the base class.

        Internally represented by an So3 rotation and a translation 3-vector.

        Args:
            rotation: So3 group encompassing a rotation, or a Quaternion to wrap in one; it is stored as given, and
                a non-unit quaternion stands for the rotation of its direction.
            translation: translation torch.Tensor with the shape of ``(B, 3)``, or a Vector3 wrapping one; the
                tensor is what gets stored. It must have the rotation's dtype and device.

        Raises:
            TypeCheckError: if the translation is an integer, bool or complex tensor (an ``nn.Parameter`` too), or its
                dtype differs from the rotation's (``inverse``, ``adjoint`` and the other methods cannot combine them).
            DeviceError: if the translation is on another device than the rotation.

        Example:
            >>> from kornia.geometry.quaternion import Quaternion
            >>> q = Quaternion.identity(batch_size=1)
            >>> s = Se3(q, torch.ones((1, 3)))
            >>> s.r
            tensor([[1., 0., 0., 0.]])
            >>> s.t
            Parameter containing:
            tensor([[1., 1., 1.]], requires_grad=True)

        """
        super().__init__()
        # KORNIA_CHECK_TYPE(rotation, (Quaternion, So3))
        if not isinstance(rotation, (Quaternion, So3)):
            raise TypeError(f"rotation type is {type(rotation)}")
        # KORNIA_CHECK_TYPE(translation, (Vector3, torch.Tensor))
        if not isinstance(translation, (Vector3, torch.Tensor)):
            raise TypeError(f"translation type is {type(translation)}")
        _t_data = translation.data if isinstance(translation, Vector3) else translation
        KORNIA_CHECK_SHAPE(_t_data, ["*", "3"])
        _check_se3_r_t_shape(rotation, _t_data)
        _check_se3_r_t_dtype_device(rotation, _t_data)
        self._translation: torch.Tensor
        self._rotation: So3
        register_module_state(self, "_translation", _t_data)
        if isinstance(rotation, Quaternion):
            self._rotation = So3(rotation)
        else:
            self._rotation = rotation

    def __repr__(self) -> str:
        return f"rotation: {self.r}\ntranslation: {self.t}"

    def __getitem__(self, idx: int | slice) -> Se3:
        return Se3(self._rotation[idx], self._translation[idx])

    def _mul_se3(self, right: Se3) -> Se3:
        _r = self.r * right.r
        # t + R t2 is rounded once, so float16/bfloat16 sum the float32 rotated translation
        _t = (self.t + _so3_rotate(self.r.q, right.t)).to(torch.promote_types(self.t.dtype, right.t.dtype))
        return Se3(_r, _t)

    @overload
    def __mul__(self, right: Se3) -> Se3: ...

    @overload
    def __mul__(self, right: Vector3) -> Vector3: ...

    @overload
    def __mul__(self, right: torch.Tensor) -> torch.Tensor: ...

    def __mul__(self, right: Se3 | Vector3 | torch.Tensor) -> Se3 | Vector3 | torch.Tensor:
        """Compose two Se3 transformations, or transform points.

        Both compute in float32 for float16/bfloat16 inputs and return the input dtype.

        Args:
            right: the other Se3 transformation, or points as a tensor or ``Vector3``: ``(3,)`` or ``(N, 3)`` for
                an unbatched pose, ``(B, 3)`` for a batch of ``B`` (a batch of one also maps ``(N, 3)``).

        Return:
            The resulting Se3 transformation, or the transformed points.

        """
        so3 = self.so3
        t = self.t
        if isinstance(right, Se3):
            # https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se3.py#L97
            return self._mul_se3(right)
        if isinstance(right, (Vector3, torch.Tensor)):
            _right_data = right if isinstance(right, torch.Tensor) else right.data
            KORNIA_CHECK_SHAPE(_right_data, ["*", "N"])
            _check_points_layout(so3.q, _right_data)
            # R p + t is rounded once, so float16/bfloat16 add the translation to the float32 rotated point
            out = (_so3_rotate(so3.q, _right_data) + t).to(torch.promote_types(t.dtype, _right_data.dtype))
            return Vector3(out) if isinstance(right, Vector3) else out
        raise TypeError(f"Unsupported type: {type(right)}")

    @property
    def so3(self) -> So3:
        """Return the underlying rotation(So3)."""
        return self._rotation

    @property
    def quaternion(self) -> Quaternion:
        """Return the underlying rotation(Quaternion)."""
        return self._rotation.q

    @property
    def r(self) -> So3:
        """Return the underlying rotation(So3)."""
        return self._rotation

    @property
    def t(self) -> torch.Tensor:
        """Return the underlying translation vector of shape ``(B,3)``."""
        return self._translation

    @property
    def rotation(self) -> So3:
        """Return the underlying `rotation(So3)`."""
        return self._rotation

    @property
    def translation(self) -> torch.Tensor:
        """Return the underlying translation vector of shape ``(B,3)``."""
        return self._translation

    @staticmethod
    def exp(v: torch.Tensor) -> Se3:
        """Convert elements of lie algebra to elements of lie group.

        Args:
            v: vector of shape ``(B, 6)``.

        Example:
            >>> v = torch.zeros((1, 6))
            >>> s = Se3.exp(v)
            >>> s.r
            tensor([[1., 0., 0., 0.]])
            >>> s.t
            Parameter containing:
            tensor([[0., 0., 0.]], requires_grad=True)

        """
        KORNIA_CHECK_SHAPE(v, ["*", "6"])
        upsilon = v[..., :3]
        omega = v[..., 3:]
        omega_hat = So3.hat(omega)
        omega_hat_sq = omega_hat @ omega_hat
        theta_sq = batched_dot_product(omega, omega)
        nonzero = theta_sq > 0
        # sqrt has an unbounded derivative at 0, so it is taken on a substituted 1 at omega = 0 and
        # the where selects the exact 0 (kornia#4229's shape). The coefficients of V are finite there
        # (1/2 and 1/6) and even in theta, so V @ upsilon is taken for every element, which keeps
        # d t / d omega = -0.5 [upsilon]_x at the identity (kornia#4953); V(0) = I exactly.
        safe_theta_sq = torch.where(nonzero, theta_sq, torch.ones_like(theta_sq))
        theta = torch.where(nonzero, safe_theta_sq.sqrt(), torch.zeros_like(theta_sq))
        R = So3.exp(omega)
        a, b, _ = _so3_small_angle_coefficients(theta)
        V = (
            torch.eye(3, device=v.device, dtype=v.dtype)
            + a[..., None, None] * omega_hat
            + b[..., None, None] * omega_hat_sq
        )
        return Se3(R, (upsilon[..., None, :] * V).sum(-1))

    def log(self) -> torch.Tensor:
        """Convert elements of lie group  to elements of lie algebra.

        Example:
            >>> from kornia.geometry.quaternion import Quaternion
            >>> q = Quaternion.identity()
            >>> Se3(q, torch.zeros(3)).log()
            tensor([0., 0., 0., 0., 0., 0.], grad_fn=<CatBackward0>)

        """
        omega = self.r.log()
        theta_sq = batched_dot_product(omega, omega)
        nonzero = theta_sq > 0
        # clamp_min(1e-12) bounds the value, not the gradient: on torch < 2.14 clamp passes the
        # incoming gradient straight through at its bound, and 1e-12 underflows to 0 in float16
        # anyway, so sqrt's unbounded derivative at 0 reached v.grad as nan (kornia#4229). Keep
        # the floor on the branch that is selected -- byte-identical for every theta_sq > 0 --
        # and take the exact zero elsewhere, where V_inv is the identity.
        safe_theta_sq = torch.where(nonzero, theta_sq.clamp_min(1e-12), torch.ones_like(theta_sq))
        theta = torch.where(nonzero, safe_theta_sq.sqrt(), torch.zeros_like(theta_sq))
        t = self.t
        omega_hat = So3.hat(omega)
        omega_hat_sq = omega_hat @ omega_hat
        # c is finite at theta = 0 (1/12), so V^-1 @ t is taken for every element, which keeps
        # d upsilon / d q_vec = [t]_x at the identity (kornia#4953).
        _, _, c = _so3_small_angle_coefficients(theta)
        V_inv = (
            torch.eye(3, device=omega.device, dtype=omega.dtype) - 0.5 * omega_hat + c[..., None, None] * omega_hat_sq
        )
        return torch.cat(((t[..., None, :] * V_inv).sum(-1), omega), -1)

    @staticmethod
    def hat(v: torch.Tensor) -> torch.Tensor:
        """Convert elements from vector space to lie algebra.

        Args:
            v: vector of shape ``(B, 6)``.

        Returns:
            matrix of shape ``(B, 4, 4)``.

        Example:
            >>> v = torch.ones((1, 6))
            >>> m = Se3.hat(v)
            >>> m
            tensor([[[ 0., -1.,  1.,  1.],
                     [ 1.,  0., -1.,  1.],
                     [-1.,  1.,  0.,  1.],
                     [ 0.,  0.,  0.,  0.]]])

        """
        KORNIA_CHECK_SHAPE(v, ["*", "6"])
        upsilon, omega = v[..., :3], v[..., 3:]
        rt = torch.cat((So3.hat(omega), upsilon[..., None]), -1)
        return F.pad(rt, (0, 0, 0, 1))  # add torch.zeros bottom

    @staticmethod
    def vee(omega: torch.Tensor) -> torch.Tensor:
        """Convert elements from lie algebra to vector space.

        Args:
            omega: 4x4-matrix representing lie algebra of shape ``(B,4,4)``.

        Returns:
            vector of shape ``(B,6)``.

        Example:
            >>> v = torch.ones((1, 6))
            >>> omega_hat = Se3.hat(v)
            >>> Se3.vee(omega_hat)
            tensor([[1., 1., 1., 1., 1., 1.]])

        """
        KORNIA_CHECK_SHAPE(omega, ["*", "4", "4"])
        head = omega[..., :3, -1]
        tail = So3.vee(omega[..., :3, :3])
        return torch.cat((head, tail), -1)

    @classmethod
    def identity(
        cls,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> Se3:
        """Create a Se3 group representing an identity rotation and zero translation.

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> s = Se3.identity()
            >>> s.r
            tensor([1., 0., 0., 0.])
            >>> s.t
            Parameter containing:
            tensor([0., 0., 0.], requires_grad=True)

        """
        t = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        if batch_size is not None:
            t = t.repeat(batch_size, 1)

        return cls(So3.identity(batch_size, device, dtype), t)

    def matrix(self) -> torch.Tensor:
        """Return the matrix representation of shape ``(B, 4, 4)``.

        Example:
            >>> s = Se3(So3.identity(), torch.ones(3))
            >>> s.matrix()
            tensor([[1., 0., 0., 1.],
                    [0., 1., 0., 1.],
                    [0., 0., 1., 1.],
                    [0., 0., 0., 1.]], grad_fn=<CopySlices>)

        """
        r_matrix = self.r.matrix()
        b_batch = torch.broadcast_shapes(r_matrix.shape[:-2], self.t.shape[:-1])
        if r_matrix.shape[:-2] != b_batch:
            r_matrix = r_matrix.expand(b_batch + (3, 3))
        t = self.t.to(r_matrix.dtype)
        if t.shape[:-1] != b_batch:
            t = t.expand(b_batch + (3,))
        rt = torch.cat((r_matrix, t[..., None]), -1)
        rt_4x4 = F.pad(rt, (0, 0, 0, 1))  # add last row torch.zeros
        rt_4x4[..., -1, -1] = 1.0
        return rt_4x4

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor, check_rotation: bool = False) -> Se3:
        """Create a Se3 group from a matrix.

        Args:
            matrix: torch.Tensor of shape ``(B, 4, 4)``.
            check_rotation: if ``True``, raise ``ValueError`` unless the rotation
                block of every input is a rotation matrix. The default ``False``
                keeps the unchecked behaviour, under which a reflection block
                such as ``diag(-1, 1, 1)`` yields a non-unit quaternion.

        Example:
            >>> s = Se3.from_matrix(torch.eye(4))
            >>> s.r
            tensor([1., 0., 0., 0.])
            >>> s.t
            Parameter containing:
            tensor([0., 0., 0.], requires_grad=True)

        """
        KORNIA_CHECK_SHAPE(matrix, ["*", "4", "4"])
        r = So3.from_matrix(matrix[..., :3, :3], check_rotation=check_rotation)
        t = matrix[..., :3, -1]
        return cls(r, t)

    @classmethod
    def from_qxyz(cls, qxyz: torch.Tensor) -> Se3:
        """Create a Se3 group from a quaternion and a translation vector.

        Args:
            qxyz: torch.Tensor of shape ``(B, 7)`` laid out as ``[qw, qx, qy, qz, x, y, z]``, scalar first; the
                quaternion is stored as given, and a non-unit one stands for the rotation of its direction.

        Example:
            >>> qxyz = torch.tensor([0., 0., 0., 1., 0., 0., 1.])
            >>> s = Se3.from_qxyz(qxyz)
            >>> s.r
            tensor([0., 0., 0., 1.])
            >>> s.t
            Parameter containing:
            tensor([0., 0., 1.], requires_grad=True)

        """
        KORNIA_CHECK_SHAPE(qxyz, ["*", "7"])
        q, xyz = qxyz[..., :4], qxyz[..., 4:]
        return cls(So3.from_wxyz(q), xyz)

    def inverse(self) -> Se3:
        """Return the inverse transformation.

        Example:
            >>> s = Se3(So3.identity(), torch.ones(3))
            >>> s_inv = s.inverse()
            >>> s_inv.r
            tensor([1., -0., -0., -0.])
            >>> s_inv.t
            tensor([-1., -1., -1.], grad_fn=<SliceBackward0>)

        """
        r_inv = self.r.inverse()
        return Se3(r_inv, r_inv * (-1 * self.t))

    @classmethod
    def random(
        cls,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> Se3:
        """Create a Se3 group representing a random transformation.

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> s = Se3.random()
            >>> s = Se3.random(batch_size=3)

        """
        shape: tuple[int, ...]
        if batch_size is None:
            shape = ()
        else:
            KORNIA_CHECK(batch_size >= 1, msg="batch_size must be positive")
            shape = (batch_size,)
        r = So3.random(batch_size, device, dtype)
        t = torch.rand((*shape, 3), device=device, dtype=dtype)
        return cls(r, t)

    @classmethod
    def rot_x(cls, x: torch.Tensor) -> Se3:
        """Construct a x-axis rotation.

        Args:
            x: the x-axis rotation angle.

        """
        zs = torch.zeros_like(x)
        return cls(So3.rot_x(x), torch.stack((zs, zs, zs), -1))

    @classmethod
    def rot_y(cls, y: torch.Tensor) -> Se3:
        """Construct a y-axis rotation.

        Args:
            y: the y-axis rotation angle.

        """
        zs = torch.zeros_like(y)
        return cls(So3.rot_y(y), torch.stack((zs, zs, zs), -1))

    @classmethod
    def rot_z(cls, z: torch.Tensor) -> Se3:
        """Construct a z-axis rotation.

        Args:
            z: the z-axis rotation angle.

        """
        zs = torch.zeros_like(z)
        return cls(So3.rot_z(z), torch.stack((zs, zs, zs), -1))

    @classmethod
    def trans(cls, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Se3:
        """Construct a translation only Se3 instance.

        Args:
            x: the x-axis translation.
            y: the y-axis translation.
            z: the z-axis translation.

        """
        KORNIA_CHECK(x.shape == y.shape)
        KORNIA_CHECK(y.shape == z.shape)
        KORNIA_CHECK_SAME_DEVICES([x, y, z])
        batch_size = x.shape[0] if len(x.shape) > 0 else None
        rotation = So3.identity(batch_size, x.device, x.dtype)
        return cls(rotation, torch.stack((x, y, z), -1))

    @classmethod
    def trans_x(cls, x: torch.Tensor) -> Se3:
        """Construct a x-axis translation.

        Args:
            x: the x-axis translation.

        """
        zs = torch.zeros_like(x)
        return cls.trans(x, zs, zs)

    @classmethod
    def trans_y(cls, y: torch.Tensor) -> Se3:
        """Construct a y-axis translation.

        Args:
            y: the y-axis translation.

        """
        zs = torch.zeros_like(y)
        return cls.trans(zs, y, zs)

    @classmethod
    def trans_z(cls, z: torch.Tensor) -> Se3:
        """Construct a z-axis translation.

        Args:
            z: the z-axis translation.

        """
        zs = torch.zeros_like(z)
        return cls.trans(zs, zs, z)

    def adjoint(self) -> torch.Tensor:
        """Return the adjoint matrix of shape ``(B, 6, 6)``.

        Example:
            >>> s = Se3.identity()
            >>> s.adjoint()
            tensor([[1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 1.]], grad_fn=<CatBackward0>)

        """
        R = self.so3.matrix()
        z = torch.zeros_like(R)
        row0 = torch.cat((R, So3.hat(self.t) @ R), -1)
        row1 = torch.cat((z, R), -1)
        return torch.cat((row0, row1), -2)

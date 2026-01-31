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

# kornia.geometry.so3 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se3.py
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_DEVICES, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE
from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.vector import Vector3


class Se3(nn.Module):
    r"""Base class to represent the Se3 group.

    The SE(3) is the group of rigid body transformations about the origin of three-dimensional Euclidean
    space :math:`R^3` under the operation of composition.
    See more: https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf

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

        Internally represented by a unit quaternion `q` and a translation 3-vector.

        Args:
            rotation: So3 group encompassing a rotation.
            translation: Vector3 or translation torch.Tensor with the shape of :math:`(B, 3)`.

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
        KORNIA_CHECK_TYPE(rotation, (Quaternion, So3))
        KORNIA_CHECK_TYPE(translation, (Vector3, torch.Tensor))
        KORNIA_CHECK_SHAPE(translation, ["*", "3"])
        self._translation: Vector3 | nn.Parameter
        self._rotation: So3
        self._translation = translation
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
        _t = self.t + self.r * right.t
        return Se3(_r, _t)

    def __mul__(self, right: Se3) -> Se3 | Vector3 | torch.Tensor:
        """Compose two Se3 transformations.

        Args:
            right: the other Se3 transformation.

        Return:
            The resulting Se3 transformation.

        """
        so3 = self.so3
        t = self.t
        if isinstance(right, Se3):
            # https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se3.py#L97
            return self._mul_se3(right)
        elif isinstance(right, (Vector3, torch.Tensor)):
            KORNIA_CHECK_SHAPE(right, ["*", "3"])
            return so3 * right + t.data
        else:
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
    def t(self) -> Vector3 | torch.Tensor:
        """Return the underlying translation vector of shape :math:`(B,3)`."""
        return self._translation

    @property
    def rotation(self) -> So3:
        """Return the underlying `rotation(So3)`."""
        return self._rotation

    @property
    def translation(self) -> Vector3 | torch.Tensor:
        """Return the underlying translation vector of shape :math:`(B,3)`."""
        return self._translation

    @staticmethod
    def exp(v: torch.Tensor) -> Se3:
        """Convert elements of lie algebra to elements of lie group.

        Args:
            v: vector of shape :math:`(B, 6)`.

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
        theta = omega.norm(dim=-1, keepdim=True)[..., None]
        R = So3.exp(omega)

        eps = torch.finfo(v.dtype).eps * 1e3
        small_theta = theta <= eps

        # V matrix components
        # (1 - cos(theta)) / theta^2
        c1_large = (1 - torch.cos(theta)) / (theta**2)
        c1_small = 0.5 - (theta**2) / 24.0
        c1 = torch.where(small_theta, c1_small, c1_large)

        # (theta - sin(theta)) / theta^3
        c2_large = (theta - torch.sin(theta)) / (theta**3)
        c2_small = 1.0 / 6.0 - (theta**2) / 120.0
        c2 = torch.where(small_theta, c2_small, c2_large)

        V = torch.eye(3, device=v.device, dtype=v.dtype) + c1 * omega_hat + c2 * omega_hat_sq
        U = (V @ upsilon[..., None]).squeeze(-1)
        return Se3(R, U)

    def log(self) -> torch.Tensor:
        """Convert elements of lie group  to elements of lie algebra.

        Example:
            >>> from kornia.geometry.quaternion import Quaternion
            >>> q = Quaternion.identity()
            >>> Se3(q, torch.zeros(3)).log()
            tensor([0., 0., 0., 0., 0., 0.])

        """
        omega = self.r.log()
        theta = omega.norm(dim=-1)
        t = self.t.data
        omega_hat = So3.hat(omega)
        omega_hat_sq = omega_hat @ omega_hat
        eps = torch.finfo(omega.dtype).eps * 1e3
        small_theta = theta <= eps
        # (1 - theta * cos(theta/2) / (2 * sin(theta/2))) / theta^2
        c_large = (1 - theta * torch.cos(theta / 2) / (2 * torch.sin(theta / 2))) / theta.pow(2)
        c_small = 1.0 / 12.0 + (theta**2) / 720.0
        c = torch.where(small_theta, c_small, c_large)

        V_inv = torch.eye(3, device=omega.device, dtype=omega.dtype) - 0.5 * omega_hat + c[..., None, None] * omega_hat_sq
        t = (V_inv @ t[..., None]).squeeze(-1)
        return torch.cat((t, omega), -1)

    @staticmethod
    def hat(v: torch.Tensor) -> torch.Tensor:
        """Convert elements from vector space to lie algebra.

        Args:
            v: vector of shape :math:`(B, 6)`.

        Returns:
            matrix of shape :math:`(B, 4, 4)`.

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
            omega: 4x4-matrix representing lie algebra of shape :math:`(B,4,4)`.

        Returns:
            vector of shape :math:`(B,6)`.

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
        device: Optional[Union[str, torch.device, None]] = None,
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
            x: 0.0
            y: 0.0
            z: 0.0

        """
        t = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        if batch_size is not None:
            t = t.repeat(batch_size, 1)

        return cls(So3.identity(batch_size, device, dtype), Vector3(t))

    def matrix(self) -> torch.Tensor:
        """Return the matrix representation of shape :math:`(B, 4, 4)`.

        Example:
            >>> s = Se3(So3.identity(), torch.ones(3))
            >>> s.matrix()
            tensor([[1., 0., 0., 1.],
                    [0., 1., 0., 1.],
                    [0., 0., 1., 1.],
                    [0., 0., 0., 1.]])

        """
        t = self.t
        if isinstance(t, Vector3):
            t = t._data
        rt = torch.cat((self.r.matrix(), t[..., None]), -1)
        rt_4x4 = F.pad(rt, (0, 0, 0, 1))  # add last row torch.zeros
        rt_4x4[..., -1, -1] = 1.0
        return rt_4x4

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> Se3:
        """Create a Se3 group from a matrix.

        Args:
            matrix: torch.Tensor of shape :math:`(B, 4, 4)`.

        Example:
            >>> s = Se3.from_matrix(torch.eye(4))
            >>> s.r
            tensor([1., 0., 0., 0.])
            >>> s.t
            Parameter containing:
            tensor([0., 0., 0.], requires_grad=True)

        """
        KORNIA_CHECK_SHAPE(matrix, ["*", "4", "4"])
        r = So3.from_matrix(matrix[..., :3, :3])
        t = matrix[..., :3, -1]
        return cls(r, t)

    @classmethod
    def from_qxyz(cls, qxyz: torch.Tensor) -> Se3:
        """Create a Se3 group a quaternion and translation vector.

        Args:
            qxyz: torch.Tensor of shape :math:`(B, 7)`.

        Example:
            >>> qxyz = torch.tensor([1., 2., 3., 0., 0., 0., 1.])
            >>> s = Se3.from_qxyz(qxyz)
            >>> s.r
            tensor([1., 2., 3., 0.])
            >>> s.t
            x: 0.0
            y: 0.0
            z: 1.0

        """
        KORNIA_CHECK_SHAPE(qxyz, ["*", "7"])
        q, xyz = qxyz[..., :4], qxyz[..., 4:]
        return cls(So3.from_wxyz(q), Vector3(xyz))

    def inverse(self) -> Se3:
        """Return the inverse transformation.

        Example:
            >>> s = Se3(So3.identity(), torch.ones(3))
            >>> s_inv = s.inverse()
            >>> s_inv.r
            tensor([1., -0., -0., -0.])
            >>> s_inv.t
            Parameter containing:
            tensor([-1., -1., -1.], requires_grad=True)

        """
        r_inv = self.r.inverse()
        _t = -1 * self.t
        if isinstance(_t, int):
            raise TypeError("Unexpected integer from `-1 * translation`")

        return Se3(r_inv, r_inv * _t)

    @classmethod
    def random(
        cls,
        batch_size: Optional[int] = None,
        device: Optional[Union[str, torch.device, None]] = None,
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
        t = Vector3.random(shape, device, dtype)
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
        """Return the adjoint matrix of shape :math:`(B, 6, 6)`.

        Example:
            >>> s = Se3.identity()
            >>> s.adjoint()
            tensor([[1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 1.]])

        """
        R = self.so3.matrix()
        z = torch.zeros_like(R)
        row0 = torch.cat((R, So3.hat(self.t) @ R), -1)
        row1 = torch.cat((z, R), -1)
        return torch.cat((row0, row1), -2)

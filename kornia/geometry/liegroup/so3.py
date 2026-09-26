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
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE
from kornia.core.tensor_wrapper import _unwrap
from kornia.geometry.conversions import quaternion_to_axis_angle, vector_to_skew_symmetric_matrix
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.vector import Vector3


def _so3_small_angle_coefficients(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Evaluate the three angle coefficients of the SO(3) and SE(3) closed forms without cancellation.

    Returns :math:`(1 - \cos\theta) / \theta^2`, :math:`(\theta - \sin\theta) / \theta^3` and
    :math:`(1 - \tfrac{\theta}{2}\cot\tfrac{\theta}{2}) / \theta^2`, the coefficients of :math:`[\omega]_\times`
    and :math:`[\omega]_\times^2` in the SO(3) Jacobians, in the SE(3) :math:`V` matrix and in its inverse.

    Each closed form is a 0/0 at :math:`\theta = 0` and keeps only about :math:`\epsilon / \theta^2` of its
    relative accuracy near it: all three evaluate to exactly 0 in float32 for :math:`\theta \le 10^{-4}` and in
    float64 for :math:`\theta \le 10^{-8}`. Below ``theta_small`` the Taylor series through :math:`\theta^{10}`
    is used instead. At the switch point the series is accurate to a fraction of an ulp and the closed forms to a
    few hundred ulps, so the two branches agree there.

    Args:
        theta: rotation angles of any shape, non-negative.

    Returns:
        the three coefficients, each with the shape of ``theta``.
    """
    theta_small = 0.2 if theta.dtype == torch.float64 else 0.5
    small = theta < theta_small
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    t2 = theta * theta
    a_series = 0.5 + t2 * (-1 / 24 + t2 * (1 / 720 + t2 * (-1 / 40320 + t2 * (1 / 3628800 - t2 / 479001600))))
    b_series = 1 / 6 + t2 * (-1 / 120 + t2 * (1 / 5040 + t2 * (-1 / 362880 + t2 * (1 / 39916800 - t2 / 6227020800))))
    c_series = 1 / 12 + t2 * (
        1 / 720 + t2 * (1 / 30240 + t2 * (1 / 1209600 + t2 * (1 / 47900160 + t2 * (691 / 1307674368000))))
    )
    safe_sq = safe_theta * safe_theta
    half = 0.5 * safe_theta
    a = torch.where(small, a_series, (1 - torch.cos(safe_theta)) / safe_sq)
    b = torch.where(small, b_series, (safe_theta - torch.sin(safe_theta)) / (safe_sq * safe_theta))
    c = torch.where(small, c_series, (1 - half * torch.cos(half) / torch.sin(half)) / safe_sq)
    return a, b, c


class So3(nn.Module):
    r"""Base class to represent the So3 group.

    The SO(3) is the group of all rotations about the origin of three-dimensional Euclidean space
    :math:`R^3` under the operation of composition.
    See more: https://en.wikipedia.org/wiki/3D_rotation_group

    We internally represent the rotation by a unit quaternion.

    Example:
        >>> q = Quaternion.identity()
        >>> s = So3(q)
        >>> s.q
        tensor([1., 0., 0., 0.])

    """

    def __init__(self, q: Quaternion) -> None:
        """Construct the base class.

        Internally represented by a unit quaternion `q`.

        Args:
            q: Quaternion with the shape of :math:`(B, 4)`.

        Example:
            >>> data = torch.ones((2, 4))
            >>> q = Quaternion(data)
            >>> So3(q)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 1., 1.]])

        """
        super().__init__()
        KORNIA_CHECK_TYPE(q, Quaternion)
        self._q = q

    def __repr__(self) -> str:
        return f"{self.q}"

    def __getitem__(self, idx: int | slice) -> So3:
        return So3(self._q[idx])

    def __mul__(self, right: So3) -> So3:
        """Compose two So3 transformations.

        Args:
            right: the other So3 transformation.

        Return:
            The resulting So3 transformation.

        """
        # https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py#L98
        if isinstance(right, So3):
            return So3(self.q * right.q)
        if isinstance(right, (torch.Tensor, Vector3)):
            _right_data = _unwrap(right)
            KORNIA_CHECK_SHAPE(_right_data, ["*", "3"])
            w = torch.zeros(*right.shape[:-1], 1, device=right.device, dtype=right.dtype)
            quat = Quaternion(torch.cat((w, _right_data), -1))
            out = (self.q * quat * self.q.conj()).vec
            return Vector3(out) if isinstance(right, Vector3) else out
        raise TypeError(f"Not So3 or torch.Tensor type. Got: {type(right)}")

    @property
    def q(self) -> Quaternion:
        """Return the underlying data with shape :math:`(B,4)`."""
        return self._q

    @staticmethod
    def exp(v: torch.Tensor) -> So3:
        """Convert elements of lie algebra to elements of lie group.

        See more: https://vision.in.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

        Args:
            v: vector of shape :math:`(B,3)`.

        Example:
            >>> v = torch.zeros((2, 3))
            >>> s = So3.exp(v)
            >>> s
            tensor([[1., 0., 0., 0.],
                    [1., 0., 0., 0.]])

        """
        KORNIA_CHECK_SHAPE(v, ["*", "3"])
        theta = v.norm(dim=-1, keepdim=True)
        w = torch.cos(0.5 * theta)
        # sin(theta / 2) / theta is a 0/0 at theta = 0 (the identity, and the standard initialisation
        # for pose optimisation), and torch.where differentiates the branch it does not select, so
        # the closed form is evaluated on a substituted 1.0 there and its series through theta**10 is
        # used below 0.5 rad, where the truncation error is under 1e-17 for every dtype. The old
        # bound, finfo(dtype).eps * 1e3 with a two-term series, was 7.8 rad in bfloat16, so every
        # bfloat16 exp used the two terms and theta = 3 gave a quaternion of norm 0.94 (kornia#4928).
        small = theta < 0.5
        safe_theta = torch.where(small, torch.ones_like(theta), theta)
        t2 = theta * theta
        b_series = 0.5 + t2 * (-1 / 48 + t2 * (1 / 3840 + t2 * (-1 / 645120 + t2 * (1 / 185794560 - t2 / 81749606400))))
        b = torch.where(small, b_series, torch.sin(0.5 * safe_theta) / safe_theta)
        return So3(Quaternion(torch.cat((w, b * v), dim=-1)))

    def log(self) -> torch.Tensor:
        """Convert elements of lie group  to elements of lie algebra.

        Example:
            >>> So3.identity(batch_size=2).log()
            tensor([[0., 0., 0.],
                    [0., 0., 0.]])

        """
        # quaternion_to_axis_angle is the principal logarithm and it measures the angle with atan2: q and -q, the
        # same rotation, give the same vector with |theta| <= pi (#4925), a rotation below 1e-4 rad in float32 keeps
        # its digits instead of collapsing to 0 through 2 * acos(real) (#4897), and the identity keeps the finite
        # gradient 2 * vec / real of #4404.
        # NOTE: this differs from https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py#L33
        return quaternion_to_axis_angle(_unwrap(self.q.data))

    @staticmethod
    def hat(v: Vector3 | torch.Tensor) -> torch.Tensor:
        """Convert elements from vector space to lie algebra. Returns matrix of shape :math:`(B,3,3)`.

        Args:
            v: Vector3 or torch.Tensor of shape :math:`(B,3)`.

        Example:
            >>> v = torch.ones((1,3))
            >>> m = So3.hat(v)
            >>> m
            tensor([[[ 0., -1.,  1.],
                     [ 1.,  0., -1.],
                     [-1.,  1.,  0.]]])

        """
        KORNIA_CHECK_SHAPE(v, ["*", "3"])
        if isinstance(v, torch.Tensor):
            # TODO: Figure out why mypy think `v` can be a Vector3 which didn't allow ellipsis on index
            a, b, c = v[..., 0], v[..., 1], v[..., 2]  # type: ignore[index]
        else:
            a, b, c = v.x, v.y, v.z
        z = torch.zeros_like(a)
        row0 = torch.stack((z, -c, b), -1)
        row1 = torch.stack((c, z, -a), -1)
        row2 = torch.stack((-b, a, z), -1)
        return torch.stack((row0, row1, row2), -2)

    @staticmethod
    def vee(omega: torch.Tensor) -> torch.Tensor:
        r"""Convert elements from lie algebra to vector space. Returns vector of shape :math:`(B,3)`.

        .. math::
            omega = \begin{bmatrix} 0 & -c & b \\
            c & 0 & -a \\
            -b & a & 0\end{bmatrix}

        Args:
            omega: 3x3-matrix representing lie algebra.

        Example:
            >>> v = torch.ones((1,3))
            >>> omega = So3.hat(v)
            >>> So3.vee(omega)
            tensor([[1., 1., 1.]])

        """
        KORNIA_CHECK_SHAPE(omega, ["*", "3", "3"])
        a, b, c = omega[..., 2, 1], omega[..., 0, 2], omega[..., 1, 0]
        return torch.stack((a, b, c), -1)

    def matrix(self) -> torch.Tensor:
        r"""Convert the quaternion to a rotation matrix of shape :math:`(B,3,3)`.

        The matrix is of the form:

        .. math::
            \begin{bmatrix} 1-2y^2-2z^2 & 2xy-2zw & 2xy+2yw \\
            2xy+2zw & 1-2x^2-2z^2 & 2yz-2xw \\
            2xz-2yw & 2yz+2xw & 1-2x^2-2y^2\end{bmatrix}

        Example:
            >>> s = So3.identity()
            >>> m = s.matrix()
            >>> m
            tensor([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]])

        """
        w = self.q.w[..., None]
        x, y, z = self.q.x[..., None], self.q.y[..., None], self.q.z[..., None]
        q0 = 1 - 2 * y**2 - 2 * z**2
        q1 = 2 * x * y - 2 * z * w
        q2 = 2 * x * z + 2 * y * w
        row0 = torch.cat((q0, q1, q2), -1)
        q0 = 2 * x * y + 2 * z * w
        q1 = 1 - 2 * x**2 - 2 * z**2
        q2 = 2 * y * z - 2 * x * w
        row1 = torch.cat((q0, q1, q2), -1)
        q0 = 2 * x * z - 2 * y * w
        q1 = 2 * y * z + 2 * x * w
        q2 = 1 - 2 * x**2 - 2 * y**2
        row2 = torch.cat((q0, q1, q2), -1)
        return torch.stack((row0, row1, row2), -2)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> So3:
        """Create So3 from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B,3,3)`.

        Example:
            >>> m = torch.eye(3)
            >>> s = So3.from_matrix(m)
            >>> s
            tensor([1., 0., 0., 0.])

        """
        return cls(Quaternion.from_matrix(matrix))

    @classmethod
    def from_wxyz(cls, wxyz: torch.Tensor) -> So3:
        """Create So3 from a torch.Tensor representing a quaternion.

        Args:
            wxyz: the quaternion to convert of shape :math:`(B,4)`.

        Example:
            >>> q = torch.tensor([1., 0., 0., 0.])
            >>> s = So3.from_wxyz(q)
            >>> s
            tensor([1., 0., 0., 0.])

        """
        KORNIA_CHECK_SHAPE(wxyz, ["*", "4"])
        return cls(Quaternion(wxyz))

    @classmethod
    def identity(
        cls,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> So3:
        """Create a So3 group representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> s = So3.identity()
            >>> s
            tensor([1., 0., 0., 0.])

            >>> s = So3.identity(batch_size=2)
            >>> s
            tensor([[1., 0., 0., 0.],
                    [1., 0., 0., 0.]])

        """
        return cls(Quaternion.identity(batch_size, device, dtype))

    def inverse(self) -> So3:
        """Return the inverse transformation.

        Example:
            >>> s = So3.identity()
            >>> s.inverse()
            tensor([1., -0., -0., -0.])

        """
        return So3(self.q.conj())

    @classmethod
    def random(
        cls,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> So3:
        """Create a So3 group representing a random rotation.

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> s = So3.random()
            >>> s = So3.random(batch_size=3)

        """
        return cls(Quaternion.random(batch_size, device, dtype))

    @classmethod
    def rot_x(cls, x: torch.Tensor) -> So3:
        """Construct a x-axis rotation.

        Args:
            x: the x-axis rotation angle.

        """
        zs = torch.zeros_like(x)
        return cls.exp(torch.stack((x, zs, zs), -1))

    @classmethod
    def rot_y(cls, y: torch.Tensor) -> So3:
        """Construct a z-axis rotation.

        Args:
            y: the y-axis rotation angle.

        """
        zs = torch.zeros_like(y)
        return cls.exp(torch.stack((zs, y, zs), -1))

    @classmethod
    def rot_z(cls, z: torch.Tensor) -> So3:
        """Construct a z-axis rotation.

        Args:
            z: the z-axis rotation angle.

        """
        zs = torch.zeros_like(z)
        return cls.exp(torch.stack((zs, zs, z), -1))

    def adjoint(self) -> torch.Tensor:
        """Return the adjoint matrix of shape :math:`(B, 3, 3)`.

        Example:
            >>> s = So3.identity()
            >>> s.adjoint()
            tensor([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]])

        """
        return self.matrix()

    @staticmethod
    def right_jacobian(vec: torch.Tensor) -> torch.Tensor:
        """Compute the right Jacobian of So3.

        Args:
            vec: the input point of shape :math:`(B, 3)`.

        Example:
            >>> vec = torch.tensor([1., 2., 3.])
            >>> So3.right_jacobian(vec)
            tensor([[-0.0687,  0.5556, -0.0141],
                    [-0.2267,  0.1779,  0.6236],
                    [ 0.5074,  0.3629,  0.5890]])

        """
        KORNIA_CHECK_SHAPE(vec, ["*", "3"])
        R_skew = vector_to_skew_symmetric_matrix(vec)
        theta = vec.norm(dim=-1, keepdim=True)[..., None]
        a, b, _ = _so3_small_angle_coefficients(theta)
        I = torch.eye(3, device=vec.device, dtype=vec.dtype)  # noqa: E741
        return I - a * R_skew + b * (R_skew @ R_skew)

    @staticmethod
    def Jr(vec: torch.Tensor) -> torch.Tensor:
        """Alias for right jacobian.

        Args:
            vec: the input point of shape :math:`(B, 3)`.

        """
        return So3.right_jacobian(vec)

    @staticmethod
    def left_jacobian(vec: torch.Tensor) -> torch.Tensor:
        """Compute the left Jacobian of So3.

        Args:
            vec: the input point of shape :math:`(B, 3)`.

        Example:
            >>> vec = torch.tensor([1., 2., 3.])
            >>> So3.left_jacobian(vec)
            tensor([[-0.0687, -0.2267,  0.5074],
                    [ 0.5556,  0.1779,  0.3629],
                    [-0.0141,  0.6236,  0.5890]])

        """
        KORNIA_CHECK_SHAPE(vec, ["*", "3"])
        R_skew = vector_to_skew_symmetric_matrix(vec)
        theta = vec.norm(dim=-1, keepdim=True)[..., None]
        a, b, _ = _so3_small_angle_coefficients(theta)
        I = torch.eye(3, device=vec.device, dtype=vec.dtype)  # noqa: E741
        return I + a * R_skew + b * (R_skew @ R_skew)

    @staticmethod
    def Jl(vec: torch.Tensor) -> torch.Tensor:
        """Alias for left jacobian.

        Args:
            vec: the input point of shape :math:`(B, 3)`.

        """
        return So3.left_jacobian(vec)

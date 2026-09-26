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

# kornia.geometry.se2 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se2.py
from __future__ import annotations

from typing import Optional, Union, overload

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_DEVICES, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE
from kornia.core.tensor_wrapper import _unwrap
from kornia.core.utils import register_module_state
from kornia.geometry.liegroup.so2 import So2
from kornia.geometry.liegroup.so3 import _so3_small_angle_coefficients
from kornia.geometry.vector import Vector2


def _check_se2_r_t_shape(r: So2, t: torch.Tensor) -> None:
    z_shape = r.z.shape
    if ((len(z_shape) == 1) and (len(t.shape) == 2)) or ((len(z_shape) == 0) and len(t.shape) == 1):
        # check_se2_t_shape
        is_batch_shape = KORNIA_CHECK_SHAPE(t, ["B", "2"], raises=False)
        is_single_shape = KORNIA_CHECK_SHAPE(t, ["2"], raises=False)
        if not (is_batch_shape or is_single_shape):
            raise ValueError(f"Invalid translation shape, we expect [B, 2], or [2] Got: {t.shape}")
    else:
        raise ValueError(
            f"Invalid input, both the inputs should be either batched or unbatched. Got: {r.z.shape} and {t.shape}"
        )


class Se2(nn.Module):
    r"""Base class to represent the Se2 group.

    The SE(2) is the group of rigid body transformations about the origin of two-dimensional Euclidean
    space :math:`R^2` under the operation of composition.

    Convention:
        - Composition and point action follow :class:`~kornia.geometry.liegroup.Se3`; ``matrix()`` is the 3x3
          :math:`[[R, t], [0, 1]]`. The rotation is an :class:`~kornia.geometry.liegroup.So2`, whose storage and
          direction conventions apply.
        - The tangent vector is :math:`(v_x, v_y, \theta)`, angle last: ``exp`` rotates by :math:`\theta` and
          translates by :math:`V(\theta) (v_x, v_y)`, and ``log`` returns :math:`\theta` in :math:`[-\pi, \pi]`.
          ``adjoint()`` is :math:`[[R, (t_y, -t_x)^\top], [0, 1]]`.
        - ``from_matrix`` ignores the bottom row and rejects a rotation block that is not of the form
          :math:`[[a, -b], [b, a]]`, but not a scaled one.
        - Known defects: ``hat`` and ``vee`` put the translation in the bottom row and the angle in a symmetric block
          (`#4929 <https://github.com/kornia/kornia/issues/4929>`_); ``random`` takes the non-unit rotation of
          ``So2.random`` (`#4930 <https://github.com/kornia/kornia/issues/4930>`_); for ``identity`` and ``random``,
          ``t`` and ``g * points`` are a ``Vector2`` instead of a tensor
          (`#4931 <https://github.com/kornia/kornia/issues/4931>`_) and the translation is missing from
          ``state_dict`` (`#4923 <https://github.com/kornia/kornia/issues/4923>`_).

    Example:
        >>> so2 = So2.identity(1)
        >>> t = torch.ones((1, 2))
        >>> se2 = Se2(so2, t)
        >>> se2
        rotation: Parameter containing:
        tensor([1.+0.j], requires_grad=True)
        translation: Parameter containing:
        tensor([[1., 1.]], requires_grad=True)

    """

    def __init__(self, rotation: So2, translation: Vector2 | torch.Tensor) -> None:
        """Construct the base class.

        Internally represented by a torch.complex number `z` and a translation 2-vector.

        Args:
            rotation: So2 group encompassing a rotation.
            translation: translation vector with the shape of :math:`(B, 2)`.

        Example:
            >>> so2 = So2.identity(1)
            >>> t = torch.ones((1, 2))
            >>> se2 = Se2(so2, t)
            >>> se2
            rotation: Parameter containing:
            tensor([1.+0.j], requires_grad=True)
            translation: Parameter containing:
            tensor([[1., 1.]], requires_grad=True)

        """
        super().__init__()
        KORNIA_CHECK_TYPE(rotation, So2)
        if not isinstance(translation, (Vector2, torch.Tensor)):
            raise TypeError(f"translation type is {type(translation)}")
        self._translation: Vector2 | torch.Tensor
        self._rotation: So2 = rotation
        if isinstance(translation, torch.Tensor):
            _check_se2_r_t_shape(rotation, translation)  # TODO remove
            register_module_state(self, "_translation", translation)
        else:
            self._translation = translation

    def __repr__(self) -> str:
        return f"rotation: {self.r}\ntranslation: {self.t}"

    def __getitem__(self, idx: int | slice) -> Se2:
        return Se2(self._rotation[idx], self._translation[idx])

    def _mul_se2(self, right: Se2) -> Se2:
        so2 = self.so2
        t = self.t
        _r = so2 * right.so2
        _t = t + so2 * right.t
        return Se2(_r, _t)

    @overload
    def __mul__(self, right: Se2) -> Se2: ...

    @overload
    def __mul__(self, right: torch.Tensor) -> torch.Tensor: ...

    def __mul__(self, right: Se2 | torch.Tensor) -> Se2 | torch.Tensor:
        """Compose two Se2 transformations, or transform points.

        Args:
            right: the other Se2 transformation, or points of shape :math:`(B, 2)` or :math:`(2,)`.

        Return:
            The resulting Se2 transformation, or the transformed points.

        """
        so2 = self.so2
        t = self.t
        if isinstance(right, Se2):
            KORNIA_CHECK_TYPE(right, Se2)
            return self._mul_se2(right)
        if isinstance(right, (Vector2, torch.Tensor)):
            # _check_se2_r_t_shape(so2, risght)
            return so2 * right + t
        raise TypeError(f"Unsupported type: {type(right)}")

    @property
    def so2(self) -> So2:
        """Return the underlying `rotation(So2)`."""
        return self._rotation

    @property
    def r(self) -> So2:
        """Return the underlying `rotation(So2)`."""
        return self._rotation

    @property
    def t(self) -> Vector2 | torch.Tensor:
        """Return the underlying translation vector of shape :math:`(B,2)`."""
        return self._translation

    @property
    def rotation(self) -> So2:
        """Return the underlying `rotation(So2)`."""
        return self._rotation

    @property
    def translation(self) -> Vector2 | torch.Tensor:
        """Return the underlying translation vector of shape :math:`(B,2)`."""
        return self._translation

    @staticmethod
    def exp(v: torch.Tensor) -> Se2:
        """Convert elements of lie algebra to elements of lie group.

        Args:
            v: vector of shape :math:`(B, 3)`.

        Example:
            >>> v = torch.ones((1, 3))
            >>> s = Se2.exp(v)
            >>> s.r
            Parameter containing:
            tensor([0.5403+0.8415j], requires_grad=True)
            >>> s.t
            Parameter containing:
            tensor([[0.3818, 1.3012]], requires_grad=True)

        """
        # check_v_shape
        is_batch = KORNIA_CHECK_SHAPE(v, ["B", "3"], raises=False)
        is_single = KORNIA_CHECK_SHAPE(v, ["3"], raises=False)
        if not (is_batch or is_single):
            raise ValueError(f"Invalid input shape, we expect [B, 3], [3] Got: {v.shape}")
        theta = v[..., 2]
        so2 = So2.exp(theta)
        # V = [[a, -b], [b, a]] with a = sin(theta) / theta and b = (1 - cos(theta)) / theta. Both are
        # 0/0 at theta = 0, 1 - cos(theta) cancels just above it, and so does the autograd derivative of
        # sin(theta) / theta, so below 0.5 rad write them through the cancellation-free So3 coefficients:
        # a = 1 - theta^2 (theta - sin(theta)) / theta^3 and
        # b = theta (1 - cos(theta)) / theta^2 (kornia#4924), evaluated at |theta| because both are even.
        # Above it take sin(theta) / theta and 2 sin(theta / 2)^2 / theta directly: 1 - theta^2 (...)
        # cancels where sin(theta) / theta is small, and theta^3 overflows float16 above 40.3 rad. Each
        # branch sees a substituted angle where it is not selected, since torch.where differentiates both.
        small = theta.abs() < 0.5
        theta_s = torch.where(small, theta, torch.zeros_like(theta))
        theta_l = torch.where(small, torch.ones_like(theta), theta)
        coef_a, coef_b, _ = _so3_small_angle_coefficients(theta_s.abs())
        a = torch.where(small, 1.0 - theta_s * theta_s * coef_b, torch.sin(theta_l) / theta_l)
        b = torch.where(small, theta_s * coef_a, 2.0 * torch.sin(0.5 * theta_l) ** 2 / theta_l)
        x = v[..., 0]
        y = v[..., 1]
        t = torch.stack((a * x - b * y, b * x + a * y), -1)
        return Se2(so2, t)

    def log(self) -> torch.Tensor:
        """Convert elements of lie group  to elements of lie algebra.

        Example:
            >>> v = torch.ones((1, 3))
            >>> s = Se2.exp(v).log()
            >>> s.shape
            torch.Size([1, 3])
            >>> torch.allclose(s, v)
            True
            >>> s.requires_grad
            True

        """
        theta = self.so2.log()
        half_theta = 0.5 * theta
        # V^-1 = [[a, theta / 2], [-theta / 2, a]] with a = (theta / 2) cot(theta / 2), a 0/0 at
        # theta = 0 that cancels just above it: a = 1 - theta^2 (1 - (theta / 2) cot(theta / 2)) / theta^2
        # through the cancellation-free So3 coefficient (kornia#4924).
        _, _, coef_c = _so3_small_angle_coefficients(theta.abs())
        a = 1.0 - theta * theta * coef_c
        row0 = torch.stack((a, half_theta), -1)
        row1 = torch.stack((-half_theta, a), -1)
        V_inv = torch.stack((row0, row1), -2)
        upsilon = V_inv @ _unwrap(self.t)[..., None]
        return torch.stack((upsilon[..., 0, 0], upsilon[..., 1, 0], theta), -1)

    @staticmethod
    def hat(v: torch.Tensor) -> torch.Tensor:
        """Convert a tangent vector to the matrix that :meth:`vee` inverts. Returns matrix of shape :math:`(B, 3, 3)`.

        The matrix is not the se(2) generator (`#4929 <https://github.com/kornia/kornia/issues/4929>`_).

        Args:
            v: vector of shape :math:`(B, 3)`.

        Example:
            >>> v = torch.tensor([1.0, 2.0, 0.5])
            >>> Se2.hat(v)
            tensor([[0.0000, 0.5000, 0.0000],
                    [0.5000, 0.0000, 0.0000],
                    [1.0000, 2.0000, 0.0000]])

        """
        # check_v_shape
        is_batch = KORNIA_CHECK_SHAPE(v, ["B", "3"], raises=False)
        is_single = KORNIA_CHECK_SHAPE(v, ["3"], raises=False)
        if not (is_batch or is_single):
            raise ValueError(f"Invalid input shape, we expect [B, 3], [3] Got: {v.shape}")
        upsilon = torch.stack((v[..., 0], v[..., 1]), -1)
        theta = v[..., 2]
        col0 = torch.cat((So2.hat(theta), upsilon.unsqueeze(-2)), -2)
        return F.pad(col0, (0, 1))

    @staticmethod
    def vee(omega: torch.Tensor) -> torch.Tensor:
        """Read the tangent vector back from a :meth:`hat` matrix.

        It reads kornia's layout, not the se(2) generator (`#4929 <https://github.com/kornia/kornia/issues/4929>`_).

        Args:
            omega: 3x3-matrix built by :meth:`hat`, of shape :math:`(B, 3, 3)`.

        Returns:
            vector of shape :math:`(B, 3)`.

        Example:
            >>> v = torch.ones(3)
            >>> omega_hat = Se2.hat(v)
            >>> Se2.vee(omega_hat)
            tensor([1., 1., 1.])

        """
        # check_se2_omega_shape
        is_batch = KORNIA_CHECK_SHAPE(omega, ["B", "3", "3"], raises=False)
        is_single = KORNIA_CHECK_SHAPE(omega, ["3", "3"], raises=False)
        if not (is_batch or is_single):
            raise ValueError(f"Invalid input size, we expect [B, 3, 3] or [3, 3]. Got: {omega.shape}")
        upsilon = omega[..., 2, :2]
        theta = So2.vee(omega[..., :2, :2])
        return torch.cat((upsilon, theta[..., None]), -1)

    @classmethod
    def identity(
        cls,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> Se2:
        """Create a Se2 group representing an identity rotation and zero translation.

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> s = Se2.identity(1)
            >>> s.r
            Parameter containing:
            tensor([1.+0.j], requires_grad=True)
            >>> s.t
            x: tensor([0.])
            y: tensor([0.])

        """
        t: torch.Tensor = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        if batch_size is not None:
            KORNIA_CHECK(batch_size >= 1, msg="batch_size must be positive")
            t = t.repeat(batch_size, 1)
        return cls(So2.identity(batch_size, device, dtype), Vector2(t))

    def matrix(self) -> torch.Tensor:
        """Return the matrix representation of shape :math:`(B, 3, 3)`.

        Example:
            >>> s = Se2(So2.identity(1), torch.ones(1, 2))
            >>> s.matrix()
            tensor([[[1., -0., 1.],
                     [0., 1., 1.],
                     [0., 0., 1.]]], grad_fn=<CopySlices>)

        """
        rt = torch.cat((self.r.matrix(), _unwrap(self.t)[..., None]), -1)
        rt_3x3 = F.pad(rt, (0, 0, 0, 1))  # add last row torch.zeros
        rt_3x3[..., -1, -1] = 1.0
        return rt_3x3

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> Se2:
        """Create an Se2 group from a matrix.

        Args:
            matrix: torch.Tensor of shape :math:`(B, 3, 3)`.

        Example:
            >>> s = Se2.from_matrix(torch.eye(3).repeat(2, 1, 1))
            >>> s.r
            Parameter containing:
            tensor([1.+0.j, 1.+0.j], requires_grad=True)
            >>> s.t
            Parameter containing:
            tensor([[0., 0.],
                    [0., 0.]], requires_grad=True)

        """
        KORNIA_CHECK_SHAPE(matrix, ["*", "3", "3"])
        r = So2.from_matrix(matrix[..., :2, :2])
        t = matrix[..., :2, -1]
        return cls(r, t)

    def inverse(self) -> Se2:
        """Return the inverse transformation.

        Example:
            >>> s = Se2(So2.identity(1), torch.ones(1,2))
            >>> s_inv = s.inverse()
            >>> s_inv.r
            tensor([1.+0.j], grad_fn=<MulBackward0>)
            >>> s_inv.t
            tensor([[-1., -1.]], grad_fn=<StackBackward0>)

        """
        r_inv: So2 = self.r.inverse()
        _t = -1 * self.t
        if isinstance(_t, int):
            raise TypeError("Unexpected integer from `-1 * translation`")

        return Se2(r_inv, r_inv * _t)

    @classmethod
    def random(
        cls,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> Se2:
        """Create a Se2 group from ``So2.random`` and a translation drawn from :math:`U[0, 1)`.

        The rotation is not a uniform unit rotation (`#4930 <https://github.com/kornia/kornia/issues/4930>`_).

        Args:
            batch_size: the batch size of the underlying data.
            device: device to place the result on.
            dtype: dtype of the result.

        Example:
            >>> s = Se2.random()
            >>> s = Se2.random(batch_size=3)

        """
        r = So2.random(batch_size, device, dtype)
        shape: tuple[int, ...]
        if batch_size is None:
            shape = (2,)
        else:
            KORNIA_CHECK(batch_size >= 1, msg="batch_size must be positive")
            shape = (batch_size, 2)
        return cls(r, Vector2(torch.rand(shape, device=device, dtype=dtype)))

    @classmethod
    def trans(cls, x: torch.Tensor, y: torch.Tensor) -> Se2:
        """Construct a translation only Se2 instance.

        Args:
            x: the x-axis translation.
            y: the y-axis translation.

        """
        KORNIA_CHECK(x.shape == y.shape)
        KORNIA_CHECK_SAME_DEVICES([x, y])
        batch_size = x.shape[0] if len(x.shape) > 0 else None
        rotation = So2.identity(batch_size, x.device, x.dtype)
        return cls(rotation, torch.stack((x, y), -1))

    @classmethod
    def trans_x(cls, x: torch.Tensor) -> Se2:
        """Construct a x-axis translation.

        Args:
            x: the x-axis translation.

        """
        zs = torch.zeros_like(x)
        return cls.trans(x, zs)

    @classmethod
    def trans_y(cls, y: torch.Tensor) -> Se2:
        """Construct a y-axis translation.

        Args:
            y: the y-axis translation.

        """
        zs = torch.zeros_like(y)
        return cls.trans(zs, y)

    def adjoint(self) -> torch.Tensor:
        """Return the adjoint matrix of shape :math:`(B, 3, 3)`.

        Example:
            >>> s = Se2.identity()
            >>> s.adjoint()
            tensor([[1., -0., 0.],
                    [0., 1., -0.],
                    [0., 0., 1.]], grad_fn=<CopySlices>)

        """
        rt = self.matrix()
        t = _unwrap(self.t)
        rt[..., 0:2, 2] = torch.stack((t[..., 1], -t[..., 0]), -1)
        return rt

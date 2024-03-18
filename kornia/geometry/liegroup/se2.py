# kornia.geometry.se2 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se2.py
from __future__ import annotations

from typing import Optional, overload

from kornia.core import (
    Device,
    Dtype,
    Module,
    Parameter,
    Tensor,
    concatenate,
    pad,
    rand,
    stack,
    tensor,
    where,
    zeros_like,
)
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_DEVICES, KORNIA_CHECK_TYPE
from kornia.geometry.liegroup._utils import check_se2_omega_shape, check_se2_t_shape, check_v_shape
from kornia.geometry.liegroup.so2 import So2
from kornia.geometry.vector import Vector2


def _check_se2_r_t_shape(r: So2, t: Tensor) -> None:
    z_shape = r.z.shape
    if ((len(z_shape) == 1) and (len(t.shape) == 2)) or ((len(z_shape) == 0) and len(t.shape) == 1):
        check_se2_t_shape(t)
    else:
        raise ValueError(
            f"Invalid input, both the inputs should be either batched or unbatched. Got: {r.z.shape} and {t.shape}"
        )


class Se2(Module):
    r"""Base class to represent the Se2 group.

    The SE(2) is the group of rigid body transformations about the origin of two-dimensional Euclidean
    space :math:`R^2` under the operation of composition.
    See more:

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

    def __init__(self, rotation: So2, translation: Vector2 | Tensor) -> None:
        """Constructor for the base class.

        Internally represented by a complex number `z` and a translation 2-vector.

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
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        # KORNIA_CHECK_TYPE(translation, (Vector3, Tensor))
        if not isinstance(translation, (Vector2, Tensor)):
            raise TypeError(f"translation type is {type(translation)}")
        self._translation: Vector2 | Parameter
        self._rotation: So2 = rotation
        if isinstance(translation, Tensor):
            _check_se2_r_t_shape(rotation, translation)  # TODO remove
            self._translation = Parameter(translation)
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
    def __mul__(self, right: Tensor) -> Tensor: ...

    def __mul__(self, right: Se2 | Tensor) -> Se2 | Tensor:
        """Compose two Se2 transformations.

        Args:
            right: the other Se2 transformation.

        Return:
            The resulting Se2 transformation.
        """
        so2 = self.so2
        t = self.t
        if isinstance(right, Se2):
            KORNIA_CHECK_TYPE(right, Se2)
            return self._mul_se2(right)
        elif isinstance(right, (Vector2, Tensor)):
            # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
            # _check_se2_r_t_shape(so2, risght)
            return so2 * right + t
        else:
            raise TypeError(f"Unsupported type: {type(right)}")

    @property
    def so2(self) -> So2:
        """Return the underlying rotation(So2)."""
        return self._rotation

    @property
    def r(self) -> So2:
        """Return the underlying rotation(So2)."""
        return self._rotation

    @property
    def t(self) -> Vector2 | Parameter:
        """Return the underlying translation vector of shape :math:`(B,2)`."""
        return self._translation

    @property
    def rotation(self) -> So2:
        """Return the underlying rotation(So2)."""
        return self._rotation

    @property
    def translation(self) -> Vector2 | Parameter:
        """Return the underlying translation vector of shape :math:`(B,2)`."""
        return self._translation

    @staticmethod
    def exp(v: Tensor) -> Se2:
        """Converts elements of lie algebra to elements of lie group.

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
        check_v_shape(v)
        theta = v[..., 2]
        so2 = So2.exp(theta)
        z = tensor(0.0, device=v.device, dtype=v.dtype)
        theta_nonzeros = theta != 0.0
        a = where(theta_nonzeros, so2.z.imag / theta, z)
        b = where(theta_nonzeros, (1.0 - so2.z.real) / theta, z)
        x = v[..., 0]
        y = v[..., 1]
        t = stack((a * x - b * y, b * x + a * y), -1)
        return Se2(so2, t)

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> v = torch.ones((1, 3))
            >>> s = Se2.exp(v).log()
            >>> s
            tensor([[1.0000, 1.0000, 1.0000]], grad_fn=<StackBackward0>)
        """
        theta = self.so2.log()
        half_theta = 0.5 * theta
        denom = self.so2.z.real - 1
        a = where(
            denom != 0, -(half_theta * self.so2.z.imag) / denom, tensor(0.0, device=theta.device, dtype=theta.dtype)
        )
        row0 = stack((a, half_theta), -1)
        row1 = stack((-half_theta, a), -1)
        V_inv = stack((row0, row1), -2)
        upsilon = V_inv @ self.t.data[..., None]
        return stack((upsilon[..., 0, 0], upsilon[..., 1, 0], theta), -1)

    @staticmethod
    def hat(v: Tensor) -> Tensor:
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B, 3, 3)`.

        Args:
            v: vector of shape:math:`(B, 3)`.

        Example:
            >>> theta = torch.tensor(3.1415/2)
            >>> So2.hat(theta)
            tensor([[0.0000, 1.5707],
                    [1.5707, 0.0000]])
        """
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_v_shape(v)
        upsilon = stack((v[..., 0], v[..., 1]), -1)
        theta = v[..., 2]
        col0 = concatenate((So2.hat(theta), upsilon.unsqueeze(-2)), -2)
        return pad(col0, (0, 1))

    @staticmethod
    def vee(omega: Tensor) -> Tensor:
        """Converts elements from lie algebra to vector space.

        Args:
            omega: 3x3-matrix representing lie algebra of shape :math:`(B, 3, 3)`.

        Returns:
            vector of shape :math:`(B, 3)`.

        Example:
            >>> v = torch.ones(3)
            >>> omega_hat = Se2.hat(v)
            >>> Se2.vee(omega_hat)
            tensor([1., 1., 1.])
        """
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_se2_omega_shape(omega)
        upsilon = omega[..., 2, :2]
        theta = So2.vee(omega[..., :2, :2])
        return concatenate((upsilon, theta[..., None]), -1)

    @classmethod
    def identity(cls, batch_size: Optional[int] = None, device: Optional[Device] = None, dtype: Dtype = None) -> Se2:
        """Create a Se2 group representing an identity rotation and zero translation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = Se2.identity(1)
            >>> s.r
            Parameter containing:
            tensor([1.+0.j], requires_grad=True)
            >>> s.t
            x: tensor([0.])
            y: tensor([0.])
        """
        t: Tensor = tensor([0.0, 0.0], device=device, dtype=dtype)
        if batch_size is not None:
            KORNIA_CHECK(batch_size >= 1, msg="batch_size must be positive")
            t = t.repeat(batch_size, 1)
        return cls(So2.identity(batch_size, device, dtype), Vector2(t))

    def matrix(self) -> Tensor:
        """Returns the matrix representation of shape :math:`(B, 3, 3)`.

        Example:
            >>> s = Se2(So2.identity(1), torch.ones(1, 2))
            >>> s.matrix()
            tensor([[[1., -0., 1.],
                     [0., 1., 1.],
                     [0., 0., 1.]]], grad_fn=<CopySlices>)
        """
        rt = concatenate((self.r.matrix(), self.t.data[..., None]), -1)
        rt_3x3 = pad(rt, (0, 0, 0, 1))  # add last row zeros
        rt_3x3[..., -1, -1] = 1.0
        return rt_3x3

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> Se2:
        """Create an Se2 group from a matrix.

        Args:
            matrix: tensor of shape :math:`(B, 3, 3)`.

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
        # KORNIA_CHECK_SHAPE(matrix, ["B", "3", "3"])  # FIXME: resolve shape bugs. @edgarriba
        r = So2.from_matrix(matrix[..., :2, :2])
        t = matrix[..., :2, -1]
        return cls(r, t)

    def inverse(self) -> Se2:
        """Returns the inverse transformation.

        Example:
            >>> s = Se2(So2.identity(1), torch.ones(1,2))
            >>> s_inv = s.inverse()
            >>> s_inv.r
            Parameter containing:
            tensor([1.+0.j], requires_grad=True)
            >>> s_inv.t
            Parameter containing:
            tensor([[-1., -1.]], requires_grad=True)
        """
        r_inv: So2 = self.r.inverse()
        _t = -1 * self.t
        if isinstance(_t, int):
            raise TypeError("Unexpected integer from `-1 * translation`")

        return Se2(r_inv, r_inv * _t)

    @classmethod
    def random(cls, batch_size: Optional[int] = None, device: Optional[Device] = None, dtype: Dtype = None) -> Se2:
        """Create a Se2 group representing a random transformation.

        Args:
            batch_size: the batch size of the underlying data.

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
        return cls(r, Vector2(rand(shape, device=device, dtype=dtype)))

    @classmethod
    def trans(cls, x: Tensor, y: Tensor) -> Se2:
        """Construct a translation only Se2 instance.

        Args:
            x: the x-axis translation.
            y: the y-axis translation.
        """
        KORNIA_CHECK(x.shape == y.shape)
        KORNIA_CHECK_SAME_DEVICES([x, y])
        batch_size = x.shape[0] if len(x.shape) > 0 else None
        rotation = So2.identity(batch_size, x.device, x.dtype)
        return cls(rotation, stack((x, y), -1))

    @classmethod
    def trans_x(cls, x: Tensor) -> Se2:
        """Construct a x-axis translation.

        Args:
            x: the x-axis translation.
        """
        zs = zeros_like(x)
        return cls.trans(x, zs)

    @classmethod
    def trans_y(cls, y: Tensor) -> Se2:
        """Construct a y-axis translation.

        Args:
            y: the y-axis translation.
        """
        zs = zeros_like(y)
        return cls.trans(zs, y)

    def adjoint(self) -> Tensor:
        """Returns the adjoint matrix of shape :math:`(B, 3, 3)`.

        Example:
            >>> s = Se2.identity()
            >>> s.adjoint()
            tensor([[1., -0., 0.],
                    [0., 1., -0.],
                    [0., 0., 1.]], grad_fn=<CopySlices>)
        """
        rt = self.matrix()
        rt[..., 0:2, 2] = stack((self.t.data[..., 1], -self.t.data[..., 0]), -1)
        return rt

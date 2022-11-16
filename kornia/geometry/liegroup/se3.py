# kornia.geometry.so3 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se3.py
from typing import Optional

from kornia.core import Module, Parameter, Tensor, concatenate, eye, pad, stack, tensor, where, zeros_like
from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.linalg import batched_dot_product
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_SAME_DEVICES, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE


class Se3(Module):
    r"""Base class to represent the Se3 group.

    The SE(3) is the group of rigid body transformations about the origin of three-dimensional Euclidean
    space :math:`R^3` under the operation of composition.
    See more: https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf

    Example:
        >>> from kornia.geometry.quaternion import Quaternion
        >>> q = Quaternion.identity()
        >>> s = Se3(So3(q), torch.ones(3))
        >>> s.r
        Parameter containing:
        tensor([1., 0., 0., 0.], requires_grad=True)
        >>> s.t
        Parameter containing:
        tensor([1., 1., 1.], requires_grad=True)
    """

    def __init__(self, r: So3, t: Tensor) -> None:
        """Constructor for the base class.

        Internally represented by a unit quaternion `q` and a translation 3-vector.

        Args:
            r: So3 group encompassing a rotation.
            t: translation vector with the shape of :math:`(B, 3)`.

        Example:
            >>> from kornia.geometry.quaternion import Quaternion
            >>> q = Quaternion.identity(batch_size=1)
            >>> s = Se3(So3(q), torch.ones((1, 3)))
            >>> s.r
            Parameter containing:
            tensor([[1., 0., 0., 0.]], requires_grad=True)
            >>> s.t
            Parameter containing:
            tensor([[1., 1., 1.]], requires_grad=True)
        """
        super().__init__()
        KORNIA_CHECK_TYPE(r, So3)
        KORNIA_CHECK_SHAPE(t, ["B", "3"])
        self._r = r
        self._t = Parameter(t)

    def __repr__(self) -> str:
        return f"rotation: {self.r}\ntranslation: {self.t}"

    def __getitem__(self, idx) -> 'Se3':
        return Se3(self._r[idx], self._t[idx][None])

    def __mul__(self, right: "Se3") -> "Se3":
        """Compose two Se3 transformations.

        Args:
            right: the other Se3 transformation.

        Return:
            The resulting Se3 transformation.
        """
        if isinstance(right, Se3):
            KORNIA_CHECK_TYPE(right, Se3)
            # https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se3.py#L97
            r = self.so3 * right.so3
            t = self.t + self.so3 * right.t
            return Se3(r, t)
        elif isinstance(right, Tensor):
            KORNIA_CHECK_TYPE(right, Tensor)
            KORNIA_CHECK_SHAPE(right, ["B", "N"])
            return self.so3 * right + self.t
        else:
            raise TypeError(f"Unsupported type: {type(right)}")

    @property
    def so3(self) -> So3:
        """Return the underlying rotation(So3)."""
        return self._r

    @property
    def r(self) -> So3:
        """Return the underlying rotation(So3)."""
        return self._r

    @property
    def t(self) -> Tensor:
        """Return the underlying translation vector of shape :math:`(B,3)`."""
        return self._t

    @staticmethod
    def exp(v) -> 'Se3':
        """Converts elements of lie algebra to elements of lie group.

        Args:
            v: vector of shape :math:`(B, 6)`.

        Example:
            >>> v = torch.zeros((1, 6))
            >>> s = Se3.exp(v)
            >>> s.r
            Parameter containing:
            tensor([[1., 0., 0., 0.]], requires_grad=True)
            >>> s.t
            Parameter containing:
            tensor([[0., 0., 0.]], requires_grad=True)
        """
        KORNIA_CHECK_SHAPE(v, ["B", "6"])
        upsilon = v[..., :3]
        omega = v[..., 3:]
        omega_hat = So3.hat(omega)
        omega_hat_sq = omega_hat @ omega_hat
        theta = batched_dot_product(omega, omega).sqrt()
        R = So3.exp(omega)
        V = (
            eye(3, device=v.device, dtype=v.dtype)
            + ((1 - theta.cos()) / (theta**2))[..., None, None] * omega_hat
            + ((theta - theta.sin()) / (theta**3))[..., None, None] * omega_hat_sq
        )
        U = where(theta[..., None] != 0.0, (upsilon[..., None, :] * V).sum(-1), upsilon)
        return Se3(R, U)

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> from kornia.geometry.quaternion import Quaternion
            >>> q = Quaternion.identity()
            >>> Se3(So3(q), torch.zeros(3)).log()
            tensor([0., 0., 0., 0., 0., 0.], grad_fn=<CatBackward0>)
        """
        omega = self.r.log()
        theta = batched_dot_product(omega, omega).sqrt()
        omega_hat = So3.hat(omega)
        omega_hat_sq = omega_hat @ omega_hat
        V_inv = (
            eye(3, device=omega.device, dtype=omega.dtype)
            - 0.5 * omega_hat
            + ((1 - theta * (theta / 2).cos() / (2 * (theta / 2).sin())) / theta.pow(2))[..., None, None] * omega_hat_sq
        )
        t = where(theta[..., None] != 0.0, (self.t[..., None, :] * V_inv).sum(-1), self.t)
        return concatenate((t, omega), -1)

    @staticmethod
    def hat(v) -> Tensor:
        """Converts elements from vector space to lie algebra.

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
        KORNIA_CHECK_SHAPE(v, ["B", "6"])
        upsilon, omega = v[..., :3], v[..., 3:]
        rt = concatenate((So3.hat(omega), upsilon[..., None]), -1)
        return pad(rt, (0, 0, 0, 1))  # add zeros bottom

    @staticmethod
    def vee(omega) -> Tensor:
        """Converts elements from lie algebra to vector space.

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
        KORNIA_CHECK_SHAPE(omega, ["B", "4", "4"])
        head = omega[..., :3, -1]
        tail = So3.vee(omega[..., :3, :3])
        return concatenate((head, tail), -1)

    @classmethod
    def identity(cls, batch_size: Optional[int] = None, device=None, dtype=None) -> 'Se3':
        """Create a Se3 group representing an identity rotation and zero translation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = Se3.identity()
            >>> s.r
            Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
            >>> s.t
            Parameter containing:
            tensor([0., 0., 0.], requires_grad=True)
        """
        t: Tensor = tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        if batch_size is not None:
            t = t.repeat(batch_size, 1)

        return cls(So3.identity(batch_size, device, dtype), t)

    def matrix(self) -> Tensor:
        """Returns the matrix representation of shape :math:`(B, 4, 4)`.

        Example:
            >>> s = Se3(So3.identity(), torch.ones(3))
            >>> s.matrix()
            tensor([[1., 0., 0., 1.],
                    [0., 1., 0., 1.],
                    [0., 0., 1., 1.],
                    [0., 0., 0., 1.]], grad_fn=<CopySlices>)
        """
        rt = concatenate((self.r.matrix(), self.t[..., None]), -1)
        rt_4x4 = pad(rt, (0, 0, 0, 1))  # add last row zeros
        rt_4x4[..., -1, -1] = 1.0
        return rt_4x4

    def inverse(self) -> 'Se3':
        """Returns the inverse transformation.

        Example:
            >>> s = Se3(So3.identity(), torch.ones(3))
            >>> s_inv = s.inverse()
            >>> s_inv.r
            Parameter containing:
            tensor([1., -0., -0., -0.], requires_grad=True)
            >>> s_inv.t
            Parameter containing:
            tensor([-1., -1., -1.], requires_grad=True)
        """
        r_inv = self.r.inverse()
        return Se3(r_inv, r_inv * (-1 * self.t))

    @classmethod
    def rot_x(cls, x: Tensor) -> "Se3":
        """Construct a x-axis rotation.

        Args:
            x: the x-axis rotation angle.
        """
        zs = zeros_like(x)
        return cls(So3.rot_x(x), stack((zs, zs, zs), -1))

    @classmethod
    def rot_y(cls, y: Tensor) -> "Se3":
        """Construct a y-axis rotation.

        Args:
            y: the y-axis rotation angle.
        """
        zs = zeros_like(y)
        return cls(So3.rot_y(y), stack((zs, zs, zs), -1))

    @classmethod
    def rot_z(cls, z: Tensor) -> "Se3":
        """Construct a z-axis rotation.

        Args:
            z: the z-axis rotation angle.
        """
        zs = zeros_like(z)
        return cls(So3.rot_z(z), stack((zs, zs, zs), -1))

    @classmethod
    def trans(cls, x: Tensor, y: Tensor, z: Tensor) -> "Se3":
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
        return cls(rotation, stack((x, y, z), -1))

    @classmethod
    def trans_x(cls, x: Tensor) -> "Se3":
        """Construct a x-axis translation.

        Args:
            x: the x-axis translation.
        """
        zs = zeros_like(x)
        return cls.trans(x, zs, zs)

    @classmethod
    def trans_y(cls, y: Tensor) -> "Se3":
        """Construct a y-axis translation.

        Args:
            y: the y-axis translation.
        """
        zs = zeros_like(y)
        return cls.trans(zs, y, zs)

    @classmethod
    def trans_z(cls, z: Tensor) -> "Se3":
        """Construct a z-axis translation.

        Args:
            z: the z-axis translation.
        """
        zs = zeros_like(z)
        return cls.trans(zs, zs, z)

    def adjoint(self) -> Tensor:
        """Returns the adjoint matrix of shape :math:`(B, 6, 6)`.

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
        z = zeros_like(R)
        row0 = concatenate((R, So3.hat(self.t) @ R), -1)
        row1 = concatenate((z, R), -1)
        return concatenate((row0, row1), -2)

# kornia.geometry.so3 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se3.py
from kornia.core import Tensor, as_tensor, concatenate, eye, pad, where
from kornia.geometry.liegroup._utils import squared_norm
from kornia.geometry.liegroup.so3 import So3
from kornia.testing import KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE


class Se3:
    r"""Base class to represent the Se3 group.

    The SE(3) is the group of rigid body transformations about the origin of three-dimensional Euclidean
    space :math:`R^3` under the operation of composition.
    See more: https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf

    Example:
        >>> from kornia.geometry.quaternion import Quaternion
        >>> q = Quaternion.identity(batch_size=1)
        >>> s = Se3(So3(q), torch.ones((1, 3)))
        >>> s.r
        real: tensor([[1.]], grad_fn=<SliceBackward0>)
        vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
        >>> s.t
        tensor([[1., 1., 1.]])
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
            >>> s = Se3(So3(q), torch.ones((1,3)))
            >>> s.r
            real: tensor([[1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
            >>> s.t
            tensor([[1., 1., 1.]])
        """
        KORNIA_CHECK_TYPE(r, So3)
        KORNIA_CHECK_SHAPE(t, ["B", "3"])
        self._r = r
        self._t = t

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
        KORNIA_CHECK_TYPE(right, Se3)
        # https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se3.py#L97
        r = self.r * right.r
        t = self.t + self.r * right.t
        return Se3(r, t)

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
            real: tensor([[1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
            >>> s.t
            tensor([[0., 0., 0.]])
        """
        KORNIA_CHECK_SHAPE(v, ["B", "6"])
        t = v[..., 0:3]
        omega = v[..., 3:]
        omega_hat = So3.hat(omega)
        theta = squared_norm(omega).sqrt()
        R = So3.exp(omega)
        V = (
            eye(3, device=v.device, dtype=v.dtype)
            + ((1 - theta.cos()) / (theta**2))[..., None] * omega_hat
            + ((theta - theta.sin()) / (theta**3))[..., None] * (omega_hat @ omega_hat)
        )
        U = where(theta != 0.0, (t.reshape(-1, 1, 3) * V).sum(-1), t)
        return Se3(R, U)

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> from kornia.geometry.quaternion import Quaternion
            >>> q = Quaternion.identity(batch_size=1)
            >>> Se3(So3(q), torch.zeros((1,3))).log()
            tensor([[0., 0., 0., 0., 0., 0.]], grad_fn=<CatBackward0>)
        """
        omega = self.r.log()
        theta = squared_norm(omega).sqrt()
        omega_hat = So3.hat(omega)
        V_inv = (
            eye(3, device=omega.device, dtype=omega.dtype)
            - 0.5 * omega_hat
            + ((1 - theta * (theta / 2).cos() / (2 * (theta / 2).sin())) / theta.pow(2))[..., None]
            * (omega_hat @ omega_hat)
        )
        t = where(theta != 0.0, (self.t.reshape(-1, 1, 3) * V_inv).sum(-1), self.t)
        return concatenate((t, omega), -1)

    @staticmethod
    def hat(v) -> Tensor:
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B,4,4)`.

        Args:
            v: vector of shape :math:`(B,6)`.

        Example:
            >>> v = torch.ones((1,6))
            >>> m = Se3.hat(v)
            >>> m
            tensor([[[ 0.,  0., -1.,  1.],
                     [ 0.,  1.,  0., -1.],
                     [ 0., -1.,  1.,  0.],
                     [ 0.,  1.,  1.,  1.]]])
        """
        KORNIA_CHECK_SHAPE(v, ["B", "6"])
        t, omega = v[..., None, :3], v[..., 3:]
        rt = concatenate((So3.hat(omega), t), 1)
        return pad(rt, (1, 0, 0, 0))  # add zeros column

    @staticmethod
    def vee(omega) -> Tensor:
        """Converts elements from lie algebra to vector space.

        Args:
            omega: 4x4-matrix representing lie algebra of shape :math:`(B,4,4)`.

        Returns:
            vector of shape :math:`(B,6)`.

        Example:
            >>> v = torch.ones((1,6))
            >>> omega_hat = Se3.hat(v)
            >>> Se3.vee(omega_hat)
            tensor([[1., 1., 1., 1., 1., 1.]])
        """
        KORNIA_CHECK_SHAPE(omega, ["B", "4", "4"])
        t = omega[..., 3, 1:]
        v = So3.vee(omega[..., 0:3, 1:])
        return concatenate((t, v), 1)

    @classmethod
    def identity(cls, batch_size: int, device=None, dtype=None) -> 'Se3':
        """Create a Se3 group representing an identity rotation and zero translation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = Se3.identity(batch_size=1)
            >>> s.r
            real: tensor([[1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
            >>> s.t
            tensor([[0., 0., 0.]])
        """
        t: Tensor = as_tensor(batch_size * [[0.0, 0.0, 0.0]], device=device, dtype=dtype)
        return cls(So3.identity(batch_size, device, dtype), t)

    def matrix(self) -> Tensor:
        """Returns the matrix representation of shape :math:`(B, 4, 4)`.

        Example:
            >>> s = Se3(So3.identity(batch_size=1), torch.ones((1,3)))
            >>> s.matrix()
            tensor([[[1., 0., 0., 1.],
                     [0., 1., 0., 1.],
                     [0., 0., 1., 1.],
                     [0., 0., 0., 1.]]], grad_fn=<CopySlices>)
        """
        rt = concatenate((self.r.matrix(), self.t.reshape(-1, 3, 1)), 2)
        rt_4x4 = pad(rt, (0, 0, 0, 1))  # add last row zeros
        rt_4x4[..., -1, -1] = 1.0
        return rt_4x4

    def inverse(self) -> 'Se3':
        """Returns the inverse transformation.

        Example:
            >>> s = Se3(So3.identity(batch_size=1), torch.ones((1,3)))
            >>> s_inv = s.inverse()
            >>> s_inv.r
            real: tensor([[1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[-0., -0., -0.]], grad_fn=<SliceBackward0>)
            >>> s_inv.t
            tensor([[-1., -1., -1.]], grad_fn=<SliceBackward0>)
        """
        r_inv = self.r.inverse()
        return Se3(r_inv, r_inv * (-1 * self.t))

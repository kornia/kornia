# kornia.geometry.so2 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se2.py
from typing import Optional

from kornia.core import Module, Parameter, Tensor, concatenate, pad, stack, tensor
from kornia.geometry.liegroup._utils import check_se2_t_shape, check_v_shape
from kornia.geometry.liegroup.so2 import So2
from kornia.testing import KORNIA_CHECK_TYPE


class Se2(Module):
    r"""Base class to represent the Se2 group.

    The SE(2) is the group of rigid body transformations about the origin of two-dimensional Euclidean
    space :math:`R^2` under the operation of composition.
    See more:

    Example:
        >>> so2 = So2.identity()
        >>> t = torch.ones((1, 2))
        >>> se2 = Se2(so2, t)
        >>> se2
        rotation: (1+0j)
        translation: Parameter containing:
        tensor([[1., 1.]], requires_grad=True)
    """

    def __init__(self, r: So2, t: Tensor) -> None:
        """Constructor for the base class.

        Internally represented by a complex number `z` and a translation 2-vector.

        Args:
            r: So2 group encompassing a rotation.
            t: translation vector with the shape of :math:`(B, 2)`.

        Example:
            >>> so2 = So2.identity()
            >>> t = torch.ones((1, 2))
            >>> se2 = Se2(so2, t)
            >>> se2
            rotation: (1+0j)
            translation: Parameter containing:
            tensor([[1., 1.]], requires_grad=True)
        """
        super().__init__()
        KORNIA_CHECK_TYPE(r, So2)
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_se2_t_shape(t)
        self._r = r
        self._t = Parameter(t)

    def __repr__(self) -> str:
        return f"rotation: {self.r}\ntranslation: {self.t}"

    def __getitem__(self, idx) -> 'Se2':
        return Se2(self._r[idx], self._t[idx][None])

    def __mul__(self, right: "Se2") -> "Se2":
        """Compose two Se2 transformations.

        Args:
            right: the other Se2 transformation.

        Return:
            The resulting Se2 transformation.
        """
        if isinstance(right, Se2):
            KORNIA_CHECK_TYPE(right, Se2)
            r = self.so2 * right.so2
            t = self.t + self.so2 * right.t
            return Se2(r, t)
        elif isinstance(right, Tensor):
            KORNIA_CHECK_TYPE(right, Tensor)
            # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
            check_se2_t_shape(right)
            return self.so2 * right + self.t
        else:
            raise TypeError(f"Unsupported type: {type(right)}")

    @property
    def so2(self) -> So2:
        """Return the underlying rotation(So2)."""
        return self._r

    @property
    def r(self) -> So2:
        """Return the underlying rotation(So2)."""
        return self._r

    @property
    def t(self) -> Tensor:
        """Return the underlying translation vector of shape :math:`(B,3)`."""
        return self._t

    @staticmethod
    def exp(v) -> 'Se2':
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
        # TODO when theta is 0
        check_v_shape(v)
        theta = v[..., 2]
        so2 = So2.exp(theta)
        a = so2.z.imag / theta
        b = (1.0 - so2.z.real) / theta
        t = stack((a * v[..., 0] - b * v[..., 1], b * v[..., 0] + a * v[..., 1]), -1)
        return Se2(so2, t)

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> v = torch.ones((1, 3))
            >>> s = Se2.exp(v).log()
            tensor([[1.0000, 1.0000, 1.0000]], grad_fn=<StackBackward0>)
        """
        theta = self.so2.log()
        half_theta = 0.5 * theta
        a = -(half_theta * self.so2.z.imag) / (self.so2.z.real - 1)
        row0 = stack((a, half_theta), -1)
        row1 = stack((-half_theta, a), -1)
        V_inv = stack((row0, row1), -2)
        upsilon = V_inv @ self.t[..., None]
        return stack((upsilon[..., 0, 0], upsilon[..., 1, 0], theta), -1)

    @staticmethod
    def hat(v):
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B, 2, 2)`.

        Args:
            theta: angle in radians of shape :math:`(B)`.

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
        col0 = concatenate((So2.hat(theta), upsilon.reshape(-1, 1, 2)), -2)
        return pad(col0, (0, 1))

    @classmethod
    def identity(cls, batch_size: Optional[int] = None, device=None, dtype=None) -> 'Se2':
        """Create a Se2 group representing an identity rotation and zero translation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = Se2.identity()
            >>> s.r
            (1+0j)
            >>> s.t
            Parameter containing:
            tensor([0., 0.], requires_grad=True)
        """
        t: Tensor = tensor([0.0, 0.0], device=device, dtype=dtype)
        if batch_size is not None:
            t = t.repeat(batch_size, 1)
        return cls(So2.identity(batch_size, device, dtype), t)

    def matrix(self) -> Tensor:
        """Returns the matrix representation of shape :math:`(B, 3, 3)`.

        Example:
            >>> s = Se2(So2.identity(), torch.ones(2))
            >>> s.matrix()
            tensor([[1., -0., 1.],
                    [0., 1., 1.],
                    [0., 0., 1.]], grad_fn=<CopySlices>)
        """
        rt = concatenate((self.r.matrix(), self.t[..., None]), -1)
        rt_3x3 = pad(rt, (0, 0, 0, 1))  # add last row zeros
        rt_3x3[..., -1, -1] = 1.0
        return rt_3x3

    def inverse(self) -> 'Se2':
        """Returns the inverse transformation.

        Example:
            >>> s = Se2(So2.identity(), torch.ones(2))
            >>> s_inv = s.inverse()
            >>> s_inv.r
            (1+0j)
            >>> s_inv.t
            Parameter containing:
            tensor([-1., -1.], requires_grad=True)
        """
        r_inv = self.r.inverse()
        return Se2(r_inv, r_inv * (-1 * self.t))

# kornia.geometry.so2 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se2.py
from typing import Optional

from kornia.core import Module, Parameter, Tensor, concatenate, stack, eye, pad, tensor, where
from kornia.geometry.liegroup.so2 import So2
from kornia.geometry.linalg import batched_dot_product
from kornia.testing import KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE


class Se2(Module):
    r"""Base class to represent the Se2 group.

    The SE(2) is the group of rigid body transformations about the origin of two-dimensional Euclidean
    space :math:`R^2` under the operation of composition.
    See more: 

    TODO:
    Example:
        >>> real = torch.tensor([1.0])
        >>> imag = torch.tensor([2.0])
        >>> So2(torch.complex(real, imag))
        Parameter containing:
        tensor([[1.+2.j]], requires_grad=True)
    """

    def __init__(self, r: So2, t: Tensor) -> None:
        """Constructor for the base class.

        Internally represented by a complex number `z` and a translation 2-vector.

        Args:
            r: So2 group encompassing a rotation.
            t: translation vector with the shape of :math:`(B, 2)`.
        
        TODO
        Example:
            >>> from kornia.geometry.quaternion import Quaternion
            >>> q = Quaternion.identity(batch_size=1)
            >>> s = Se2(So2(q), torch.ones((1,3)))
            >>> s.r
            real: tensor([[1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
            >>> s.t
            tensor([[1., 1., 1.]])
        """
        super().__init__()
        KORNIA_CHECK_TYPE(r, So2)
        KORNIA_CHECK_SHAPE(t, ["B", "2"])
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
            # https://github.com/strasdat/Sophus/blob/master/sympy/sophus/se2.py#L97
            r = self.so2 * right.so2
            t = self.t + self.so2 * right.t
            return Se2(r, t)
        elif isinstance(right, Tensor):
            KORNIA_CHECK_TYPE(right, Tensor)
            KORNIA_CHECK_SHAPE(right, ["B", "N"])
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
            >>> v = torch.zeros((1, 3))
            >>> s = Se2.exp(v)
            >>> s.r
            Parameter containing:
            tensor([[1.+0.j]], requires_grad=True)
            >>> s.t
            Parameter containing:
            tensor([[nan, nan]], requires_grad=True)
        """
        KORNIA_CHECK_SHAPE(v, ["B", "3"])
        #TODO when theta is 0
        theta = v[..., 2]
        so2 = So2.exp(theta)
        aa = so2.z.imag / theta
        bb = (1.0 - so2.z.real) / theta
        t = stack((aa @ v[..., 0] - bb @ v[..., 1], bb @ v[..., 0] + aa @ v[..., 1]), -1)
        return Se2(so2, t)

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> from kornia.geometry.quaternion import Quaternion
            >>> q = Quaternion.identity()
            >>> Se2(So2(q), torch.zeros(3)).log()
            tensor([0., 0., 0., 0., 0., 0.], grad_fn=<CatBackward0>)
        """
        theta = self.so2.log()
        half_theta = 0.5 * theta
        aa = -(half_theta * self.so2.z.imag) / (self.so2.z.real - 1)
        row0 = concatenate((aa[..., None], half_theta[..., None]), -1)
        row1 = concatenate((-half_theta[..., None], aa[..., None]), -1)
        V_inv = concatenate((row0, row1), 1)
        upsilon = V_inv @ self.t[..., None]
        return concatenate((upsilon[:, 0], upsilon[:, 1], theta), -1)

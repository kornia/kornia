from kornia.core import Tensor, concatenate, zeros, eye, where
from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.liegroup._utils import squared_norm
from kornia.testing import KORNIA_CHECK_TYPE, KORNIA_CHECK_SHAPE

class Se3:
    r"""Base class to represent the Se3 group.

    The SE(3) is the group of rigid body transformations
    https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf

    Example:
        >>> q = Quaternion.identity(batch_size=1)
        >>> s = Se3(So3(q), torch.rand((1, 3)))
        >>> s
        rotation:real: tensor([[-0.4420]], grad_fn=<SliceBackward0>) 
        vec: tensor([[ 0.3347,  0.7869, -0.2711]], grad_fn=<SliceBackward0>)
        translation:tensor([[0.0631, 0.0164, 0.9543]])
    """

    def __init__(self, r: So3, t: Tensor) -> None:
        """Constructor for the base class.

        Args:
            rot: So3 group encompassing a rotation.
            t: translation vector with the shape of :math:`(B, 3)`.

        Example:
            >>> q = Quaternion.random(batch_size=1)
            >>> s = Se3(So3(q), torch.rand((1,3)))
            >>> s
            rotation:real: tensor([[-0.7368]], grad_fn=<SliceBackward0>) 
            vec: tensor([[-0.5644,  0.0390,  0.3702]], grad_fn=<SliceBackward0>)
            translation:tensor([[0.6353, 0.5230, 0.5110]])
        """
        KORNIA_CHECK_TYPE(r, So3)
        KORNIA_CHECK_SHAPE(t, ["B", "3"])
        self._r = r
        self._t = t

    def __repr__(self) -> str:
        return f"rotation:{self.r}\ntranslation:{self.t}"

    def __getitem__(self, idx) -> 'Se3':
        return Se3(self._r[idx], self._t[idx].unsqueeze(0))

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
            >>> v = torch.rand((1, 6))
            >>> s = Se3.exp(v)
            >>> s
            rotation:real: tensor([[0.9032]], grad_fn=<SliceBackward0>) 
            vec: tensor([[0.0506, 0.2116, 0.3700]], grad_fn=<SliceBackward0>)
            translation:tensor([[0.4024, 0.7548, 0.6127]])
        """
        import pdb
        tt = v[..., 0:3]
        omega = v[..., 3:]
        omega_hat = So3.hat(omega)
        theta = squared_norm(omega).sqrt()
        R = So3.exp(omega)
        V = (eye(3) + ((1 - theta.cos()) / (theta**2)).unsqueeze(-1) * omega_hat + ((theta - theta.sin()) / (theta**3)).unsqueeze(-1) * (omega_hat @ omega_hat))
        U = where(theta != 0.0, (tt.reshape(-1, 1, 3)*V).sum(-1), tt)
        return Se3(R, U)

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> q = Quaternion.random(batch_size=1)
            >>> Se3(So3(q), torch.rand((1,3))).log()
            tensor([[-0.3175,  1.0273,  1.1459,  1.5137,  3.5590, -1.2261]],
                   grad_fn=<CatBackward0>)
        """
        omega = self.r.log()
        theta = squared_norm(omega).sqrt()
        omega_hat = So3.hat(omega)
        V_inv = eye(3) - 0.5 * omega_hat + ((1 - theta * (theta / 2).cos() / (2 * (theta/2).sin())) / theta.pow(2)).unsqueeze(-1) * (omega_hat @ omega_hat)
        t = where(theta != 0.0, (self.t.reshape(-1, 1, 3) * V_inv).sum(-1), self.t)
        return concatenate((t, omega), -1)

    @staticmethod
    def hat(v) -> Tensor:
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B,4,4)`.

        Args:
            v: vector of shape :math:`(B,6)`.

        Example:
            >>> v = torch.rand((1,6))
            >>> m = Se3.hat(v)
            >>> m
            tensor([[[ 0.0000,  0.0000, -0.7649,  0.4374],
                     [ 0.0000,  0.7649,  0.0000, -0.1046],
                     [ 0.0000, -0.4374,  0.1046,  0.0000],
                     [ 0.0000,  0.4076,  0.6015,  0.6532]]])
        """
        t = v[..., 0:3].reshape(-1, 1, 3)
        omega = v[..., 3:]
        rt = concatenate((So3.hat(omega), t.reshape(-1, 1, 3)), 1)
        return concatenate((zeros(v.shape[0], 4, 1), rt), -1)

    @staticmethod
    def vee(omega) -> Tensor:
        """Converts elements from lie algebra to vector space. Returns vector of shape :math:`(B,6)`.

        Args:
            omega: 4x4-matrix representing lie algebra

        Example:
            >>> v = torch.rand((1,6))
            >>> omega = Se3.hat(v)
            >>> Se3.vee(omega)
            tensor([[0.5979, 0.5425, 0.2085, 0.5961, 0.2532, 0.5150]])
        """
        t = omega[..., 3, 1:4]
        v  = So3.vee(omega[..., 0:3, 1:4])
        return concatenate((t, v), 1)

    def matrix(self) -> Tensor:
        rt = concatenate((self.r.matrix(), self.t.reshape(-1, 1, 3)), 1)
        return concatenate((zeros(1, 4, 1), rt), -1)

    def inverse(self) -> 'Se3':
        R_inv = self.r.inverse()
        return Se3(R_inv, -1 * self.t)

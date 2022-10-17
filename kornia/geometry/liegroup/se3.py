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
        >>> s = Se3(So3(q), torch.ones((1, 3)))
        >>> s
        rotation: real: tensor([[1.]], grad_fn=<SliceBackward0>) 
        vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
        translation: tensor([[1., 1., 1.]])
    """

    def __init__(self, r: So3, t: Tensor) -> None:
        """Constructor for the base class.

        Args:
            rot: So3 group encompassing a rotation.
            t: translation vector with the shape of :math:`(B, 3)`.

        Example:
            >>> q = Quaternion.identity(batch_size=1)
            >>> s = Se3(So3(q), torch.ones((1,3)))
            >>> s.r
            rotation: real: tensor([[1.]], grad_fn=<SliceBackward0>) 
            vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
            translation: tensor([[1., 1., 1.]])
       """
        KORNIA_CHECK_TYPE(r, So3)
        KORNIA_CHECK_SHAPE(t, ["B", "3"])
        self._r = r
        self._t = t

    def __repr__(self) -> str:
        return f"rotation: {self.r}\ntranslation: {self.t}"

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
            >>> v = torch.zeros((1, 6))
            >>> s = Se3.exp(v)
            >>> s
            rotation: real: tensor([[1.]], grad_fn=<SliceBackward0>) 
            vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
            translation: tensor([[0., 0., 0.]])
        """
        t = v[..., 0:3]
        omega = v[..., 3:]
        omega_hat = So3.hat(omega)
        theta = squared_norm(omega).sqrt()
        R = So3.exp(omega)
        V = (eye(3) + ((1 - theta.cos()) / (theta**2)).unsqueeze(-1) * omega_hat + ((theta - theta.sin()) / (theta**3)).unsqueeze(-1) * (omega_hat @ omega_hat))
        U = where(theta != 0.0, (t.reshape(-1, 1, 3) * V).sum(-1), t)
        return Se3(R, U)

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> q = Quaternion.identity(batch_size=1)
            >>> Se3(So3(q), torch.zeros((1,3))).log()
            tensor([[0., 0., 0., 0., 0., 0.]], grad_fn=<CatBackward0>)
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
            >>> v = torch.ones((1,6))
            >>> m = Se3.hat(v)
            >>> m
            tensor([[[ 0.,  0., -1.,  1.],
                     [ 0.,  1.,  0., -1.],
                     [ 0., -1.,  1.,  0.],
                     [ 0.,  1.,  1.,  1.]]])
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
            >>> v = torch.ones((1,6))
            >>> omega_hat = Se3.hat(v)
            >>> Se3.vee(omega_hat)
            tensor([[1., 1., 1., 1., 1., 1.]])
        """
        t = omega[..., 3, 1:]
        v = So3.vee(omega[..., 0:3, 1:])
        return concatenate((t, v), 1)

    def matrix(self) -> Tensor:
        """Returns the matrix representation of shape :math:`(B, 4, 4)`.

        Example:
            >>> s = Se3(So3.identity(batch_size=1), torch.ones((1,3)))
            >>> s
            tensor([[[0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 1., 1., 1.]]], grad_fn=<CatBackward0>)
        """
        rt = concatenate((self.r.matrix(), self.t.reshape(-1, 1, 3)), 1)
        return concatenate((zeros(self.t.shape[0], 4, 1), rt), -1)

    def inverse(self) -> 'Se3':
        """Returns the inverse transformation.

        Example:
            >>> s = Se3(So3.identity(batch_size=1), torch.ones((1,3)))
            >>> s.inverse()
            rotation: real: tensor([[1.]], grad_fn=<SliceBackward0>) 
            vec: tensor([[-0., -0., -0.]], grad_fn=<SliceBackward0>)
            translation: tensor([[-1., -1., -1.]])
        """
        R_inv = self.r.inverse()
        return Se3(R_inv, -1 * self.t)

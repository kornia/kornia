from kornia.core import Tensor, concatenate, stack, where, zeros, zeros_like, eye
from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.liegroup._utils import squared_norm
from kornia.testing import KORNIA_CHECK_TYPE, KORNIA_CHECK_SHAPE

class Se3:
    r"""Base class to represent the Se3 group.

    The SE(3) is the group of rigid body transformations
    See more: http://lavalle.pl/planning/node147.html

    Example: To do

    """

    def __init__(self, rot: So3, t: Tensor) -> None:
        """Constructor for the base class.

        Args:
            rot: So3
            t: translation vector with the shape of :math:`(B, 3)`.

        Example:
            >>> q = Quaternion.identity(batch_size=1)
            >>> s = Se3(So3(q), torch.rand((1,3))
            >>> s
            real: tensor([[1.]], grad_fn=<SliceBackward0>)#todo
            vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
        """
        KORNIA_CHECK_TYPE(rot, So3)
        KORNIA_CHECK_SHAPE(t, ["B", "3"])
        self._rot = rot
        self._t = t

    def __repr__(self) -> str:
        return f"{self.rot.q}\n{self.t}"

    def __getitem__(self, idx) -> 'Se3':
        return Se3(self._rot[idx], self._t[idx].unsqueeze(0))

    @property
    def rot(self) -> So3:
        """Return the underlying rotation(So3)."""
        return self._rot

    @property
    def t(self) -> So3:
        """Return the underlying translation vector of shape :math:`(B,3)`."""
        return self._t

    @staticmethod
    def exp(vv) -> 'Se3':
        """Converts elements of lie algebra to elements of lie group.

        See more: https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf

        Args:
            v: vector of shape :math:`(B, 6)`.

        Example:todo
        """
        # import pdb
        # pdb.set_trace()
        tt = vv[..., 0:3]
        omega = vv[..., 3:]
        omega_hat = So3.hat(omega)
        theta = squared_norm(omega).sqrt()
        rot = So3.exp(omega)
        vvv = eye(3) + ((1 - theta.cos()) / theta.pow(2)) * omega_hat + ((theta - theta.sin()) / theta.pow(3)) * omega_hat.pow(2)
        return Se3(rot, (tt * vvv).sum(-1))

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> data = torch.rand((2, 4))
            >>> q = Quaternion(data)
            >>> So3(q).log()
            tensor([[2.3822, 0.2638, 0.1771],
                    [0.3699, 1.8639, 0.3685]], grad_fn=<MulBackward0>)
        """
        omega = self.rot.log()
        theta = squared_norm(omega).sqrt()
        omega_hat = So3.hat(omega)
        vv_inv = eye(3) - (omega_hat/ 2) + ((1 - (theta * (theta / 2).cos()) / 2 * (theta/2).sin()) / theta.pow(2)) * omega_hat.pow(2)
        return concatenate(((self.t * vv_inv).sum(-1), omega), -1)

    def hat(self, vv):
        tt = vv[..., 0:3].reshape(-1, 1, 3)
        omega = vv[..., 3:]
        rt = concatenate((So3.hat(omega), tt.reshape(-1, 1, 3)), 1)
        return concatenate((zeros(1, 4, 1), rt),-1)

    def vee(self, omega):
        tt = omega[..., 3, 1:4]
        vee  = So3.vee(omega[..., 1:4, 1:4])
        return concatenate((tt, vee), 1)

    # def identity():

    # def inverse():
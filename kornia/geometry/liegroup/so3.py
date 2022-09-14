from kornia.core import Tensor, concatenate, stack

from ._utils import squared_norm
from kornia.utils.misc import zeros_like
from kornia.geometry.quaternion import Quaternion

from kornia.testing import KORNIA_CHECK_TYPE

class So3:
    def __init__(self, q: Quaternion) -> None:
        KORNIA_CHECK_TYPE(q, Quaternion)
        self._q = q

    def __repr__(self) -> str:
        return f"{self.q}"

    @property
    def q(self):
        return self._q

    @classmethod
    def exp(cls, v) -> 'So3':
        theta_sq = squared_norm(v)
        theta = theta_sq.sqrt()
        return cls(Quaternion(concatenate(((0.5 * theta).cos(), (0.5 * theta).sin().div(theta).mul(v[:])),1)))

    def log(self):
        n = squared_norm(self.q.vec).sqrt()
        return 2 * (n / self.q.real).atan() / n * self.q.vec

    @staticmethod
    def hat(o):
        zeros = zeros_like(o)[..., 0]
        row0 = concatenate([zeros, -o[..., 2], o[..., 1]], -1)
        row1 = concatenate([o[..., 2], zeros, -o[..., 0]], -1)
        row2 = concatenate([-o[..., 1], o[..., 0], zeros], -1)
        return stack([row0, row1, row2], -2)

    def matrix(self):
        e1 = (1 - 2 * self.q.vec[..., 1] ** 2 - 2 * self.q.vec[..., 2] ** 2).reshape(1, 1,-1)
        e2 = (2 * self.q.vec[..., 0] * self.q.vec[..., 1] - 2 * self.q.vec[..., 2] * self.q[..., 3]).reshape(1,1,-1)
        e3 = (2 * self.q.vec[..., 0] * self.q.vec[..., 2] + 2 * self.q.vec[..., 1] * self.q[..., 3]).reshape(1,1,-1)
        col1 = concatenate([e1,e2,e3], 1)
        e1 = (2 * self.q.vec[..., 0] * self.q.vec[..., 1] + 2 * self.q.vec[..., 2] * self.q[..., 3]).reshape(1,1,-1)
        e2 = (1 - 2 * self.q.vec[..., 0] ** 2 - 2 * self.q.vec[..., 2] ** 2).reshape(1,1,-1)
        e3 = (2 * self.q.vec[..., 1] * self.q.vec[..., 2] - 2 * self.q.vec[..., 0] * self.q[..., 3]).reshape(1,1,-1)
        col2 = concatenate([e1,e2,e3], 1)
        e1 = (2 * self.q.vec[..., 0] * self.q.vec[..., 2] - 2 * self.q.vec[..., 1] * self.q[..., 3]).reshape(1,1,-1)
        e2 = (2 * self.q.vec[..., 1] * self.q.vec[..., 2] + 2 * self.q.vec[..., 0] * self.q[..., 3]).reshape(1,1,-1)
        e3 = (1 - 2 * self.q.vec[..., 0] ** 2 - 2 * self.q.vec[..., 1] ** 2).reshape(1,1,-1)
        col3 = concatenate([e1,e2,e3], 1)
        return concatenate([col1,col2,col3], 0)

    @classmethod
    def identity(cls):
        return cls(Quaternion.identity())

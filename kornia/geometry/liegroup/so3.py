from kornia.core import Tensor, concatenate, stack, zeros_like

from kornia.geometry.quaternion import Quaternion
from kornia.geometry.liegroup._utils import squared_norm

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
        ox, oy, oz = o[..., 0], o[..., 1], o[..., 2]
        zeros = zeros_like(o)[..., 0]
        row0 = concatenate([zeros, -oz, oy], -1)
        row1 = concatenate([oz, zeros, -ox], -1)
        row2 = concatenate([-oy, ox, zeros], -1)
        return stack([row0, row1, row2], -2)

    def matrix(self):
        w, x, y, z = self.q[..., 0], self.q.vec[..., 0], self.q.vec[..., 1], self.q.vec[..., 2]
        q1 = (1 - 2 * y ** 2 - 2 * z ** 2).reshape(1, 1,-1)
        q2 = (2 * x * y - 2 * z * w).reshape(1,1,-1)
        q3 = (2 * x * z + 2 * y * w).reshape(1,1,-1)
        col1 = concatenate([q1,q2,q3], 1)
        q1 = (2 * x * y + 2 * z * w).reshape(1,1,-1)
        q2 = (1 - 2 * x ** 2 - 2 * z ** 2).reshape(1,1,-1)
        q3 = (2 * y * z - 2 * x * w).reshape(1,1,-1)
        col2 = concatenate([q1,q2,q3], 1)
        q1 = (2 * x * z - 2 * y * w).reshape(1,1,-1)
        q2 = (2 * y * z + 2 * x * w).reshape(1,1,-1)
        q3 = (1 - 2 * x ** 2 - 2 * y ** 2).reshape(1,1,-1)
        col3 = concatenate([q1,q2,q3], 1)
        return concatenate([col1,col2,col3], 0)

    @classmethod
    def identity(cls):
        return cls(Quaternion.identity())

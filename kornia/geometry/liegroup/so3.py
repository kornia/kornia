from kornia.core import Tensor, concatenate, stack, zeros_like

from kornia.geometry.quaternion import Quaternion
from ._utils import squared_norm
from kornia.testing import KORNIA_CHECK_TYPE, KORNIA_CHECK


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
    def hat(oo):
        aa, bb, cc = oo[None, None, ..., 0], oo[None, None, ..., 1], oo[None, None, ..., 2]
        zeros = zeros_like(oo)[None, None, ..., 0]
        row0 = concatenate([zeros, -cc, bb], 1)
        row1 = concatenate([cc, zeros, -aa], 1)
        row2 = concatenate([-bb, aa, zeros], 1)
        return concatenate([row0, row1, row2], 0)

    @staticmethod
    def vee(omega):
        a, b, c = omega[2, 1, ...], omega[0, 2, ...], omega[1, 0, ...]
        return stack([a, b, c], 1)

    def matrix(self):
        w, x, y, z = self.q[None, None, ..., 0], self.q.vec[None, None, ..., 0], self.q.vec[None, None, ..., 1], self.q.vec[None, None, ..., 2]
        q0 = (1 - 2 * y ** 2 - 2 * z ** 2)
        q1 = (2 * x * y - 2 * z * w)
        q2 = (2 * x * z + 2 * y * w)
        row0 = concatenate([q0, q1, q2], 1)
        q0 = (2 * x * y + 2 * z * w)
        q1 = (1 - 2 * x ** 2 - 2 * z ** 2)
        q2 = (2 * y * z - 2 * x * w)
        row1 = concatenate([q0, q1, q2], 1)
        q0 = (2 * x * z - 2 * y * w)
        q1 = (2 * y * z + 2 * x * w)
        q2 = (1 - 2 * x ** 2 - 2 * y ** 2)
        row2 = concatenate([q0, q1, q2], 1)
        return concatenate([row0, row1, row2], 0)

    @classmethod
    def identity(cls, batch_size: int) -> 'So3':
        return cls(Quaternion.identity(batch_size))

    def inverse(self) -> 'So3':
        return So3(self.q.conj())

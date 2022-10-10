# kornia.geometry.so3 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py
from kornia.core import Tensor, concatenate, stack, zeros_like, zeros, where
from kornia.geometry.quaternion import Quaternion
from kornia.testing import KORNIA_CHECK_TYPE, KORNIA_CHECK


class So3:
    r"""Base class to represent the So3 group.

    The SO(3) is the group of all rotations about the origin of three-dimensional Euclidean space
    R^3 under the operation of composition.
    See more: https://en.wikipedia.org/wiki/3D_rotation_group

    We internally represent the rotation by a unit quaternion.

    Example:
        >>> q = Quaternion.identity(batch_size=1)
        >>> s = So3(q)
        >>> s.q
        real: tensor([[1.]], grad_fn=<SliceBackward0>) 
        vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
    """

    def __init__(self, q: Quaternion) -> None:
        """Constructor for the base class.

        Args:
            data: Quaternion with the sape of :math:`(B, 4)`.

        Example:
            >>> data = torch.rand((2, 4))
            >>> q = Quaternion(data)
            >>> So3(q)
            real: tensor([[0.2734],
                          [0.7782]], grad_fn=<SliceBackward0>) 
            vec: tensor([[0.2420, 0.2716, 0.6159],
                         [0.8727, 0.7592, 0.7212]], grad_fn=<SliceBackward0>)
        """
        KORNIA_CHECK_TYPE(q, Quaternion)
        self._q = q
        self.epsilon = 2.220446049250313e-16

    def __repr__(self) -> str:
        return f"{self.q}"

    def __getitem__(self, idx) -> 'So3':
        return So3(Quaternion(self._q[idx].reshape(1, -1))) #change once quaternion bug fix is merged

    @property
    def q(self) -> Quaternion:
        """Return the underlying data with shape :math:`(B,4)`."""
        return self._q

    def exp(self, v) -> 'So3':
        """Converts elements of lie algebra to elements of lie group.
 
        See more: https://vision.in.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

        Args:
            v: vector of shape :math:`(B,3)`.

        Example:
            >>> v = torch.rand((2,3))
            >>> s = So3.identity(batch_size=1).exp(v)
            >>> s
            real: tensor([[0.8891],
                    [0.9774]], grad_fn=<SliceBackward0>) 
            vec: tensor([[0.0893, 0.3285, 0.3060],
                    [0.0567, 0.1315, 0.1558]], grad_fn=<SliceBackward0>)
        """
        theta = self.squared_norm(v).sqrt()
        small_angles_indices = where(theta < self.epsilon)[0]
        large_angles_indices = where(theta > self.epsilon)[0]
        large_angles = theta[large_angles_indices]

        qtensor = zeros((v.shape[0],4))
        qtensor[small_angles_indices] = Tensor([1, 0, 0, 0]) #identity quaternion for small angles
       
        w = (0.5 * large_angles).cos()
        xyz = (0.5 * large_angles).sin().div(large_angles).mul(v[large_angles_indices])
        qtensor[large_angles_indices]  = concatenate((w, xyz), 1)
        return So3(Quaternion(qtensor))

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> data = torch.rand((2, 4))
            >>> q = Quaternion(data)
            >>> So3(q).log()
            tensor([[2.3822, 0.2638, 0.1771],
                    [0.3699, 1.8639, 0.3685]], grad_fn=<MulBackward0>)
        """
        theta = self.squared_norm(self.q.vec).sqrt()
        small_angles_indices = where(theta < self.epsilon)[0]
        large_angles_indices = where(theta > self.epsilon)[0]
        large_angles = theta[large_angles_indices]

        q_real = self.q.real
        q_vec =  self.q.vec
        omega_t = zeros((self.q.shape[0], 3))
        
        o1 = 2 / q_real[small_angles_indices]
        o2 = theta[small_angles_indices].pow(2)/ q_real[small_angles_indices].pow(3)
        omega_t[small_angles_indices] = (o1 - (2 * o2) / 3) * q_vec[small_angles_indices]
        o1 = (q_real[large_angles_indices].pow(2) + theta[large_angles_indices].pow(2)).sqrt()
        omega_t[large_angles_indices] = (4 * (large_angles / (q_real[large_angles_indices] + o1)).atan()) / large_angles * q_vec[large_angles_indices]
        return omega_t

    @staticmethod
    def hat(v) -> Tensor:
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B,3,3)`.

        Args:
            v: vector of shape :math:`(B,3)`.

        Example:
            >>> v = torch.rand((1,3))
            >>> m = So3.hat(v)
            >>> m
            tensor([[[ 0.0000, -0.4011,  0.7219],
                     [ 0.4011,  0.0000, -0.3723],
                     [-0.7219,  0.3723,  0.0000]]])
        """
        a, b, c = v[..., 0, None, None], v[..., 1, None, None], v[..., 2, None, None]
        zeros = zeros_like(v)[..., 0, None, None]
        row0 = concatenate([zeros, -c, b], 2)
        row1 = concatenate([c, zeros, -a], 2)
        row2 = concatenate([-b, a, zeros], 2)
        return concatenate([row0, row1, row2], 1)

    @staticmethod
    def vee(omega) -> Tensor:
        """Converts elements from lie algebra to vector space. Returns vector of shape :math:`(B,3)`.

        Args:
            omega: 3x3-matrix representing lie algebra of the following structure:
            ::

                          [0  -c   b ]
                omega  =  [c   0  -a ]
                          [-b  a   0 ]

        Example:
            >>> v = torch.rand((1,3))
            >>> omega = So3.hat(v)
            >>> So3.vee(omega)
            tensor([[0.1802, 0.8256, 0.6205]])
        """
        a, b, c = omega[..., 2, 1], omega[..., 0, 2], omega[..., 1, 0]
        return stack([a, b, c], 1)

    def matrix(self) -> Tensor:
        """Convert the quaternion to a rotation matrix of shape :math:`(B,3,3)`.

        The matrix is of the form:
            ::

                        [(1-2y^2-2z^2)  (2xy-2zw)       (2xy+2yw)    ]
                        [(2xy+2zw)      (1-2x^2-2z^2)   (2yz-2xw)    ]
                        [(2xz-2yw)      (2yz+2xw)       (1-2x^2-2y^2)]

        Example:
            >>> s = So3.identity(batch_size=1)
            >>> m = s.matrix()
            >>> m
            tensor([[[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]], grad_fn=<CatBackward0>)
        """
        w = self.q.w.unsqueeze(2)
        x, y, z = self.q.x.unsqueeze(2), self.q.y.unsqueeze(2), self.q.z.unsqueeze(2)
        q0 = (1 - 2 * y ** 2 - 2 * z ** 2)
        q1 = (2 * x * y - 2 * z * w)
        q2 = (2 * x * z + 2 * y * w)
        row0 = concatenate([q0, q1, q2], 2)
        q0 = (2 * x * y + 2 * z * w)
        q1 = (1 - 2 * x ** 2 - 2 * z ** 2)
        q2 = (2 * y * z - 2 * x * w)
        row1 = concatenate([q0, q1, q2], 2)
        q0 = (2 * x * z - 2 * y * w)
        q1 = (2 * y * z + 2 * x * w)
        q2 = (1 - 2 * x ** 2 - 2 * y ** 2)
        row2 = concatenate([q0, q1, q2], 2)
        return concatenate([row0, row1, row2], 1)

    @classmethod
    def identity(cls, batch_size: int) -> 'So3':
        """Create a So3 group representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = So3.identity(batch_size=2)
            >>> s.data
            real: tensor([[1.],
                    [1.]], grad_fn=<SliceBackward0>) 
            vec: tensor([[0., 0., 0.],
                    [0., 0., 0.]], grad_fn=<SliceBackward0>)
        """
        return cls(Quaternion.identity(batch_size))

    def inverse(self) -> 'So3':
        return So3(self.q.conj())

    def squared_norm(self, x, y=None) -> Tensor:
        return self._batched_squared_norm(x, y)

    def _batched_squared_norm(self, x, y=None):
        if y is None:
            y = x
        KORNIA_CHECK(x.shape == y.shape)
        return (x[..., None, :] @ y[..., :, None])[..., 0]

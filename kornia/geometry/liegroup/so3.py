# kornia.geometry.so3 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py
from kornia.core import Tensor, concatenate, stack, tensor, where, zeros
from kornia.geometry.liegroup._utils import squared_norm
from kornia.geometry.quaternion import Quaternion
from kornia.testing import KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE


class So3:
    r"""Base class to represent the So3 group.

    The SO(3) is the group of all rotations about the origin of three-dimensional Euclidean space
    :math:`R^3` under the operation of composition.
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

        Internally represented by a unit quaternion `q`.

        Args:
            data: Quaternion with the shape of :math:`(B, 4)`.

        Example:
            >>> data = torch.ones((2, 4))
            >>> q = Quaternion(data)
            >>> So3(q)
            real: tensor([[1.],
                    [1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[1., 1., 1.],
                    [1., 1., 1.]], grad_fn=<SliceBackward0>)
        """
        KORNIA_CHECK_TYPE(q, Quaternion)
        self._q = q

    def __repr__(self) -> str:
        return f"{self.q}"

    def __getitem__(self, idx) -> 'So3':
        return So3(self._q[idx])

    def __mul__(self, right):
        """Compose two So3 transformations.

        Args:
            right: the other So3 transformation.

        Return:
            The resulting So3 transformation.
        """
        # https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py#L98
        if isinstance(right, So3):
            return So3(self.q * right.q)
        elif isinstance(right, Tensor):
            KORNIA_CHECK_SHAPE(right, ["B", "3"])
            w = zeros(right.shape[0], 1).to(right.device, right.dtype)
            return (self.q * Quaternion(concatenate((w, right), 1)) * self.q.conj()).vec
        else:
            raise TypeError(f"Not So3 or Tensor type. Got: {type(right)}")

    @property
    def q(self) -> Quaternion:
        """Return the underlying data with shape :math:`(B,4)`."""
        return self._q

    @staticmethod
    def exp(v) -> 'So3':
        """Converts elements of lie algebra to elements of lie group.

        See more: https://vision.in.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

        Args:
            v: vector of shape :math:`(B,3)`.

        Example:
            >>> v = torch.zeros((2,3))
            >>> s = So3.identity(batch_size=1).exp(v)
            >>> s
            real: tensor([[1.],
                    [1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[0., 0., 0.],
                    [0., 0., 0.]], grad_fn=<SliceBackward0>)
        """
        KORNIA_CHECK_SHAPE(v, ["B", "3"])
        theta = squared_norm(v).sqrt()
        theta_nonzeros = theta != 0.0
        theta_half = 0.5 * theta
        # TODO: uncomment me after deprecate pytorch 10.2
        # w = where(theta_nonzeros, theta_half.cos(), 1.0)
        # b = where(theta_nonzeros, theta_half.sin() / theta, 0.0)
        w = where(theta_nonzeros, theta_half.cos(), tensor(1.0, device=v.device, dtype=v.dtype))
        b = where(theta_nonzeros, theta_half.sin() / theta, tensor(0.0, device=v.device, dtype=v.dtype))
        xyz = b * v
        return So3(Quaternion(concatenate((w, xyz), 1)))

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> data = torch.ones((2, 4))
            >>> q = Quaternion(data)
            >>> So3(q).log()
            tensor([[0., 0., 0.],
                    [0., 0., 0.]], grad_fn=<WhereBackward0>)
        """
        theta = squared_norm(self.q.vec).sqrt()
        # NOTE: this differs from https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py#L33
        omega = where(theta != 0, 2 * self.q.real.acos() * self.q.vec / theta, 2 * self.q.vec / self.q.real)
        return omega

    @staticmethod
    def hat(v) -> Tensor:
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B,3,3)`.

        Args:
            v: vector of shape :math:`(B,3)`.

        Example:
            >>> v = torch.ones((1,3))
            >>> m = So3.hat(v)
            >>> m
            tensor([[[ 0., -1.,  1.],
                     [ 1.,  0., -1.],
                     [-1.,  1.,  0.]]])
        """
        KORNIA_CHECK_SHAPE(v, ["B", "3"])
        v = v[..., None, None]
        a, b, c = v[:, 0], v[:, 1], v[:, 2]
        z = zeros(v.shape[0], 1, 1, device=v.device, dtype=v.dtype)
        row0 = concatenate([z, -c, b], 2)
        row1 = concatenate([c, z, -a], 2)
        row2 = concatenate([-b, a, z], 2)
        return concatenate([row0, row1, row2], 1)

    @staticmethod
    def vee(omega) -> Tensor:
        r"""Converts elements from lie algebra to vector space. Returns vector of shape :math:`(B,3)`.

        .. math::
            omega = \begin{bmatrix} 0 & -c & b \\
            c & 0 & -a \\
            -b & a & 0\end{bmatrix}

        Args:
            omega: 3x3-matrix representing lie algebra.

        Example:
            >>> v = torch.ones((1,3))
            >>> omega = So3.hat(v)
            >>> So3.vee(omega)
            tensor([[1., 1., 1.]])
        """
        KORNIA_CHECK_SHAPE(omega, ["B", "3", "3"])
        a, b, c = omega[..., 2, 1], omega[..., 0, 2], omega[..., 1, 0]
        return stack([a, b, c], 1)

    # NOTE: the math style won't render well with sphinx
    def matrix(self) -> Tensor:
        r"""Convert the quaternion to a rotation matrix of shape :math:`(B,3,3)`.

        The matrix is of the form:

        .. math::
            \begin{bmatrix} 1-2y^2-2z^2 & 2xy-2zw & 2xy+2yw \\
            2xy+2zw & 1-2x^2-2z^2 & 2yz-2xw \\
            2xz-2yw & 2yz+2xw & 1-2x^2-2y^2\end{bmatrix}

        Example:
            >>> s = So3.identity(batch_size=1)
            >>> m = s.matrix()
            >>> m
            tensor([[[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]], grad_fn=<CatBackward0>)
        """
        w = self.q.w[..., None]
        x, y, z = self.q.x[..., None], self.q.y[..., None], self.q.z[..., None]
        q0 = 1 - 2 * y**2 - 2 * z**2
        q1 = 2 * x * y - 2 * z * w
        q2 = 2 * x * z + 2 * y * w
        row0 = concatenate([q0, q1, q2], 2)
        q0 = 2 * x * y + 2 * z * w
        q1 = 1 - 2 * x**2 - 2 * z**2
        q2 = 2 * y * z - 2 * x * w
        row1 = concatenate([q0, q1, q2], 2)
        q0 = 2 * x * z - 2 * y * w
        q1 = 2 * y * z + 2 * x * w
        q2 = 1 - 2 * x**2 - 2 * y**2
        row2 = concatenate([q0, q1, q2], 2)
        return concatenate([row0, row1, row2], 1)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> 'So3':
        """Create So3 from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B,3,3)`.

        Example:
            >>> m = torch.eye(3)[None]
            >>> s = So3.from_matrix(m)
            >>> s
            real: tensor([[1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
        """
        return cls(Quaternion.from_matrix(matrix))

    @classmethod
    def identity(cls, batch_size: int, device=None, dtype=None) -> 'So3':
        """Create a So3 group representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = So3.identity(batch_size=2)
            >>> s
            real: tensor([[1.],
                    [1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[0., 0., 0.],
                    [0., 0., 0.]], grad_fn=<SliceBackward0>)
        """
        return cls(Quaternion.identity(batch_size, device, dtype))

    def inverse(self) -> 'So3':
        """Returns the inverse transformation.

        Example:
            >>> s = So3.identity(batch_size=1)
            >>> s.inverse()
            real: tensor([[1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[-0., -0., -0.]], grad_fn=<SliceBackward0>)
        """
        return So3(self.q.conj())

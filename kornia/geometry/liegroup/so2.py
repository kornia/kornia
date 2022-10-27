# kornia.geometry.so2 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so2.py
from kornia.core import Tensor, concatenate, stack, tensor, where, zeros
from kornia.geometry.liegroup._utils import squared_norm
from kornia.geometry.quaternion import Quaternion
from kornia.testing import KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE


class So2:
    r"""Base class to represent the So2 group.

    The SO(2) is the group of all rotations about the origin of three-dimensional Euclidean space
    :math:`R^2` under the operation of composition.
    See more: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group

    We internally represent the rotation by a unit quaternion.
    TODO
    Example:
        >>> q = Quaternion.identity(batch_size=1)
        >>> s = So2(q)
        >>> s.q
        real: tensor([[1.]], grad_fn=<SliceBackward0>)
        vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
    """

    def __init__(self, z: Tensor) -> None:
        """Constructor for the base class.

        Internally represented by complex number `z`.

        Args:
            z: Complex number with the shape of :math:`(B, 1)`.

        Example:
            >>> z = torch.randn((2, 1), dtype=torch.cfloat)
            >>> So2(z)
            tensor([[ 1.5691+0.9840j],
                    [-0.6943-0.8857j]])
        """
        KORNIA_CHECK_SHAPE(z, ["B", "1"])
        self._z = z

    def __repr__(self) -> str:
        return f"{self.z}"

    def __getitem__(self, idx) -> 'So2':
        return So2(self._z[idx][..., None])

    # def __mul__(self, right):
    #     """Compose two So2 transformations.

    #     Args:
    #         right: the other So2 transformation.

    #     Return:
    #         The resulting So2 transformation.
    #     """
    #     # https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so2.py#L98
    #     if isinstance(right, So2):
    #         return So2(self.q * right.q)
    #     elif isinstance(right, Tensor):
    #         KORNIA_CHECK_SHAPE(right, ["B", "3"])
    #         w = zeros(right.shape[0], 1).to(right.device, right.dtype)
    #         return (self.q * Quaternion(concatenate((w, right), 1)) * self.q.conj()).vec
    #     else:
    #         raise TypeError(f"Not So2 or Tensor type. Got: {type(right)}")

    @property
    def z(self) -> Tensor:
        """Return the underlying data with shape :math:`(B,4)`."""
        return self._z

    # @staticmethod
    # def exp(v) -> 'So2':
    #     """Converts elements of lie algebra to elements of lie group.

    #     See more: https://vision.in.tum.de/_media/members/demmeln/nurlanov2021so2log.pdf

    #     Args:
    #         v: vector of shape :math:`(B,3)`.

    #     Example:
    #         >>> v = torch.zeros((2,3))
    #         >>> s = So2.identity(batch_size=1).exp(v)
    #         >>> s
    #         real: tensor([[1.],
    #                 [1.]], grad_fn=<SliceBackward0>)
    #         vec: tensor([[0., 0., 0.],
    #                 [0., 0., 0.]], grad_fn=<SliceBackward0>)
    #     """
    #     KORNIA_CHECK_SHAPE(v, ["B", "3"])
    #     theta = squared_norm(v).sqrt()
    #     theta_nonzeros = theta != 0.0
    #     theta_half = 0.5 * theta
    #     # TODO: uncomment me after deprecate pytorch 10.2
    #     # w = where(theta_nonzeros, theta_half.cos(), 1.0)
    #     # b = where(theta_nonzeros, theta_half.sin() / theta, 0.0)
    #     w = where(theta_nonzeros, theta_half.cos(), tensor(1.0, device=v.device, dtype=v.dtype))
    #     b = where(theta_nonzeros, theta_half.sin() / theta, tensor(0.0, device=v.device, dtype=v.dtype))
    #     xyz = b * v
    #     return So2(Quaternion(concatenate((w, xyz), 1)))

    # def log(self) -> Tensor:
    #     """Converts elements of lie group  to elements of lie algebra.

    #     Example:
    #         >>> data = torch.ones((2, 4))
    #         >>> q = Quaternion(data)
    #         >>> So2(q).log()
    #         tensor([[0., 0., 0.],
    #                 [0., 0., 0.]], grad_fn=<WhereBackward0>)
    #     """
    #     theta = squared_norm(self.q.vec).sqrt()
    #     # NOTE: this differs from https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so2.py#L33
    #     omega = where(theta != 0, 2 * self.q.real.acos() * self.q.vec / theta, 2 * self.q.vec / self.q.real)
    #     return omega

    # @staticmethod
    # def hat(v) -> Tensor:
    #     """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B,3,3)`.

    #     Args:
    #         v: vector of shape :math:`(B,3)`.

    #     Example:
    #         >>> v = torch.ones((1,3))
    #         >>> m = So2.hat(v)
    #         >>> m
    #         tensor([[[ 0., -1.,  1.],
    #                  [ 1.,  0., -1.],
    #                  [-1.,  1.,  0.]]])
    #     """
    #     KORNIA_CHECK_SHAPE(v, ["B", "3"])
    #     v = v[..., None, None]
    #     a, b, c = v[:, 0], v[:, 1], v[:, 2]
    #     z = zeros(v.shape[0], 1, 1, device=v.device, dtype=v.dtype)
    #     row0 = concatenate([z, -c, b], 2)
    #     row1 = concatenate([c, z, -a], 2)
    #     row2 = concatenate([-b, a, z], 2)
    #     return concatenate([row0, row1, row2], 1)

    # @staticmethod
    # def vee(omega) -> Tensor:
    #     r"""Converts elements from lie algebra to vector space. Returns vector of shape :math:`(B,3)`.

    #     .. math::
    #         omega = \begin{bmatrix} 0 & -c & b \\
    #         c & 0 & -a \\
    #         -b & a & 0\end{bmatrix}

    #     Args:
    #         omega: 3x3-matrix representing lie algebra.

    #     Example:
    #         >>> v = torch.ones((1,3))
    #         >>> omega = So2.hat(v)
    #         >>> So2.vee(omega)
    #         tensor([[1., 1., 1.]])
    #     """
    #     KORNIA_CHECK_SHAPE(omega, ["B", "3", "3"])
    #     a, b, c = omega[..., 2, 1], omega[..., 0, 2], omega[..., 1, 0]
    #     return stack([a, b, c], 1)

    def matrix(self) -> Tensor:
        #TODO
        """Convert the quaternion to a rotation matrix of shape :math:`(B,3,3)`.

        Example:
            >>> s = So2.identity(batch_size=1)
            >>> m = s.matrix()
            >>> m
            tensor([[[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]], grad_fn=<CatBackward0>)
        """
        row0 = concatenate((self.z.real, -self.z.imag), -1)
        row1 = concatenate((self.z.imag, -self.z.real), -1)
        return stack((row0, row1), -1)

    # @classmethod
    # def from_matrix(cls, matrix: Tensor) -> 'So2':
    #     """Create So2 from a rotation matrix.

    #     Args:
    #         matrix: the rotation matrix to convert of shape :math:`(B,3,3)`.

    #     Example:
    #         >>> m = torch.eye(3)[None]
    #         >>> s = So2.from_matrix(m)
    #         >>> s
    #         real: tensor([[1.]], grad_fn=<SliceBackward0>)
    #         vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
    #     """
    #     return cls(Quaternion.from_matrix(matrix))

    # @classmethod
    # def identity(cls, batch_size: int, device=None, dtype=None) -> 'So2':
    #     """Create a So2 group representing an identity rotation.

    #     Args:
    #         batch_size: the batch size of the underlying data.

    #     Example:
    #         >>> s = So2.identity(batch_size=2)
    #         >>> s
    #         real: tensor([[1.],
    #                 [1.]], grad_fn=<SliceBackward0>)
    #         vec: tensor([[0., 0., 0.],
    #                 [0., 0., 0.]], grad_fn=<SliceBackward0>)
    #     """
    #     return cls(Quaternion.identity(batch_size, device, dtype))

    # def inverse(self) -> 'So2':
    #     """Returns the inverse transformation.

    #     Example:
    #         >>> s = So2.identity(batch_size=1)
    #         >>> s.inverse()
    #         real: tensor([[1.]], grad_fn=<SliceBackward0>)
    #         vec: tensor([[-0., -0., -0.]], grad_fn=<SliceBackward0>)
    #     """
    #     return So2(self.q.conj())

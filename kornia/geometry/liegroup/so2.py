# kornia.geometry.so2 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so2.py
from kornia.core import Tensor, concatenate, stack, tensor, complex_
from kornia.geometry.liegroup._utils import squared_norm
from kornia.testing import KORNIA_CHECK_SHAPE


class So2:
    r"""Base class to represent the So2 group.

    The SO(2) is the group of all rotations about the origin of two-dimensional Euclidean space
    :math:`R^2` under the operation of composition.
    See more: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group

    We internally represent the rotation by a complex number.
    Example:
        >>> real = torch.tensor([1, 2], dtype=torch.float32)
        >>> imag = torch.tensor([3, 4], dtype=torch.float32)
        >>> So2(torch.complex(real, imag))
        tensor([1.+3.j, 2.+4.j])
    """

    def __init__(self, z: Tensor) -> None:
        """Constructor for the base class.

        Internally represented by complex number `z`.

        Args:
            z: Complex number with the shape of :math:`(B, 1)`.

        Example:
            >>> real = torch.tensor([1, 2], dtype=torch.float32)
            >>> imag = torch.tensor([3, 4], dtype=torch.float32)
            >>> So2(torch.complex(real, imag))
            tensor([1.+3.j, 2.+4.j])
        """
        KORNIA_CHECK_SHAPE(z, ["B", "1"])
        self._z = z

    def __repr__(self) -> str:
        return f"{self.z}"

    def __getitem__(self, idx) -> 'So2':
        return So2(self._z[idx][..., None])

    def __mul__(self, right):
        """Compose two So2 transformations.

        Args:
            right: the other So2 transformation.

        Return:
            The resulting So2 transformation.
        """
        if isinstance(right, So2):
            return So2(self.z * right.z)
        elif isinstance(right, Tensor):
            KORNIA_CHECK_SHAPE(right, ["B", "2", "1"])
            return (self.matrix() @ right)
        else:
            raise TypeError(f"Not So2 or Tensor type. Got: {type(right)}")

    @property
    def z(self) -> Tensor:
        """Return the underlying data with shape :math:`(B, 1)`."""
        return self._z

    @staticmethod
    def exp(theta) -> 'So2':
        """Converts elements of lie algebra to elements of lie group.

        Args:
            theta: angle in radians  of shape :math:`(B, 1)`.

        Example:
            >>> v = torch.tensor([3.1415/2])[..., None]
            >>> s = So2.exp(v)
            >>> s
            tensor([[4.6329e-05+1.j]])
        """
        KORNIA_CHECK_SHAPE(theta, ["B", "1"])
        return So2(complex_(theta.cos(), theta.sin()))

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.
        
        Example:
            >>> real = torch.tensor([1, 2], dtype=torch.float32)
            >>> imag = torch.tensor([3, 4], dtype=torch.float32)
            >>> So2(torch.complex(real, imag)).log()
            Tensorsor([1.2490, 1.1071])
        """
        return self.z.imag.atan2(self.z.real)

    @staticmethod
    def hat(theta) -> Tensor:
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B,3,3)`.

        Args:
            v: vector of shape :math:`(B,3)`.

        Example:
            >>> v = torch.ones((1,3))
            >>> m = So2.hat(v)
            >>> m
            tensor([[[ 0., -1.,  1.],
                     [ 1.,  0., -1.],
                     [-1.,  1.,  0.]]])
        """
        KORNIA_CHECK_SHAPE(theta, ["B", "1"])
        batch_size = theta.shape[0]
        z = tensor([0.0] * batch_size, device=theta.device, dtype=theta.dtype)[..., None]
        row0 = concatenate((z, theta), -1)
        row1 = concatenate((theta, z), -1)
        return stack((row0, row1), -1)

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
        row1 = concatenate((self.z.imag, self.z.real), -1)
        return stack((row0, row1), -1)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> 'So3':
        """Create So2 from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B,3,3)`.

        Example:
            >>> m = torch.eye(3)[None]
            >>> s = So3.from_matrix(m)
            >>> s
            real: tensor([[1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[0., 0., 0.]], grad_fn=<SliceBackward0>)
        """
        return cls(complex_(matrix[..., 0, 0], matrix[..., 1, 0]))

    @classmethod
    def identity(cls, batch_size: int, device=None, dtype=None) -> 'So2':
        #TODO
        """Create a So2 group representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = So2.identity(batch_size=2)
            >>> s
            real: tensor([[1.],
                    [1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[0., 0., 0.],
                    [0., 0., 0.]], grad_fn=<SliceBackward0>)
        """
        z = complex_(tensor([[1.0]] * batch_size, device=device, dtype=dtype), tensor([[0.0]] * batch_size, device=device, dtype=dtype))
        return cls(z)

    def inverse(self) -> 'So2':
        #TODO
        """Returns the inverse transformation.

        Example:
            >>> s = So2.identity(batch_size=1)
            >>> s.inverse()
            real: tensor([[1.]], grad_fn=<SliceBackward0>)
            vec: tensor([[-0., -0., -0.]], grad_fn=<SliceBackward0>)
        """
        # complex_(self.z.real, -self.z.imag)
        return So2(1 / self.z)

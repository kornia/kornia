# kornia.geometry.so2 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so2.py
from kornia.core import Module, Parameter, Tensor, _complex, concatenate, stack, tensor
from kornia.testing import KORNIA_CHECK_SHAPE


class So2(Module):
    r"""Base class to represent the So2 group.

    The SO(2) is the group of all rotations about the origin of two-dimensional Euclidean space
    :math:`R^2` under the operation of composition.
    See more: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group

    We internally represent the rotation by a complex number.

    Example:
        >>> real = torch.tensor([[1.0]])
        >>> imag = torch.tensor([[2.0]])
        >>> So2(torch.complex(real, imag))
        tensor([1.+2.j], requires_grad=True)
    """

    def __init__(self, z: Tensor) -> None:
        """Constructor for the base class.

        Internally represented by complex number `z`.

        Args:
            z: Complex number with the shape of :math:`(B, 1)`.

        Example:
            >>> real = torch.tensor([[1.0]])
            >>> imag = torch.tensor([[2.0]])
            >>> So2(torch.complex(real, imag))
            tensor([1.+2.j], requires_grad=True)
        """
        super().__init__()
        KORNIA_CHECK_SHAPE(z, ["B", "1"])
        self._z = Parameter(z)

    def __repr__(self) -> str:
        return f"{self.z}"

    def __getitem__(self, idx: int) -> 'So2':
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
            return self.matrix() @ right
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
            theta: angle in radians of shape :math:`(B, 1)`.

        Example:
            >>> v = torch.tensor([[3.1415/2]])
            >>> s = So2.exp(v)
            >>> s
            tensor([[4.6329e-05+1.j]], requires_grad=True)
        """
        KORNIA_CHECK_SHAPE(theta, ["B", "1"])
        return So2(_complex(theta.cos(), theta.sin()))

    def log(self) -> Tensor:
        """Converts elements of lie group to elements of lie algebra.

        Example:
            >>> real = torch.tensor([[1.0]])
            >>> imag = torch.tensor([[3.0]])
            >>> So2(torch.complex(real, imag)).log()
            tensor([1.2490], grad_fn=<Atan2Backward0>)
        """
        return self.z.imag.atan2(self.z.real)

    @staticmethod
    def hat(theta) -> Tensor:
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B, 2, 2)`.

        Args:
            theta: angle in radians of shape :math:`(B, 1)`.

        Example:
            >>> theta = torch.tensor([[3.1415/2]])
            >>> So2.hat(theta)
            tensor([[[0.0000, 1.5707],
                     [1.5707, 0.0000]]])
        """
        KORNIA_CHECK_SHAPE(theta, ["B", "1"])
        batch_size = theta.shape[0]
        z = tensor([0.0] * batch_size, device=theta.device, dtype=theta.dtype)[..., None]
        row0 = stack((z, theta), -1)
        row1 = stack((theta, z), -1)
        return concatenate((row0, row1), 1)

    def matrix(self) -> Tensor:
        """Convert the complex number to a rotation matrix of shape :math:`(B, 2, 2)`.

        Example:
            >>> s = So2.identity(batch_size=1)
            >>> m = s.matrix()
            >>> m
            tensor([[[1., 0.],
                     [0., 1.]]], grad_fn=<CatBackward0>)
        """
        row0 = stack((self.z.real, -self.z.imag), -1)
        row1 = stack((self.z.imag, self.z.real), -1)
        return concatenate((row0, row1), 1)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> 'So2':
        """Create So2 from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B, 2, 2)`.

        Example:
            >>> m = torch.eye(2)[None]
            >>> s = So2.from_matrix(m)
            >>> s
            Parameter containing:
            tensor([1.+0.j], requires_grad=True)
        """
        KORNIA_CHECK_SHAPE(matrix, ["B", "2", "2"])
        return cls(_complex(matrix[..., 0, 0], matrix[..., 1, 0]))

    @classmethod
    def identity(cls, batch_size: int, device=None, dtype=None) -> 'So2':
        """Create a So2 group representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = So2.identity(batch_size=2)
            >>> s
            tensor([[1.+0.j],
                    [1.+0.j]], requires_grad=True)
        """
        z = _complex(
            tensor([[1.0]] * batch_size, device=device, dtype=dtype),
            tensor([[0.0]] * batch_size, device=device, dtype=dtype),
        )
        return cls(z)

    def inverse(self) -> 'So2':
        """Returns the inverse transformation.

        Example:
            >>> s = So2.identity(batch_size=1)
            >>> s.inverse()
            Parameter containing:
            tensor([[1.+0.j]], requires_grad=True)
        """
        return So2(1 / self.z)

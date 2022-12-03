# kornia.geometry.so2 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so2.py
from typing import Optional, Union

from kornia.core import Module, Parameter, Tensor, complex, rand, stack, tensor, zeros_like
from kornia.geometry.liegroup._utils import (
    check_so2_matrix_shape,
    check_so2_t_shape,
    check_so2_theta_shape,
    check_so2_z_shape,
)
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR


class So2(Module):
    r"""Base class to represent the So2 group.

    The SO(2) is the group of all rotations about the origin of two-dimensional Euclidean space
    :math:`R^2` under the operation of composition.
    See more: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group

    We internally represent the rotation by a complex number.

    Example:
        >>> real = torch.tensor([1.0])
        >>> imag = torch.tensor([2.0])
        >>> So2(torch.complex(real, imag))
        Parameter containing:
        tensor([1.+2.j], requires_grad=True)
    """

    def __init__(self, z: Tensor) -> None:
        """Constructor for the base class.

        Internally represented by complex number `z`.

        Args:
            z: Complex number with the shape of :math:`(B, 1)` or :math:`(B)`.

        Example:
            >>> real = torch.tensor(1.0)
            >>> imag = torch.tensor(2.0)
            >>> So2(torch.complex(real, imag)).z
            Parameter containing:
            tensor(1.+2.j, requires_grad=True)
        """
        super().__init__()
        KORNIA_CHECK_IS_TENSOR(z)
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_so2_z_shape(z)
        self._z = Parameter(z)

    def __repr__(self) -> str:
        return f"{self.z}"

    def __getitem__(self, idx: int) -> 'So2':
        return So2(self._z[idx][..., None])

    def __mul__(self, right: Union['So2', Tensor]) -> Union['So2', Tensor]:
        """Performs a left-multiplication either rotation concatenation or point-transform.

        Args:
            right: the other So2 transformation.

        Return:
            The resulting So2 transformation.
        """
        out: Union['So2', Tensor]
        if isinstance(right, So2):
            out = So2(self.z * right.z)
        elif isinstance(right, Tensor):
            # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
            check_so2_t_shape(right)
            x = right[..., 0]
            y = right[..., 1]
            real = self.z.real
            imag = self.z.imag
            out = stack((real * x - imag * y, imag * x + real * y), -1)
        else:
            raise TypeError(f"Not So2 or Tensor type. Got: {type(right)}")
        return out

    @property
    def z(self) -> Tensor:
        """Return the underlying data with shape :math:`(B, 1)`."""
        return self._z

    @staticmethod
    def exp(theta: Tensor) -> 'So2':
        """Converts elements of lie algebra to elements of lie group.

        Args:
            theta: angle in radians of shape :math:`(B, 1)` or :math:`(B)`.

        Example:
            >>> v = torch.tensor([3.1415/2])
            >>> s = So2.exp(v)
            >>> s
            Parameter containing:
            tensor([4.6329e-05+1.j], requires_grad=True)
        """
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_so2_theta_shape(theta)
        return So2(complex(theta.cos(), theta.sin()))

    def log(self) -> Tensor:
        """Converts elements of lie group to elements of lie algebra.

        Example:
            >>> real = torch.tensor([1.0])
            >>> imag = torch.tensor([3.0])
            >>> So2(torch.complex(real, imag)).log()
            tensor([1.2490], grad_fn=<Atan2Backward0>)
        """
        return self.z.imag.atan2(self.z.real)

    @staticmethod
    def hat(theta: Tensor) -> Tensor:
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B, 2, 2)`.

        Args:
            theta: angle in radians of shape :math:`(B)`.

        Example:
            >>> theta = torch.tensor(3.1415/2)
            >>> So2.hat(theta)
            tensor([[0.0000, 1.5707],
                    [1.5707, 0.0000]])
        """
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_so2_theta_shape(theta)
        z = zeros_like(theta)
        row0 = stack((z, theta), -1)
        row1 = stack((theta, z), -1)
        return stack((row0, row1), -1)

    def matrix(self) -> Tensor:
        """Convert the complex number to a rotation matrix of shape :math:`(B, 2, 2)`.

        Example:
            >>> s = So2.identity()
            >>> m = s.matrix()
            >>> m
            tensor([[1., -0.],
                    [0., 1.]], grad_fn=<StackBackward0>)
        """
        row0 = stack((self.z.real, -self.z.imag), -1)
        row1 = stack((self.z.imag, self.z.real), -1)
        return stack((row0, row1), -2)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> 'So2':
        """Create So2 from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B, 2, 2)`.

        Example:
            >>> m = torch.eye(2)
            >>> s = So2.from_matrix(m)
            >>> s.z
            Parameter containing:
            tensor(1.+0.j, requires_grad=True)
        """
        # TODO change to KORNIA_CHECK_SHAPE once there is multiple shape support
        check_so2_matrix_shape(matrix)
        z = complex(matrix[..., 0, 0], matrix[..., 1, 0])
        return cls(z)

    @classmethod
    def identity(cls, batch_size: Optional[int] = None, device=None, dtype=None) -> 'So2':
        """Create a So2 group representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = So2.identity(batch_size=2)
            >>> s
            Parameter containing:
            tensor([1.+0.j, 1.+0.j], requires_grad=True)
        """
        real_data = tensor(1.0, device=device, dtype=dtype)
        imag_data = tensor(0.0, device=device, dtype=dtype)
        if batch_size is not None:
            KORNIA_CHECK(batch_size >= 1, msg="batch_size must be positive")
            real_data = real_data.repeat(batch_size)
            imag_data = imag_data.repeat(batch_size)
        return cls(complex(real_data, imag_data))

    def inverse(self) -> 'So2':
        """Returns the inverse transformation.

        Example:
            >>> s = So2.identity()
            >>> s.inverse().z
            Parameter containing:
            tensor(1.+0.j, requires_grad=True)
        """
        return So2(1 / self.z)

    @classmethod
    def random(cls, batch_size: Optional[int] = None, device=None, dtype=None) -> 'So2':
        """Create a So2 group representing a random rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = So2.random()
            >>> s = So2.random(batch_size=3)
        """
        if batch_size is not None:
            KORNIA_CHECK(batch_size >= 1, msg="batch_size must be positive")
            real_data = rand((batch_size,), device=device, dtype=dtype)
            imag_data = rand((batch_size,), device=device, dtype=dtype)
        else:
            real_data = rand((), device=device, dtype=dtype)
            imag_data = rand((), device=device, dtype=dtype)
        return cls(complex(real_data, imag_data))

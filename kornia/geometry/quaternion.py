# kornia.geometry.quaternion module inspired by Eigen, Sophus-sympy, and PyQuaternion.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/quaternion.py
# https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
# https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Quaternion.h
from math import pi
from typing import Tuple, Union

from kornia.core import Module, Parameter, Tensor, as_tensor, concatenate, rand, stack
from kornia.geometry.conversions import (
    QuaternionCoeffOrder,
    angle_axis_to_quaternion,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE


class Quaternion(Module):
    r"""Base class to represent a Quaternion.

    A quaternion is a four dimensional vector representation of a rotation transformation in 3d.
    See more: https://en.wikipedia.org/wiki/Quaternion

    The general definition of a quaternion is given by:

    .. math ::

        Q = a + b \cdot \mathbf{i} + c \cdot \mathbf{j} + d \cdot \mathbf{k}

    Thus, we represent a rotation quaternion as a contiguous tensor structure to
    perform rigid bodies transformations:

    .. math ::

        Q = \begin{bmatrix} q_w & q_x & q_y & q_z \end{bmatrix}

    Example:
        >>> q = Quaternion.identity(batch_size=4)
        >>> q.data
        Parameter containing:
        tensor([[1., 0., 0., 0.],
                [1., 0., 0., 0.],
                [1., 0., 0., 0.],
                [1., 0., 0., 0.]], requires_grad=True)
        >>> q.real
        tensor([[1.],
                [1.],
                [1.],
                [1.]], grad_fn=<SliceBackward0>)
        >>> q.vec
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]], grad_fn=<SliceBackward0>)
    """

    def __init__(self, data: Tensor) -> None:
        """Constructor for the base class.

        Args:
            data: tensor containing the quaternion data with the sape of :math:`(B, 4)`.

        Example:
            >>> data = torch.rand(2, 4)
            >>> q = Quaternion(data)
            >>> q.shape
            (2, 4)
        """
        super().__init__()
        KORNIA_CHECK_SHAPE(data, ["B", "4"])
        self._data = Parameter(data)

    def __repr__(self) -> str:
        return f"real: {self.real} \nvec: {self.vec}"

    def __getitem__(self, idx) -> 'Quaternion':
        return Quaternion(self.data[idx].reshape(1, -1))

    def __neg__(self) -> 'Quaternion':
        """Inverts the sign of the quaternion data.

        Example:
            >>> q = Quaternion.identity(batch_size=1)
            >>> -q.data
            tensor([[-1., -0., -0., -0.]], grad_fn=<NegBackward0>)
        """
        return Quaternion(-self.data)

    def __add__(self, right: 'Quaternion') -> 'Quaternion':
        """Add a given quaternion.

        Args:
            right: the quaternion to add.

        Example:
            >>> q1 = Quaternion.identity(batch_size=1)
            >>> q2 = Quaternion(Tensor([[2., 0., 1., 1.]]))
            >>> q3 = q1 + q2
            >>> q3.data
            Parameter containing:
            tensor([[3., 0., 1., 1.]], requires_grad=True)
        """
        KORNIA_CHECK_TYPE(right, Quaternion)
        return Quaternion(self.data + right.data)

    def __sub__(self, right: 'Quaternion') -> 'Quaternion':
        """Subtract a given quaternion.

        Args:
            right: the quaternion to subtract.

        Example:
            >>> q1 = Quaternion(Tensor([[2., 0., 1., 1.]]))
            >>> q2 = Quaternion.identity(batch_size=1)
            >>> q3 = q1 - q2
            >>> q3.data
            Parameter containing:
            tensor([[1., 0., 1., 1.]], requires_grad=True)
        """
        KORNIA_CHECK_TYPE(right, Quaternion)
        return Quaternion(self.data - right.data)

    def __mul__(self, right: 'Quaternion') -> 'Quaternion':
        KORNIA_CHECK_TYPE(right, Quaternion)
        # NOTE: borrowed from sophus sympy. Produce less multiplications compared to others.
        # https://github.com/strasdat/Sophus/blob/785fef35b7d9e0fc67b4964a69124277b7434a44/sympy/sophus/quaternion.py#L19
        new_real = self.real * right.real - self._batched_squared_norm(self.vec, right.vec)
        new_vec = self.real * right.vec + right.real * self.vec + self.vec.cross(right.vec)
        return Quaternion(concatenate((new_real, new_vec), -1))

    def __div__(self, right: Union[Tensor, 'Quaternion']) -> 'Quaternion':
        if isinstance(right, Tensor):
            return Quaternion(self.data / right)
        KORNIA_CHECK_TYPE(right, Quaternion)
        return self * right.inv()

    def __truediv__(self, right: 'Quaternion') -> 'Quaternion':
        return self.__div__(right)

    @property
    def data(self) -> Tensor:
        """Return the underlying data with shape :math:`(B,4).`"""
        return self._data

    @property
    def coeffs(self) -> Tensor:
        """Return the underlying data with shape :math:`(B,4)`.

        Alias for :func:`~kornia.geometry.quaternion.Quaternion.data`
        """
        return self._data

    @property
    def real(self) -> Tensor:
        """Return the real part with shape :math:`(B,1)`.

        Alias for :func:`~kornia.geometry.quaternion.Quaternion.w`
        """
        return self.w

    @property
    def vec(self) -> Tensor:
        """Return the vector with the imaginary part with shape :math:`(B,3)`."""
        return self.data[..., 1:]

    @property
    def q(self) -> Tensor:
        """Return the underlying data with shape :math:`(B,4)`.

        Alias for :func:`~kornia.geometry.quaternion.Quaternion.data`
        """
        return self.data

    @property
    def scalar(self) -> Tensor:
        """Return a scalar with the real with shape :math:`(B,1)`.

        Alias for :func:`~kornia.geometry.quaternion.Quaternion.w`
        """
        return self.real

    @property
    def w(self) -> Tensor:
        """Return the :math:`q_w` with shape :math:`(B,1)`."""
        return self.data[..., 0:1]

    @property
    def x(self) -> Tensor:
        """Return the :math:`q_x` with shape :math:`(B,1)`."""
        return self.data[..., 1:2]

    @property
    def y(self) -> Tensor:
        """Return the :math:`q_y` with shape :math:`(B,1)`."""
        return self.data[..., 2:3]

    @property
    def z(self) -> Tensor:
        """Return the :math:`q_z` with shape :math:`(B,1)`."""
        return self.data[..., 3:4]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying data with shape :math:`(B,4)`."""
        return tuple(self.data.shape)

    @property
    def polar_angle(self) -> Tensor:
        """Return the polar angle with shape :math:`(B,1)`.

        Example:
            >>> q = Quaternion.identity(batch_size=1)
            >>> q.polar_angle
            tensor([[0.]], grad_fn=<AcosBackward0>)
        """
        return (self.scalar / self.norm()).acos()

    def matrix(self) -> Tensor:
        """Convert the quaternion to a rotation matrix of shape :math:`(B,3,3)`.

        Example:
            >>> q = Quaternion.identity(batch_size=1)
            >>> m = q.matrix()
            >>> m
            tensor([[[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]], grad_fn=<ViewBackward0>)
        """
        return quaternion_to_rotation_matrix(self.data, order=QuaternionCoeffOrder.WXYZ)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> 'Quaternion':
        """Create a quaternion from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B,3,3)`.

        Example:
            >>> m = torch.eye(3)[None]
            >>> q = Quaternion.from_matrix(m)
            >>> q.data
            Parameter containing:
            tensor([[1., 0., 0., 0.]], requires_grad=True)
        """
        return cls(rotation_matrix_to_quaternion(matrix, order=QuaternionCoeffOrder.WXYZ))

    @classmethod
    def from_axis_angle(cls, axis_angle: Tensor) -> 'Quaternion':
        """Create a quaternion from axis-angle representation.

        Args:
            axis_angle: rotation vector of shape :math:`(B,3)`.

        Example:
            >>> axis_angle = torch.tensor([[1., 0., 0.]])
            >>> q = Quaternion.from_axis_angle(axis_angle)
            >>> q.data
            Parameter containing:
            tensor([[0.8776, 0.4794, 0.0000, 0.0000]], requires_grad=True)
        """
        return cls(angle_axis_to_quaternion(axis_angle, order=QuaternionCoeffOrder.WXYZ))

    @classmethod
    def identity(cls, batch_size: int) -> 'Quaternion':
        """Create a quaternion representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> q = Quaternion.identity(batch_size=2)
            >>> q.data
            Parameter containing:
            tensor([[1., 0., 0., 0.],
                    [1., 0., 0., 0.]], requires_grad=True)
        """
        data: Tensor = as_tensor([1.0, 0.0, 0.0, 0.0])
        data = data.repeat(batch_size, 1)
        return cls(data)

    @classmethod
    def from_coeffs(cls, w: float, x: float, y: float, z: float) -> 'Quaternion':
        """Create a quaternion from the data coefficients.

        Args:
            w: a float representing the :math:`q_w` component.
            x: a float representing the :math:`q_x` component.
            y: a float representing the :math:`q_y` component.
            z: a float representing the :math:`q_z` component.

        Example:
            >>> q = Quaternion.from_coeffs(1., 0., 0., 0.)
            >>> q.data
            Parameter containing:
            tensor([[1., 0., 0., 0.]], requires_grad=True)
        """
        return cls(as_tensor([[w, x, y, z]]))

    @classmethod
    def random(cls, batch_size: int) -> 'Quaternion':
        """Create a random unit quaternion of shape :math:`(B,4)`.

        Uniformly distributed across the rotation space as per: http://planning.cs.uiuc.edu/node198.html

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> q = Quaternion.random(batch_size=2)
            >>> q.norm()
            tensor([1.0000, 1.0000], grad_fn=<NormBackward1>)
        """
        r1, r2, r3 = rand(3, batch_size)
        q1 = (1.0 - r1).sqrt() * ((2 * pi * r2).sin())
        q2 = (1.0 - r1).sqrt() * ((2 * pi * r2).cos())
        q3 = r1.sqrt() * (2 * pi * r3).sin()
        q4 = r1.sqrt() * (2 * pi * r3).cos()
        return cls(stack((q1, q2, q3, q4), -1))

    def norm(self) -> Tensor:
        return self.data.norm(p=2, dim=-1)

    def normalize(self) -> 'Quaternion':
        return Quaternion(normalize_quaternion(self.data))

    def conj(self) -> 'Quaternion':
        return Quaternion(concatenate((self.real, -self.vec), -1))

    def inv(self) -> 'Quaternion':
        return self.conj() / self.squared_norm()

    def squared_norm(self) -> Tensor:
        return self._batched_squared_norm(self.vec) + self.real**2

    def _batched_squared_norm(self, x, y=None):
        if y is None:
            y = x
        KORNIA_CHECK(x.shape == y.shape)
        return (x[..., None, :] @ y[..., :, None])[..., 0]

    # TODO: implement me
    def slerp(self):
        raise NotImplementedError

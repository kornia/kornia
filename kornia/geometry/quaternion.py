# kornia.geometry.quaternion module inspired by Eigen, Sophus-sympy, and PyQuaternion.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/quaternion.py
# https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
# https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Quaternion.h
from math import pi
from typing import Optional, Tuple, Union

from kornia.core import Device, Dtype, Module, Parameter, Tensor, concatenate, rand, stack, tensor, where
from kornia.core.check import KORNIA_CHECK_TYPE
from kornia.geometry.conversions import (
    axis_angle_to_quaternion,
    euler_from_quaternion,
    normalize_quaternion,
    quaternion_from_euler,
    quaternion_to_axis_angle,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from kornia.geometry.linalg import batched_dot_product


class Quaternion(Module):
    r"""Base class to represent a Quaternion.

    A quaternion is a four dimensional vector representation of a rotation transformation in 3d.
    See more: https://en.wikipedia.org/wiki/Quaternion

    The general definition of a quaternion is given by:

    .. math::

        Q = a + b \cdot \mathbf{i} + c \cdot \mathbf{j} + d \cdot \mathbf{k}

    Thus, we represent a rotation quaternion as a contiguous tensor structure to
    perform rigid bodies transformations:

    .. math::

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
        tensor([1., 1., 1., 1.], grad_fn=<SelectBackward0>)
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
        # KORNIA_CHECK_SHAPE(data, ["B", "4"])  # FIXME: resolve shape bugs. @edgarriba
        self._data = Parameter(data)

    def __repr__(self) -> str:
        return f"{self.data}"

    def __getitem__(self, idx: Union[int, slice]) -> "Quaternion":
        return Quaternion(self.data[idx])

    def __neg__(self) -> "Quaternion":
        """Inverts the sign of the quaternion data.

        Example:
            >>> q = Quaternion.identity()
            >>> -q.data
            tensor([-1., -0., -0., -0.], grad_fn=<NegBackward0>)
        """
        return Quaternion(-self.data)

    def __add__(self, right: "Quaternion") -> "Quaternion":
        """Add a given quaternion.

        Args:
            right: the quaternion to add.

        Example:
            >>> q1 = Quaternion.identity()
            >>> q2 = Quaternion(tensor([2., 0., 1., 1.]))
            >>> q3 = q1 + q2
            >>> q3.data
            Parameter containing:
            tensor([3., 0., 1., 1.], requires_grad=True)
        """
        KORNIA_CHECK_TYPE(right, Quaternion)
        return Quaternion(self.data + right.data)

    def __sub__(self, right: "Quaternion") -> "Quaternion":
        """Subtract a given quaternion.

        Args:
            right: the quaternion to subtract.

        Example:
            >>> q1 = Quaternion(tensor([2., 0., 1., 1.]))
            >>> q2 = Quaternion.identity()
            >>> q3 = q1 - q2
            >>> q3.data
            Parameter containing:
            tensor([1., 0., 1., 1.], requires_grad=True)
        """
        KORNIA_CHECK_TYPE(right, Quaternion)
        return Quaternion(self.data - right.data)

    def __mul__(self, right: "Quaternion") -> "Quaternion":
        KORNIA_CHECK_TYPE(right, Quaternion)
        # NOTE: borrowed from sophus sympy. Produce less multiplications compared to others.
        # https://github.com/strasdat/Sophus/blob/785fef35b7d9e0fc67b4964a69124277b7434a44/sympy/sophus/quaternion.py#L19
        new_real = self.real * right.real - batched_dot_product(self.vec, right.vec)
        new_vec = self.real[..., None] * right.vec + right.real[..., None] * self.vec + self.vec.cross(right.vec)
        return Quaternion(concatenate((new_real[..., None], new_vec), -1))

    def __div__(self, right: Union[Tensor, "Quaternion"]) -> "Quaternion":
        if isinstance(right, Tensor):
            return Quaternion(self.data / right[..., None])
        KORNIA_CHECK_TYPE(right, Quaternion)
        return self * right.inv()

    def __truediv__(self, right: "Quaternion") -> "Quaternion":
        return self.__div__(right)

    def __pow__(self, t: float) -> "Quaternion":
        """Return the power of a quaternion raised to exponent t.

        Args:
            t: raised exponent.

        Example:
            >>> q = Quaternion(tensor([1., .5, 0., 0.]))
            >>> q_pow = q**2
        """
        theta = self.polar_angle[..., None]
        vec_norm = self.vec.norm(dim=-1, keepdim=True)
        n = where(vec_norm != 0, self.vec / vec_norm, self.vec * 0)
        w = (t * theta).cos()
        xyz = (t * theta).sin() * n
        return Quaternion(concatenate((w, xyz), -1))

    @property
    def data(self) -> Tensor:
        """Return the underlying data with shape :math:`(B, 4).`"""
        return self._data

    @property
    def coeffs(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return a tuple with the underlying coefficients in WXYZ order."""
        return self.w, self.x, self.y, self.z

    @property
    def real(self) -> Tensor:
        """Return the real part with shape :math:`(B,)`.

        Alias for
        :func: `~kornia.geometry.quaternion.Quaternion.w`
        """
        return self.w

    @property
    def vec(self) -> Tensor:
        """Return the vector with the imaginary part with shape :math:`(B, 3)`."""
        return self.data[..., 1:]

    @property
    def q(self) -> Tensor:
        """Return the underlying data with shape :math:`(B, 4)`.

        Alias for :func:`~kornia.geometry.quaternion.Quaternion.data`
        """
        return self.data

    @property
    def scalar(self) -> Tensor:
        """Return a scalar with the real with shape :math:`(B,)`.

        Alias for
        :func: `~kornia.geometry.quaternion.Quaternion.w`
        """
        return self.real

    @property
    def w(self) -> Tensor:
        """Return the :math:`q_w` with shape :math:`(B,)`."""
        return self.data[..., 0]

    @property
    def x(self) -> Tensor:
        """Return the :math:`q_x` with shape :math:`(B,)`."""
        return self.data[..., 1]

    @property
    def y(self) -> Tensor:
        """Return the :math:`q_y` with shape :math:`(B,)`."""
        return self.data[..., 2]

    @property
    def z(self) -> Tensor:
        """Return the :math:`q_z` with shape :math:`(B,)`."""
        return self.data[..., 3]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying data with shape :math:`(B, 4)`."""
        return tuple(self.data.shape)

    @property
    def polar_angle(self) -> Tensor:
        """Return the polar angle with shape :math:`(B,1)`.

        Example:
            >>> q = Quaternion.identity()
            >>> q.polar_angle
            tensor(0., grad_fn=<AcosBackward0>)
        """
        return (self.scalar / self.norm()).acos()

    def matrix(self) -> Tensor:
        """Convert the quaternion to a rotation matrix of shape :math:`(B, 3, 3)`.

        Example:
            >>> q = Quaternion.identity()
            >>> m = q.matrix()
            >>> m
            tensor([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]], grad_fn=<ViewBackward0>)
        """
        return quaternion_to_rotation_matrix(self.data)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> "Quaternion":
        """Create a quaternion from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B, 3, 3)`.

        Example:
            >>> m = torch.eye(3)[None]
            >>> q = Quaternion.from_matrix(m)
            >>> q.data
            Parameter containing:
            tensor([[1., 0., 0., 0.]], requires_grad=True)
        """
        return cls(rotation_matrix_to_quaternion(matrix))

    @classmethod
    def from_euler(cls, roll: Tensor, pitch: Tensor, yaw: Tensor) -> "Quaternion":
        """Create a quaternion from Euler angles.

        Args:
            roll: the roll euler angle.
            pitch: the pitch euler angle.
            yaw: the yaw euler angle.

        Example:
            >>> roll, pitch, yaw = tensor(0), tensor(1), tensor(0)
            >>> q = Quaternion.from_euler(roll, pitch, yaw)
            >>> q.data
            Parameter containing:
            tensor([0.8776, 0.0000, 0.4794, 0.0000], requires_grad=True)
        """
        w, x, y, z = quaternion_from_euler(roll=roll, pitch=pitch, yaw=yaw)
        q = stack((w, x, y, z), -1)
        return cls(q)

    def to_euler(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert the quaternion to a triple of Euler angles (roll, pitch, yaw).

        Example:
            >>> q = Quaternion(tensor([2., 0., 1., 1.]))
            >>> roll, pitch, yaw = q.to_euler()
            >>> roll
            tensor(2.0344, grad_fn=<Atan2Backward0>)
            >>> pitch
            tensor(1.5708, grad_fn=<AsinBackward0>)
            >>> yaw
            tensor(2.2143, grad_fn=<Atan2Backward0>)
        """
        return euler_from_quaternion(self.w, self.x, self.y, self.z)

    @classmethod
    def from_axis_angle(cls, axis_angle: Tensor) -> "Quaternion":
        """Create a quaternion from axis-angle representation.

        Args:
            axis_angle: rotation vector of shape :math:`(B, 3)`.

        Example:
            >>> axis_angle = torch.tensor([[1., 0., 0.]])
            >>> q = Quaternion.from_axis_angle(axis_angle)
            >>> q.data
            Parameter containing:
            tensor([[0.8776, 0.4794, 0.0000, 0.0000]], requires_grad=True)
        """
        return cls(axis_angle_to_quaternion(axis_angle))

    def to_axis_angle(self) -> Tensor:
        """Converts the quaternion to an axis-angle representation.

        Example:
            >>> q = Quaternion.identity()
            >>> axis_angle = q.to_axis_angle()
            >>> axis_angle
            tensor([0., 0., 0.], grad_fn=<AsStridedBackward0>)
        """
        return quaternion_to_axis_angle(self.data)

    @classmethod
    def identity(
        cls, batch_size: Optional[int] = None, device: Optional[Device] = None, dtype: Dtype = None
    ) -> "Quaternion":
        """Create a quaternion representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> q = Quaternion.identity()
            >>> q.data
            Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
        """
        data = tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        if batch_size is not None:
            data = data.repeat(batch_size, 1)
        return cls(data)

    @classmethod
    def from_coeffs(cls, w: float, x: float, y: float, z: float) -> "Quaternion":
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
            tensor([1., 0., 0., 0.], requires_grad=True)
        """
        return cls(tensor([w, x, y, z]))

    # TODO: update signature
    # def random(cls, shape: Optional[List] = None, device = None, dtype = None) -> 'Quaternion':
    @classmethod
    def random(
        cls, batch_size: Optional[int] = None, device: Optional[Device] = None, dtype: Dtype = None
    ) -> "Quaternion":
        """Create a random unit quaternion of shape :math:`(B, 4)`.

        Uniformly distributed across the rotation space as per: http://planning.cs.uiuc.edu/node198.html

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> q = Quaternion.random()
            >>> q = Quaternion.random(batch_size=2)
        """
        rand_shape = (batch_size,) if batch_size is not None else ()

        r1, r2, r3 = rand((3, *rand_shape), device=device, dtype=dtype)
        q1 = (1.0 - r1).sqrt() * ((2 * pi * r2).sin())
        q2 = (1.0 - r1).sqrt() * ((2 * pi * r2).cos())
        q3 = r1.sqrt() * (2 * pi * r3).sin()
        q4 = r1.sqrt() * (2 * pi * r3).cos()
        return cls(stack((q1, q2, q3, q4), -1))

    def slerp(self, q1: "Quaternion", t: float) -> "Quaternion":
        """Returns a unit quaternion spherically interpolated between quaternions self.q and q1.

        See more: https://en.wikipedia.org/wiki/Slerp

        Args:
            q1: second quaternion to be interpolated between.
            t: interpolation ratio, range [0-1]

        Example:
            >>> q0 = Quaternion.identity()
            >>> q1 = Quaternion(torch.tensor([1., .5, 0., 0.]))
            >>> q2 = q0.slerp(q1, .3)
        """
        KORNIA_CHECK_TYPE(q1, Quaternion)
        q0 = self.normalize()
        q1 = q1.normalize()
        return q0 * (q0.inv() * q1) ** t

    # TODO: add docs
    def norm(self, keepdim: bool = False) -> Tensor:
        # p==2, dim|axis==-1, keepdim
        return self.data.norm(2, -1, keepdim)

    # TODO: add docs
    def normalize(self) -> "Quaternion":
        return Quaternion(normalize_quaternion(self.data))

    # TODO: add docs
    def conj(self) -> "Quaternion":
        return Quaternion(concatenate((self.real[..., None], -self.vec), -1))

    # TODO: add docs
    def inv(self) -> "Quaternion":
        return self.conj() / self.squared_norm()

    # TODO: add docs
    def squared_norm(self) -> Tensor:
        return batched_dot_product(self.vec, self.vec) + self.real**2

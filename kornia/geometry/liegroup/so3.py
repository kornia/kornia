# kornia.geometry.so3 module inspired by Sophus-sympy.
# https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py
from __future__ import annotations

from typing import Optional

from kornia.core import Device, Dtype, Module, Tensor, concatenate, eye, stack, tensor, where, zeros, zeros_like
from kornia.core.check import KORNIA_CHECK_TYPE
from kornia.geometry.conversions import vector_to_skew_symmetric_matrix
from kornia.geometry.linalg import batched_dot_product
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.vector import Vector3


class So3(Module):
    r"""Base class to represent the So3 group.

    The SO(3) is the group of all rotations about the origin of three-dimensional Euclidean space
    :math:`R^3` under the operation of composition.
    See more: https://en.wikipedia.org/wiki/3D_rotation_group

    We internally represent the rotation by a unit quaternion.

    Example:
        >>> q = Quaternion.identity()
        >>> s = So3(q)
        >>> s.q
        Parameter containing:
        tensor([1., 0., 0., 0.], requires_grad=True)
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
            Parameter containing:
            tensor([[1., 1., 1., 1.],
                    [1., 1., 1., 1.]], requires_grad=True)
        """
        super().__init__()
        KORNIA_CHECK_TYPE(q, Quaternion)
        self._q = q

    def __repr__(self) -> str:
        return f"{self.q}"

    def __getitem__(self, idx: int | slice) -> So3:
        return So3(self._q[idx])

    def __mul__(self, right: So3) -> So3:
        """Compose two So3 transformations.

        Args:
            right: the other So3 transformation.

        Return:
            The resulting So3 transformation.
        """
        # https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py#L98
        if isinstance(right, So3):
            return So3(self.q * right.q)
        elif isinstance(right, (Tensor, Vector3)):
            # KORNIA_CHECK_SHAPE(right, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
            w = zeros(*right.shape[:-1], 1, device=right.device, dtype=right.dtype)
            quat = Quaternion(concatenate((w, right.data), -1))
            out = (self.q * quat * self.q.conj()).vec
            if isinstance(right, Tensor):
                return out
            elif isinstance(right, Vector3):
                return Vector3(out)
        else:
            raise TypeError(f"Not So3 or Tensor type. Got: {type(right)}")

    @property
    def q(self) -> Quaternion:
        """Return the underlying data with shape :math:`(B,4)`."""
        return self._q

    @staticmethod
    def exp(v: Tensor) -> So3:
        """Converts elements of lie algebra to elements of lie group.

        See more: https://vision.in.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

        Args:
            v: vector of shape :math:`(B,3)`.

        Example:
            >>> v = torch.zeros((2, 3))
            >>> s = So3.exp(v)
            >>> s
            Parameter containing:
            tensor([[1., 0., 0., 0.],
                    [1., 0., 0., 0.]], requires_grad=True)
        """
        # KORNIA_CHECK_SHAPE(v, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
        theta = batched_dot_product(v, v).sqrt()[..., None]
        theta_nonzeros = theta != 0.0
        theta_half = 0.5 * theta
        # TODO: uncomment me after deprecate pytorch 10.2
        # w = where(theta_nonzeros, theta_half.cos(), 1.0)
        # b = where(theta_nonzeros, theta_half.sin() / theta, 0.0)
        w = where(theta_nonzeros, theta_half.cos(), tensor(1.0, device=v.device, dtype=v.dtype))
        b = where(theta_nonzeros, theta_half.sin() / theta, tensor(0.0, device=v.device, dtype=v.dtype))
        xyz = b * v
        return So3(Quaternion(concatenate((w, xyz), -1)))

    def log(self) -> Tensor:
        """Converts elements of lie group  to elements of lie algebra.

        Example:
            >>> data = torch.ones((2, 4))
            >>> q = Quaternion(data)
            >>> So3(q).log()
            tensor([[0., 0., 0.],
                    [0., 0., 0.]], grad_fn=<WhereBackward0>)
        """
        theta = batched_dot_product(self.q.vec, self.q.vec).sqrt()
        # NOTE: this differs from https://github.com/strasdat/Sophus/blob/master/sympy/sophus/so3.py#L33
        omega = where(
            theta[..., None] != 0,
            2 * self.q.real[..., None].acos() * self.q.vec / theta[..., None],
            2 * self.q.vec / self.q.real[..., None],
        )
        return omega

    @staticmethod
    def hat(v: Vector3 | Tensor) -> Tensor:
        """Converts elements from vector space to lie algebra. Returns matrix of shape :math:`(B,3,3)`.

        Args:
            v: Vector3 or tensor of shape :math:`(B,3)`.

        Example:
            >>> v = torch.ones((1,3))
            >>> m = So3.hat(v)
            >>> m
            tensor([[[ 0., -1.,  1.],
                     [ 1.,  0., -1.],
                     [-1.,  1.,  0.]]])
        """
        # KORNIA_CHECK_SHAPE(v, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
        if isinstance(v, Tensor):
            # TODO: Figure out why mypy think `v` can be a Vector3 which didn't allow ellipsis on index
            a, b, c = v[..., 0], v[..., 1], v[..., 2]  # type: ignore[index]
        else:
            a, b, c = v.x, v.y, v.z
        z = zeros_like(a)
        row0 = stack((z, -c, b), -1)
        row1 = stack((c, z, -a), -1)
        row2 = stack((-b, a, z), -1)
        return stack((row0, row1, row2), -2)

    @staticmethod
    def vee(omega: Tensor) -> Tensor:
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
        # KORNIA_CHECK_SHAPE(omega, ["B", "3", "3"])  # FIXME: resolve shape bugs. @edgarriba
        a, b, c = omega[..., 2, 1], omega[..., 0, 2], omega[..., 1, 0]
        return stack((a, b, c), -1)

    def matrix(self) -> Tensor:
        r"""Convert the quaternion to a rotation matrix of shape :math:`(B,3,3)`.

        The matrix is of the form:

        .. math::
            \begin{bmatrix} 1-2y^2-2z^2 & 2xy-2zw & 2xy+2yw \\
            2xy+2zw & 1-2x^2-2z^2 & 2yz-2xw \\
            2xz-2yw & 2yz+2xw & 1-2x^2-2y^2\end{bmatrix}

        Example:
            >>> s = So3.identity()
            >>> m = s.matrix()
            >>> m
            tensor([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]], grad_fn=<StackBackward0>)
        """
        w = self.q.w[..., None]
        x, y, z = self.q.x[..., None], self.q.y[..., None], self.q.z[..., None]
        q0 = 1 - 2 * y**2 - 2 * z**2
        q1 = 2 * x * y - 2 * z * w
        q2 = 2 * x * z + 2 * y * w
        row0 = concatenate((q0, q1, q2), -1)
        q0 = 2 * x * y + 2 * z * w
        q1 = 1 - 2 * x**2 - 2 * z**2
        q2 = 2 * y * z - 2 * x * w
        row1 = concatenate((q0, q1, q2), -1)
        q0 = 2 * x * z - 2 * y * w
        q1 = 2 * y * z + 2 * x * w
        q2 = 1 - 2 * x**2 - 2 * y**2
        row2 = concatenate((q0, q1, q2), -1)
        return stack((row0, row1, row2), -2)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> So3:
        """Create So3 from a rotation matrix.

        Args:
            matrix: the rotation matrix to convert of shape :math:`(B,3,3)`.

        Example:
            >>> m = torch.eye(3)
            >>> s = So3.from_matrix(m)
            >>> s
            Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
        """
        return cls(Quaternion.from_matrix(matrix))

    @classmethod
    def from_wxyz(cls, wxyz: Tensor) -> So3:
        """Create So3 from a tensor representing a quaternion.

        Args:
            wxyz: the quaternion to convert of shape :math:`(B,4)`.

        Example:
            >>> q = torch.tensor([1., 0., 0., 0.])
            >>> s = So3.from_wxyz(q)
            >>> s
            Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
        """
        # KORNIA_CHECK_SHAPE(wxyz, ["B", "4"])  # FIXME: resolve shape bugs. @edgarriba
        return cls(Quaternion(wxyz))

    @classmethod
    def identity(
        cls, batch_size: Optional[int] = None, device: Optional[Device] = None, dtype: Optional[Dtype] = None
    ) -> So3:
        """Create a So3 group representing an identity rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = So3.identity()
            >>> s
            Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)

            >>> s = So3.identity(batch_size=2)
            >>> s
            Parameter containing:
            tensor([[1., 0., 0., 0.],
                    [1., 0., 0., 0.]], requires_grad=True)
        """
        return cls(Quaternion.identity(batch_size, device, dtype))

    def inverse(self) -> So3:
        """Returns the inverse transformation.

        Example:
            >>> s = So3.identity()
            >>> s.inverse()
            Parameter containing:
            tensor([1., -0., -0., -0.], requires_grad=True)
        """
        return So3(self.q.conj())

    @classmethod
    def random(
        cls, batch_size: Optional[int] = None, device: Optional[Device] = None, dtype: Optional[Dtype] = None
    ) -> So3:
        """Create a So3 group representing a random rotation.

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> s = So3.random()
            >>> s = So3.random(batch_size=3)
        """
        return cls(Quaternion.random(batch_size, device, dtype))

    @classmethod
    def rot_x(cls, x: Tensor) -> So3:
        """Construct a x-axis rotation.

        Args:
            x: the x-axis rotation angle.
        """
        zs = zeros_like(x)
        return cls.exp(stack((x, zs, zs), -1))

    @classmethod
    def rot_y(cls, y: Tensor) -> So3:
        """Construct a z-axis rotation.

        Args:
            y: the y-axis rotation angle.
        """
        zs = zeros_like(y)
        return cls.exp(stack((zs, y, zs), -1))

    @classmethod
    def rot_z(cls, z: Tensor) -> So3:
        """Construct a z-axis rotation.

        Args:
            z: the z-axis rotation angle.
        """
        zs = zeros_like(z)
        return cls.exp(stack((zs, zs, z), -1))

    def adjoint(self) -> Tensor:
        """Returns the adjoint matrix of shape :math:`(B, 3, 3)`.

        Example:
            >>> s = So3.identity()
            >>> s.adjoint()
            tensor([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]], grad_fn=<StackBackward0>)
        """
        return self.matrix()

    @staticmethod
    def right_jacobian(vec: Tensor) -> Tensor:
        """Computes the right Jacobian of So3.

        Args:
            vec: the input point of shape :math:`(B, 3)`.

        Example:
            >>> vec = torch.tensor([1., 2., 3.])
            >>> So3.right_jacobian(vec)
            tensor([[-0.0687,  0.5556, -0.0141],
                    [-0.2267,  0.1779,  0.6236],
                    [ 0.5074,  0.3629,  0.5890]])
        """
        # KORNIA_CHECK_SHAPE(vec, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
        R_skew = vector_to_skew_symmetric_matrix(vec)
        theta = vec.norm(dim=-1, keepdim=True)[..., None]
        I = eye(3, device=vec.device, dtype=vec.dtype)  # noqa: E741
        Jr = I - ((1 - theta.cos()) / theta**2) * R_skew + ((theta - theta.sin()) / theta**3) * (R_skew @ R_skew)
        return Jr

    @staticmethod
    def Jr(vec: Tensor) -> Tensor:
        """Alias for right jacobian.

        Args:
            vec: the input point of shape :math:`(B, 3)`.
        """
        return So3.right_jacobian(vec)

    @staticmethod
    def left_jacobian(vec: Tensor) -> Tensor:
        """Computes the left Jacobian of So3.

        Args:
            vec: the input point of shape :math:`(B, 3)`.

        Example:
            >>> vec = torch.tensor([1., 2., 3.])
            >>> So3.left_jacobian(vec)
            tensor([[-0.0687, -0.2267,  0.5074],
                    [ 0.5556,  0.1779,  0.3629],
                    [-0.0141,  0.6236,  0.5890]])
        """
        # KORNIA_CHECK_SHAPE(vec, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
        R_skew = vector_to_skew_symmetric_matrix(vec)
        theta = vec.norm(dim=-1, keepdim=True)[..., None]
        I = eye(3, device=vec.device, dtype=vec.dtype)  # noqa: E741
        Jl = I + ((1 - theta.cos()) / theta**2) * R_skew + ((theta - theta.sin()) / theta**3) * (R_skew @ R_skew)
        return Jl

    @staticmethod
    def Jl(vec: Tensor) -> Tensor:
        """Alias for left jacobian.

        Args:
            vec: the input point of shape :math:`(B, 3)`.
        """
        return So3.left_jacobian(vec)

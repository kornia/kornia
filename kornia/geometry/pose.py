from __future__ import annotations

import uuid

from kornia.core import Tensor, eye
from kornia.geometry.liegroup import Se2, Se3, So2, So3
from kornia.geometry.quaternion import Quaternion


def check_R_shape(R: Tensor) -> None:
    if len(R.shape) > 3 or len(R.shape) < 2:
        raise ValueError(f"R must be either 2x2 or 3x3, got {R.shape}")
    if list(R.shape[-2:]) not in ([2, 2], [3, 3]):
        raise ValueError(f"R must be either 2x2 or 3x3, got {R.shape}")


class NamedPose:
    r"""Class to represent a named pose between two frames.

    Internally represented by either Se2 or Se3.

    Example:
        >>> b_from_a = NamedPose(Se3.identity(), frame1="frame_a", frame2="frame_b")
        >>> b_from_a
        NamedPose(frame2_from_frame1=rotation: Parameter containing:
        tensor([1., 0., 0., 0.], requires_grad=True)
        translation: x: 0.0
        y: 0.0
        z: 0.0,
        frame1: frame_a -> frame2: frame_b)
    """

    def __init__(self, frame2_from_frame1: Se2 | Se3, frame1: str | None = None, frame2: str | None = None) -> None:
        """Constructor for NamedPose.

        Args:
            frame2_from_frame1: Pose from frame 1 to frame 2.
            frame1: Name of frame a.
            frame2: Name of frame b.
        """
        self._frame2_from_frame1 = frame2_from_frame1
        self._frame1 = frame1 or uuid.uuid4().hex
        self._frame2 = frame2 or uuid.uuid4().hex

    def __repr__(self) -> str:
        return f"NamedPose(frame2_from_frame1={self._frame2_from_frame1}, \
                \nframe1: {self._frame1} -> frame2: {self._frame2})"

    def __mul__(self, other: NamedPose) -> NamedPose:
        """Compose two NamedPoses.

        Args:
            other: NamedPose to compose with.

        Returns:
            NamedPose: Composed NamedPose.

        Example:
            >>> b_from_a = NamedPose(Se3.identity(), frame1="frame_a", frame2="frame_b")
            >>> c_from_b = NamedPose(Se3.identity(), frame1="frame_b", frame2="frame_c")
            >>> b_from_a * c_from_b
            NamedPose(frame2_from_frame1=rotation: Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
            translation: x: 0.0
            y: 0.0
            z: 0.0,
            frame1: frame_a -> frame2: frame_c)
        """
        if self.frame2 != other.frame1:
            raise ValueError(f"Cannot compose {self} with {other}")
        return NamedPose(self.pose * other.pose, self.frame1, other.frame2)

    @property
    def pose(self) -> Se2 | Se3:
        """Pose from frame 1 to frame 2."""
        return self._frame2_from_frame1

    @property
    def rotation(self) -> So3 | So2:
        """Rotation part of the pose."""
        return self._frame2_from_frame1.rotation

    @property
    def translation(self) -> Tensor:
        """Translation part of the pose."""
        return self._frame2_from_frame1.translation

    @property
    def frame1(self) -> str:
        """Name of frame 1."""
        return self._frame1

    @property
    def frame2(self) -> str:
        """Name of frame 2."""
        return self._frame2

    @classmethod
    def from_RT(
        cls, R: So3 | So2 | Tensor | Quaternion, T: Tensor, frame1: str | None = None, frame2: str | None = None
    ) -> NamedPose:
        """Construct NamedPose from rotation and translation.

        Args:
            R : Rotation part of the pose.
            T : Translation part of the pose.
            frame1 : Name of frame a.
            frame2 : Name of frame b.

        Returns:
            NamedPose: NamedPose constructed from rotation and translation.

        Example:
            >>> b_from_a_rotation = So3.identity()
            >>> b_from_a_trans = torch.tensor([1., 2., 3.])
            >>> b_from_a = NamedPose.from_RT(b_from_a_rotation, b_from_a_trans, frame1="frame_a", frame2="frame_b")
            >>> b_from_a
            NamedPose(frame2_from_frame1=rotation: Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
            translation: Parameter containing:
            tensor([1., 2., 3.], requires_grad=True),
            frame1: frame_a -> frame2: frame_b)
        """
        # TODO: Add support for RT matrix
        if isinstance(R, (So3, Quaternion)):
            return cls(Se3(R, T), frame1, frame2)
        elif isinstance(R, So2):
            return cls(Se2(R, T), frame1, frame2)
        elif isinstance(R, Tensor):
            check_R_shape(R)
            dim = R.shape[-1]
            RT = eye(dim + 1, device=R.device, dtype=R.dtype)
            RT[..., :dim, :dim] = R
            RT[..., :dim, dim] = T
            if dim == 2:
                return cls(Se2.from_matrix(RT), frame1, frame2)
            elif dim == 3:
                return cls(Se3.from_matrix(RT), frame1, frame2)
        else:
            raise ValueError(f"R must be either So2, So3, Quaternion, or Tensor, got {type(R)}")

    @classmethod
    def from_colmap() -> NamedPose:
        raise NotImplementedError

    def inverse(self) -> NamedPose:
        """Inverse of the NamedPose.

        Returns:
            NamedPose: Inverse of the NamedPose.

        Example:
            >>> b_from_a = NamedPose(Se3.identity(), frame1="frame_a", frame2="frame_b")
            >>> b_from_a.inverse()
            NamedPose(frame2_from_frame1=rotation: Parameter containing:
            tensor([1., -0., -0., -0.], requires_grad=True)
            translation: x: 0.0
            y: 0.0
            z: 0.0,
            frame1: frame_b -> frame2: frame_a)
        """
        return NamedPose(self._frame2_from_frame1.inverse(), self.frame2, self.frame1)

    def transform(self, points_in_frame1: Tensor) -> Tensor:
        """Transform points from frame 1 to frame 2.

        Args:
            points_in_frame1: Points in frame 1.

        Returns:
            Tensor: Points in frame 2.

        Example:
            >>> b_from_a = NamedPose(Se3.identity(), frame1="frame_a", frame2="frame_b")
            >>> b_from_a.transform(torch.tensor([1., 2., 3.]))
            tensor([1., 2., 3.], grad_fn=<AddBackward0>)
        """
        return self._frame2_from_frame1 * points_in_frame1

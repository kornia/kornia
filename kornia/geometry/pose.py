from __future__ import annotations

import uuid

from kornia.core import Tensor, eye
from kornia.geometry.liegroup import Se2, Se3, So2, So3
from kornia.geometry.quaternion import Quaternion


def check_matrix_shape(matrix: Tensor, matrix_type: str = "R") -> None:
    target_shapes = []
    if matrix_type == "R":
        target_shapes = [[2, 2], [3, 3]]
    elif matrix_type == "RT":
        target_shapes = [[3, 3], [4, 4]]
    if len(matrix.shape) > 3 or len(matrix.shape) < 2 or list(matrix.shape[-2:]) not in target_shapes:
        raise ValueError(
            f"{matrix_type} must be either {target_shapes[0]}x{target_shapes[0]} or \
              {target_shapes[1]}x{target_shapes[1]}, got {matrix.shape}"
        )


class NamedPose:
    r"""Class to represent a named pose between two frames.

    Internally represented by either Se2 or Se3.

    Example:
        >>> b_from_a = NamedPose(Se3.identity(), frame_src="frame_a", frame_dst="frame_b")
        >>> b_from_a
        NamedPose(dst_from_src=rotation: Parameter containing:
        tensor([1., 0., 0., 0.], requires_grad=True)
        translation: x: 0.0
        y: 0.0
        z: 0.0,
        frame_src: frame_a -> frame_dst: frame_b)
    """

    def __init__(self, dst_from_src: Se2 | Se3, frame_src: str | None = None, frame_dst: str | None = None) -> None:
        """Constructor for NamedPose.

        Args:
            dst_from_src: Pose from frame 1 to frame 2.
            src: Name of frame a.
            dst: Name of frame b.
        """
        self._dst_from_src = dst_from_src
        self._frame_src = frame_src or uuid.uuid4().hex
        self._frame_dst = frame_dst or uuid.uuid4().hex

    def __repr__(self) -> str:
        return f"NamedPose(dst_from_src={self._dst_from_src},\nframe_src: {self._frame_src} -> frame_dst: {self._frame_dst})"

    def __mul__(self, other: NamedPose) -> NamedPose:
        """Compose two NamedPoses.

        Args:
            other: NamedPose to compose with.

        Returns:
            NamedPose: Composed NamedPose.

        Example:
            >>> b_from_a = NamedPose(Se3.identity(), frame_src="frame_a", frame_dst="frame_b")
            >>> c_from_b = NamedPose(Se3.identity(), frame_src="frame_b", frame_dst="frame_c")
            >>> c_from_b * b_from_a
            NamedPose(dst_from_src=rotation: Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
            translation: x: 0.0
            y: 0.0
            z: 0.0,
            frame_src: frame_a -> frame_dst: frame_c)
        """
        if self._frame_src != other._frame_dst:
            raise ValueError(f"Cannot compose {self} with {other}")
        # TODO: fix mypy error
        return NamedPose(other.pose * self.pose, other._frame_src, self._frame_dst)

    @property
    def pose(self) -> Se2 | Se3:
        """Pose from frame 1 to frame 2."""
        return self._dst_from_src

    @property
    def rotation(self) -> So3 | So2:
        """Rotation part of the pose."""
        return self._dst_from_src.rotation

    @property
    def translation(self) -> Tensor:
        """Translation part of the pose."""
        return self._dst_from_src.translation

    @property
    def frame_src(self) -> str:
        """Name of the source frame."""
        return self._frame_src

    @property
    def frame_dst(self) -> str:
        """Name of the destination frame."""
        return self._frame_dst

    @classmethod
    def from_rt(
        cls,
        rotation: So3 | So2 | Tensor | Quaternion,
        translation: Tensor,
        frame_src: str | None = None,
        frame_dst: str | None = None,
    ) -> NamedPose | None:
        """Construct NamedPose from rotation and translation.

        Args:
            rotation: Rotation part of the pose.
            T : Translation part of the pose.
            frame_src : Name of the source frame.
            frame_dst : Name of the destination frame.

        Returns:
            NamedPose: NamedPose constructed from rotation and translation.

        Example:
            >>> b_from_a_rot = So3.identity()
            >>> b_from_a_trans = torch.tensor([1., 2., 3.])
            >>> b_from_a = NamedPose.from_rt(b_from_a_rot, b_from_a_trans, frame_src="frame_a", frame_dst="frame_b")
            >>> b_from_a
            NamedPose(dst_from_src=rotation: Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
            translation: Parameter containing:
            tensor([1., 2., 3.], requires_grad=True),
            frame_src: frame_a -> frame_dst: frame_b)
        """
        if isinstance(rotation, (So3, Quaternion)):
            return cls(Se3(rotation, translation), frame_src, frame_dst)
        elif isinstance(rotation, So2):
            return cls(Se2(rotation, translation), frame_src, frame_dst)
        elif isinstance(rotation, Tensor):
            check_matrix_shape(rotation)
            dim = rotation.shape[-1]
            RT = eye(dim + 1, device=rotation.device, dtype=rotation.dtype)
            RT[..., :dim, :dim] = rotation
            RT[..., :dim, dim] = translation
            if dim == 2:
                return cls(Se2.from_matrix(RT), frame_src, frame_dst)
            elif dim == 3:
                return cls(Se3.from_matrix(RT), frame_src, frame_dst)
        else:
            raise ValueError(f"R must be either So2, So3, Quaternion, or Tensor, got {type(rotation)}")
        return None

    @classmethod
    def from_matrix(
        cls, matrix: Tensor, frame_src: str | None = None, frame_dst: str | None = None
    ) -> NamedPose | None:
        """Construct NamedPose from a matrix.

        Args:
            matrix : Matrix representation of the pose.
            frame_src : Name of the source frame.
            frame_dst : Name of the destination frame.

        Returns:
            NamedPose: NamedPose constructed from a matrix.

        Example:
            >>> b_from_a_matrix = Se3.identity().matrix()
            >>> b_from_a = NamedPose.from_matrix(b_from_a_matrix, frame_src="frame_a", frame_dst="frame_b")
            >>> b_from_a
            NamedPose(dst_from_src=rotation: Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
            translation: Parameter containing:
            tensor([0., 0., 0.], requires_grad=True),
            frame_src: frame_a -> frame_dst: frame_b)
        """
        check_matrix_shape(matrix, matrix_type="RT")
        dim = matrix.shape[-1]
        if dim == 3:
            return cls(Se2.from_matrix(matrix), frame_src, frame_dst)
        elif dim == 4:
            return cls(Se3.from_matrix(matrix), frame_src, frame_dst)
        return None

    @classmethod
    def from_colmap() -> NamedPose:
        raise NotImplementedError

    def inverse(self) -> NamedPose:
        """Inverse of the NamedPose.

        Returns:
            NamedPose: Inverse of the NamedPose.

        Example:
            >>> b_from_a = NamedPose(Se3.identity(), frame_src="frame_a", frame_dst="frame_b")
            >>> b_from_a.inverse()
            NamedPose(dst_from_src=rotation: Parameter containing:
            tensor([1., -0., -0., -0.], requires_grad=True)
            translation: x: 0.0
            y: 0.0
            z: 0.0,
            frame_src: frame_b -> frame_dst: frame_a)
        """
        return NamedPose(self._dst_from_src.inverse(), self._frame_dst, self._frame_src)

    def transform_points(self, points_in_src: Tensor) -> Tensor:
        """Transform points from frame 1 to frame 2.

        Args:
            points_in_src: Points in frame 1.

        Returns:
            Tensor: Points in frame 2.

        Example:
            >>> b_from_a = NamedPose(Se3.identity(), frame_src="frame_a", frame_dst="frame_b")
            >>> b_from_a.transform_points(torch.tensor([1., 2., 3.]))
            tensor([1., 2., 3.], grad_fn=<AddBackward0>)
        """
        return self._dst_from_src * points_in_src

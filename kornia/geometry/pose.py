from __future__ import annotations

import uuid

from kornia.core import Tensor, eye
from kornia.geometry.liegroup import Se2, Se3, So2, So3
from kornia.geometry.quaternion import Quaternion


def check_R_shape(R: Tensor):
    if len(R.shape) > 3 or len(R.shape) < 2:
        raise ValueError(f"R must be either 2x2 or 3x3, got {R.shape}")
    if list(R.shape[-2:]) not in ([2, 2], [3, 3]):
        raise ValueError(f"R must be either 2x2 or 3x3, got {R.shape}")


class NamedPose:
    r"""Class to represent a named pose between two frames.

    Internally represented by either Se2 or Se3.
    """

    def __init__(self, t_a_b: Se2 | Se3, frame_a: str | None = None, frame_b: str | None = None) -> None:
        self.t_a_b = t_a_b
        self._frame_a = frame_a or uuid.uuid4().hex
        self._frame_b = frame_b or uuid.uuid4().hex

    def __repr__(self) -> str:
        return f"NamedPose(t_a_b={self.t_a_b}, \n frame: {self._frame_a} -> frame: {self._frame_b})"

    def __mul__(self, other: NamedPose) -> NamedPose:
        t_a_b = self.t_a_b * other.t_a_b
        return NamedPose(t_a_b, self._frame_a, other.frame_b)

    @property
    def rotation(self) -> So3 | So2:
        return self.t_a_b.rotation

    @property
    def translation(self) -> Tensor:
        return self.t_a_b.translation

    @property
    def frame_a(self) -> str:
        return self._frame_a

    @property
    def frame_b(self) -> str:
        return self._frame_b

    @classmethod
    def from_RT(
        cls, R: So3 | So2 | Tensor | Quaternion, T: Tensor, frame_a: str | None = None, frame_b: str | None = None
    ) -> NamedPose:
        if isinstance(R, (So3, Quaternion)):
            return cls(Se3(R, T), frame_a, frame_b)
        elif isinstance(R, So2):
            return cls(Se2(R, T), frame_a, frame_b)
        elif isinstance(R, Tensor):
            check_R_shape(R)
            dim = R.shape[-1]
            RT = eye(dim + 1, device=R.device, dtype=R.dtype)
            RT[..., :dim, :dim] = R
            RT[..., :dim, dim] = T
            if dim == 2:
                return cls(Se2.from_matrix(RT), frame_a, frame_b)
            elif dim == 3:
                return cls(Se3.from_matrix(RT), frame_a, frame_b)
        else:
            raise ValueError(f"R must be either So2, So3, Quaternion, or Tensor, got {type(R)}")

    def from_colmap():
        raise NotImplementedError

    def inverse(self) -> NamedPose:
        return NamedPose(self.t_a_b.inverse(), self.frame_b, self.frame_a)

    def transform(self, points: Tensor) -> Tensor:
        return self.t_a_b * points

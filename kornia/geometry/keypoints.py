from typing import List, Optional, Tuple, Union

from torch import Tensor
from kornia.geometry import transform_points


def _merge_keypoint_list(keypoints: List[Tensor]) -> Tensor:
    raise NotImplementedError


class Keypoints:

    def __init__(
        self,
        keypoints: Union[Tensor, List[Tensor]],
        raise_if_not_floating_point: bool = True,
    ) -> None:

        self._N: Optional[List[int]] = None

        if isinstance(keypoints, list):
            keypoints, self._N = _merge_keypoint_list(keypoints)

        if not isinstance(keypoints, Tensor):
            raise TypeError(f"Input keypoints is not a Tensor. Got: {type(keypoints)}.")

        if not keypoints.is_floating_point():
            if raise_if_not_floating_point:
                raise ValueError(f"Coordinates must be in floating point. Got {keypoints.dtype}")

            keypoints = keypoints.float()

        if len(keypoints.shape) == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            keypoints = keypoints.reshape((-1, 2))

        if not (2 <= keypoints.ndim <= 3 and keypoints.shape[-1:] == (2,)):
            raise ValueError(f"Keypoints shape must be (N, 2) or (B, N, 2). Got {keypoints.shape}.")

        self._is_batched = False if keypoints.ndim == 2 else True

        self._data = keypoints

    def __getitem__(self, key) -> "Keypoints":
        new_obj = Keypoints(self._data[key], False)
        return new_obj

    def __setitem__(self, key, value: "Keypoints") -> "Keypoints":
        self._data[key] = value._data
        return self

    @property
    def data(self,):
        return self._data

    def pad(
        self,
        padding_size: Tensor,
    ) -> "Keypoints":
        """Pad a bounding keypoints.

        Args:
            padding_size: (B, 4)
        """
        self._data[..., 0] += padding_size[..., :1]  # left padding
        self._data[..., 1] += padding_size[..., 2:3]  # top padding

    def unpad(
        self,
        padding_size: Tensor,
    ) -> "Keypoints":
        """Pad a bounding keypoints.

        Args:
            padding_size: (B, 4)
        """
        self._data[..., 0] -= padding_size[..., :1]  # left padding
        self._data[..., 1] -= padding_size[..., 2:3]  # top padding

    def transform_keypoints(self, M: Tensor, inplace: bool = False) -> "Keypoints":
        r"""Apply a transformation matrix to the 2D keypoints.

        Args:
            M: The transformation matrix to be applied, shape of :math:`(3, 3)` or :math:`(B, 3, 3)`.
            inplace: do transform in-place and return self.

        Returns:
            The transformed keypoints.
        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (3, 3):
            raise ValueError(f"The transformation matrix shape must be (3, 3) or (B, 3, 3). Got {M.shape}.")

        transformed_boxes = transform_points(self._data, M)
        if inplace:
            self._data = transformed_boxes
            return self

        return Keypoints(transformed_boxes, False)

    def transform_keypoints_(self, M: Tensor) -> "Keypoints":
        """Inplace version of :func:`Keypoints.transform_keypoints`"""
        return self.transform_keypoints(M, inplace=True)

    def to_tensor(
        self, as_padded_sequence: bool = False
    ) -> Union[Tensor, List[Tensor]]:
        r"""Cast :class:`Keypoints` to a tensor. ``mode`` controls which 2D keypoints format should be use to represent
        keypoints in the tensor.

        Args:
            as_padded_sequence: whether to keep the pads for a list of keypoints. This parameter is only valid
                if the keypoints are from a keypoint list.

        Returns:
            Keypoints tensor :math:`(B, N, 2)`
        """

        if not as_padded_sequence:
            raise NotImplementedError
        return self._data

    def clone(self) -> "Keypoints":
        return Keypoints(self._data.clone(), False)

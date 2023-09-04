from typing import List, Optional, Tuple, Union

import torch

from kornia.core import Tensor
from kornia.geometry import transform_points


def _merge_keypoint_list(keypoints: List[Tensor]) -> Tuple[Tensor, list[int]]:
    raise NotImplementedError


def _transform_points(points: Tensor, M: Tensor) -> Tensor:
    """Transforms 3D points by applying the transformation matrix M.

    Args:
        points: points of shape :math:`(B, N, 3)`.
        M: the transformation matrix of shape :math:`(3, 3)` or :math:`(B, 3, 3)`.
    """
    M = M if M.is_floating_point() else M.float()

    if points.shape[0] == 0:
        return points

    M = M if M.ndim == 3 else M.unsqueeze(0)

    if points.shape[0] != M.shape[0]:
        raise ValueError(
            f"Batch size mismatch. Got {points.shape[0]} for boxes and {M.shape[0]} for the transformation matrix."
        )

    transformed_points: Tensor = transform_points(M, points)
    transformed_points = transformed_points.view_as(points)
    return transformed_points


class PointCloud:
    def __init__(self, keypoints: Union[Tensor, List[Tensor]],) -> None:
        self._N: Optional[List[int]] = None

        if isinstance(keypoints, list):
            keypoints, self._N = _merge_keypoint_list(keypoints)

        if not isinstance(keypoints, Tensor):
            raise TypeError(f"Input keypoints is not a Tensor. Got: {type(keypoints)}.")

        if not keypoints.is_floating_point():
            keypoints = keypoints.float()

        if not (keypoints.ndim == 3 and keypoints.shape[-1] >= 3):
            raise ValueError(f"Keypoints shape must be (B, N, C) and C >= 3. Got {keypoints.shape}.")

        self._is_batched = True

        self._data = keypoints

    def __getitem__(self, key: Union[slice, int, Tensor]) -> "PointCloud":
        new_obj = type(self)(self._data[key], False)
        return new_obj

    def __setitem__(self, key: Union[slice, int, Tensor], value: "PointCloud") -> "PointCloud":
        self._data[key] = value._data
        return self

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def data(self) -> Tensor:
        return self._data

    def transform_keypoints(self, M: Tensor, inplace: bool = False) -> "PointCloud":
        r"""Apply a transformation matrix to point clouds.

        Args:
            M: The transformation matrix to be applied, shape of :math:`(3, 3)` or :math:`(B, 3, 3)`.
            inplace: do transform in-place and return self.

        Returns:
            The transformed point clouds.
        """
        if inplace:
            self._data[..., :3] = _transform_points(self._data[..., :3], M)
            return self

        obj = self.clone()
        return obj.transform_keypoints_(M)

    def transform_keypoints_(self, M: Tensor) -> "PointCloud":
        """Inplace version of :func:`Keypoints.transform_keypoints`"""
        return self.transform_keypoints(M, inplace=True)

    @classmethod
    def from_tensor(cls, keypoints: Tensor) -> "PointCloud":
        return cls(keypoints)

    def to_tensor(self, as_padded_sequence: bool = False) -> Union[Tensor, List[Tensor]]:
        r"""Cast :class:`Keypoints` to a tensor. ``mode`` controls which 2D keypoints format should be use to
        represent keypoints in the tensor.

        Args:
            as_padded_sequence: whether to keep the pads for a list of keypoints. This parameter is only valid
                if the keypoints are from a keypoint list.

        Returns:
            Keypoints tensor :math:`(B, N, 3)`
        """
        if as_padded_sequence:
            raise NotImplementedError
        return self._data

    def clone(self) -> "PointCloud":
        return PointCloud(self._data.clone(), False)

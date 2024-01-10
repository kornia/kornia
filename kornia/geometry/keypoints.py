from typing import Dict, List, Optional, Tuple, Union, cast

import torch
from torch import Size

from kornia.core import Tensor
from kornia.geometry import transform_points

__all__ = ["Keypoints", "Keypoints3D"]


def _merge_keypoint_list(keypoints: List[Tensor]) -> Tensor:
    raise NotImplementedError


def inside_image(coords: Tensor, image_size: Tensor):
    """
    Args:
        coords: (B, N, 2) or (N, 2) in WH order
        image_size: (B, 2) or (2,) in WH order
    """
    return (
        (coords[..., 0] >= 0)
        & (coords[..., 0] <= image_size[..., None, 0] - 1)
        & (coords[..., 1] >= 0)
        & (coords[..., 1] <= image_size[..., None, 1] - 1)
    )

def transform_valid_mask(data: Tensor, valid_mask: Tensor, params: Optional[Dict[str, Tensor]]):
    if params is not None:
        if "output_size" in params:
            image_size = params["output_size"]
        else:
            image_size = params["forward_input_shape"][-2:]
            if data.ndim == 3:
                image_size = image_size[None].repeat(data.shape[0], 1)

    return valid_mask & inside_image(data, image_size)

class Keypoints:
    """2D Keypoints containing Nx2 or BxNx2 points.

    Args:
        keypoints: Raw tensor or a list of Tensors with the Nx2 coordinates
        raise_if_not_floating_point: will raise if the Tensor isn't float
        valid_mask: (B, N) or (N,) bool tensor indicating whether the keypoint is valid or not
    """

    def __init__(
        self,
        keypoints: Union[Tensor, List[Tensor]],
        raise_if_not_floating_point: bool = True,
        valid_mask: Optional[Tensor] = None,
    ) -> None:
        self._N: Optional[List[int]] = None
        self._valid_mask: Optional[Tensor] = valid_mask

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
        self._valid_mask = self._data.new_ones(self._data.shape[:-1]).bool() if valid_mask is None else valid_mask

    def __getitem__(self, key: Union[slice, int, Tensor]) -> "Keypoints":
        new_obj = type(self)(self._data[key], False)
        return new_obj

    def __setitem__(self, key: Union[slice, int, Tensor], value: "Keypoints") -> "Keypoints":
        self._data[key] = value._data
        return self

    @property
    def shape(self) -> Union[Tuple[int, ...], Size]:
        return self.data.shape

    @property
    def data(self) -> Tensor:
        return self._data

    @property
    def device(self) -> torch.device:
        """Returns keypoints device."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Returns keypoints dtype."""
        return self._data.dtype

    @property
    def valid_mask(self) -> Tensor:
        """Returns keypoints visibility."""
        return self._valid_mask

    def index_put(
        self,
        indices: Union[Tuple[Tensor, ...], List[Tensor]],
        values: Union[Tensor, "Keypoints"],
        params: Dict[str, Tensor],
        inplace: bool = False,
    ) -> "Keypoints":
        if inplace:
            _data = self._data
        else:
            _data = self._data.clone()

        if isinstance(values, Keypoints):
            _data.index_put_(indices, values.data)
        else:
            _data.index_put_(indices, values)

        valid_mask = transform_valid_mask(_data, self._valid_mask, params)

        if inplace:
            self._valid_mask = valid_mask
            return self
        
        obj = self.clone()
        obj._data = _data
        obj._valid_mask = valid_mask
        return obj

    def pad(self, padding_size: Tensor) -> "Keypoints":
        """Pad a bounding keypoints.

        Args:
            padding_size: (B, 4)
        """
        if not (len(padding_size.shape) == 2 and padding_size.size(1) == 4):
            raise RuntimeError(f"Expected padding_size as (B, 4). Got {padding_size.shape}.")
        self._data[..., 0] += padding_size[..., :1]  # left padding
        self._data[..., 1] += padding_size[..., 2:3]  # top padding
        return self

    def unpad(self, padding_size: Tensor) -> "Keypoints":
        """Pad a bounding keypoints.

        Args:
            padding_size: (B, 4)
        """
        if not (len(padding_size.shape) == 2 and padding_size.size(1) == 4):
            raise RuntimeError(f"Expected padding_size as (B, 4). Got {padding_size.shape}.")
        self._data[..., 0] -= padding_size[..., :1]  # left padding
        self._data[..., 1] -= padding_size[..., 2:3]  # top padding
        return self

    def transform_keypoints(self, M: Tensor, params: Dict[str, Tensor], inplace: bool = False) -> "Keypoints":
        r"""Apply a transformation matrix to the 2D keypoints.

        Args:
            M: The transformation matrix to be applied, shape of :math:`(3, 3)` or :math:`(B, 3, 3)`.
            inplace: do transform in-place and return self.

        Returns:
            The transformed keypoints.
        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (3, 3):
            raise ValueError(f"The transformation matrix shape must be (3, 3) or (B, 3, 3). Got {M.shape}.")

        transformed_boxes = transform_points(M, self._data)
        transformed_valid_mask = transform_valid_mask(transformed_boxes, self._valid_mask, params)

        if inplace:
            self._data = transformed_boxes
            self._valid_mask = transformed_valid_mask
            return self
                
        return Keypoints(transformed_boxes, False, transformed_valid_mask)

    def transform_keypoints_(self, M: Tensor, params: Dict[str, Tensor]) -> "Keypoints":
        """Inplace version of :func:`Keypoints.transform_keypoints`"""
        return self.transform_keypoints(M, params=params, inplace=True)

    @classmethod
    def from_tensor(cls, keypoints: Tensor) -> "Keypoints":
        return cls(keypoints)

    def to_tensor(self, as_padded_sequence: bool = False) -> Union[Tensor, List[Tensor]]:
        r"""Cast :class:`Keypoints` to a tensor. ``mode`` controls which 2D keypoints format should be use to
        represent keypoints in the tensor.

        Args:
            as_padded_sequence: whether to keep the pads for a list of keypoints. This parameter is only valid
                if the keypoints are from a keypoint list.

        Returns:
            Keypoints tensor :math:`(B, N, 2)`
        """
        if as_padded_sequence:
            raise NotImplementedError
        return self._data

    def clone(self) -> "Keypoints":
        return Keypoints(self._data.clone(), False, self._valid_mask.clone())

    def type(self, dtype: torch.dtype) -> "Keypoints":
        self._data = self._data.type(dtype)
        return self


class VideoKeypoints(Keypoints):
    temporal_channel_size: int

    @classmethod
    def from_tensor(cls, boxes: Union[Tensor, List[Tensor]], validate_boxes: bool = True) -> "VideoKeypoints":
        if isinstance(boxes, (list,)) or (boxes.dim() != 4 or boxes.shape[-1] != 2):
            raise ValueError("Input box type is not yet supported. Please input an `BxTxNx2` tensor directly.")

        temporal_channel_size = boxes.size(1)

        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        out = cls(boxes.view(boxes.size(0) * boxes.size(1), -1, boxes.size(3)))
        out.temporal_channel_size = temporal_channel_size
        return out

    def to_tensor(self) -> Tensor:  # type: ignore[override]
        out = super().to_tensor(as_padded_sequence=False)
        out = cast(Tensor, out)
        return out.view(-1, self.temporal_channel_size, *out.shape[1:])

    def transform_keypoints(self, M: Tensor, inplace: bool = False) -> "VideoKeypoints":
        out = super().transform_keypoints(M, inplace=inplace)
        if inplace:
            return self
        out = VideoKeypoints(out.data, False)
        out.temporal_channel_size = self.temporal_channel_size
        return out

    def clone(self) -> "VideoKeypoints":
        out = VideoKeypoints(self._data.clone(), False)
        out.temporal_channel_size = self.temporal_channel_size
        return out


class Keypoints3D:
    """3D Keypoints containing Nx3 or BxNx3 points.

    Args:
        keypoints: Raw tensor or a list of Tensors with the Nx3 coordinates
        raise_if_not_floating_point: will raise if the Tensor isn't float
    """

    def __init__(self, keypoints: Union[Tensor, List[Tensor]], raise_if_not_floating_point: bool = True) -> None:
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
            keypoints = keypoints.reshape((-1, 3))

        if not (2 <= keypoints.ndim <= 3 and keypoints.shape[-1:] == (3,)):
            raise ValueError(f"Keypoints shape must be (N, 3) or (B, N, 3). Got {keypoints.shape}.")

        self._is_batched = False if keypoints.ndim == 2 else True

        self._data = keypoints

    def __getitem__(self, key: Union[slice, int, Tensor]) -> "Keypoints3D":
        new_obj = type(self)(self._data[key], False)
        return new_obj

    def __setitem__(self, key: Union[slice, int, Tensor], value: "Keypoints3D") -> "Keypoints3D":
        self._data[key] = value._data
        return self

    @property
    def shape(self) -> Size:
        return self.data.shape

    @property
    def data(self) -> Tensor:
        return self._data

    def pad(self, padding_size: Tensor) -> "Keypoints3D":
        """Pad a bounding keypoints.

        Args:
            padding_size: (B, 6)
        """
        raise NotImplementedError

    def unpad(self, padding_size: Tensor) -> "Keypoints3D":
        """Pad a bounding keypoints.

        Args:
            padding_size: (B, 6)
        """
        raise NotImplementedError

    def transform_keypoints(self, M: Tensor, inplace: bool = False) -> "Keypoints3D":
        r"""Apply a transformation matrix to the 2D keypoints.

        Args:
            M: The transformation matrix to be applied, shape of :math:`(3, 3)` or :math:`(B, 3, 3)`.
            inplace: do transform in-place and return self.

        Returns:
            The transformed keypoints.
        """
        raise NotImplementedError

    def transform_keypoints_(self, M: Tensor) -> "Keypoints3D":
        """Inplace version of :func:`Keypoints.transform_keypoints`"""
        return self.transform_keypoints(M, inplace=True)

    @classmethod
    def from_tensor(cls, keypoints: Tensor) -> "Keypoints3D":
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

    def clone(self) -> "Keypoints3D":
        return Keypoints3D(self._data.clone(), False)

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List, Optional, Tuple, Union, cast

import torch
from torch import Size

from kornia.core._compat import _emit_deprecation_warning
from kornia.geometry import transform_points
from kornia.geometry._coordinate import _cast_to_floating_point

__all__ = ["Keypoints", "Keypoints3D"]


def _merge_keypoint_list(keypoints: List[torch.torch.Tensor]) -> torch.torch.Tensor:
    raise NotImplementedError(
        "Building `Keypoints` or `Keypoints3D` from a list of tensors is not implemented; pass one tensor of "
        "shape `(N, 2)` or `(B, N, 2)` for `Keypoints`, `(N, 3)` or `(B, N, 3)` for `Keypoints3D` (kornia#5023)."
    )


class Keypoints:
    r"""2D keypoints, stored as an ``(N, 2)`` or ``(B, N, 2)`` tensor of ``(x, y)`` points.

    Args:
        keypoints: tensor of ``(N, 2)`` or ``(B, N, 2)`` coordinates. A list of tensors is not implemented
            (see Known defects).
        raise_if_not_floating_point: ``True`` raises ``ValueError`` for a non-floating-point tensor, ``False`` casts
            it to ``float32``, or ``float64`` when that is the default dtype. Floating-point input keeps its dtype.

    Convention:
        - A point is ``(x, y)`` in pixel coordinates (:ref:`Coordinates and sizes <coordinate-conventions>`), the
          frame of the :class:`~kornia.geometry.boxes.Boxes` vertices.
        - :meth:`transform_keypoints` maps a point to :math:`M [x, y, 1]^\top` and converts it back with
          :func:`~kornia.geometry.conversions.convert_points_from_homogeneous`. ``inplace=False`` returns a new
          :class:`Keypoints`; ``inplace=True`` and :meth:`transform_keypoints_` rebind ``self`` to the transformed
          tensor and return ``self``. Neither writes into the tensor that ``self`` wrapped.
        - The constructor and :meth:`from_tensor` wrap a floating-point tensor without copying it. :meth:`pad`,
          :meth:`unpad`, item assignment and ``index_put(inplace=True)`` write into the stored tensor, so they
          change the caller's tensor; call :meth:`clone` first to keep it.
        - Integer indexing selects one batch for batched input, or one point with shape ``(1, 2)`` for unbatched
          input. Scalar integer tensors follow the same rule.
        - Known defects: list input and ``to_tensor(as_padded_sequence=True)`` raise ``NotImplementedError``
          (`#5023 <https://github.com/kornia/kornia/issues/5023>`_).

    """

    def __init__(self, keypoints: torch.Tensor, raise_if_not_floating_point: bool = True) -> None:
        self._N: Optional[List[int]] = None

        if isinstance(keypoints, list):
            keypoints, self._N = _merge_keypoint_list(keypoints)

        if not isinstance(keypoints, torch.torch.Tensor):
            raise TypeError(f"Input keypoints is not a torch.Tensor. Got: {type(keypoints)}.")

        if not keypoints.is_floating_point():
            if raise_if_not_floating_point:
                raise ValueError(f"Coordinates must be in floating point. Got {keypoints.dtype}")

            keypoints = _cast_to_floating_point(keypoints)

        if len(keypoints.shape) == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            keypoints = keypoints.reshape((-1, 2))

        if not (2 <= keypoints.ndim <= 3 and keypoints.shape[-1:] == (2,)):
            raise ValueError(f"Keypoints shape must be (N, 2) or (B, N, 2). Got {keypoints.shape}.")

        self._is_batched = keypoints.ndim != 2

        self._data = keypoints

    def __getitem__(self, key: Union[slice, int, torch.torch.Tensor]) -> "Keypoints":
        data = self._data[key]
        if data.ndim == 1 and not isinstance(key, tuple):
            # an integer or 0-d integer tensor selects one point of unbatched keypoints: keep the point axis
            data = data.unsqueeze(0)
        return type(self)(data, False)

    def __setitem__(self, key: Union[slice, int, torch.torch.Tensor], value: "Keypoints") -> "Keypoints":
        self._data[key] = value._data
        return self

    @property
    def shape(self) -> Union[Tuple[int, ...], Size]:
        """Return the tensor shape used to store 2D keypoints.

        Returns:
            Shape of :attr:`data`. The common layouts are ``(N, 2)`` for
            unbatched keypoints and ``(B, N, 2)`` for batched keypoints,
            where :math:`B` is the batch size, :math:`N` is the number of
            keypoints, and the final dimension stores ``(x, y)`` coordinates.
        """
        return self.data.shape

    @property
    def data(self) -> torch.torch.Tensor:
        """Return the raw 2D keypoint coordinate tensor.

        Returns:
            Tensor storing keypoint coordinates in ``(..., 2)`` format, where
            the last dimension contains ``x`` and ``y``.
        """
        return self._data

    @property
    def device(self) -> torch.device:
        """Returns keypoints device."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Returns keypoints dtype."""
        return self._data.dtype

    def index_put(
        self,
        indices: Union[Tuple[torch.torch.Tensor, ...], List[torch.torch.Tensor]],
        values: Union[torch.torch.Tensor, "Keypoints"],
        inplace: bool = False,
    ) -> "Keypoints":
        """Write keypoint coordinates at selected tensor indices.

        See the Convention block on :class:`Keypoints`.

        Args:
            indices: Index tuple or list accepted by ``Tensor.index_put_`` for
                the stored coordinate tensor.
            values: Replacement coordinates, either as a raw tensor or another
                :class:`Keypoints` object.
            inplace: If ``True``, update this object in place. Otherwise,
                clone the coordinates first and return a new wrapper.

        Returns:
            :class:`Keypoints` object containing the updated coordinates.
        """
        _data = self._data if inplace else self._data.clone()

        if isinstance(values, Keypoints):
            _data.index_put_(indices, values.data)
        else:
            _data.index_put_(indices, values)

        if inplace:
            return self

        obj = self.clone()
        obj._data = _data
        return obj

    def _broadcast_over_points(self, values: torch.torch.Tensor) -> torch.torch.Tensor:
        """Shape a per-image ``(B, 1)`` column so it broadcasts over ``self._data[..., i]``.

        The batched container indexes as ``(B, N)``, which a ``(B, 1)`` column already broadcasts over. The
        unbatched ``(N, 2)`` form carries a single implicit image and indexes as ``(N,)``, so the column must
        drop its batch axis, otherwise the in-place update would broadcast to ``(1, N)``.
        """
        if self._is_batched:
            return values
        if values.size(0) != 1:
            raise RuntimeError(
                f"Unbatched (N, 2) keypoints carry a single image, so a per-image tensor must have one row. "
                f"Got {values.size(0)}."
            )
        return values[0]

    def pad(self, padding_size: torch.torch.Tensor) -> "Keypoints":
        """Pad the keypoints in place.

        See the Convention block on :class:`Keypoints`.

        ``padding_size`` is ordered as ``(left, right, top, bottom)``. Only ``left`` and ``top`` shift the
        coordinate origin. Both the batched ``(B, N, 2)`` and the unbatched ``(N, 2)`` container are
        supported; the unbatched form carries a single image, so ``padding_size`` must have exactly one row.

        Args:
            padding_size: per-image padding in pixels, shaped ``(B, 4)``. A single row broadcasts across the
                batch.

        """
        if not (len(padding_size.shape) == 2 and padding_size.size(1) == 4):
            raise RuntimeError(f"Expected padding_size as (B, 4). Got {padding_size.shape}.")
        offset = padding_size.to(device=self._data.device)
        self._data[..., 0] += self._broadcast_over_points(offset[..., :1])  # left padding
        self._data[..., 1] += self._broadcast_over_points(offset[..., 2:3])  # top padding
        return self

    def unpad(self, padding_size: torch.torch.Tensor) -> "Keypoints":
        """Undo :meth:`pad` in place.

        See the Convention block on :class:`Keypoints`.

        Accepts the same batched and unbatched containers and ``padding_size`` layout as :meth:`pad`.

        Args:
            padding_size: per-image padding in pixels, shaped ``(B, 4)``. A single row broadcasts across the
                batch.

        """
        if not (len(padding_size.shape) == 2 and padding_size.size(1) == 4):
            raise RuntimeError(f"Expected padding_size as (B, 4). Got {padding_size.shape}.")
        offset = padding_size.to(device=self._data.device)
        self._data[..., 0] -= self._broadcast_over_points(offset[..., :1])  # left padding
        self._data[..., 1] -= self._broadcast_over_points(offset[..., 2:3])  # top padding
        return self

    def transform_keypoints(self, M: torch.torch.Tensor, inplace: bool = False) -> "Keypoints":
        r"""Apply a transformation matrix to the 2D keypoints.

        See the Convention block on :class:`Keypoints`.

        Args:
            M: The transformation matrix to be applied, shape of ``(3, 3)`` or ``(B, 3, 3)``.
            inplace: ``True`` rebinds this object to the transformed tensor and returns it; ``False`` returns a new
                :class:`Keypoints`.

        Returns:
            The transformed keypoints.

        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (3, 3):
            raise ValueError(f"The transformation matrix shape must be (3, 3) or (B, 3, 3). Got {M.shape}.")

        transformed_boxes = transform_points(M, self._data)
        if inplace:
            self._data = transformed_boxes
            return self

        return Keypoints(transformed_boxes, False)

    def transform_keypoints_(self, M: torch.torch.Tensor) -> "Keypoints":
        """Inplace version of :func:`Keypoints.transform_keypoints`."""
        return self.transform_keypoints(M, inplace=True)

    @classmethod
    def from_tensor(cls, keypoints: torch.torch.Tensor) -> "Keypoints":
        """Validate and wrap a tensor of 2D keypoint coordinates.

        Args:
            keypoints: Floating-point tensor in ``(N, 2)`` or ``(B, N, 2)`` format; an integer tensor
                raises ``ValueError``. The last dimension stores ``(x, y)`` coordinates.

        Returns:
            New :class:`Keypoints` instance containing the input coordinates.
        """
        return cls(keypoints)

    def to_tensor(self, as_padded_sequence: bool = False) -> Union[torch.torch.Tensor, List[torch.torch.Tensor]]:
        r"""Cast :class:`Keypoints` to a tensor.

        Args:
            as_padded_sequence: not implemented; ``True`` raises ``NotImplementedError``
                (`#5023 <https://github.com/kornia/kornia/issues/5023>`_).

        Returns:
            The stored tensor itself, ``(N, 2)`` or ``(B, N, 2)``.

        """
        if as_padded_sequence:
            raise NotImplementedError(
                "`Keypoints.to_tensor(as_padded_sequence=True)` is not implemented; it needs the keypoint-list "
                "input that `Keypoints` does not support. Call `to_tensor()` for the `(N, 2)` or `(B, N, 2)` "
                "tensor instead (kornia#5023)."
            )
        return self._data

    def clone(self) -> "Keypoints":
        """Create an independent copy of the 2D keypoints."""
        return Keypoints(self._data.clone(), False)

    def type(self, dtype: torch.dtype) -> "Keypoints":
        """Cast stored keypoint coordinates to a target dtype.

        Args:
            dtype: Destination dtype for the coordinate tensor.

        Returns:
            ``self``, rebound to the converted tensor; the tensor it wrapped before is not modified.
        """
        self._data = self._data.type(dtype)
        return self


class VideoKeypoints(Keypoints):
    """Video keypoints shaped ``(B, T, N, 2)``, stored internally as ``(B * T, N, 2)``.

    Non-contiguous inputs are copied only when flattening requires it. Empty batches and point sets
    retain their shape on conversion back to a tensor; the temporal dimension must be nonzero.

    Indexing and item assignment select whole videos on the video-batch axis, as for
    :class:`~kornia.geometry.boxes.VideoBoxes`: ``vk[i]`` is a one-video :class:`VideoKeypoints` whose
    :meth:`to_tensor` is ``(1, T, N, 2)``, and a boolean mask has length ``B``. Index frames with ``vk.data``.
    """

    temporal_channel_size: int

    @classmethod
    def from_tensor(
        cls, boxes: Union[torch.torch.Tensor, List[torch.torch.Tensor]], validate_boxes: bool = True
    ) -> "VideoKeypoints":
        if isinstance(boxes, (list,)) or (boxes.dim() != 4 or boxes.shape[-1] != 2):
            raise ValueError("Input box type is not yet supported. Please input an `BxTxNx2` tensor directly.")

        temporal_channel_size = boxes.size(1)

        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        out = cls(boxes.reshape(boxes.size(0) * boxes.size(1), boxes.size(2), boxes.size(3)))
        out.temporal_channel_size = temporal_channel_size
        return out

    def __getitem__(self, key: Union[slice, int, torch.torch.Tensor]) -> "VideoKeypoints":
        r"""Select videos while preserving the temporal axis, as :meth:`VideoBoxes.__getitem__` does.

        An integer or scalar tensor keeps a singleton video-batch dimension. Slices, integer tensors, and boolean
        tensors follow the corresponding PyTorch indexing behavior on the video-batch axis.
        """
        batch_size = self._data.shape[0] // self.temporal_channel_size
        selected = self._data.view(batch_size, self.temporal_channel_size, *self._data.shape[1:])[key]
        if selected.ndim == self._data.ndim:
            selected = selected.unsqueeze(0)
        out = type(self)(selected.flatten(0, 1), False)
        out.temporal_channel_size = self.temporal_channel_size
        return out

    def __setitem__(self, key: Union[slice, int, torch.torch.Tensor], value: "Keypoints") -> "VideoKeypoints":
        r"""Write whole videos, selected by ``key`` on the video-batch axis as in :meth:`__getitem__`.

        A :class:`VideoKeypoints` value must have the same ``temporal_channel_size``: otherwise its frames would be
        regrouped into videos of another length.
        """
        size = self.temporal_channel_size
        if isinstance(value, VideoKeypoints) and value.temporal_channel_size != size:
            raise ValueError(
                f"Cannot assign videos of {value.temporal_channel_size} frames to videos of {size} frames."
            )
        # Explicit video counts: a clip with no points has 0 elements, so a -1 in the view would be ambiguous.
        n, m = self._data.shape[0] // size, value.data.shape[0] // size
        self._data.view(n, size, *self._data.shape[1:])[key] = value.data.view(m, size, *value.data.shape[1:])
        return self

    def to_tensor(self) -> torch.torch.Tensor:
        out = super().to_tensor(as_padded_sequence=False)
        out = cast(torch.torch.Tensor, out)
        batch_size = out.shape[0] // self.temporal_channel_size
        return out.reshape(batch_size, self.temporal_channel_size, *out.shape[1:])

    def transform_keypoints(self, M: torch.torch.Tensor, inplace: bool = False) -> "VideoKeypoints":
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
    """3D keypoints, stored as an ``(N, 3)`` or ``(B, N, 3)`` tensor of ``(x, y, z)`` points.

    The constructor validates the tensor as :class:`Keypoints` does and wraps a floating-point one without copying it.

    Args:
        keypoints: tensor of ``(N, 3)`` or ``(B, N, 3)`` coordinates. A list of tensors is not implemented
            (see Known defects).
        raise_if_not_floating_point: ``True`` raises ``ValueError`` for a non-floating-point tensor, ``False`` casts
            it to ``float32``, or ``float64`` when that is the default dtype. Floating-point input keeps its dtype.

    Convention:
        - Integer indexing selects one batch for batched input, or one point with shape ``(1, 3)`` for unbatched
          input. Scalar integer tensors follow the same rule.
        - Known defects: list input and ``to_tensor(as_padded_sequence=True)`` raise ``NotImplementedError``;
          :meth:`pad` and :meth:`unpad` are deprecated and also raise it
          (`#5023 <https://github.com/kornia/kornia/issues/5023>`_).

    """

    def __init__(self, keypoints: torch.Tensor, raise_if_not_floating_point: bool = True) -> None:
        self._N: Optional[List[int]] = None

        if isinstance(keypoints, list):
            keypoints, self._N = _merge_keypoint_list(keypoints)

        if not isinstance(keypoints, torch.torch.Tensor):
            raise TypeError(f"Input keypoints is not a torch.Tensor. Got: {type(keypoints)}.")

        if not keypoints.is_floating_point():
            if raise_if_not_floating_point:
                raise ValueError(f"Coordinates must be in floating point. Got {keypoints.dtype}")

            keypoints = _cast_to_floating_point(keypoints)

        if len(keypoints.shape) == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            keypoints = keypoints.reshape((-1, 3))

        if not (2 <= keypoints.ndim <= 3 and keypoints.shape[-1:] == (3,)):
            raise ValueError(f"Keypoints shape must be (N, 3) or (B, N, 3). Got {keypoints.shape}.")

        self._is_batched = keypoints.ndim != 2

        self._data = keypoints

    def __getitem__(self, key: Union[slice, int, torch.Tensor]) -> "Keypoints3D":
        data = self._data[key]
        if data.ndim == 1 and not isinstance(key, tuple):
            # an integer or 0-d integer tensor selects one point of unbatched keypoints: keep the point axis
            data = data.unsqueeze(0)
        return type(self)(data, False)

    def __setitem__(self, key: Union[slice, int, torch.Tensor], value: "Keypoints3D") -> "Keypoints3D":
        self._data[key] = value._data
        return self

    @property
    def shape(self) -> Size:
        """Return the tensor shape used to store 3D keypoints.

        Returns:
            Shape of :attr:`data`. The common layouts are ``(N, 3)`` and
            ``(B, N, 3)``, where the final dimension stores ``(x, y, z)``
            coordinates.
        """
        return self.data.shape

    @property
    def data(self) -> torch.torch.Tensor:
        """Return the raw 3D keypoint coordinate tensor.

        Returns:
            Tensor storing coordinates in ``(..., 3)`` format, with the final
            dimension ordered as ``(x, y, z)``.
        """
        return self._data

    def pad(self, padding_size: torch.torch.Tensor) -> "Keypoints3D":
        """Deprecated since 0.9.1 and never implemented: raises ``NotImplementedError``."""
        _emit_deprecation_warning(
            "Keypoints3D.pad",
            None,
            "0.9.1",
            "It has never been implemented: the padding layout of 3D keypoints is not defined (kornia#5023).",
        )
        raise NotImplementedError("`Keypoints3D.pad` is not implemented (kornia#5023).")

    def unpad(self, padding_size: torch.torch.Tensor) -> "Keypoints3D":
        """Deprecated since 0.9.1 and never implemented: raises ``NotImplementedError``."""
        _emit_deprecation_warning(
            "Keypoints3D.unpad",
            None,
            "0.9.1",
            "It has never been implemented: the padding layout of 3D keypoints is not defined (kornia#5023).",
        )
        raise NotImplementedError("`Keypoints3D.unpad` is not implemented (kornia#5023).")

    def transform_keypoints(self, M: torch.Tensor, inplace: bool = False) -> "Keypoints3D":
        r"""Apply a transformation matrix to the 3D keypoints.

        A point is mapped to :math:`M [x, y, z, 1]^\top` and converted back with
        :func:`~kornia.geometry.conversions.convert_points_from_homogeneous`.

        Args:
            M: The transformation matrix to be applied, shape of ``(4, 4)`` or ``(B, 4, 4)``.
            inplace: ``True`` rebinds this object to the transformed tensor and returns it, without writing into the
                tensor it wrapped; ``False`` returns a new :class:`Keypoints3D`.

        Returns:
            The transformed keypoints.

        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (4, 4):
            raise ValueError(f"The transformation matrix shape must be (4, 4) or (B, 4, 4). Got {M.shape}.")

        transformed = transform_points(M, self._data)
        if inplace:
            self._data = transformed
            return self

        return Keypoints3D(transformed, False)

    def transform_keypoints_(self, M: torch.Tensor) -> "Keypoints3D":
        """Inplace version of :func:`Keypoints3D.transform_keypoints`."""
        return self.transform_keypoints(M, inplace=True)

    @classmethod
    def from_tensor(cls, keypoints: torch.Tensor) -> "Keypoints3D":
        """Validate and wrap a tensor of 3D keypoint coordinates.

        Args:
            keypoints: Tensor in ``(N, 3)`` or ``(B, N, 3)`` format,
                where the last dimension stores ``(x, y, z)``.

        Returns:
            New :class:`Keypoints3D` instance containing the input coordinates.
        """
        return cls(keypoints)

    def to_tensor(self, as_padded_sequence: bool = False) -> Union[torch.torch.Tensor, List[torch.torch.Tensor]]:
        r"""Cast :class:`Keypoints3D` to a tensor.

        Args:
            as_padded_sequence: not implemented; ``True`` raises ``NotImplementedError``
                (`#5023 <https://github.com/kornia/kornia/issues/5023>`_).

        Returns:
            The stored tensor itself, ``(N, 3)`` or ``(B, N, 3)``.

        """
        if as_padded_sequence:
            raise NotImplementedError(
                "`Keypoints3D.to_tensor(as_padded_sequence=True)` is not implemented; it needs the keypoint-list "
                "input that `Keypoints3D` does not support. Call `to_tensor()` for the `(N, 3)` or `(B, N, 3)` "
                "tensor instead (kornia#5023)."
            )
        return self._data

    def clone(self) -> "Keypoints3D":
        """Create an independent copy of the 3D keypoints."""
        return Keypoints3D(self._data.clone(), False)

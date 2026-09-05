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

from __future__ import annotations

from typing import Optional, Tuple, cast

import torch
from torch import Size

from kornia.core.ops import eye_like
from kornia.core.utils import is_exporting
from kornia.geometry.linalg import transform_points

__all__ = ["Boxes", "Boxes3D", "VideoBoxes"]


def _is_floating_point_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.half)


def _merge_box_list(boxes: list[torch.Tensor], method: str = "pad") -> tuple[torch.Tensor, list[int]]:
    r"""Merge a list of boxes into one tensor."""
    if not all(box.shape[-2:] == torch.Size([4, 2]) and box.dim() == 3 for box in boxes):
        raise TypeError(f"Input boxes must be a list of (N, 4, 2) shaped. Got: {[box.shape for box in boxes]}.")

    if method == "pad":
        max_N = max(box.shape[0] for box in boxes)
        stats = [max_N - box.shape[0] for box in boxes]
        output = torch.nn.utils.rnn.pad_sequence(boxes, batch_first=True)
    else:
        raise NotImplementedError(f"`{method}` is not implemented.")

    return output, stats


def _transform_boxes(boxes: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Transform 3D and 2D in kornia format by applying the transformation matrix M.

    Boxes and the transformation matrix could be batched or not.

    Args:
        boxes: 2D quadrilaterals or 3D hexahedrons in kornia format.
        M: the transformation matrix of shape :math:`(3, 3)` or :math:`(B, 3, 3)` for 2D and :math:`(4, 4)` or
            :math:`(B, 4, 4)` for 3D hexahedron.

    """
    M = M if M.is_floating_point() else M.float()

    # Work with batch as kornia.transform_points only supports a batch of points.
    boxes_per_batch, n_points_per_box, coordinates_dimension = boxes.shape[-3:]
    if boxes_per_batch == 0:
        return boxes
    points = boxes.view(-1, n_points_per_box * boxes_per_batch, coordinates_dimension)
    M = M if M.ndim == 3 else M.unsqueeze(0)

    if points.shape[0] != M.shape[0]:
        raise ValueError(
            f"Batch size mismatch. Got {points.shape[0]} for boxes and {M.shape[0]} for the transformation matrix."
        )

    transformed_boxes: torch.Tensor = transform_points(M, points)
    transformed_boxes = transformed_boxes.view_as(boxes)
    return transformed_boxes


def _boxes_to_polygons(
    xmin: torch.Tensor, ymin: torch.Tensor, width: torch.Tensor, height: torch.Tensor
) -> torch.Tensor:
    if not xmin.ndim == ymin.ndim == width.ndim == height.ndim == 2:
        raise ValueError("We expect to create a batch of 2D boxes (quadrilaterals) in vertices format (B, N, 4, 2)")

    # Create (B,N,4,2) with all points in top left position of boxes
    polygons = torch.zeros((xmin.shape[0], xmin.shape[1], 4, 2), device=xmin.device, dtype=xmin.dtype)
    polygons[..., 0] = xmin.unsqueeze(-1)
    polygons[..., 1] = ymin.unsqueeze(-1)
    # Shift top-right, bottom-right, bottom-left points to the right coordinates
    polygons[..., 1, 0] += width - 1  # Top right
    polygons[..., 2, 0] += width - 1  # Bottom right
    polygons[..., 2, 1] += height - 1  # Bottom right
    polygons[..., 3, 1] += height - 1  # Bottom left
    return polygons


def _boxes_to_quadrilaterals(boxes: torch.Tensor, mode: str = "xyxy", validate_boxes: bool = True) -> torch.Tensor:
    """Convert from boxes to quadrilaterals."""
    mode = mode.lower()

    if mode.startswith("vertices"):
        batched = boxes.ndim == 4
        if not (3 <= boxes.ndim <= 4 and boxes.shape[-2:] == torch.Size([4, 2])):
            raise ValueError(f"Boxes shape must be (N, 4, 2) or (B, N, 4, 2) when {mode} mode. Got {boxes.shape}.")
    elif mode.startswith("xy"):
        batched = boxes.ndim == 3
        if not (2 <= boxes.ndim <= 3 and boxes.shape[-1] == 4):
            raise ValueError(f"Boxes shape must be (N, 4) or (B, N, 4) when {mode} mode. Got {boxes.shape}.")
    else:
        raise ValueError(f"Unknown mode {mode}")

    boxes = boxes if boxes.is_floating_point() else boxes.float()
    boxes = boxes if batched else boxes.unsqueeze(0)

    if mode.startswith("vertices"):
        if mode == "vertices":
            quadrilaterals = boxes.clone()
            # Here, vertices are quadrilaterals with width and height defined as `width = xmax - xmin`  and
            # `height = ymax - ymin`. We need to convert to `width = xmax - xmin + 1` and `height = ymax - ymin + 1` to
            # match with internal Boxes Kornia representation.
            quadrilaterals[..., 1:3, 0] = quadrilaterals[..., 1:3, 0] - 1
            quadrilaterals[..., 2:, 1] = quadrilaterals[..., 2:, 1] - 1
        elif mode == "vertices_plus":
            # Avoid passing reference
            quadrilaterals = boxes.clone()
        else:
            raise ValueError(f"Unknown mode {mode}")
    elif mode.startswith("xy"):
        if mode == "xyxy":
            height, width = boxes[..., 3] - boxes[..., 1], boxes[..., 2] - boxes[..., 0]
        elif mode == "xyxy_plus":
            height, width = boxes[..., 3] - boxes[..., 1] + 1, boxes[..., 2] - boxes[..., 0] + 1
        elif mode == "xywh":
            height, width = boxes[..., 3], boxes[..., 2]
        else:
            raise ValueError(f"Unknown mode {mode}")

        # Value validation reads the data, which graph capture cannot do; skip it under export.
        if validate_boxes and not is_exporting():
            if (width <= 0).any():
                raise ValueError("Some boxes have negative widths or 0.")
            if (height <= 0).any():
                raise ValueError("Some boxes have negative heights or 0.")

        xmin, ymin = boxes[..., 0], boxes[..., 1]
        quadrilaterals = _boxes_to_polygons(xmin, ymin, width, height)
    else:
        raise ValueError(f"Unknown mode {mode}")

    quadrilaterals = quadrilaterals if batched else quadrilaterals.squeeze(0)

    return quadrilaterals


def _boxes3d_to_polygons3d(
    xmin: torch.Tensor,
    ymin: torch.Tensor,
    zmin: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    if not xmin.ndim == ymin.ndim == zmin.ndim == width.ndim == height.ndim == depth.ndim == 2:
        raise ValueError("We expect to create a batch of 3D boxes (hexahedrons) in vertices format (B, N, 8, 3)")

    # Front
    # Create (B,N,4,3) with all points in front top left position of boxes
    front_vertices = torch.zeros((xmin.shape[0], xmin.shape[1], 4, 3), device=xmin.device, dtype=xmin.dtype)
    front_vertices[..., 0] = xmin.unsqueeze(-1)
    front_vertices[..., 1] = ymin.unsqueeze(-1)
    front_vertices[..., 2] = zmin.unsqueeze(-1)
    # Shift front-top-right, front-bottom-right, front-bottom-left points to the right coordinates
    front_vertices[..., 1, 0] += width - 1  # Top right
    front_vertices[..., 2, 0] += width - 1  # Bottom right
    front_vertices[..., 2, 1] += height - 1  # Bottom right
    front_vertices[..., 3, 1] += height - 1  # Bottom left

    # Back
    back_vertices = front_vertices.clone()
    back_vertices[..., 2] += depth.unsqueeze(-1) - 1

    polygons3d = torch.cat([front_vertices, back_vertices], dim=-2)
    return polygons3d


class Boxes:
    r"""2D boxes containing N or BxN boxes.

    Args:
        boxes: 2D boxes, shape of :math:`(N, 4, 2)`, :math:`(B, N, 4, 2)` or a list of :math:`(N, 4, 2)`.
            See below for more details.
        raise_if_not_floating_point: flag to control floating point casting behaviour when `boxes` is not a
            floating point tensor. True to raise an error when `boxes` isn't a floating point tensor, False
            to cast to float.
        mode: Representation label reused as the default output mode by :meth:`to_tensor`. The constructor does not
            convert ``boxes``; use :meth:`from_tensor` to import another representation.

    Convention:
        - A box is a quadrilateral of four floating-point ``(x, y)`` vertices, stored as :math:`(N, 4, 2)` or
          :math:`(B, N, 4, 2)` data. Axis-aligned boxes are built in clockwise top-left, top-right,
          bottom-right, bottom-left order, but the vertices stay arbitrary: :meth:`transform_boxes` produces
          rotated quadrilaterals. :meth:`compute_area` sorts vertices by angle about their arithmetic centroid
          before applying the shoelace formula, so it assumes a convex quadrilateral and uses an exclusive area
          convention that can disagree with :meth:`get_boxes_shape` and :meth:`to_mask`.
        - The stored form is inclusive (``'vertices_plus'``): ``width = xmax - xmin + 1``. The exclusive
          ``'xyxy'``, ``'xywh'``, and ``'vertices'`` modes convert to this form in :meth:`from_tensor` and back
          in :meth:`to_tensor`. The constructor converts nothing and stores ``mode`` as a label. For exclusive
          vertex data ``d``, ``Boxes(d, mode='vertices').to_tensor()`` applies export offsets to unconverted data,
          while ``Boxes.from_tensor(d, mode='vertices')`` imports it before export; the round-trip conditions and
          limitations are described next.
        - An axis-aligned box in the documented vertex order whose extent is at least one unit per axis
          round-trips exactly in its own mode when every intermediate conversion result is exactly representable
          in the tensor dtype. A sub-unit extent does not: the inclusive ``- 1`` inverts the stored quadrilateral,
          so normalized ``[0, 1]`` boxes with a sub-unit span on either axis are silently corrupted by the three
          converting modes.
          The ``'xyxy_plus'`` mode is unaffected because its ``+ 1`` cancels the ``- 1``;
          ``'vertices_plus'`` applies no offset at all.
        - :meth:`to_tensor` reduces the stored vertices with ``amin``/``amax``, so every export is an
          axis-aligned bounding box. It is lossy for rotated boxes, and ``to_tensor('vertices_plus')`` is
          therefore not the identity on :attr:`data`.
        - :meth:`get_boxes_shape` returns ``(heights, widths)`` in that order, with exactly the values from
          ``to_tensor('xywh', as_padded_sequence=True)``. For a list-backed object this includes its padding
          entries, which report as 1-by-1 boxes under the inclusive ``+1`` convention even though an ordinary
          :meth:`to_tensor` export trims them.
        - :func:`~kornia.geometry.bbox.infer_bbox_shape`, :func:`~kornia.geometry.bbox.bbox_to_mask`,
          :func:`~kornia.geometry.bbox.validate_bbox`, and :func:`~kornia.geometry.bbox.nms` have no mode argument,
          but the module does not use one convention throughout. :func:`~kornia.geometry.bbox.infer_bbox_shape`
          adds one per axis and
          :func:`~kornia.geometry.bbox.bbox_to_mask` fills through the bottom-right vertex's row and column, so
          both read their input as inclusive: pass them the ``'vertices_plus'`` export rather than
          ``'vertices'``, which they read as one pixel larger per axis. Both consumers require unbatched
          :math:`(N, 4, 2)` input and raise :class:`~kornia.core.exceptions.ShapeError` for a batched
          :math:`(B, N, 4, 2)` export, so index or flatten it before passing it.
          :func:`~kornia.geometry.bbox.validate_bbox` is invariant in exact arithmetic, because its ``+1``
          terms cancel;
          :func:`~kornia.geometry.bbox.nms` computes exclusive areas, and
          :func:`~kornia.geometry.bbox.transform_bbox` converts ``'xywh'`` with the exclusive
          ``xmax = xmin + width``.
        - With ``validate_boxes=True``, the ``'xy*'`` modes reject non-positive extents measured in that mode's
          convention.
        - The constructor rejects an integer tensor unless ``raise_if_not_floating_point=False``. A list input is
          padded into a tensor of its *first* element's dtype before that check, so a mixed-dtype list is accepted
          or rejected by its first box alone and the remaining boxes are cast to that dtype. For a single tensor,
          :meth:`from_tensor` silently casts integer input to ``float32``. For a list, it converts each element
          independently and then pads into the first converted element's dtype, recasting the remaining elements.
        - :meth:`merge` concatenates boxes along the box axis and repacks list-backed batch rows so their padding
          remains at the end, while :meth:`index_put` replaces selected coordinates. Both methods are non-mutating
          by default.

    .. warning::
        The inclusive ``+1`` arithmetic differs from torchvision, COCO, and albumentations and is tracked as a
        coordinated repair in `#3934 <https://github.com/kornia/kornia/issues/3934>`_. The sub-unit conversion
        corruption is `#4061 <https://github.com/kornia/kornia/issues/4061>`_, the cross-module export trap is
        `#4009 <https://github.com/kornia/kornia/issues/4009>`_, the exclusive :meth:`compute_area` convention is
        `#4010 <https://github.com/kornia/kornia/issues/4010>`_, and the exclusive
        :func:`~kornia.geometry.bbox.nms` convention is
        `#4008 <https://github.com/kornia/kornia/issues/4008>`_. The differing ``width``/``height`` argument order
        of :func:`~kornia.geometry.bbox.bbox_to_mask` and :meth:`to_mask` is tracked in
        `#4014 <https://github.com/kornia/kornia/issues/4014>`_. The integer-input policy split between the
        constructor and :meth:`from_tensor` is
        `#4012 <https://github.com/kornia/kornia/issues/4012>`_. With ``validate_boxes=True``, vertex modes remain
        unvalidated, and ``'vertices'`` also subtracts one from fixed vertex positions, potentially deforming the
        input rather than rejecting it; this is tracked in `#4177 <https://github.com/kornia/kornia/issues/4177>`_.
        The unimplemented ``trim``, ``translate(method='fast')``, and tuple-bound ``clamp`` paths are tracked in
        `#4017 <https://github.com/kornia/kornia/issues/4017>`_. The pad, unpad, and clamp operations fail for
        unbatched containers even though the class accepts :math:`(N, 4, 2)` data; this is tracked in
        `#4244 <https://github.com/kornia/kornia/issues/4244>`_.

    """

    def __init__(
        self,
        boxes: torch.Tensor | list[torch.Tensor],
        raise_if_not_floating_point: bool = True,
        mode: str = "vertices_plus",
    ) -> None:
        self._N: Optional[list[int]] = None

        if isinstance(boxes, list):
            boxes, self._N = _merge_box_list(boxes)

        if not isinstance(boxes, torch.Tensor):
            raise TypeError(f"Input boxes is not a Tensor. Got: {type(boxes)}.")

        if not boxes.is_floating_point():
            if raise_if_not_floating_point:
                raise ValueError(f"Coordinates must be in floating point. Got {boxes.dtype}")

            boxes = boxes.float()

        if len(boxes.shape) == 0:
            boxes = boxes.reshape((-1, 4))

        if not (3 <= boxes.ndim <= 4 and boxes.shape[-2:] == (4, 2)):
            raise ValueError(f"Boxes shape must be (N, 4, 2) or (B, N, 4, 2). Got {boxes.shape}.")

        self._is_batched = False if boxes.ndim == 3 else True

        self._data = boxes
        self._mode = mode

    def __getitem__(self, key: slice | int | torch.Tensor) -> Boxes:
        new_box = type(self)(self._data[key], False)
        new_box._mode = self._mode
        return new_box

    def __setitem__(self, key: slice | int | torch.Tensor, value: Boxes) -> Boxes:
        self._data[key] = value._data
        return self

    @property
    def shape(self) -> tuple[int, ...] | Size:
        """Return the tensor shape used to store the boxes.

        Returns:
            Shape of :attr:`data`. For unbatched boxes this is usually
            :math:`(N, 4, 2)`, where :math:`N` is the number of boxes, ``4``
            is the number of corner vertices, and ``2`` stores ``(x, y)``.
            For batched boxes the shape is usually :math:`(B, N, 4, 2)`, where
            :math:`B` is the batch size.
        """
        return self.data.shape

    def get_boxes_shape(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute boxes heights and widths.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        Returns:
            - Boxes heights, shape of :math:`(N,)` or :math:`(B,N)`.
            - Boxes widths, shape of :math:`(N,)` or :math:`(B,N)`.

        Example:
            >>> boxes_xyxy = torch.tensor([[[1,1,2,2],[1,1,3,2]]])
            >>> boxes = Boxes.from_tensor(boxes_xyxy)
            >>> boxes.get_boxes_shape()
            (tensor([[1., 1.]]), tensor([[1., 2.]]))

        """
        boxes_xywh = cast(torch.Tensor, self.to_tensor("xywh", as_padded_sequence=True))
        widths, heights = boxes_xywh[..., 2], boxes_xywh[..., 3]
        return heights, widths

    def merge(self, boxes: Boxes, inplace: bool = False) -> Boxes:
        """Merge boxes.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        For batched boxes, if the current instance holds :math:`(B, N, 4, 2)` and
        the incoming boxes holds :math:`(B, M, 4, 2)`, the merge results in
        :math:`(B, N + M, 4, 2)`.

        For unbatched boxes, if the current instance holds :math:`(N, 4, 2)` and
        the incoming boxes holds :math:`(M, 4, 2)`, the merge results in
        :math:`(N + M, 4, 2)`.

        Args:
            boxes: 2D boxes.
            inplace: do transform in-place and return self.

        Note:
            When either input was created from a list, each batch row is repacked
            so that its real boxes precede all trailing padding. The merged object
            keeps the combined per-image padding counts in ``_N``.

        """
        padding: Optional[list[int]] = None
        if self._N is not None or boxes._N is not None:
            if self._data.shape[0] != boxes.data.shape[0]:
                raise ValueError(
                    f"Batch size mismatch. Got {self._data.shape[0]} for self and {boxes.data.shape[0]} for boxes."
                )

            self_padding = self._N if self._N is not None else [0] * self._data.shape[0]
            boxes_padding = boxes._N if boxes._N is not None else [0] * boxes.data.shape[0]
            data = torch.stack(
                [
                    torch.cat(
                        [
                            self._data[i, : self._data.shape[-3] - self_pad],
                            boxes.data[i, : boxes.data.shape[-3] - boxes_pad],
                            self._data[i, self._data.shape[-3] - self_pad :],
                            boxes.data[i, boxes.data.shape[-3] - boxes_pad :],
                        ]
                    )
                    for i, (self_pad, boxes_pad) in enumerate(zip(self_padding, boxes_padding))
                ]
            )
            padding = [self_pad + boxes_pad for self_pad, boxes_pad in zip(self_padding, boxes_padding)]
        else:
            data = torch.cat([self._data, boxes.data], dim=-3)

        if inplace:
            self._data = data
            self._N = padding
            return self

        obj = self.clone()
        obj._data = data
        obj._N = padding
        return obj

    def index_put(
        self,
        indices: tuple[torch.Tensor, ...] | list[torch.Tensor],
        values: torch.Tensor | Boxes,
        inplace: bool = False,
    ) -> Boxes:
        """Write box coordinates at selected tensor indices.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        This mirrors :meth:`torch.Tensor.index_put_` for the internal
        quadrilateral tensor. It is useful when a subset of boxes in a batch
        must be replaced while keeping the :class:`Boxes` wrapper and metadata.

        Args:
            indices: Index tuple or list accepted by ``Tensor.index_put_``.
                The indices address entries in the stored tensor, commonly
                shaped :math:`(B, N, 4, 2)` or :math:`(N, 4, 2)`.
            values: Replacement coordinates. If a :class:`Boxes` object is
                passed, its :attr:`data` tensor is used.
            inplace: If ``True``, update this object and return ``self``. If
                ``False``, clone the current data first and return a new
                :class:`Boxes` instance.

        Returns:
            :class:`Boxes` containing the updated coordinates.
        """
        if inplace:
            _data = self._data
        else:
            _data = self._data.clone()

        if isinstance(values, Boxes):
            _data.index_put_(indices, values.data)
        else:
            _data.index_put_(indices, values)

        if inplace:
            return self

        obj = self.clone()
        obj._data = _data
        return obj

    def pad(self, padding_size: torch.Tensor) -> Boxes:
        """Pad every box in place.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        ``padding_size`` is ordered as ``(left, right, top, bottom)``. Only
        ``left`` and ``top`` change the coordinate origin; this method returns
        ``self`` after adding those two values to every vertex. This operation
        supports only batched :math:`(B, N, 4, 2)` containers.

        Note:
            Padded :class:`~kornia.augmentation.RandomCrop` uses this method
            and therefore requires batched boxes.

        Args:
            padding_size: Per-batch padding in ``(left, right, top, bottom)``
                order, shaped :math:`(B, 4)`.

        """
        if not (len(padding_size.shape) == 2 and padding_size.size(1) == 4):
            raise RuntimeError(f"Expected padding_size as (B, 4). Got {padding_size.shape}.")
        self._data[..., 0] += padding_size[..., None, :1].to(device=self._data.device)  # left padding
        self._data[..., 1] += padding_size[..., None, 2:3].to(device=self._data.device)  # top padding
        return self

    def unpad(self, padding_size: torch.Tensor) -> Boxes:
        """Undo :meth:`pad` in place.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        ``padding_size`` is ordered as ``(left, right, top, bottom)``. Only
        ``left`` and ``top`` change the coordinate origin; this method returns
        ``self`` after subtracting those two values from every vertex. This
        operation supports only batched :math:`(B, N, 4, 2)` containers.

        Args:
            padding_size: Per-batch padding in ``(left, right, top, bottom)``
                order, shaped :math:`(B, 4)`.

        """
        if not (len(padding_size.shape) == 2 and padding_size.size(1) == 4):
            raise RuntimeError(f"Expected padding_size as (B, 4). Got {padding_size.shape}.")
        self._data[..., 0] -= padding_size[..., None, :1].to(device=self._data.device)  # left padding
        self._data[..., 1] -= padding_size[..., None, 2:3].to(device=self._data.device)  # top padding
        return self

    def clamp(
        self,
        topleft: Optional[torch.Tensor | tuple[int, int]] = None,
        botright: Optional[torch.Tensor | tuple[int, int]] = None,
        inplace: bool = False,
    ) -> Boxes:
        """Clamp every box vertex inside per-image coordinate limits.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        Convention:
            Bounds must be tensors with one ``(x, y)`` pair per batch element.
            Every vertex is clamped independently, so a box wholly outside the
            bounds collapses onto the nearest boundary instead of being removed.
            This operation supports only batched :math:`(B, N, 4, 2)` containers.

        Coordinates below ``topleft`` are raised to the lower bound and
        coordinates above ``botright`` are lowered to the upper bound. The
        implementation accepts only tensor bounds with one ``(x, y)`` pair per
        batch element.

        Args:
            topleft: Tensor of shape :math:`(B, 2)` containing the minimum
                ``x`` and ``y`` coordinate allowed for each batch item.
            botright: Tensor of shape :math:`(B, 2)` containing the maximum
                ``x`` and ``y`` coordinate allowed for each batch item.
            inplace: If ``True``, clamp this object in place. Otherwise, return
                a new :class:`Boxes` object with clamped data.

        Returns:
            :class:`Boxes` whose vertex coordinates are restricted to the
            provided bounds.
        """
        if not (isinstance(topleft, torch.Tensor) and isinstance(botright, torch.Tensor)):
            raise NotImplementedError
        if inplace:
            _data = self._data
        else:
            _data = self._data.clone()
        topleft_x = topleft[:, None, :1].repeat(1, _data.size(1), 4)
        _data[..., 0][_data[..., 0] < topleft_x] = topleft_x[_data[..., 0] < topleft_x]

        topleft_y = topleft[:, None, 1:].repeat(1, _data.size(1), 4)
        _data[..., 1][_data[..., 1] < topleft_y] = topleft_y[_data[..., 1] < topleft_y]

        botright_x = botright[:, None, :1].repeat(1, _data.size(1), 4)
        _data[..., 0][_data[..., 0] > botright_x] = botright_x[_data[..., 0] > botright_x]

        botright_y = botright[:, None, 1:].repeat(1, _data.size(1), 4)
        _data[..., 1][_data[..., 1] > botright_y] = botright_y[_data[..., 1] > botright_y]
        if inplace:
            return self

        obj = self.clone()
        obj._data = _data
        return obj

    def trim(self, correspondence_preserve: bool = False, inplace: bool = False) -> Boxes:
        """Raise because trimming padded boxes is not implemented.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        Args:
            correspondence_preserve: Reserved for a future implementation.
            inplace: Reserved for a future implementation.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError

    def filter_boxes_by_area(
        self, min_area: Optional[float] = None, max_area: Optional[float] = None, inplace: bool = False
    ) -> Boxes:
        """Zero boxes whose polygon area is outside the requested range.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        The box area is computed from its four vertices. Boxes smaller than
        ``min_area`` or larger than ``max_area`` are not dropped from the
        tensor; their coordinates are replaced with zeros so the original batch
        and box dimensions stay unchanged. See :meth:`compute_area` for the
        area convention used by the thresholds.

        Args:
            min_area: Optional lower inclusive area threshold. Boxes with area
                below this value are zeroed.
            max_area: Optional upper inclusive area threshold. Boxes with area
                above this value are zeroed.
            inplace: If ``True``, update this object in place. Otherwise,
                return a filtered clone.

        Returns:
            :class:`Boxes` with the same shape as the input container and
            out-of-range boxes replaced by zero coordinates.
        """
        area = self.compute_area()
        if inplace:
            _data = self._data
        else:
            _data = self._data.clone()
        if min_area is not None:
            _data[area < min_area] = 0.0
        if max_area is not None:
            _data[area > max_area] = 0.0
        if inplace:
            return self

        obj = self.clone()
        obj._data = _data
        return obj

    def compute_area(self) -> torch.Tensor:
        """Compute polygon area with the shoelace formula.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        Convention:
            For axis-aligned boxes, the shoelace result over stored inclusive
            vertices is ``(width - 1) * (height - 1)``, while the inclusive
            terms from :meth:`get_boxes_shape` multiply to ``width * height``.
            Rotated or otherwise non-axis-aligned quadrilaterals use their
            polygon area instead.

        .. warning::
            The differing area conventions are tracked in
            `#4010 <https://github.com/kornia/kornia/issues/4010>`_.

        Returns:
            Area for each box, shaped :math:`(N,)` or :math:`(B, N)`.
        """
        coords = self._data.view((-1, 4, 2)) if self._data.ndim == 4 else self._data
        # calculate centroid of the box
        centroid = coords.mean(dim=1, keepdim=True)
        # calculate the angle from centroid to each corner
        angles = torch.atan2(coords[..., 1] - centroid[..., 1], coords[..., 0] - centroid[..., 0])
        # sort the corners by angle to get an order for shoelace formula
        _, clockwise_indices = torch.sort(angles, dim=1, descending=True)
        # gather the corners in the new order
        ordered_corners = torch.gather(coords, 1, clockwise_indices.unsqueeze(-1).expand(-1, -1, 2))
        x, y = ordered_corners[..., 0], ordered_corners[..., 1]
        # Gaussian/Shoelace formula https://en.wikipedia.org/wiki/Shoelace_formula
        area = 0.5 * torch.abs(torch.sum((x * torch.roll(y, 1, 1)) - (y * torch.roll(x, 1, 1)), dim=1))
        return area.view(self._data.shape[:2]) if self._data.ndim == 4 else area

    @classmethod
    def from_tensor(
        cls, boxes: torch.Tensor | list[torch.Tensor], mode: str = "xyxy", validate_boxes: bool = True
    ) -> Boxes:
        r"""Create :class:`Boxes` from boxes stored in another format.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        Args:
            boxes: 2D boxes, shape of :math:`(N, 4)`, :math:`(B, N, 4)`, :math:`(N, 4, 2)` or
                :math:`(B, N, 4, 2)`, or a list of :math:`(N, 4)` or :math:`(N, 4, 2)` tensors matching ``mode``.
            mode: The format in which the boxes are provided:

                * 'xyxy': ``xmin, ymin, xmax, ymax`` with exclusive extent. With shape :math:`(N, 4)`,
                  :math:`(B, N, 4)`.
                * 'xyxy_plus': ``xmin, ymin, xmax, ymax`` with inclusive extent. With shape :math:`(N, 4)`,
                  :math:`(B, N, 4)`.
                * 'xywh': ``xmin, ymin, width, height`` with exclusive extent. With shape :math:`(N, 4)`,
                  :math:`(B, N, 4)`.
                * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                  *top-left, top-right, bottom-right, bottom-left*. Vertices coordinates are in (x,y) order. This is
                  the exclusive input form. With shape :math:`(N, 4, 2)`, :math:`(B, N, 4, 2)`.
                * 'vertices_plus': the inclusive stored vertex form. With shape :math:`(N, 4, 2)`,
                  :math:`(B, N, 4, 2)`.

            validate_boxes: Check extents for the ``'xy*'`` modes in each mode's convention. This flag has no
                validation effect for vertex modes; see the warning on :class:`~kornia.geometry.boxes.Boxes`.

        Returns:
            :class:`Boxes` containing the converted inclusive vertex representation.

        Examples:
            >>> boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
            >>> boxes = Boxes.from_tensor(boxes_xyxy, mode='xyxy')
            >>> boxes.data  # (2, 4, 2)
            tensor([[[0., 3.],
                     [0., 3.],
                     [0., 3.],
                     [0., 3.]],
            <BLANKLINE>
                    [[5., 1.],
                     [7., 1.],
                     [7., 3.],
                     [5., 3.]]])

        """
        quadrilaterals: torch.Tensor | list[torch.Tensor]
        if isinstance(boxes, torch.Tensor):
            quadrilaterals = _boxes_to_quadrilaterals(boxes, mode=mode, validate_boxes=validate_boxes)
        else:
            quadrilaterals = [_boxes_to_quadrilaterals(box, mode, validate_boxes) for box in boxes]

        return cls(quadrilaterals, False, mode)

    def to_tensor(
        self, mode: Optional[str] = None, as_padded_sequence: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        r"""Cast :class:`Boxes` to a tensor.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        ``mode`` selects the output representation. ``'xyxy'``, ``'xywh'``, and ``'vertices'`` are exclusive
        exports; the ``'_plus'`` variants are inclusive. Every mode exports the axis-aligned bounds of the
        stored vertices, reduced with ``amin``/``amax``, so the export is lossy for rotated boxes:
        ``to_tensor('vertices_plus')`` does not return :attr:`data` unchanged after a :meth:`transform_boxes`
        call that rotates, shears, or otherwise reorders the vertices. A quarter turn keeps the box
        axis-aligned and still yields an export that differs from :attr:`data`, because the reduction
        re-canonicalizes the vertex order.

        Args:
            mode: the output box format, or ``None`` to reuse :attr:`mode`. That attribute depends on the
                construction path: the constructor defaults to ``'vertices_plus'``, while :meth:`from_tensor`
                records the mode its input was given in. It could be:

                * 'xyxy': ``xmin, ymin, xmax, ymax`` with exclusive extent.
                * 'xyxy_plus': ``xmin, ymin, xmax, ymax`` with inclusive extent.
                * 'xywh': ``xmin, ymin, width, height`` with exclusive extent.
                * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                  *top-left, top-right, bottom-right, bottom-left*. Vertices coordinates are in (x,y) order. This is
                  the exclusive export form.
                * 'vertices_plus': the inclusive stored vertex form.
            as_padded_sequence: If this object was created from a list, return its padded tensor rather than a list
                of tensors trimmed to their original lengths. The padded values follow the selected output mode.
                Indexing with ``[]`` drops the list metadata, so a sliced object always returns the padded tensor;
                see `#4179 <https://github.com/kornia/kornia/issues/4179>`_.

        Returns:
            Boxes tensor in the ``mode`` format, or a list of tensors when the object was created from a list and
            ``as_padded_sequence=False``. The tensor shape depends on the ``mode`` value:

                * 'vertices' or 'vertices_plus': :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
                * Any other value: :math:`(N, 4)` or :math:`(B, N, 4)`.

        Examples:
            >>> boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
            >>> boxes = Boxes.from_tensor(boxes_xyxy)
            >>> assert (boxes_xyxy == boxes.to_tensor(mode='xyxy')).all()

        """
        batched_boxes = self._data if self._is_batched else self._data.unsqueeze(0)

        boxes: torch.Tensor | list[torch.Tensor]

        # Create boxes in xyxy_plus format.
        boxes = torch.stack([batched_boxes.amin(dim=-2), batched_boxes.amax(dim=-2)], dim=-2).view(
            batched_boxes.shape[0], batched_boxes.shape[1], 4
        )

        if mode is None:
            mode = self.mode

        mode = mode.lower()

        if mode in ("xyxy", "xyxy_plus"):
            pass
        elif mode in ("xywh", "vertices", "vertices_plus"):
            height, width = boxes[..., 3] - boxes[..., 1] + 1, boxes[..., 2] - boxes[..., 0] + 1
            boxes[..., 2] = width
            boxes[..., 3] = height
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode in ("xyxy", "vertices"):
            offset = torch.as_tensor([0, 0, 1, 1], device=boxes.device, dtype=boxes.dtype)
            boxes = boxes + offset

        if mode.startswith("vertices"):
            boxes = _boxes_to_polygons(boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3])

        if self._N is not None and not as_padded_sequence:
            boxes = [torch.nn.functional.pad(o, (len(o.shape) - 1) * [0, 0] + [0, -n]) for o, n in zip(boxes, self._N)]
        else:
            boxes = boxes if self._is_batched else boxes.squeeze(0)
        return boxes

    def to_mask(self, height: int, width: int) -> torch.Tensor:
        """Convert 2D boxes to masks. Covered area is 1 and the remaining is 0.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        Convention:
            The size is ``(height, width)`` and the mask is :math:`(N, height, width)` or
            :math:`(B, N, height, width)` in the box dtype; :func:`~kornia.geometry.bbox.bbox_to_mask` takes
            ``(width, height)`` for the same result. The boxes are exported as exclusive ``'xyxy'`` bounds (the
            axis-aligned bounding box of a rotated quadrilateral), clamped to ``[0, width]`` and ``[0, height]``,
            rounded to the nearest integer, and filled over the half-open ranges ``[xmin, xmax)`` and
            ``[ymin, ymax)``, so a box entirely outside the image fills nothing and a fractional box can fill a
            different area than :func:`~kornia.geometry.bbox.bbox_to_mask` gives for the same vertices. A
            list-backed object also fills a mask channel for each padding entry, whose zero row exports as a
            one-pixel box at the origin. The loop taken on CPU and MPS and the vectorized path taken on CUDA and
            under graph capture produce the same mask. A box tensor that requires grad is rejected with
            ``RuntimeError``.

        .. warning::
            The argument-order split with :func:`~kornia.geometry.bbox.bbox_to_mask` is tracked in
            `#4014 <https://github.com/kornia/kornia/issues/4014>`_ and the rounding split in
            `#4015 <https://github.com/kornia/kornia/issues/4015>`_. The padding-entry pixel is documented as it is and
            pinned by ``test_wart_to_mask_fills_the_origin_pixel_for_list_padding_rows`` in
            ``tests/geometry/test_boxes.py``.

        Args:
            height: height of the masked image/images.
            width: width of the masked image/images.

        Returns:
            the output mask tensor, shape of :math:`(N, height, width)` or :math:`(B, N, height, width)` and dtype of
            :func:`Boxes.dtype` (it can be any floating point dtype).

        Note:
            It is currently non-differentiable.

        Examples:
            >>> boxes = Boxes(torch.tensor([[  # Equivalent to Boxes.from_tensor([[1, 1, 4, 3]], mode='xyxy_plus')
            ...        [1., 1.],
            ...        [4., 1.],
            ...        [4., 3.],
            ...        [1., 3.],
            ...   ]]))  # 1x4x2
            >>> boxes.to_mask(5, 5)
            tensor([[[0., 0., 0., 0., 0.],
                     [0., 1., 1., 1., 1.],
                     [0., 1., 1., 1., 1.],
                     [0., 1., 1., 1., 1.],
                     [0., 0., 0., 0., 0.]]])

        """
        if self._data.requires_grad:
            raise RuntimeError(
                "Boxes.to_tensor isn't differentiable. Please, create boxes from tensors with `requires_grad=False`."
            )

        is_batched = self._is_batched
        dtype = self.dtype
        device = self.device

        # -----------------
        # CPU Hotpath (loop)
        # -----------------
        # The loop slices with data-dependent bounds, which graph capture cannot do; export takes the vectorized path.
        if device.type != "cuda" and not is_exporting():
            if self._is_batched:  # (B, N, 4, 2)
                mask = torch.zeros(
                    (self._data.shape[0], self._data.shape[1], height, width), dtype=self.dtype, device=self.device
                )
            else:  # (N, 4, 2)
                mask = torch.zeros((self._data.shape[0], height, width), dtype=self.dtype, device=self.device)

            # Boxes coordinates can be outside the image size after transforms. Clamp values to the image size
            clipped_boxes_xyxy = cast(torch.Tensor, self.to_tensor("xyxy", as_padded_sequence=True))
            clipped_boxes_xyxy[..., ::2].clamp_(0, width)
            clipped_boxes_xyxy[..., 1::2].clamp_(0, height)

            # Reshape mask to (BxN, H, W) and boxes to (BxN, 4) to iterate over all of them.
            # Cast boxes coordinates to be integer to use them as indexes. Use round to handle decimal values.
            for mask_channel, box_xyxy in zip(
                mask.view(-1, height, width), clipped_boxes_xyxy.view(-1, 4).round().int()
            ):
                # Mask channel dimensions: (height, width)
                mask_channel[box_xyxy[1] : box_xyxy[3], box_xyxy[0] : box_xyxy[2]] = 1

            return mask

        # -----------------
        # GPU Hotpath (vectorized)
        # -----------------
        out_shape: Tuple[int, ...]
        if is_batched:
            out_shape = (self.shape[0], self.shape[1], height, width)
        else:
            out_shape = (self.shape[0], height, width)

        clipped_boxes_xyxy = cast(torch.Tensor, self.to_tensor("xyxy", as_padded_sequence=True))
        clipped_boxes_xyxy[..., ::2].clamp_(0, width)
        clipped_boxes_xyxy[..., 1::2].clamp_(0, height)

        xyxy = clipped_boxes_xyxy.view(-1, 4).round().long()

        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        x1 = x1.clamp(0, width)
        x2 = x2.clamp(0, width)
        y1 = y1.clamp(0, height)
        y2 = y2.clamp(0, height)

        ys = torch.arange(height, device=device)
        xs = torch.arange(width, device=device)

        y_mask = (ys[None, :] >= y1[:, None]) & (ys[None, :] < y2[:, None])
        x_mask = (xs[None, :] >= x1[:, None]) & (xs[None, :] < x2[:, None])

        masks = (y_mask.unsqueeze(2) & x_mask.unsqueeze(1)).to(dtype)
        return masks.view(*out_shape)

    def transform_boxes(self, M: torch.Tensor, inplace: bool = False) -> Boxes:
        r"""Apply a transformation matrix to the 2D boxes.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        Args:
            M: The transformation matrix to be applied, shape of :math:`(3, 3)` or :math:`(B, 3, 3)`.
            inplace: do transform in-place and return self.

        Returns:
            The transformed boxes.

        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (3, 3):
            raise ValueError(f"The transformation matrix shape must be (3, 3) or (B, 3, 3). Got {M.shape}.")

        transformed_boxes = _transform_boxes(self._data, M)
        if inplace:
            self._data = transformed_boxes
            return self

        obj = self.clone()
        obj._data = transformed_boxes
        return obj

    def transform_boxes_(self, M: torch.Tensor) -> Boxes:
        """Apply :meth:`transform_boxes` in place and return ``self``.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        Convention:
            The in-place operation rebinds this object's internal tensor to a
            transformed result for nonempty containers. A tensor reference
            obtained from :attr:`data` before that call therefore remains
            unchanged and no longer aliases the container's data. Empty
            containers retain their original tensor reference.

        Returns:
            This :class:`Boxes` object after the transformation.
        """
        return self.transform_boxes(M, inplace=True)

    def translate(self, size: torch.Tensor, method: str = "warp", inplace: bool = False) -> Boxes:
        """Translate boxes by the provided size.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

        ``size`` supplies one ``(x, y)`` translation per batch item. Only the
        ``"warp"`` method is implemented; ``"fast"`` raises
        :class:`NotImplementedError`.

        Args:
            size: translate size for x, y direction, shape of :math:`(B, 2)`.
            method: "warp" or "fast".
            inplace: do transform in-place and return self.

        Returns:
            The transformed boxes.

        """
        if method == "fast":
            raise NotImplementedError
        elif method == "warp":
            pass
        else:
            raise NotImplementedError

        M: torch.Tensor = eye_like(3, size)
        M[:, :2, 2] = size
        return self.transform_boxes(M, inplace=inplace)

    @property
    def data(self) -> torch.Tensor:
        """Return the raw quadrilateral coordinate tensor.

        Returns:
            Tensor storing four vertices per box in ``(x, y)`` order. The
            common shapes are :math:`(N, 4, 2)` for unbatched boxes and
            :math:`(B, N, 4, 2)` for batched boxes, where :math:`B` is batch
            size and :math:`N` is the number of boxes.
        """
        return self._data

    @property
    def mode(self) -> str:
        """Return the box format remembered by this container.

        Returns:
            Mode string used as the default by :meth:`to_tensor`, such as
            ``"xyxy"``, ``"xywh"``, ``"vertices"``, or their ``"_plus"``
            variants.
        """
        return self._mode

    @property
    def device(self) -> torch.device:
        """Returns boxes device."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Returns boxes dtype."""
        return self._data.dtype

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Boxes:
        """Like :func:`torch.nn.Module.to()` method."""
        # In torchscript, dtype is a int and not a class. https://github.com/pytorch/pytorch/issues/51941
        if dtype is not None and not _is_floating_point_dtype(dtype):
            raise ValueError("Boxes must be in floating point")
        self._data = self._data.to(device=device, dtype=dtype)
        return self

    def clone(self) -> Boxes:
        """Create an independent copy of the box container.

        Returns:
            New :class:`Boxes` object with cloned tensor storage. Metadata such
            as the current mode, original list lengths, and batched flag is
            preserved.
        """
        obj = type(self)(self._data.clone(), False)
        obj._mode = self._mode
        obj._N = self._N
        obj._is_batched = self._is_batched
        return obj

    def type(self, dtype: torch.dtype) -> Boxes:
        """Cast the stored box coordinates to a new dtype.

        Args:
            dtype: Target floating-point dtype for the coordinate tensor.

        Returns:
            ``self`` after converting :attr:`data` in place.
        """
        self._data = self._data.type(dtype)
        return self


class VideoBoxes(Boxes):
    r"""2D boxes with an explicit temporal channel for video sequences.

    Accepts and returns box corners for a batch of videos as
    :math:`(B, T, N, 4, 2)` in ``vertices_plus`` mode. Internally the corners
    are stored flattened as :math:`(B \cdot T, N, 4, 2)`.
    :class:`~kornia.augmentation.AugmentationSequential` uses this wrapper when
    the pipeline contains a video sequential so that ``to_tensor`` restores the
    temporal axis after geometric transforms.

    See the Convention block on :class:`~kornia.geometry.boxes.Boxes`.

    Convention:
        - :meth:`from_tensor` stores the :math:`(B, T, N, 4, 2)` input unchanged as batched
          :math:`(B \cdot T, N, 4, 2)` ``'vertices_plus'`` data; there is no mode argument, no conversion and no
          validation, and integer input is cast to ``float32``. Any other rank or last dimensions, and list input,
          raise ``ValueError``.
        - :meth:`to_tensor` accepts every :class:`Boxes` export mode and restores the temporal axis, so
          ``to_tensor('xyxy')`` is :math:`(B, T, N, 4)`. Its default is the stored ``'vertices_plus'`` mode.
        - A transformation matrix must carry one entry per flattened frame, :math:`(B \cdot T, 3, 3)`; a
          :math:`(3, 3)` matrix raises ``ValueError``. Methods that copy through :meth:`clone`, among them
          :meth:`transform_boxes`, :meth:`translate`, :meth:`clamp`, :meth:`filter_boxes_by_area`, :meth:`pad`,
          :meth:`merge` and :meth:`to`, return a :class:`VideoBoxes` with the same
          :attr:`temporal_channel_size`.

    .. warning::
        :meth:`get_boxes_shape` and :meth:`to_mask` raise ``TypeError`` because the :meth:`to_tensor` override
        does not accept the ``as_padded_sequence`` keyword they pass, and indexing returns a wrapper without
        :attr:`temporal_channel_size`, so its :meth:`to_tensor` raises ``AttributeError``. Both are documented as
        they are and pinned by ``test_wart_inherited_methods_break_on_the_temporal_wrapper`` in
        ``tests/geometry/test_boxes.py``. The inert ``validate_boxes`` flag is part of
        `#4177 <https://github.com/kornia/kornia/issues/4177>`_.

    Attributes:
        temporal_channel_size: Number of frames :math:`T` stored with the boxes.

    """

    temporal_channel_size: int

    @classmethod
    def from_tensor(  # type: ignore[override]
        cls, boxes: torch.Tensor | list[torch.Tensor], validate_boxes: bool = True
    ) -> VideoBoxes:
        r"""Create :class:`VideoBoxes` from a video box tensor.

        Args:
            boxes: Box corners with shape :math:`(B, T, N, 4, 2)` in
                ``vertices_plus`` order (top-left, top-right, bottom-right,
                bottom-left), stored unchanged; integer input is cast to
                ``float32``. Lists of tensors are not supported yet.
            validate_boxes: Forwarded to ``_boxes_to_quadrilaterals``. The
                ``vertices_plus`` path used here builds corners directly and
                performs no size check, so this flag currently has no effect.

        Returns:
            :class:`VideoBoxes` with :attr:`temporal_channel_size` set to
            ``boxes.size(1)``.

        Raises:
            ValueError: If ``boxes`` is a list or does not have shape
                :math:`(B, T, N, 4, 2)`.
        """
        if isinstance(boxes, (list,)) or (boxes.dim() != 5 or boxes.shape[-2:] != torch.Size([4, 2])):
            raise ValueError("Input box type is not yet supported. Please input an `BxTxNx4x2` tensor directly.")

        temporal_channel_size = boxes.size(1)

        quadrilaterals = _boxes_to_quadrilaterals(
            boxes.view(boxes.size(0) * boxes.size(1), -1, boxes.size(3), boxes.size(4)),
            mode="vertices_plus",
            validate_boxes=validate_boxes,
        )
        out = cls(quadrilaterals, False, "vertices_plus")
        out.temporal_channel_size = temporal_channel_size
        return out

    def to_tensor(self, mode: Optional[str] = None) -> torch.Tensor | list[torch.Tensor]:  # type: ignore[override]
        r"""Cast :class:`VideoBoxes` to a tensor with the temporal axis restored.

        The ``as_padded_sequence`` keyword of :meth:`Boxes.to_tensor` is not
        accepted; see the warning on the class.

        Args:
            mode: Output box format forwarded to :meth:`Boxes.to_tensor`. When
                ``None``, uses the stored mode (``vertices_plus`` by default).

        Returns:
            Tensor shaped :math:`(B, T, \ldots)` where :math:`T` is
            :attr:`temporal_channel_size`.
        """
        out = super().to_tensor(mode, as_padded_sequence=False)
        if isinstance(out, torch.Tensor):
            return out.view(-1, self.temporal_channel_size, *out.shape[1:])
        # If returns a list of boxes.
        return [_out.view(-1, self.temporal_channel_size, *_out.shape[1:]) for _out in out]

    def clone(self) -> VideoBoxes:
        """Create an independent copy of the video box container.

        Returns:
            New :class:`VideoBoxes` with cloned tensor storage and the same
            :attr:`temporal_channel_size`, mode, and batch metadata.
        """
        obj = type(self)(self._data.clone(), False)
        obj._mode = self._mode
        obj._N = self._N
        obj._is_batched = self._is_batched
        obj.temporal_channel_size = self.temporal_channel_size
        return obj


class Boxes3D:
    r"""3D boxes containing N or BxN boxes.

    Args:
        boxes: 3D boxes, shape of :math:`(N,8,3)` or :math:`(B,N,8,3)`. See below for more details.
        raise_if_not_floating_point: flag to control floating point casting behaviour when `boxes` is not a floating
            point tensor. True to raise an error when `boxes` isn't a floating point tensor, False to cast to float.
        mode: Representation label reported by :attr:`mode`. The constructor does not convert ``boxes`` and
            :meth:`to_tensor` does not consult the label; use :meth:`from_tensor` to import another representation.

    Convention:
        - A box is a `hexahedron <https://en.wikipedia.org/wiki/Hexahedron>`_ of eight floating-point
          ``(x, y, z)`` vertices, stored as :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)` data, in the order
          front-top-left, front-top-right, front-bottom-right, front-bottom-left, then the same four back
          vertices. The vertices stay arbitrary after :meth:`transform_boxes`.
        - The stored form is inclusive: ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and
          ``depth = zmax - zmin + 1``. The exclusive ``'xyzxyz'`` and ``'xyzwhd'`` modes subtract one from the
          max corner in :meth:`from_tensor` and add it back in :meth:`to_tensor`; ``'xyzxyz_plus'`` is stored as
          given. :meth:`from_tensor` accepts only :math:`(N, 6)` or :math:`(B, N, 6)` input in those three modes;
          the vertex modes ``'vertices'`` (exclusive) and ``'vertices_plus'`` (the stored form) exist for
          :meth:`to_tensor` only. Mode strings are lowercased before use.
        - :meth:`to_tensor` reduces the stored vertices with ``amin``/``amax``, so every export is an axis-aligned
          bounding box and ``to_tensor('vertices_plus')`` is not the identity on :attr:`data` for a rotated box.
          Its default mode is ``'xyzxyz'`` whatever the stored label, unlike :meth:`Boxes.to_tensor`, which
          defaults to the stored mode.
        - :meth:`get_boxes_shape` returns ``(depths, heights, widths)`` in that order, in the inclusive terms.
        - :func:`~kornia.geometry.bbox.validate_bbox3d`, :func:`~kornia.geometry.bbox.infer_bbox_shape3d` and
          :func:`~kornia.geometry.bbox.bbox_to_mask3d` have no mode argument and read their input as inclusive:
          pass them the ``'vertices_plus'`` export, never ``'vertices'``, which they read as one larger per axis.
          They also require unbatched :math:`(N, 8, 3)` input; see their warnings.
        - With ``validate_boxes=True``, :meth:`from_tensor` rejects extents that are not positive in the given
          mode's convention, so ``xmax == xmin`` is rejected in ``'xyzxyz'`` and accepted in ``'xyzxyz_plus'``.
        - The constructor rejects an integer tensor unless ``raise_if_not_floating_point=False``;
          :meth:`from_tensor` silently casts integer input to ``float32``.
        - :meth:`transform_boxes` leaves the source unchanged and returns a new object labelled
          ``'xyzxyz_plus'``; :meth:`transform_boxes_` rebinds the internal tensor of ``self`` and keeps the label.

    .. warning::
        The inclusive ``+1`` arithmetic differs from torchvision, COCO, and albumentations and is tracked as a
        coordinated repair in `#3934 <https://github.com/kornia/kornia/issues/3934>`_. The exclusive-export trap is
        `#4009 <https://github.com/kornia/kornia/issues/4009>`_, the integer-input policy split is
        `#4012 <https://github.com/kornia/kornia/issues/4012>`_, the validator contract split with
        :func:`~kornia.geometry.bbox.validate_bbox` is `#4013 <https://github.com/kornia/kornia/issues/4013>`_, and
        boxes built by :func:`~kornia.geometry.bbox.bbox_generator3d` measure one larger than requested,
        `#4018 <https://github.com/kornia/kornia/issues/4018>`_. The :meth:`to_tensor` default-mode split with
        :class:`Boxes` is documented as it is and pinned by
        ``test_wart_to_tensor_default_mode_ignores_the_stored_label`` in
        ``tests/geometry/test_boxes.py``. :meth:`to_mask` rejects boxes that require grad even though
        :meth:`to_tensor` is differentiable; see the note on :meth:`to_tensor`.

    """

    def __init__(
        self, boxes: torch.Tensor, raise_if_not_floating_point: bool = True, mode: str = "xyzxyz_plus"
    ) -> None:
        if not isinstance(boxes, torch.Tensor):
            raise TypeError(f"Input boxes is not a Tensor. Got: {type(boxes)}.")

        if not boxes.is_floating_point():
            if raise_if_not_floating_point:
                raise ValueError(f"Coordinates must be in floating point. Got {boxes.dtype}.")

            boxes = boxes.float()

        if len(boxes.shape) == 0:
            boxes = boxes.reshape((-1, 6))

        if not (3 <= boxes.ndim <= 4 and boxes.shape[-2:] == (8, 3)):
            raise ValueError(f"3D bbox shape must be (N, 8, 3) or (B, N, 8, 3). Got {boxes.shape}.")

        self._is_batched = False if boxes.ndim == 3 else True

        self._data = boxes
        self._mode = mode

    def __getitem__(self, key: slice | int | torch.Tensor) -> Boxes3D:
        new_box = Boxes3D(self._data[key], False, mode="xyzxyz_plus")
        new_box._mode = self._mode
        return new_box

    def __setitem__(self, key: slice | int | torch.Tensor, value: Boxes3D) -> Boxes3D:
        self._data[key] = value._data
        return self

    @property
    def shape(self) -> tuple[int, ...] | Size:
        """Return the tensor shape used to store 3D boxes.

        Returns:
            Shape of :attr:`data`. For unbatched boxes this is usually
            :math:`(N, 8, 3)`, where :math:`N` is the number of boxes, ``8`` is
            the number of cuboid corners, and ``3`` stores ``(x, y, z)``. For
            batched boxes the shape is usually :math:`(B, N, 8, 3)`.
        """
        return self.data.shape

    def get_boxes_shape(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute boxes depths, heights and widths.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes3D`.

        Returns:
            - Boxes depths, shape of :math:`(N,)` or :math:`(B,N)`.
            - Boxes heights, shape of :math:`(N,)` or :math:`(B,N)`.
            - Boxes widths, shape of :math:`(N,)` or :math:`(B,N)`.

        Example:
            >>> boxes_xyzxyz = torch.tensor([[ 0,  1,  2, 10, 21, 32], [3, 4, 5, 43, 54, 65]])
            >>> boxes3d = Boxes3D.from_tensor(boxes_xyzxyz)
            >>> boxes3d.get_boxes_shape()
            (tensor([30., 60.]), tensor([20., 50.]), tensor([10., 40.]))

        """
        boxes_xyzwhd = self.to_tensor(mode="xyzwhd")
        widths, heights, depths = boxes_xyzwhd[..., 3], boxes_xyzwhd[..., 4], boxes_xyzwhd[..., 5]
        return depths, heights, widths

    @classmethod
    def from_tensor(cls, boxes: torch.Tensor, mode: str = "xyzxyz", validate_boxes: bool = True) -> Boxes3D:
        r"""Create :class:`Boxes3D` from 3D boxes stored in another format.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes3D`.

        Args:
            boxes: 3D boxes, shape of :math:`(N,6)` or :math:`(B,N,6)`; integer input is cast to ``float32``.
            mode: The format in which the 3D boxes are provided, matched case-insensitively.

                * 'xyzxyz': boxes are assumed to be in the format ``xmin, ymin, zmin, xmax, ymax, zmax`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'xyzxyz_plus': similar to 'xyzxyz' mode but where box width, length and depth are defined as
                  ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and ``depth = zmax - zmin + 1``.
                * 'xyzwhd': boxes are assumed to be in the format ``xmin, ymin, zmin, width, height, depth`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.

            validate_boxes: reject boxes whose width, height or depth is not positive when measured in the given
                mode's convention, so ``xmax == xmin`` is rejected in ``'xyzxyz'`` and accepted in ``'xyzxyz_plus'``.

        Returns:
            :class:`Boxes3D` containing the converted inclusive vertex representation, labelled with ``mode``.

        Examples:
            >>> boxes_xyzxyz = torch.as_tensor([[0, 3, 6, 1, 4, 8], [5, 1, 3, 8, 4, 9]])
            >>> boxes = Boxes3D.from_tensor(boxes_xyzxyz, mode='xyzxyz')
            >>> boxes.data  # (2, 8, 3)
            tensor([[[0., 3., 6.],
                     [0., 3., 6.],
                     [0., 3., 6.],
                     [0., 3., 6.],
                     [0., 3., 7.],
                     [0., 3., 7.],
                     [0., 3., 7.],
                     [0., 3., 7.]],
            <BLANKLINE>
                    [[5., 1., 3.],
                     [7., 1., 3.],
                     [7., 3., 3.],
                     [5., 3., 3.],
                     [5., 1., 8.],
                     [7., 1., 8.],
                     [7., 3., 8.],
                     [5., 3., 8.]]])

        """
        if not (2 <= boxes.ndim <= 3 and boxes.shape[-1] == 6):
            raise ValueError(f"BBox shape must be (N, 6) or (B, N, 6). Got {boxes.shape}.")

        batched = boxes.ndim == 3
        boxes = boxes if batched else boxes.unsqueeze(0)
        boxes = boxes if boxes.is_floating_point() else boxes.float()

        xmin, ymin, zmin = boxes[..., 0], boxes[..., 1], boxes[..., 2]
        mode = mode.lower()
        if mode == "xyzxyz":
            width = boxes[..., 3] - boxes[..., 0]
            height = boxes[..., 4] - boxes[..., 1]
            depth = boxes[..., 5] - boxes[..., 2]
        elif mode == "xyzxyz_plus":
            width = boxes[..., 3] - boxes[..., 0] + 1
            height = boxes[..., 4] - boxes[..., 1] + 1
            depth = boxes[..., 5] - boxes[..., 2] + 1
        elif mode == "xyzwhd":
            width, height, depth = boxes[..., 3], boxes[..., 4], boxes[..., 5]
        else:
            raise ValueError(f"Unknown mode {mode}")

        # Value validation reads the data, which graph capture cannot do; skip it under export.
        if validate_boxes and not is_exporting():
            if (width <= 0).any():
                raise ValueError("Some boxes have negative widths or 0.")
            if (height <= 0).any():
                raise ValueError("Some boxes have negative heights or 0.")
            if (depth <= 0).any():
                raise ValueError("Some boxes have negative depths or 0.")

        hexahedrons = _boxes3d_to_polygons3d(xmin, ymin, zmin, width, height, depth)
        hexahedrons = hexahedrons if batched else hexahedrons.squeeze(0)
        return cls(hexahedrons, raise_if_not_floating_point=False, mode=mode)

    def to_tensor(self, mode: str = "xyzxyz") -> torch.Tensor:
        r"""Cast :class:`Boxes3D` to a tensor.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes3D`.

        Convention:
            ``mode`` selects the export format and defaults to ``'xyzxyz'`` regardless of the label stored by
            :meth:`from_tensor` or the constructor; :meth:`Boxes.to_tensor` defaults to its stored mode instead.
            Every export starts from the ``amin``/``amax`` bounds of the stored vertices, so it is the
            axis-aligned bounding box of a rotated hexahedron.

        Args:
            mode: The format in which the boxes are provided, matched case-insensitively.

                * 'xyzxyz': boxes are assumed to be in the format ``xmin, ymin, zmin, xmax, ymax, zmax`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'xyzxyz_plus': similar to 'xyzxyz' mode but where box width, length and depth are defined as
                   ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and ``depth = zmax - zmin + 1``.
                * 'xyzwhd': boxes are assumed to be in the format ``xmin, ymin, zmin, width, height, depth`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                  *front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
                  back-top-right, back-bottom-right,  back-bottom-left*. Vertices coordinates are in (x,y, z) order.
                  Finally, box width, height and depth are defined as ``width = xmax - xmin``, ``height = ymax - ymin``
                  and ``depth = zmax - zmin``.
                * 'vertices_plus': similar to 'vertices' mode but where box width, height and depth are defined as
                  ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and ``depth = zmax - zmin + 1``; this is
                  the stored form.

        Returns:
            3D Boxes tensor in the ``mode`` format. The shape depends with the ``mode`` value:

                * 'vertices' or 'vertices_plus': :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)`.
                * Any other value: :math:`(N, 6)` or :math:`(B, N, 6)`.

        Note:
            The vertex-to-corner reduction below is ``amin``/``amax`` over the 8 vertices, which is
            differentiable everywhere except where multiple vertices exactly tie for an axis extremum --
            e.g. every face of an axis-aligned box, where 4 vertices share each min/max coordinate. At an
            exact tie PyTorch's ``amin``/``amax`` backward splits the gradient evenly among the tied
            vertices (``1/k`` for ``k`` ties), a valid subgradient usable for optimization, but one that
            :func:`torch.autograd.gradcheck`'s central-difference estimate will not exactly match at that
            point -- the same non-uniqueness any reduction has at a kink (compare ``torch.max`` or
            :class:`~torch.nn.ReLU` at their own kinks). This was previously (see `#1396
            <https://github.com/kornia/kornia/issues/1396>`_) mistaken for an actual gradient bug and
            gated behind a ``RuntimeError``; :class:`Boxes` (2D) uses the same reduction and was never
            gated, because an axis-aligned rectangle always ties 2-way per extremum, where the ``1/2``
            split happens to coincide with the central-difference estimate -- gradcheck cannot see the
            same kink there. A degenerate 2D box -- zero extent on an axis, so four vertices tie -- reaches
            the same 4-way split and is not guaranteed that coincidence.

        Examples:
            >>> boxes_xyzxyz = torch.as_tensor([[0, 3, 6, 1, 4, 8], [5, 1, 3, 8, 4, 9]])
            >>> boxes = Boxes3D.from_tensor(boxes_xyzxyz, mode='xyzxyz')
            >>> assert (boxes.to_tensor(mode='xyzxyz') == boxes_xyzxyz).all()

        """
        batched_boxes = self._data if self._is_batched else self._data.unsqueeze(0)

        # Create boxes in xyzxyz_plus format.
        boxes = torch.stack([batched_boxes.amin(dim=-2), batched_boxes.amax(dim=-2)], dim=-2).view(
            batched_boxes.shape[0], batched_boxes.shape[1], 6
        )

        mode = mode.lower()
        if mode in ("xyzxyz", "xyzxyz_plus"):
            pass
        elif mode in ("xyzwhd", "vertices", "vertices_plus"):
            width = boxes[..., 3] - boxes[..., 0] + 1
            height = boxes[..., 4] - boxes[..., 1] + 1
            depth = boxes[..., 5] - boxes[..., 2] + 1
            boxes[..., 3] = width
            boxes[..., 4] = height
            boxes[..., 5] = depth
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode in ("xyzxyz", "vertices"):
            offset = torch.as_tensor([0, 0, 0, 1, 1, 1], device=boxes.device, dtype=boxes.dtype)
            boxes = boxes + offset

        if mode.startswith("vertices"):
            xmin, ymin, zmin = boxes[..., 0], boxes[..., 1], boxes[..., 2]
            width, height, depth = boxes[..., 3], boxes[..., 4], boxes[..., 5]

            boxes = _boxes3d_to_polygons3d(xmin, ymin, zmin, width, height, depth)

        boxes = boxes if self._is_batched else boxes.squeeze(0)
        return boxes

    def to_mask(self, depth: int, height: int, width: int) -> torch.Tensor:
        """Convert 3D boxes to masks. Covered area is 1 and the remaining is 0.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes3D`.

        Convention:
            The size is ``(depth, height, width)`` and the mask is :math:`(N, depth, height, width)` or
            :math:`(B, N, depth, height, width)` in the box dtype, where :func:`~kornia.geometry.bbox.bbox_to_mask3d`
            returns ``float32`` with an extra channel axis. The boxes are exported as exclusive ``'xyzxyz'``
            bounds, clamped to the volume, rounded to the nearest integer, and filled over half-open ranges, so a
            box entirely outside the volume fills nothing and a fractional box can fill a different volume than
            :func:`~kornia.geometry.bbox.bbox_to_mask3d`, which truncates. The loop and the grid comparison taken
            under graph capture produce the same mask. A box tensor that requires grad is rejected with
            ``RuntimeError``.

        .. warning::
            The rounding split with :func:`~kornia.geometry.bbox.bbox_to_mask3d` is tracked in
            `#4015 <https://github.com/kornia/kornia/issues/4015>`_.

        Args:
            depth: depth of the masked image/images.
            height: height of the masked image/images.
            width: width of the masked image/images.

        Returns:
            the output mask tensor, shape of :math:`(N, depth, height, width)` or :math:`(B, N, depth, height, width)`
            and dtype of :func:`Boxes3D.dtype` (it can be any floating point dtype).

        Note:
            It is currently non-differentiable.

        Examples:
            >>> boxes = Boxes3D(torch.tensor([[  # Same as Boxes3D.from_tensor([[1, 1, 1, 3, 3, 2]], 'xyzxyz_plus')
            ...     [1., 1., 1.],
            ...     [3., 1., 1.],
            ...     [3., 3., 1.],
            ...     [1., 3., 1.],
            ...     [1., 1., 2.],
            ...     [3., 1., 2.],
            ...     [3., 3., 2.],
            ...     [1., 3., 2.],
            ... ]]))  # 1x8x3
            >>> boxes.to_mask(4, 5, 5)
            tensor([[[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                     [[0., 0., 0., 0., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                     [[0., 0., 0., 0., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                     [[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]]])

        """
        if self._data.requires_grad:
            raise RuntimeError(
                "Boxes.to_tensor isn't differentiable. Please, create boxes from tensors with `requires_grad=False`."
            )

        if self._is_batched:  # (B, N, 8, 3)
            mask = torch.zeros(
                (self._data.shape[0], self._data.shape[1], depth, height, width),
                dtype=self._data.dtype,
                device=self._data.device,
            )
        else:  # (N, 8, 3)
            mask = torch.zeros(
                (self._data.shape[0], depth, height, width), dtype=self._data.dtype, device=self._data.device
            )

        # Boxes coordinates can be outside the image size after transforms. Clamp values to the image size
        clipped_boxes_xyzxyz = self.to_tensor("xyzxyz")
        clipped_boxes_xyzxyz[..., ::3].clamp_(0, width)
        clipped_boxes_xyzxyz[..., 1::3].clamp_(0, height)
        clipped_boxes_xyzxyz[..., 2::3].clamp_(0, depth)

        # Cast boxes coordinates to be integer to use them as bounds. Use round to handle decimal values.
        xyzxyz = clipped_boxes_xyzxyz.view(-1, 6).round().long()

        if is_exporting():
            # The loop below slices with data-dependent bounds, which graph capture cannot do; compare a
            # coordinate grid against the bounds instead.
            device = self._data.device
            zs = torch.arange(depth, device=device)
            ys = torch.arange(height, device=device)
            xs = torch.arange(width, device=device)
            z_mask = (zs[None, :] >= xyzxyz[:, 2:3]) & (zs[None, :] < xyzxyz[:, 5:6])
            y_mask = (ys[None, :] >= xyzxyz[:, 1:2]) & (ys[None, :] < xyzxyz[:, 4:5])
            x_mask = (xs[None, :] >= xyzxyz[:, 0:1]) & (xs[None, :] < xyzxyz[:, 3:4])
            masks = z_mask[:, :, None, None] & y_mask[:, None, :, None] & x_mask[:, None, None, :]
            return masks.to(mask.dtype).view(mask.shape)

        # Reshape mask to (BxN, D, H, W) and boxes to (BxN, 6) to iterate over all of them.
        for mask_channel, box_xyzxyz in zip(mask.view(-1, depth, height, width), xyzxyz):
            # Mask channel dimensions: (depth, height, width)
            mask_channel[
                box_xyzxyz[2] : box_xyzxyz[5], box_xyzxyz[1] : box_xyzxyz[4], box_xyzxyz[0] : box_xyzxyz[3]
            ] = 1

        return mask

    def transform_boxes(self, M: torch.Tensor, inplace: bool = False) -> Boxes3D:
        r"""Apply a transformation matrix to the 3D boxes.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes3D`.

        Args:
            M: The transformation matrix to be applied, shape of :math:`(4, 4)` or :math:`(B, 4, 4)`, where ``B``
                must equal the box batch size and an unbatched box counts as one.
            inplace: do transform in-place and return self.

        Returns:
            The transformed boxes: a new :class:`Boxes3D` labelled ``'xyzxyz_plus'`` when ``inplace`` is false,
            otherwise ``self``.

        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (4, 4):
            raise ValueError(f"The transformation matrix shape must be (4, 4) or (B, 4, 4). Got {M.shape}.")

        transformed_boxes = _transform_boxes(self._data, M)
        if inplace:
            self._data = transformed_boxes
            return self

        return Boxes3D(transformed_boxes, False, "xyzxyz_plus")

    def transform_boxes_(self, M: torch.Tensor) -> Boxes3D:
        """Apply :meth:`transform_boxes` in place and return ``self``.

        See the Convention block on :class:`~kornia.geometry.boxes.Boxes3D`.

        Convention:
            The in-place operation rebinds this object's internal tensor to the transformed result. A tensor
            reference obtained from :attr:`data` before that call therefore remains unchanged and no longer
            aliases the container's data. The stored mode label is kept.

        Returns:
            This :class:`Boxes3D` object after the transformation.
        """
        return self.transform_boxes(M, inplace=True)

    @property
    def data(self) -> torch.Tensor:
        """Return the raw 3D corner-coordinate tensor.

        Returns:
            Tensor containing eight 3D corner coordinates per box, usually
            shaped :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)`.
        """
        return self._data

    @property
    def mode(self) -> str:
        """Return the 3D box format remembered by this container.

        Returns:
            Mode string describing how this container should be interpreted
            during tensor conversion.
        """
        return self._mode

    @property
    def device(self) -> torch.device:
        """Returns boxes device."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Returns boxes dtype."""
        return self._data.dtype

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Boxes3D:
        """Like :func:`torch.nn.Module.to()` method."""
        # In torchscript, dtype is a int and not a class. https://github.com/pytorch/pytorch/issues/51941
        if dtype is not None and not _is_floating_point_dtype(dtype):
            raise ValueError("Boxes must be in floating point")
        self._data = self._data.to(device=device, dtype=dtype)
        return self

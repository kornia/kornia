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

from typing import Optional

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import is_exporting

from .linalg import transform_points

__all__ = [
    "bbox_generator",
    "bbox_generator3d",
    "bbox_to_mask",
    "bbox_to_mask3d",
    "infer_bbox_shape",
    "infer_bbox_shape3d",
    "nms",
    "transform_bbox",
    "validate_bbox",
    "validate_bbox3d",
]


def validate_bbox(boxes: torch.Tensor) -> bool:
    """Validate whether a 2D box has matching top and bottom edge vectors.

    Convention:
        Vertices use inclusive coordinates in clockwise top-left, top-right, bottom-right, bottom-left order.
        The function accepts :math:`(B, 4, 2)` tensors and :math:`(B, N, 4, 2)` tensors. It returns ``False`` for an
        invalid shape or when corresponding components of the top and bottom edge vectors differ by more than
        ``1e-4``; it does not raise for those inputs. It does not check right angles, positive area, or clockwise
        direction. A parallelogram whose
        vertices follow a cyclic order passes, including cyclic rotations, either direction, rotated rectangles,
        and zero-area boxes; other vertex relabelings can fail. The inclusive ``+1`` terms, tracked in
        `#3934 <https://github.com/kornia/kornia/issues/3934>`_, cancel in exact arithmetic, but finite-precision
        rounding can make the result differ from exclusive arithmetic, particularly for low-precision dtypes.
        Rank-4 input is flattened with ``reshape``, so any stride layout is accepted, matching the shape the
        docstring documents.

    .. warning::
        :func:`validate_bbox3d` raises ``AssertionError`` where this function returns ``False``. That
        inconsistency is tracked in `#4013 <https://github.com/kornia/kornia/issues/4013>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of :math:`(B, 4, 2)` or :math:`(B, N, 4, 2)`, where each box is defined in the
            following ``clockwise`` order: top-left, top-right, bottom-right, bottom-left. The coordinates must be in
            the x, y order.

    """
    if not (len(boxes.shape) in [3, 4] and boxes.shape[-2:] == torch.Size([4, 2])):
        return False

    if len(boxes.shape) == 4:
        boxes = boxes.reshape(-1, 4, 2)

    x_tl, y_tl = boxes[..., 0, 0], boxes[..., 0, 1]
    x_tr, y_tr = boxes[..., 1, 0], boxes[..., 1, 1]
    x_br, y_br = boxes[..., 2, 0], boxes[..., 2, 1]
    x_bl, y_bl = boxes[..., 3, 0], boxes[..., 3, 1]

    width_t, width_b = x_tr - x_tl + 1, x_br - x_bl + 1
    height_t, height_b = y_tr - y_tl + 1, y_br - y_bl + 1

    # Replace torch.allclose with exportable operations
    width_diff = torch.abs(width_t - width_b)
    height_diff = torch.abs(height_t - height_b)

    # Check if differences are within tolerance (1e-4)
    if torch.any(width_diff > 1e-4):
        return False

    if torch.any(height_diff > 1e-4):
        return False

    return True


def validate_bbox3d(boxes: torch.Tensor) -> bool:
    """Validate that a 3D box has equal inclusive edge extents along each axis, raising when it does not.

    Convention:
        Vertices use inclusive coordinates in the order front-top-left, front-top-right, front-bottom-right,
        front-bottom-left, then the same four back vertices. The function accepts :math:`(B, 8, 3)` and
        :math:`(B, N, 8, 3)` tensors and raises ``AssertionError`` for any other shape. It compares the inclusive
        ``+1`` extents of the four edges parallel to each axis and raises ``AssertionError`` when they differ, so a
        sheared parallelepiped with equal edge lengths and a zero-extent box both pass; it does not check right
        angles or positive extent. Under graph capture the extent checks are skipped and the shape check alone
        returns ``True``. The ``+1`` terms cancel in exact arithmetic and are tracked in
        `#3934 <https://github.com/kornia/kornia/issues/3934>`_.

    .. warning::
        :func:`validate_bbox` returns ``False`` where this function raises; that inconsistency is tracked in
        `#4013 <https://github.com/kornia/kornia/issues/4013>`_. Rank-4 input passes this check but breaks
        :func:`infer_bbox_shape3d` and :func:`bbox_to_mask3d`, tracked in
        `#4248 <https://github.com/kornia/kornia/issues/4248>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of :math:`(B, 8, 3)` or :math:`(B, N, 8, 3)`, where each box is defined in the following ``clockwise``
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in the x, y, z order.

    Returns:
        ``True``. Invalid input raises instead of returning ``False``.

    """
    if not (len(boxes.shape) in [3, 4] and boxes.shape[-2:] == torch.Size([8, 3])):
        raise AssertionError(f"Box shape must be (B, 8, 3) or (B, N, 8, 3). Got {boxes.shape}.")

    if len(boxes.shape) == 4:
        boxes = boxes.reshape(-1, 8, 3)

    # The cube checks below read the data, which graph capture cannot do; skip them under export.
    if is_exporting():
        return True

    left = torch.index_select(boxes, 1, torch.tensor([1, 2, 5, 6], device=boxes.device, dtype=torch.long))[:, :, 0]
    right = torch.index_select(boxes, 1, torch.tensor([0, 3, 4, 7], device=boxes.device, dtype=torch.long))[:, :, 0]
    widths = left - right + 1
    if not torch.allclose(widths.permute(1, 0), widths[:, 0]):
        raise AssertionError(f"Boxes must have be cube, while get different widths {widths}.")

    bot = torch.index_select(boxes, 1, torch.tensor([2, 3, 6, 7], device=boxes.device, dtype=torch.long))[:, :, 1]
    upper = torch.index_select(boxes, 1, torch.tensor([0, 1, 4, 5], device=boxes.device, dtype=torch.long))[:, :, 1]
    heights = bot - upper + 1
    if not torch.allclose(heights.permute(1, 0), heights[:, 0]):
        raise AssertionError(f"Boxes must have be cube, while get different heights {heights}.")

    depths = boxes[:, 4:, 2] - boxes[:, :4, 2] + 1
    if not torch.allclose(depths.permute(1, 0), depths[:, 0]):
        raise AssertionError(f"Boxes must have be cube, while get different depths {depths}.")

    return True


def infer_bbox_shape(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Auto-infer the output sizes for the given 2D bounding boxes.

    Convention:
        Vertices use inclusive coordinates in clockwise top-left, top-right, bottom-right, bottom-left order.
        The returned tuple is ``(heights, widths)``, in that order. Both extents are read from fixed vertex
        indices, as ``width = boxes[:, 1, 0] - boxes[:, 0, 0] + 1`` and
        ``height = boxes[:, 2, 1] - boxes[:, 0, 1] + 1``, rather than from a ``maximum - minimum`` reduction.
        The two agree for an axis-aligned box in the documented order; for any other vertex order, including the
        rotated quadrilaterals that :func:`transform_bbox` produces for polygon input, they can diverge and the
        result can be negative.
        :meth:`kornia.geometry.boxes.Boxes.get_boxes_shape` is reduction based and does not share that behavior.
        The fixed-index reading also lets zero-width boxes emitted by :func:`bbox_generator` report width ``0``
        rather than ``2`` under a reduction; :class:`~kornia.augmentation.RandomCutMixV2` relies on that behavior.

    .. warning::
        The inclusive ``+1`` arithmetic differs from torchvision, COCO, and albumentations and is tracked in
        `#3934 <https://github.com/kornia/kornia/issues/3934>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have shape
            :math:`(N, 4, 2)`, where each box is defined in the following ``clockwise`` order: top-left, top-right,
            bottom-right, bottom-left. The coordinates must be in the x, y order.

    Raises:
        ShapeError: if ``boxes`` does not have shape :math:`(N, 4, 2)`.

    Returns:
        - Bounding box heights, shape of :math:`(N,)`.
        - Bounding box widths, shape of :math:`(N,)`.

    Example:
        >>> boxes = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ], [
        ...     [1., 1.],
        ...     [3., 1.],
        ...     [3., 2.],
        ...     [1., 2.],
        ... ]])  # 2x4x2
        >>> infer_bbox_shape(boxes)
        (tensor([2., 2.]), tensor([2., 3.]))

    """
    KORNIA_CHECK_SHAPE(boxes, ["N", "4", "2"])

    width: torch.Tensor = boxes[:, 1, 0] - boxes[:, 0, 0] + 1
    height: torch.Tensor = boxes[:, 2, 1] - boxes[:, 0, 1] + 1
    return height, width


def infer_bbox_shape3d(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Auto-infer the output sizes for the given 3D bounding boxes.

    Convention:
        Vertices use inclusive coordinates in the order front-top-left, front-top-right, front-bottom-right,
        front-bottom-left, then the same four back vertices. The returned tuple is ``(depths, heights, widths)``,
        in that order, each read as ``max - min + 1`` along one edge of the box after :func:`validate_bbox3d`
        has established that the edges parallel to each axis have equal extent. Like :func:`infer_bbox_shape`,
        the function adds one per axis, so pass the ``'vertices_plus'`` export of
        :class:`~kornia.geometry.boxes.Boxes3D` rather than ``'vertices'``, which it reads as one larger per axis.

    .. warning::
        The inclusive ``+1`` arithmetic differs from torchvision, COCO, and albumentations and is tracked in
        `#3934 <https://github.com/kornia/kornia/issues/3934>`_; the exclusive-export trap is
        `#4009 <https://github.com/kornia/kornia/issues/4009>`_. Validation raises ``AssertionError`` rather than
        returning ``False``, see `#4013 <https://github.com/kornia/kornia/issues/4013>`_. Batched :math:`(B, N, 8, 3)`
        input passes validation but is then indexed as if the box axis were the vertex axis, which raises an
        indexing error or returns wrong-shaped values depending on ``N``; flatten to :math:`(B \cdot N, 8, 3)`
        first. Tracked in `#4248 <https://github.com/kornia/kornia/issues/4248>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of :math:`(B, 8, 3)`, where each box is defined in the following ``clockwise`` order: front-top-left,
            front-top-right, front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right,
            back-bottom-left. The coordinates must be in the x, y, z order.

    Returns:
        - Bounding box depths, shape of :math:`(B,)`.
        - Bounding box heights, shape of :math:`(B,)`.
        - Bounding box widths, shape of :math:`(B,)`.

    Example:
        >>> boxes = torch.tensor([[[ 0,  1,  2],
        ...         [10,  1,  2],
        ...         [10, 21,  2],
        ...         [ 0, 21,  2],
        ...         [ 0,  1, 32],
        ...         [10,  1, 32],
        ...         [10, 21, 32],
        ...         [ 0, 21, 32]],
        ...        [[ 3,  4,  5],
        ...         [43,  4,  5],
        ...         [43, 54,  5],
        ...         [ 3, 54,  5],
        ...         [ 3,  4, 65],
        ...         [43,  4, 65],
        ...         [43, 54, 65],
        ...         [ 3, 54, 65]]]) # 2x8x3
        >>> infer_bbox_shape3d(boxes)
        (tensor([31, 61]), tensor([21, 51]), tensor([11, 41]))

    """
    validate_bbox3d(boxes)

    left = torch.index_select(boxes, 1, torch.tensor([1, 2, 5, 6], device=boxes.device, dtype=torch.long))[:, :, 0]
    right = torch.index_select(boxes, 1, torch.tensor([0, 3, 4, 7], device=boxes.device, dtype=torch.long))[:, :, 0]
    widths = (left - right + 1)[:, 0]

    bot = torch.index_select(boxes, 1, torch.tensor([2, 3, 6, 7], device=boxes.device, dtype=torch.long))[:, :, 1]
    upper = torch.index_select(boxes, 1, torch.tensor([0, 1, 4, 5], device=boxes.device, dtype=torch.long))[:, :, 1]
    heights = (bot - upper + 1)[:, 0]

    depths = (boxes[:, 4:, 2] - boxes[:, :4, 2] + 1)[:, 0]
    return depths, heights, widths


def bbox_to_mask(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert 2D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Convention:
        The image size is given as ``(width, height)`` while the mask comes back as :math:`(B, height, width)`;
        :meth:`kornia.geometry.boxes.Boxes.to_mask` takes ``(height, width)`` for the same result. Only the top-left
        (index 0) and bottom-right (index 2) vertices are read and the other two are ignored, so a non-rectangular
        quadrilateral is masked by the axis-aligned box those two vertices span. A pixel is covered when its integer
        coordinates satisfy ``xmin <= x <= xmax`` and ``ymin <= y <= ymax`` on the raw, unrounded values, which
        reads the vertices as inclusive: pass the ``'vertices_plus'`` export of
        :class:`~kornia.geometry.boxes.Boxes`, not ``'vertices'``. The mask has the input dtype, including integer
        dtypes, and no gradient path. Input must be unbatched :math:`(B, 4, 2)`; :math:`(B, N, 4, 2)` raises
        :class:`~kornia.core.exceptions.ShapeError`.

    .. warning::
        The ``(width, height)`` argument order is tracked in `#4014 <https://github.com/kornia/kornia/issues/4014>`_.
        The inclusive raw-float comparison differs from the rounding in
        :meth:`~kornia.geometry.boxes.Boxes.to_mask` and the truncation in :func:`bbox_to_mask3d` for fractional
        coordinates and is tracked in `#4015 <https://github.com/kornia/kornia/issues/4015>`_; the exclusive-export
        trap is `#4009 <https://github.com/kornia/kornia/issues/4009>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of :math:`(B, 4, 2)`, where each box is defined in the following ``clockwise`` order: top-left, top-right,
            bottom-right and bottom-left. The coordinates must be in the x, y order.
        width: width of the masked image.
        height: height of the masked image.

    Returns:
        the output mask tensor, shape of :math:`(B, height, width)` and dtype of ``boxes``.

    Raises:
        ShapeError: if ``boxes`` does not have shape :math:`(B, 4, 2)`.

    Note:
        It is currently non-differentiable.

    Examples:
        >>> boxes = torch.tensor([[
        ...        [1., 1.],
        ...        [3., 1.],
        ...        [3., 2.],
        ...        [1., 2.],
        ...   ]])  # 1x4x2
        >>> bbox_to_mask(boxes, 5, 5)
        tensor([[[0., 0., 0., 0., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]]])

    """
    KORNIA_CHECK_SHAPE(boxes, ["B", "4", "2"])

    # NOTE: `validate_bbox`'s boolean result was previously computed here and discarded — it
    # never raised, so it performed no validation while adding a data-dependent graph break
    # (`torch.any(...)` -> Python `if`) that blocked torch.compile fullgraph (e.g. RandomErasing,
    # which builds its mask through this function). Dropped; behaviour is byte-identical.
    # zero padding the surroundings
    yy = torch.arange(height, device=boxes.device, dtype=boxes.dtype).view(height, 1)
    xx = torch.arange(width, device=boxes.device, dtype=boxes.dtype).view(1, width)
    x_min = boxes[:, 0, 0].view(-1, 1, 1)
    y_min = boxes[:, 0, 1].view(-1, 1, 1)
    x_max = boxes[:, 2, 0].view(-1, 1, 1)
    y_max = boxes[:, 2, 1].view(-1, 1, 1)
    # Reduce along each axis first (cheap ``(B, 1, W)`` and ``(B, H, 1)`` ands), then combine once.
    # The previous ``a & b & c & d`` chained two full ``(B, H, W)`` ands; this does a single one —
    # byte-identical result, half the full-grid work (the dominant cost when masking large images).
    x_in = (xx >= x_min) & (xx <= x_max)
    y_in = (yy >= y_min) & (yy <= y_max)
    mask = x_in & y_in
    return mask.to(boxes.dtype)


def bbox_to_mask3d(boxes: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """Convert 3D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Convention:
        ``size`` is ``(depth, height, width)`` and the mask comes back as :math:`(B, 1, depth, height, width)` in
        ``float32`` whatever the input dtype, unlike :func:`bbox_to_mask`, which keeps the input dtype and has no
        channel axis, and :meth:`kornia.geometry.boxes.Boxes3D.to_mask`, which keeps the box dtype and returns
        :math:`(N, depth, height, width)`. After :func:`validate_bbox3d`, which raises ``AssertionError`` for an
        invalid box, the bounds are read from fixed vertex positions, truncated toward zero with ``.long()``, and
        compared inclusively, which reads the vertices as inclusive: pass the ``'vertices_plus'`` export. The
        intersection of the three axis ranges is recovered only when the box leaves at least one index uncovered
        on every axis; see the warning. There is no gradient path.

    .. warning::
        A box that covers or overhangs a whole output axis fills the entire volume instead of the intersection:
        the union-of-planes intermediate becomes all true and its reductions lose the other two bounds, where
        :meth:`~kornia.geometry.boxes.Boxes3D.to_mask` fills the clamped region. Tracked in
        `#4255 <https://github.com/kornia/kornia/issues/4255>`_. The truncation differs from the inclusive
        raw-float comparison of :func:`bbox_to_mask` and the rounding of
        :meth:`~kornia.geometry.boxes.Boxes3D.to_mask` for fractional coordinates and is tracked in
        `#4015 <https://github.com/kornia/kornia/issues/4015>`_. The ``float32`` output with a channel axis is
        tracked in `#4250 <https://github.com/kornia/kornia/issues/4250>`_. Validation raises rather than returning
        ``False``, `#4013 <https://github.com/kornia/kornia/issues/4013>`_. Batched :math:`(B, N, 8, 3)` input passes
        validation and then raises an indexing or broadcasting error; flatten it first. Tracked in
        `#4248 <https://github.com/kornia/kornia/issues/4248>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of :math:`(B, 8, 3)`, where each box is defined in the following ``clockwise`` order: front-top-left,
            front-top-right, front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right,
            back-bottom-left. The coordinates must be in the x, y, z order.
        size: depth, height and width of the masked image.

    Returns:
        the output mask tensor, shape of :math:`(B, 1, depth, height, width)` and dtype ``float32``.

    Examples:
        >>> boxes = torch.tensor([[
        ...     [1., 1., 1.],
        ...     [2., 1., 1.],
        ...     [2., 2., 1.],
        ...     [1., 2., 1.],
        ...     [1., 1., 2.],
        ...     [2., 1., 2.],
        ...     [2., 2., 2.],
        ...     [1., 2., 2.],
        ... ]])  # 1x8x3
        >>> bbox_to_mask3d(boxes, (4, 5, 5))
        tensor([[[[[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 1., 1., 0., 0.],
                   [0., 1., 1., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 1., 1., 0., 0.],
                   [0., 1., 1., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]]]]])

    """
    validate_bbox3d(boxes)
    D0, D1, D2 = size  # get depth, height, width

    z_min = boxes[:, 0, 2].long()
    z_max = boxes[:, 4, 2].long()
    y_min = boxes[:, 1, 1].long()
    y_max = boxes[:, 2, 1].long()
    x_min = boxes[:, 0, 0].long()
    x_max = boxes[:, 1, 0].long()

    z = torch.arange(D0, device=boxes.device, dtype=torch.long)
    y = torch.arange(D1, device=boxes.device, dtype=torch.long)
    x = torch.arange(D2, device=boxes.device, dtype=torch.long)

    # Compute mask as union of planes in one step
    m = (
        ((z[None, :] >= z_min[:, None]) & (z[None, :] <= z_max[:, None]))[:, None, :, None, None]
        | ((y[None, :] >= y_min[:, None]) & (y[None, :] <= y_max[:, None]))[:, None, None, :, None]
        | ((x[None, :] >= x_min[:, None]) & (x[None, :] <= x_max[:, None]))[:, None, None, None, :]
    ).float()  # Shape: (N, 1, D0, D1, D2)

    # Compute conditions
    cond1 = m.all(dim=3, keepdim=True).all(dim=2, keepdim=True)
    cond2 = m.all(dim=4, keepdim=True).all(dim=2, keepdim=True)
    cond3 = m.all(dim=3, keepdim=True).all(dim=4, keepdim=True)

    m_out = cond1 * cond2 * cond3  # Broadcasting to (N, 1, D0, D1, D2)
    return m_out.float()


def bbox_generator(
    x_start: torch.Tensor, y_start: torch.Tensor, width: torch.Tensor, height: torch.Tensor
) -> torch.Tensor:
    """Generate 2D bounding boxes according to the provided start coords, width and height.

    Convention:
        The far corner is placed at ``start + size - 1`` on each axis, so the generated box is inclusive:
        :func:`infer_bbox_shape` reads back exactly ``width`` and ``height``, and a zero size places the far
        corner one before the start. The vertex order is top-left, top-right, bottom-right, bottom-left. A scalar
        input produces a batch of one. All four tensors must share dtype and device, otherwise ``AssertionError``
        is raised; the output has that dtype and device and keeps a gradient path to the inputs.

    .. warning::
        The inclusive arithmetic is tracked in `#3934 <https://github.com/kornia/kornia/issues/3934>`_.
        :func:`bbox_generator3d` places its far corner at ``start + size`` instead, see
        `#4018 <https://github.com/kornia/kornia/issues/4018>`_.

    Args:
        x_start: a tensor containing the x coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        y_start: a tensor containing the y coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        width: widths of the bounding boxes. Shape must be a scalar tensor or :math:`(B,)`.
        height: heights of the bounding boxes. Shape must be a scalar tensor or :math:`(B,)`.

    Returns:
        the bounding box tensor, shape of :math:`(B, 4, 2)`.

    Examples:
        >>> x_start = torch.tensor([0, 1])
        >>> y_start = torch.tensor([1, 0])
        >>> width = torch.tensor([5, 3])
        >>> height = torch.tensor([7, 4])
        >>> bbox_generator(x_start, y_start, width, height)
        tensor([[[0, 1],
                 [4, 1],
                 [4, 7],
                 [0, 7]],
        <BLANKLINE>
                [[1, 0],
                 [3, 0],
                 [3, 3],
                 [1, 3]]])

    """
    if not (x_start.shape == y_start.shape and x_start.dim() in [0, 1]):
        raise AssertionError(f"`x_start` and `y_start` must be a scalar or (B,). Got {x_start}, {y_start}.")
    if not (width.shape == height.shape and width.dim() in [0, 1]):
        raise AssertionError(f"`width` and `height` must be a scalar or (B,). Got {width}, {height}.")
    if not x_start.dtype == y_start.dtype == width.dtype == height.dtype:
        raise AssertionError(
            "All tensors must be in the same dtype. Got "
            f"`x_start`({x_start.dtype}), `y_start`({x_start.dtype}), `width`({width.dtype}), `height`({height.dtype})."
        )
    if not x_start.device == y_start.device == width.device == height.device:
        raise AssertionError(
            "All tensors must be in the same device. Got "
            f"`x_start`({x_start.device}), `y_start`({x_start.device}), "
            f"`width`({width.device}), `height`({height.device})."
        )

    # Build the four corners (TL, TR, BR, BL) directly by stacking instead of allocating a zero
    # tensor and mutating it with six indexed in-place adds (each of which is a separate kernel /
    # copy). `.view(-1)` treats a scalar input as batch-1, matching the previous ``repeat`` shape.
    x0 = x_start.view(-1)
    y0 = y_start.view(-1)
    x1 = x0 + width.view(-1) - 1
    y1 = y0 + height.view(-1) - 1
    bbox = torch.stack(
        [
            torch.stack([x0, y0], dim=-1),
            torch.stack([x1, y0], dim=-1),
            torch.stack([x1, y1], dim=-1),
            torch.stack([x0, y1], dim=-1),
        ],
        dim=-2,
    )

    return bbox


def bbox_generator3d(
    x_start: torch.Tensor,
    y_start: torch.Tensor,
    z_start: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Generate 3D bounding boxes according to the provided start coords, width, height and depth.

    Convention:
        The far corner is placed at ``start + size`` on each axis, one further than :func:`bbox_generator`, so
        :func:`infer_bbox_shape3d` reads back ``size + 1`` on every axis; the example below shows it. The four
        front vertices precede the four back vertices. A scalar input produces a batch of one. All six tensors
        must share dtype and device, otherwise ``AssertionError`` is raised; the output has that dtype and device
        and keeps a gradient path to the inputs.

    .. warning::
        The extra unit of extent relative to :func:`bbox_generator` and to the inclusive
        :func:`infer_bbox_shape3d` is tracked as a coordinated repair in
        `#4018 <https://github.com/kornia/kornia/issues/4018>`_ and is documented as it is.

    Args:
        x_start: a tensor containing the x coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        y_start: a tensor containing the y coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        z_start: a tensor containing the z coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        width: widths of the bounding boxes. Shape must be a scalar tensor or :math:`(B,)`.
        height: heights of the bounding boxes. Shape must be a scalar tensor or :math:`(B,)`.
        depth: depths of the bounding boxes. Shape must be a scalar tensor or :math:`(B,)`.

    Returns:
        the 3d bounding box tensor :math:`(B, 8, 3)`.

    Examples:
        >>> x_start = torch.tensor([0, 3])
        >>> y_start = torch.tensor([1, 4])
        >>> z_start = torch.tensor([2, 5])
        >>> width = torch.tensor([10, 40])
        >>> height = torch.tensor([20, 50])
        >>> depth = torch.tensor([30, 60])
        >>> bbox_generator3d(x_start, y_start, z_start, width, height, depth)
        tensor([[[ 0,  1,  2],
                 [10,  1,  2],
                 [10, 21,  2],
                 [ 0, 21,  2],
                 [ 0,  1, 32],
                 [10,  1, 32],
                 [10, 21, 32],
                 [ 0, 21, 32]],
        <BLANKLINE>
                [[ 3,  4,  5],
                 [43,  4,  5],
                 [43, 54,  5],
                 [ 3, 54,  5],
                 [ 3,  4, 65],
                 [43,  4, 65],
                 [43, 54, 65],
                 [ 3, 54, 65]]])

    """
    if not (x_start.shape == y_start.shape == z_start.shape and x_start.dim() in [0, 1]):
        raise AssertionError(
            f"`x_start`, `y_start` and `z_start` must be a scalar or (B,). Got {x_start}, {y_start}, {z_start}."
        )
    if not (width.shape == height.shape == depth.shape and width.dim() in [0, 1]):
        raise AssertionError(f"`width`, `height` and `depth` must be a scalar or (B,). Got {width}, {height}, {depth}.")
    if not x_start.dtype == y_start.dtype == z_start.dtype == width.dtype == height.dtype == depth.dtype:
        raise AssertionError(
            "All tensors must be in the same dtype. "
            f"Got `x_start`({x_start.dtype}), `y_start`({x_start.dtype}), `z_start`({x_start.dtype}), "
            f"`width`({width.dtype}), `height`({height.dtype}) and `depth`({depth.dtype})."
        )
    if not x_start.device == y_start.device == z_start.device == width.device == height.device == depth.device:
        raise AssertionError(
            "All tensors must be in the same device. "
            f"Got `x_start`({x_start.device}), `y_start`({x_start.device}), `z_start`({x_start.device}), "
            f"`width`({width.device}), `height`({height.device}) and `depth`({depth.device})."
        )

    # front
    bbox = torch.tensor(
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], device=x_start.device, dtype=x_start.dtype
    ).repeat(x_start.numel(), 1, 1)

    bbox[:, :, 0] += x_start.view(-1, 1)
    bbox[:, :, 1] += y_start.view(-1, 1)
    bbox[:, :, 2] += z_start.view(-1, 1)
    bbox[:, 1, 0] += width.view(-1)
    bbox[:, 2, 0] += width.view(-1)
    bbox[:, 2, 1] += height.view(-1)
    bbox[:, 3, 1] += height.view(-1)

    # back
    bbox_back = bbox.clone()
    bbox_back[:, :, -1] += depth.view(-1, 1).expand(-1, 4)
    bbox = torch.cat([bbox, bbox_back], dim=1)

    return bbox


def transform_bbox(
    trans_mat: torch.Tensor, boxes: torch.Tensor, mode: str = "xyxy", restore_coordinates: Optional[bool] = None
) -> torch.Tensor:
    r"""Apply a transformation matrix to a box or batch of boxes.

    Convention:
        ``xyxy`` and ``xywh`` inputs are transformed through endpoint pairs. Coordinate
        restoration sorts their endpoints after a flip; polygon vertices retain their
        transformed cyclic order.

    Args:
        trans_mat: The transformation matrix to be applied, with supported shape :math:`(B, 3, 3)`.
            For boxes shaped :math:`(N, 4)`, ``B`` is one or ``N``. For boxes shaped
            :math:`(B, N, 4)` or :math:`(B, N, 4, 2)`, it is one or the boxes' ``B``.
        boxes: The boxes to be transformed with a common shape of :math:`(N, 4)` or batched as :math:`(B, N, 4)`, the
            polygon shape of :math:`(B, N, 4, 2)` is also supported.
        mode: The format in which the boxes are provided. If set to 'xyxy' the boxes are assumed to be in the format
            ``xmin, ymin, xmax, ymax``. If set to 'xywh' the boxes are assumed to be in the format
            ``xmin, ymin, width, height``
        restore_coordinates: Reorder endpoints after a flipped ``xyxy`` or ``xywh`` transform.
            Enabled by default (``None`` behaves as ``True``); pass ``False`` to preserve raw
            transformed endpoints. Polygon vertices retain their transformed order.

    Returns:
        The transformed boxes in the specified mode.

    """
    if not isinstance(mode, str):
        raise TypeError(f"Mode must be a string. Got {type(mode)}")

    if mode not in ("xyxy", "xywh"):
        raise ValueError(f"Mode must be one of 'xyxy', 'xywh'. Got {mode}")

    # convert boxes to format xyxy
    if mode == "xywh":
        boxes = boxes.clone()
        boxes[..., 2] = boxes[..., 0] + boxes[..., 2]  # x + w
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3]  # y + h

    transformed_boxes: torch.Tensor = transform_points(trans_mat, boxes.view(boxes.shape[0], -1, 2))
    transformed_boxes = transformed_boxes.view_as(boxes)

    if (restore_coordinates is None or restore_coordinates) and not (boxes.shape[-2:] == torch.Size([4, 2])):
        restored_boxes = transformed_boxes.clone()
        # In case the boxes are flipped, we ensure it is ordered like left-top -> right-bot points
        restored_boxes[..., 0] = torch.min(transformed_boxes[..., [0, 2]], dim=-1)[0]
        restored_boxes[..., 1] = torch.min(transformed_boxes[..., [1, 3]], dim=-1)[0]
        restored_boxes[..., 2] = torch.max(transformed_boxes[..., [0, 2]], dim=-1)[0]
        restored_boxes[..., 3] = torch.max(transformed_boxes[..., [1, 3]], dim=-1)[0]
        transformed_boxes = restored_boxes

    if mode == "xywh":
        transformed_boxes[..., 2] = transformed_boxes[..., 2] - transformed_boxes[..., 0]
        transformed_boxes[..., 3] = transformed_boxes[..., 3] - transformed_boxes[..., 1]

    return transformed_boxes


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Perform non-maxima suppression (NMS) on tensor of bounding boxes according to the intersection-over-union (IoU).

    Convention:
        Boxes use exclusive ``xyxy`` coordinates. IoU area is
        ``(x_2 - x_1) * (y_2 - y_1)`` and the result contains kept indices rather
        than a boolean mask.

    Args:
        boxes: tensor containing exclusive ``xyxy`` bounding boxes with shape
            :math:`(N, 4)`, ordered as ``(x_1, y_1, x_2, y_2)``.
        scores: tensor containing the scores associated to each bounding box with shape :math:`(N,)`.
        iou_threshold: the threshold to discard the overlapping boxes.

    Returns:
        Indices of the boxes kept from the input set, ordered by descending score.

    .. warning::
        This differs from the inclusive coordinate arithmetic used by several other
        :mod:`kornia.geometry.bbox` operations and is documented in
        `#4008 <https://github.com/kornia/kornia/issues/4008>`_.

    Example:
        >>> boxes = torch.tensor([
        ...     [10., 10., 20., 20.],
        ...     [15., 5., 15., 25.],
        ...     [100., 100., 200., 200.],
        ...     [100., 100., 200., 200.]])
        >>> scores = torch.tensor([0.9, 0.8, 0.7, 0.9])
        >>> nms(boxes, scores, iou_threshold=0.8)
        tensor([0, 3, 1])

    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes expected as Nx4. Got: {boxes.shape}.")

    if len(scores.shape) != 1:
        raise ValueError(f"scores expected as N. Got: {scores.shape}.")

    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"boxes and scores mus have same shape. Got: {boxes.shape, scores.shape}.")

    x1, y1, x2, y2 = boxes.unbind(-1)
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(descending=True)

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    if len(keep) > 0:
        return torch.stack(keep)

    return boxes.new_empty((0,), dtype=torch.long)

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

"""Box functions on plain corner tensors: build, measure, validate, mask, transform and NMS.

A 2D box is four ``(x, y)`` corners in clockwise order, top-left, top-right, bottom-right, bottom-left, so one box per
image is ``(B, 4, 2)``. A 3D box is those four corners at the front followed by the same four at the back, ``(B, 8, 3)``
with ``(x, y, z)`` corners. Corners are inclusive pixel centres: a box covering pixels ``0..9`` has corners at ``0`` and
``9`` and width ``x_max - x_min + 1 = 10``, unlike the exclusive boxes of torchvision, COCO and albumentations
(`#3934 <https://github.com/kornia/kornia/issues/3934>`_). The generators and the shape and mask functions share this
convention; :func:`nms` takes exclusive ``(N, 4)`` ``xyxy`` boxes instead
(`#4008 <https://github.com/kornia/kornia/issues/4008>`_), and :func:`transform_bbox` takes ``xyxy`` or ``xywh`` boxes,
or corners, and adds no ``+1``. From an unbatched :class:`~kornia.geometry.boxes.Boxes`, pass its
``to_tensor('vertices_plus')`` export; the ``'vertices'`` export reads one pixel larger
(`#4009 <https://github.com/kornia/kornia/issues/4009>`_). See
:doc:`Conventions & Pitfalls </get-started/conventions>` for the box formats of the rest of kornia.

Which function to use:

- Corners from a start corner and a size: :func:`bbox_generator` in 2D, :func:`bbox_generator3d` in 3D.
- The size of axis-aligned corners: :func:`infer_bbox_shape` returns heights and widths, :func:`infer_bbox_shape3d`
  depths, heights and widths.
- Whether corners are laid out as a box: :func:`validate_bbox` returns ``False`` when they are not,
  :func:`validate_bbox3d` raises ``AssertionError``.
- A filled mask per box: :func:`bbox_to_mask` returns ``(B, H, W)``, :func:`bbox_to_mask3d` ``(B, 1, D, H, W)``.
- Boxes moved by ``3 x 3`` matrices: :func:`transform_bbox` for ``xyxy`` or ``xywh`` tensors and corners;
  :meth:`~kornia.geometry.boxes.Boxes.transform_boxes` moves all four corners of a box container.
- The highest-scored of overlapping boxes: :func:`nms`, by IoU. The local maxima of a response map are
  :func:`~kornia.geometry.subpix.nms2d` instead.
- Boxes as objects with format conversions, several boxes per image, padding and filtering:
  :class:`~kornia.geometry.boxes.Boxes` and :class:`~kornia.geometry.boxes.Boxes3D`. The image patch inside each box:
  :func:`~kornia.geometry.transform.crop_and_resize`.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch

from kornia.core._compat import _emit_deprecation_warning
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import is_compiling

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

_VALIDATE_BBOX_FALSE_MESSAGE = (
    "`validate_bbox` returned False for an invalid box. Returning False is deprecated since kornia 0.9.1: a "
    "future release raises AssertionError instead, as `validate_bbox3d` does. To work with both, treat an "
    "AssertionError like a False result (kornia#4013)."
)
_VALIDATE_BBOX3D_FALSE_MESSAGE = (
    "`validate_bbox3d` returned False for a non-finite coordinate. Returning False is deprecated since kornia "
    "0.9.1: a future release raises AssertionError, as it already does for every other invalid box. To work with "
    "both, treat an AssertionError like a False result (kornia#4013)."
)


def validate_bbox(boxes: torch.Tensor) -> bool:
    """Check that 2D box corners have equal top and bottom edge vectors, returning a bool.

    ``ok = validate_bbox(boxes)`` takes corners ``(B, 4, 2)`` or ``(B, N, 4, 2)`` and returns ``True`` when every
    coordinate is finite and, in every box, the top edge (top-left to top-right) equals the bottom edge (bottom-left
    to bottom-right) within ``1e-4`` on each axis. Otherwise it returns ``False``, without raising, and an eager call
    warns with a ``FutureWarning``. It is a check of the corner layout only; for 3D corners, use
    :func:`validate_bbox3d`, which raises instead.

    Convention:
        Vertices are inclusive, in clockwise top-left, top-right, bottom-right, bottom-left order (see
        :func:`infer_bbox_shape`). Returns ``False`` for any other shape too. Right angles, positive area and
        direction are not checked, so any parallelogram in cyclic vertex order passes. An empty batch passes.

    .. warning::
        Returning ``False`` is deprecated; a future release raises ``AssertionError``
        (`#4013 <https://github.com/kornia/kornia/issues/4013>`_).

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of ``(B, 4, 2)`` or ``(B, N, 4, 2)``, where each box is defined in the
            following ``clockwise`` order: top-left, top-right, bottom-right, bottom-left. The coordinates must be in
            the x, y order.

    Returns:
        ``True`` for valid boxes, ``False`` otherwise.

    See Also:
        - :func:`validate_bbox3d`: the check for ``(B, 8, 3)`` corners, which raises ``AssertionError`` instead of
          returning ``False``.
        - :func:`infer_bbox_shape`: the inclusive height and width of corners that pass.
        - :meth:`~kornia.geometry.boxes.Boxes.from_tensor`: imports boxes into a container, rejecting non-finite
          coordinates, and non-positive extents in the ``'xy*'`` modes, but not checking the corner layout.

    """
    if not (len(boxes.shape) in [3, 4] and boxes.shape[-2:] == torch.Size([4, 2])):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling():
            warnings.warn(_VALIDATE_BBOX_FALSE_MESSAGE, FutureWarning, stacklevel=2)
        return False

    if len(boxes.shape) == 4:
        boxes = boxes.reshape(-1, 4, 2)

    x_tl, y_tl = boxes[..., 0, 0], boxes[..., 0, 1]
    x_tr, y_tr = boxes[..., 1, 0], boxes[..., 1, 1]
    x_br, y_br = boxes[..., 2, 0], boxes[..., 2, 1]
    x_bl, y_bl = boxes[..., 3, 0], boxes[..., 3, 1]

    width_t, width_b = x_tr - x_tl + 1, x_br - x_bl + 1
    height_t, height_b = y_tr - y_tl + 1, y_br - y_bl + 1

    # Non-finite, width mismatch, height mismatch, read back in one host sync. Every failure returns False, so
    # the order in which the flags are read does not change the result.
    flags = torch.stack(
        [
            ~torch.isfinite(boxes).all(),
            (torch.abs(width_t - width_b) > 1e-4).any(),
            (torch.abs(height_t - height_b) > 1e-4).any(),
        ]
    ).cpu()
    if bool(flags.any()):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling():
            warnings.warn(_VALIDATE_BBOX_FALSE_MESSAGE, FutureWarning, stacklevel=2)
        return False

    return True


def validate_bbox3d(boxes: torch.Tensor) -> bool:
    r"""Check that 3D box corners measure one extent per axis, raising if not.

    ``ok = validate_bbox3d(boxes)`` takes corners ``(B, 8, 3)`` or ``(B, N, 8, 3)`` and returns ``True`` when, in every
    box, the ``x`` extents of the four left-to-right edges agree, the ``y`` extents of the two diagonals of the front
    face and of the back face agree, and the ``z`` extents of the four front-to-back edges agree, to the default
    tolerances of :func:`torch.isclose`; for a box with right angles these are its edge extents. Another shape or a
    disagreement raises ``AssertionError``; a non-finite coordinate returns ``False``, with a ``FutureWarning`` on an
    eager call. For 2D corners, use :func:`validate_bbox`, which returns ``False`` instead of raising.

    Convention:
        Vertices are inclusive, front-top-left, front-top-right, front-bottom-right, front-bottom-left, then the
        same four back vertices. Right angles and positive extent are not checked as such: a box whose right face is
        shifted in ``y`` raises, because its face diagonals then differ in ``y``, while one whose bottom face is
        shifted in ``x`` or ``z``, whose back face is shifted in ``x`` or ``y``, or whose right face is shifted in
        ``z`` passes. Under graph capture only the shape is checked.

    .. warning::
        Returning ``False`` is deprecated; a future release raises ``AssertionError``
        (`#4013 <https://github.com/kornia/kornia/issues/4013>`_).

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of ``(B, 8, 3)`` or ``(B, N, 8, 3)``, where each box is defined in the following ``clockwise``
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in the x, y, z order.

    Returns:
        ``True``, or ``False`` when any coordinate is non-finite. Invalid input otherwise raises instead of
        returning ``False``.

    See Also:
        - :func:`validate_bbox`: the check for ``(B, 4, 2)`` corners, which returns ``False`` instead of raising.
        - :func:`infer_bbox_shape3d`: runs this check, then returns the depths, heights and widths of the corners.
        - :meth:`~kornia.geometry.boxes.Boxes3D.from_tensor`: imports ``(N, 6)`` boxes into a container, rejecting
          non-finite coordinates and non-positive extents.

    """
    if _validate_bbox3d(boxes):
        return True
    if not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling():
        warnings.warn(_VALIDATE_BBOX3D_FALSE_MESSAGE, FutureWarning, stacklevel=2)
    return False


def _validate_bbox3d(boxes: torch.Tensor) -> bool:
    # ``validate_bbox3d`` without its deprecation warning, for kornia's own callers, which turn a ``False``
    # result into an ``AssertionError`` themselves.
    if not (len(boxes.shape) in [3, 4] and boxes.shape[-2:] == torch.Size([8, 3])):
        raise AssertionError(f"Box shape must be (B, 8, 3) or (B, N, 8, 3). Got {boxes.shape}.")

    if len(boxes.shape) == 4:
        boxes = boxes.reshape(-1, 8, 3)

    # The value checks below read the data, which graph capture cannot do; skip them under any capture
    # (torch.compile, torch.export). TorchScript cannot call is_compiling(), and a scripted call keeps the checks.
    if not torch.jit.is_scripting() and is_compiling():
        return True

    # Vertices (1, 2, 5, 6) - (0, 3, 4, 7) along x and (2, 3, 6, 7) - (0, 1, 4, 5) along y, by slicing, so a
    # call does not copy an index tensor from the host.
    x, y = boxes[..., 0], boxes[..., 1]
    widths = torch.cat([x[:, 1:3] - x[:, 0:4:3], x[:, 5:7] - x[:, 4:8:3]], 1) + 1
    heights = torch.cat([y[:, 2:4] - y[:, 0:2], y[:, 6:8] - y[:, 4:6]], 1) + 1
    depths = boxes[:, 4:, 2] - boxes[:, :4, 2] + 1

    # Non-finite coordinates are rejected as a predicate result rather than through the extent comparisons,
    # which see ``nan != nan`` and would raise a "different widths" AssertionError that names the wrong defect.
    # All four flags reach the host in one sync; ``isclose`` keeps the ``allclose`` default tolerances.
    flags = torch.stack(
        [
            ~torch.isfinite(boxes).all(),
            ~torch.isclose(widths.permute(1, 0), widths[:, 0]).all(),
            ~torch.isclose(heights.permute(1, 0), heights[:, 0]).all(),
            ~torch.isclose(depths.permute(1, 0), depths[:, 0]).all(),
        ]
    ).cpu()
    if bool(flags[0]):
        return False
    if bool(flags[1]):
        raise AssertionError(f"Boxes must have be cube, while get different widths {widths}.")
    if bool(flags[2]):
        raise AssertionError(f"Boxes must have be cube, while get different heights {heights}.")
    if bool(flags[3]):
        raise AssertionError(f"Boxes must have be cube, while get different depths {depths}.")

    return True


def infer_bbox_shape(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Read the inclusive heights and widths of (N, 4, 2) box corners from fixed vertices.

    ``heights, widths = infer_bbox_shape(boxes)`` takes ``(N, 4, 2)`` corners and returns two ``(N,)`` tensors in the
    dtype of ``boxes``, the inclusive heights ``y_br - y_tl + 1`` and widths ``x_tr - x_tl + 1`` read from the
    top-left, top-right and bottom-right vertices. It does not validate the corners. For ``(B, N, 4, 2)`` corners,
    flatten them to ``(B * N, 4, 2)`` first; for corners in any order, such as a rotated box, use
    :meth:`~kornia.geometry.boxes.Boxes.get_boxes_shape`, which measures the axis-aligned bounds; for 3D corners,
    :func:`infer_bbox_shape3d`.

    Convention:
        Vertices are **inclusive**: a box covering pixels ``0..9`` has corners at ``0`` and ``9`` and width
        ``10``. The vertex-based helpers of this module share it; :func:`nms` takes exclusive ``xyxy`` and
        :func:`transform_bbox` converts ``'xywh'`` exclusively (``xmax = xmin + width``).
        The order is clockwise top-left, top-right, bottom-right, bottom-left. The result is
        ``(heights, widths)``, read from fixed vertex indices, ``width = boxes[:, 1, 0] - boxes[:, 0, 0] + 1`` and
        ``height = boxes[:, 2, 1] - boxes[:, 0, 1] + 1``, not from a ``max - min`` reduction, so a box in
        another vertex order (such as a rotated polygon from :func:`transform_bbox`) can give a negative extent.
        :meth:`kornia.geometry.boxes.Boxes.get_boxes_shape` is reduction based.

    .. warning::
        The inclusive ``+1`` arithmetic differs from torchvision, COCO and albumentations, which are exclusive:
        `#3934 <https://github.com/kornia/kornia/issues/3934>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have shape
            ``(N, 4, 2)``, where each box is defined in the following ``clockwise`` order: top-left, top-right,
            bottom-right, bottom-left. The coordinates must be in the x, y order.

    Raises:
        ShapeError: if ``boxes`` does not have shape ``(N, 4, 2)``.

    Returns:
        - Bounding box heights, shape of ``(N,)``.
        - Bounding box widths, shape of ``(N,)``.

    See Also:
        - :func:`infer_bbox_shape3d`: the depths, heights and widths of ``(B, 8, 3)`` corners, which it validates first.
        - :meth:`~kornia.geometry.boxes.Boxes.get_boxes_shape`: the inclusive size of the axis-aligned bounds of the
          boxes in a container, for corners in any order.
        - :func:`bbox_generator`: the inverse, corners from a start corner and this height and width.

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
    r"""Read the inclusive depths, heights and widths of (B, 8, 3) box corners, validating them.

    ``depths, heights, widths = infer_bbox_shape3d(boxes)`` takes ``(B, 8, 3)`` corners, four at the front and then
    four at the back, and returns three ``(B,)`` tensors in the dtype of ``boxes``, the inclusive extent of each box
    along ``z``, ``y`` and ``x``. It first checks the corners as :func:`validate_bbox3d` does and raises
    ``AssertionError`` where that check finds extents that disagree, such as for a box whose right face is shifted in
    ``y``, or a non-finite coordinate; another shape than ``(B, 8, 3)`` raises
    :class:`~kornia.core.exceptions.ShapeError`. For ``(N, 4, 2)`` corners, use :func:`infer_bbox_shape`, which does
    not validate.

    Convention:
        Vertices are inclusive (see :func:`infer_bbox_shape`), in :func:`validate_bbox3d` order. The result is
        ``(depths, heights, widths)``, each read from fixed vertex indices like :func:`infer_bbox_shape` (for
        example ``width = boxes[:, 1, 0] - boxes[:, 0, 0] + 1``), so another vertex order can give a negative
        extent. Pass the ``'vertices_plus'``
        export of :class:`~kornia.geometry.boxes.Boxes3D`; ``'vertices'`` reads one larger per axis.
        ``(B, N, 8, 3)`` input raises :class:`~kornia.core.exceptions.ShapeError`; flatten it first.

    .. warning::
        The ``+1`` is `#3934 <https://github.com/kornia/kornia/issues/3934>`_ and the exclusive-export trap
        `#4009 <https://github.com/kornia/kornia/issues/4009>`_. An invalid box, including a non-finite
        coordinate, raises ``AssertionError`` rather than being reported:
        `#4013 <https://github.com/kornia/kornia/issues/4013>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of ``(B, 8, 3)``, where each box is defined in the following ``clockwise`` order: front-top-left,
            front-top-right, front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right,
            back-bottom-left. The coordinates must be in the x, y, z order.

    Returns:
        - Bounding box depths, shape of ``(B,)``.
        - Bounding box heights, shape of ``(B,)``.
        - Bounding box widths, shape of ``(B,)``.

    See Also:
        - :func:`infer_bbox_shape`: the heights and widths of ``(N, 4, 2)`` corners, without validation.
        - :meth:`~kornia.geometry.boxes.Boxes3D.get_boxes_shape`: the same three sizes for the boxes of a container,
          from the axis-aligned bounds of the corners.
        - :func:`bbox_generator3d`: the inverse, corners from a start corner and these sizes.

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
    # validate_bbox3d also accepts (B, N, 8, 3) and reshapes internally, but the
    # indexing below reads dim 1 as the vertex axis, so a rank-4 input would be
    # read as if the box axis were the vertices. Reject it here, the way #4218
    # did for the 2D twin.
    KORNIA_CHECK_SHAPE(boxes, ["N", "8", "3"])
    # ``validate_bbox3d`` raises for every failure it detects except a non-finite coordinate, which is a
    # ``False`` predicate result; this call site relies on the raise, so it converts that ``False`` itself.
    if not _validate_bbox3d(boxes):
        raise AssertionError("Boxes must have finite coordinates, got non-finite values.")

    left = torch.index_select(boxes, 1, torch.tensor([1, 2, 5, 6], device=boxes.device, dtype=torch.long))[:, :, 0]
    right = torch.index_select(boxes, 1, torch.tensor([0, 3, 4, 7], device=boxes.device, dtype=torch.long))[:, :, 0]
    widths = (left - right + 1)[:, 0]

    bot = torch.index_select(boxes, 1, torch.tensor([2, 3, 6, 7], device=boxes.device, dtype=torch.long))[:, :, 1]
    upper = torch.index_select(boxes, 1, torch.tensor([0, 1, 4, 5], device=boxes.device, dtype=torch.long))[:, :, 1]
    heights = (bot - upper + 1)[:, 0]

    depths = (boxes[:, 4:, 2] - boxes[:, :4, 2] + 1)[:, 0]
    return depths, heights, widths


def bbox_to_mask(
    boxes: torch.Tensor,
    width: Optional[int] = None,
    height: Optional[int] = None,
    size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Draw (B, 4, 2) inclusive box corners as filled (B, H, W) masks.

    ``mask = bbox_to_mask(boxes, size=(H, W))`` takes one box per image, ``(B, 4, 2)`` corners, and returns
    ``(B, H, W)`` masks in the dtype of ``boxes``: ``1`` on each pixel ``(x, y)`` with ``xmin <= x <= xmax`` and
    ``ymin <= y <= ymax``, read from the top-left and bottom-right corners, and ``0`` elsewhere, with no gradient path.
    The bounds are compared inclusively and unrounded, and a box partly outside the image fills only its part inside.
    For several boxes per image, use :meth:`~kornia.geometry.boxes.Boxes.to_mask`, which returns ``(N, H, W)`` for
    unbatched boxes and ``(B, N, H, W)`` for batched ones; for volumes, :func:`bbox_to_mask3d`.

    Convention:
        The size is ``size=(height, width)``, as for :func:`bbox_to_mask3d` and
        :meth:`~kornia.geometry.boxes.Boxes.to_mask`; the deprecated ``width, height`` arguments come in the
        opposite order. Only vertices 0 (top-left) and 2 (bottom-right) are read. Pass the ``'vertices_plus'`` export
        of :class:`~kornia.geometry.boxes.Boxes`. ``(B, N, 4, 2)`` raises :class:`~kornia.core.exceptions.ShapeError`.

    .. warning::
        The ``(width, height)`` form is deprecated in favour of ``size=(height, width)``:
        `#4014 <https://github.com/kornia/kornia/issues/4014>`_. For fractional
        coordinates the raw comparison differs from the rounding of :meth:`~kornia.geometry.boxes.Boxes.to_mask`
        and the truncation of :func:`bbox_to_mask3d`: `#4015 <https://github.com/kornia/kornia/issues/4015>`_.
        The exclusive-export trap is `#4009 <https://github.com/kornia/kornia/issues/4009>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of ``(B, 4, 2)``, where each box is defined in the following ``clockwise`` order: top-left, top-right,
            bottom-right and bottom-left. The coordinates must be in the x, y order.
        width: deprecated since 0.9.1; pass ``size=(height, width)`` instead. Width of the masked image.
        height: deprecated since 0.9.1; pass ``size=(height, width)`` instead. Height of the masked image.
        size: (height, width) of the masked image.

    Returns:
        the output mask tensor, shape of ``(B, height, width)`` and dtype of ``boxes``.

    Raises:
        ShapeError: if ``boxes`` does not have shape ``(B, 4, 2)``.
        ValueError: if ``size`` is passed together with ``width`` or ``height``.
        TypeError: if neither ``size`` nor both ``width`` and ``height`` are given.

    See Also:
        - :func:`bbox_to_mask3d`: the 3D form, ``(B, 1, D, H, W)`` masks of ``(B, 8, 3)`` corners, whose bounds are
          truncated toward zero.
        - :meth:`~kornia.geometry.boxes.Boxes.to_mask`: masks of several boxes per image, ``(N, H, W)`` for unbatched
          boxes and ``(B, N, H, W)`` for batched ones, filled over the rounded exclusive bounds.
        - :func:`infer_bbox_shape`: the inclusive height and width of the same corners.

    Note:
        The pixel-position grid is built in ``float32`` regardless of ``boxes.dtype``, so
        ``float16``/``bfloat16`` boxes are masked exactly even on images wider or taller than
        the integer-exact range of those dtypes (2048 px for ``float16``, 256 px for
        ``bfloat16``). The result is still returned in ``boxes.dtype``.

    Examples:
        >>> boxes = torch.tensor([[
        ...        [1., 1.],
        ...        [3., 1.],
        ...        [3., 2.],
        ...        [1., 2.],
        ...   ]])  # 1x4x2
        >>> bbox_to_mask(boxes, size=(5, 5))
        tensor([[[0., 0., 0., 0., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]]])

    """
    # `size` carries the canonical (height, width) order; `width`/`height` stay in place for one
    # release cycle in the opposite order, which is exactly the trap #4014 records, so the
    # deprecated form is resolved here, with a warning on every eager call, rather than silently transposing.
    if size is not None:
        if width is not None or height is not None:
            raise ValueError("Pass either size=(height, width) or the deprecated width and height, not both.")
        mask_height, mask_width = size
    elif width is not None and height is not None:
        if not torch.jit.is_scripting():
            _emit_deprecation_warning(
                "bbox_to_mask(boxes, width, height)",
                "bbox_to_mask(boxes, size=(height, width))",
                "0.9.1",
                "The `width` and `height` arguments will be removed in a future release; note that `size` is "
                "`(height, width)`, the order of `bbox_to_mask3d` and `Boxes.to_mask`.",
            )
        mask_height, mask_width = height, width
    else:
        raise TypeError("bbox_to_mask() missing required argument 'size' (height, width).")

    KORNIA_CHECK_SHAPE(boxes, ["B", "4", "2"])

    # NOTE: `validate_bbox`'s boolean result was previously computed here and discarded — it
    # never raised, so it performed no validation while adding a data-dependent graph break
    # (`torch.any(...)` -> Python `if`) that blocked torch.compile fullgraph (e.g. RandomErasing,
    # which builds its mask through this function). Dropped; behaviour is byte-identical.
    # zero padding the surroundings
    # Built at a fixed float32 regardless of boxes.dtype: these are exact pixel
    # indices, and float16 can only represent integers exactly up to 2048 --
    # beyond that, consecutive positions collapse onto the same value, so a
    # float16 `boxes` on an image taller/wider than 2048px silently mismasks
    # rows/columns near the collapse. The grid-vs-boxes comparisons below
    # promote to the wider of the two dtypes (float32 for float16/bfloat16/
    # float32 boxes, float64 for float64 boxes), so nothing downstream loses
    # precision; the returned mask still casts back to boxes.dtype as documented.
    yy = torch.arange(mask_height, device=boxes.device, dtype=torch.float32).view(mask_height, 1)
    xx = torch.arange(mask_width, device=boxes.device, dtype=torch.float32).view(1, mask_width)
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
    r"""Draw (B, 8, 3) inclusive 3D box corners as filled (B, 1, D, H, W) masks.

    ``mask = bbox_to_mask3d(boxes, (D, H, W))`` takes one box per volume, ``(B, 8, 3)`` corners, and returns
    ``(B, 1, D, H, W)`` masks in the dtype of ``boxes``: ``1`` on the voxels inside each box and ``0`` elsewhere, with
    no gradient path. It first checks the corners as :func:`validate_bbox3d` does and raises ``AssertionError`` where
    that check finds extents that disagree, or for a non-finite coordinate. For several boxes per volume, use
    :meth:`~kornia.geometry.boxes.Boxes3D.to_mask`; for images, :func:`bbox_to_mask`.

    Convention:
        ``size`` is ``(depth, height, width)`` and the mask ``(B, 1, depth, height, width)`` in the input
        dtype, with a channel axis that :func:`bbox_to_mask` does not have
        (:meth:`kornia.geometry.boxes.Boxes3D.to_mask` returns ``(N, depth, height, width)`` for unbatched boxes and
        ``(B, N, depth, height, width)`` for batched ones). The bounds are truncated toward zero and compared
        inclusively: pass the ``'vertices_plus'`` export. ``(B, N, 8, 3)`` input raises
        :class:`~kornia.core.exceptions.ShapeError`.

        The bounds come from fixed vertices, ``x`` from the first two, ``y`` from the second and third and ``z``
        from the first and fifth, and fill axis-aligned ranges. A sheared box that :func:`validate_bbox3d` accepts,
        such as one whose back face is shifted in ``x``, is masked as the axis-aligned box of those vertices, not
        rasterized.

    .. warning::
        For fractional coordinates the truncation differs from :func:`bbox_to_mask` and
        :meth:`~kornia.geometry.boxes.Boxes3D.to_mask`: `#4015 <https://github.com/kornia/kornia/issues/4015>`_.
        An invalid box, including a non-finite coordinate, raises ``AssertionError``:
        `#4013 <https://github.com/kornia/kornia/issues/4013>`_.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of ``(B, 8, 3)``, where each box is defined in the following ``clockwise`` order: front-top-left,
            front-top-right, front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right,
            back-bottom-left. The coordinates must be in the x, y, z order.
        size: depth, height and width of the masked image.

    Returns:
            the output mask tensor, shape of ``(B, 1, depth, height, width)`` and dtype of ``boxes``.

    See Also:
        - :func:`bbox_to_mask`: the 2D form, ``(B, H, W)`` masks of ``(B, 4, 2)`` corners with no channel axis, whose
          bounds are compared unrounded.
        - :meth:`~kornia.geometry.boxes.Boxes3D.to_mask`: masks of several boxes per volume, ``(N, D, H, W)`` for
          unbatched boxes and ``(B, N, D, H, W)`` for batched ones, filled over the rounded exclusive bounds.
        - :func:`infer_bbox_shape3d`: the inclusive depth, height and width of the same corners.

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
    # Same as infer_bbox_shape3d: boxes[:, 4, 2] below reads dim 1 as the vertex
    # axis, which a rank-4 (B, N, 8, 3) input silently is not.
    KORNIA_CHECK_SHAPE(boxes, ["B", "8", "3"])
    if not _validate_bbox3d(boxes):
        raise AssertionError("Boxes must have finite coordinates, got non-finite values.")
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

    # Intersection of the three broadcast axis slabs, computed directly with `&`. This used to
    # `|` the three slabs and recover the intersection with a three-way `all()` reduction/product;
    # that recovery breaks the moment any one slab covers a whole axis, since the union is then
    # all-true on that axis and the reduction can no longer see the other two slabs' bounds. `&`
    # needs no such recovery step -- each broadcast slab already carries its own axis's bound at
    # every position, so the elementwise intersection is exactly the answer.
    m = (
        ((z[None, :] >= z_min[:, None]) & (z[None, :] <= z_max[:, None]))[:, None, :, None, None]
        & ((y[None, :] >= y_min[:, None]) & (y[None, :] <= y_max[:, None]))[:, None, None, :, None]
        & ((x[None, :] >= x_min[:, None]) & (x[None, :] <= x_max[:, None]))[:, None, None, None, :]
    )  # Shape: (N, 1, D0, D1, D2)
    return m.to(boxes.dtype)


def bbox_generator(
    x_start: torch.Tensor, y_start: torch.Tensor, width: torch.Tensor, height: torch.Tensor
) -> torch.Tensor:
    """Build (B, 4, 2) inclusive box corners from top-left starts, widths and heights.

    ``boxes = bbox_generator(x_start, y_start, width, height)`` takes ``(B,)`` or scalar tensors and returns
    ``(B, 4, 2)`` corners in clockwise order from the top-left one, with the far corner at ``start + size - 1``, in
    the dtype of the inputs and with a gradient path to them. These are the corners
    :meth:`~kornia.geometry.boxes.Boxes.from_tensor` stores for the same boxes in ``'xywh'`` mode, which also
    validates them and wraps them in a container; to crop the boxes from images, pass the result to
    :func:`~kornia.geometry.transform.crop_and_resize`. For 3D boxes, use :func:`bbox_generator3d`.

    Convention:
        The far corner is at ``start + size - 1`` (inclusive, see :func:`infer_bbox_shape`), so
        :func:`infer_bbox_shape` reads back ``width`` and ``height``; a zero size puts the far corner one before
        the start. Scalar inputs give a batch of one, and scalar sizes broadcast against ``(B,)`` starts; scalar
        starts with ``(B,)`` sizes, ``B > 1``, raise ``RuntimeError``. The four tensors must share dtype and device,
        otherwise ``AssertionError``.

    Args:
        x_start: a tensor containing the x coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or ``(B,)``.
        y_start: a tensor containing the y coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or ``(B,)``.
        width: widths of the bounding boxes. Shape must be a scalar tensor or ``(B,)``.
        height: heights of the bounding boxes. Shape must be a scalar tensor or ``(B,)``.

    Returns:
        the bounding box tensor, shape of ``(B, 4, 2)``.

    See Also:
        - :func:`bbox_generator3d`: the 3D form, ``(B, 8, 3)`` corners from a start corner and three sizes.
        - :func:`infer_bbox_shape`: the inverse, the height and width of the corners.
        - :meth:`~kornia.geometry.boxes.Boxes.from_tensor`: the same corners from ``'xywh'`` boxes, validated and kept
          in a container with format conversions.
        - :func:`~kornia.geometry.transform.crop_and_resize`: crops one such box per image and resizes it.

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
    return torch.stack(
        [
            torch.stack([x0, y0], dim=-1),
            torch.stack([x1, y0], dim=-1),
            torch.stack([x1, y1], dim=-1),
            torch.stack([x0, y1], dim=-1),
        ],
        dim=-2,
    )


def bbox_generator3d(
    x_start: torch.Tensor,
    y_start: torch.Tensor,
    z_start: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Build (B, 8, 3) inclusive 3D box corners from start corners and sizes.

    ``boxes = bbox_generator3d(x_start, y_start, z_start, width, height, depth)`` takes ``(B,)`` or scalar tensors
    and returns ``(B, 8, 3)`` corners: four at the front, at ``z_start``, in clockwise order from the top-left one,
    then the same four at the back, at ``z_start + depth - 1``. They are in the dtype of the inputs, with a gradient
    path to them. These are the corners :meth:`~kornia.geometry.boxes.Boxes3D.from_tensor` stores for the same boxes
    in ``'xyzwhd'`` mode. For 2D boxes, use :func:`bbox_generator`.

    Convention:
        The four front vertices precede the four back vertices. Scalar inputs give a batch of one, and scalar sizes
        broadcast against ``(B,)`` starts; scalar starts with ``(B,)`` sizes, ``B > 1``, raise ``RuntimeError``. The
        six tensors must share dtype and device, otherwise ``AssertionError``.

        The far corner is at ``start + size - 1`` (inclusive, see :func:`infer_bbox_shape3d`), so
        :func:`infer_bbox_shape3d` reads back ``size`` on every axis, matching :func:`bbox_generator`.

    Args:
        x_start: a tensor containing the x coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or ``(B,)``.
        y_start: a tensor containing the y coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or ``(B,)``.
        z_start: a tensor containing the z coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or ``(B,)``.
        width: widths of the bounding boxes. Shape must be a scalar tensor or ``(B,)``.
        height: heights of the bounding boxes. Shape must be a scalar tensor or ``(B,)``.
        depth: depths of the bounding boxes. Shape must be a scalar tensor or ``(B,)``.

    Returns:
        the 3d bounding box tensor ``(B, 8, 3)``.

    See Also:
        - :func:`bbox_generator`: the 2D form, ``(B, 4, 2)`` corners from a start corner, a width and a height.
        - :func:`infer_bbox_shape3d`: the inverse, the depth, height and width of the corners.
        - :meth:`~kornia.geometry.boxes.Boxes3D.from_tensor`: the same corners from ``'xyzwhd'`` boxes, validated and
          kept in a container with format conversions.
        - :func:`~kornia.geometry.transform.crop_and_resize3d`: crops one such box per volume and resizes it; pass
          ``align_corners=True`` there.

    Examples:
        >>> x_start = torch.tensor([0, 3])
        >>> y_start = torch.tensor([1, 4])
        >>> z_start = torch.tensor([2, 5])
        >>> width = torch.tensor([10, 40])
        >>> height = torch.tensor([20, 50])
        >>> depth = torch.tensor([30, 60])
        >>> bbox_generator3d(x_start, y_start, z_start, width, height, depth)
        tensor([[[ 0,  1,  2],
                 [ 9,  1,  2],
                 [ 9, 20,  2],
                 [ 0, 20,  2],
                 [ 0,  1, 31],
                 [ 9,  1, 31],
                 [ 9, 20, 31],
                 [ 0, 20, 31]],
        <BLANKLINE>
                [[ 3,  4,  5],
                 [42,  4,  5],
                 [42, 53,  5],
                 [ 3, 53,  5],
                 [ 3,  4, 64],
                 [42,  4, 64],
                 [42, 53, 64],
                 [ 3, 53, 64]]])

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
    bbox = torch.zeros((x_start.numel(), 4, 3), device=x_start.device, dtype=x_start.dtype)

    bbox[:, :, 0] += x_start.view(-1, 1)
    bbox[:, :, 1] += y_start.view(-1, 1)
    bbox[:, :, 2] += z_start.view(-1, 1)
    bbox[:, 1, 0] += width.view(-1) - 1
    bbox[:, 2, 0] += width.view(-1) - 1
    bbox[:, 2, 1] += height.view(-1) - 1
    bbox[:, 3, 1] += height.view(-1) - 1

    # back
    bbox_back = bbox.clone()
    bbox_back[:, :, -1] += (depth.view(-1, 1) - 1).expand(-1, 4)
    return torch.cat([bbox, bbox_back], dim=1)


def transform_bbox(
    trans_mat: torch.Tensor, boxes: torch.Tensor, mode: str = "xyxy", restore_coordinates: Optional[bool] = None
) -> torch.Tensor:
    r"""Transform xyxy or xywh box tensors by 3x3 matrices through their two endpoints.

    ``out = transform_bbox(trans_mat, boxes)`` takes ``(B, 3, 3)`` matrices and ``(N, 4)`` or ``(B, N, 4)`` boxes in
    ``mode`` ``'xyxy'`` (the default) or ``'xywh'``, and returns the transformed boxes in the same mode and shape.
    Only the two endpoints ``(xmin, ymin)`` and ``(xmax, ymax)`` are transformed, and by default sorted again so that
    ``xmin <= xmax`` and ``ymin <= ymax``, so under a rotation or a shear the result is not the bounding box of the
    transformed box. ``(B, N, 4, 2)`` corners are transformed vertex by vertex instead. For the bounding box of a
    rotated box, use :meth:`~kornia.geometry.boxes.Boxes.transform_boxes`, which transforms all four corners, and
    export it with :meth:`~kornia.geometry.boxes.Boxes.to_tensor`.

    Convention:
        ``xyxy`` and ``xywh`` inputs are transformed through endpoint pairs. Coordinate
        restoration sorts their endpoints after a flip; polygon vertices retain their
        transformed cyclic order. ``'xywh'`` boxes are converted with ``xmax = xmin + width``, without the
        inclusive ``+1`` of the corner functions of this module. ``boxes`` is not modified.

    Args:
        trans_mat: The transformation matrix to be applied, with supported shape ``(B, 3, 3)``.
            For boxes shaped ``(N, 4)``, ``B`` is one or ``N``. For boxes shaped
            ``(B, N, 4)`` or ``(B, N, 4, 2)``, it is one or the boxes' ``B``.
        boxes: The boxes to be transformed with a common shape of ``(N, 4)`` or batched as ``(B, N, 4)``, the
            polygon shape of ``(B, N, 4, 2)`` is also supported.
        mode: The format in which the boxes are provided. If set to 'xyxy' the boxes are assumed to be in the format
            ``xmin, ymin, xmax, ymax``. If set to 'xywh' the boxes are assumed to be in the format
            ``xmin, ymin, width, height``
        restore_coordinates: Reorder endpoints after a flipped ``xyxy`` or ``xywh`` transform.
            Enabled by default (``None`` behaves as ``True``); pass ``False`` to preserve raw
            transformed endpoints. Polygon vertices retain their transformed order.

    Returns:
        The transformed boxes in the specified mode.

    See Also:
        - :meth:`~kornia.geometry.boxes.Boxes.transform_boxes`: transforms all four corners of the boxes in a
          container, whose ``to_tensor`` export is the bounding box of a rotated box.
        - :func:`~kornia.geometry.linalg.transform_points`: the point transform applied to the endpoints and corners.
        - :func:`~kornia.geometry.transform.warp_perspective`: warps the images by the same source-to-destination
          matrix.

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

    # Name the point count: `-1` cannot be inferred for an empty batch. Sizes read from `.shape` stay symbolic in
    # traced and exported graphs, which `unflatten` does not guarantee for the legacy ONNX exporter.
    num_points = math.prod(boxes.shape[1:]) // 2
    transformed_boxes: torch.Tensor = transform_points(trans_mat, boxes.reshape(boxes.shape[0], num_points, 2))
    transformed_boxes = transformed_boxes.view_as(boxes)

    if (restore_coordinates is None or restore_coordinates) and boxes.shape[-2:] != torch.Size([4, 2]):
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
    """Suppress overlapping (N, 4) xyxy boxes by IoU, keeping the highest-scored ones.

    ``keep = nms(boxes, scores, iou_threshold)`` takes ``N`` exclusive ``xyxy`` boxes ``(N, 4)`` and their scores
    ``(N,)``, and returns the ``int64`` indices of the boxes it keeps, highest score first: it keeps the
    highest-scored remaining box, drops every remaining box whose IoU with it exceeds ``iou_threshold``, and repeats
    until no box remains. It takes one set of boxes per call; see the notes for how it compares with
    ``torchvision.ops.nms``. To keep the local maxima of a response map rather than boxes, use
    :func:`~kornia.geometry.subpix.nms2d`.

    Convention:
        Boxes use exclusive ``xyxy`` coordinates. IoU area is
        ``(x_2 - x_1) * (y_2 - y_1)`` and the result contains kept indices rather
        than a boolean mask.

    Args:
        boxes: tensor containing exclusive ``xyxy`` bounding boxes with shape
            ``(N, 4)``, ordered as ``(x_1, y_1, x_2, y_2)``.
        scores: tensor containing the scores associated to each bounding box with shape ``(N,)``.
        iou_threshold: the threshold to discard the overlapping boxes.

    Returns:
        Indices of the boxes kept from the input set, ordered by descending score.

    Note:
        Coordinates in ``float16`` or ``bfloat16`` are computed in ``float32``, and ``uint8``, ``int8``, ``int16``
        or ``int32`` coordinates in ``int64``. In their own dtype the sum of two areas overflows: two 181-pixel
        squares exceed the ``float16`` maximum of 65504, and two 32768-pixel squares exceed ``int32``.
        ``bfloat16`` does not overflow, but its rounding can move an IoU across ``iou_threshold``.

    Note:
        For one box or more, ``torch.compile`` cannot capture ``nms`` in one graph: the number of loop iterations and
        the size of the output depend on the box values. ``fullgraph=True`` raises, and a default ``torch.compile``
        runs it eagerly, so it is outside the :ref:`torch.compile promise <export-support-promise>`.

    Note:
        For ``float32`` or ``float64`` boxes with finite coordinates, distinct scores and ``iou_threshold >= 0`` it
        keeps the same boxes, in the same order, as ``torchvision.ops.nms``. Equal scores are ordered by
        :func:`torch.sort`, which is not stable, so with ties the order, and which of two overlapping tied boxes is
        kept, can differ from torchvision. A box with a NaN coordinate is dropped by any higher-scored kept box and,
        when kept, drops every lower-scored box, where torchvision keeps them
        (`#6477 <https://github.com/kornia/kornia/issues/6477>`_).

    Note:
        Two boxes whose union is empty, such as two zero-width boxes, have an IoU of ``0``, so at an ``iou_threshold``
        of at least ``0`` neither suppresses the other, as in ``torchvision.ops.nms``. With a negative threshold the
        lower-scored one is dropped, where torchvision, whose IoU for them is NaN, keeps both.

    .. warning::
        This differs from the inclusive coordinate arithmetic used by several other
        :mod:`kornia.geometry.bbox` operations and is documented in
        `#4008 <https://github.com/kornia/kornia/issues/4008>`_.

    See Also:
        - :func:`~kornia.geometry.subpix.nms2d`: keeps the strict local maxima of a ``(B, C, H, W)`` response map, not
          boxes.
        - :meth:`~kornia.geometry.boxes.Boxes.to_tensor`: exports the boxes of a container as the exclusive
          ``'xyxy'`` boxes this function takes.

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

    if boxes.dtype in (torch.float16, torch.bfloat16):
        boxes = boxes.float()
    elif boxes.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32):
        boxes = boxes.to(torch.int64)

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
        union = areas[i] + areas[order[1:]] - inter
        # Two zero-area boxes have a zero union, and their IoU is 0 rather than 0 / 0: a NaN would compare as an
        # overlap below and suppress a disjoint box. torchvision keeps such a box too. The intersection is 0 whenever
        # the union is not positive, so dividing it by 1 there gives 0, and a NaN coordinate still gives a NaN IoU.
        ovr = inter / torch.where(union > 0, union, torch.ones_like(union))

        inds = torch.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    if len(keep) > 0:
        return torch.stack(keep)

    return boxes.new_empty((0,), dtype=torch.long)

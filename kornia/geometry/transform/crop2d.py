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

"""Crop image regions by integer slicing, by boxes or by a matrix, at a given output size.

The crop functions take channels-first images ``(B, C, H, W)`` and raise for an unbatched one. A box is
``(B, 4, 2)``: four ``(x, y)`` pixel corners, top-left, top-right, bottom-right, bottom-left, with **inclusive**
coordinates, so box ``(1, 1)``..``(2, 2)`` is a 2x2 pixel block, and the corners of an exclusive ``xyxy`` box
(torchvision) select one more row and column (`#3934 <https://github.com/kornia/kornia/issues/3934>`_). Sizes are
``(h, w)``. Except for the integer slicing of :func:`crop_by_indices` and of the default mode of
:class:`CenterCrop2D`, the crops resample by a source→destination pixel matrix through
:func:`~kornia.geometry.transform.warp_affine` or :func:`~kornia.geometry.transform.warp_perspective`, into an output
window that starts at destination pixel ``(0, 0)``. For these the matrix is in pixels, so along a source axis longer
than one pixel ``align_corners`` changes only how ``'reflection'`` padding mirrors (about the outer pixel centres
with ``True``, about the outer pixel edges with ``False``) and, in ``'nearest'`` mode, which pixel a sample exactly
halfway between two pixels takes. Along a one-pixel source axis, ``align_corners=True`` repeats that pixel where the
window should be padded (`#6433 <https://github.com/kornia/kornia/issues/6433>`_).

Which function to use:

- An axis-aligned box by integer slicing, without interpolation, resized or padded where the slices differ in size:
  :func:`crop_by_indices`.
- Any four-corner box resampled to a fixed ``(h, w)``: :func:`crop_and_resize`; mapped onto a second box whose size
  sets the output: :func:`crop_by_boxes`.
- A crop given as a source→destination pixel matrix, ``(B, 2, 3)`` or ``(B, 3, 3)``: :func:`crop_by_transform_mat`.
- The central window of each image: :func:`center_crop`, or :class:`CenterCrop2D` (module form, slicing by default).
- For volumes, use :func:`~kornia.geometry.transform.crop_and_resize3d` and the other ``3d`` crops; in an
  augmentation pipeline, :class:`~kornia.augmentation.RandomCrop`, :class:`~kornia.augmentation.RandomResizedCrop` or
  :class:`~kornia.augmentation.CenterCrop`.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from kornia.constants import Resample
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.utils import is_compiling, is_exporting
from kornia.geometry.bbox import infer_bbox_shape
from kornia.geometry.transform._crop import _compiled_slice_resize

from .affwarp import resize
from .imgwarp import get_perspective_transform, warp_affine, warp_perspective

__all__ = [
    "CenterCrop2D",
    "center_crop",
    "crop_and_resize",
    "crop_by_boxes",
    "crop_by_indices",
    "crop_by_transform_mat",
]


def crop_and_resize(
    input_tensor: torch.Tensor,
    boxes: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    r"""Crop a four-corner box from each image and resample it to a fixed output size.

    ``patches = crop_and_resize(input_tensor, boxes, size)`` takes images ``(B, C, H, W)`` and boxes ``(B, 4, 2)``, one
    per image or ``(1, 4, 2)`` for all of them, and returns the ``(B, C, h, w)`` patches for ``size = (h, w)``. The box
    corners land on the corner pixels of the patch: a rectangle is resized corner to corner, any other quadrilateral is
    rectified by the homography between the two, and the patch is differentiable with respect to the corners. A single
    image does not broadcast over a batch of boxes. For an exact integer slice without interpolation, use
    :func:`crop_by_indices`; to place the crop by a second box, :func:`crop_by_boxes`; for volumes,
    :func:`crop_and_resize3d`.

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for the ``align_corners``
        defaults and sampling rules.

        - input: ``(B, C, H, W)``; ``size`` is ``(h, w)``
        - ``boxes``: ``(B, 4, 2)`` corner points in ``(x, y)`` order
          top-left, top-right, bottom-right, bottom-left; coordinates are
          **inclusive** pixel positions (box ``(1, 1)``..``(2, 2)`` selects a
          :math:`2 \times 2` pixel block), origin at top-left
        - a box axis of one pixel is cropped exactly when ``size`` is one pixel on that axis too; one pixel in the box
          but not in ``size``, or the reverse, has no invertible matrix and returns NaN (a meaningless finite value
          with ``mode='nearest'``), and so does a box with no homography to the patch, such as a triangle
          (`#4747 <https://github.com/kornia/kornia/issues/4747>`_)

    Args:
        input_tensor: the 2D image torch.Tensor with shape ``(B, C, H, W)``.
        boxes : a torch.Tensor containing the coordinates of the bounding boxes to be extracted.
            The torch.Tensor must have the shape of ``(B, 4, 2)``, where each box is defined in the following
            (clockwise) order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y
            order.
        size: a tuple with the height and width that will be
            used to resize the extracted patches.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.

    Returns:
        the patches, with shape ``(B, C, h, w)`` for ``size = (h, w)``.

    See Also:
        - :func:`crop_by_indices`: integer slicing of axis-aligned boxes, without interpolation.
        - :func:`crop_by_boxes`: maps the box onto a second box, whose size sets the output size.
        - :func:`crop_and_resize3d`: the same for volumes and ``(B, 8, 3)`` boxes; pass ``align_corners=True`` there.
        - :func:`resize`: the whole image; for an image and a ``size`` of at least 2 pixels on each axis, bilinear
          ``crop_and_resize`` of the image's corner box equals ``resize(input, size, align_corners=True)``.
        - :class:`~kornia.augmentation.RandomResizedCrop`: random boxes resized to a fixed size, in an augmentation
          pipeline.

    Example:
        >>> input = torch.tensor([[[
        ...     [1., 2., 3., 4.],
        ...     [5., 6., 7., 8.],
        ...     [9., 10., 11., 12.],
        ...     [13., 14., 15., 16.],
        ... ]]])
        >>> boxes = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]])  # 1x4x2
        >>> crop_and_resize(input, boxes, (2, 2), mode='nearest', align_corners=True)
        tensor([[[[ 6.,  7.],
                  [10., 11.]]]])

    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Input torch.tensor type is not a torch.Tensor. Got {type(input_tensor)}")

    if not isinstance(boxes, torch.Tensor):
        raise TypeError(f"Input boxes type is not a torch.Tensor. Got {type(boxes)}")

    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"Input size must be a tuple/list of length 2. Got {size}")

    if len(input_tensor.shape) != 4:
        raise AssertionError(f"Only torch.Tensor with shape (B, C, H, W) supported. Got {input_tensor.shape}.")

    # unpack input data
    dst_h, dst_w = size

    # [x, y] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src = boxes.to(input_tensor)

    # [x, y] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst = torch.tensor(
        [[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    ).expand(points_src.shape[0], -1, -1)

    return _crop_by_boxes_to_size(
        input_tensor, points_src, points_dst, (dst_h, dst_w), mode, padding_mode, align_corners
    )


def _crop_translation(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Return the matrix of a crop, which only moves the first ``src`` vertex onto the first ``dst`` vertex.

    Solving the perspective system from the four vertices instead fails for a crop with a size-1 axis, whose
    vertices are collinear (#4751).
    """
    transform = torch.eye(3, device=src.device, dtype=src.dtype).repeat(src.shape[0], 1, 1)
    transform[:, :2, 2] = dst[:, 0] - src[:, 0]
    return transform


def _crop_scale_translation(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Return the matrix of an axis-aligned crop, which maps ``src`` vertices 0 and 2 onto ``dst`` vertices 0 and 2.

    An axis where both boxes have zero extent (a size-1 crop) is a plain translation. An axis where only one
    of them does has no invertible matrix and gives a non-finite one.
    """
    src_extent = src[:, 2] - src[:, 0]
    dst_extent = dst[:, 2] - dst[:, 0]
    both_zero = (src_extent == 0) & (dst_extent == 0)
    scale = torch.where(both_zero, torch.ones_like(src_extent), dst_extent / src_extent)
    transform = torch.eye(3, device=src.device, dtype=src.dtype).repeat(src.shape[0], 1, 1)
    transform[:, 0, 0] = scale[:, 0]
    transform[:, 1, 1] = scale[:, 1]
    transform[:, :2, 2] = dst[:, 0] - scale * src[:, 0]
    return transform


def center_crop(
    input_tensor: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    r"""Crop the central window of each image to a given size, in one call.

    ``patches = center_crop(input_tensor, size)`` takes images ``(B, C, H, W)`` and returns their central
    ``(B, C, h, w)`` windows for ``size = (h, w)``: a sub-image of the input, with the extra pixel of an odd margin at
    the bottom/right. A ``size`` larger than the input pads by ``padding_mode`` (zeros by default), again with the extra
    pixel at the bottom/right, except along a one-pixel input axis, which the default ``align_corners=True`` repeats
    (`#6433 <https://github.com/kornia/kornia/issues/6433>`_). The crop is an integer translation sampled by
    :func:`~kornia.geometry.transform.warp_perspective`, so it is exact up to float rounding. For the module form, which
    slices by default, use :class:`CenterCrop2D`; for volumes, :func:`center_crop3d`.

    Convention:
        - input: ``(B, C, H, W)`` (strictly 4D — no unbatched ``(C, H, W)``/
          ``(H, W)`` input is accepted); ``size`` is ``(h, w)``
        - offset: the crop starts at the integer pixel ``int(H / 2 - h / 2)`` (likewise for the width), as in
          :class:`CenterCrop2D`, so the output is a sub-image of the input. With an odd margin the extra pixel is
          left at the bottom/right; for a ``size`` larger than the input, the extra padding pixel goes there too.

    Args:
        input_tensor: the 2D image torch.Tensor with shape ``(B, C, H, W)``.
        size: a tuple with the expected height and width
          of the output patch.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.

    Returns:
        the central windows, with shape ``(B, C, h, w)``.

    See Also:
        - :class:`CenterCrop2D`: the module form; slices with integer indices by default instead of resampling.
        - :func:`center_crop3d`: the central window of each volume; keep its default ``align_corners=True``.
        - :func:`crop_and_resize`: a window at any box, resampled to a fixed size.
        - :class:`~kornia.augmentation.CenterCrop`: the same windows as an augmentation, for augmentation pipelines;
          it raises for a ``size`` larger than the input instead of padding.

    Examples:
        >>> input = torch.tensor([[[
        ...     [1., 2., 3., 4.],
        ...     [5., 6., 7., 8.],
        ...     [9., 10., 11., 12.],
        ...     [13., 14., 15., 16.],
        ...  ]]])
        >>> center_crop(input, (2, 4), mode='nearest', align_corners=True)
        tensor([[[[ 5.,  6.,  7.,  8.],
                  [ 9., 10., 11., 12.]]]])

    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Input torch.tensor type is not a torch.Tensor. Got {type(input_tensor)}")

    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"Input size must be a tuple/list of length 2. Got {size}")

    if len(input_tensor.shape) != 4:
        raise AssertionError(f"Only torch.Tensor with shape (B, C, H, W) supported. Got {input_tensor.shape}.")

    # unpack input sizes
    dst_h, dst_w = size
    src_h, src_w = input_tensor.shape[-2:]

    # compute start/end offsets
    dst_h_half: float = dst_h / 2
    dst_w_half: float = dst_w / 2
    src_h_half: float = src_h / 2
    src_w_half: float = src_w / 2

    # an integer start, as CenterCrop2D takes it: a fractional one would resample between pixels
    start_x: int = int(src_w_half - dst_w_half)
    start_y: int = int(src_h_half - dst_h_half)

    end_x: int = start_x + dst_w - 1
    end_y: int = start_y + dst_h - 1

    # [y, x] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src: torch.Tensor = torch.tensor(
        [[[start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y]]],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    # [y, x] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: torch.Tensor = torch.tensor(
        [[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    ).expand(points_src.shape[0], -1, -1)

    transform = _crop_translation(points_src, points_dst)

    return crop_by_transform_mat(
        input_tensor, transform, (dst_h, dst_w), mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )


def crop_by_boxes(
    input_tensor: torch.Tensor,
    src_box: torch.Tensor,
    dst_box: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    validate_boxes: bool = True,
) -> torch.Tensor:
    """Crop images by mapping a source box onto a destination box, whose size sets the output size.

    ``patches = crop_by_boxes(input_tensor, src_box, dst_box)`` takes images ``(B, C, H, W)`` and two boxes per image,
    ``(B, 4, 2)`` each (both ``(1, 4, 2)`` for all images), warps each image by the homography that maps ``src_box``
    onto ``dst_box`` and returns the ``(B, C, h, w)`` window at destination pixels ``(0, 0)``..``(w - 1, h - 1)``, where
    ``(h, w)`` is the inclusive size of ``dst_box``. Every ``dst_box`` in the batch must have that same size, since the
    output is one tensor, and a ``dst_box`` that does not start at ``(0, 0)`` shifts the crop by its offset, partly out
    of the window. The size is read from the ``dst_box`` values on the host. When ``dst_box`` is the window
    ``(0, 0)``..``(w - 1, h - 1)`` itself and ``(h, w)`` is known, :func:`crop_and_resize` with ``size=(h, w)`` gives
    the same crop without that read; it always places the destination box at the origin, so it cannot reproduce an
    offset ``dst_box``. For volumes, use :func:`crop_by_boxes3d`.

    Convention:
        - input: ``(B, C, H, W)``
        - ``src_box``/``dst_box``: ``(B, 4, 2)`` corner points in ``(x, y)`` order
          top-left, top-right, bottom-right, bottom-left — same convention as
          :func:`crop_and_resize`; ``dst_box`` determines the output resolution
        - a single box (batch size 1) broadcasts over a batch of images, but a single
          image does not broadcast over a batch of boxes; ``src_box`` and ``dst_box`` have the same batch size
        - a box axis of one pixel in one box but not in the other returns NaN (a meaningless finite value with
          ``mode='nearest'``), as in :func:`crop_and_resize`
          (`#4747 <https://github.com/kornia/kornia/issues/4747>`_)

    Args:
        input_tensor: the 2D image torch.Tensor with shape ``(B, C, H, W)``.
        src_box: a torch.Tensor with shape ``(B, 4, 2)`` containing the coordinates of the bounding boxes
            to be extracted, where each box is defined
            in the clockwise order: top-left, top-right, bottom-right and bottom-left.
            The coordinates must be in x, y order.
        dst_box: a torch.Tensor with shape ``(B, 4, 2)`` containing the coordinates of the bounding boxes
            to be placed, where each box is defined
            in the clockwise order: top-left, top-right, bottom-right and bottom-left.
            The coordinates must be in x, y order.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.
        validate_boxes: has no effect: the boxes are not validated
          (`#6401 <https://github.com/kornia/kornia/issues/6401>`_).

    Returns:
        the patches, with shape ``(B, C, h, w)`` for the size ``(h, w)`` of ``dst_box``.

    Raises:
        AssertionError: if the boxes of ``dst_box`` differ in size.

    See Also:
        - :func:`crop_and_resize`: takes the output size as ``size`` and places the box at the patch corners.
        - :func:`crop_by_transform_mat`: takes the source→destination matrix itself.
        - :func:`crop_by_boxes3d`: the same for volumes and ``(B, 8, 3)`` boxes.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
        >>> src_box = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]])  # 1x4x2
        >>> dst_box = torch.tensor([[
        ...     [0., 0.],
        ...     [1., 0.],
        ...     [1., 1.],
        ...     [0., 1.],
        ... ]])  # 1x4x2
        >>> crop_by_boxes(input, src_box, dst_box, align_corners=True)
        tensor([[[[ 5.0000,  6.0000],
                  [ 9.0000, 10.0000]]]])

    """
    bbox: Tuple[torch.Tensor, torch.Tensor] = infer_bbox_shape(dst_box)
    if not ((bbox[0] == bbox[0][0]).all() and (bbox[1] == bbox[1][0]).all()):
        raise AssertionError(
            f"Cropping height, width and depth must be exact same in a batch. Got height {bbox[0]} and width {bbox[1]}."
        )

    h_out: int = int(bbox[0][0].item())
    w_out: int = int(bbox[1][0].item())

    return _crop_by_boxes_to_size(input_tensor, src_box, dst_box, (h_out, w_out), mode, padding_mode, align_corners)


def _crop_by_boxes_to_size(
    input_tensor: torch.Tensor,
    src_box: torch.Tensor,
    dst_box: torch.Tensor,
    out_size: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    # ``crop_by_boxes`` with the output size already known as Python ints. ``crop_and_resize`` builds
    # ``dst_box`` from ``size``, so it takes this path and never reads the box values back from the
    # device -- which also keeps it capturable by ``torch.onnx.export``.
    if len(input_tensor.shape) != 4:
        raise AssertionError(f"Only torch.Tensor with shape (B, C, H, W) supported. Got {input_tensor.shape}.")

    # compute transformation between points and warp
    # Note: torch.Tensor.dtype must be float. "solve_cpu" not implemented for 'Long'
    src_box = src_box.to(input_tensor)
    dst_box = dst_box.to(input_tensor)
    dst_trans_src: torch.Tensor = get_perspective_transform(src_box, dst_box)
    # A box with a size-1 axis has collinear vertices and the solve returns NaN for it (#4747). Fall back to
    # the matrix built from the box extents there; every other box keeps the solved matrix.
    solved = dst_trans_src.isfinite().all(dim=-1).all(dim=-1)
    size_one = ((src_box[:, 2] - src_box[:, 0]) == 0).any(dim=-1) | ((dst_box[:, 2] - dst_box[:, 0]) == 0).any(dim=-1)
    fallback = (~solved & size_one)[:, None, None]
    # ``torch.where`` also differentiates the branch it discards, so a box that keeps its solved matrix builds the
    # extent matrix from a unit source square: its own vertices 0 and 2 can share a coordinate (a square turned by
    # 45 degrees), and dividing by that zero extent would make the gradient with respect to the box NaN.
    unit = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=src_box.device, dtype=src_box.dtype)
    extent_matrix = _crop_scale_translation(torch.where(fallback, src_box, unit), dst_box)
    dst_trans_src = torch.where(fallback, extent_matrix, dst_trans_src)

    return crop_by_transform_mat(
        input_tensor, dst_trans_src, out_size, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )


def crop_by_transform_mat(
    input_tensor: torch.Tensor,
    transform: torch.Tensor,
    out_size: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Crop images by a source→destination pixel matrix, affine or homography, into a given output size.

    ``patches = crop_by_transform_mat(input_tensor, transform, out_size)`` takes images ``(B, C, H, W)`` and a
    source→destination pixel matrix per image, affine ``(B, 2, 3)`` or homography ``(B, 3, 3)`` (a batch of 1 serves
    every image), and returns the ``(B, C, h, w)`` destination window starting at pixel ``(0, 0)`` for
    ``out_size = (h, w)``. It calls :func:`~kornia.geometry.transform.warp_affine` for an affine matrix and
    :func:`~kornia.geometry.transform.warp_perspective` for a homography, after casting the matrix to the image dtype
    and expanding it to the batch: a float64 or integer matrix crops a float32 image, which those two reject, and a
    batch-1 homography serves a batch, which ``warp_perspective`` alone does not allow. To build the matrix from boxes,
    use :func:`crop_by_boxes` or :func:`crop_and_resize`; for volumes, :func:`crop_by_transform_mat3d`.

    Convention:
        - input: ``(B, C, H, W)``; ``out_size`` is ``(h, w)``
        - ``transform`` is the source→destination **pixel** transform, accepted as
          either ``(B, 2, 3)`` affine or ``(B, 3, 3)`` homogeneous; dispatch is
          by shape — ``(B, 2, 3)`` takes the cheaper :func:`warp_affine` path,
          while ``(B, 3, 3)`` takes :func:`warp_perspective` and uses the **full**
          matrix, so a non-trivial third (projective) row changes the output; an
          ``out_size`` dimension of ``1`` is handled like any other size

    Args:
        input_tensor: the 2D image torch.Tensor with shape ``(B, C, H, W)``.
        transform: a perspective transformation matrix with shape ``(B, 3, 3)``, or an
          affine matrix with shape ``(B, 2, 3)`` that takes the cheaper warp_affine path.
        out_size: size of the output image (height, width).
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode (str): padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.

    Returns:
        the patches, with shape ``(B, C, h, w)``.

    See Also:
        - :func:`warp_affine`: the affine warp it calls, with a ``fill_value`` for the padding.
        - :func:`warp_perspective`: the homography warp it calls, with a ``fill_value`` for the padding.
        - :func:`crop_by_boxes`: builds the homography from a source and a destination box.
        - :func:`crop_by_transform_mat3d`: the same for volumes, by ``(B, 3, 4)`` or ``(B, 4, 4)`` matrices.

    """
    # simulate broadcasting
    dst_trans_src = torch.as_tensor(
        transform.expand(input_tensor.shape[0], -1, -1), device=input_tensor.device, dtype=input_tensor.dtype
    )

    # dispatch on the transform shape, which is static and therefore safe for torch.export:
    # (B, 2, 3) is affine by construction and keeps the cheaper warp_affine path (no perspective
    # divide, shared-grid broadcast), while (B, 3, 3) takes the full perspective warp.
    if transform.shape[-2:] == (2, 3):
        return warp_affine(
            input_tensor,
            dst_trans_src,
            out_size,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    patches: torch.Tensor = warp_perspective(
        input_tensor,
        dst_trans_src,
        out_size,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    return patches


def _crop_inverse_by_transform_mat(
    input_tensor: torch.Tensor,
    transform: torch.Tensor,
    out_size: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """:func:`crop_by_transform_mat` for warping a crop back onto its canvas, also from a one-pixel axis.

    ``grid_sample`` cannot place a source axis of length 1: with ``align_corners=True`` every sample lands on its
    only pixel, so the whole canvas axis gets its value, and with ``False`` it blends it into the next pixel. One
    pixel of padding on each side of such an axis, in the warp's own padding (a reflected one-pixel axis repeats
    its pixel), gives the sampler a neighbour to interpolate towards, and ``transform`` is shifted to match.
    ``transform`` is the ``(B, 3, 3)`` crop-to-canvas pixel matrix.
    """
    pad_h = int(input_tensor.shape[-2] == 1)
    pad_w = int(input_tensor.shape[-1] == 1)
    if pad_h or pad_w:
        pad_mode = "constant" if padding_mode == "zeros" else "replicate"
        input_tensor = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=pad_mode)
        shift = torch.eye(3, device=transform.device, dtype=transform.dtype)
        shift[0, 2] = -pad_w
        shift[1, 2] = -pad_h
        transform = transform @ shift
    return crop_by_transform_mat(input_tensor, transform[:, :2, :], out_size, mode, padding_mode, align_corners)


def crop_by_indices(
    input_tensor: torch.Tensor,
    src_box: torch.Tensor,
    size: Optional[Tuple[int, int]] = None,
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    antialias: bool = False,
    shape_compensation: str = "resize",
) -> torch.Tensor:
    """Crop axis-aligned boxes by integer slicing, then resize or pad each crop to a common size.

    ``patches = crop_by_indices(input_tensor, src_box, size)`` takes images ``(B, C, H, W)`` and one box per image,
    ``(B, 4, 2)``, slices each box out with integer indices and returns ``(B, C, h, w)`` for ``size = (h, w)``. Outside
    graph export (see the notes) it slices by the top-left corner, the right edge (``x`` of vertex 1) and the bottom
    edge (``y`` of vertex 3) of a box, and truncates float coordinates toward zero; without ``size`` it takes the size
    from the box as :func:`~kornia.geometry.bbox.infer_bbox_shape` does, whose height reads ``y`` of vertex 2, and every
    box must have the same size. A slice of another size than ``size``, from a box of another size or one clipped by the
    image border, is resized (``shape_compensation='resize'``) or zero-padded or trimmed to it (``'pad'``).
    The crop is differentiable with respect to the image, not the box: for sub-pixel, rotated or differentiable boxes,
    use :func:`crop_and_resize`.

    Convention:
        - input: ``(B, C, H, W)``; ``size`` is ``(h, w)`` if given, else inferred
          from ``src_box``; a single box does not broadcast over a batch of images
        - ``src_box``: ``(B, 4, 2)`` corner points in ``(x, y)`` order top-left,
          top-right, bottom-right, bottom-left — same convention as :func:`crop_and_resize`
        - unlike the other crop operators in this module: ``interpolation=`` (not
          ``mode=``), ``align_corners=None`` by default (not ``True``), and an
          ``antialias=False`` option
        - ``shape_compensation`` (``'resize'`` by default) applies whenever the cropped
          slice does not match ``size``, whether or not ``src_box`` is identical across
          the batch — each row's output depends only on its own box. It must be ``'resize'``,
          ``'pad'``, or ``'F.pad'`` (a legacy alias for ``'pad'``).
        - the part of a box outside the image is clipped on every side, a box that starts left of or above the
          image included, and ``shape_compensation`` then applies. With ``'pad'`` the padding goes on the side
          where the box overflowed, so every kept pixel stays at its position in the box.

    .. note::
        Under :func:`torch.compile` a non-empty batch with ``size`` given, ``'bilinear'``,
        ``'bicubic'`` or ``'nearest'`` interpolation, ``antialias=False`` and
        ``shape_compensation='resize'`` is cropped by a gather that keeps the box coordinates
        on the device, so it traces with ``fullgraph=True`` and matches the eager result up to
        float rounding. Every other call keeps the eager path, which reads the box coordinates
        back to Python and breaks the graph. Graph export is described in the note below.

    Args:
        input_tensor: the 2D image torch.Tensor with shape ``(B, C, H, W)``.
        src_box: a torch.Tensor with shape ``(B, 4, 2)`` containing the coordinates of the bounding boxes
            to be extracted, where each box is defined
            in the clockwise order: top-left, top-right, bottom-right and bottom-left.
            The coordinates must be in x, y order.
        size: output size. An auto resize or F.pad will be performed according to ``shape_compensation``
            if the cropped slice sizes are not exactly align `size`.
            If None, will auto-infer from src_box.
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'nearest-exact'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'area'``.
        align_corners: interpolation flag.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.
        shape_compensation: how to compensate when the cropped slice size differs from `size`: ``'resize'``
            resamples, while ``'pad'`` pads or trims. ``'F.pad'`` is a legacy alias for ``'pad'``.

    Returns:
        The cropped torch.Tensor with shape ``(B, C, h, w)``, where
        ``(h, w)`` is ``size`` if given, otherwise the shape inferred
        from ``src_box``.

    Raises:
        ValueError: if ``size`` is ``None`` and the boxes differ in size.

    See Also:
        - :func:`crop_and_resize`: resamples any four-corner box, differentiably with respect to its corners.
        - :class:`CenterCrop2D`: slices the central window with this function in its default mode, unless the size
          exceeds the input on an axis, where it zero-pads and slices directly.
        - :class:`~kornia.augmentation.RandomCrop`: random boxes, sliced with this function in its default mode, in an
          augmentation pipeline.

    .. note::
        Under graph export (:func:`torch.export.export` / :func:`torch.onnx.export`) the box
        coordinates cannot be read back to Python, so the crop is captured as the same warp as
        :func:`crop_and_resize` (``align_corners=True``), which keeps the box coordinates dynamic.
        This needs ``size``: a box that is exactly ``size`` pixels (what
        :class:`~kornia.augmentation.RandomCrop` and :class:`CenterCrop2D` produce) is reproduced
        exactly, any other box is resampled to ``size`` with ``align_corners=True`` regardless of
        ``shape_compensation`` and ``align_corners``. Without ``size`` the eager path is kept: it
        reads the box coordinates back to Python, which breaks the graph under
        :func:`torch.compile` and is rejected by ``torch.export`` as data-dependent.

    """
    KORNIA_CHECK_SHAPE(input_tensor, ["B", "C", "H", "W"])
    KORNIA_CHECK_SHAPE(src_box, ["B", "4", "2"])
    if shape_compensation not in ("resize", "pad", "F.pad"):
        raise ValueError(
            f"`shape_compensation` must be one of 'resize', 'pad', or 'F.pad'. Got {shape_compensation!r}."
        )

    B, C, _, _ = input_tensor.shape
    # Kept on the device ``src_box`` lives on: the coordinates are read back to Python below, which costs no device
    # synchronization for the CPU boxes the augmentations pass. A float box truncates to the same integers either way:
    # the former copy to a CUDA input's device also cast on the host.
    src = torch.as_tensor(src_box, dtype=torch.long)
    x1 = src[:, 0, 0]
    x2 = src[:, 1, 0] + 1
    y1 = src[:, 0, 1]
    y2 = src[:, 3, 1] + 1

    if size is not None and is_exporting():
        return _crop_by_indices_export(input_tensor, src_box, size, interpolation)
    if (
        size is not None
        and is_compiling()
        and B > 0
        and (interpolation in ("bilinear", "bicubic") or (interpolation == "nearest" and align_corners is None))
        and not antialias
        and shape_compensation == "resize"
    ):
        # ``_compiled_slice_resize`` implements only these three modes (any other one would be resampled as
        # bicubic) and cannot gather from an empty batch; ``interpolate`` rejects ``align_corners`` for nearest.
        return _compiled_slice_resize(input_tensor, src_box, size, interpolation, align_corners)

    # ``RandomCrop`` passes its ``size`` through as given, and a list never compares equal to a ``torch.Size``:
    # every slice would look mismatched and be resized to its own size.
    if size is not None:
        size = (size[0], size[1])

    # Move the four coordinate columns to Python in a single device sync (one ``tolist`` over a
    # stacked tensor) instead of a ``unique`` per column plus a device-to-host ``int(...)`` inside
    # every loop iteration — the coordinates index Python-level slicing, so they must be host ints.
    x1l, x2l, y1l, y2l = torch.stack([x1, x2, y1, y2], dim=0).tolist()
    # A negative index would wrap to the far edge, so the part of a box left of or above the image is clipped,
    # like the part past the right or bottom edge, which the slice clips. ``'pad'`` pads it back on that side.
    left = [max(-v, 0) for v in x1l]
    top = [max(-v, 0) for v in y1l]
    x1l, x2l = [max(v, 0) for v in x1l], [max(v, 0) for v in x2l]
    y1l, y2l = [max(v, 0) for v in y1l], [max(v, 0) for v in y2l]
    # With 'pad' a clipped box needs that padding even when its clipped slice happens to have the requested size.
    shifted = [shape_compensation != "resize" and (left[i] > 0 or top[i] > 0) for i in range(B)]

    # ``B > 0``: an empty batch has no first box to compare against, and takes the general path below,
    # which returns an empty ``(0, C, *size)`` tensor (#4429).
    if (
        B > 0
        and x1l.count(x1l[0]) == B
        and x2l.count(x2l[0]) == B
        and y1l.count(y1l[0]) == B
        and y2l.count(y2l[0]) == B
        and left.count(left[0]) == B
        and top.count(top[0]) == B
    ):
        out = input_tensor[..., y1l[0] : y2l[0], x1l[0] : x2l[0]]
        if size is not None and (out.shape[-2:] != size or shifted[0]):
            if shape_compensation == "resize":
                return resize(
                    out,
                    size,
                    interpolation=interpolation,
                    align_corners=align_corners,
                    side="short",
                    antialias=antialias,
                )
            return _pad_to_size(out, size, left[0], top[0])

    if size is None:
        h, w = infer_bbox_shape(src)
        if B > 0 and ((h != h[0]).any() | (w != w[0]).any()).item():
            raise ValueError(
                "All boxes in the batch must have the same height and width when `size` is None. "
                "Please pass `size` explicitly when box dimensions vary across the batch."
            )
        size = (int(h[0].item()), int(w[0].item())) if B > 0 else (0, 0)
    # Unbind only when batch-slice gradients would allocate a full input buffer for every crop. On that
    # path the rows are also joined with one ``cat``: writing them into ``out`` row by row would make the
    # backward clone the gradient of the whole output once per row.
    rows = input_tensor.unbind(0) if B > 1 and input_tensor.requires_grad and torch.is_grad_enabled() else None
    slices = [
        input_tensor[i : i + 1, :, y1l[i] : y2l[i], x1l[i] : x2l[i]]
        if rows is None
        else rows[i][None, :, y1l[i] : y2l[i], x1l[i] : x2l[i]]
        for i in range(B)
    ]
    # When every row's slice already has the requested size (always the case for RandomCrop), join the
    # views with one ``cat``: one copy kernel for the batch instead of a ``copy_`` per row (#4531).
    # ``contiguous`` keeps the result in the default memory format, as the ``torch.empty`` below does,
    # when the input is channels-last. MPS keeps the per-row copies: on an M1 the ``cat`` of strided
    # views ran at 0.75-0.96x their speed without gradients (B = 4-64, 256² -> 128², #4987).
    if (
        B > 0
        and input_tensor.device.type != "mps"
        and not any(shifted)
        and all(crop.shape[-2:] == size for crop in slices)
    ):
        return torch.cat(slices).contiguous()
    out = torch.empty(B, C, *size, device=input_tensor.device, dtype=input_tensor.dtype) if rows is None else None
    crops: list[torch.Tensor] = []
    for i, _out in enumerate(slices):
        if _out.shape[-2:] != size or shifted[i]:
            if shape_compensation == "resize":
                _out = resize(
                    _out,
                    size,
                    interpolation=interpolation,
                    align_corners=align_corners,
                    side="short",
                    antialias=antialias,
                )
            else:
                _out = _pad_to_size(_out, size, left[i], top[i])
        if out is None:
            crops.append(_out)
        else:
            out[i] = _out
    if out is None:
        return torch.cat(crops).contiguous()
    return out


def _pad_to_size(crop: torch.Tensor, size: Tuple[int, int], left: int, top: int) -> torch.Tensor:
    # ``left`` and ``top`` are the box columns and rows clipped before the image, so every kept pixel stays at its
    # position in the box; the rest of ``size`` is padded (or, for a larger slice, trimmed) at the right and bottom.
    # A box entirely outside the image clips more than ``size``: it is all padding, as past the right edge.
    left, top = min(left, size[1]), min(top, size[0])
    return F.pad(crop, [left, size[1] - crop.shape[-1] - left, top, size[0] - crop.shape[-2] - top])


def _crop_by_indices_export(
    input_tensor: torch.Tensor,
    src_box: torch.Tensor,
    size: Tuple[int, int],
    interpolation: str,
) -> torch.Tensor:
    # Export path of ``crop_by_indices``: the box is resampled onto a ``size`` grid through the same
    # warp as :func:`crop_and_resize`, so the box coordinates stay graph inputs instead of being read
    # back to Python. Sampling at ``align_corners=True`` reproduces the integer slice exactly when the
    # box is already ``size`` pixels and bilinearly resizes it otherwise.
    if interpolation not in ("bilinear", "nearest", "bicubic"):
        raise ValueError(
            f"`crop_by_indices` can export with bilinear, nearest or bicubic interpolation. Got {interpolation}."
        )
    dst_h, dst_w = size
    points_dst = torch.tensor(
        [[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    ).expand(src_box.shape[0], -1, -1)
    return _crop_by_boxes_to_size(
        input_tensor, src_box.to(input_tensor), points_dst, size, interpolation, "zeros", align_corners=True
    )


class CenterCrop2D(nn.Module):
    """Crop the central window of each image to a given size, by slicing or resampling (module form).

    Build ``cropper = CenterCrop2D(size)`` with an ``int`` or an ``(h, w)`` tuple, then ``out = cropper(input)`` returns
    the central ``(B, C, h, w)`` windows of ``(B, C, H, W)`` images, at the offsets of :func:`center_crop`. The default
    ``cropping_mode='slice'`` slices with integer indices, through :func:`crop_by_indices`, or after zero-padding when
    ``size`` exceeds the input on an axis; ``'resample'`` samples the same windows, up to float rounding, by a
    translation through :func:`~kornia.geometry.transform.warp_affine` with the ``resample`` mode, except that with
    ``align_corners=True`` it repeats a one-pixel input axis that ``'slice'`` zero-pads
    (`#6433 <https://github.com/kornia/kornia/issues/6433>`_). For one call, use
    :func:`center_crop`; in an augmentation pipeline, :class:`~kornia.augmentation.CenterCrop`.

    Convention:
        - ``align_corners`` (and ``resample``) only take effect when
          ``cropping_mode='resample'``; the default ``cropping_mode='slice'`` performs
          integer-index slicing and ignores both
        - See the convention block of :func:`~kornia.geometry.transform.center_crop`.

    Args:
        size: Size ``(h, w)`` in pixels of the cropped region, or one ``int`` for both sides; a list raises
          ``TypeError``.
        align_corners: interpolation flag.
        resample: Resampling mode.
        cropping_mode: Cropping mode, "resample" or "slice".

    Note:
        ``torch.jit.script`` compiles neither cropping mode, and ``torch.jit.trace`` records the crop offsets of the
        traced input size.

    See Also:
        - :func:`center_crop`: the one-call form, which always resamples.
        - :func:`crop_by_indices`: the integer slicing of the default mode, for any axis-aligned box.
        - :class:`~kornia.augmentation.CenterCrop`: the same windows as an augmentation, for augmentation pipelines;
          it raises for a ``size`` larger than the input instead of padding.

    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        align_corners: bool = True,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        cropping_mode: str = "slice",
    ) -> None:
        super().__init__()
        if isinstance(size, tuple):
            self.size = (size[0], size[1])
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise TypeError(f"Invalid size type. Expected int or tuple[int, int]. Got: {type(size)}.")

        dst_h, dst_w = self.size
        points_dst = torch.tensor([[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]], dtype=torch.long)
        self.register_buffer("points_src", points_dst.clone())
        self.register_buffer("points_dst", points_dst)

        self.flags = {
            "resample": Resample.get(resample),
            "cropping_mode": cropping_mode,
            "align_corners": align_corners,
            "size": self.size,
            "padding_mode": "zeros",
        }

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Crop the central region of each input image to ``self.size``.

        Depending on ``cropping_mode``, the crop is produced either by
        perspective-resampling (subpixel/interpolated) or by direct slicing.

        Args:
            input: Image batch with shape ``(B, C, H, W)``, where
                :math:`B` is batch size, :math:`C` is channels, :math:`H` is
                source height, and :math:`W` is source width.

        Returns:
            Center-cropped tensor with shape ``(B, C, h, w)``, where
            :math:`h` and :math:`w` are the requested crop height and width
            stored in ``self.size``.
        """
        batch_size = input.shape[0]

        dst_h, dst_w = self.size
        src_h, src_w = input.shape[-2:]

        dst_h_half, dst_w_half = dst_h / 2, dst_w / 2
        src_h_half, src_w_half = src_h / 2, src_w / 2

        start_x, start_y = int(src_w_half - dst_w_half), int(src_h_half - dst_h_half)
        end_x, end_y = start_x + dst_w - 1, start_y + dst_h - 1

        # [y, x] origin
        # top-left, top-right, bottom-right, bottom-left
        self.points_src[0, 0, 0] = start_x
        self.points_src[0, 0, 1] = start_y
        self.points_src[0, 1, 0] = end_x
        self.points_src[0, 1, 1] = start_y
        self.points_src[0, 2, 0] = end_x
        self.points_src[0, 2, 1] = end_y
        self.points_src[0, 3, 0] = start_x
        self.points_src[0, 3, 1] = end_y

        if self.flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            transform = _crop_translation(
                self.points_src.expand(batch_size, -1, -1).to(input),
                self.points_dst.expand(batch_size, -1, -1).to(input),
            )
            transform = transform.expand(batch_size, -1, -1)

            return crop_by_transform_mat(
                input,
                transform[:, :2, :],
                self.size,
                self.flags["resample"].name.lower(),
                "zeros",
                self.flags["align_corners"],
            )

        if self.flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            if dst_h > src_h or dst_w > src_w:
                # A size larger than the input: zero-pad around the image, as the resample mode does, then slice.
                # The offsets are Python ints from the shapes, so this compiles with fullgraph=True.
                left, top = max(-start_x, 0), max(-start_y, 0)
                padded = F.pad(input, [left, max(end_x + 1 - src_w, 0), top, max(end_y + 1 - src_h, 0)])
                return padded[..., start_y + top : end_y + 1 + top, start_x + left : end_x + 1 + left].contiguous()
            return crop_by_indices(input, self.points_src.expand(batch_size, -1, -1).to(input), self.flags["size"])

        raise NotImplementedError(f"Not supported type: {self.flags['cropping_mode']}.")

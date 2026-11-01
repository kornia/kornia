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

"""Crop volume regions by boxes or by a matrix, resampled to a given output size.

The crops take channels-first volumes ``(B, C, D, H, W)``. A box is ``(B, 8, 3)``: eight ``(x, y, z)`` voxel
corners, the front face (smaller ``z``) and then the back face, each face top-left, top-right, bottom-right,
bottom-left, with **inclusive** coordinates as in the 2D crops of
:func:`~kornia.geometry.transform.crop_and_resize`. Sizes are ``(d, h, w)``. Every crop resamples by a
source→destination voxel matrix through :func:`~kornia.geometry.transform.warp_affine3d`, into an output window that
starts at voxel ``(0, 0, 0)``. Pass ``align_corners=True``: at ``False`` the matrix is normalized for corner-aligned
voxels while ``grid_sample`` reads half-voxel ones, so the samples miss the box, and even a box of the output size is
interpolated rather than copied (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). :func:`crop_and_resize3d`
and :func:`crop_by_boxes3d` default to ``False``. In eager mode the box crops reject, with ``AssertionError``, a box
whose four edges along an axis span different extents on it, one with a non-finite coordinate, and a pair of boxes
that no affine map relates, such as a box that is not a parallelepiped in :func:`crop_and_resize3d`. They raise
``torch.linalg.LinAlgError`` for a box or an output size with a one-voxel axis
(`#4747 <https://github.com/kornia/kornia/issues/4747>`_).

Which function to use:

- A box resampled to a fixed ``(d, h, w)``: :func:`crop_and_resize3d`; mapped onto a second box whose size sets the
  output: :func:`crop_by_boxes3d`.
- A crop given as a source→destination voxel matrix, ``(B, 3, 4)`` or ``(B, 4, 4)``: :func:`crop_by_transform_mat3d`.
- The central window of each volume: :func:`center_crop3d`.
- For images, use :func:`~kornia.geometry.transform.crop_and_resize` and the other 2D crops; in an augmentation
  pipeline, :class:`~kornia.augmentation.RandomCrop3D` or :class:`~kornia.augmentation.CenterCrop3D`.
"""

from typing import Tuple

import torch

from kornia.core.utils import is_compiling
from kornia.geometry.bbox import _validate_bbox3d, infer_bbox_shape3d

from .imgwarp import get_perspective_transform3d, warp_affine3d

__all__ = ["center_crop3d", "crop_and_resize3d", "crop_by_boxes3d", "crop_by_transform_mat3d"]


def crop_and_resize3d(
    tensor: torch.Tensor,
    boxes: torch.Tensor,
    size: Tuple[int, int, int],
    interpolation: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Crop an eight-corner box from each volume and resample it to a fixed output size.

    ``patches = crop_and_resize3d(tensor, boxes, size)`` takes volumes ``(B, C, D, H, W)`` and boxes ``(B, 8, 3)``, one
    per volume or ``(1, 8, 3)`` for all of them, and returns the ``(B, C, d, h, w)`` patches for ``size = (d, h, w)``.
    Each box must be a parallelepiped, so that an affine map takes it onto the patch; with ``align_corners=True`` its
    corners land on the corner voxels of the patch. Pass ``align_corners=True``, since the default
    ``False`` samples off the box (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). For images, use
    :func:`crop_and_resize`; to place the crop by a second box, :func:`crop_by_boxes3d`.

    Convention:
        - input: ``(B, C, D, H, W)``; ``size`` is ``(d, h, w)``
        - ``boxes``: ``(B, 8, 3)`` corner points in ``(x, y, z)`` order, front face
          then back face, each face top-left, top-right, bottom-right, bottom-left —
          same ``(x, y)``/inclusive-pixel convention as :func:`crop_and_resize`, with
          ``z`` anchored at the top-left of the first depth slice (``z = 0``);
          reproducing the exact integer-voxel slice requires ``align_corners=True`` —
          the default ``False`` interpolates instead
        - align_corners: ``False`` by default
        - a single volume does not broadcast over a batch of boxes; a box or ``size`` with a one-voxel axis raises
          ``torch.linalg.LinAlgError`` (`#4747 <https://github.com/kornia/kornia/issues/4747>`_)

    Args:
        tensor: the 3D volume tensor with shape ``(B, C, D, H, W)``.
        boxes: a tensor with shape ``(B, 8, 3)`` containing the coordinates of the bounding boxes
            to be extracted, where each box is defined in the clockwise
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
        size: a tuple with the depth, height and width that will be
            used to resize the extracted patches.
        interpolation: Interpolation flag.
        align_corners: mode for grid_generation.

    Returns:
        tensor containing the patches with shape ``(B, C, d, h, w)``.

    Raises:
        AssertionError: if a box has a non-finite coordinate, edges along one axis that span different extents on
            it, or is not a parallelepiped.

    See Also:
        - :func:`crop_and_resize`: the same for images and ``(B, 4, 2)`` boxes, where, along an axis longer than one
          pixel, ``align_corners`` changes only ``'reflection'`` padding and, with ``mode='nearest'``, which pixel a
          sample exactly halfway takes.
        - :func:`crop_by_boxes3d`: maps the box onto a second box, whose size sets the output size.
        - :func:`center_crop3d`: the central window of each volume, with ``align_corners=True`` by default.

    Example:
        >>> input = torch.arange(64, dtype=torch.float32).view(1, 1, 4, 4, 4)
        >>> input
        tensor([[[[[ 0.,  1.,  2.,  3.],
                   [ 4.,  5.,  6.,  7.],
                   [ 8.,  9., 10., 11.],
                   [12., 13., 14., 15.]],
        <BLANKLINE>
                  [[16., 17., 18., 19.],
                   [20., 21., 22., 23.],
                   [24., 25., 26., 27.],
                   [28., 29., 30., 31.]],
        <BLANKLINE>
                  [[32., 33., 34., 35.],
                   [36., 37., 38., 39.],
                   [40., 41., 42., 43.],
                   [44., 45., 46., 47.]],
        <BLANKLINE>
                  [[48., 49., 50., 51.],
                   [52., 53., 54., 55.],
                   [56., 57., 58., 59.],
                   [60., 61., 62., 63.]]]]])
        >>> boxes = torch.tensor([[
        ...     [1., 1., 1.],
        ...     [3., 1., 1.],
        ...     [3., 3., 1.],
        ...     [1., 3., 1.],
        ...     [1., 1., 2.],
        ...     [3., 1., 2.],
        ...     [3., 3., 2.],
        ...     [1., 3., 2.],
        ... ]])  # 1x8x3
        >>> crop_and_resize3d(input, boxes, (2, 2, 2), align_corners=True)
        tensor([[[[[21.0000, 23.0000],
                   [29.0000, 31.0000]],
        <BLANKLINE>
                  [[37.0000, 39.0000],
                   [45.0000, 47.0000]]]]])

    """
    if not isinstance(tensor, (torch.Tensor)):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")
    if not isinstance(boxes, (torch.Tensor)):
        raise TypeError(f"Input boxes type is not a torch.Tensor. Got {type(boxes)}")
    if not isinstance(size, (tuple, list)) or len(size) != 3:
        raise ValueError(f"Input size must be a tuple/list of length 3. Got {size}")
    if len(tensor.shape) != 5:
        raise AssertionError(f"Only tensor with shape (B, C, D, H, W) supported. Got {tensor.shape}.")
    # unpack input data
    dst_d, dst_h, dst_w = size[0], size[1], size[2]

    # [x, y, z] origin
    # from front to back
    # top-left, top-right, bottom-right, bottom-left
    points_src: torch.Tensor = boxes

    # [x, y, z] destination
    # from front to back
    # top-left, top-right, bottom-right, bottom-left
    points_dst: torch.Tensor = torch.tensor(
        [
            [
                [0, 0, 0],
                [dst_w - 1, 0, 0],
                [dst_w - 1, dst_h - 1, 0],
                [0, dst_h - 1, 0],
                [0, 0, dst_d - 1],
                [dst_w - 1, 0, dst_d - 1],
                [dst_w - 1, dst_h - 1, dst_d - 1],
                [0, dst_h - 1, dst_d - 1],
            ]
        ],
        dtype=tensor.dtype,
        device=tensor.device,
    ).expand(points_src.shape[0], -1, -1)

    return _crop_by_boxes3d_to_size(tensor, points_src, points_dst, (dst_d, dst_h, dst_w), interpolation, align_corners)


def center_crop3d(
    tensor: torch.Tensor, size: Tuple[int, int, int], interpolation: str = "bilinear", align_corners: bool = True
) -> torch.Tensor:
    r"""Crop the central window of each volume to a given size.

    ``patches = center_crop3d(tensor, size)`` takes volumes ``(B, C, D, H, W)`` and returns their central
    ``(B, C, d, h, w)`` windows for ``size = (d, h, w)``: with the default ``align_corners=True``, a sub-volume of the
    input, with the extra voxel of an odd margin at the back/bottom/right. A ``size`` larger than the input pads with
    zeros, except along a one-voxel input axis, which ``align_corners=True`` repeats
    (`#6433 <https://github.com/kornia/kornia/issues/6433>`_). For images, use :func:`center_crop`; for a crop at any
    box, :func:`crop_and_resize3d`.

    Convention:
        - input: ``(B, C, D, H, W)``; ``size`` is ``(d, h, w)``
        - align_corners: ``True`` by default
        - offset: the crop starts at the integer voxel ``int(D / 2 - d / 2)`` (likewise for the height and width),
          as in :class:`~kornia.augmentation.CenterCrop3D`, so with the default ``align_corners=True`` the output is a
          sub-volume of the input. With an odd margin the extra voxel is left at the back/bottom/right.
          ``align_corners=False`` resamples off the voxel grid even for an even margin (see
          :func:`crop_and_resize3d`, `#4503 <https://github.com/kornia/kornia/issues/4503>`_).
        - a ``size`` with a one-voxel axis raises ``torch.linalg.LinAlgError``
          (`#4747 <https://github.com/kornia/kornia/issues/4747>`_)

    Args:
        tensor: the 3D volume tensor with shape ``(B, C, D, H, W)``.
        size: a tuple with the expected depth, height and width
            of the output patch.
        interpolation: Interpolation flag.
        align_corners : mode for grid_generation.

    Returns:
        the central windows, with shape ``(B, C, d, h, w)``.

    See Also:
        - :func:`center_crop`: the central window of each image.
        - :func:`crop_and_resize3d`: a window at any box, resampled to a fixed size.
        - :class:`~kornia.augmentation.CenterCrop3D`: the same windows as an augmentation, for augmentation pipelines;
          it raises for a ``size`` larger than the input instead of padding.

    Examples:
        >>> input = torch.arange(64, dtype=torch.float32).view(1, 1, 4, 4, 4)
        >>> input
        tensor([[[[[ 0.,  1.,  2.,  3.],
                   [ 4.,  5.,  6.,  7.],
                   [ 8.,  9., 10., 11.],
                   [12., 13., 14., 15.]],
        <BLANKLINE>
                  [[16., 17., 18., 19.],
                   [20., 21., 22., 23.],
                   [24., 25., 26., 27.],
                   [28., 29., 30., 31.]],
        <BLANKLINE>
                  [[32., 33., 34., 35.],
                   [36., 37., 38., 39.],
                   [40., 41., 42., 43.],
                   [44., 45., 46., 47.]],
        <BLANKLINE>
                  [[48., 49., 50., 51.],
                   [52., 53., 54., 55.],
                   [56., 57., 58., 59.],
                   [60., 61., 62., 63.]]]]])
        >>> center_crop3d(input, (2, 2, 2), align_corners=True)
        tensor([[[[[21.0000, 22.0000],
                   [25.0000, 26.0000]],
        <BLANKLINE>
                  [[37.0000, 38.0000],
                   [41.0000, 42.0000]]]]])

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 5:
        raise AssertionError(f"Only tensor with shape (B, C, D, H, W) supported. Got {tensor.shape}.")

    if not isinstance(size, (tuple, list)) or len(size) != 3:
        raise ValueError(f"Input size must be a tuple/list of length 3. Got {size}")

    # unpack input sizes
    dst_d, dst_h, dst_w = size
    src_d, src_h, src_w = tensor.shape[-3:]

    # compute start/end offsets
    dst_d_half = dst_d / 2
    dst_h_half = dst_h / 2
    dst_w_half = dst_w / 2
    src_d_half = src_d / 2
    src_h_half = src_h / 2
    src_w_half = src_w / 2

    # an integer start, as CenterCrop3D takes it: a fractional one would resample between voxels
    start_x = int(src_w_half - dst_w_half)
    start_y = int(src_h_half - dst_h_half)
    start_z = int(src_d_half - dst_d_half)

    end_x = start_x + dst_w - 1
    end_y = start_y + dst_h - 1
    end_z = start_z + dst_d - 1
    # [x, y, z] origin
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    points_src: torch.Tensor = torch.tensor(
        [
            [
                [start_x, start_y, start_z],
                [end_x, start_y, start_z],
                [end_x, end_y, start_z],
                [start_x, end_y, start_z],
                [start_x, start_y, end_z],
                [end_x, start_y, end_z],
                [end_x, end_y, end_z],
                [start_x, end_y, end_z],
            ]
        ],
        device=tensor.device,
    )

    # [x, y, z] destination
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    points_dst: torch.Tensor = torch.tensor(
        [
            [
                [0, 0, 0],
                [dst_w - 1, 0, 0],
                [dst_w - 1, dst_h - 1, 0],
                [0, dst_h - 1, 0],
                [0, 0, dst_d - 1],
                [dst_w - 1, 0, dst_d - 1],
                [dst_w - 1, dst_h - 1, dst_d - 1],
                [0, dst_h - 1, dst_d - 1],
            ]
        ],
        device=tensor.device,
    ).expand(points_src.shape[0], -1, -1)

    return _crop_by_boxes3d_to_size(
        tensor,
        points_src.to(tensor.dtype),
        points_dst.to(tensor.dtype),
        (dst_d, dst_h, dst_w),
        interpolation,
        align_corners,
    )


def crop_by_boxes3d(
    tensor: torch.Tensor,
    src_box: torch.Tensor,
    dst_box: torch.Tensor,
    interpolation: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Crop volumes by mapping a source box onto a destination box, whose size sets the output size.

    ``patches = crop_by_boxes3d(tensor, src_box, dst_box)`` takes volumes ``(B, C, D, H, W)`` and two boxes per
    volume, ``(B, 8, 3)`` each (both ``(1, 8, 3)`` for all volumes), warps each volume by the affine map that takes
    the corners of ``src_box`` onto those of ``dst_box``, which must exist, as it does for any two parallelepipeds or
    two translated copies of one box, and returns the ``(B, C, d, h, w)`` window that starts at voxel ``(0, 0, 0)``,
    where ``(d, h, w)`` is the inclusive size of ``dst_box``; every ``dst_box`` in the batch must have that size. Pass
    ``align_corners=True``: the default ``False`` samples off the box
    (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). When the output size is known, :func:`crop_and_resize3d`
    takes it instead of a second box; for images, use :func:`crop_by_boxes`.

    Convention:
        - input: ``(B, C, D, H, W)``
        - ``src_box``/``dst_box``: ``(B, 8, 3)`` corner points in ``(x, y, z)``
          order, front face then back face, each face top-left, top-right,
          bottom-right, bottom-left — same convention as :func:`crop_and_resize3d`;
          ``dst_box`` determines the output resolution
        - align_corners: ``False`` by default

    Args:
        tensor : the 3D volume tensor with shape ``(B, C, D, H, W)``.
        src_box : a tensor with shape ``(B, 8, 3)`` containing the coordinates of the bounding boxes
            to be extracted, where each box is defined in the clockwise
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
        dst_box: a tensor with shape ``(B, 8, 3)`` containing the coordinates of the bounding boxes
            to be placed, where each box is defined in the clockwise
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
        interpolation: Interpolation flag.
        align_corners: mode for grid_generation.

    Returns:
        the patches, with shape ``(B, C, d, h, w)`` for the size ``(d, h, w)`` of ``dst_box``.

    Raises:
        AssertionError: if the boxes of ``dst_box`` differ in size, a box has a non-finite coordinate or edges along
            one axis that span different extents on it, or no affine map takes a ``src_box`` onto its ``dst_box``.

    See Also:
        - :func:`crop_by_boxes`: the same for images and ``(B, 4, 2)`` boxes.
        - :func:`crop_and_resize3d`: takes the output size as ``size`` and places the box at the patch corners.
        - :func:`crop_by_transform_mat3d`: takes the source→destination matrix itself.

    Examples:
        >>> input = torch.tensor([[[
        ...         [[ 0.,  1.,  2.,  3.],
        ...          [ 4.,  5.,  6.,  7.],
        ...          [ 8.,  9., 10., 11.],
        ...          [12., 13., 14., 15.]],
        ...         [[16., 17., 18., 19.],
        ...          [20., 21., 22., 23.],
        ...          [24., 25., 26., 27.],
        ...          [28., 29., 30., 31.]],
        ...         [[32., 33., 34., 35.],
        ...          [36., 37., 38., 39.],
        ...          [40., 41., 42., 43.],
        ...          [44., 45., 46., 47.]]]]])
        >>> src_box = torch.tensor([[
        ...     [1., 1., 1.],
        ...     [3., 1., 1.],
        ...     [3., 3., 1.],
        ...     [1., 3., 1.],
        ...     [1., 1., 2.],
        ...     [3., 1., 2.],
        ...     [3., 3., 2.],
        ...     [1., 3., 2.],
        ... ]])  # 1x8x3
        >>> dst_box = torch.tensor([[
        ...     [0., 0., 0.],
        ...     [2., 0., 0.],
        ...     [2., 2., 0.],
        ...     [0., 2., 0.],
        ...     [0., 0., 1.],
        ...     [2., 0., 1.],
        ...     [2., 2., 1.],
        ...     [0., 2., 1.],
        ... ]])  # 1x8x3
        >>> crop_by_boxes3d(input, src_box, dst_box, interpolation='nearest', align_corners=True)
        tensor([[[[[21., 22., 23.],
                   [25., 26., 27.],
                   [29., 30., 31.]],
        <BLANKLINE>
                  [[37., 38., 39.],
                   [41., 42., 43.],
                   [45., 46., 47.]]]]])

    """
    bbox = infer_bbox_shape3d(dst_box)
    if not ((bbox[0] == bbox[0][0]).all() and (bbox[1] == bbox[1][0]).all() and (bbox[2] == bbox[2][0]).all()):
        raise AssertionError(
            "Cropping depth, height and width must be exact same in a batch. "
            f"Got depth {bbox[0]}, height {bbox[1]} and width {bbox[2]}."
        )

    out_size = (int(bbox[0][0].item()), int(bbox[1][0].item()), int(bbox[2][0].item()))
    return _crop_by_boxes3d_to_size(tensor, src_box, dst_box, out_size, interpolation, align_corners)


def _crop_by_boxes3d_to_size(
    tensor: torch.Tensor,
    src_box: torch.Tensor,
    dst_box: torch.Tensor,
    out_size: Tuple[int, int, int],
    interpolation: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    # ``crop_by_boxes3d`` with the output size already known as Python ints. ``crop_and_resize3d`` and
    # ``center_crop3d`` build ``dst_box`` from ``size``, so they take this path and never read the box
    # values back from the device -- which also keeps them capturable by ``torch.onnx.export``.
    # ``validate_bbox3d`` raises for every failure it detects except a non-finite coordinate, which is a
    # ``False`` predicate result; these call sites rely on the raise, so they convert that ``False``.
    if not (_validate_bbox3d(src_box) and _validate_bbox3d(dst_box)):
        raise AssertionError("Boxes must have finite coordinates, got non-finite values.")

    if len(tensor.shape) != 5:
        raise AssertionError(f"Only tensor with shape (B, C, D, H, W) supported. Got {tensor.shape}.")

    # compute transformation between points and warp
    # Note: Tensor.dtype must be float. "solve_cpu" not implemented for 'Long'
    src_box, dst_box = src_box.to(tensor.dtype), dst_box.to(tensor.dtype)
    dst_trans_src: torch.Tensor = get_perspective_transform3d(src_box, dst_box)
    _check_boxes_related_by_affine(src_box, dst_box, dst_trans_src)
    # simulate broadcasting
    dst_trans_src = dst_trans_src.expand(tensor.shape[0], -1, -1).type_as(tensor)

    patches: torch.Tensor = crop_by_transform_mat3d(
        tensor, dst_trans_src, out_size, mode=interpolation, align_corners=align_corners
    )

    return patches


def _check_boxes_related_by_affine(src_box: torch.Tensor, dst_box: torch.Tensor, dst_trans_src: torch.Tensor) -> None:
    # The crops warp by the top three rows of the box homography, an affine map, so they place the 8 corners of
    # ``src_box`` on those of ``dst_box`` only when an affine map relates the two boxes, as one relates any two
    # parallelepipeds. Refuse every other pair instead of cropping it with a wrong mapping. Mapping a corner sums
    # terms of up to ``scale`` voxels, rounded to the matrix dtype, so a pair passes when no corner misses by more
    # than ``sqrt(eps)`` of that, with the float32 ``eps`` for float64: a parallelepiped misses by about ``eps`` of it
    # (more on MPS, which solves for the matrix in float32), boxes off a parallelepiped by as much as the extents
    # ``_validate_bbox3d`` lets differ (``rtol=1e-5``), such as a gradcheck step, still pass, and moving a single
    # corner of a small box by one voxel misses by far more. Like the value checks of ``_validate_bbox3d``, this one
    # reads the data back and is skipped under graph capture.
    if not torch.jit.is_scripting() and is_compiling():
        return
    work_dtype = torch.promote_types(dst_trans_src.dtype, torch.float32)
    mat = dst_trans_src[:, :3].detach().to(work_dtype)
    src, dst = src_box.detach().to(work_dtype), dst_box.detach().to(work_dtype)
    linear, translation = mat[..., :3].transpose(-1, -2), mat[..., 3:].transpose(-1, -2)
    miss = (src @ linear + translation - dst).abs().amax((-2, -1))
    scale = (src.abs() @ linear.abs() + translation.abs() + dst.abs()).amax((-2, -1))
    tolerance = max(torch.finfo(dst_trans_src.dtype).eps, torch.finfo(torch.float32).eps) ** 0.5 * scale
    if bool((miss > tolerance).any()):
        raise AssertionError(
            "Boxes must be related by an affine map, as two parallelepipeds are: the affine map the crop warps by "
            f"misses the corners of dst_box by up to {miss.max().item():.3g} voxels."
        )


def crop_by_transform_mat3d(
    tensor: torch.Tensor,
    transform: torch.Tensor,
    out_size: Tuple[int, int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Crop volumes by a source→destination voxel matrix into a given output size.

    ``patches = crop_by_transform_mat3d(tensor, transform, out_size)`` takes volumes ``(B, C, D, H, W)`` and a
    source→destination voxel matrix per volume, ``(B, 3, 4)`` or ``(B, 4, 4)`` (a batch of 1 serves every volume, which
    :func:`warp_affine3d` alone does not allow), and returns the ``(B, C, d, h, w)`` window that starts at voxel
    ``(0, 0, 0)`` for ``out_size = (d, h, w)``. It warps with :func:`warp_affine3d` and uses only the top three rows,
    so a projective last row is ignored. Unlike :func:`crop_by_transform_mat`, it does not cast the matrix: its dtype
    must be the volume's. For images, use :func:`crop_by_transform_mat`; to build the matrix from boxes,
    :func:`crop_by_boxes3d`.

    Convention:
        - input: ``(B, C, D, H, W)``; ``out_size`` is ``(d, h, w)``
        - ``transform`` is the source→destination **pixel** transform, accepted as
          either ``(B, 3, 4)`` affine or ``(B, 4, 4)`` homogeneous; only the top
          three rows are used, so a projective last row is ignored (unlike the 2D
          :func:`crop_by_transform_mat`)
        - align_corners: ``True`` by default; ``False`` warps by a matrix normalized for corner-aligned voxels
          (`#4503 <https://github.com/kornia/kornia/issues/4503>`_)

    Args:
        tensor: the 3D volume tensor with shape ``(B, C, D, H, W)``.
        transform: the source->destination pixel transform, either affine with
          shape ``(B, 3, 4)`` or homogeneous with shape ``(B, 4, 4)``.
        out_size: size of the output image (depth, height, width).
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.

    Returns:
        the patches, with shape ``(B, C, d, h, w)``.

    See Also:
        - :func:`crop_by_transform_mat`: the same for images, which also applies a projective row.
        - :func:`warp_affine3d`: the warp it calls, which takes one ``(B, 3, 4)`` matrix per volume.
        - :func:`crop_by_boxes3d`: builds the matrix from a source and a destination box.

    """
    # simulate broadcasting
    dst_trans_src = transform.expand(tensor.shape[0], -1, -1)

    patches: torch.Tensor = warp_affine3d(
        tensor, dst_trans_src[:, :3, :], out_size, flags=mode, padding_mode=padding_mode, align_corners=align_corners
    )

    return patches

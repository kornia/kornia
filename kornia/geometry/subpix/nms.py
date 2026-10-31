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

"""Strict non-maxima suppression of 2D response maps and 3D response volumes."""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _split_window(k: int) -> Tuple[int, int]:
    """Return the neighbour extents of a length-``k`` window either side of its centre."""
    before = (k - 1) // 2
    return before, k - before - 1


def _reduce_max(parts: List[torch.Tensor]) -> torch.Tensor:
    """Reduce the maxima from the non-empty neighbourhood regions."""
    out = parts[0]
    for i in range(1, len(parts)):
        out = torch.maximum(out, parts[i])
    return out


def _neighbourhood_max2d(x: torch.Tensor, ky: int, kx: int) -> torch.Tensor:
    """Max over each ``ky x kx`` window with its own centre excluded.

    The window minus its centre is partitioned into four rectangles -- the rows above the centre
    row, the rows below it, and the centre row either side of the centre -- and each is reduced with
    ``max_pool2d``.  The two full-width slabs share a single ``(1, kx)`` column pass, so the cost is
    ``O(ky + kx)`` taps per position rather than the ``ky * kx - 1`` a literal neighbourhood costs.

    Only positions whose full window lies inside the image are computed, so the result has shape
    ``(B, C, H - ky + 1, W - kx + 1)``; entry ``(i, j)`` belongs to the centre
    ``x[..., i + (k_y - 1) // 2, j + (k_x - 1) // 2]``.
    """
    cy, by = _split_window(ky)
    cx, bx = _split_window(kx)
    H, W = x.shape[-2], x.shape[-1]
    parts: List[torch.Tensor] = []
    if cy > 0 or by > 0:
        row_max = F.max_pool2d(x, (1, kx), stride=1)
        if cy > 0:
            parts.append(F.max_pool2d(row_max[..., : H - by - 1, :], (cy, 1), stride=1))
        if by > 0:
            parts.append(F.max_pool2d(row_max[..., cy + 1 :, :], (by, 1), stride=1))
    centre_row = x[..., cy : H - by, :]
    if cx > 0:
        parts.append(F.max_pool2d(centre_row[..., : W - bx - 1], (1, cx), stride=1))
    if bx > 0:
        parts.append(F.max_pool2d(centre_row[..., cx + 1 :], (1, bx), stride=1))
    return _reduce_max(parts)


def _neighbourhood_max3d(x: torch.Tensor, kd: int, ky: int, kx: int) -> torch.Tensor:
    """Max over each ``kd x ky x kx`` window with its own centre excluded.

    The 3-D analogue of :func:`_neighbourhood_max2d`: six boxes -- the depth slabs either side of
    the centre depth, then the row slabs either side of the centre row within it, then the centre
    row either side of the centre -- sharing the column and row passes they have in common.
    """
    cd, bd = _split_window(kd)
    cy, by = _split_window(ky)
    cx, bx = _split_window(kx)
    D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
    parts: List[torch.Tensor] = []
    col_max = F.max_pool3d(x, (1, 1, kx), stride=1) if (cd > 0 or bd > 0 or cy > 0 or by > 0) else x
    if cd > 0 or bd > 0:
        plane_max = F.max_pool3d(col_max, (1, ky, 1), stride=1)
        if cd > 0:
            parts.append(F.max_pool3d(plane_max[..., : D - bd - 1, :, :], (cd, 1, 1), stride=1))
        if bd > 0:
            parts.append(F.max_pool3d(plane_max[..., cd + 1 :, :, :], (bd, 1, 1), stride=1))
    centre_plane = col_max[..., cd : D - bd, :, :]
    if cy > 0:
        parts.append(F.max_pool3d(centre_plane[..., : H - by - 1, :], (1, cy, 1), stride=1))
    if by > 0:
        parts.append(F.max_pool3d(centre_plane[..., cy + 1 :, :], (1, by, 1), stride=1))
    centre_row = x[..., cd : D - bd, cy : H - by, :]
    if cx > 0:
        parts.append(F.max_pool3d(centre_row[..., : W - bx - 1], (1, 1, cx), stride=1))
    if bx > 0:
        parts.append(F.max_pool3d(centre_row[..., cx + 1 :], (1, 1, bx), stride=1))
    return _reduce_max(parts)


class NonMaximaSuppression2d(nn.Module):
    r"""Keep the strict local maxima of a (B, C, H, W) map; the module form of nms2d.

    ``nms = NonMaximaSuppression2d((ky, kx))`` stores the window size, and ``out = nms(response)`` returns what
    ``nms2d(response, (ky, kx))`` returns: the ``(B, C, H, W)`` map with every position that is not a strict local
    maximum set to ``0``, or with ``nms(response, mask_only=True)`` the boolean mask. The module holds no parameters
    or buffers; a state dict saved by an older kornia release with a ``kernel`` buffer still loads strictly.

    Convention:
        - See the convention block of :func:`~kornia.geometry.subpix.nms2d`.

    Args:
        kernel_size: the window size ``(ky, kx)``, a tuple of two ints; a list or another length raises
          ``AssertionError``.

    See Also:
        - :func:`nms2d`: the function form, with the window size passed on each call.
        - :class:`NonMaximaSuppression3d`: the module for ``(B, C, D, H, W)`` volumes.
        - :func:`nms3d_minmax`: strict maxima and minima of a volume in one call.

    """

    def __init__(self, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        if not isinstance(kernel_size, tuple):
            raise AssertionError(type(kernel_size))
        if len(kernel_size) != 2:
            raise AssertionError(kernel_size)
        self.kernel_size: tuple[int, int] = kernel_size

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # The convolution implementation registered this derived tensor as a persistent buffer.
        # It is unused by the pooled implementation, but accepting it preserves strict loading of
        # checkpoints saved by older Kornia releases, including when NMS is nested in another module.
        state_dict.pop(prefix + "kernel", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor, mask_only: bool = False) -> torch.Tensor:
        """Keep only strict local maxima in a 2D response map.

        Each spatial location is compared with the surrounding values inside
        ``self.kernel_size``. Locations that are not strictly larger than their
        neighbors are suppressed. This is commonly used to turn dense corner or
        keypoint response maps into sparse candidate locations.

        Args:
            x: Response tensor with shape ``(B, C, H, W)``, where
                :math:`B` is the batch size, :math:`C` is the number of
                response channels, :math:`H` is height, and :math:`W` is width.
            mask_only: If ``True``, return the boolean maxima mask. If
                ``False``, return ``x`` masked by local-maxima positions.

        Returns:
            If ``mask_only`` is ``True``, a boolean tensor with shape
            ``(B, C, H, W)``. Otherwise, a tensor with the same shape and
            dtype as ``x`` where non-maxima have been set to zero.
        """
        if len(x.shape) != 4:
            raise AssertionError(x.shape)
        B, CH, H, W = x.size()

        if self.kernel_size == (1, 1):
            mask = torch.ones(B, CH, H, W, device=x.device, dtype=torch.bool)
        elif self.kernel_size == (3, 3):
            # 8-comparison explicit path: no extra memory for conv kernel.
            left = slice(0, -2)
            center = slice(1, -1)
            right = slice(2, None)
            mask = torch.zeros(B, CH, H, W, device=x.device, dtype=torch.bool)
            ct = x[..., center, center]
            mask[..., 1:-1, 1:-1] = (
                (ct > x[..., left, left])
                & (ct > x[..., left, center])
                & (ct > x[..., left, right])
                & (ct > x[..., center, left])
                & (ct > x[..., center, right])
                & (ct > x[..., right, left])
                & (ct > x[..., right, center])
                & (ct > x[..., right, right])
            )
        elif self.kernel_size == (5, 5):
            # 24-comparison explicit path for 5x5 neighbourhood.
            c2 = slice(0, -4)
            c1 = slice(1, -3)
            c0 = slice(2, -2)
            p1 = slice(3, -1)
            p2 = slice(4, None)
            mask = torch.zeros(B, CH, H, W, device=x.device, dtype=torch.bool)
            ct = x[..., c0, c0]
            mask[..., 2:-2, 2:-2] = (
                (ct > x[..., c2, c2])
                & (ct > x[..., c2, c1])
                & (ct > x[..., c2, c0])
                & (ct > x[..., c2, p1])
                & (ct > x[..., c2, p2])
                & (ct > x[..., c1, c2])
                & (ct > x[..., c1, c1])
                & (ct > x[..., c1, c0])
                & (ct > x[..., c1, p1])
                & (ct > x[..., c1, p2])
                & (ct > x[..., c0, c2])
                & (ct > x[..., c0, c1])
                & (ct > x[..., c0, p1])
                & (ct > x[..., c0, p2])
                & (ct > x[..., p1, c2])
                & (ct > x[..., p1, c1])
                & (ct > x[..., p1, c0])
                & (ct > x[..., p1, p1])
                & (ct > x[..., p1, p2])
                & (ct > x[..., p2, c2])
                & (ct > x[..., p2, c1])
                & (ct > x[..., p2, c0])
                & (ct > x[..., p2, p1])
                & (ct > x[..., p2, p2])
            )
        elif self.kernel_size == (7, 7):
            # 48-comparison explicit path for 7x7 neighbourhood.
            c3 = slice(0, -6)
            c2 = slice(1, -5)
            c1 = slice(2, -4)
            c0 = slice(3, -3)
            p1 = slice(4, -2)
            p2 = slice(5, -1)
            p3 = slice(6, None)
            mask = torch.zeros(B, CH, H, W, device=x.device, dtype=torch.bool)
            ct = x[..., c0, c0]
            mask[..., 3:-3, 3:-3] = (
                (ct > x[..., c3, c3])
                & (ct > x[..., c3, c2])
                & (ct > x[..., c3, c1])
                & (ct > x[..., c3, c0])
                & (ct > x[..., c3, p1])
                & (ct > x[..., c3, p2])
                & (ct > x[..., c3, p3])
                & (ct > x[..., c2, c3])
                & (ct > x[..., c2, c2])
                & (ct > x[..., c2, c1])
                & (ct > x[..., c2, c0])
                & (ct > x[..., c2, p1])
                & (ct > x[..., c2, p2])
                & (ct > x[..., c2, p3])
                & (ct > x[..., c1, c3])
                & (ct > x[..., c1, c2])
                & (ct > x[..., c1, c1])
                & (ct > x[..., c1, c0])
                & (ct > x[..., c1, p1])
                & (ct > x[..., c1, p2])
                & (ct > x[..., c1, p3])
                & (ct > x[..., c0, c3])
                & (ct > x[..., c0, c2])
                & (ct > x[..., c0, c1])
                & (ct > x[..., c0, p1])
                & (ct > x[..., c0, p2])
                & (ct > x[..., c0, p3])
                & (ct > x[..., p1, c3])
                & (ct > x[..., p1, c2])
                & (ct > x[..., p1, c1])
                & (ct > x[..., p1, c0])
                & (ct > x[..., p1, p1])
                & (ct > x[..., p1, p2])
                & (ct > x[..., p1, p3])
                & (ct > x[..., p2, c3])
                & (ct > x[..., p2, c2])
                & (ct > x[..., p2, c1])
                & (ct > x[..., p2, c0])
                & (ct > x[..., p2, p1])
                & (ct > x[..., p2, p2])
                & (ct > x[..., p2, p3])
                & (ct > x[..., p3, c3])
                & (ct > x[..., p3, c2])
                & (ct > x[..., p3, c1])
                & (ct > x[..., p3, c0])
                & (ct > x[..., p3, p1])
                & (ct > x[..., p3, p2])
                & (ct > x[..., p3, p3])
            )
        else:
            # General path: the same rule as the explicit paths above, for any window size. A
            # position whose full window does not fit inside the image is not a maximum -- it has
            # not been compared against the neighbours that would decide it -- so the border strip
            # stays False and only the interior is computed (#4239).
            ky, kx = self.kernel_size
            cy, by = _split_window(ky)
            cx, bx = _split_window(kx)
            mask = torch.zeros(B, CH, H, W, device=x.device, dtype=torch.bool)
            if H >= ky and W >= kx:
                centre = x[..., cy : H - by, cx : W - bx]
                mask[..., cy : H - by, cx : W - bx] = centre > _neighbourhood_max2d(x, ky, kx)

        if mask_only:
            return mask
        return torch.where(mask, x, 0)


class NonMaximaSuppression3d(nn.Module):
    r"""Keep the strict local maxima of a (B, C, D, H, W) volume; the module form of nms3d.

    ``nms = NonMaximaSuppression3d((kd, ky, kx))`` stores the window size, and ``out = nms(volume)`` returns what
    ``nms3d(volume, (kd, ky, kx))`` returns: the ``(B, C, D, H, W)`` volume with every voxel that is not a strict
    local maximum set to ``0``, or with ``nms(volume, mask_only=True)`` the boolean mask.

    Convention:
        - See the convention block of :func:`~kornia.geometry.subpix.nms2d`.

    Args:
        kernel_size: the window size ``(kd, ky, kx)``, a tuple of three ints; a list or another length raises
          ``AssertionError``.

    See Also:
        - :func:`nms3d`: the function form, with the window size passed on each call.
        - :class:`NonMaximaSuppression2d`: the module for ``(B, C, H, W)`` maps.
        - :func:`nms3d_minmax`: strict maxima and minima in a ``3 x 3 x 3`` window in one call.

    """

    def __init__(self, kernel_size: tuple[int, int, int]) -> None:
        super().__init__()
        if not isinstance(kernel_size, tuple):
            raise AssertionError(type(kernel_size))
        if len(kernel_size) != 3:
            raise AssertionError(kernel_size)
        self.kernel_size: tuple[int, int, int] = kernel_size

    def forward(self, x: torch.Tensor, mask_only: bool = False) -> torch.Tensor:
        """Keep only strict local maxima in a 3D response volume.

        Each voxel is compared with its neighbors across depth, height, and
        width. This is used by scale-space detectors to keep responses that are
        locally maximal both in image position and in scale/depth.

        Args:
            x: Response tensor with shape ``(B, C, D, H, W)``, where
                :math:`B` is batch size, :math:`C` is channel count,
                :math:`D` is depth or scale level, :math:`H` is height, and
                :math:`W` is width.
            mask_only: If ``True``, return only the maxima mask; otherwise
                return suppressed responses.

        Returns:
            Boolean maxima mask with shape ``(B, C, D, H, W)`` when
            ``mask_only`` is ``True``. Otherwise, an NMS-filtered tensor with
            the same shape and dtype as ``x``.
        """
        if len(x.shape) != 5:
            raise AssertionError(x.shape)
        # find local maximum values
        B, CH, D, H, W = x.size()
        if self.kernel_size == (1, 1, 1):
            mask = torch.ones(B, CH, D, H, W, device=x.device, dtype=torch.bool)
        elif self.kernel_size == (3, 3, 3):
            # 26-comparison explicit path: strict local maximum, works on CPU and CUDA.
            # Using integer slice literals (not slice objects) makes this torch.jit.script-friendly,
            # which fuses the ops and runs ~13x faster on CUDA than the eager path.
            mask = torch.zeros(B, CH, D, H, W, device=x.device, dtype=torch.bool)
            ct = x[..., 1:-1, 1:-1, 1:-1]
            mask[..., 1:-1, 1:-1, 1:-1] = (
                (ct > x[..., 0:-2, 0:-2, 0:-2])
                & (ct > x[..., 0:-2, 0:-2, 1:-1])
                & (ct > x[..., 0:-2, 0:-2, 2:])
                & (ct > x[..., 0:-2, 1:-1, 0:-2])
                & (ct > x[..., 0:-2, 1:-1, 1:-1])
                & (ct > x[..., 0:-2, 1:-1, 2:])
                & (ct > x[..., 0:-2, 2:, 0:-2])
                & (ct > x[..., 0:-2, 2:, 1:-1])
                & (ct > x[..., 0:-2, 2:, 2:])
                & (ct > x[..., 1:-1, 0:-2, 0:-2])
                & (ct > x[..., 1:-1, 0:-2, 1:-1])
                & (ct > x[..., 1:-1, 0:-2, 2:])
                & (ct > x[..., 1:-1, 1:-1, 0:-2])
                & (ct > x[..., 1:-1, 1:-1, 2:])
                & (ct > x[..., 1:-1, 2:, 0:-2])
                & (ct > x[..., 1:-1, 2:, 1:-1])
                & (ct > x[..., 1:-1, 2:, 2:])
                & (ct > x[..., 2:, 0:-2, 0:-2])
                & (ct > x[..., 2:, 0:-2, 1:-1])
                & (ct > x[..., 2:, 0:-2, 2:])
                & (ct > x[..., 2:, 1:-1, 0:-2])
                & (ct > x[..., 2:, 1:-1, 1:-1])
                & (ct > x[..., 2:, 1:-1, 2:])
                & (ct > x[..., 2:, 2:, 0:-2])
                & (ct > x[..., 2:, 2:, 1:-1])
                & (ct > x[..., 2:, 2:, 2:])
            )
        else:
            # General path: the same rule as the explicit path above, for any window size. See
            # `NonMaximaSuppression2d.forward` for the border convention (#4239). The old path
            # padded by the full kernel size rather than half of it and raised for every kernel
            # other than (3, 3, 3) (#4241); nothing pads here.
            kd, ky, kx = self.kernel_size
            cd, bd = _split_window(kd)
            cy, by = _split_window(ky)
            cx, bx = _split_window(kx)
            mask = torch.zeros(B, CH, D, H, W, device=x.device, dtype=torch.bool)
            if D >= kd and H >= ky and W >= kx:
                centre = x[..., cd : D - bd, cy : H - by, cx : W - bx]
                mask[..., cd : D - bd, cy : H - by, cx : W - bx] = centre > _neighbourhood_max3d(x, kd, ky, kx)
        if mask_only:
            return mask
        return torch.where(mask, x, 0)


# functional api


def nms2d(input: torch.Tensor, kernel_size: tuple[int, int], mask_only: bool = False) -> torch.Tensor:
    r"""Keep the strict local maxima of a (B, C, H, W) map in a 2D window, zeroing the rest.

    ``out = nms2d(response, (ky, kx))`` takes a ``(B, C, H, W)`` response map and returns it with every position that
    is not a strict local maximum of its ``ky x kx`` window set to ``0``, ``(B, C, H, W)`` in the input dtype; with
    ``mask_only=True`` it returns the boolean maxima mask instead. Each channel is suppressed on its own. For
    ``(B, C, D, H, W)`` volumes, use :func:`nms3d`, or :func:`nms3d_minmax` for the maxima and minima together; to
    suppress overlapping boxes by their IoU, use :func:`~kornia.geometry.bbox.nms`.

    Convention:
        - Only strict maxima survive: a position must be greater than every other value in its ``kernel_size``
          window, so with ``k >= 3`` on every axis every pixel of a plateau is suppressed, even a plateau at the top
          of the map. :ref:`Coordinates and sizes <coordinate-conventions>` compares this with scikit-image, scipy
          and OpenCV.
        - The window spans ``(k - 1) // 2`` positions before the centre and ``k // 2`` after it on each axis, so an
          even ``k`` reaches one further forward, and the first ``(k - 1) // 2`` and last ``k // 2`` positions of
          each axis are never maxima.
        - The default output keeps the input value at each maximum and is ``0`` elsewhere, which is larger than a
          negative maximum, so use ``mask_only=True`` for a signed response.

    Args:
        input: response map with shape ``(B, C, H, W)``; another number of dimensions raises ``AssertionError``.
        kernel_size: the window size ``(ky, kx)``, a tuple of two ints. ``(1, 1)`` keeps every position, and a
          window larger than the map keeps none.
        mask_only: return the boolean maxima mask instead of the suppressed map.

    Returns:
        the suppressed map ``(B, C, H, W)`` in the dtype of ``input``, or with ``mask_only=True`` a boolean mask of
        the same shape.

    See Also:
        - :func:`nms3d`: the same rule over the depth, height and width of a ``(B, C, D, H, W)`` volume.
        - :func:`nms3d_minmax`: strict maxima and minima of a volume in a fixed ``3 x 3 x 3`` window, in one call.
        - :class:`NonMaximaSuppression2d`: the module form.
        - :func:`~kornia.geometry.bbox.nms`: suppresses overlapping boxes by their IoU, not positions of a map.

    """
    return NonMaximaSuppression2d(kernel_size)(input, mask_only)


def nms3d(input: torch.Tensor, kernel_size: tuple[int, int, int], mask_only: bool = False) -> torch.Tensor:
    r"""Keep the strict local maxima of a (B, C, D, H, W) volume in a 3D window, zeroing the rest.

    ``out = nms3d(volume, (kd, ky, kx))`` takes a ``(B, C, D, H, W)`` volume, such as a scale-space response with the
    scale levels along ``D``, and returns it with every voxel that is not a strict local maximum of its
    ``kd x ky x kx`` window set to ``0``; with ``mask_only=True`` it returns the boolean maxima mask instead. For the
    minima as well, in a ``3 x 3 x 3`` window, use :func:`nms3d_minmax`; for a ``(B, C, H, W)`` map, :func:`nms2d`.

    Convention:
        - See the convention block of :func:`nms2d`, applied over depth, height and width.

    Args:
        input: response volume with shape ``(B, C, D, H, W)``; another number of dimensions raises
          ``AssertionError``.
        kernel_size: the window size ``(kd, ky, kx)``, a tuple of three ints. ``(1, 1, 1)`` keeps every voxel, and a
          window larger than the volume keeps none.
        mask_only: return the boolean maxima mask instead of the suppressed volume.

    Returns:
        the suppressed volume ``(B, C, D, H, W)`` in the dtype of ``input``, or with ``mask_only=True`` a boolean
        mask of the same shape.

    See Also:
        - :func:`nms2d`: the same rule over the height and width of a ``(B, C, H, W)`` map.
        - :func:`nms3d_minmax`: the maxima and the minima in a fixed ``3 x 3 x 3`` window, in one call.
        - :class:`NonMaximaSuppression3d`: the module form.
        - :func:`iterative_quad_interp3d`: refines the maxima of a scale-space volume to sub-voxel positions; it
          takes this function's mask as ``precomputed_nms_mask``.

    """
    return NonMaximaSuppression3d(kernel_size)(input, mask_only)


def nms3d_minmax(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Mask the strict local maxima and minima of a (B, C, D, H, W) volume in a 3x3x3 window.

    ``max_mask, min_mask = nms3d_minmax(volume)`` takes a ``(B, C, D, H, W)`` volume and returns two boolean masks of
    the same shape: the voxels strictly greater than all 26 neighbours, and those strictly smaller. For a floating
    input they equal ``nms3d(volume, (3, 3, 3), mask_only=True)`` and
    ``nms3d(-volume, (3, 3, 3), mask_only=True)``. :class:`~kornia.feature.ScaleSpaceDetector` uses it on every
    octave; for another window size, call :func:`nms3d`.

    Under ``torch.compile`` the comparisons can fuse; in eager mode on the CPU and MPS the call costs about as much as
    the two :func:`nms3d` calls.

    Convention:
        - The strict and border rules of :func:`nms2d` with a :math:`3 \times 3 \times 3` window, for maxima and
          minima alike.

    Args:
        input: 5-D tensor of shape ``(B, C, D, H, W)``; another number of dimensions raises ``AssertionError``.

    Returns:
        A pair ``(max_mask, min_mask)`` of bool tensors with the same shape as
        *input*.  ``max_mask[..., d, h, w]`` is ``True`` when the voxel is
        strictly greater than all 26 neighbours; ``min_mask`` is the same for
        strict local minima.

    See Also:
        - :func:`nms3d`: the maxima only, in a window of any size.
        - :func:`nms2d`: the maxima of a ``(B, C, H, W)`` map.
        - :class:`NonMaximaSuppression3d`: the maxima only, as a module.

    Example:
        >>> x = torch.randn(1, 1, 5, 10, 10)
        >>> max_mask, min_mask = nms3d_minmax(x)
        >>> max_mask.shape
        torch.Size([1, 1, 5, 10, 10])

    """
    if input.dim() != 5:
        raise AssertionError(input.shape)
    B, CH, D, H, W = input.shape
    max_mask = torch.zeros(B, CH, D, H, W, device=input.device, dtype=torch.bool)
    min_mask = torch.zeros(B, CH, D, H, W, device=input.device, dtype=torch.bool)
    ct = input[..., 1:-1, 1:-1, 1:-1]
    # 26 explicit comparisons with integer slice literals — no Python loop so the
    # compiler sees all ops at trace time and can fuse them into a single kernel.
    is_max = (
        (ct > input[..., 0:-2, 0:-2, 0:-2])
        & (ct > input[..., 0:-2, 0:-2, 1:-1])
        & (ct > input[..., 0:-2, 0:-2, 2:])
        & (ct > input[..., 0:-2, 1:-1, 0:-2])
        & (ct > input[..., 0:-2, 1:-1, 1:-1])
        & (ct > input[..., 0:-2, 1:-1, 2:])
        & (ct > input[..., 0:-2, 2:, 0:-2])
        & (ct > input[..., 0:-2, 2:, 1:-1])
        & (ct > input[..., 0:-2, 2:, 2:])
        & (ct > input[..., 1:-1, 0:-2, 0:-2])
        & (ct > input[..., 1:-1, 0:-2, 1:-1])
        & (ct > input[..., 1:-1, 0:-2, 2:])
        & (ct > input[..., 1:-1, 1:-1, 0:-2])
        & (ct > input[..., 1:-1, 1:-1, 2:])
        & (ct > input[..., 1:-1, 2:, 0:-2])
        & (ct > input[..., 1:-1, 2:, 1:-1])
        & (ct > input[..., 1:-1, 2:, 2:])
        & (ct > input[..., 2:, 0:-2, 0:-2])
        & (ct > input[..., 2:, 0:-2, 1:-1])
        & (ct > input[..., 2:, 0:-2, 2:])
        & (ct > input[..., 2:, 1:-1, 0:-2])
        & (ct > input[..., 2:, 1:-1, 1:-1])
        & (ct > input[..., 2:, 1:-1, 2:])
        & (ct > input[..., 2:, 2:, 0:-2])
        & (ct > input[..., 2:, 2:, 1:-1])
        & (ct > input[..., 2:, 2:, 2:])
    )
    is_min = (
        (ct < input[..., 0:-2, 0:-2, 0:-2])
        & (ct < input[..., 0:-2, 0:-2, 1:-1])
        & (ct < input[..., 0:-2, 0:-2, 2:])
        & (ct < input[..., 0:-2, 1:-1, 0:-2])
        & (ct < input[..., 0:-2, 1:-1, 1:-1])
        & (ct < input[..., 0:-2, 1:-1, 2:])
        & (ct < input[..., 0:-2, 2:, 0:-2])
        & (ct < input[..., 0:-2, 2:, 1:-1])
        & (ct < input[..., 0:-2, 2:, 2:])
        & (ct < input[..., 1:-1, 0:-2, 0:-2])
        & (ct < input[..., 1:-1, 0:-2, 1:-1])
        & (ct < input[..., 1:-1, 0:-2, 2:])
        & (ct < input[..., 1:-1, 1:-1, 0:-2])
        & (ct < input[..., 1:-1, 1:-1, 2:])
        & (ct < input[..., 1:-1, 2:, 0:-2])
        & (ct < input[..., 1:-1, 2:, 1:-1])
        & (ct < input[..., 1:-1, 2:, 2:])
        & (ct < input[..., 2:, 0:-2, 0:-2])
        & (ct < input[..., 2:, 0:-2, 1:-1])
        & (ct < input[..., 2:, 0:-2, 2:])
        & (ct < input[..., 2:, 1:-1, 0:-2])
        & (ct < input[..., 2:, 1:-1, 1:-1])
        & (ct < input[..., 2:, 1:-1, 2:])
        & (ct < input[..., 2:, 2:, 0:-2])
        & (ct < input[..., 2:, 2:, 1:-1])
        & (ct < input[..., 2:, 2:, 2:])
    )
    max_mask[..., 1:-1, 1:-1, 1:-1] = is_max
    min_mask[..., 1:-1, 1:-1, 1:-1] = is_min
    return max_mask, min_mask

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


from typing import Optional

import torch

from kornia.core.utils import is_compiling


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`. A singleton axis is
    represented by ``0``, the centre of the normalized range. A zero spatial size
    produces a correspondingly empty grid; this differs from pixel-coordinate
    normalization, where a zero-sized coordinate system is undefined.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])

    """
    # Build the pixel ramp with ``arange`` in both eager and capture. ``linspace`` rounds its
    # endpoint into the coordinate dtype first and fills the upper half of the ramp backwards from
    # that rounded value, so its step is exactly 1 only while ``size - 1`` is an exactly
    # representable integer -- ``size <= 2 ** p + 1`` for a ``p``-bit significand. Past that it
    # lands one ulp off the correctly rounded column index. The bound is 257 in bfloat16 and 2049
    # in float16, and 2 ** 24 + 1 / 2 ** 53 + 1 in float32/float64, so no non-half grid can move.
    # Inductor lowers the same call to a per-index computation that rounds correctly, so eager and
    # compiled grids disagreed by up to one ulp of the width. ``arange`` also retains the dynamic
    # output length under export, where ``linspace`` specializes symbolic ``steps``. Match
    # ``linspace``'s default floating dtype when none is given -- read off an empty tensor rather
    # than via ``torch.get_default_dtype()``, for which TorchScript has no builtin. The empty
    # tensor takes ``device`` so that the read depends on an argument: a no-input factory call is
    # constant-folded by TorchScript's optimizing executor after the first run, which would freeze
    # the ramp at whatever the default dtype happened to be on that call.
    ramp_dtype = dtype if dtype is not None else torch.empty(0, device=device).dtype
    # Normalizing at float16/bfloat16 rounds three times in eager -- once per op in
    # ``(xs / (size - 1) - 0.5) * 2`` -- where inductor computes the chain in float32 and rounds
    # once on store. Eager also materializes the half ramp before the capture branch widens it,
    # while inductor folds that narrow-then-widen round trip away and keeps the exact index, so
    # above ``2 ** p`` the two paths were dividing different numerators. Running the whole
    # normalization in float32 and narrowing once removes both: the arithmetic is then the same
    # on every path and only the final rounding is left. float32/float64 and the pixel ramp are
    # bit-for-bit untouched, since neither widens.
    widened = normalized_coordinates and ramp_dtype in (torch.float16, torch.bfloat16)
    work_dtype = torch.float32 if widened else ramp_dtype
    xs = torch.arange(width, device=device, dtype=work_dtype)
    ys = torch.arange(height, device=device, dtype=work_dtype)
    # Fix TracerWarning
    # Note: keeping this formula inline avoids the extra tensors and shape checks incurred by
    #       normalize_pixel_coordinates. The two paths use the same singleton-centre policy.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        if not torch.jit.is_scripting() and (torch.jit.is_tracing() or is_compiling()):
            # Graph capture needs the tensor form to keep a symbolic size dynamic. The ramp is
            # already in ``work_dtype``, so the divisor is exact and the quotient no longer has
            # to be rounded back down mid-expression.
            width_t = torch.scalar_tensor(width, device=xs.device, dtype=work_dtype)
            height_t = torch.scalar_tensor(height, device=ys.device, dtype=work_dtype)
            xs = torch.where(width_t > 1, (xs / (width_t - 1) - 0.5) * 2, xs * 0.0)
            ys = torch.where(height_t > 1, (ys / (height_t - 1) - 0.5) * 2, ys * 0.0)
        else:
            # ``* 0.0`` rather than ``zeros_like`` so that a singleton axis follows the
            # same integer-to-float promotion the non-singleton branch performs.
            xs = (xs / (width - 1) - 0.5) * 2 if width > 1 else xs * 0.0
            ys = (ys / (height - 1) - 0.5) * 2 if height > 1 else ys * 0.0
    if widened:
        # The single rounding, after the whole normalization.
        xs = xs.to(ramp_dtype)
        ys = ys.to(ramp_dtype)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def create_meshgrid3d(
    depth: int,
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`. A singleton axis is
    represented by ``0``, the centre of the normalized range. A zero spatial size
    produces a correspondingly empty grid; this differs from pixel-coordinate
    normalization, where a zero-sized coordinate system is undefined.

    Args:
        depth: the image depth (channels).
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, D, H, W, 3)`.

    """
    # See ``create_meshgrid``: ``arange`` keeps eager and captured pixel ramps identical at
    # float16/bfloat16, keeps the output length dynamic under export, and reads the default dtype
    # off a ``device``-bearing empty tensor so TorchScript cannot constant-fold the read.
    ramp_dtype = dtype if dtype is not None else torch.empty(0, device=device).dtype
    # See ``create_meshgrid``: a normalized half-precision ramp is built and normalized in
    # float32 and narrowed once, so eager and captured grids agree.
    widened = normalized_coordinates and ramp_dtype in (torch.float16, torch.bfloat16)
    work_dtype = torch.float32 if widened else ramp_dtype
    xs = torch.arange(width, device=device, dtype=work_dtype)
    ys = torch.arange(height, device=device, dtype=work_dtype)
    zs = torch.arange(depth, device=device, dtype=work_dtype)
    # Fix TracerWarning
    if normalized_coordinates:
        if not torch.jit.is_scripting() and (torch.jit.is_tracing() or is_compiling()):
            width_t = torch.scalar_tensor(width, device=xs.device, dtype=work_dtype)
            height_t = torch.scalar_tensor(height, device=ys.device, dtype=work_dtype)
            depth_t = torch.scalar_tensor(depth, device=zs.device, dtype=work_dtype)
            xs = torch.where(width_t > 1, (xs / (width_t - 1) - 0.5) * 2, xs * 0.0)
            ys = torch.where(height_t > 1, (ys / (height_t - 1) - 0.5) * 2, ys * 0.0)
            zs = torch.where(depth_t > 1, (zs / (depth_t - 1) - 0.5) * 2, zs * 0.0)
        else:
            xs = (xs / (width - 1) - 0.5) * 2 if width > 1 else xs * 0.0
            ys = (ys / (height - 1) - 0.5) * 2 if height > 1 else ys * 0.0
            zs = (zs / (depth - 1) - 0.5) * 2 if depth > 1 else zs * 0.0
    if widened:
        xs = xs.to(ramp_dtype)
        ys = ys.to(ramp_dtype)
        zs = zs.to(ramp_dtype)
    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([zs, xs, ys], indexing="ij"), dim=-1)  # DxWxHx3
    return base_grid.permute(0, 2, 1, 3).unsqueeze(0)  # 1xDxHxWx3

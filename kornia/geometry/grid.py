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
    if not torch.jit.is_scripting() and is_compiling():
        # ``linspace`` specializes symbolic ``steps`` under export; ``arange`` retains the
        # dynamic output length. Match ``linspace``'s default floating dtype when none is given.
        arange_dtype = dtype if dtype is not None else torch.get_default_dtype()
        xs = torch.arange(width, device=device, dtype=arange_dtype)
        ys = torch.arange(height, device=device, dtype=arange_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: keeping this formula inline avoids the extra tensors and shape checks incurred by
    #       normalize_pixel_coordinates. The two paths use the same singleton-centre policy.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        if torch.jit.is_tracing() or is_compiling():
            # Graph capture needs the tensor form to keep a symbolic size dynamic. Low-precision
            # floating types cannot represent every practical image size exactly (e.g. bfloat16
            # rounds 257 to 256 and 299 to 300), so the size arithmetic runs in float32. Divide
            # against the *unrounded* divisor and round the quotient, which is what eager does
            # against its Python-int divisor: casting the divisor down first would round it into
            # the coordinate dtype, which eager never does.
            work_dtype = torch.float32 if xs.dtype in (torch.float16, torch.bfloat16) else xs.dtype
            width_t = torch.scalar_tensor(width, device=xs.device, dtype=work_dtype)
            height_t = torch.scalar_tensor(height, device=ys.device, dtype=work_dtype)
            xs_norm = xs.to(work_dtype) / (width_t - 1)
            ys_norm = ys.to(work_dtype) / (height_t - 1)
            if xs.is_floating_point():
                # A widened half type rounds back down here, which is exactly where eager rounds.
                # An integral grid dtype must instead stay promoted into the default float, as the
                # eager true-division leaves it.
                xs_norm = xs_norm.to(xs.dtype)
                ys_norm = ys_norm.to(ys.dtype)
            xs = torch.where(width_t > 1, (xs_norm - 0.5) * 2, torch.zeros_like(xs_norm))
            ys = torch.where(height_t > 1, (ys_norm - 0.5) * 2, torch.zeros_like(ys_norm))
        else:
            # ``* 0.0`` rather than ``zeros_like`` so that a singleton axis follows the
            # same integer-to-float promotion the non-singleton branch performs.
            xs = (xs / (width - 1) - 0.5) * 2 if width > 1 else xs * 0.0
            ys = (ys / (height - 1) - 0.5) * 2 if height > 1 else ys * 0.0
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
    if not torch.jit.is_scripting() and is_compiling():
        arange_dtype = dtype if dtype is not None else torch.get_default_dtype()
        xs = torch.arange(width, device=device, dtype=arange_dtype)
        ys = torch.arange(height, device=device, dtype=arange_dtype)
        zs = torch.arange(depth, device=device, dtype=arange_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
        zs = torch.linspace(0, depth - 1, depth, device=device, dtype=dtype)
    # Fix TracerWarning
    if normalized_coordinates:
        if torch.jit.is_tracing() or is_compiling():
            # See ``create_meshgrid``: low-precision dtypes round large sizes, so the size
            # arithmetic runs in float32 and the quotient, not the divisor, is what gets cast down.
            work_dtype = torch.float32 if xs.dtype in (torch.float16, torch.bfloat16) else xs.dtype
            width_t = torch.scalar_tensor(width, device=xs.device, dtype=work_dtype)
            height_t = torch.scalar_tensor(height, device=ys.device, dtype=work_dtype)
            depth_t = torch.scalar_tensor(depth, device=zs.device, dtype=work_dtype)
            xs_norm = xs.to(work_dtype) / (width_t - 1)
            ys_norm = ys.to(work_dtype) / (height_t - 1)
            zs_norm = zs.to(work_dtype) / (depth_t - 1)
            if xs.is_floating_point():
                # As in ``create_meshgrid``: round a widened half type back down, but leave an
                # integral grid dtype promoted into the default float.
                xs_norm = xs_norm.to(xs.dtype)
                ys_norm = ys_norm.to(ys.dtype)
                zs_norm = zs_norm.to(zs.dtype)
            xs = torch.where(width_t > 1, (xs_norm - 0.5) * 2, torch.zeros_like(xs_norm))
            ys = torch.where(height_t > 1, (ys_norm - 0.5) * 2, torch.zeros_like(ys_norm))
            zs = torch.where(depth_t > 1, (zs_norm - 0.5) * 2, torch.zeros_like(zs_norm))
        else:
            xs = (xs / (width - 1) - 0.5) * 2 if width > 1 else xs * 0.0
            ys = (ys / (height - 1) - 0.5) * 2 if height > 1 else ys * 0.0
            zs = (zs / (depth - 1) - 0.5) * 2 if depth > 1 else zs * 0.0
    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([zs, xs, ys], indexing="ij"), dim=-1)  # DxWxHx3
    return base_grid.permute(0, 2, 1, 3).unsqueeze(0)  # 1xDxHxWx3

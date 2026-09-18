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

from typing import Optional, Tuple

import torch


def _resize_coordinates(
    start: torch.Tensor,
    end: torch.Tensor,
    input_size: int,
    output_size: int,
    mode: str,
    align_corners: Optional[bool],
) -> torch.Tensor:
    length = end - start
    positions = torch.arange(output_size, device=start.device, dtype=start.dtype)
    if mode == "nearest":
        # ATen computes the scale on the host (usually float32; see the CPU-double path). CUDA/Inductor
        # tensor float32 division can change pixels at integer boundaries (e.g. 26 -> 22).
        # Build scales in CPU float64 before casting, like host scalar arithmetic. Tensor
        # arange keeps input_size symbolic under dynamic=True; a Python range specializes it.
        # No float64 GPU/MPS ops or device-to-host copies of sampled coordinates are needed.
        scales = (torch.arange(input_size + 1, device="cpu", dtype=torch.float64) / output_size).to(start)
        return (positions * scales[length.long()]).float().floor()
    if align_corners:
        coordinates = positions * ((length - 1) / (output_size - 1) if output_size > 1 else length * 0.0)
    else:
        coordinates = (positions + 0.5) * (length / output_size) - 0.5
    # Reciprocal multiplication can round an identity scale below one (e.g. 41/41).
    # Keep exact integer indices for unchanged axes, including the no-resize center tap.
    return torch.where(length == output_size, positions, coordinates)


def _cubic_weights(fraction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # PyTorch interpolate's cubic convolution, A=-0.75 (ATen/native/UpSample.h).
    def inner(x: torch.Tensor) -> torch.Tensor:
        return ((1.25 * x - 2.25) * x) * x + 1.0

    def outer(x: torch.Tensor) -> torch.Tensor:
        return ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0

    return outer(fraction + 1), inner(fraction), inner(1 - fraction), outer(2 - fraction)


def _compiled_slice_resize(
    input: torch.Tensor, src: torch.Tensor, size: Tuple[int, int], mode: str, align_corners: Optional[bool]
) -> torch.Tensor:
    """Slice and resize without ragged intermediate tensors or Python crop coordinates.

    Match ``interpolate`` pixel locations and clamp each interpolation tap to the *crop*
    boundary, including bicubic's outer taps. A warp of the entire image would read outside
    that boundary. Integer box conversion matches ``crop_by_indices`` and stops box gradients.
    """
    batch, channels, height, width = input.shape
    # Nearest normally uses float32 index arithmetic; handle ATen's CPU-double exception below.
    coordinate_dtype = torch.float64 if input.dtype == torch.float64 and mode != "nearest" else torch.float32
    src = src.to(device=input.device, dtype=torch.long)
    # Match Python slicing for negative and out-of-bounds replay coordinates.
    x0, x1 = src[:, 0, 0:1], src[:, 1, 0:1] + 1
    y0, y1 = src[:, 0, 1:2], src[:, 3, 1:2] + 1
    x0 = torch.where(x0 < 0, x0 + width, x0).clamp(0, width).to(coordinate_dtype)
    x1 = torch.where(x1 < 0, x1 + width, x1).clamp(0, width).to(coordinate_dtype)
    y0 = torch.where(y0 < 0, y0 + height, y0).clamp(0, height).to(coordinate_dtype)
    y1 = torch.where(y1 < 0, y1 + height, y1).clamp(0, height).to(coordinate_dtype)
    x = _resize_coordinates(x0, x1, width, size[1], mode, align_corners)
    y = _resize_coordinates(y0, y1, height, size[0], mode, align_corners)
    if mode == "nearest" and input.device.type == "cpu" and input.dtype == torch.float64 and sum(size) > 128:
        # ATen's generic CPU nearest kernel uses double scales for double images,
        # then floorf; its channels-last kernel uses float scales. Match the layout
        # of the actual slices (including crop_by_indices' identical-box batch path).
        # _use_vectorized_kernel_cond_2d selects float scales for Hout + Wout <= 128
        # or channels-last slices with C > 3; otherwise HelperInterpNearest uses doubles.
        # The threshold is an ATen benchmark heuristic, present in both supported endpoints:
        # https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cpu/UpSampleKernel.cpp#L1539-L1574
        # https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        # Keep test_compile_nearest_large_output across torch versions: a rounding change
        # selects a different pixel, so this cannot be covered by a looser value tolerance.
        xd = _resize_coordinates(x0.double(), x1.double(), width, size[1], mode, align_corners)
        yd = _resize_coordinates(y0.double(), y1.double(), height, size[0], mode, align_corners)
        if channels > 3:
            h, w = y1 - y0, x1 - x0
            sn, sc, sh, sw = input.stride()
            channels_last = (sc == 1) & ((w == 1) | (sw == channels)) & ((h == 1) | (sh == channels * w))
            if batch > 1:
                corners = src[:, [0, 1, 0, 3], [0, 0, 1, 1]]
                identical = (corners == corners[:1]).all()
                channels_last = channels_last & (~identical | (sn == channels * h * w))
            x, y = torch.where(channels_last, x, xd), torch.where(channels_last, y, yd)
        else:
            x, y = xd, yd
    # Accumulate gradients in opmath precision before casting to half/bfloat16.
    # Nearest upsampling also needs this: many outputs can gather the same pixel,
    # and half atomic additions can saturate even with a single gather operation.
    work = input.float() if input.dtype in (torch.float16, torch.bfloat16) else input
    if mode != "nearest":
        work = work.to(coordinate_dtype)
    flat = work.reshape(batch, channels, height * width)

    def gather(x_index: torch.Tensor, y_index: torch.Tensor) -> torch.Tensor:
        x_index = (x0 + x_index.clamp(min=0)).minimum(x1 - 1).long()
        y_index = (y0 + y_index.clamp(min=0)).minimum(y1 - 1).long()
        indices = (y_index.unsqueeze(-1) * width + x_index.unsqueeze(-2)).reshape(batch, 1, -1)
        return flat.gather(2, indices.expand(-1, channels, -1)).reshape(batch, channels, *size)

    if mode == "nearest":
        return gather(x, y).to(input.dtype)

    def preserve_slice(result: torch.Tensor, slice_values: torch.Tensor) -> torch.Tensor:
        # crop_by_indices skips interpolate entirely when the slice already has the
        # requested size. Avoid spreading NaNs/Infs through zero-weight neighbors.
        unchanged = ((x1 - x0 == size[1]) & (y1 - y0 == size[0]))[:, :, None, None]
        # The central tap already equals the exact slice when both dimensions match.
        return torch.where(unchanged, slice_values, result).to(input.dtype)

    if mode == "bilinear":
        x, y = x.clamp(min=0), y.clamp(min=0)
    ix, iy = x.floor(), y.floor()
    fx, fy = x - ix, y - iy
    center = gather(ix, iy)
    if mode == "bilinear":
        wx, wy = fx[:, None, None, :], fy[:, None, :, None]
        top = center * (1 - wx) + gather(ix + 1, iy) * wx
        bottom = gather(ix, iy + 1) * (1 - wx) + gather(ix + 1, iy + 1) * wx
        return preserve_slice(top * (1 - wy) + bottom * wy, center)

    wxs, wys = _cubic_weights(fx), _cubic_weights(fy)
    result = input.new_zeros((batch, channels, *size), dtype=coordinate_dtype)
    for j in range(4):
        row = input.new_zeros((batch, channels, *size), dtype=coordinate_dtype)
        for i in range(4):
            tap = center if i == 1 and j == 1 else gather(ix + i - 1, iy + j - 1)
            row = row + tap * wxs[i][:, None, None, :]
        result = result + row * wys[j][:, None, :, None]
    return preserve_slice(result, center)

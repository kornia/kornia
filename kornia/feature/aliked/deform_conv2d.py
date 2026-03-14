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

"""Pure-PyTorch implementation of deform_conv2d matching torchvision.ops.deform_conv2d.

Implements both DCNv1 (mask=None) and DCNv2 (mask provided) as described in:
  - Deformable Convolutional Networks (https://arxiv.org/abs/1703.06211)
  - Deformable ConvNets v2 (https://arxiv.org/abs/1811.11168)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Deformable 2D convolution (DCNv1/v2) without torchvision dependency.

    Matches the signature and behaviour of ``torchvision.ops.deform_conv2d``.

    Args:
        input: input feature map ``(B, C_in, H_in, W_in)``.
        offset: offset tensor ``(B, 2 * n_offset_groups * kH * kW, H_out, W_out)``.
            Layout per kernel point: ``(dy, dx)`` interleaved.
        weight: convolution kernel ``(C_out, C_in / groups, kH, kW)``.
        bias: optional bias ``(C_out,)``.
        stride: convolution stride ``(sH, sW)``.
        padding: zero-padding ``(pH, pW)``.
        dilation: kernel dilation ``(dH, dW)``.
        mask: optional DCNv2 modulation mask
            ``(B, n_offset_groups * kH * kW, H_out, W_out)`` in ``[0, 1]``.

    Returns:
        Output feature map ``(B, C_out, H_out, W_out)``.
    """
    B, C_in, H_in, W_in = input.shape
    C_out, C_in_per_group, kH, kW = weight.shape

    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(stride, int):
        stride = (stride, stride)

    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    groups = C_in // C_in_per_group
    K = kH * kW
    n_off_grps = offset.shape[1] // (2 * K)
    c_per_off_grp = C_in // n_off_grps

    H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    # parse offsets: interleaved (dy, dx) per kernel point per group
    offset = offset.reshape(B, n_off_grps, 2 * K, H_out, W_out)
    off_y = offset[:, :, 0::2, :, :]  # [B, n_off_grps, K, H_out, W_out]
    off_x = offset[:, :, 1::2, :, :]

    if mask is not None:
        mask = mask.reshape(B, n_off_grps, K, H_out, W_out)

    # base sampling grid
    grid_oh = torch.arange(H_out, device=input.device, dtype=input.dtype)
    grid_ow = torch.arange(W_out, device=input.device, dtype=input.dtype)

    base_h = (grid_oh * sH - pH).reshape(1, 1, 1, H_out, 1)
    base_w = (grid_ow * sW - pW).reshape(1, 1, 1, 1, W_out)

    kh_offs = torch.arange(kH, device=input.device, dtype=input.dtype) * dH
    kw_offs = torch.arange(kW, device=input.device, dtype=input.dtype) * dW
    kern_h = kh_offs.repeat_interleave(kW).reshape(1, 1, K, 1, 1)
    kern_w = kw_offs.repeat(kH).reshape(1, 1, K, 1, 1)

    # Absolute fractional sampling positions [B, n_off_grps, K, H_out, W_out]
    sample_h = base_h + kern_h + off_y
    sample_w = base_w + kern_w + off_x

    # bilinear interpolation with zero-padding
    sampled = _bilinear_sample(input, sample_h, sample_w, n_off_grps, c_per_off_grp)
    # sampled: [B, n_off_grps, c_per_off_grp, K, H_out, W_out]

    # apply mask (DCNv2 modulation)
    if mask is not None:
        sampled = sampled * mask.unsqueeze(2)  # broadcast over channels

    # reshape for grouped matmul
    sampled = sampled.reshape(B, C_in, K, H_out, W_out)

    C_out_per_group = C_out // groups
    sampled = sampled.reshape(B, groups, C_in_per_group, K, H_out * W_out)
    weight_flat = weight.reshape(groups, C_out_per_group, C_in_per_group, K)

    out = torch.einsum("bgckn,gock->bgon", sampled, weight_flat)
    out = out.reshape(B, C_out, H_out, W_out)

    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)

    return out


def _bilinear_sample(
    input: Tensor,
    sample_h: Tensor,
    sample_w: Tensor,
    n_off_grps: int,
    c_per_off_grp: int,
) -> Tensor:
    """Sample input at fractional (sample_h, sample_w) positions with zero-padding.

    Uses the same corner-validity bilinear interpolation as torchvision's
    deformable-conv C++ kernel.

    Args:
        input: ``(B, C_in, H_in, W_in)``
        sample_h: ``(B, n_off_grps, K, H_out, W_out)``
        sample_w: ``(B, n_off_grps, K, H_out, W_out)``
        n_off_grps: number of offset groups.
        c_per_off_grp: input channels per offset group.

    Returns:
        Sampled tensor ``(B, n_off_grps, c_per_off_grp, K, H_out, W_out)``.
    """
    B, _C_in, H_in, W_in = input.shape

    h0 = sample_h.floor()
    w0 = sample_w.floor()

    lh = sample_h - h0
    lw = sample_w - w0

    h0 = h0.long()
    w0 = w0.long()
    h1 = h0 + 1
    w1 = w0 + 1

    def _valid(h: Tensor, w: Tensor) -> Tensor:
        return ((h >= 0) & (h < H_in) & (w >= 0) & (w < W_in)).to(input.dtype)

    m00 = _valid(h0, w0)
    m01 = _valid(h0, w1)
    m10 = _valid(h1, w0)
    m11 = _valid(h1, w1)

    h0c = h0.clamp(0, max(H_in - 1, 0))
    h1c = h1.clamp(0, max(H_in - 1, 0))
    w0c = w0.clamp(0, max(W_in - 1, 0))
    w1c = w1.clamp(0, max(W_in - 1, 0))

    # input -> [B, n_off_grps, c_per_off_grp, H_in * W_in]
    input_flat = input.reshape(B, n_off_grps, c_per_off_grp, H_in * W_in)

    flat_size = sample_h.shape[2] * sample_h.shape[3] * sample_h.shape[4]  # K*H_out*W_out

    def _gather(hh: Tensor, ww: Tensor) -> Tensor:
        idx = (hh * W_in + ww).reshape(B, n_off_grps, flat_size)
        idx = idx.unsqueeze(2).expand(-1, -1, c_per_off_grp, -1)
        return torch.gather(input_flat, 3, idx)

    v00 = _gather(h0c, w0c)
    v01 = _gather(h0c, w1c)
    v10 = _gather(h1c, w0c)
    v11 = _gather(h1c, w1c)

    K = sample_h.shape[2]
    H_out = sample_h.shape[3]
    W_out = sample_h.shape[4]
    shape = (B, n_off_grps, c_per_off_grp, K, H_out, W_out)

    v00 = v00.reshape(shape)
    v01 = v01.reshape(shape)
    v10 = v10.reshape(shape)
    v11 = v11.reshape(shape)

    lh = lh.unsqueeze(2)
    lw = lw.unsqueeze(2)
    m00 = m00.unsqueeze(2)
    m01 = m01.unsqueeze(2)
    m10 = m10.unsqueeze(2)
    m11 = m11.unsqueeze(2)

    return v00 * m00 * (1 - lh) * (1 - lw) + v01 * m01 * (1 - lh) * lw + v10 * m10 * lh * (1 - lw) + v11 * m11 * lh * lw

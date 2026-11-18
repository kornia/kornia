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
"""Optional Triton kernels of the CLIDD deformable sampling: bilinear sampling fused with the projection.

Adapted from https://github.com/HITCSC/CLIDD (``model/triton_plugin.py``), MIT License,
Copyright (c) 2024 HITCSC-Robotics; see ``kornia/feature/clidd/clidd.py`` for the full notice.

This module imports ``triton`` at import time. :mod:`kornia.feature.clidd.clidd` imports it only when Triton is
installed and the feature maps are on CUDA; everywhere else CLIDD uses the PyTorch formulation of the same operation,
:func:`torch.nn.functional.grid_sample` followed by :func:`torch.nn.functional.conv2d`.

Changes from the authors' kernels: the ``tl.dot`` precision follows ``torch.backends.cudnn.allow_tf32`` instead of
Triton's TF32 default, so the kernels agree with the PyTorch formulation (a cuDNN convolution) to float32 rounding
when TF32 is off; the forward takes the maps in NHWC only; and the weight of each level is laid out once per call.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl

from kornia.core._compat import torch_version_lt


@triton.jit
def _forward_kernel(  # noqa: PLR0913, PLR0917 (a Triton kernel takes every pointer, stride and size)
    output_ptr,
    input_ptr,
    grid_ptr,
    weight_ptr,
    bias_ptr,
    stride_out_b,
    stride_out_n,
    stride_out_c,
    stride_in_b,
    stride_in_h,
    stride_in_w,
    stride_in_c,
    stride_grid_b,
    stride_grid_n,
    stride_grid_m,
    stride_grid_d,
    stride_w_m,
    stride_w_cin,
    stride_w_cout,
    H,
    W,
    C_in,
    N,
    C_out,
    with_bias: tl.constexpr,
    align_corners: tl.constexpr,
    M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    b_idx = pid // num_n_blocks
    n_block_idx = pid % num_n_blocks

    n_offsets = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    for c_out_start in range(0, C_out, BLOCK_SIZE_C_OUT):
        c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
        mask_c_out = c_out_offsets < C_out
        acc = tl.zeros((BLOCK_N, BLOCK_SIZE_C_OUT), dtype=tl.float32)

        for m in range(M):
            grid_offset = b_idx * stride_grid_b + n_offsets * stride_grid_n + m * stride_grid_m
            x = tl.load(grid_ptr + grid_offset, mask=mask_n, other=0.0).to(tl.float32)
            y = tl.load(grid_ptr + grid_offset + stride_grid_d, mask=mask_n, other=0.0).to(tl.float32)

            if align_corners:
                ix = (x + 1) / 2.0 * (W - 1)
                iy = (y + 1) / 2.0 * (H - 1)
            else:
                ix = ((x + 1) * W - 1) / 2.0
                iy = ((y + 1) * H - 1) / 2.0

            ix0 = tl.math.floor(ix).to(tl.int32)
            iy0 = tl.math.floor(iy).to(tl.int32)
            dx = ix - ix0
            dy = iy - iy0
            w00 = (1.0 - dx) * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w10 = dx * (1.0 - dy)
            w11 = dx * dy
            ix1 = ix0 + 1
            iy1 = iy0 + 1

            mask_x0 = (ix0 >= 0) & (ix0 < W)
            mask_x1 = (ix1 >= 0) & (ix1 < W)
            mask_y0 = (iy0 >= 0) & (iy0 < H)
            mask_y1 = (iy1 >= 0) & (iy1 < H)

            m_00 = mask_n[:, None] & mask_x0[:, None] & mask_y0[:, None]
            m_01 = mask_n[:, None] & mask_x0[:, None] & mask_y1[:, None]
            m_10 = mask_n[:, None] & mask_x1[:, None] & mask_y0[:, None]
            m_11 = mask_n[:, None] & mask_x1[:, None] & mask_y1[:, None]

            for c_in_start in range(0, C_in, BLOCK_SIZE_C_IN):
                c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
                mask_c_in = c_in_offsets < C_in

                ptr_base = input_ptr + b_idx * stride_in_b + c_in_offsets[None, :] * stride_in_c
                val_00 = tl.load(
                    ptr_base + (iy0 * stride_in_h + ix0 * stride_in_w)[:, None],
                    mask=m_00 & mask_c_in[None, :],
                    other=0.0,
                ).to(tl.float32)
                val_01 = tl.load(
                    ptr_base + (iy1 * stride_in_h + ix0 * stride_in_w)[:, None],
                    mask=m_01 & mask_c_in[None, :],
                    other=0.0,
                ).to(tl.float32)
                val_10 = tl.load(
                    ptr_base + (iy0 * stride_in_h + ix1 * stride_in_w)[:, None],
                    mask=m_10 & mask_c_in[None, :],
                    other=0.0,
                ).to(tl.float32)
                val_11 = tl.load(
                    ptr_base + (iy1 * stride_in_h + ix1 * stride_in_w)[:, None],
                    mask=m_11 & mask_c_in[None, :],
                    other=0.0,
                ).to(tl.float32)

                s = w00[:, None] * val_00 + w01[:, None] * val_01 + w10[:, None] * val_10 + w11[:, None] * val_11

                w_ptr = (
                    weight_ptr
                    + m * stride_w_m
                    + c_in_offsets[:, None] * stride_w_cin
                    + c_out_offsets[None, :] * stride_w_cout
                )
                w_block = tl.load(w_ptr, mask=mask_c_in[:, None] & mask_c_out[None, :], other=0.0).to(tl.float32)
                acc += tl.dot(s, w_block, input_precision=INPUT_PRECISION)

        if with_bias:
            acc += tl.load(bias_ptr + c_out_offsets, mask=mask_c_out, other=0.0).to(tl.float32)[None, :]

        out_ptr = (
            output_ptr
            + b_idx * stride_out_b
            + n_offsets[:, None] * stride_out_n
            + c_out_offsets[None, :] * stride_out_c
        )
        tl.store(out_ptr, acc.to(output_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_c_out[None, :])


@triton.jit
def _backward_kernel(  # noqa: PLR0913, PLR0917 (a Triton kernel takes every pointer, stride and size)
    input_ptr,
    grid_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_grid_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    dout_ptr,
    stride_in_b,
    stride_in_h,
    stride_in_w,
    stride_in_c,
    stride_grid_b,
    stride_grid_n,
    stride_grid_m,
    stride_grid_d,
    stride_w_m,
    stride_w_cin,
    stride_w_cout,
    stride_gin_b,
    stride_gin_h,
    stride_gin_w,
    stride_gin_c,
    stride_ggrid_b,
    stride_ggrid_n,
    stride_ggrid_m,
    stride_ggrid_d,
    stride_gw_m,
    stride_gw_cin,
    stride_gw_cout,
    stride_dout_b,
    stride_dout_n,
    stride_dout_c,
    H,
    W,
    C_in,
    N,
    C_out,
    with_bias: tl.constexpr,
    align_corners: tl.constexpr,
    M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    b_idx = pid // num_n_blocks
    n_block_idx = pid % num_n_blocks

    n_offsets = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    if with_bias:
        for c_out_start in range(0, C_out, BLOCK_SIZE_C_OUT):
            c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
            mask_c_out = c_out_offsets < C_out
            dout_offset = (
                b_idx * stride_dout_b + n_offsets[:, None] * stride_dout_n + c_out_offsets[None, :] * stride_dout_c
            )
            dout_block = tl.load(dout_ptr + dout_offset, mask=mask_n[:, None] & mask_c_out[None, :], other=0.0).to(
                tl.float32
            )
            tl.atomic_add(grad_bias_ptr + c_out_offsets, tl.sum(dout_block, axis=0), mask=mask_c_out)

    for m in range(M):
        grid_base = b_idx * stride_grid_b + n_offsets * stride_grid_n + m * stride_grid_m
        x = tl.load(grid_ptr + grid_base, mask=mask_n, other=0.0).to(tl.float32)
        y = tl.load(grid_ptr + grid_base + stride_grid_d, mask=mask_n, other=0.0).to(tl.float32)

        if align_corners:
            ix = (x + 1.0) / 2.0 * (W - 1)
            iy = (y + 1.0) / 2.0 * (H - 1)
            scale_x = (W - 1) / 2.0
            scale_y = (H - 1) / 2.0
        else:
            ix = ((x + 1.0) * W - 1.0) / 2.0
            iy = ((y + 1.0) * H - 1.0) / 2.0
            scale_x = W / 2.0
            scale_y = H / 2.0

        ix0 = tl.math.floor(ix).to(tl.int32)
        iy0 = tl.math.floor(iy).to(tl.int32)
        dx = ix - ix0
        dy = iy - iy0
        w00 = (1.0 - dx) * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w10 = dx * (1.0 - dy)
        w11 = dx * dy
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        mask_x0 = (ix0 >= 0) & (ix0 < W)
        mask_x1 = (ix1 >= 0) & (ix1 < W)
        mask_y0 = (iy0 >= 0) & (iy0 < H)
        mask_y1 = (iy1 >= 0) & (iy1 < H)

        m_00 = mask_n[:, None] & mask_x0[:, None] & mask_y0[:, None]
        m_01 = mask_n[:, None] & mask_x0[:, None] & mask_y1[:, None]
        m_10 = mask_n[:, None] & mask_x1[:, None] & mask_y0[:, None]
        m_11 = mask_n[:, None] & mask_x1[:, None] & mask_y1[:, None]

        dl_dix = tl.zeros((BLOCK_N,), dtype=tl.float32)
        dl_diy = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for c_in_start in range(0, C_in, BLOCK_SIZE_C_IN):
            c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
            mask_c_in = c_in_offsets < C_in

            off_00 = (iy0 * stride_in_h + ix0 * stride_in_w)[:, None]
            off_01 = (iy1 * stride_in_h + ix0 * stride_in_w)[:, None]
            off_10 = (iy0 * stride_in_h + ix1 * stride_in_w)[:, None]
            off_11 = (iy1 * stride_in_h + ix1 * stride_in_w)[:, None]
            ptr_base = input_ptr + b_idx * stride_in_b + c_in_offsets[None, :] * stride_in_c
            val_00 = tl.load(ptr_base + off_00, mask=m_00 & mask_c_in[None, :], other=0.0).to(tl.float32)
            val_01 = tl.load(ptr_base + off_01, mask=m_01 & mask_c_in[None, :], other=0.0).to(tl.float32)
            val_10 = tl.load(ptr_base + off_10, mask=m_10 & mask_c_in[None, :], other=0.0).to(tl.float32)
            val_11 = tl.load(ptr_base + off_11, mask=m_11 & mask_c_in[None, :], other=0.0).to(tl.float32)

            s = w00[:, None] * val_00 + w01[:, None] * val_01 + w10[:, None] * val_10 + w11[:, None] * val_11
            g_s = tl.zeros((BLOCK_N, BLOCK_SIZE_C_IN), dtype=tl.float32)

            for c_out_start in range(0, C_out, BLOCK_SIZE_C_OUT):
                c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
                mask_c_out = c_out_offsets < C_out

                dout_offset = (
                    b_idx * stride_dout_b + n_offsets[:, None] * stride_dout_n + c_out_offsets[None, :] * stride_dout_c
                )
                dout_block = tl.load(dout_ptr + dout_offset, mask=mask_n[:, None] & mask_c_out[None, :], other=0.0).to(
                    tl.float32
                )

                w_ptr = (
                    weight_ptr
                    + m * stride_w_m
                    + c_in_offsets[:, None] * stride_w_cin
                    + c_out_offsets[None, :] * stride_w_cout
                )
                w_block = tl.load(w_ptr, mask=mask_c_in[:, None] & mask_c_out[None, :], other=0.0).to(tl.float32)
                g_s += tl.dot(dout_block, tl.trans(w_block), input_precision=INPUT_PRECISION)

                contrib_w = tl.dot(tl.trans(s), dout_block, input_precision=INPUT_PRECISION)
                gw_ptr = (
                    grad_weight_ptr
                    + m * stride_gw_m
                    + c_in_offsets[:, None] * stride_gw_cin
                    + c_out_offsets[None, :] * stride_gw_cout
                )
                tl.atomic_add(gw_ptr, contrib_w, mask=mask_c_in[:, None] & mask_c_out[None, :])

            gptr_base = grad_input_ptr + b_idx * stride_gin_b + c_in_offsets[None, :] * stride_gin_c
            tl.atomic_add(
                gptr_base + (iy0 * stride_gin_h + ix0 * stride_gin_w)[:, None],
                w00[:, None] * g_s,
                mask=m_00 & mask_c_in[None, :],
            )
            tl.atomic_add(
                gptr_base + (iy1 * stride_gin_h + ix0 * stride_gin_w)[:, None],
                w01[:, None] * g_s,
                mask=m_01 & mask_c_in[None, :],
            )
            tl.atomic_add(
                gptr_base + (iy0 * stride_gin_h + ix1 * stride_gin_w)[:, None],
                w10[:, None] * g_s,
                mask=m_10 & mask_c_in[None, :],
            )
            tl.atomic_add(
                gptr_base + (iy1 * stride_gin_h + ix1 * stride_gin_w)[:, None],
                w11[:, None] * g_s,
                mask=m_11 & mask_c_in[None, :],
            )

            ds_dix = (
                -(1.0 - dy)[:, None] * val_00
                - dy[:, None] * val_01
                + (1.0 - dy)[:, None] * val_10
                + dy[:, None] * val_11
            )
            ds_diy = (
                -(1.0 - dx)[:, None] * val_00
                + (1.0 - dx)[:, None] * val_01
                - dx[:, None] * val_10
                + dx[:, None] * val_11
            )
            dl_dix += tl.sum(g_s * ds_dix, axis=1)
            dl_diy += tl.sum(g_s * ds_diy, axis=1)

        ggrid_x_ptr = grad_grid_ptr + b_idx * stride_ggrid_b + n_offsets * stride_ggrid_n + m * stride_ggrid_m
        tl.store(ggrid_x_ptr, dl_dix * scale_x, mask=mask_n)
        tl.store(ggrid_x_ptr + stride_ggrid_d, dl_diy * scale_y, mask=mask_n)


def _next_pow2(a: int) -> int:
    return 1 if a <= 1 else 1 << ((a - 1).bit_length())


# The launchers, untyped: a type checker reads ``kernel[grid](...)`` as a call of the kernel's own signature, which a
# launch does not follow (it passes the ``tl.constexpr`` parameters by keyword and adds ``num_warps``). The same
# ``JITFunction`` objects, so Dynamo still traces the launches as user-defined Triton kernels.
_launch_forward: Any = _forward_kernel
_launch_backward: Any = _backward_kernel


def _launch_config(n: int, b: int, c_in: int, c_out: int) -> Tuple[Tuple[int], int, int, int]:
    block_n = 32
    block_c_in = max(min(32, _next_pow2(c_in)), 16)
    block_c_out = max(min(128, _next_pow2(c_out)), 16)
    return (triton.cdiv(n, block_n) * b,), block_n, block_c_in, block_c_out


def _input_precision() -> str:
    return "tf32" if torch.backends.cudnn.allow_tf32 else "ieee"


class _DeformableSampleProject(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input_nhwc: torch.Tensor,
        grid: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        align_corners: bool,
    ) -> torch.Tensor:
        B, H, W, C_in = input_nhwc.shape
        N, M = grid.shape[1], grid.shape[2]
        C_out = weight.shape[0]
        precision = _input_precision()
        ctx.save_for_backward(input_nhwc, grid, weight)
        ctx.with_bias = bias is not None
        ctx.align_corners = align_corners
        ctx.precision = precision

        # (C_out, C_in, 1, M) -> (M, C_in, C_out)
        weight_mcc = weight[:, :, 0].permute(2, 1, 0).contiguous()
        output = torch.empty((B, N, C_out), device=input_nhwc.device, dtype=input_nhwc.dtype)
        if N == 0 or B == 0:
            return output
        launch, block_n, block_c_in, block_c_out = _launch_config(N, B, C_in, C_out)
        _launch_forward[launch](
            output,
            input_nhwc,
            grid,
            weight_mcc,
            bias if bias is not None else weight_mcc,
            *output.stride(),
            *input_nhwc.stride(),
            *grid.stride(),
            *weight_mcc.stride(),
            H,
            W,
            C_in,
            N,
            C_out,
            with_bias=bias is not None,
            align_corners=align_corners,
            M=M,
            INPUT_PRECISION=precision,
            BLOCK_SIZE_C_OUT=block_c_out,
            BLOCK_SIZE_C_IN=block_c_in,
            BLOCK_N=block_n,
            num_warps=4,
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        input_nhwc, grid, weight = ctx.saved_tensors
        B, H, W, C_in = input_nhwc.shape
        N, M = grid.shape[1], grid.shape[2]
        C_out = weight.shape[0]
        weight_mcc = weight[:, :, 0].permute(2, 1, 0).contiguous()

        # Accumulate in float32, as the forward does, rather than with half-precision atomics. On U128 at 480 x 640 with
        # 4,096 keypoints this changed no gradient measurably: half-precision rounding elsewhere dominates.
        grad_input = torch.zeros(input_nhwc.shape, device=input_nhwc.device, dtype=torch.float32)
        grad_grid = torch.zeros(grid.shape, device=grid.device, dtype=torch.float32)
        grad_weight_mcc = torch.zeros(weight_mcc.shape, device=weight.device, dtype=torch.float32)
        grad_bias = torch.zeros((C_out,), device=weight.device, dtype=torch.float32)
        if N > 0 and B > 0:
            launch, block_n, block_c_in, block_c_out = _launch_config(N, B, C_in, C_out)
            _launch_backward[launch](
                input_nhwc,
                grid,
                weight_mcc,
                grad_input,
                grad_grid,
                grad_weight_mcc,
                grad_bias,
                grad_output,
                *input_nhwc.stride(),
                *grid.stride(),
                *weight_mcc.stride(),
                *grad_input.stride(),
                *grad_grid.stride(),
                *grad_weight_mcc.stride(),
                *grad_output.stride(),
                H,
                W,
                C_in,
                N,
                C_out,
                with_bias=ctx.with_bias,
                align_corners=ctx.align_corners,
                M=M,
                INPUT_PRECISION=ctx.precision,
                BLOCK_SIZE_C_OUT=block_c_out,
                BLOCK_SIZE_C_IN=block_c_in,
                BLOCK_N=block_n,
                num_warps=4,
            )
        grad_weight = grad_weight_mcc.permute(2, 1, 0).unsqueeze(2).to(weight.dtype)
        return (
            grad_input.to(input_nhwc.dtype),
            grad_grid.to(grid.dtype),
            grad_weight,
            grad_bias.to(weight.dtype) if ctx.with_bias else None,
            None,
        )


def deformable_sample_project(
    input_nhwc: torch.Tensor,
    grid: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    align_corners: bool,
) -> torch.Tensor:
    """Sample ``input_nhwc`` bilinearly at ``grid`` and project the samples with a ``(1, M)`` convolution.

    Equal to ``conv2d(grid_sample(input, grid, align_corners=align_corners), weight, bias)`` with zero padding, as a
    ``(B, N, C_out)`` tensor, without materialising the ``(B, C_in, N, M)`` samples.

    Args:
        input_nhwc: CUDA feature map ``(B, H, W, C_in)``, contiguous for coalesced reads.
        grid: normalised ``(x, y)`` sampling positions ``(B, N, M, 2)``.
        weight: projection ``(C_out, C_in, 1, M)``.
        bias: optional projection bias ``(C_out,)``.
        align_corners: as in :func:`torch.nn.functional.grid_sample`.

    Returns:
        The projected samples ``(B, N, C_out)`` in the dtype of ``input_nhwc``.
    """
    return _DeformableSampleProject.apply(input_nhwc, grid, weight, bias, align_corners)


if torch_version_lt(2, 7, 0):
    # Inductor before torch 2.7 miscompiles these kernels. In torch 2.5 a compiled describe returned wrong descriptors
    # (0.46 away from eager) or failed with "NameError: name 's2' is not defined" with gradients and dynamic shapes. In
    # torch 2.6, Dynamo traces the autograd function only without gradients (with them its strict mode rejects
    # Tensor.stride); passing strides computed from the shapes made it trace with gradients too, and the compiled
    # gradient with respect to the image was then 0.9 % of the largest away from eager (G128, float32, TF32 off),
    # against 1e-6 in torch 2.7.1, 2.9.1 and 2.15. Keep the kernels out of the graph there: eager calls are unchanged,
    # and a compiled caller breaks the graph around them, or raises under fullgraph=True. Retire with torch 2.6 support.
    deformable_sample_project = torch.compiler.disable(deformable_sample_project)

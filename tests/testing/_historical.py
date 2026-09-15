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

"""Graph-capture branches of kornia functions as they stood at named commits of kornia#4006.

Each function is the exact arithmetic of its commit's ``torch.jit.is_tracing()`` branch, with the
eager branch kept so that ``assert_capture_matches_eager`` can compare the two. They are the
regression fixtures for ``testing.precision``: the helper must FAIL on each of them.
"""

from __future__ import annotations

import torch


def _xs_ys(height: int, width: int, device: torch.device | None, dtype: torch.dtype | None):
    xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    return xs, ys


def _stack(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    base_grid = torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1)
    return base_grid.permute(1, 0, 2).unsqueeze(0)


def create_meshgrid_9ed891c5(height, width, device=None, dtype=None):
    """kornia#4006 @ 9ed891c5 — divisor built directly in the coordinate dtype (wave-5 finding)."""
    xs, ys = _xs_ys(height, width, device, dtype)
    if torch.jit.is_tracing():
        width_t = torch.scalar_tensor(width, device=xs.device, dtype=xs.dtype)
        height_t = torch.scalar_tensor(height, device=ys.device, dtype=ys.dtype)
        xs = torch.where(width_t > 1, (xs / (width_t - 1) - 0.5) * 2, torch.zeros_like(xs))
        ys = torch.where(height_t > 1, (ys / (height_t - 1) - 0.5) * 2, torch.zeros_like(ys))
    else:
        xs = (xs / (width - 1) - 0.5) * 2 if width > 1 else xs * 0.0
        ys = (ys / (height - 1) - 0.5) * 2 if height > 1 else ys * 0.0
    return _stack(xs, ys)


def create_meshgrid_32ab0eeb(height, width, device=None, dtype=None):
    """kornia#4006 @ 32ab0eeb — float32 arithmetic, but the divisor is cast down first (wave-8)."""
    xs, ys = _xs_ys(height, width, device, dtype)
    if torch.jit.is_tracing():
        work_dtype = torch.float32 if xs.dtype in (torch.float16, torch.bfloat16) else xs.dtype
        width_t = torch.scalar_tensor(width, device=xs.device, dtype=work_dtype)
        height_t = torch.scalar_tensor(height, device=ys.device, dtype=work_dtype)
        w_den = (width_t - 1).to(xs.dtype)
        h_den = (height_t - 1).to(ys.dtype)
        xs = torch.where(width_t > 1, (xs / w_den - 0.5) * 2, torch.zeros_like(xs))
        ys = torch.where(height_t > 1, (ys / h_den - 0.5) * 2, torch.zeros_like(ys))
    else:
        xs = (xs / (width - 1) - 0.5) * 2 if width > 1 else xs * 0.0
        ys = (ys / (height - 1) - 0.5) * 2 if height > 1 else ys * 0.0
    return _stack(xs, ys)


def normal_transform_pixel_1522441d(height, width, device=None, dtype=None):
    """kornia#4006 @ 1522441d — promotion keyed off the ``dtype`` ARGUMENT, so ``dtype=None`` under a
    half default dtype rounds the size (wave-9 finding)."""
    if not torch.jit.is_tracing():
        sx = 1.0 if width == 1 else 2.0 / (width - 1.0)
        sy = 1.0 if height == 1 else 2.0 / (height - 1.0)
        tx = 0.0 if width == 1 else -1.0
        ty = 0.0 if height == 1 else -1.0
        return torch.tensor([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], device=device, dtype=dtype)[None]
    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    if work_dtype not in (torch.float32, torch.float64):
        work_dtype = torch.get_default_dtype()
    width_t = torch.scalar_tensor(width, device=device, dtype=work_dtype)
    height_t = torch.scalar_tensor(height, device=device, dtype=work_dtype)
    one = torch.ones((), device=device, dtype=work_dtype)
    zero = torch.zeros((), device=device, dtype=work_dtype)
    sx_t = torch.where(width_t == 1, one, 2.0 / (width_t - 1.0))
    sy_t = torch.where(height_t == 1, one, 2.0 / (height_t - 1.0))
    tx_t = torch.where(width_t == 1, zero, -one)
    ty_t = torch.where(height_t == 1, zero, -one)
    return torch.stack(
        [torch.stack([sx_t, zero, tx_t]), torch.stack([zero, sy_t, ty_t]), torch.stack([zero, zero, one])]
    ).to(dtype=dtype)[None]

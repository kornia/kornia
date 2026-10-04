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

import torch


def _legacy_nearest_affine(
    source_size: torch.Tensor, output_size: torch.Tensor, source_offset: torch.Tensor, max_source_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit an affine map to the centres of legacy nearest's copied pixel blocks.

    PyTorch's legacy nearest interpolation maps output index ``u`` to
    ``floor(u * source_size / output_size)``. Since this is piecewise constant,
    the returned affine is the least-squares fit over source pixels represented
    in the output; ``source_offset`` converts crop-local coordinates to input
    coordinates.
    """
    source_size = source_size.to(dtype=torch.long)
    output_size = output_size.to(device=source_size.device, dtype=torch.long)
    source_offset = source_offset.to(device=source_size.device)
    source_index = torch.arange(max_source_size, device=source_size.device).unsqueeze(0)
    source_index = source_index.expand(source_size.shape[0], -1)
    source_size = source_size.unsqueeze(-1)
    output_size = output_size.unsqueeze(-1)

    # The output indices copying source pixel q form the integer interval
    # [ceil(q * O / I), ceil((q + 1) * O / I) - 1].
    first = torch.div(source_index * output_size + source_size - 1, source_size, rounding_mode="floor")
    end = torch.div((source_index + 1) * output_size + source_size - 1, source_size, rounding_mode="floor")
    last = end - 1
    represented = (source_index < source_size) & (first <= last) & (first < output_size)
    centres = (first + last).to(torch.float32) / 2
    weights = represented.to(torch.float32)
    count = weights.sum(dim=-1, keepdim=True).clamp_min(1)
    x = source_index.to(torch.float32)
    mean_x = (x * weights).sum(dim=-1, keepdim=True) / count
    mean_y = (centres * weights).sum(dim=-1, keepdim=True) / count
    centered_x = x - mean_x
    variance = (centered_x.square() * weights).sum(dim=-1, keepdim=True)
    covariance = (centered_x * (centres - mean_y) * weights).sum(dim=-1, keepdim=True)
    slope = torch.where(
        variance > 0, covariance / variance.clamp_min(torch.finfo(variance.dtype).tiny), torch.zeros_like(variance)
    )
    intercept = mean_y - slope * mean_x

    slope = slope.squeeze(-1).to(dtype=source_offset.dtype)
    intercept = intercept.squeeze(-1).to(dtype=source_offset.dtype) - slope * source_offset
    return slope, intercept

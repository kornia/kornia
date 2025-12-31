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

import math
from typing import Tuple

import torch


def arange_sequence(ranges: torch.Tensor) -> torch.Tensor:
    """Return a sequence of the ranges specified by the argument.

    Example:
    [2, 5, 1, 2] -> [0, 1, 0, 1, 2, 3, 4, 0, 0, 1]

    """
    maxcnt = torch.max(ranges).item()
    numuni = ranges.shape[0]
    complete_ranges = torch.arange(maxcnt, device=ranges.device).unsqueeze(0).expand(numuni, -1)

    return complete_ranges[complete_ranges < ranges.unsqueeze(-1)]


def dist_matrix(d1: torch.Tensor, d2: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
    """Distance between two tensors."""
    if is_normalized:
        return 2 - 2.0 * d1 @ d2.t()
    x_norm = (d1**2).sum(1).view(-1, 1)
    y_norm = (d2**2).sum(1).view(1, -1)
    # print(x_norm, y_norm)
    distmat = x_norm + y_norm - 2.0 * d1 @ d2.t()
    # distmat[torch.isnan(distmat)] = np.inf
    return distmat


def orientation_diff(o1: torch.Tensor, o2: torch.Tensor) -> torch.Tensor:
    """Orientation difference between two tensors."""
    diff = o2 - o1
    diff[diff < -180] += 360
    diff[diff >= 180] -= 360
    return diff


def piecewise_arange(piecewise_idxer: torch.Tensor) -> torch.Tensor:
    """Count repeated indices.

    Example:
    [0, 0, 0, 3, 3, 3, 3, 1, 1, 2] -> [0, 1, 2, 0, 1, 2, 3, 0, 1, 0]
    """
    dv = piecewise_idxer.device
    # print(piecewise_idxer)
    uni: torch.Tensor
    uni, counts = torch.unique_consecutive(piecewise_idxer, return_counts=True)
    # print(counts)
    maxcnt = int(torch.max(counts).item())
    numuni = uni.shape[0]
    tmp = torch.zeros(size=(numuni, maxcnt), device=dv).bool()
    ranges = torch.arange(maxcnt, device=dv).unsqueeze(0).expand(numuni, -1)
    tmp[ranges < counts.unsqueeze(-1)] = True
    return ranges[tmp]


def batch_2x2_inv(m: torch.Tensor, check_dets: bool = False) -> torch.Tensor:
    """Return inverse of batch of 2x2 matrices."""
    a = m[..., 0, 0]
    b = m[..., 0, 1]
    c = m[..., 1, 0]
    d = m[..., 1, 1]
    minv = torch.empty_like(m)
    det = a * d - b * c
    if check_dets:
        det[torch.abs(det) < 1e-10] = 1e-10
    minv[..., 0, 0] = d
    minv[..., 1, 1] = a
    minv[..., 0, 1] = -b
    minv[..., 1, 0] = -c
    return minv / det.unsqueeze(-1).unsqueeze(-1)


def batch_2x2_Q(m: torch.Tensor) -> torch.Tensor:
    """Return Q of batch of 2x2 matrices."""
    return batch_2x2_inv(batch_2x2_invQ(m), check_dets=True)


def batch_2x2_invQ(m: torch.Tensor) -> torch.Tensor:
    """Return inverse Q of batch of 2x2 matrices."""
    return m @ m.transpose(-1, -2)


def batch_2x2_det(m: torch.Tensor) -> torch.Tensor:
    """Return determinant of batch of 2x2 matrices."""
    a = m[..., 0, 0]
    b = m[..., 0, 1]
    c = m[..., 1, 0]
    d = m[..., 1, 1]
    return a * d - b * c


def batch_2x2_ellipse(m: torch.Tensor, *, eps: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Eigenvalues and Eigenvectors of batch of 2x2 matrices."""
    am = m[..., 0, 0]
    bm = m[..., 0, 1]
    cm = m[..., 1, 0]
    dm = m[..., 1, 1]

    a = am * am + bm * bm
    b = am * cm + bm * dm
    d = cm * cm + dm * dm

    trh = 0.5 * (a + d)
    diff = 0.5 * (a - d)

    # stable hypot
    sqrtdisc = torch.hypot(diff, b)

    e1 = trh + sqrtdisc
    e2 = trh - sqrtdisc
    if eps > 0:
        e1 = e1.clamp(min=eps)
        e2 = e2.clamp(min=eps)
    else:
        e1 = e1.clamp(min=0.0)
        e2 = e2.clamp(min=0.0)
    eigenvals = torch.stack([e1, e2], dim=-1)

    theta = 0.5 * torch.atan2(2.0 * b, a - d)
    c = torch.cos(theta)
    s = torch.sin(theta)

    ev1 = torch.stack([c, s], dim=-1)  # (...,2)
    ev2 = torch.stack([-s, c], dim=-1)  # orthogonal (...,2)
    eigenvecs = torch.stack([ev1, ev2], dim=-1)  # (...,2,2) columns are eigenvectors
    return eigenvals, eigenvecs


def draw_first_k_couples(k: int, rdims: torch.Tensor, dv: torch.device) -> torch.Tensor:
    """Return first k couples.

    Exhaustive search over the first n samples:
     * n(n+1)/2 = n2/2 + n/2 couples
    Max n for which we can exhaustively sample with k couples:
    * n2/2 + n/2 = k
    * n = sqrt(1/4 + 2k)-1/2 = (sqrt(8k+1)-1)/2
    """
    max_exhaustive_search = int(math.sqrt(2 * k + 0.25) - 0.5)
    residual_search = int(k - max_exhaustive_search * (max_exhaustive_search + 1) / 2)

    repeats = torch.cat(
        [
            torch.arange(max_exhaustive_search, dtype=torch.long, device=dv) + 1,
            torch.tensor([residual_search], dtype=torch.long, device=dv),
        ]
    )
    idx_sequence = torch.stack([repeats.repeat_interleave(repeats), arange_sequence(repeats)], dim=-1)
    return torch.remainder(idx_sequence.unsqueeze(-1), rdims)


def random_samples_indices(iters: int, rdims: torch.Tensor, dv: torch.device) -> torch.Tensor:
    """Randomly sample indices of torch.tensor."""
    rands = torch.rand(size=(iters, 2, rdims.shape[0]), device=dv)
    scaled_rands = rands * (rdims - 1e-8).float()
    rand_samples_rel = scaled_rands.long()
    return rand_samples_rel

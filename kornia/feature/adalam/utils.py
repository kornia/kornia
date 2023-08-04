import math
from typing import Tuple

import torch

from kornia.core import Tensor


def arange_sequence(ranges: Tensor) -> Tensor:
    """Returns a sequence of the ranges specified by the argument.

    Example:
    [2, 5, 1, 2] -> [0, 1, 0, 1, 2, 3, 4, 0, 0, 1]
    """
    maxcnt = torch.max(ranges).item()
    numuni = ranges.shape[0]
    complete_ranges = torch.arange(maxcnt, device=ranges.device).unsqueeze(0).expand(numuni, -1)

    return complete_ranges[complete_ranges < ranges.unsqueeze(-1)]


def dist_matrix(d1: Tensor, d2: Tensor, is_normalized: bool = False) -> Tensor:
    if is_normalized:
        return 2 - 2.0 * d1 @ d2.t()
    x_norm = (d1**2).sum(1).view(-1, 1)
    y_norm = (d2**2).sum(1).view(1, -1)
    # print(x_norm, y_norm)
    distmat = x_norm + y_norm - 2.0 * d1 @ d2.t()
    # distmat[torch.isnan(distmat)] = np.inf
    return distmat


def orientation_diff(o1: Tensor, o2: Tensor) -> Tensor:
    diff = o2 - o1
    diff[diff < -180] += 360
    diff[diff >= 180] -= 360
    return diff


def piecewise_arange(piecewise_idxer: Tensor) -> Tensor:
    """
    count repeated indices
    Example:
    [0, 0, 0, 3, 3, 3, 3, 1, 1, 2] -> [0, 1, 2, 0, 1, 2, 3, 0, 1, 0]
    """
    dv = piecewise_idxer.device
    # print(piecewise_idxer)
    uni: Tensor
    uni, counts = torch.unique_consecutive(piecewise_idxer, return_counts=True)
    # print(counts)
    maxcnt = int(torch.max(counts).item())
    numuni = uni.shape[0]
    tmp = torch.zeros(size=(numuni, maxcnt), device=dv).bool()
    ranges = torch.arange(maxcnt, device=dv).unsqueeze(0).expand(numuni, -1)
    tmp[ranges < counts.unsqueeze(-1)] = True
    return ranges[tmp]


def batch_2x2_inv(m: Tensor, check_dets: bool = False) -> Tensor:
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


def batch_2x2_Q(m: Tensor) -> Tensor:
    return batch_2x2_inv(batch_2x2_invQ(m), check_dets=True)


def batch_2x2_invQ(m: Tensor) -> Tensor:
    return m @ m.transpose(-1, -2)


def batch_2x2_det(m: Tensor) -> Tensor:
    a = m[..., 0, 0]
    b = m[..., 0, 1]
    c = m[..., 1, 0]
    d = m[..., 1, 1]
    return a * d - b * c


def batch_2x2_ellipse(m: Tensor) -> Tuple[Tensor, Tensor]:
    am = m[..., 0, 0]
    bm = m[..., 0, 1]
    cm = m[..., 1, 0]
    dm = m[..., 1, 1]

    a = am**2 + bm**2
    b = am * cm + bm * dm
    d = cm**2 + dm**2

    trh = (a + d) / 2
    # det = a * d - b * c
    sqrtdisc = torch.sqrt(((a - d) / 2) ** 2 + b**2)

    eigenvals = torch.stack([trh + sqrtdisc, trh - sqrtdisc], dim=-1).clamp(min=0)
    dens = eigenvals - a.unsqueeze(-1)
    dens[torch.abs(dens) < 1e-6] = 1e-6
    eigenvecs = torch.stack([b.unsqueeze(-1) / dens, torch.ones_like(dens)], dim=-2)
    eigenvecs = eigenvecs / torch.norm(eigenvecs, dim=-2, keepdim=True)

    # err = eigenvecs @ torch.diag_embed(eigenvals) @ eigenvecs.transpose(-2, -1) - q
    return eigenvals, eigenvecs


def draw_first_k_couples(k: int, rdims: Tensor, dv: torch.device) -> Tensor:
    # exhaustive search over the first n samples:
    # n(n+1)/2 = n2/2 + n/2 couples
    # max n for which we can exhaustively sample with k couples:
    # n2/2 + n/2 = k
    # n = sqrt(1/4 + 2k)-1/2 = (sqrt(8k+1)-1)/2
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


def random_samples_indices(iters: int, rdims: Tensor, dv: torch.device) -> Tensor:
    rands = torch.rand(size=(iters, 2, rdims.shape[0]), device=dv)
    scaled_rands = rands * (rdims - 1e-8).float()
    rand_samples_rel = scaled_rands.long()
    return rand_samples_rel

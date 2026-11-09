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

from typing import Any, Dict, Tuple, Union

import torch

from kornia.geometry.solvers.affine import _find_affine_linear, _find_affine_linear_pair

from .utils import arange_sequence, batch_2x2_eigenvals, draw_first_k_couples


def stable_sort_residuals(residuals: torch.Tensor, ransidx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort residuals."""
    if residuals.dtype == torch.float32:
        # Non-negative IEEE-754 float32 bit patterns are order-preserving; packing group IDs
        # in the high 32 bits sorts by (group, residual) in one pass without precision loss.
        bits = (residuals + 0.0).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        key = (ransidx.to(torch.int64).unsqueeze(0) << 32) | bits
        sorting_idxes = torch.argsort(key, dim=-1, stable=True)
        return residuals.gather(1, sorting_idxes), sorting_idxes
    # Two stable sorts preserve exact residual ordering for float64 or other dtypes.
    sorting_idxes = torch.argsort(residuals, dim=-1, stable=True)
    by_group = torch.argsort(ransidx[sorting_idxes], dim=-1, stable=True)
    sorting_idxes = sorting_idxes.gather(1, by_group)

    return residuals.gather(1, sorting_idxes), sorting_idxes


def _distinct_samples(xsamples: torch.Tensor, ysamples: torch.Tensor, ransidx: torch.Tensor) -> torch.Tensor:
    """Mark correspondences whose source and target locations are both first seen in their group."""
    distinct = torch.ones_like(ransidx, dtype=torch.bool)
    for points in (xsamples, ysamples):
        if points.dtype == torch.float32:
            xy = (points + 0.0).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            key = (xy[:, 0] << 32) | xy[:, 1]
            order = torch.argsort(key, stable=True)
            order = order[torch.argsort(ransidx[order], stable=True)]
            repeated = (ransidx[order[1:]] == ransidx[order[:-1]]) & (key[order[1:]] == key[order[:-1]])
        else:
            order = torch.argsort(points[:, 1], stable=True)
            order = order[torch.argsort(points[order, 0], stable=True)]
            order = order[torch.argsort(ransidx[order], stable=True)]
            repeated = (ransidx[order[1:]] == ransidx[order[:-1]]) & (points[order[1:]] == points[order[:-1]]).all(
                dim=-1
            )
        distinct[order[1:][repeated]] = False
    return distinct


def group_sum_and_cumsum(
    scores_mat: torch.Tensor, end_group_idx: torch.Tensor, group_idx: Union[torch.Tensor, slice, None] = None
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """Calculate cumulative sum over group."""
    cumulative_scores = torch.cumsum(scores_mat, dim=1)
    ending_cumusums = cumulative_scores[:, end_group_idx]
    shifted_ending_cumusums = torch.cat(
        [
            torch.zeros(size=(ending_cumusums.shape[0], 1), dtype=ending_cumusums.dtype, device=scores_mat.device),
            ending_cumusums[:, :-1],
        ],
        dim=1,
    )
    grouped_sums = ending_cumusums - shifted_ending_cumusums

    if group_idx is not None:
        grouped_cumsums = cumulative_scores - shifted_ending_cumusums[:, group_idx]
        return grouped_sums, grouped_cumsums
    return grouped_sums, None


def _select_inliers(
    residuals: torch.Tensor,
    xsamples: torch.Tensor,
    ysamples: torch.Tensor,
    ransidx: torch.Tensor,
    rdims: torch.Tensor,
    idxoffsets: torch.Tensor,
    affinities: torch.Tensor,
    config: Dict[str, Any],
    valid_affinities: Union[torch.Tensor, None] = None,
    distinct: Union[torch.Tensor, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a consensus using independent geometric support, then return its samples.

    Equal residuals do not imply duplicate keypoints. Count each source and target
    location at most once per neighbourhood, and reserve three correspondences for
    the seed and fitting pair. Duplicate correspondences may be returned but cannot
    manufacture evidence. Counts always describe the returned set, not its evidence.
    """
    dv = residuals.device
    min_inliers, min_confidence = config["min_inliers"], config["min_confidence"]
    sorted_res, sorting_idxes = stable_sort_residuals(residuals, ransidx)
    # Keep the first input representative (the descriptor-score preference from
    # extract_local_patterns), as the former exact-fit recovery did. Conflicting
    # duplicates cannot switch representatives to support each fitted hypothesis.
    # Cost: CPU float32, 4 threads, SIFT2048 ubc/graf 1->2:
    # public match_adalam 121/128 ms (main) -> 83/84 ms.
    if distinct is None:
        distinct = _distinct_samples(xsamples, ysamples, ransidx)

    ends = rdims.cumsum(0) - 1
    weights = distinct.to(residuals.dtype)
    dist_cumsum = weights.cumsum(0)
    end_totals = dist_cumsum[ends]
    start_totals = torch.cat([end_totals.new_zeros(1), end_totals[:-1]])
    group_totals = (end_totals - start_totals).clamp_min_(1)
    min_denom = (group_totals * min_confidence)[ransidx].unsqueeze(0)

    evidence = weights[sorting_idxes].cumsum_(dim=1)
    evidence.sub_(start_totals[ransidx] + 3).clamp_min_(0)
    # A finite floor prevents exact fits receiving infinite confidence. No residual
    # quantization: the former float16 bins merged distinct subpixel correspondences.
    areas = sorted_res.square().clamp_min_(2.0**-24)
    eligible = (evidence >= min_inliers) & (evidence >= min_denom * areas)
    win = int(min_inliers)
    if 0 < win < sorted_res.shape[1]:
        # Require the marginal density over the last min_inliers samples to also meet
        # min_confidence once past the initial minimal support, so a dense exact core
        # cannot subsidize a sparse outlier tail.
        d_area = (areas[:, win:] - areas[:, :-win]).clamp_min_(2.0**-24)
        eligible[:, win:] &= (evidence[:, win:] - evidence[:, :-win]) >= (min_denom[:, win:] * d_area)
    if valid_affinities is None:
        eigenvals = batch_2x2_eigenvals(affinities)
        scale_limit = config["detected_scale_rate_threshold"] ** 2
        valid_affinities = (eigenvals[..., 1] >= 1 / scale_limit) & (eigenvals[..., 0] <= scale_limit)

    # The largest qualifying radius defines one coherent consensus per hypothesis.
    # Earlier samples belong to it even when their own prefixes had too little evidence.
    positions = torch.arange(len(ransidx), device=dv, dtype=torch.int32).expand_as(sorted_res)
    boundary = torch.cummax(torch.where(eligible, positions, -1), dim=1).values[:, ends].to(torch.long)
    boundary = torch.where(valid_affinities & (boundary >= idxoffsets), boundary, -1)
    safe_boundary = boundary.clamp_min(0)
    support = evidence.gather(1, safe_boundary)
    support = torch.where(boundary >= 0, support, 0)
    best_support = support.max(dim=0).values
    hypothesis_confidence = support / (group_totals.unsqueeze(0) * areas.gather(1, safe_boundary))
    # Equal-size consensuses prefer the tighter fit, rather than whichever pair
    # happened to be sampled first.
    best_iters = torch.where(support == best_support, hypothesis_confidence, -1).max(dim=0).indices
    groups = torch.arange(len(rdims), device=dv)
    best_boundary = boundary[best_iters, groups]
    counts = (best_boundary - idxoffsets + 1).clamp_min(0)
    sample_positions = torch.arange(len(ransidx), device=dv)
    selected = sample_positions <= best_boundary[ransidx]
    samples = sorting_idxes[best_iters[ransidx], sample_positions][selected]
    best_confidence = hypothesis_confidence[best_iters, groups]
    best_confidence = torch.where(best_boundary >= 0, best_confidence, 0)
    return ransidx[selected], samples, counts, best_iters, best_confidence


def sample_padded_inliers(
    xsamples: torch.Tensor,
    ysamples: torch.Tensor,
    inlier_counts: torch.Tensor,
    inl_ransidx: torch.Tensor,
    inl_sampleidx: torch.Tensor,
    numransacs: int,
    dv: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample from padded inliers."""
    maxinliers = int(torch.max(inlier_counts).item())
    dtype = xsamples.dtype
    padded_inlier_x = torch.zeros(size=(numransacs, maxinliers, 2), device=dv, dtype=dtype)
    padded_inlier_y = torch.zeros(size=(numransacs, maxinliers, 2), device=dv, dtype=dtype)

    col_idx = arange_sequence(inlier_counts)
    padded_inlier_x[inl_ransidx, col_idx] = xsamples[inl_sampleidx]
    padded_inlier_y[inl_ransidx, col_idx] = ysamples[inl_sampleidx]

    return padded_inlier_x, padded_inlier_y


def _reset_large_scale_changes(
    affinities: torch.Tensor, scale_rate_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace affinities whose scale change exceeds the threshold by the identity, as AdaLAM does.

    Returns the affinities and the mask of those that pass. Rank-deficient (NaN) affinities fail the test but stay
    NaN, so that :func:`_finite_residuals` gives them no support instead of the identity's.
    """
    eigenvals = batch_2x2_eigenvals(affinities)
    valid = (eigenvals[..., 1] >= 1 / scale_rate_threshold**2) & (eigenvals[..., 0] <= scale_rate_threshold**2)
    reset = ~valid & ~eigenvals[..., 0].isnan()
    identity = torch.eye(2, device=affinities.device, dtype=affinities.dtype)
    return torch.where(reset[..., None, None], identity, affinities), valid


def _finite_residuals(y_pred: torch.Tensor, ysamples: torch.Tensor) -> torch.Tensor:
    """Residuals for the confidence sorter, which needs finite values.

    A rank-deficient (NaN) fit gets residual 2: outside both normalized unit neighborhoods, it cannot provide
    support at min_confidence > 1.
    """
    return torch.norm(y_pred - ysamples, dim=-1).nan_to_num(nan=2.0)


def _reprojection_residuals(
    affinities: torch.Tensor, xsamples: torch.Tensor, ysamples: torch.Tensor, ransidx: torch.Tensor
) -> torch.Tensor:
    """Compute finite reprojection residuals without materializing a ``(..., N, 2, 2)`` matrix tensor."""
    x0, x1 = xsamples[:, 0], xsamples[:, 1]
    dx = torch.addcmul(affinities[..., 0, 0][..., ransidx] * x0, affinities[..., 0, 1][..., ransidx], x1).sub_(
        ysamples[:, 0]
    )
    dy = torch.addcmul(affinities[..., 1, 0][..., ransidx] * x0, affinities[..., 1, 1][..., ransidx], x1).sub_(
        ysamples[:, 1]
    )
    return dx.square_().addcmul_(dy, dy).sqrt_().nan_to_num_(nan=2.0)


def ransac(
    xsamples: torch.Tensor,
    ysamples: torch.Tensor,
    rdims: torch.Tensor,
    config: Dict[str, Any],
    iters: int = 128,
    refit: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ransac."""
    DET_THR = config["detected_scale_rate_threshold"]
    dv: torch.device = config["device"]

    numransacs = rdims.shape[0]
    ransidx = torch.arange(numransacs, device=dv).repeat_interleave(rdims, output_size=xsamples.shape[0])
    idxoffsets = torch.cat([torch.tensor([0], device=dv), torch.cumsum(rdims[:-1], dim=0)], dim=0)
    distinct = _distinct_samples(xsamples, ysamples, ransidx)

    rand_samples_rel = draw_first_k_couples(iters, rdims, dv)
    rand_samples_abs = rand_samples_rel + idxoffsets
    sampled_x = torch.transpose(
        xsamples[rand_samples_abs], dim0=1, dim1=2
    )  # (niters, 2, numransacs, 2) -> (niters, numransacs, 2, 2)
    sampled_y = torch.transpose(ysamples[rand_samples_abs], dim0=1, dim1=2)

    # minimal fit for sampled_x @ A^T = sampled_y; rank-deficient pairs are NaN
    affinities_fit = _find_affine_linear_pair(sampled_x, sampled_y)
    valid_affinities = None
    if not refit:
        affinities_fit, valid_affinities = _reset_large_scale_changes(affinities_fit, DET_THR)

    residuals = _reprojection_residuals(affinities_fit, xsamples, ysamples, ransidx)  # (niters, numsamples)

    # When refit=True, allow a minimal hypothesis with at least min_inliers total distinct
    # correspondences (3 fitted + min_inliers - 3 independent) to propose inliers for
    # least-squares refitting; the refitted model must still satisfy the full min_inliers.
    proposal_config = dict(config, min_inliers=max(config["min_inliers"] - 3, 1)) if refit else config
    inl_ransidx, inl_sampleidx, inl_counts, inl_iters, inl_confidence = _select_inliers(
        residuals,
        xsamples,
        ysamples,
        ransidx,
        rdims,
        idxoffsets,
        affinities_fit,
        proposal_config,
        valid_affinities,
        distinct,
    )

    if len(inl_sampleidx) == 0:
        # If no inliers have been found, there is nothing to re-fit!
        refit = False

    if not refit:
        return (
            inl_sampleidx,
            affinities_fit[inl_iters, torch.arange(inl_iters.shape[0], device=dv)],
            inl_confidence,
            inl_counts,
        )

    # Organize inliers found into a matrix for efficient GPU re-fitting.
    # Cope with the irregular number of inliers per sample by padding with torch.zeros
    padded_inlier_x, padded_inlier_y = sample_padded_inliers(
        xsamples, ysamples, inl_counts, inl_ransidx, inl_sampleidx, numransacs, dv
    )

    # Keep translation anchored at the seed; zero padding contributes no evidence.
    refit_affinity = _find_affine_linear(padded_inlier_x, padded_inlier_y)
    refit_affinity, valid_refits = _reset_large_scale_changes(refit_affinity, DET_THR)
    residuals = _reprojection_residuals(refit_affinity, xsamples, ysamples, ransidx)

    inl_ransidx, inl_sampleidx, inl_counts, inl_iters, inl_confidence = _select_inliers(
        residuals.unsqueeze(0),
        xsamples,
        ysamples,
        ransidx,
        rdims,
        idxoffsets,
        refit_affinity.unsqueeze(0),
        config,
        valid_refits.unsqueeze(0),
        distinct,
    )
    return inl_sampleidx, refit_affinity, inl_confidence, inl_counts

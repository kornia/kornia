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

from .utils import arange_sequence, batch_2x2_ellipse, draw_first_k_couples, piecewise_arange


def stable_sort_residuals(residuals: torch.Tensor, ransidx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort residuals."""
    logres = torch.log(residuals + 1e-10)
    minlogres = torch.min(logres)
    maxlogres = torch.max(logres)

    span = (maxlogres - minlogres).clamp_min(torch.finfo(logres.dtype).eps)
    sorting_score = ransidx.unsqueeze(0).float() + 0.99 * (logres - minlogres) / span

    sorting_idxes = torch.argsort(sorting_score, dim=-1)  # (niters, numsamples)

    return residuals.gather(1, sorting_idxes), sorting_idxes


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


def confidence_based_inlier_selection(
    residuals: torch.Tensor,
    ransidx: torch.Tensor,
    rdims: torch.Tensor,
    idxoffsets: torch.Tensor,
    dv: torch.device,
    min_confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select inliers from confidence scores."""
    sorted_res, sorting_idxes = stable_sort_residuals(residuals, ransidx)
    return _confidence_from_sorted_residuals(sorted_res, sorting_idxes, ransidx, rdims, idxoffsets, dv, min_confidence)


def _confidence_from_sorted_residuals(
    sorted_res: torch.Tensor,
    sorting_idxes: torch.Tensor,
    ransidx: torch.Tensor,
    rdims: torch.Tensor,
    idxoffsets: torch.Tensor,
    dv: torch.device,
    min_confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the original confidence heuristic without sorting the residuals again."""
    numransacs = rdims.shape[0]
    numiters = sorted_res.shape[0]

    sorted_res_sqr = sorted_res**2

    too_perfect_fits = sorted_res_sqr <= 1e-8
    end_rans_indexing = torch.cumsum(rdims, dim=0) - 1

    _, inv_indices, res_dup_counts = torch.unique_consecutive(
        sorted_res_sqr.half().float(), dim=1, return_counts=True, return_inverse=True
    )

    duplicates_per_sample = res_dup_counts[inv_indices]
    inlier_weights = (1.0 / duplicates_per_sample).repeat(numiters, 1)
    inlier_weights[too_perfect_fits] = 0.0

    balanced_rdims, weights_cumsums = group_sum_and_cumsum(inlier_weights, end_rans_indexing, ransidx)
    if not isinstance(weights_cumsums, torch.Tensor):
        raise TypeError("Expected the `weights_cumsums` to be a torch.Tensor!")

    progressive_inl_rates = (
        weights_cumsums.float() / balanced_rdims.repeat_interleave(rdims, dim=1, output_size=ransidx.numel()).float()
    )

    good_inl_mask = (sorted_res_sqr * min_confidence <= progressive_inl_rates) | too_perfect_fits

    inlier_weights[~good_inl_mask] = 0.0
    inlier_counts_matrix, _ = group_sum_and_cumsum(inlier_weights, end_rans_indexing)

    inl_counts, inl_iters = torch.max(inlier_counts_matrix.long(), dim=0)

    relative_inl_idxes = arange_sequence(inl_counts)
    num_inliers = relative_inl_idxes.numel()
    inl_ransidx = torch.arange(numransacs, device=dv).repeat_interleave(inl_counts, output_size=num_inliers)
    inl_sampleidx = sorting_idxes[
        inl_iters.repeat_interleave(inl_counts, output_size=num_inliers), idxoffsets[inl_ransidx] + relative_inl_idxes
    ]
    highest_accepted_sqr_residuals = sorted_res_sqr[inl_iters, idxoffsets + inl_counts - 1]
    expected_extra_inl = (
        balanced_rdims[inl_iters, torch.arange(numransacs, device=dv)].float() * highest_accepted_sqr_residuals
    )
    return inl_ransidx, inl_sampleidx, inl_counts, inl_iters, inl_counts.float() / expected_extra_inl


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover well-supported near-exact correspondences missed by the original selection.

    Unlike the reference AdaLAM heuristic, equal tiny residuals need not be duplicate
    keypoints. Count distinct geometry instead, reserving three correspondences for
    the seed and minimal pair. They may be returned, but supply no independent evidence.
    """
    dv = residuals.device
    min_inliers, min_confidence = config["min_inliers"], config["min_confidence"]
    sorted_res, sorting_idxes = stable_sort_residuals(residuals, ransidx)
    # Ordinary noisy fits have only the seed and fitting pair near zero. Inspect one
    # already-sorted residual per hypothesis/neighbourhood before scanning all samples.
    # Use a loose gate because the log-based sort can tie close residuals in float32;
    # the recovery below still applies the exact 2**-12 threshold.
    # Skip the gate for very many groups, where adding the group ID to the sorting
    # score can lose too much residual precision even for this loose threshold.
    probe = idxoffsets + (rdims - 1).clamp(max=min_inliers + 2)
    if len(rdims) <= 4096 and not ((sorted_res[:, probe] <= 2.0**-11) & (rdims >= min_inliers + 3)).any():
        return _confidence_from_sorted_residuals(
            sorted_res, sorting_idxes, ransidx, rdims, idxoffsets, dv, min_confidence
        )
    # Cover both the "too perfect" cutoff and the first float16 residual bin used
    # by the original duplicate heuristic (2**-24 is the smallest float16 subnormal).
    residual_area = 2.0**-24
    near_exact = residuals <= 2.0**-12
    ends = rdims.cumsum(0) - 1
    raw_counts, _ = group_sum_and_cumsum(near_exact.float(), ends)
    raw_counts = raw_counts.long()
    candidates = raw_counts >= min_inliers + 3
    if not candidates.any():
        return _confidence_from_sorted_residuals(
            sorted_res, sorting_idxes, ransidx, rdims, idxoffsets, dv, min_confidence
        )

    if valid_affinities is None:
        eigenvals, _ = batch_2x2_ellipse(affinities)
        scale_limit = config["detected_scale_rate_threshold"] ** 2
        valid_affinities = (eigenvals[..., 1] >= 1 / scale_limit) & (eigenvals[..., 0] <= scale_limit)
    candidates &= valid_affinities

    # Use at most one occurrence of each source and target location in each neighbourhood.
    # Stable scalar sorts avoid torch.unique(dim=0), unavailable on older MPS versions.
    distinct = torch.ones_like(ransidx, dtype=torch.bool)
    for points in (xsamples, ysamples):
        order = torch.argsort(points[:, 1], stable=True)
        order = order[torch.argsort(points[order, 0], stable=True)]
        order = order[torch.argsort(ransidx[order], stable=True)]
        repeated = (ransidx[order[1:]] == ransidx[order[:-1]]) & (points[order[1:]] == points[order[:-1]]).all(dim=-1)
        distinct[order[1:][repeated]] = False

    support, _ = group_sum_and_cumsum((near_exact & distinct.unsqueeze(0)).float(), ends)
    support = support.long()
    support = (support - 3).clamp_min(0)
    support = torch.where(candidates, support, 0)
    best_support, best_iters = support.max(dim=0)
    best_counts = raw_counts[best_iters, torch.arange(len(rdims), device=dv)]
    # The fixed residual-area floor makes confidence finite even for a zero residual.
    # It is conservative: every recovered residual is bounded by this same area.
    exact_confidence = best_support.float() / (rdims.float() * residual_area)
    recover = (best_support >= min_inliers) & (exact_confidence >= min_confidence)
    # When every neighbourhood is fully explained, no larger consensus exists.
    # Avoid evaluating the residual-duplicate heuristic that rejects these exact fits.
    if (recover & (best_counts == rdims)).all():
        return ransidx, torch.arange(len(ransidx), device=dv), rdims, best_iters, exact_confidence

    inl_ransidx, inl_sampleidx, counts, iters, confidence = _confidence_from_sorted_residuals(
        sorted_res, sorting_idxes, ransidx, rdims, idxoffsets, dv, min_confidence
    )
    accepted = (confidence >= min_confidence) & (counts * (1 - 1 / confidence) >= min_inliers)
    recover &= ~accepted | (best_counts > counts)
    if not recover.any():
        return inl_ransidx, inl_sampleidx, counts, iters, confidence

    recovered_samples = torch.where(
        recover[ransidx] & near_exact[best_iters[ransidx], torch.arange(len(ransidx), device=dv)]
    )[0]
    keep = ~recover[inl_ransidx]
    inl_sampleidx = torch.cat([inl_sampleidx[keep], recovered_samples])
    inl_ransidx = torch.cat([inl_ransidx[keep], ransidx[recovered_samples]])
    order = torch.argsort(inl_ransidx, stable=True)
    counts = torch.where(recover, best_counts, counts)
    iters = torch.where(recover, best_iters, iters)
    confidence = torch.where(recover, exact_confidence, confidence)
    return inl_ransidx[order], inl_sampleidx[order], counts, iters, confidence


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

    padded_inlier_x[inl_ransidx, piecewise_arange(inl_ransidx)] = xsamples[inl_sampleidx]
    padded_inlier_y[inl_ransidx, piecewise_arange(inl_ransidx)] = ysamples[inl_sampleidx]

    return padded_inlier_x, padded_inlier_y


def _reset_large_scale_changes(
    affinities: torch.Tensor, scale_rate_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace affinities whose scale change exceeds the threshold by the identity, as AdaLAM does.

    Returns the affinities and the mask of those that pass. Rank-deficient (NaN) affinities fail the test but stay
    NaN, so that :func:`_finite_residuals` gives them no support instead of the identity's.
    """
    eigenvals, _eigenvecs = batch_2x2_ellipse(affinities)
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
    y_pred = (affinities_fit[:, ransidx] @ xsamples.unsqueeze(-1)).squeeze(-1)

    residuals = _finite_residuals(y_pred, ysamples)  # (niters, numsamples)

    inl_ransidx, inl_sampleidx, inl_counts, inl_iters, inl_confidence = _select_inliers(
        residuals, xsamples, ysamples, ransidx, rdims, idxoffsets, affinities_fit, config, valid_affinities
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
    y_pred = (refit_affinity[ransidx] @ xsamples.unsqueeze(-1)).squeeze(-1)

    residuals = _finite_residuals(y_pred, ysamples)

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
    )
    return inl_sampleidx, refit_affinity, inl_confidence, inl_counts

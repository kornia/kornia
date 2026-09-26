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

"""Module containing RANSAC modules."""

from __future__ import annotations

import math
import sys
from functools import lru_cache, partial
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.epipolar import find_essential, find_fundamental, project_to_essential, sampson_epipolar_distance
from kornia.geometry.homography import (
    find_homography_dlt,
    find_homography_dlt_iterated,
    find_homography_lines_dlt,
    find_homography_lines_dlt_iterated,
    line_segment_transfer_error_one_way,
    oneway_transfer_error,
    sample_is_valid_for_homography,
)

__all__ = ["RANSAC"]


@lru_cache(maxsize=32)
def _prosac_growth(sample_size: int, pop_size: int, budget: int) -> Tuple[int, ...]:
    """Cumulative draw counts T'_n (Chum & Matas, CVPR 2005, eqs. 3--4)."""
    expected = float(budget)
    for i in range(sample_size):
        expected *= (sample_size - i) / (pop_size - i)
    ends = [1]
    for n in range(sample_size + 1, pop_size + 1):
        next_expected = expected * n / (n - sample_size)
        ends.append(ends[-1] + max(1, math.ceil(next_expected - expected)))
        expected = next_expected
    return tuple(ends)


def _squared_line_distance(ls1: torch.Tensor, ls2: torch.Tensor, models: torch.Tensor) -> torch.Tensor:
    """Convert the line helper's algebraic residual into squared coordinate distance."""
    residual = line_segment_transfer_error_one_way(ls1, ls2, models)
    length = (ls2[..., 1, :] - ls2[..., 0, :]).norm(dim=-1)
    distance = residual / torch.where(length > 0, length, torch.ones_like(length))
    return torch.where(length > 0, distance.square(), torch.full_like(distance, float("inf")))


class RANSAC(nn.Module):
    """Module for robust geometry estimation with RANSAC. https://en.wikipedia.org/wiki/Random_sample_consensus.

    Convention:
        - ``kp1`` and ``kp2`` are passed as the first- and second-image arguments of the estimator selected by
          ``model_type``, so a homography maps ``kp1`` to ``kp2``. ``forward`` returns a ``(3, 3)`` model and an
          ``(N,)`` bool inlier mask, or an all-zero model and no inliers when no candidate has more inliers than
          its minimal sample (four correspondences for homographies, five for ``"essential"``, seven or eight
          for the fundamental models).
        - ``inl_th`` is in the keypoints' own units (pixels, or calibrated units for ``"essential"``): a
          correspondence is an inlier when its one-way transfer error for ``"homography"``, its Sampson distance
          for ``"fundamental"``, ``"fundamental_7pt"`` and ``"essential"``, or the mean distance of its transferred
          endpoints from the image-2 segment's line for ``"homography_from_linesegments"`` is at most ``inl_th``.
          :ref:`two-view-conventions` compares this with OpenCV.
        - ``score_type="msac"`` ranks candidates by ``sum(1 - min(e / inl_th**2, 1))`` over the squared errors
          ``e``; acceptance and early stopping count inliers for either score.
        - ``prosac_sampling=True`` expects correspondences sorted best-first and stops with PROSAC's
          termination-length test; ``confidence=1`` runs the whole ``batch_size * max_iter`` budget.
        - A seeded call uses a private generator and leaves torch's global RNG state unchanged; ``seed=None``
          draws from the global generator.
        - Known defects: for ``"homography_from_linesegments"``, local optimization weights segments by the
          length-scaled residual of :func:`~kornia.geometry.homography.line_segment_transfer_error_one_way`, which
          down-weights long segments (`#4867 <https://github.com/kornia/kornia/issues/4867>`_), and the endpoint
          pairing of :func:`~kornia.geometry.homography.find_homography_lines_dlt` applies
          (`#4866 <https://github.com/kornia/kornia/issues/4866>`_).

    Args:
        model_type: "homography", "fundamental", "fundamental_7pt", "essential", or
            "homography_from_linesegments".
        inl_th: positive inlier threshold, in the units given above.
        batch_size: number of generated samples at once.
        max_iter: maximum batches to generate, giving a budget of ``batch_size * max_iter`` minimal samples.
            The seven- and five-point solvers can return multiple models per sample.
        confidence: stopping confidence in ``(0, 1]``; 1 disables early stopping.
        max_lo_iters: maximum local refitting iterations; zero disables polishing.
        score_type: "ransac" for support count, or "msac" for truncated squared residuals.
        prosac_sampling: use PROSAC sampling on best-first ordered correspondences. The growth schedule
            advances per sampled set within each batch; stopping tests the incumbent's support within ranked
            prefixes (Chum and Matas, 2005, section 2.2) as well as within the whole set.
        seed: optional seed, reset on each call for reproducible estimation on the same device.
        lo_sample_size: optional inlier-subset size for a batch of ``max_lo_iters`` randomized local
            refits followed by a full-inlier refit. None uses iterative full-inlier refitting.

    """

    def __init__(
        self,
        model_type: str = "homography",
        inl_th: float = 2.0,
        batch_size: int = 2048,
        max_iter: int = 10,
        confidence: float = 0.99,
        max_lo_iters: int = 5,
        score_type: str = "ransac",
        prosac_sampling: bool = False,
        seed: Optional[int] = None,
        lo_sample_size: Optional[int] = None,
    ) -> None:
        """Initialize the RANSAC estimator.

        Args:
            model_type: type of model to estimate: "homography", "fundamental", "fundamental_7pt", "essential",
                "homography_from_linesegments".
            inl_th: inlier threshold; the class docstring gives its unit per ``model_type``.
            batch_size: number of generated samples at once.
            max_iter: maximum batches to generate. At most ``batch_size * max_iter`` minimal samples are drawn.
            confidence: desired confidence of the result, used for the early stopping. 1 runs the full budget.
            max_lo_iters: number of local optimization (polishing) iterations.
            score_type: scoring method to use: "ransac" or "msac".
            prosac_sampling: use PROSAC's progressive sampling schedule. Inputs must be sorted best-first
                by match quality. The schedule advances for every sampled set, including within batches.
                Stops when a ranked prefix, or the whole set, certifies the incumbent with ``confidence``.
            seed: optional random seed for reproducible results. If None, uses global random state.
            lo_sample_size: optional cap on the number of inliers used by each randomized local refit.
                Fits ``max_lo_iters`` independent subsets in one batch, followed by one full-inlier refit.
                None uses iterative full-inlier refitting. Subset refits must raise the score; a full-inlier
                refit may also tie it, since it is more precise than the minimal-sample model.

        """
        super().__init__()
        self.supported_models = [
            "homography",
            "fundamental",
            "fundamental_7pt",
            "homography_from_linesegments",
            "essential",
        ]
        self.supported_scores = ["msac", "ransac"]
        if score_type not in self.supported_scores:
            raise ValueError(f"Unsupported score type: {score_type}")
        if not math.isfinite(inl_th * inl_th) or inl_th <= 0 or inl_th * inl_th == 0:
            raise ValueError("inl_th and its square must be positive and finite")
        if batch_size <= 0 or max_iter <= 0 or max_lo_iters < 0:
            raise ValueError("batch_size and max_iter must be positive; max_lo_iters must be nonnegative")
        if not 0 < confidence <= 1:
            raise ValueError("confidence must lie in (0, 1]")
        self.score_type = score_type
        self.inl_th = inl_th
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model_type = model_type
        self.confidence = confidence
        self.max_lo_iters = max_lo_iters
        self.model_type = model_type
        self.prosac_sampling = prosac_sampling
        self.seed = seed
        self.lo_sample_size = lo_sample_size
        # The PROSAC growth schedule as a device tensor, reused across the batches of a call.
        self._prosac_ends: Optional[Tuple[Tuple[int, int, int, torch.device], torch.Tensor]] = None

        self.error_fn: Callable[..., torch.Tensor]
        self.minimal_solver: Callable[..., torch.Tensor]
        self.polisher_solver: Callable[..., torch.Tensor]

        if model_type == "homography":
            self.error_fn = oneway_transfer_error
            self.minimal_solver = find_homography_dlt
            # The polisher's Gaussian re-weighting uses the inlier threshold as its standard deviation.
            self.polisher_solver = partial(find_homography_dlt_iterated, soft_inl_th=inl_th)
            self.minimal_sample_size = 4
            self.polisher_sample_size = 4
        elif model_type == "homography_from_linesegments":
            self.error_fn = _squared_line_distance
            self.minimal_solver = find_homography_lines_dlt
            # Known defect: this IRLS polisher weights segments by the length-scaled residual of
            # line_segment_transfer_error_one_way, not by the distance used for scoring, so local
            # optimization down-weights long segments (https://github.com/kornia/kornia/issues/4867).
            self.polisher_solver = partial(find_homography_lines_dlt_iterated, soft_inl_th=inl_th)
            self.minimal_sample_size = 4
            self.polisher_sample_size = 4
        elif model_type == "fundamental":
            self.error_fn = sampson_epipolar_distance
            self.minimal_solver = find_fundamental
            self.minimal_sample_size = 8
            self.polisher_solver = find_fundamental
            self.polisher_sample_size = 8
        elif model_type == "fundamental_7pt":
            self.error_fn = sampson_epipolar_distance
            self.minimal_solver = partial(find_fundamental, method="7POINT")
            self.minimal_sample_size = 7
            self.polisher_solver = find_fundamental
            self.polisher_sample_size = 8
        elif model_type == "essential":
            self.error_fn = sampson_epipolar_distance
            self.minimal_solver = find_essential
            self.minimal_sample_size = 5
            self.polisher_solver = find_fundamental
            self.polisher_sample_size = 8
        else:
            raise NotImplementedError(f"{model_type} is unknown. Try one of {self.supported_models}")
        if lo_sample_size is not None and lo_sample_size < self.polisher_sample_size:
            raise ValueError(f"lo_sample_size must be at least {self.polisher_sample_size}")

    def sample(
        self,
        sample_size: int,
        pop_size: int,
        batch_size: int,
        iteration: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """Minimal sampler, but unlike traditional RANSAC we sample in batches.

        Yields the benefit of the parallel processing, esp. on GPU.

        Args:
            sample_size: number of samples to draw from the population.
            pop_size: size of the population to sample from.
            batch_size: number of sample sets to generate.
            iteration: zero-based batch index (used for PROSAC scheduling and the random seed).
            device: device to place the samples on.

        Returns:
            Tensor of sampled indices with shape :math:`(batch_size, sample_size)`.

        """
        device = torch.device("cpu") if device is None else torch.device(device)
        if not 0 < sample_size <= pop_size:
            raise ValueError("sample_size must be positive and no larger than pop_size")
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed + iteration)
        if device.type != "cpu" and not self.prosac_sampling:
            return (
                torch.rand(batch_size, pop_size, device=device, generator=generator)
                .topk(k=sample_size, dim=1, sorted=False)
                .indices
            )
        # The PROSAC schedule is indexed by sampled SETS, not outer batches or solver roots.
        # See https://cmp.felk.cvut.cz/~matas/papers/chum-prosac-cvpr05.pdf, eqs. 3--6.
        population = torch.full((batch_size,), pop_size, device=device, dtype=torch.long)
        force_newest = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if self.prosac_sampling:
            ends = self._prosac_schedule(sample_size, pop_size, device)
            draws = torch.arange(batch_size, device=device) + iteration * batch_size + 1
            population = (torch.searchsorted(ends, draws) + sample_size).clamp(max=pop_size)
            force_newest = draws <= ends[-1]

        if device.type == "cpu":
            # Floyd's sampling without replacement: O(B*m^2) comparisons and O(B*m)
            # storage, independent of N. Vectorize over the entire hypothesis batch.
            # Random-key topk is faster on CUDA, where these m small steps are launch-bound.
            rand = torch.rand(batch_size, sample_size, device=device, dtype=torch.float64, generator=generator)
            out = torch.empty(batch_size, sample_size, device=device, dtype=torch.long)
            for i in range(sample_size):
                last = population - sample_size + i
                candidate = (rand[:, i] * (last + 1)).long()
                if i > 0:
                    duplicate = (out[:, :i] == candidate[:, None]).any(dim=1)
                    candidate = torch.where(duplicate, last, candidate)
                if i == sample_size - 1:
                    candidate = torch.where(force_newest, last, candidate)
                out[:, i] = candidate
            return out

        rand = torch.rand(batch_size, pop_size, device=device, generator=generator)
        if self.prosac_sampling:
            rand.masked_fill_(torch.arange(pop_size, device=device)[None] >= population[:, None], -1.0)
            # Making the newest point largest forces its inclusion, leaving a uniform
            # (m-1)-subset of the preceding prefix. After growth, sample uniformly.
            newest = population[:, None] - 1
            rand.scatter_(1, newest, torch.where(force_newest[:, None], 2.0, rand.gather(1, newest)))
        return rand.topk(k=sample_size, dim=1, sorted=False).indices

    def _is_supported(self, num_inliers: float) -> bool:
        """Whether a model has support beyond the minimal sample it may have been fitted to.

        A minimal-sample model always fits its own sample, so only further inliers are evidence of a
        consensus; without them, input with no consensus returns the all-zero failure matrix.
        """
        return num_inliers > self.minimal_sample_size

    def _prosac_schedule(self, sample_size: int, pop_size: int, device: torch.device) -> torch.Tensor:
        """Return the cumulative PROSAC draw counts on ``device``, converting them once per configuration."""
        key = (sample_size, pop_size, self.batch_size * self.max_iter, device)
        if self._prosac_ends is None or self._prosac_ends[0] != key:
            self._prosac_ends = (key, torch.tensor(_prosac_growth(*key[:3]), device=device))
        return self._prosac_ends[1]

    def _prosac_max_samples(self, inliers: torch.Tensor, num_inliers: int) -> int:
        """PROSAC termination length test (Chum and Matas, CVPR 2005, section 2.2).

        A ranked prefix ``U_n`` certifies the incumbent when its support ``I_n`` there is non-random and
        ``k_n = log(1 - confidence) / log(1 - C(I_n, m) / C(n, m))`` draws fit inside the prefix's growth
        interval ``T'_n``, so that every counted draw was taken from ``U_n`` or a shorter prefix. The
        binomial tail of ``I_n - m`` accidental inliers with per-correspondence probability ``beta = 0.05`` is
        bounded by Chernoff's inequality at significance 0.05. As in OpenCV's USAC, prefixes shorter than
        ``min(N / 2, 100)`` correspondences or supporting fewer than 20% of all correspondences cannot
        terminate: a handful of top-ranked inliers to a model fitted to their neighbours is no evidence of a
        good model. The whole set is always a candidate, which is the uniform-sampling bound.
        """
        budget = self.batch_size * self.max_iter
        m, total = self.minimal_sample_size, inliers.numel()
        if self.confidence >= 1.0 or total <= m:
            return budget
        bound = min(budget, self.max_samples_by_conf(num_inliers, total, m, self.confidence))
        n_min = max(m + 1, min(total // 2, 100))
        if n_min > total:
            return bound
        # Independent of the keypoint dtype, including half precision. Double precision keeps a
        # near-integer bound from rounding down; MPS has no float64.
        dtype = torch.float32 if inliers.device.type == "mps" else torch.float64
        n = torch.arange(n_min, total + 1, device=inliers.device, dtype=dtype)
        support = inliers.cumsum(0)[n_min - 1 :].to(dtype)
        # Chernoff: P[Binomial(n - m, beta) >= I - m] <= exp(-(n - m) * D(q || beta)) for q > beta.
        beta = 0.05
        q = (support - m).clamp(min=0) / (n - m)
        divergence = torch.special.xlogy(q, q / beta) + torch.special.xlogy(1 - q, (1 - q) / (1 - beta))
        non_random = (q > beta) & ((n - m) * divergence > -math.log(0.05))
        enough = support >= 0.2 * total
        # Exact without-replacement probability of an all-inlier sample from the prefix, eq. 10.
        offsets = torch.arange(m, device=inliers.device, dtype=dtype)
        probability = ((support[:, None] - offsets).clamp(min=0) / (n[:, None] - offsets)).prod(1)
        required = torch.where(
            probability > 0,
            (math.log1p(-self.confidence) / torch.log1p(-probability)).ceil().clamp(min=1),
            torch.full_like(probability, float("inf")),
        )
        ends = self._prosac_schedule(m, total, inliers.device)[n_min - m :].to(dtype)
        eligible = non_random & enough & (required <= ends)
        candidate = torch.where(eligible, required, torch.full_like(required, float(budget))).min()
        return min(bound, int(candidate.item()))

    @staticmethod
    def max_samples_by_conf(n_inl: int, num_tc: int, sample_size: int, conf: float) -> int:
        """Update max_iter to stop iterations earlier https://en.wikipedia.org/wiki/Random_sample_consensus.

        Args:
            n_inl: number of inliers.
            num_tc: total number of correspondences.
            sample_size: size of minimal sample.
            conf: desired confidence level.

        Returns:
            Number of samples needed to achieve the desired confidence, rounded up.
            Returns ``sys.maxsize`` when no finite stopping bound is available.

        """
        if conf <= 0.0:
            return 1
        # conf >= 1 disables early stopping, even when every correspondence is an inlier.
        if conf >= 1.0 or num_tc < sample_size or n_inl < sample_size:
            return sys.maxsize
        if n_inl >= num_tc:
            return 1
        # Proper RANSAC formula for sampling without replacement
        # P(all samples are inliers) = (n_inl/num_tc) * ((n_inl-1)/(num_tc-1)) * ...
        # ... * ((n_inl-sample_size+1)/(num_tc-sample_size+1))
        prob_inlier = 1.0
        for i in range(sample_size):
            prob_inlier *= (n_inl - i) / (num_tc - i)

        if prob_inlier == 0.0:
            return sys.maxsize
        return min(sys.maxsize, math.ceil(math.log1p(-conf) / math.log1p(-prob_inlier)))

    def estimate_model_from_minsample(self, kp1: torch.Tensor, kp2: torch.Tensor) -> torch.Tensor:
        """Estimate models from minimal samples.

        Args:
            kp1: source keypoints with shape :math:`(batch_size, sample_size, 2)`.
            kp2: target keypoints with shape :math:`(batch_size, sample_size, 2)`.

        Returns:
            Estimated models tensor.

        """
        batch_size, sample_size = kp1.shape[:2]
        return self.minimal_solver(kp1, kp2, torch.ones(batch_size, sample_size, dtype=kp1.dtype, device=kp1.device))

    def verify(
        self, kp1: torch.Tensor, kp2: torch.Tensor, models: torch.Tensor, inl_th: float
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """Verify models by computing inliers and selecting the best model.

        Args:
            kp1: source keypoints.
            kp2: target keypoints.
            models: candidate models to verify.
            inl_th: positive squared inlier threshold.

        Returns:
            Tuple containing:
                - Best model
                - Inlier mask for the best model
                - Score of the best model
                - Number of inliers (distinct from the MSAC score)

        """
        if not math.isfinite(inl_th) or inl_th <= 0:
            raise ValueError("The squared inlier threshold must be positive and finite")
        if len(kp1.shape) == 2:
            kp1 = kp1[None]
        if len(kp2.shape) == 2:
            kp2 = kp2[None]
        batch_size = models.shape[0]
        if self.model_type == "homography_from_linesegments":
            errors = self.error_fn(kp1.expand(batch_size, -1, 2, 2), kp2.expand(batch_size, -1, 2, 2), models)
        else:
            # The point metrics broadcast over models. Expanding points first makes
            # homogeneous conversion allocate B copies of identical coordinates.
            errors = self.error_fn(kp1, kp2, models)
        # Non-finite residuals must not poison the reduction or win argmax. One kernel: this runs for
        # every batch and every LO step, where accelerator launches dominate the cost.
        inf = float("inf")
        errors = errors.nan_to_num(nan=inf, posinf=inf, neginf=inf)
        inl_mask = errors <= inl_th
        score_ransac = inl_mask.sum(dim=1)
        if self.score_type == "msac":
            # Equivalent to minimizing the truncated squared loss (MSAC), normalized
            # to [0, N]. This is a quality score, NOT an inlier count. Accumulate in at
            # least float32: a half-precision total rounds to steps of 2-4 in the hundreds.
            score = (1.0 - errors.clamp(min=0.0, max=inl_th) / inl_th).sum(
                dim=1, dtype=torch.promote_types(errors.dtype, torch.float32)
            )
            # A high-quality but under-supported model must not hide a viable candidate
            # elsewhere in the same batch. An all-invalid batch is rejected by forward.
            # The same support rule as _is_supported.
            score = score.masked_fill(score_ransac <= self.minimal_sample_size, -1)
        elif self.score_type == "ransac":
            # The score is the support, so argmax already prefers any sufficiently supported
            # candidate; forward rejects an insufficient best support.
            score = score_ransac
        else:
            raise ValueError(f"Unsupported score type: {self.score_type}")
        best_model_idx = score.argmax()
        best_model_score = score[best_model_idx].item()
        num_inliers = score_ransac[best_model_idx].item()
        model_best = models[best_model_idx].clone()
        inliers_best = inl_mask[best_model_idx]
        return model_best, inliers_best, best_model_score, num_inliers

    def remove_bad_samples(self, kp1: torch.Tensor, kp2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove degenerate samples based on model-specific constraints.

        Args:
            kp1: source keypoints.
            kp2: target keypoints.

        Returns:
            Tuple of filtered keypoints (kp1, kp2).

        """
        # ToDo: add (model-specific) verification of the samples,
        # E.g. constraints on not to be a degenerate sample
        if self.model_type == "homography":
            mask = sample_is_valid_for_homography(kp1, kp2)
            return kp1[mask], kp2[mask]
        return kp1, kp2

    def remove_bad_models(self, models: torch.Tensor) -> torch.Tensor:
        """Remove degenerate models based on simple heuristics.

        Args:
            models: candidate models to filter.

        Returns:
            Filtered models tensor.

        """
        # Filter out NaN or Inf models
        mask = torch.isfinite(models).all(dim=-1).all(dim=-1) & (models.abs().amax(dim=(-2, -1)) > 0)
        return models[mask]

    def polish_model(self, kp1: torch.Tensor, kp2: torch.Tensor, inliers: torch.Tensor) -> torch.Tensor:
        """Polish the model using inliers through local optimization.

        Args:
            kp1: source keypoints.
            kp2: target keypoints.
            inliers: boolean mask indicating inlier correspondences.

        Returns:
            Polished model tensor.

        """
        # TODO: Replace this with MAGSAC++ polisher
        kp1_inl = kp1[inliers][None]
        kp2_inl = kp2[inliers][None]
        num_inl = kp1_inl.size(1)
        model = self.polisher_solver(
            kp1_inl, kp2_inl, torch.ones(1, num_inl, dtype=kp1_inl.dtype, device=kp1_inl.device)
        )
        # The polisher fits a fundamental matrix via the 8-point DLT, which does not enforce the
        # essential-matrix constraint (two equal non-zero singular values and a zero singular value).
        # Project it back onto the essential manifold so that downstream decompose_essential_matrix /
        # motion_from_essential do not silently fail.
        # See https://github.com/kornia/kornia/issues/3874
        return self._project_refits(model)

    def _project_refits(self, models: torch.Tensor) -> torch.Tensor:
        """Project essential refits onto the manifold, dropping invalid ones before the projection's SVD."""
        if self.model_type != "essential":
            return models
        models = self.remove_bad_models(models)
        return project_to_essential(models) if len(models) > 0 else models

    def _subset_refits(
        self, kp1: torch.Tensor, kp2: torch.Tensor, inliers: torch.Tensor, lo_sample_size: int, iteration: int
    ) -> torch.Tensor:
        """Fit ``max_lo_iters`` random ``lo_sample_size``-subsets of the inliers in one solver batch.

        Randomized non-minimal refits let LO escape a bad consensus. Bounded subsets follow Lebeda et al.,
        BMVC 2012 (LO+), but this is not the full LO+ algorithm with threshold scheduling.
        """
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=kp1.device)
            generator.manual_seed(self.seed + self.max_iter + iteration)
        indices = inliers.nonzero().flatten()
        # Independent refits share the current consensus and run in one
        # solver batch, rather than max_lo_iters tiny accelerator calls.
        subset = torch.rand(self.max_lo_iters, len(indices), device=kp1.device, generator=generator)
        selected = indices[subset.topk(lo_sample_size, dim=1).indices]
        models = self.polisher_solver(kp1[selected], kp2[selected], torch.ones_like(selected, dtype=kp1.dtype))
        return self._project_refits(models)

    def _local_optimization(
        self,
        kp1: torch.Tensor,
        kp2: torch.Tensor,
        model: torch.Tensor,
        inliers: torch.Tensor,
        model_score: float,
        num_inliers: float,
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float, bool]:
        """Refit a newly accepted model from its inliers.

        Returns:
            The refined model, its inlier mask, score and support, and whether the model is still the
            minimal solver's (no refit replaced it).

        """
        from_minimal_solver = True
        lo_sample_size = self.lo_sample_size
        bounded_lo = lo_sample_size is not None and num_inliers > lo_sample_size and self.max_lo_iters > 0
        for lo_iteration in range(2 if bounded_lo else self.max_lo_iters):
            if num_inliers < self.polisher_sample_size:
                break
            use_subset = bounded_lo and lo_iteration == 0
            if use_subset and lo_sample_size is not None:
                model_lo = self._subset_refits(kp1, kp2, inliers, lo_sample_size, iteration)
            else:
                model_lo = self.polish_model(kp1, kp2, inliers)
            if model_lo is not None and len(model_lo) > 0:
                model_lo = self.remove_bad_models(model_lo)
            if model_lo is None or len(model_lo) == 0:
                # A failed subset batch still leaves the full refit. A failed full refit would
                # only repeat itself, since the inliers it was fitted to have not changed.
                if use_subset:
                    continue
                break
            model_lo_best, inliers_lo, score_lo, num_inliers_lo = self.verify(kp1, kp2, model_lo, self.inl_th**2)
            improved = score_lo > model_score
            # A full-inlier least-squares refit that keeps the score is still more precise than
            # the minimal-sample model; RANSAC scoring ties whenever support does not grow.
            tied_refit = score_lo == model_score and not use_subset
            if (improved or tied_refit) and self._is_supported(num_inliers_lo):
                model = model_lo_best
                inliers = inliers_lo.clone()
                model_score = score_lo
                num_inliers = num_inliers_lo
                from_minimal_solver = False
            if not improved and not use_subset:
                break
        return model, inliers, model_score, num_inliers, from_minimal_solver

    def validate_inputs(self, kp1: torch.Tensor, kp2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> None:
        """Validate input tensors for shape and size requirements.

        Args:
            kp1: source keypoints.
            kp2: target keypoints.
            weights: optional correspondence weights (not used currently).

        Raises:
            ValueError: if ``kp1`` and ``kp2`` differ in length or hold fewer correspondences than the minimal
                sample.
            ShapeError: if the keypoint shape is wrong.

        """
        if self.model_type != "homography_from_linesegments":
            KORNIA_CHECK_SHAPE(kp1, ["N", "2"])
            KORNIA_CHECK_SHAPE(kp2, ["N", "2"])
            if not (kp1.shape[0] == kp2.shape[0]) or (kp1.shape[0] < self.minimal_sample_size):
                raise ValueError(
                    "kp1 and kp2 should be                                  equal shape at least"
                    f" [{self.minimal_sample_size}, 2],                                  got {kp1.shape}, {kp2.shape}"
                )
        if self.model_type == "homography_from_linesegments":
            KORNIA_CHECK_SHAPE(kp1, ["N", "2", "2"])
            KORNIA_CHECK_SHAPE(kp2, ["N", "2", "2"])
            if not (kp1.shape[0] == kp2.shape[0]) or (kp1.shape[0] < self.minimal_sample_size):
                raise ValueError(
                    "kp1 and kp2 should be                                  equal shape at least"
                    f" [{self.minimal_sample_size}, 2, 2],                                  got {kp1.shape},"
                    f" {kp2.shape}"
                )

    def forward(
        self, kp1: torch.Tensor, kp2: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Call main forward method to execute the RANSAC algorithm.

        Args:
            kp1: source image keypoints :math:`(N, 2)` (or line segments :math:`(N, 2, 2)`).
            kp2: target image keypoints with the same shape. For PROSAC, both inputs must be sorted
                best-first using the same correspondence-quality ordering. Essential estimation uses
                camera-normalized coordinates and a threshold in those units.
            weights: optional correspondences weights. Not used now.

        Returns:
            - Estimated model, shape of :math:`(3, 3)`, or zeros if no valid model is found. A model needs
              more inliers than its minimal sample, since a model fitted to a sample fits that sample.
            - Boolean inlier mask, shape of :math:`(N,)`, in the supplied correspondence order.

        """
        self.validate_inputs(kp1, kp2, weights)
        best_score_total = -float("inf")
        max_samples = self.max_iter * self.batch_size
        num_tc: int = len(kp1)
        best_model_total = torch.zeros(3, 3, dtype=kp1.dtype, device=kp1.device)
        inliers_best_total: torch.Tensor = torch.zeros(num_tc, device=kp1.device, dtype=torch.bool)
        # Only a minimal-solver model needs projecting onto the essential manifold; LO refits already are.
        best_needs_projection = False
        for i in range(self.max_iter):
            if i * self.batch_size >= max_samples:
                break
            # Sample minimal samples in batch to estimate models
            idxs = self.sample(self.minimal_sample_size, num_tc, self.batch_size, i, kp1.device)
            kp1_sampled = kp1[idxs]
            kp2_sampled = kp2[idxs]
            kp1_sampled, kp2_sampled = self.remove_bad_samples(kp1_sampled, kp2_sampled)
            if len(kp1_sampled) == 0:
                continue
            # Estimate models
            models = self.estimate_model_from_minsample(kp1_sampled, kp2_sampled)
            if self.model_type in ["essential", "fundamental_7pt"]:
                models = models.reshape(-1, 3, 3)
            models = self.remove_bad_models(models)
            if (models is None) or (len(models) == 0):
                continue
            # Score the models and select the best one
            model, inliers, model_score, num_inliers = self.verify(kp1, kp2, models, self.inl_th**2)
            # Store far-the-best model and (optionally) do a local optimization
            if (model_score > best_score_total) and self._is_supported(num_inliers):
                model, inliers, model_score, num_inliers, from_minimal_solver = self._local_optimization(
                    kp1, kp2, model, inliers, model_score, num_inliers, i
                )
                # Now storing the best model
                best_model_total = model.clone()
                inliers_best_total = inliers.clone()
                best_score_total = model_score
                best_needs_projection = from_minimal_solver

                # Should we already stop?
                # The score may be MSAC; confidence depends on support, and counts
                # sampled sets, not the number of roots returned by a minimal solver.
                # The bound follows the incumbent's own support, as in OpenCV's USAC: under
                # MSAC a better-scoring model can have less support than the one it replaced.
                # PROSAC also tests ranked prefixes, which is where its speed comes from.
                if self.prosac_sampling:
                    max_samples = self._prosac_max_samples(inliers, int(num_inliers))
                else:
                    max_samples = min(
                        self.max_iter * self.batch_size,
                        self.max_samples_by_conf(int(num_inliers), num_tc, self.minimal_sample_size, self.confidence),
                    )
        # The best model may come from the 5-point minimal solver (find_essential), which is not
        # guaranteed to return a matrix on the essential manifold. Project the returned model once
        # instead of projecting every candidate inside the loop, so that model selection is unaffected.
        # See https://github.com/kornia/kornia/issues/3874
        if self.model_type == "essential" and best_needs_projection:
            best_model_total = project_to_essential(best_model_total[None])[0]
            _, inliers_best_total, _, support = self.verify(kp1, kp2, best_model_total[None], self.inl_th**2)
            # Projection moves the residuals; a model that loses its support is no model.
            if not self._is_supported(support):
                best_model_total = torch.zeros_like(best_model_total)
                inliers_best_total = torch.zeros_like(inliers_best_total)
        return best_model_total, inliers_best_total

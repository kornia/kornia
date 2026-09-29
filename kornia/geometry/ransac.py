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
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry._degensac import (
    _h_degenerate_sample,
    _h_degenerate_samples,
    _inner_homography,
    _inside_plane,
    _plane_parallax_search,
    _repeats_plane,
)
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import find_essential, find_fundamental, project_to_essential, sampson_epipolar_distance
from kornia.geometry.epipolar._metrics import _sampson_from_quadratic_basis, _sampson_quadratic_basis
from kornia.geometry.epipolar.essential import _five_point_candidates, _refine_essential_lm
from kornia.geometry.epipolar.fundamental import (
    _eight_point_fundamental,
    _epipolar_design_rows,
    _refine_fundamental_lm,
    _seven_point_candidates,
    normalize_points,
    normalize_transformation,
)
from kornia.geometry.homography import (
    _four_point_homography,
    _line_segment_squared_distance_one_way,
    _refine_homography_lm,
    _transfer_basis,
    _transfer_from_basis,
    find_homography_dlt,
    find_homography_dlt_iterated,
    find_homography_lines_dlt,
    find_homography_lines_dlt_iterated,
    oneway_transfer_error,
    sample_is_valid_for_homography,
    sampson_homography_distance,
)

__all__ = ["RANSAC"]

# The batch size that the ``max_iter`` budget counts in when ``batch_size="auto"``.
_DEFAULT_BATCH = 2048

# Model types that local_optimization="lm" supports, and how many minimal models it refines.
_LM_MODELS = ("homography", "fundamental", "fundamental_7pt", "fundamental_8pt", "essential")
_LM_CANDIDATES = 8
# Model types that DEGENSAC supports: seven-point fundamental matrices (Chum, Werner and Matas, CVPR 2005).
_DEGENSAC_MODELS = ("fundamental", "fundamental_7pt")


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


def _resolve_degensac(degensac: Optional[bool], model_type: str, local_optimization: str) -> bool:
    """The ``degensac`` setting of :class:`RANSAC`: None leaves it off; True requires support."""
    if degensac is not None and not isinstance(degensac, bool):
        raise ValueError(f"degensac must be None, True or False, got {degensac!r}")
    supported = model_type in _DEGENSAC_MODELS and local_optimization == "lm"
    if degensac and not supported:
        raise ValueError(
            'degensac=True requires model_type "fundamental" or "fundamental_7pt" with local_optimization="lm"'
        )
    return False if degensac is None else degensac


def _normalize_correspondences(
    kp1: torch.Tensor, kp2: torch.Tensor, shared_scale: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    r"""Hartley-normalize both images' correspondences with :func:`normalize_points`, once per RANSAC call.

    Statistics use the correspondences finite in both images: the others enter :func:`normalize_points` as zero-weight
    placeholders, since a weight of 0 does not remove a NaN from its weighted sums, and come out as NaN in both images,
    so they are never counted as inliers. ``shared_scale`` gives both images the scale of the mean of their two mean
    radii, which keeps the Sampson distance a multiple of the pixel one: :func:`normalize_points` scales by
    :math:`\sqrt{2} / (r + \epsilon)`, so that scale is the harmonic mean of the two.

    Returns:
        Homogeneous normalized points ``(N, 3)`` of each image, the ``(3, 3)`` transforms that map pixels to them, and
        the two scales in pixels per normalized unit.
    """
    finite = torch.isfinite(kp1).all(1) & torch.isfinite(kp2).all(1)
    stacked = torch.stack([kp1, kp2])
    placeholders = torch.where(finite[None, :, None], stacked, torch.zeros_like(stacked))
    points, transforms = normalize_points(placeholders, weights=finite.to(kp1.dtype).expand(2, -1))
    if shared_scale:
        scale = transforms[:, 0, 0]
        ratio = (2.0 / (1.0 / scale).sum()) / scale
        points = points * ratio[:, None, None]
        transforms = torch.cat([transforms[:, :2] * ratio[:, None, None], transforms[:, 2:]], 1)
    points = torch.where(finite[None, :, None], points, torch.full_like(points, float("nan")))
    points = convert_points_to_homogeneous(points)
    s1, s2 = (1.0 / transforms[:, 0, 0]).tolist()
    return points[0], points[1], transforms[0], transforms[1], s1, s2


class RANSAC(nn.Module):
    """Module for robust geometry estimation with RANSAC. https://en.wikipedia.org/wiki/Random_sample_consensus.

    Convention:
        - ``kp1`` and ``kp2`` are passed as the first- and second-image arguments of the estimator selected by
          ``model_type``, so a homography maps ``kp1`` to ``kp2``. ``forward`` returns a ``(3, 3)`` model and an
          ``(N,)`` bool inlier mask, or an all-zero model and no inliers when no candidate has more inliers than
          its minimal sample (four correspondences for homographies, five for ``"essential"``, seven for
          ``"fundamental"`` and ``"fundamental_7pt"``, eight for ``"fundamental_8pt"``).
        - ``"fundamental"`` draws seven-point samples, like ``"fundamental_7pt"``; ``"fundamental_8pt"`` draws
          eight-point ones.
        - ``inl_th`` is in the keypoints' own units (pixels, or calibrated units for ``"essential"``): a
          correspondence is an inlier when its one-way transfer error for ``"homography"``, its Sampson distance
          for the fundamental models and ``"essential"``, or the mean distance of its transferred endpoints from
          the image-2 segment's line for ``"homography_from_linesegments"`` is at most ``inl_th``.
          :ref:`two-view-conventions` compares this with OpenCV.
        - ``score_type="msac"`` (the default) ranks candidates by ``sum(1 - min(e / inl_th**2, 1))`` over the
          squared errors ``e``; acceptance and early stopping count inliers for either score.
        - ``local_optimization="lm"``, the default for homographies, fundamental and essential matrices, keeps the
          eight best-scoring minimal models of the whole run. After sampling, it refines them together with
          ``max_lo_iters`` Levenberg-Marquardt iterations on all correspondences with the squared error truncated
          at ``inl_th``, keeps the best-scoring refit or minimal model, and refines that one on its inliers with
          ``refine_iters`` iterations of a Cauchy loss of scale ``inl_th / 3``. The returned mask holds the inliers
          of the returned model after conversion to the input dtype; if that conversion loses sufficient support,
          the call returns the all-zero failure result. Early stopping follows the support of the best minimal model.
          The refinements run on the CPU in float64 whatever the device of the correspondences: they are a few dozen
          small operations per iteration, which launch latency makes slower on an accelerator. Homographies and
          fundamental matrices are estimated on Hartley-normalized correspondences; essential matrices in the
          caller's calibrated coordinates, which a normalization would take off the essential manifold, refined on
          that manifold and returned with unit Frobenius norm.
        - ``local_optimization="dlt"``, the only choice for ``"homography_from_linesegments"``, refits each new best
          model from its inliers: with the default ``lo_sample_size=32``, ``max_lo_iters`` randomized refits on
          32-inlier subsets followed by one full-inlier refit; ``lo_sample_size=None`` refits all inliers
          iteratively.
        - ``degensac=True``, an opt-in for ``"fundamental"`` and ``"fundamental_7pt"`` with
          ``local_optimization="lm"``, runs DEGENSAC (Chum, Werner and Matas, CVPR 2005). Every seven-point model that
          sets a new record among the raw scores is tested for an H-degenerate sample: five or more of its seven
          correspondences related by one homography. Such a model fits the dominant plane and whatever happens to
          agree with it off the plane, so the homography is refined, and fundamental matrices are drawn from it and
          pairs of correspondences off the plane (plane and parallax). The best of them, refined, joins the
          eight-model pool and, when it outscores the incumbent, sets the stopping bound. Thresholds and iteration
          counts follow Chum's implementation in pydegensac; the draws come from a private host generator. A plane
          already searched in the call is not searched again: a sample whose homography's inliers lie 95% or more
          inside it is skipped before refinement, and a refined plane whose inliers match it (Jaccard index 0.95 or
          more) before the search. The default, ``degensac=False``, keeps the plain seven-point loop.
          Explicit recovery reproduces that result exactly when no record-setting sample is degenerate.
          Chum's tolerance, three times the squared threshold for five of the seven correspondences, also flags
          samples in many scenes without a dominant plane; there the recovered models only join the competition.
        - ``prosac_sampling=True`` expects correspondences sorted best-first and stops with PROSAC's
          termination-length test; ``confidence=1`` runs the whole ``batch_size * max_iter`` budget.
        - A seeded call uses a private generator and leaves torch's global RNG state unchanged; ``seed=None``
          draws the minimal samples from the global generator. DEGENSAC's recovery draws always come from a private
          host generator, which an unseeded call seeds from the sample that triggers the recovery.

    Args:
        model_type: "homography", "fundamental", "fundamental_7pt", "fundamental_8pt", "essential", or
            "homography_from_linesegments".
        inl_th: positive inlier threshold, in the units given above.
        batch_size: hypotheses generated and verified at once, or ``"auto"`` to pick the batches per call from
            the device and the number of correspondences (see :meth:`resolve_batch_size`).
        max_iter: with an integer ``batch_size``, the maximum number of batches, for a budget of
            ``batch_size * max_iter`` minimal samples; with ``"auto"``, the budget is ``2048 * max_iter``. The
            seven- and five-point solvers can return multiple models per sample.
        confidence: stopping confidence in ``(0, 1]``; 1 disables early stopping.
        max_lo_iters: local optimization iterations: Levenberg-Marquardt iterations with
            ``local_optimization="lm"``, refits with ``"dlt"``; zero disables local optimization.
        score_type: "msac" (default) for truncated squared residuals, or "ransac" for support count.
        prosac_sampling: use PROSAC sampling on best-first ordered correspondences. The growth schedule
            advances per sampled set within each batch; stopping tests the incumbent's support within ranked
            prefixes (Chum and Matas, 2005, section 2.2) as well as within the whole set. It pays off when the
            ranking tracks inlier-ness and the inlier ratio is low (ratio-tested SIFT). On learned matches with
            85% or more inliers it can certify a model fitted to a spatially clustered top-ranked prefix after
            one batch and score below uniform sampling; use uniform sampling or ``confidence=1`` there.
        seed: optional seed, reset on each call for reproducible estimation on the same device.
        lo_sample_size: with ``local_optimization="dlt"``, the inlier-subset size for a batch of ``max_lo_iters``
            randomized local refits followed by a full-inlier refit (default 32). None uses iterative full-inlier
            refitting. Unused with ``"lm"``.
        max_samples: optional budget of minimal samples that overrides the one implied by ``batch_size`` and
            ``max_iter``; the last batch is truncated to it.
        local_optimization: ``"lm"`` or ``"dlt"``, as described above; None picks ``"lm"`` for homographies,
            fundamental and essential matrices and ``"dlt"`` for line segments.
        refine_iters: Levenberg-Marquardt iterations of the final refinement with ``local_optimization="lm"``;
            zero disables it.
        degensac: run DEGENSAC's dominant-plane recovery, as described above. Defaults to False; None also leaves
            it off. True enables it for ``"fundamental"`` and ``"fundamental_7pt"`` with ``local_optimization="lm"``,
            the only supported combinations; True with any other raises ``ValueError``.

    """

    def __init__(
        self,
        model_type: str = "homography",
        inl_th: float = 2.0,
        batch_size: Union[int, str] = "auto",
        max_iter: int = 10,
        confidence: float = 0.99,
        max_lo_iters: int = 5,
        score_type: str = "msac",
        prosac_sampling: bool = False,
        seed: Optional[int] = None,
        lo_sample_size: Optional[int] = 32,
        max_samples: Optional[int] = None,
        local_optimization: Optional[str] = None,
        refine_iters: int = 3,
        degensac: Optional[bool] = False,
    ) -> None:
        """Initialize the RANSAC estimator.

        Args:
            model_type: type of model to estimate: "homography", "fundamental", "fundamental_7pt",
                "fundamental_8pt", "essential", "homography_from_linesegments".
            inl_th: inlier threshold; the class docstring gives its unit per ``model_type``.
            batch_size: number of generated samples at once, or ``"auto"`` (see :meth:`resolve_batch_size`).
            max_iter: maximum batches to generate. At most ``batch_size * max_iter`` minimal samples are drawn
                (``2048 * max_iter`` with ``batch_size="auto"``) unless ``max_samples`` is given.
            confidence: desired confidence of the result, used for the early stopping. 1 runs the full budget.
            max_lo_iters: number of local optimization iterations.
            score_type: scoring method to use: "msac" (default) or "ransac".
            prosac_sampling: use PROSAC's progressive sampling schedule. Inputs must be sorted best-first
                by match quality. The schedule advances for every sampled set, including within batches.
                Stops when a ranked prefix, or the whole set, certifies the incumbent with ``confidence``.
            seed: optional random seed for reproducible results. If None, uses global random state.
            lo_sample_size: with ``local_optimization="dlt"``, the cap on the number of inliers used by each
                randomized local refit (default 32). Fits ``max_lo_iters`` independent subsets in one batch,
                followed by one full-inlier refit. None uses iterative full-inlier refitting. Subset refits must
                raise the score; a full-inlier refit may also tie it, since it is more precise than the
                minimal-sample model.
            max_samples: optional budget of minimal samples, overriding ``batch_size * max_iter``.
            local_optimization: ``"lm"`` (batched Levenberg-Marquardt refinement of the best minimal models and a
                final robust refinement) or ``"dlt"`` (refits of each new best model); None picks ``"lm"`` where it
                is supported, for homographies, fundamental and essential matrices.
            refine_iters: Levenberg-Marquardt iterations of the final refinement on the inliers with ``"lm"``.
            degensac: opt-in DEGENSAC dominant-plane recovery for seven-point fundamental matrices.
                Defaults to False; None also leaves it off. True requires seven-point matrices with
                local_optimization="lm".

        """
        super().__init__()
        self.supported_models = [
            "homography",
            "fundamental",
            "fundamental_7pt",
            "fundamental_8pt",
            "homography_from_linesegments",
            "essential",
        ]
        self.supported_scores = ["msac", "ransac"]
        if score_type not in self.supported_scores:
            raise ValueError(f"Unsupported score type: {score_type}")
        if not math.isfinite(inl_th * inl_th) or inl_th <= 0 or inl_th * inl_th == 0:
            raise ValueError("inl_th and its square must be positive and finite")
        if isinstance(batch_size, str):
            if batch_size != "auto":
                raise ValueError('batch_size must be a positive integer or "auto"')
        elif isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer or "auto"')
        if max_iter <= 0 or max_lo_iters < 0:
            raise ValueError("max_iter must be positive; max_lo_iters must be nonnegative")
        if max_samples is not None and (
            isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0
        ):
            raise ValueError("max_samples must be a positive integer")
        if not 0 < confidence <= 1:
            raise ValueError("confidence must lie in (0, 1]")
        if isinstance(refine_iters, bool) or not isinstance(refine_iters, int) or refine_iters < 0:
            raise ValueError("refine_iters must be a nonnegative integer")
        if local_optimization is None:
            local_optimization = "lm" if model_type in _LM_MODELS else "dlt"
        if local_optimization not in ("lm", "dlt"):
            raise ValueError(f'local_optimization must be "lm" or "dlt", got {local_optimization!r}')
        if local_optimization == "lm" and model_type not in _LM_MODELS:
            raise ValueError(f'local_optimization="lm" supports {", ".join(_LM_MODELS)}, not {model_type!r}')
        degensac = _resolve_degensac(degensac, model_type, local_optimization)
        self.score_type = score_type
        self.inl_th = inl_th
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.model_type = model_type
        self.confidence = confidence
        self.max_lo_iters = max_lo_iters
        self.model_type = model_type
        self.prosac_sampling = prosac_sampling
        self.seed = seed
        self.lo_sample_size = lo_sample_size
        self.local_optimization = local_optimization
        self.refine_iters = refine_iters
        self.degensac = degensac
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
            self.error_fn = _line_segment_squared_distance_one_way
            self.minimal_solver = find_homography_lines_dlt
            # The polisher re-weights by the same perpendicular pixel distance the score uses.
            self.polisher_solver = partial(find_homography_lines_dlt_iterated, soft_inl_th=inl_th)
            self.minimal_sample_size = 4
            self.polisher_sample_size = 4
        elif model_type == "fundamental_8pt":
            self.error_fn = sampson_epipolar_distance
            self.minimal_solver = find_fundamental
            self.minimal_sample_size = 8
            self.polisher_solver = find_fundamental
            self.polisher_sample_size = 8
        elif model_type in ("fundamental", "fundamental_7pt"):
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
        offset: Optional[int] = None,
    ) -> torch.Tensor:
        """Minimal sampler, but unlike traditional RANSAC we sample in batches.

        Yields the benefit of the parallel processing, esp. on GPU.

        Args:
            sample_size: number of samples to draw from the population.
            pop_size: size of the population to sample from.
            batch_size: number of sample sets to generate.
            iteration: zero-based batch index (used for the random seed and, without ``offset``, for PROSAC
                scheduling).
            device: device to place the samples on.
            offset: number of sets drawn by the previous batches, for PROSAC scheduling when batches differ in
                size; None assumes ``iteration`` batches of ``batch_size``.

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
            start = iteration * batch_size if offset is None else offset
            draws = torch.arange(batch_size, device=device) + start + 1
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

    @property
    def sample_budget(self) -> int:
        """Minimal samples drawn per call: ``max_samples`` if given, else ``batch_size * max_iter``.

        ``batch_size="auto"`` counts ``max_iter`` in batches of 2048, the historical batch, whatever batch the
        call resolves. Read at call time, like ``confidence``, so the attributes can be changed after construction.
        """
        if self.max_samples is not None:
            return self.max_samples
        if isinstance(self.batch_size, int):
            return self.batch_size * self.max_iter
        return _DEFAULT_BATCH * self.max_iter

    def resolve_batch_size(self, num_tc: int, device: torch.device) -> int:
        """Return the batch size of a ``local_optimization="dlt"`` call: the configured one, or the ``"auto"`` choice.

        On CUDA and MPS a homography batch costs about the same from a few hundred up to 8192 hypotheses
        (the four-point solve is launch-bound), so the whole :attr:`sample_budget` is drawn in batches of
        up to 8192; the epipolar solvers are compute-bound past 2048 hypotheses, so their batches stop there
        and early stopping is checked in between. The verification holds a ``batch x N`` residual matrix and a
        few temporaries of that size, so past ``2**27`` entries (about 1 GiB at peak in float32) the batch
        shrinks with ``N``, down to the 2048 of the historical fixed batch. On CPU the cost is linear in
        ``batch * N`` residuals, and the eight-point solver is about ten times a DLT, so the batch aims at a
        millisecond or so of work, 256 to 2048 hypotheses for homographies and 128 to 512 for the epipolar
        models. Other devices retain the historical 2048-sample batch.

        With ``local_optimization="lm"`` an ``"auto"`` batch instead starts at 256 samples on CPU (512 for
        homographies) and doubles after every batch, up to 2048 (4096 for homographies), so that inputs with
        many inliers stop after a small first batch while the rest pay the per-batch overhead a few times only.
        On CUDA and MPS it is the whole budget up to 8192 samples. Essential matrices start at 64 samples on CPU,
        up to 1024, and at 256 on CUDA and MPS, up to 8192: a five-point sample needs few draws at high inlier
        ratios, and its host eigenvalue solve costs the same on every device. All shrink, down to 64 samples, when a
        batch would score more than ``2**22`` (CPU) or ``2**25`` (accelerators) residuals, counting the three models
        of a seven-point sample and the ten candidate slots of a five-point one. On accelerators scoring holds two or
        three times that many entries at its peak, about 0.5 GiB in float32; on CPU it scores tiles of about a million
        residuals, independently of the batch. An integer ``batch_size`` is kept for every batch.
        """
        if isinstance(self.batch_size, int):
            return self.batch_size
        planar = self.model_type in ("homography", "homography_from_linesegments")
        if device.type == "cpu":
            work, lower, upper = (1 << 19, 256, 2048) if planar else (1 << 17, 128, 512)
        elif device.type in ("cuda", "mps"):
            work, lower, upper = 1 << 27, 2048, 8192 if planar else 2048
        else:
            return min(_DEFAULT_BATCH, self.sample_budget)
        batch = min(max(work // max(num_tc, 1), lower), upper)
        return min(batch, self.sample_budget)

    def _lm_batch_range(self, num_tc: int, device: torch.device) -> Tuple[int, int]:
        """First and largest batch of a ``local_optimization="lm"`` call; see :meth:`resolve_batch_size`."""
        budget = self.sample_budget
        if isinstance(self.batch_size, int):
            return min(self.batch_size, budget), min(self.batch_size, budget)
        # Candidate slots per sample: the cubic's three roots, the degree-ten polynomial's ten for "essential".
        models = {7: 3, 5: 10}.get(self.minimal_sample_size, 1)
        planar = self.model_type == "homography"
        # A five-point sample needs few draws at high inlier ratios, and the host eigenvalue solve costs about 7 us
        # per sample on every device: essential matrices start small everywhere (tuned on PhotoTourism).
        essential = self.model_type == "essential"
        if device.type == "cpu":
            first, upper, work = (
                (512, 4096, 1 << 22) if planar else (64, 1024, 1 << 22) if essential else (256, 2048, 1 << 22)
            )
        elif device.type in ("cuda", "mps"):
            first, upper, work = (256 if essential else 8192), 8192, 1 << 25
        else:
            first, upper, work = _DEFAULT_BATCH, _DEFAULT_BATCH, 1 << 25
        largest = max(min(upper, work // (models * max(num_tc, 1))), 64)
        return min(first, largest, budget), min(largest, budget)

    def _is_supported(self, num_inliers: float) -> bool:
        """Whether a model has support beyond the minimal sample it may have been fitted to.

        A minimal-sample model always fits its own sample, so only further inliers are evidence of a
        consensus; without them, input with no consensus returns the all-zero failure matrix.
        """
        return num_inliers > self.minimal_sample_size

    def _prosac_schedule(self, sample_size: int, pop_size: int, device: torch.device) -> torch.Tensor:
        """Return the cumulative PROSAC draw counts on ``device``, converting them once per configuration."""
        key = (sample_size, pop_size, self.sample_budget, device)
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
        budget = self.sample_budget
        m, total = self.minimal_sample_size, inliers.numel()
        if self.confidence >= 1.0 or total <= m:
            return budget
        bound = min(budget, self.max_samples_by_conf(num_inliers, total, m, self.confidence))
        n_min = max(m + 1, min(total // 2, 100))
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
            # The sampling generators use seed + batch index, at most sample_budget of them.
            generator.manual_seed(self.seed + self.sample_budget + iteration)
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
        if self.local_optimization == "lm":
            with torch.no_grad():
                return self._forward_lm(kp1, kp2)
        best_score_total = -float("inf")
        num_tc: int = len(kp1)
        budget = self.sample_budget
        batch_size = self.resolve_batch_size(num_tc, kp1.device)
        max_samples = budget
        best_model_total = torch.zeros(3, 3, dtype=kp1.dtype, device=kp1.device)
        inliers_best_total: torch.Tensor = torch.zeros(num_tc, device=kp1.device, dtype=torch.bool)
        # Only a minimal-solver model needs projecting onto the essential manifold; LO refits already are.
        best_needs_projection = False
        for i in range(-(-budget // batch_size)):
            if i * batch_size >= max_samples:
                break
            # Sample minimal samples in batch to estimate models. The last batch is truncated to the budget after
            # sampling, so that the PROSAC schedule and the seed follow the nominal batch size.
            current = min(batch_size, budget - i * batch_size)
            idxs = self.sample(self.minimal_sample_size, num_tc, batch_size, i, kp1.device)[:current]
            kp1_sampled = kp1[idxs]
            kp2_sampled = kp2[idxs]
            kp1_sampled, kp2_sampled = self.remove_bad_samples(kp1_sampled, kp2_sampled)
            if len(kp1_sampled) == 0:
                continue
            # Estimate models
            models = self.estimate_model_from_minsample(kp1_sampled, kp2_sampled)
            if self.model_type in ["essential", "fundamental", "fundamental_7pt"]:
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
                        budget,
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

    def _lm_minimal_models(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Minimal models ``(M, 3, 3)`` of normalized (for essential matrices, calibrated) samples ``(B, m, 3)``.

        Samples that :func:`~kornia.geometry.homography.sample_is_valid_for_homography` rejects and absent roots of
        the seven-point cubic are dropped on CPU; on other devices, where dropping would need a synchronization, they
        are NaN models, which score no inliers. The orientation test is unaffected by the normalization, a
        translation and a positive scale.

        Returns:
            The models and the sample row ``(M,)`` each was solved from, in draw order: sample by sample, and root by
            root within a sample, as DEGENSAC's record setters need them.
        """
        compact = x1.device.type == "cpu"
        rows = torch.arange(x1.shape[0], device=x1.device)
        if self.model_type == "homography":
            oriented = sample_is_valid_for_homography(x1[..., :2], x2[..., :2])
            if compact:
                x1, x2, rows = x1[oriented], x2[oriented], rows[oriented]
                if len(x1) == 0:
                    return x1.new_zeros(0, 3, 3), rows
                return _four_point_homography(x1, x2), rows
            models = _four_point_homography(x1, x2)
            return models.masked_fill(~oriented[:, None, None], float("nan")), rows
        design = _epipolar_design_rows(x1, x2)
        if self.model_type == "essential":
            # Samples with a non-finite correspondence, rank-deficient samples and complex roots give NaN slots.
            candidates, valid = _five_point_candidates(design)
            rows = rows.repeat_interleave(candidates.shape[1])
            if compact:
                return candidates[valid], rows[valid.flatten()]
            return candidates.flatten(0, 1), rows
        if self.minimal_sample_size == 7:
            candidates, valid = _seven_point_candidates(design)
            models = candidates.masked_fill(~valid[..., None, None], float("nan")).flatten(0, 1)
            rows = rows.repeat_interleave(candidates.shape[1])
        else:
            models = _eight_point_fundamental(design)
        if compact:
            keep = torch.isfinite(models).flatten(1).all(1)
            return models[keep], rows[keep]
        return models, rows

    @staticmethod
    def _raw_record_setters(scores: torch.Tensor, prior: float) -> List[int]:
        """Indices of the models, in draw order, whose score beats ``prior`` and every earlier score of the batch.

        The models a sequential loop would find to set a new raw record (Chum's ``maxSs``), found at once with a
        running maximum over the CPU ``scores``.
        """
        earlier = torch.cat([scores.new_full((1,), prior), torch.cummax(scores, 0).values[:-1]])
        return (scores > earlier.clamp_min(prior)).nonzero().flatten().tolist()

    def _lm_errors(self, models: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Squared residuals ``(M, N)`` of normalized models on normalized correspondences (calibrated: essential)."""
        if self.model_type == "homography":
            return _transfer_from_basis(models, _transfer_basis(x1, x2[:, :2]))
        return _sampson_from_quadratic_basis(models, _sampson_quadratic_basis(x1, x2))

    def _lm_score(self, errors: torch.Tensor, threshold: float) -> torch.Tensor:
        """MSAC scores ``sum(1 - min(e / threshold, 1))`` or RANSAC support counts ``(M,)`` of squared residuals."""
        if self.score_type == "msac":
            # fmin, unlike clamp, takes the threshold for NaN residuals: a NaN residual is an outlier.
            # The clipped residuals own their storage: reuse it for the pointwise score operations.
            contributions = torch.fmin(errors, torch.full_like(errors[:1, :1], threshold))
            return contributions.div_(-threshold).add_(1.0).sum(1)
        return (errors <= threshold).sum(1).to(errors.dtype)

    def _lm_score_models(
        self, models: torch.Tensor, basis: torch.Tensor, threshold: float, max_residuals: int = 1 << 20
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Score minimal models without retaining the whole model-by-correspondence residual matrix.

        CPU hypothesis tiles bound temporary storage independently of the solver batch. Correspondences are never
        split, so each score uses the same reduction as full verification. Accelerators keep one tile to amortize
        launch overhead. Support counts accumulate in the working dtype while they are exactly representable,
        avoiding the default int64 conversion of every entry of the inlier matrix. Only PROSAC retains the boolean
        masks: its stopping rule must use the same inlier decisions as the score, including at rounding boundaries.
        """
        planar = self.model_type == "homography"
        n = basis.shape[1] // (3 if planar else 2)
        tile = max(1, max_residuals // max(n, 1)) if models.device.type == "cpu" else max(len(models), 1)
        scores, counts, masks = [], [], []
        count_dtype = models.dtype if n <= 1 << 24 else torch.int64
        residual_fn = _transfer_from_basis if planar else _sampson_from_quadratic_basis
        for start in range(0, len(models), tile):
            errors = residual_fn(models[start : start + tile], basis)
            inliers = errors <= threshold
            support = inliers.sum(1, dtype=count_dtype)
            counts.append(support)
            scores.append(self._lm_score(errors, threshold) if self.score_type == "msac" else support.to(models.dtype))
            if self.prosac_sampling:
                masks.append(inliers)
        if len(scores) == 1:
            return scores[0], counts[0], masks[0] if masks else None
        return torch.cat(scores), torch.cat(counts), torch.cat(masks) if masks else None

    def _lm_refine(
        self,
        models: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor],
        loss: str,
        scale2: float,
        iters: int,
    ) -> torch.Tensor:
        """Refine normalized (calibrated: essential) models with Levenberg-Marquardt, on float64 host tensors."""
        if self.model_type == "homography":
            return _refine_homography_lm(models, x1, x2[:, :2], mask, loss, scale2, iters)
        if self.model_type == "essential":
            return _refine_essential_lm(models, x1, x2, mask, loss, scale2, iters)
        return _refine_fundamental_lm(models, x1, x2, mask, loss, scale2, iters)

    def _degensac_recover(
        self,
        model: torch.Tensor,
        sample: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_host: torch.Tensor,
        x2_host: torch.Tensor,
        basis: torch.Tensor,
        threshold: float,
        generator: Optional[torch.Generator],
        seen_planes: Optional[List[torch.Tensor]] = None,
        homography: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """DEGENSAC's recovery of a raw record setter (Chum, Werner and Matas, CVPR 2005; Chum's ``exp_ranF.c``).

        ``model`` ``(3, 3)`` is a seven-point model of ``sample`` ``(7,)``, both on the device of the normalized
        correspondences ``x1``, ``x2`` ``(N, 3)``; ``x1_host``, ``x2_host`` are the same correspondences on the host in
        float64, NaN where not finite; ``threshold`` is the squared threshold ``t`` of that frame. When the sample is
        H-degenerate and its homography has at least 8 inliers at ``3 t``, the homography is refined (``innerH``), and
        with more than 6 plane inliers at ``16 t`` and at least 4 correspondences beyond ``100 t``, plane-and-parallax
        models are drawn from those (:mod:`kornia.geometry._degensac`), unless ``seen_planes``, the refined planes'
        inlier masks of the call's earlier recoveries, already holds the plane (:func:`_repeats_plane`); a new plane
        is appended to it. The best-scoring model is refined like the pool, with ``max_lo_iters`` truncated
        Levenberg-Marquardt iterations, Chum's ``innerFH`` role, and the better of it and its refinement is returned
        alone, as ``rFtH`` returns one model: near-duplicates from one search would crowd the eight-model pool.

        ``homography``, when supplied, is the result of the batch's degeneracy check in float64 on the host.

        Returns:
            The recovered model ``(1, 3, 3)`` with its score, support and, with PROSAC, inlier mask, as
            :meth:`_lm_score_models` returns them; None when the sample is not H-degenerate or nothing is recovered.
        """
        m = self.minimal_sample_size
        if homography is None:
            sample_host = sample.cpu()
            homography = _h_degenerate_sample(
                model.detach().cpu().double(), x1_host[sample_host], x2_host[sample_host], 3 * threshold
            )
        if homography is None:
            return None
        rows = (torch.isfinite(x1_host).all(1) & torch.isfinite(x2_host).all(1)).nonzero().flatten()
        p1, p2 = x1_host[rows, :2], x2_host[rows, :2]
        errors = sampson_homography_distance(p1[None], p2[None], homography[None])[0]
        planes = [] if seen_planes is None else seen_planes
        # A plane already searched in this call is skipped before its refinement when the sample's own homography
        # falls inside it, and after when the refined plane matches it.
        if int((errors < 3 * threshold).sum()) < 8 or _inside_plane(errors <= 16 * threshold, planes):
            return None
        homography, errors = _inner_homography(homography, p1, p2, 16 * threshold, generator)
        plane, off_plane = errors <= 16 * threshold, errors > 100 * threshold
        if _repeats_plane(plane, planes):
            return None
        planes.append(plane)
        if int(plane.sum()) <= 6 or int(off_plane.sum()) < 4:
            return None
        off_rows = rows[off_plane].to(x1.device)
        found = _plane_parallax_search(
            homography.to(x1.device, x1.dtype),
            x1[off_rows],
            x2[off_rows],
            2 * threshold,
            256 if x1.device.type == "cpu" else 2048,
            generator,
        )
        if found is None:
            return None
        models = found[0]
        scores, counts, masks = self._lm_score_models(models, basis, threshold)
        scores = scores.masked_fill(counts <= m, -1.0)
        if float(scores.max()) < 0:
            return None
        best = int(scores.argmax())
        raw = models[best : best + 1], scores[best : best + 1], counts[best : best + 1]
        raw_mask = None if masks is None else masks[best : best + 1]
        if self.max_lo_iters == 0:
            return (*raw, raw_mask)
        refined = self._lm_refine(
            raw[0].cpu().double(), x1_host[rows], x2_host[rows], None, "truncated", threshold, self.max_lo_iters
        ).to(x1.device, x1.dtype)
        scores, counts, masks = self._lm_score_models(refined, basis, threshold)
        scores = scores.masked_fill(counts <= m, -1.0)
        # The truncated loss is not the score: with score_type="ransac" a step can trade inliers (568 -> 566 on one
        # dominant-plane sample). Keep the better model, the refit on ties.
        if float(scores[0]) < float(raw[1][0]):
            return (*raw, raw_mask)
        return refined, scores, counts, masks

    @staticmethod
    def _lm_pool(
        candidates: torch.Tensor, candidate_scores: torch.Tensor, models: torch.Tensor, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The pool's ``_LM_CANDIDATES`` best models and scores after adding ``models``, placed first for ties."""
        candidate_scores, order = torch.cat([scores, candidate_scores]).topk(
            min(_LM_CANDIDATES, len(scores) + len(candidate_scores))
        )
        return torch.cat([models, candidates])[order], candidate_scores

    def _lm_stopping_bound(
        self, masks: Optional[torch.Tensor], row: Union[int, torch.Tensor], support: int, num_tc: int
    ) -> int:
        """Samples to draw for an incumbent with ``support`` inliers, as in OpenCV's USAC.

        PROSAC's termination test on ``masks[row]`` when PROSAC keeps masks, else the classic bound.
        """
        if masks is not None:
            return self._prosac_max_samples(masks[row], support)
        return min(
            self.sample_budget, self.max_samples_by_conf(support, num_tc, self.minimal_sample_size, self.confidence)
        )

    def _degensac_batch(
        self,
        models: torch.Tensor,
        samples: torch.Tensor,
        scores: torch.Tensor,
        prior: float,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_host: torch.Tensor,
        x2_host: torch.Tensor,
        basis: torch.Tensor,
        threshold: float,
        iteration: int,
        seen_planes: Optional[List[torch.Tensor]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """DEGENSAC's recoveries of a batch's raw record setters, in draw order (:meth:`_raw_record_setters`).

        ``models`` ``(M, 3, 3)`` were solved from ``samples`` ``(M, 7)`` and scored ``scores``; ``prior`` is the raw
        record before the batch. Every recovery draw comes from one private host generator per batch: a device
        generator cannot draw on the host, and the global one, which an unseeded call's minimal samples come from,
        must not advance, or every later sample of the call would differ from ``degensac=False``. A seeded call seeds
        it ``seed + 2 * sample_budget + iteration``; an unseeded one from the first record setter's sample, drawn from
        the global generator, so that its draws still differ between calls.
        """
        # Transfer and test all record setters together; only the subsequent recovery depends on earlier planes.
        records = self._raw_record_setters(scores.cpu(), prior)
        if not records:
            return []
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed + 2 * self.sample_budget + iteration)
        else:
            # Python hashes a tuple of ints the same in every process.
            generator.manual_seed(hash(tuple(samples[records[0]].tolist())) & 0x7FFFFFFFFFFFFFFF)
        planes = [] if seen_planes is None else seen_planes
        recoveries = []
        samples_host = samples[records].cpu()
        homographies = _h_degenerate_samples(
            models[records].detach().cpu().double(), x1_host[samples_host], x2_host[samples_host], 3 * threshold
        )
        for record, homography in zip(records, homographies):
            if homography is None:
                continue
            recovered = self._degensac_recover(
                models[record],
                samples[record],
                x1,
                x2,
                x1_host,
                x2_host,
                basis,
                threshold,
                generator,
                planes,
                homography,
            )
            if recovered is not None:
                recoveries.append(recovered)
        return recoveries

    def _forward_lm(self, kp1: torch.Tensor, kp2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """RANSAC with batched minimal solvers and Levenberg-Marquardt local optimization and refinement.

        Sampling, minimal solving and scoring run on the device of the correspondences; the normalization, the
        refinements and the final selection run on the host in float64, which needs a handful of synchronizations
        per call instead of one per small operation. What differs from the public solvers is what a sampling loop
        needs: correspondences are normalized once per call rather than per sample, models stay in that normalized
        frame at unit Frobenius norm instead of ``F[2, 2] = 1``, absent candidates are NaN on accelerators rather than
        dropped, and scoring uses the quadratic Sampson and folded transfer bases, which are fast and accurate in that
        frame only. Essential matrices skip the normalization: calibrated coordinates are already of unit scale, and
        a translation or scaling of them would take the models off the essential manifold.
        """
        device, dtype = kp1.device, kp1.dtype
        work = torch.float64 if dtype == torch.float64 else torch.float32
        num_tc, m = len(kp1), self.minimal_sample_size
        planar = self.model_type == "homography"
        essential = self.model_type == "essential"
        failure = (torch.zeros(3, 3, dtype=dtype, device=device), torch.zeros(num_tc, dtype=torch.bool, device=device))
        host = torch.device("cpu")
        # The Sampson distance scales with a similarity shared by both images; the transfer error with image 2's.
        # Moved first, then cast: a single .to(host, torch.float64) out of MPS returns zeros on torch 2.14 and
        # raises on 2.5.1 (pytorch/pytorch#197715).
        kp1_host, kp2_host = kp1.detach().to(host).double(), kp2.detach().to(host).double()
        finite = torch.isfinite(kp1_host).all(1) & torch.isfinite(kp2_host).all(1)
        if not bool(finite.any()):
            return failure
        if essential:
            # Non-finite correspondences stay NaN, as the normalization leaves them, and are never inliers.
            x1_host, x2_host = (
                convert_points_to_homogeneous(kp.masked_fill(~finite[:, None], float("nan")))
                for kp in (kp1_host, kp2_host)
            )
            t1 = t2 = torch.eye(3, dtype=torch.float64)
            threshold = self.inl_th**2
        else:
            x1_host, x2_host, t1, t2, s1, s2 = _normalize_correspondences(kp1_host, kp2_host, not planar)
            threshold = (self.inl_th / (s2 if planar else s1)) ** 2
        x1, x2 = x1_host.to(device, work), x2_host.to(device, work)
        basis = _transfer_basis(x1, x2[:, :2]) if planar else _sampson_quadratic_basis(x1, x2)
        budget = self.sample_budget
        batch, largest = self._lm_batch_range(num_tc, device)
        grow = not isinstance(self.batch_size, int)
        candidates, candidate_scores = x1.new_zeros(0, 3, 3), x1.new_zeros(0)
        best_score = -1.0
        # DEGENSAC tests the models that set a new raw record (Chum's maxSs), which recovered models never raise;
        # best_score is the incumbent's, over raw and recovered models (maxS).
        degensac = self.degensac and m == 7
        best_minimal_score = -1.0
        seen_planes: List[torch.Tensor] = []
        max_samples, drawn, iteration = budget, 0, 0
        while drawn < max_samples:
            current = min(batch, max_samples - drawn)
            indices = self.sample(m, num_tc, current, iteration, device, offset=drawn)
            batch_iteration = iteration
            drawn, iteration = drawn + current, iteration + 1
            if grow:
                batch = min(2 * batch, largest)
            models, origin = self._lm_minimal_models(x1[indices], x2[indices])
            if len(models) == 0:
                continue
            # Reject insufficient support before ranking: high MSAC scores from minimal samples alone must not
            # crowd supported models out of the candidate pool.
            scores_all, counts_all, masks_all = self._lm_score_models(models, basis, threshold)
            scores_all = scores_all.masked_fill(counts_all <= m, -1.0)
            top_scores, top = scores_all.topk(min(_LM_CANDIDATES, len(models)))
            top_counts = counts_all[top]
            # The run's best minimal models so far, refined after sampling.
            candidates, candidate_scores = self._lm_pool(candidates, candidate_scores, models[top], top_scores)
            scores, counts = torch.stack([top_scores, top_counts.to(top_scores.dtype)]).tolist()
            best = max(range(len(scores)), key=scores.__getitem__)
            if scores[best] > best_score:
                # The bound follows the new incumbent's own support, as with local_optimization="dlt".
                best_score = scores[best]
                max_samples = self._lm_stopping_bound(masks_all, top[best], int(counts[best]), num_tc)
            if not degensac or scores[best] <= best_minimal_score:
                continue
            recoveries = self._degensac_batch(
                models,
                indices[origin],
                scores_all,
                best_minimal_score,
                x1,
                x2,
                x1_host,
                x2_host,
                basis,
                threshold,
                batch_iteration,
                seen_planes,
            )
            best_minimal_score = scores[best]
            for recovered_models, recovered_scores, recovered_counts, recovered_masks in recoveries:
                candidates, candidate_scores = self._lm_pool(
                    candidates, candidate_scores, recovered_models, recovered_scores
                )
                winner = int(recovered_scores.argmax())
                if float(recovered_scores[winner]) > best_score:
                    best_score = float(recovered_scores[winner])
                    max_samples = self._lm_stopping_bound(
                        recovered_masks, winner, int(recovered_counts[winner]), num_tc
                    )
        x1_host, x2_host = x1_host[finite], x2_host[finite]
        candidates = candidates.to(host).double()[candidate_scores.to(host) >= 0]
        if len(candidates) == 0:
            return failure
        if self.max_lo_iters > 0:
            refits = self._lm_refine(candidates, x1_host, x2_host, None, "truncated", threshold, self.max_lo_iters)
            # Refits first: a tie goes to the refit, which is fitted to more than a minimal sample.
            candidates = torch.cat([refits, candidates])
        errors = self._lm_errors(candidates, x1_host, x2_host)
        inliers = errors <= threshold
        scores = self._lm_score(errors, threshold).masked_fill(inliers.sum(1) <= m, -1.0)
        best = int(scores.argmax())
        if scores[best] < 0:
            return failure
        model, inliers = candidates[best], inliers[best]
        if self.refine_iters > 0:
            # A Cauchy scale of a third of the threshold: the threshold as three standard deviations of the noise.
            mask = inliers[None]
            refined = self._lm_refine(model[None], x1_host, x2_host, mask, "cauchy", threshold / 9, self.refine_iters)
            refined_inliers = self._lm_errors(refined, x1_host, x2_host)[0] <= threshold
            if self._is_supported(int(refined_inliers.sum())):
                model, inliers = refined[0], refined_inliers
        if essential:
            # A minimal model is essential only up to rounding when neither refinement ran: project it.
            model = project_to_essential(model)
            model = (model / model.norm()).to(dtype)
            # E and -E are the same model; a fixed sign makes equal inputs give equal outputs.
            model = model * model.flatten()[model.abs().argmax()].sign()
        else:
            model = (torch.linalg.inv(t2) @ model @ t1) if planar else (t2.mT @ model @ t1)
            model = normalize_transformation(model).to(dtype)
        if not bool(torch.isfinite(model).all()):
            return failure
        # Casting (especially to half precision) changes the model. Classify that exact returned matrix in pixel
        # coordinates, in float64 so metric arithmetic does not add another round of low-precision error. The public
        # line/transfer forms also avoid cancellation in the normalized quadratic sampling basis near epipoles.
        errors = self.error_fn(kp1_host[None], kp2_host[None], model[None].to(torch.float64), eps=0.0)[0]
        mask = finite & (errors <= self.inl_th**2)
        if not self._is_supported(int(mask.sum())):
            return failure
        return model.to(device), mask.to(device)

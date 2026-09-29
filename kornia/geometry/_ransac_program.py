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

"""The whole of ``RANSAC(local_optimization="lm")`` as one tensor program, compiled by ``RANSAC(compile=True)``.

:meth:`~kornia.geometry.ransac.RANSAC._forward_lm` drives its sampling loop from the host: it reads each batch's best
score to decide whether to stop, and returns early when no model is supported. Here the loop is a
:func:`torch.while_loop` whose state is a fixed-size pool of the best minimal models, the stopping bound is computed
in tensors, and every early return is a :func:`torch.where`. The sizes that depend on the data, the batch of each
iteration and the number of models that survive the degeneracy tests, are unbacked, so a single graph serves every
number of correspondences, threshold, confidence and sample budget. ``RANSAC(compile=True)`` compiles it once per
configuration and keeps the compiled function on disk (:func:`load_program`).
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import os
import platform
import sys
import threading
import warnings
from typing import Callable, Dict, Iterator, Tuple

import torch

import kornia
from kornia.geometry import homography as _homography
from kornia.geometry import ransac as _ransac
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import _metrics as _epipolar_metrics
from kornia.geometry.epipolar import essential as _essential
from kornia.geometry.epipolar import fundamental as _fundamental
from kornia.geometry.epipolar import normalize_transformation, project_to_essential
from kornia.geometry.epipolar._metrics import _sampson_errors, _sampson_from_quadratic_basis, _sampson_quadratic_basis
from kornia.geometry.epipolar.essential import _five_point_candidates, _refine_essential_lm
from kornia.geometry.epipolar.fundamental import (
    _eight_point_fundamental,
    _epipolar_design_rows,
    _refine_fundamental_lm,
    _seven_point_candidates,
)
from kornia.geometry.homography import (
    _four_point_homography,
    _refine_homography_lm,
    _transfer_basis,
    _transfer_errors,
    _transfer_from_basis,
    sample_is_valid_for_homography,
)
from kornia.geometry.ransac import _LM_CANDIDATES, _normalize_correspondences_core

__all__ = ["MAX_BATCH", "build_lm_program", "load_program"]

# The candidate pool: as many minimal models as RANSAC(local_optimization="lm") refines.
_POOL = _LM_CANDIDATES
# Upper bound of a batch; RANSAC(compile=True) rejects a larger integer batch_size.
MAX_BATCH = 1 << 20

_SAMPLE_SIZES = {"homography": 4, "fundamental": 7, "fundamental_7pt": 7, "fundamental_8pt": 8, "essential": 5}


def _draw_samples(m: int, num_tc: int, batch: int, device: torch.device) -> torch.Tensor:
    """``batch`` uniform ``m``-subsets of ``range(num_tc)`` from the global generator, as :meth:`RANSAC.sample` draws.

    Floyd's algorithm on CPU, random-key top-k elsewhere; without PROSAC and without a private generator, which a
    compiled graph cannot hold (a seeded call forks the global generator instead).
    """
    if device.type != "cpu":
        return torch.rand(batch, num_tc, device=device).topk(k=m, dim=1, sorted=False).indices
    rand = torch.rand(batch, m, device=device, dtype=torch.float64)
    columns = []
    for i in range(m):
        last = num_tc - m + i
        candidate = (rand[:, i] * (last + 1)).long()
        if i > 0:
            duplicate = (torch.stack(columns, 1) == candidate[:, None]).any(dim=1)
            candidate = torch.where(duplicate, last, candidate)
        columns.append(candidate)
    return torch.stack(columns, 1)


def _minimal_models(model_type: str, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Minimal models ``(M, 3, 3)`` of samples ``(B, m, 3)``, as :meth:`RANSAC._lm_minimal_models`.

    On CPU the degenerate samples and the absent roots are dropped (``M`` is unbacked); elsewhere they are NaN models,
    which score no inliers. Every sample is solved before the drop: the shape checks of the batched triangular solves
    would guard on an unbacked batch being empty.
    """
    compact = x1.device.type == "cpu"
    if model_type == "homography":
        oriented = sample_is_valid_for_homography(x1[..., :2], x2[..., :2])
        models = _four_point_homography(x1, x2)
        return models[oriented] if compact else models.masked_fill(~oriented[:, None, None], float("nan"))
    design = _epipolar_design_rows(x1, x2)
    if model_type == "essential":
        candidates, valid = _five_point_candidates(design)
        return candidates[valid] if compact else candidates.flatten(0, 1)
    if model_type == "fundamental_8pt":
        models = _eight_point_fundamental(design)
    else:
        candidates, valid = _seven_point_candidates(design)
        models = candidates.masked_fill(~valid[..., None, None], float("nan")).flatten(0, 1)
    return models[torch.isfinite(models).flatten(1).all(1)] if compact else models


def _scores(
    models: torch.Tensor, basis: torch.Tensor, threshold: torch.Tensor, planar: bool, msac: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MSAC scores (or support counts) in float64 and int64 support counts ``(M,)`` of minimal models.

    Both sums accumulate in 64 bits: a float32 sum longer than 4096 elements makes inductor's CPU code guard on the
    number of correspondences, which a single graph for every ``N`` cannot carry.
    """
    errors = _transfer_from_basis(models, basis) if planar else _sampson_from_quadratic_basis(models, basis)
    counts = (errors <= threshold).sum(1)
    if not msac:
        return counts.to(torch.float64), counts
    # fmin, unlike clamp, takes the threshold for NaN residuals: a NaN residual is an outlier.
    return (1.0 - torch.fmin(errors, threshold) / threshold).sum(1, dtype=torch.float64), counts


def _samples_needed(
    support: torch.Tensor, num_tc: int, m: int, confidence: torch.Tensor, budget: torch.Tensor
) -> torch.Tensor:
    """:meth:`RANSAC.max_samples_by_conf` of ``support`` inliers in tensors, capped at ``budget`` (int64)."""
    inliers = support.to(torch.float64)
    probability = torch.ones_like(inliers)
    for i in range(m):
        probability = probability * (inliers - i) / (num_tc - i)
    needed = torch.ceil(torch.log1p(-confidence) / torch.log1p(-probability))
    bounded = (confidence < 1) & (inliers >= m) & (probability > 0) & (needed < budget.to(torch.float64))
    needed = torch.where(bounded, needed, budget.to(torch.float64)).to(torch.int64)
    return torch.where((confidence < 1) & (inliers >= num_tc), torch.ones_like(needed), needed)


def _lm_errors(model_type: str, models: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Squared residuals ``(K, N)`` of normalized models on normalized points, as :meth:`RANSAC._lm_errors`."""
    if model_type == "homography":
        return _transfer_from_basis(models, _transfer_basis(x1, x2[:, :2]))
    return _sampson_from_quadratic_basis(models, _sampson_quadratic_basis(x1, x2))


def _lm_refine(
    model_type: str,
    models: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    mask: torch.Tensor | None,
    loss: str,
    scale2: torch.Tensor,
    iters: int,
) -> torch.Tensor:
    if model_type == "homography":
        return _refine_homography_lm(models, x1, x2[:, :2], mask, loss, scale2, iters)
    if model_type == "essential":
        return _refine_essential_lm(models, x1, x2, mask, loss, scale2, iters)
    return _refine_fundamental_lm(models, x1, x2, mask, loss, scale2, iters)


def build_lm_program(model_type: str, score_type: str, max_lo_iters: int, refine_iters: int) -> Callable[..., Tuple]:
    """Return the tensor program of ``RANSAC(model_type, score_type=..., local_optimization="lm")``.

    The program maps ``(kp1, kp2, inl_th, confidence, budget, first_batch, largest_batch)`` -- the correspondences
    ``(N, 2)`` and five 0-d tensors (float64 threshold and confidence, int64 sample budget and first and largest batch,
    all on the host) -- to the model ``(3, 3)`` and the inlier mask ``(N,)`` that
    :meth:`~kornia.geometry.ransac.RANSAC._forward_lm` returns for them. Batches double from ``first_batch`` up to
    ``largest_batch``; equal values keep a fixed batch.
    """
    m = _SAMPLE_SIZES[model_type]
    planar = model_type == "homography"
    essential = model_type == "essential"
    msac = score_type == "msac"

    def lm_program(
        kp1: torch.Tensor,
        kp2: torch.Tensor,
        inl_th: torch.Tensor,
        confidence: torch.Tensor,
        budget: torch.Tensor,
        first_batch: torch.Tensor,
        largest_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = kp1.device, kp1.dtype
        work = torch.float64 if dtype == torch.float64 else torch.float32
        num_tc = kp1.shape[0]
        torch._check(num_tc >= m)
        host = torch.device("cpu")
        # Moved first, then cast: a single .to(host, torch.float64) out of MPS returns zeros (pytorch/pytorch#197715).
        kp1_host, kp2_host = kp1.detach().to(host).double(), kp2.detach().to(host).double()
        finite = torch.isfinite(kp1_host).all(1) & torch.isfinite(kp2_host).all(1)
        if essential:
            x1_host, x2_host = (
                convert_points_to_homogeneous(kp.masked_fill(~finite[:, None], float("nan")))
                for kp in (kp1_host, kp2_host)
            )
            t1 = t2 = torch.eye(3, dtype=torch.float64)
            threshold = inl_th.square()
        else:
            x1_host, x2_host, t1, t2, scales = _normalize_correspondences_core(kp1_host, kp2_host, not planar)
            threshold = (inl_th / scales[1 if planar else 0]).square()
        # Copies: the normalized points are views of one tensor, and inputs of the loop body must not alias.
        x1, x2 = x1_host.to(device, work, copy=True), x2_host.to(device, work, copy=True)
        basis = _transfer_basis(x1, x2[:, :2]) if planar else _sampson_quadratic_basis(x1, x2)
        threshold_work = threshold.to(device, work)
        pad_models = x1.new_full((_POOL, 3, 3), float("nan"))
        pad_scores = torch.full((_POOL,), -2.0, dtype=torch.float64, device=device)
        pad_counts = torch.zeros(_POOL, dtype=torch.int64, device=device)

        def cond(drawn, batch, max_samples, best_score, pool, pool_scores):  # type: ignore[no-untyped-def]
            return drawn < max_samples

        def body(drawn, batch, max_samples, best_score, pool, pool_scores):  # type: ignore[no-untyped-def]
            current = torch.minimum(batch, max_samples - drawn)
            size = current.item()
            torch._check(size >= 1)
            torch._check(size <= MAX_BATCH)
            indices = _draw_samples(m, num_tc, size, device)
            models = _minimal_models(model_type, x1[indices], x2[indices])
            scores, counts = _scores(models, basis, threshold_work, planar, msac)
            # Insufficient support ranks below every supported model, as in RANSAC._forward_lm; the padding, which
            # keeps the top-k defined for a batch without models, below that.
            scores = torch.cat([scores.masked_fill(counts <= m, -1.0), pad_scores])
            top_scores, top = scores.topk(_POOL)
            top_models = torch.cat([models, pad_models])[top]
            top_counts = torch.cat([counts, pad_counts])[top]
            pool_scores, order = torch.cat([top_scores, pool_scores]).topk(_POOL)
            pool = torch.cat([top_models, pool])[order]
            # The bound follows the new incumbent's own support, as with local_optimization="dlt".
            improved = top_scores[0] > best_score
            needed = _samples_needed(top_counts[0], num_tc, m, confidence, budget)
            max_samples = torch.where(improved, torch.minimum(budget, needed), max_samples)
            best_score = torch.where(improved, top_scores[0], best_score)
            batch = torch.minimum(2 * batch, largest_batch)
            return drawn + current, batch, max_samples, best_score, pool, pool_scores

        state = (
            torch.zeros((), dtype=torch.int64),
            first_batch.clone(),
            budget.clone(),
            torch.full((), -1.0, dtype=torch.float64),
            pad_models.clone(),
            pad_scores.clone(),
        )
        _, _, _, _, pool, pool_scores = torch.while_loop(cond, body, state)

        # ---- refinement and selection on the host, in float64 ----
        # RANSAC._forward_lm drops the non-finite correspondences here. Dropping them would leave an unbacked number
        # of points, whose strides inductor cannot lower: they stay as finite stand-ins of weight 0 instead, which
        # adds zeros to the same sums.
        stand_in = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        x1_host = torch.where(finite[:, None], x1_host, stand_in)
        x2_host = torch.where(finite[:, None], x2_host, stand_in)
        weights = finite.to(torch.float64)[None]
        valid = (pool_scores >= 0).to(host)
        # An absent pool slot holds a finite stand-in, which the factorizations of the refinement accept, and is
        # never selected.
        candidates = torch.where(valid[:, None, None], pool.to(host).double(), torch.eye(3, dtype=torch.float64))
        if max_lo_iters > 0:
            refits = _lm_refine(model_type, candidates, x1_host, x2_host, weights, "truncated", threshold, max_lo_iters)
            # Refits first: a tie goes to the refit, which is fitted to more than a minimal sample.
            candidates = torch.cat([refits, candidates])
            valid = torch.cat([valid, valid])
        errors = _lm_errors(model_type, candidates, x1_host, x2_host)
        inliers = (errors <= threshold) & finite
        support = inliers.sum(1)
        if msac:
            scores = ((1.0 - torch.fmin(errors, threshold) / threshold) * weights).sum(1)
        else:
            scores = support.to(torch.float64)
        scores = scores.masked_fill(support <= m, -1.0).masked_fill(~valid, -2.0)
        best = scores.argmax().reshape(1)
        failed = scores.max() < 0
        model, inliers = candidates.index_select(0, best)[0], inliers.index_select(0, best)[0]
        if refine_iters > 0:
            # A Cauchy scale of a third of the threshold: the threshold as three standard deviations of the noise.
            mask = inliers.to(torch.float64)[None]
            refined = _lm_refine(model_type, model[None], x1_host, x2_host, mask, "cauchy", threshold / 9, refine_iters)
            refined_inliers = (_lm_errors(model_type, refined, x1_host, x2_host)[0] <= threshold) & finite
            supported = refined_inliers.sum() > m
            model = torch.where(supported, refined[0], model)
            inliers = torch.where(supported, refined_inliers, inliers)
        if essential:
            # A minimal model is essential only up to rounding when neither refinement ran: project it.
            model = project_to_essential(model)
            model = (model / model.norm()).to(dtype)
            # E and -E are the same model; a fixed sign makes equal inputs give equal outputs.
            model = model * model.flatten().index_select(0, model.abs().argmax().reshape(1)).sign()
        else:
            model = (torch.linalg.inv(t2) @ model @ t1) if planar else (t2.mT @ model @ t1)
            model = normalize_transformation(model).to(dtype)
        failed = failed | ~torch.isfinite(model).all()
        # The exact returned matrix, classified in pixel coordinates in float64, as RANSAC._forward_lm does.
        model64 = torch.where(failed, torch.eye(3, dtype=torch.float64), model.to(torch.float64))
        p1 = convert_points_to_homogeneous(kp1_host)
        if planar:
            pixel_errors = _transfer_errors(model64[None], p1, kp2_host, 0.0)[0]
        else:
            pixel_errors = _sampson_errors(model64[None], p1, convert_points_to_homogeneous(kp2_host), 0.0)[0]
        mask = finite & (pixel_errors <= inl_th.square())
        failed = failed | (mask.sum() <= m)
        model = torch.where(failed, torch.zeros_like(model), model)
        mask = mask & ~failed
        return model.to(device), mask.to(device)

    return lm_program


# ---- compiled programs, in memory and on disk ----

_LOCK = threading.Lock()
_PROGRAMS: Dict[Tuple, Callable[..., Tuple]] = {}


@functools.cache
def _source_digest() -> str:
    """Digest of the modules whose code the compiled graph embeds, read once: an edit must not load a stale artifact."""
    digest = hashlib.sha256()
    for module in (sys.modules[__name__], _homography, _fundamental, _essential, _epipolar_metrics, _ransac):
        digest.update(inspect.getsource(module).encode())
    return digest.hexdigest()


@contextlib.contextmanager
def _dynamo_config() -> Iterator[None]:
    """The dynamo settings the program needs: ``.item()`` of the batch size and the compacted model count.

    Every configuration compiles the same code object, whose cache of graphs dynamo bounds by ``recompile_limit``. The
    eigenvalues of the five-point solver are complex, which inductor leaves to ATen, with a warning that is expected.
    """
    with (
        warnings.catch_warnings(),
        torch._dynamo.config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True, recompile_limit=64
        ),
    ):
        warnings.filterwarnings(
            "ignore", message="Torchinductor does not support code generation for complex operators"
        )
        yield


def _artifact_path(key: Tuple) -> str:
    from torch._inductor.cpu_vec_isa import pick_vec_isa
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    digest = hashlib.sha256()
    # The generated CPU kernels target this machine's vector instructions, which a shared cache directory may not.
    machine = (platform.platform(), platform.machine(), str(pick_vec_isa()))
    for part in (torch.__version__, kornia.__version__, sys.version, machine, key, _source_digest()):
        digest.update(repr(part).encode())
    return os.path.join(cache_dir(), "kornia_ransac", f"{digest.hexdigest()[:32]}.bin")


def load_program(key: Tuple, example_inputs: Tuple[torch.Tensor, ...]) -> Callable[..., Tuple]:
    """Return the compiled program of ``key``: ``(model_type, score_type, max_lo_iters, refine_iters, dtype, device)``.

    Compiled once per process. The compiled function is also saved next to inductor's own cache
    (``TORCHINDUCTOR_CACHE_DIR``) and loaded from there by later processes, which then skip tracing. The number of
    correspondences is dynamic, so the program serves every ``N``. ``KORNIA_RANSAC_AOT=0`` disables the artifact.
    """
    with _LOCK:
        program = _PROGRAMS.get(key)
        if program is not None:
            return program
        compiled = torch.compile(build_lm_program(*key[:4]), fullgraph=True)
        program = _traced(compiled)
        if os.environ.get("KORNIA_RANSAC_AOT", "1") != "0":
            try:
                program = _load_or_save_artifact(key, compiled, _mark_dynamic(example_inputs))
            except Exception as error:  # noqa: BLE001 - the AOT API is experimental; the traced program is equivalent
                warnings.warn(
                    f"RANSAC(compile=True) could not use its on-disk artifact: {type(error).__name__}: {error}",
                    stacklevel=3,
                )
        _PROGRAMS[key] = program
        return program


def _mark_dynamic(inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """The inputs with the correspondence dimension marked dynamic, on new tensors: the caller's stay unmarked.

    The traced program is called with marked inputs, as its guards expect; an artifact loaded by another process is
    called with the plain ones, since its guards reject marked tensors.
    """
    kp1, kp2 = inputs[0].detach(), inputs[1].detach()
    torch._dynamo.mark_dynamic(kp1, 0)
    torch._dynamo.mark_dynamic(kp2, 0)
    return (kp1, kp2, *inputs[2:])


def _traced(compiled: Callable[..., Tuple]) -> Callable[..., Tuple]:
    """``compiled`` called with the program's dynamo settings and dynamic correspondences."""

    def run(*args: torch.Tensor) -> Tuple:
        with _dynamo_config():
            return compiled(*_mark_dynamic(args))

    return run


def _load_or_save_artifact(
    key: Tuple, compiled: Callable[..., Tuple], example_inputs: Tuple[torch.Tensor, ...]
) -> Callable[..., Tuple]:
    path = _artifact_path(key)
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return torch.compiler.load_compiled_function(handle)
    from torch._dynamo.exc import TensorifyScalarRestartAnalysis

    with _dynamo_config():
        try:
            artifact = compiled.aot_compile((example_inputs, {}))  # type: ignore[attr-defined]
        except TensorifyScalarRestartAnalysis:
            # torch.compile restarts its analysis once when a Python scalar cannot become a tensor; aot_compile
            # (torch 2.14) raises instead, and succeeds when called again.
            artifact = compiled.aot_compile((example_inputs, {}))  # type: ignore[attr-defined]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    artifact.save_compiled_function(path)
    return artifact

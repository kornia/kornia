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

"""RANSAC speed and two-view pose quality on cached IMC correspondences.

Run from the checkout being measured (copy this harness to older checkouts).
Preparation/evaluation require optional h5py and the imc2021-simple package;
measurement needs only the normal Kornia environment and NumPy. No downloads.

    python benchmarks/geometry/ransac.py prepare --data-root ../imc2021-simple/data/phototourism \
        --scenes sacre_coeur,reichstag --cache sift=raw_matches.h5 \
        --cache xfeat=raw_matches_xfeat_n2048_lighterglue.h5 --pairs 20 --npz /tmp/pairs.npz
    python benchmarks/geometry/ransac.py run --npz /tmp/pairs.npz --device cuda \
        --batches 256,2048 --sample-budget 8192 --scores ransac,msac --prosac off,on \
        --lo-iters 0,5 --seeds 0,1,2 --json /tmp/predictions.json
    python benchmarks/geometry/ransac.py evaluate --npz /tmp/pairs.npz \
        --predictions /tmp/predictions.json --json /tmp/evaluated.json
    python benchmarks/geometry/ransac.py sweep --npz /tmp/pairs.npz --device cpu --json /tmp/sweep.json
    python benchmarks/geometry/ransac.py evaluate --npz /tmp/pairs.npz \
        --predictions /tmp/sweep.json --json /tmp/sweep-evaluated.json
    python benchmarks/geometry/ransac.py plot --evaluated /tmp/sweep-evaluated.json --out /tmp/time_maa.png
    git show main:kornia/geometry/ransac.py > /tmp/base_ransac.py
    python benchmarks/geometry/ransac.py compare --npz /tmp/pairs.npz --device cpu \
        --base-source /tmp/base_ransac.py --json /tmp/compare.json

Timing covers public RANSAC.forward on preloaded device tensors, including sampling,
scoring and local optimization, excluding construction, sorting, transfers and pose
recovery. A fixed estimator seed is reset by each forward; repeated timing calls
therefore measure the same work. Sample budgets count minimal samples, not solutions
(the seven-point solver can emit multiple solutions). Confidence stopping can reduce
actual work. Batch sizes must divide the budget. Quality is evaluated separately for
every seed, using imc2021.metrics.pose_error and maa_imc (strict thresholds 1..10 deg).
Failure means nonfinite pose error; finite errors >=10 deg also miss every mAA threshold.
Scene mAA is averaged equally, then seeds equally, separately for each feature/config.
Prepared pairs come from the intersection of requested raw feature caches per scene,
selected without replacement by a pinned NumPy RNG. Cached score (preferred)
and ratio both rank lower values first, following imc2021-simple. --timing-pairs
limits repeated timing per scene/feature while preserving all quality evaluations. This is a diagnostic subset, not
an official full IMC leaderboard result. Missing caches are reported and skipped.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "benchmarks"))

from common import finish_run, save_json, setup_run, start_run, time_us  # noqa: E402

from kornia.geometry import RANSAC  # noqa: E402


def digest(path: Path) -> str:
    """Identify exact input bytes without publishing a machine-local path."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args: argparse.Namespace) -> None:
    """Export raw matches and ground truth, never an upstream estimator's inliers."""
    try:
        import h5py
        from imc2021.io import load_calibration
    except ImportError as exc:
        raise SystemExit(f"Preparation requires h5py and imc2021-simple: {exc}") from exc
    payload: dict[str, Any] = {}
    sources = []
    rng = np.random.default_rng(args.selection_seed)
    for scene in args.scenes.split(","):
        scene_dir = args.data_root / scene / args.scene_set
        calib = load_calibration(scene_dir)
        caches = {}
        for spec in args.cache:
            label, filename = spec.split("=", 1)
            path = scene_dir / filename
            if not path.is_file():
                print(f"# NOTE: skipping missing cache {scene}/{filename}")
                continue
            with h5py.File(path, "r") as handle:
                pairs = {(a, b) for a in handle for b in handle[a] if a in calib and b in calib}
            caches[label] = (path, pairs)
        if not caches:
            continue
        common_pairs = sorted(set.intersection(*(pairs for _, pairs in caches.values())))
        indices = sorted(rng.choice(len(common_pairs), min(args.pairs, len(common_pairs)), replace=False))
        for label, (path, _) in caches.items():
            sources.append({"scene": scene, "feature": label, "file": path.name, "sha256": digest(path)})
            with h5py.File(path, "r") as handle:
                for index in indices:
                    a, b = common_pairs[index]
                    group = handle[a][b]
                    prefix = f"{scene}__{label}__{index}__"
                    if "mkp_a" not in group or "mkp_b" not in group:
                        raise ValueError(f"{path.name} is not a raw correspondence cache")
                    for field in ("mkp_a", "mkp_b", "ratio", "score"):
                        if field in group:
                            payload[prefix + field] = group[field][:]
                    payload[prefix + "names"] = np.array([a, b])
                    for side, name in (("a", a), ("b", b)):
                        for field in ("K", "R", "T"):
                            payload[prefix + side + "_" + field] = calib[name][field]
    if not payload:
        raise SystemExit("No pairs found in the requested caches")
    payload["provenance"] = np.array(json.dumps({"sources": sources, "selection_seed": args.selection_seed}))
    np.savez_compressed(args.npz, **payload)
    print(f"# exported {sum(key.endswith('__mkp_a') for key in payload)} feature/pair records to {args.npz}")


def pair_keys(data: Any, args: argparse.Namespace) -> list[str]:
    """Select exported pairs without changing their deterministic preparation order."""
    keys = sorted(key.removesuffix("__mkp_a") for key in data.files if key.endswith("__mkp_a"))
    if args.scenes:
        keys = [key for key in keys if key.split("__")[0] in args.scenes.split(",")]
    if args.features:
        keys = [key for key in keys if key.split("__")[1] in args.features.split(",")]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    selected = []
    for key in keys:
        group = tuple(key.split("__")[:2])
        if args.pairs is None or counts[group] < args.pairs:
            selected.append(key)
            counts[group] += 1
    return selected


def run(args: argparse.Namespace) -> None:
    """Measure full public estimator calls and preserve their outputs for evaluation."""
    device, dtype, sync = setup_run(args, opencv=False)
    meta = start_run(
        "RANSAC",
        args,
        device,
        units="pairs/s",
        regimes=["Preloaded device tensors; full RANSAC.forward including sampling, scoring and LO"],
    )
    meta.update(
        dataset_sha256=digest(args.npz),
        sample_budget=args.sample_budget,
        confidence=args.confidence,
        threshold_px=args.threshold,
        min_run_time=args.min_run_time,
        timing_pairs_per_scene_feature=args.timing_pairs,
        actual_work="sampled_sets counts minimal sets in untimed prediction, excluding solver roots",
        ransac_source_sha256=digest(ROOT / "kornia" / "geometry" / "ransac.py"),
        timing_includes="RANSAC.forward only; fixed estimator seed per repeated call",
    )
    configs = list(
        itertools.product(
            [int(x) for x in args.batches.split(",")],
            args.scores.split(","),
            args.prosac.split(","),
            [int(x) for x in args.lo_iters.split(",")],
            [int(x) for x in args.seeds.split(",")],
        )
    )
    if any(batch <= 0 or args.sample_budget % batch for batch, *_ in configs):
        raise SystemExit("Every positive batch size must divide --sample-budget exactly")
    rows = []
    with np.load(args.npz, allow_pickle=False) as data, torch.inference_mode():
        keys = pair_keys(data, args)
        if not keys:
            raise SystemExit("No pairs selected")
        timing_counts: dict[tuple[str, str], int] = defaultdict(int)
        for key in keys:
            scene, feature, _ = key.split("__")
            timed = args.timing_pairs is None or timing_counts[scene, feature] < args.timing_pairs
            timing_counts[scene, feature] += 1
            a, b = data[key + "__mkp_a"], data[key + "__mkp_b"]
            if key + "__score" in data:
                order = np.argsort(data[key + "__score"].reshape(-1), kind="stable")
                ranking = "ascending score"
            elif key + "__ratio" in data:
                order = np.argsort(data[key + "__ratio"].reshape(-1), kind="stable")
                ranking = "ascending ratio"
            else:
                order, ranking = np.arange(len(a)), "unranked"
            # Both samplers see identical order; PROSAC assumes best matches come first.
            kp1 = torch.as_tensor(a[order], device=device, dtype=dtype)
            kp2 = torch.as_tensor(b[order], device=device, dtype=dtype)
            for batch, score, prosac, lo, seed in configs:
                row: dict[str, Any] = {
                    "op": "RANSAC",
                    "backend": "kornia (eager)",
                    "batch": batch,
                    "dtype": args.dtype,
                    "pair": key,
                    "scene": scene,
                    "feature": feature,
                    "model_type": args.model,
                    "score_type": score,
                    "prosac": prosac == "on",
                    "max_lo_iters": lo,
                    "lo_sample_size": args.lo_sample_size,
                    "seed": seed,
                    "correspondences": len(a),
                    "ranking": ranking,
                    "timed": timed,
                    "median_us": None,
                    "iqr_us": None,
                    "throughput_per_s": None,
                }
                if prosac == "on" and ranking == "unranked":
                    row["error"] = "unavailable: PROSAC requires cached ratios or scores"
                else:
                    estimator = RANSAC(
                        model_type=args.model,
                        inl_th=args.threshold,
                        batch_size=batch,
                        max_iter=args.sample_budget // batch,
                        confidence=args.confidence,
                        max_lo_iters=lo,
                        score_type=score,
                        prosac_sampling=prosac == "on",
                        seed=seed,
                        **({"lo_sample_size": args.lo_sample_size} if args.lo_sample_size is not None else {}),
                    )
                    try:
                        original_sample = estimator.sample
                        sampled_sets = 0

                        def counted_sample(
                            *sample_args: Any, _sample: Any = original_sample, **sample_kwargs: Any
                        ) -> Any:
                            nonlocal sampled_sets
                            sampled_sets += int(sample_args[2] if len(sample_args) > 2 else sample_kwargs["batch_size"])
                            return _sample(*sample_args, **sample_kwargs)

                        estimator.sample = counted_sample
                        try:
                            matrix, mask = estimator(kp1, kp2)
                        finally:
                            estimator.sample = original_sample
                            row["sampled_sets"] = sampled_sets
                        if timed:
                            median, iqr = time_us(partial(estimator, kp1, kp2), args.min_run_time, sync=sync)
                            if not np.isfinite(median):
                                raise RuntimeError("Timing call failed")
                            row.update(median_us=median, iqr_us=iqr, throughput_per_s=1e6 / median)
                        original_mask = np.zeros(len(a), dtype=bool)
                        original_mask[order] = mask.detach().cpu().numpy().reshape(-1)
                        row.update(
                            matrix=matrix.detach().cpu().tolist(),
                            inlier_indices=np.flatnonzero(original_mask).tolist(),
                        )
                    except (RuntimeError, ValueError) as exc:
                        row["error"] = type(exc).__name__
                rows.append(row)
                print(
                    f"{key} batch={batch} {score} prosac={prosac} lo={lo} seed={seed}: {row['median_us']} us",
                    flush=True,
                )
    finish_run(args, "geometry-ransac", meta, rows)


def compare(args: argparse.Namespace) -> None:
    """Time another revision's RANSAC and this checkout's variants alternately in one process.

    Separate processes on a hybrid CPU can differ by up to 2x on identical code, so an A/B timing
    must share one warmed process. ``--base-source`` is that revision's ``kornia/geometry/ransac.py``
    (``git show <rev>:kornia/geometry/ransac.py``). It is loaded next to the current module and
    imports this checkout's kornia internals, so it is only valid when that file is the sole
    library difference between the revisions. Each call is timed ``--rounds`` times, alternating
    the variant order, and the minimum median is kept.
    """
    device, _, sync = setup_run(args, opencv=False)
    variants: dict[str, tuple[Any, dict[str, Any]]] = {"new": (RANSAC, {})}
    if args.base_source is not None:
        variants = {"base": (load_ransac_module(args.base_source).RANSAC, {}), **variants}
    variants["new-lo0"] = (RANSAC, {"max_lo_iters": 0})
    for cap in [int(x) for x in args.lo_sample_sizes.split(",") if x]:
        variants[f"new-lo{cap}-subsets"] = (RANSAC, {"lo_sample_size": cap})
    variants["new-prosac"] = (RANSAC, {"prosac_sampling": True})
    meta = start_run(
        "RANSAC A/B",
        args,
        device,
        units="ms",
        regimes=["One warmed process; variants alternate per round; minimum median per variant"],
    )
    meta.update(
        dataset_sha256=digest(args.npz),
        ransac_source_sha256=digest(ROOT / "kornia" / "geometry" / "ransac.py"),
        base_ransac_source_sha256=digest(args.base_source) if args.base_source is not None else None,
        sample_budget=args.sample_budget,
        confidence=args.confidence,
        threshold_px=args.threshold,
        rounds=args.rounds,
        min_run_time=args.min_run_time,
    )
    rows = []
    with np.load(args.npz, allow_pickle=False) as data, torch.inference_mode():
        for key in pair_keys(data, args):
            scene, feature, _ = key.split("__")
            a, b = data[key + "__mkp_a"], data[key + "__mkp_b"]
            field = key + "__score" if key + "__score" in data else key + "__ratio"
            order = np.argsort(data[field].reshape(-1), kind="stable") if field in data else np.arange(len(a))
            kp1 = torch.as_tensor(a[order], device=device, dtype=torch.float32)
            kp2 = torch.as_tensor(b[order], device=device, dtype=torch.float32)
            batches = [int(x) for x in args.batches.split(",")]
            seeds = [int(x) for x in args.seeds.split(",")]
            for score, batch, seed in itertools.product(args.scores.split(","), batches, seeds):
                names = list(variants)
                estimators = {
                    name: cls(
                        model_type="fundamental",
                        inl_th=args.threshold,
                        batch_size=batch,
                        max_iter=args.sample_budget // batch,
                        confidence=args.confidence,
                        score_type=score,
                        seed=seed,
                        **{"max_lo_iters": args.lo_iters, **options},
                    )
                    for name, (cls, options) in variants.items()
                }
                best = dict.fromkeys(names, float("inf"))
                for round_index in range(args.rounds):
                    for name in names if round_index % 2 == 0 else names[::-1]:
                        median, _ = time_us(partial(estimators[name], kp1, kp2), args.min_run_time, sync=sync)
                        best[name] = min(best[name], median)
                for name in names:
                    rows.append(
                        {
                            "op": "RANSAC",
                            "backend": name,
                            "pair": key,
                            "scene": scene,
                            "feature": feature,
                            "score_type": score,
                            "batch": batch,
                            "seed": seed,
                            "median_us": best[name],
                        }
                    )
            print(f"{key}: timed {len(variants)} variants", flush=True)
    finish_run(args, "geometry-ransac-compare", meta, rows)


def load_ransac_module(path: Path) -> Any:
    """Load another revision's ``kornia/geometry/ransac.py`` next to this checkout's module.

    It imports this checkout's kornia internals, so it is only valid when that file is the sole
    library difference between the revisions (``git show <rev>:kornia/geometry/ransac.py``).
    """
    spec = importlib.util.spec_from_file_location("kornia.geometry._ransac_other_revision", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SWEEP_KORNIA = {
    "msac": {"score_type": "msac"},
    "msac-lo32": {"score_type": "msac", "lo_sample_size": 32},
    "msac-prosac": {"score_type": "msac", "prosac_sampling": True},
    "ransac": {"score_type": "ransac"},
}
SWEEP_OPENCV = {"usac_magsac": "USAC_MAGSAC", "usac_accurate": "USAC_ACCURATE", "ransac": "FM_RANSAC"}


def sweep(args: argparse.Namespace) -> None:
    """Time single calls over a budget x inlier-threshold grid for the time-mAA curve.

    Every configuration runs once per pair, all configurations of one pair back to back, so slow
    drift affects every method alike. Each call is timed on its own with device synchronization,
    as in the IMC time-mAA protocol; the curve averages the times over pairs. Kornia's budget is
    ``batch_size * max_iter`` minimal sets (a budget below ``--batch`` runs as one batch of that
    size); OpenCV's is ``maxIters``. Predictions are stored with bit-packed inlier masks and scored
    by ``evaluate``.
    """
    kornia_methods = [m for m in args.kornia.split(",") if m]
    opencv_methods = [m for m in args.opencv.split(",") if m]
    device, _, _ = setup_run(args, opencv=bool(opencv_methods))
    estimators: list[tuple[str, Any]] = [("kornia", RANSAC)]
    if args.base_source is not None:
        # Another revision's RANSAC next to this one, interleaved per pair in the same process.
        estimators.append(("kornia-base", load_ransac_module(args.base_source).RANSAC))
    cv2 = None
    if opencv_methods:
        import cv2
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    thresholds = [float(x) for x in args.thresholds.split(",")]
    meta = start_run(
        "RANSAC time-mAA sweep",
        args,
        device,
        units="ms",
        regimes=["Single synchronized call per pair and configuration; inputs preloaded on the device"],
    )
    meta.update(
        dataset_sha256=digest(args.npz),
        ransac_source_sha256=digest(ROOT / "kornia" / "geometry" / "ransac.py"),
        base_ransac_source_sha256=digest(args.base_source) if args.base_source is not None else None,
        thresholds_px=thresholds,
        confidence=args.confidence,
        seed=args.seed,
    )
    rows = []
    with np.load(args.npz, allow_pickle=False) as data, torch.inference_mode():
        for pair_index, key in enumerate(pair_keys(data, args)):
            scene, feature, _ = key.split("__")
            a, b = data[key + "__mkp_a"], data[key + "__mkp_b"]
            field = key + "__score" if key + "__score" in data else key + "__ratio"
            order = np.argsort(data[field].reshape(-1), kind="stable") if field in data else np.arange(len(a))
            kp1 = torch.as_tensor(a[order], device=device, dtype=torch.float32)
            kp2 = torch.as_tensor(b[order], device=device, dtype=torch.float32)
            points1, points2 = a[order].astype(np.float64), b[order].astype(np.float64)
            calls: list[tuple[dict[str, Any], Any]] = []
            for (prefix, ransac_class), name, budget, threshold in itertools.product(
                estimators, kornia_methods, [int(x) for x in args.budgets.split(",")], thresholds
            ):
                batch = min(args.batch, budget)
                estimator = ransac_class(
                    model_type="fundamental",
                    inl_th=threshold,
                    batch_size=batch,
                    max_iter=budget // batch,
                    confidence=args.confidence,
                    max_lo_iters=args.lo_iters,
                    seed=args.seed,
                    **SWEEP_KORNIA[name],
                )
                config = {"method": f"{prefix} {name}", "device": device.type, "batch": batch}
                config.update(sample_budget=budget, threshold_px=threshold)
                calls.append((config, partial(estimator, kp1, kp2)))
            for name, iters, threshold in itertools.product(
                opencv_methods, [int(x) for x in args.opencv_iters.split(",")], thresholds
            ):

                def estimate(
                    flag: int = getattr(cv2, SWEEP_OPENCV[name]),
                    iters: int = iters,
                    th: float = threshold,
                    p1: Any = points1,
                    p2: Any = points2,
                ) -> Any:
                    cv2.setRNGSeed(args.seed)
                    return cv2.findFundamentalMat(p1, p2, flag, th, args.confidence, iters)

                config = {"method": f"opencv {name}", "device": "cpu", "batch": None}
                config.update(sample_budget=iters, threshold_px=threshold)
                calls.append((config, estimate))
            if pair_index == 0:
                for _, call in calls:  # lazy initialization and solver handles stay out of the timings
                    with contextlib.suppress(Exception):
                        call()
            for config, call in calls:
                error = None
                sync()
                start = time.perf_counter()
                try:
                    matrix, mask = call()
                except Exception as exc:  # e.g. an OpenCV USAC assertion: scored as a failed estimate
                    matrix, mask, error = None, None, type(exc).__name__
                sync()
                elapsed_ms = (time.perf_counter() - start) * 1000
                row: dict[str, Any] = {
                    "op": "RANSAC",
                    "backend": config["method"],
                    "pair": key,
                    "scene": scene,
                    "feature": feature,
                    "model_type": "fundamental",
                    "seed": args.seed,
                    "correspondences": len(a),
                    "time_ms": elapsed_ms,
                    **config,
                }
                if error is not None:
                    row["error"] = error
                if torch.is_tensor(matrix):
                    matrix, mask = matrix.cpu().numpy(), mask.cpu().numpy()
                if matrix is not None and np.shape(matrix) == (3, 3) and np.abs(matrix).max() > 0:
                    original = np.zeros(len(a), dtype=bool)
                    original[order] = np.asarray(mask).reshape(-1).astype(bool)
                    packed = base64.b64encode(np.packbits(original).tobytes()).decode()
                    row.update(matrix=np.asarray(matrix).tolist(), inliers_packed=packed)
                rows.append(row)
            print(f"{key}: {len(calls)} configurations", flush=True)
    finish_run(args, "geometry-ransac-sweep", meta, rows)


# Line style separates the families: kornia solid in color (one validated categorical hue per
# configuration, checked all-pairs), another kornia revision dashed with hollow markers in the same
# hue, OpenCV in neutral greys with its own dash patterns. Markers differ within each family too,
# so identity never rests on color alone.
PLOT_SERIES = {  # method: (label, group, color, linestyle, marker, filled)
    "kornia msac": ("MSAC", "kornia", "#2a78d6", "-", "o", True),
    "kornia ransac": ("RANSAC score", "kornia", "#eb6834", "-", "s", True),
    "kornia msac-lo32": ("MSAC + subset LO (32)", "kornia", "#1baf7a", "-", "D", True),
    "kornia msac-prosac": ("MSAC + PROSAC", "kornia", "#4a3aa7", "-", "^", True),
    "kornia-base msac": ("MSAC", "kornia-base", "#2a78d6", "--", "o", False),
    "kornia-base ransac": ("RANSAC score", "kornia-base", "#eb6834", "--", "s", False),
    "opencv usac_magsac": ("USAC_MAGSAC (MAGSAC++)", "opencv", "#1f1f1e", ":", "s", True),
    "opencv usac_accurate": ("USAC_ACCURATE", "opencv", "#5f5e5a", "-.", "D", True),
    "opencv ransac": ("FM_RANSAC", "opencv", "#8f8e89", "--", "v", True),
}
PLOT_GROUPS = {"kornia": "kornia, this PR", "kornia-base": "kornia, main before this PR", "opencv": "OpenCV (CPU)"}
PAGE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style_axis(axis: Any) -> None:
    """Log-scale x axis with a recessive grid and no top/right spines."""
    axis.set_facecolor(PAGE)
    axis.set_xscale("log")
    axis.grid(True, which="major", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=8)


def draw_series(axis: Any, method: str, xs: list[float], ys: list[float], handles: dict[str, dict[str, Any]]) -> None:
    """Draw one method's line in its family style and remember the handle for the grouped legend."""
    label, group, color, linestyle, marker, filled = PLOT_SERIES[method]
    (line,) = axis.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        linewidth=2 if group != "opencv" else 1.6,
        marker=marker,
        markersize=5.5,
        markerfacecolor=color if filled else PAGE,
        markeredgecolor=color if not filled else PAGE,
        markeredgewidth=1.4 if not filled else 0.8,
        label=label,
    )
    handles.setdefault(group, {}).setdefault(label, line)


def grouped_legend(figure: Any, handles: dict[str, dict[str, Any]], y: float) -> None:
    """One titled legend per family, side by side below the panels."""
    groups = [g for g in PLOT_GROUPS if g in handles]
    for index, group in enumerate(groups):
        legend = figure.legend(
            list(handles[group].values()),
            list(handles[group]),
            title=PLOT_GROUPS[group],
            loc="upper center",
            bbox_to_anchor=((index + 0.5) / len(groups), y),
            fontsize=8.5,
            title_fontsize=9,
            frameon=False,
            labelcolor=INK,
            handlelength=3.2,
        )
        legend.get_title().set_color(INK)


def plot_thresholds(plt: Any, configs: dict[tuple[Any, ...], dict[str, Any]], features: list[str], args: Any) -> None:
    """Pose mAA against the inlier threshold on the CPU sweep, every method at its largest budget."""
    figure, axes = plt.subplots(1, len(features), figsize=(6.4 * len(features), 5.4), squeeze=False, facecolor=PAGE)
    thresholds = sorted({key[4] for key in configs})
    handles: dict[str, dict[str, Any]] = {}
    for axis, feature in zip(axes[0], features):
        for method in PLOT_SERIES:
            budgets = [key[3] for key in configs if key[:3] == (feature, method, "cpu")]
            if not budgets:
                continue
            largest = (feature, method, "cpu", max(budgets))
            points = sorted((key[4], value["maa"]) for key, value in configs.items() if key[:4] == largest)
            draw_series(axis, method, [t for t, _ in points], [m for _, m in points], handles)
        style_axis(axis)
        axis.set_xticks(thresholds, [f"{t:g}" for t in thresholds])
        axis.minorticks_off()
        axis.set_title({"sift": "SIFT", "xfeat": "XFeat"}.get(feature, feature), color=INK)
        axis.set_xlabel("inlier threshold (px, log scale)", color=MUTED, fontsize=9)
        axis.set_ylabel("pose mAA (1-10°)", color=MUTED, fontsize=9)
    figure.suptitle(args.title or "Pose accuracy vs. inlier threshold (CPU, largest budget)", color=INK)
    grouped_legend(figure, handles, 0.2)
    figure.text(
        0.5,
        0.005,
        "Each method at its largest budget: kornia 16384 minimal sample sets, OpenCV maxIters 25600.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.2, 1, 1))
    figure.savefig(args.threshold_out, dpi=args.dpi, facecolor=PAGE)
    print(f"# figure written to {args.threshold_out}")


def plot(args: argparse.Namespace) -> None:
    """Draw the IMC-style time-mAA curve from evaluated sweeps: best threshold per budget point."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    configs: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in args.evaluated:
        for summary in json.loads(path.read_text())["metadata"]["quality_summary"]:
            key = tuple(summary.get(f) for f in ("feature", "method", "device", "sample_budget", "threshold_px"))
            if key in configs:
                raise SystemExit(f"{key} appears in more than one evaluated file")
            configs[key] = summary
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for (feature, method, device, budget, _threshold), summary in configs.items():
        point = best.setdefault((feature, method, device, budget), summary)
        if summary["maa"] > point["maa"]:
            best[feature, method, device, budget] = summary
    curves: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for (feature, method, device, budget), summary in sorted(best.items(), key=lambda item: item[0][3]):
        curves[feature, device][method].append(
            {
                "budget": budget,
                "threshold_px": summary["threshold_px"],
                "maa": summary["maa"],
                "mean_time_ms": summary["mean_time_ms"],
            }
        )
    features = sorted({feature for feature, _ in curves})
    devices = [
        d
        for d in ("cpu", "cuda")
        if any(
            device == d and any(not m.startswith("opencv") for m in methods) for (_, device), methods in curves.items()
        )
    ]
    figure, axes = plt.subplots(
        len(devices),
        len(features),
        figsize=(6.4 * len(features), 4.6 * len(devices) + 1.4),
        squeeze=False,
        facecolor=PAGE,
    )
    handles: dict[str, dict[str, Any]] = {}
    for row, device in enumerate(devices):
        for column, feature in enumerate(features):
            axis = axes[row][column]
            lines = dict(curves[feature, device])
            if device != "cpu":  # OpenCV runs on the CPU; repeat it as the reference in every row
                lines.update({m: p for m, p in curves.get((feature, "cpu"), {}).items() if m.startswith("opencv")})
            for method in PLOT_SERIES:
                if method in lines:
                    points = sorted(lines[method], key=lambda p: p["mean_time_ms"])
                    draw_series(axis, method, [p["mean_time_ms"] for p in points], [p["maa"] for p in points], handles)
            style_axis(axis)
            name = {"sift": "SIFT", "xfeat": "XFeat"}.get(feature, feature)
            axis.set_title(f"{name}, kornia on {device.upper()}", color=INK)
            axis.set_xlabel("mean time per pair (ms, log scale)", color=MUTED, fontsize=9)
            axis.set_ylabel("pose mAA (1-10°)", color=MUTED, fontsize=9)
    figure.suptitle(args.title or "Fundamental matrix: pose accuracy vs. time", color=INK, fontsize=12.5)
    legend_space = 1.4 / (4.6 * len(devices) + 1.4)
    grouped_legend(figure, handles, legend_space)
    note = (
        "Each point is one budget at its best inlier threshold from the sweep; "
        "mean over pairs of single synchronized calls."
    )
    if "opencv" in handles:
        note += " OpenCV (maxIters budget) runs on the CPU and is repeated in the CUDA row."
    figure.text(0.5, 0.004, note, ha="center", fontsize=8, color=MUTED)
    figure.tight_layout(rect=(0, legend_space, 1, 1))
    figure.savefig(args.out, dpi=args.dpi, facecolor=PAGE)
    print(f"# figure written to {args.out}")
    if args.points_json is not None:
        points_out = {f"{f}|{d}": dict(methods) for (f, d), methods in curves.items()}
        args.points_json.write_text(json.dumps(points_out, indent=2, sort_keys=True) + "\n")
    if args.threshold_out is not None:
        plot_thresholds(plt, configs, features, args)


CONFIG_FIELDS = (
    "feature",
    "model_type",
    "batch",
    "score_type",
    "prosac",
    "max_lo_iters",
    "lo_sample_size",
    "method",
    "device",
    "sample_budget",
    "threshold_px",
)


def evaluate(args: argparse.Namespace) -> None:
    """Use the companion evaluator's exact pose-error and mAA implementations."""
    try:
        from imc2021 import metrics
    except ImportError as exc:
        raise SystemExit(f"Evaluation requires imc2021-simple and its optional dependencies: {exc}") from exc
    # Several prediction files (e.g. one sweep per seed) pool into one evaluation: each seed is its
    # own scene/seed cell, so the mAA averages over seeds as well as scenes.
    documents = [json.loads(path.read_text()) for path in args.predictions]
    dataset = digest(args.npz)
    if any(doc["metadata"]["dataset_sha256"] != dataset for doc in documents):
        raise SystemExit("NPZ content differs from the measured dataset")
    document = documents[0]
    document["results"] = [row for doc in documents for row in doc["results"]]
    if args.thresholds is not None:
        keep = {float(x) for x in args.thresholds.split(",")}
        document["results"] = [r for r in document["results"] if r.get("threshold_px") in (None, *keep)]
    if len(documents) > 1:
        document["metadata"]["pooled_predictions"] = [doc["metadata"] for doc in documents[1:]]
    groups: dict[tuple[Any, ...], dict[tuple[str, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    times: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    with np.load(args.npz, allow_pickle=False) as data:
        for row in document["results"]:
            key = row["pair"]
            errors = dict.fromkeys(("R_err", "t_err", "max_err"), float("inf"))
            # An all-zero matrix is RANSAC's failure result, whichever command wrote it.
            if "matrix" in row and np.isfinite(row["matrix"]).all() and np.abs(row["matrix"]).max() > 0:
                if "inliers_packed" in row:
                    packed = np.frombuffer(base64.b64decode(row["inliers_packed"]), dtype=np.uint8)
                    inliers = np.flatnonzero(np.unpackbits(packed)[: row["correspondences"]])
                else:
                    inliers = np.array(row["inlier_indices"], dtype=int)
                calib = [{field: data[f"{key}__{side}_{field}"] for field in ("K", "R", "T")} for side in ("a", "b")]
                errors = metrics.pose_error(
                    np.asarray(row["matrix"]),
                    data[key + "__mkp_a"][inliers],
                    data[key + "__mkp_b"][inliers],
                    *calib,
                )
            row.update(errors)
            row["failed"] = not np.isfinite(errors["max_err"])
            config = tuple(row.get(field) for field in CONFIG_FIELDS)
            groups[config][row["scene"], row["seed"]].append(errors["max_err"])
            if row.get("time_ms") is not None:
                times[config].append(row["time_ms"])
    summaries = []
    for config, cells in groups.items():
        cell_results = [
            {
                "scene": scene,
                "seed": seed,
                "pairs": len(errors),
                "maa": metrics.maa_imc(errors, list(range(1, 11))),
                "failure_rate": float(np.mean(~np.isfinite(errors))),
            }
            for (scene, seed), errors in sorted(cells.items())
        ]
        summary = {field: value for field, value in zip(CONFIG_FIELDS, config) if value is not None}
        summary.update(
            maa=float(np.mean([cell["maa"] for cell in cell_results])),
            failure_rate=float(np.mean([cell["failure_rate"] for cell in cell_results])),
            mean_time_ms=float(np.mean(times[config])) if times[config] else None,
            cells=cell_results,
        )
        summaries.append(summary)
        print(json.dumps(summary))
    meta = document["metadata"]
    meta.update(
        quality_summary=summaries,
        pose_evaluator="imc2021.metrics.pose_error",
        evaluator_sha256=digest(Path(metrics.__file__)),
        maa_thresholds_deg=list(range(1, 11)),
        aggregation="equal-weight scene/seed cells per feature/config; failures retained",
    )
    save_json(args.json, meta, document["results"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--scenes", required=True)
    prep.add_argument("--scene-set", default="set_100")
    prep.add_argument("--cache", action="append", required=True, help="label=raw_cache_filename.h5 (repeatable)")
    prep.add_argument("--pairs", type=int, default=20, help="pairs per scene, shared across features")
    prep.add_argument("--selection-seed", type=int, default=0)
    measure = commands.add_parser("run")
    measure.add_argument("--device", default="cpu")
    measure.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    measure.add_argument("--threads", type=int, default=4)
    measure.add_argument("--batches", default="256,2048")
    measure.add_argument("--sample-budget", type=int, default=8192)
    measure.add_argument("--scores", default="ransac,msac")
    measure.add_argument("--prosac", default="off,on")
    measure.add_argument("--lo-iters", default="0,5")
    measure.add_argument(
        "--lo-sample-size", type=int, help="optional bounded LO subset; omitted for base compatibility"
    )
    measure.add_argument("--seeds", default="0,1,2")
    measure.add_argument("--model", choices=("fundamental", "fundamental_7pt"), default="fundamental")
    measure.add_argument("--threshold", type=float, default=1.0)
    measure.add_argument("--confidence", type=float, default=0.999)
    measure.add_argument("--min-run-time", type=float, default=0.2)
    measure.add_argument("--scenes")
    measure.add_argument("--features")
    measure.add_argument("--pairs", type=int, help="limit pairs per scene/feature")
    measure.add_argument("--timing-pairs", type=int, help="time only this many pairs per scene/feature; score all")
    measure.add_argument("--json", type=Path, required=True)
    measure.set_defaults(contribute=None)
    sw = commands.add_parser("sweep")
    sw.add_argument("--device", default="cpu")
    sw.add_argument("--dtype", choices=("float32",), default="float32")
    sw.add_argument("--threads", type=int, default=4)
    sw.add_argument("--kornia", default=",".join(SWEEP_KORNIA), help=f"subset of {','.join(SWEEP_KORNIA)}")
    sw.add_argument("--opencv", default=",".join(SWEEP_OPENCV), help=f"subset of {','.join(SWEEP_OPENCV)}")
    sw.add_argument("--base-source", type=Path, help="also sweep another revision's ransac.py as 'kornia-base'")
    sw.add_argument("--budgets", default="256,512,1024,2048,4096,8192,16384", help="kornia minimal-sample budgets")
    sw.add_argument("--batch", type=int, default=256, help="kornia batch size (smaller budgets use one batch)")
    sw.add_argument("--opencv-iters", default="10,25,100,400,1600,6400,25600")
    sw.add_argument("--thresholds", default="0.25,0.5,0.75,1,1.5,2")
    sw.add_argument("--confidence", type=float, default=0.999)
    sw.add_argument("--lo-iters", type=int, default=5)
    sw.add_argument("--seed", type=int, default=0)
    sw.add_argument("--scenes")
    sw.add_argument("--features")
    sw.add_argument("--pairs", type=int, help="limit pairs per scene/feature")
    sw.add_argument("--json", type=Path, required=True)
    sw.set_defaults(contribute=None)
    pl = commands.add_parser("plot")
    pl.add_argument("--evaluated", type=Path, nargs="+", required=True)
    pl.add_argument("--out", type=Path, required=True)
    pl.add_argument("--dpi", type=int, default=150)
    pl.add_argument("--points-json", type=Path)
    pl.add_argument("--threshold-out", type=Path, help="also draw mAA against the inlier threshold")
    pl.add_argument("--title", help="figure title")
    ab = commands.add_parser("compare")
    ab.add_argument("--device", default="cpu")
    ab.add_argument("--dtype", choices=("float32",), default="float32")
    ab.add_argument("--threads", type=int, default=4)
    ab.add_argument("--base-source", type=Path, help="another revision's kornia/geometry/ransac.py")
    ab.add_argument("--batches", default="256,2048")
    ab.add_argument("--sample-budget", type=int, default=4096)
    ab.add_argument("--scores", default="msac,ransac")
    ab.add_argument("--lo-iters", type=int, default=5)
    ab.add_argument("--lo-sample-sizes", default="32,128", help="bounded-LO variants of this checkout")
    ab.add_argument("--seeds", default="0,1,2")
    ab.add_argument("--threshold", type=float, default=2.0)
    ab.add_argument("--confidence", type=float, default=0.999)
    ab.add_argument("--min-run-time", type=float, default=0.1)
    ab.add_argument("--rounds", type=int, default=3)
    ab.add_argument("--scenes")
    ab.add_argument("--features")
    ab.add_argument("--pairs", type=int, default=2, help="pairs per scene/feature")
    ab.add_argument("--json", type=Path, required=True)
    ab.set_defaults(contribute=None)
    score = commands.add_parser("evaluate")
    score.add_argument("--predictions", type=Path, nargs="+", required=True, help="pooled, e.g. one file per seed")
    score.add_argument("--thresholds", help="keep only sweep rows at these inlier thresholds")
    score.add_argument("--json", type=Path, required=True)
    for command in (prep, measure, sw, ab, score):
        command.add_argument("--npz", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.lo_sample_size is not None and args.lo_sample_size <= 0:
            parser.error("--lo-sample-size must be positive")
        if args.timing_pairs is not None and args.timing_pairs < 0:
            parser.error("--timing-pairs must be nonnegative")
        if set(args.scores.split(",")) - {"ransac", "msac"}:
            parser.error("--scores accepts ransac,msac")
        if set(args.prosac.split(",")) - {"off", "on"}:
            parser.error("--prosac accepts off,on")
        if args.sample_budget <= 0 or args.min_run_time <= 0 or args.threads <= 0:
            parser.error("sample budget, minimum run time and thread count must be positive")
        if any(int(value) < 0 for value in args.lo_iters.split(",")):
            parser.error("--lo-iters must be nonnegative")
    if args.command == "compare":
        if set(args.scores.split(",")) - {"ransac", "msac"}:
            parser.error("--scores accepts ransac,msac")
        batches = [int(x) for x in args.batches.split(",")]
        if any(batch <= 0 or args.sample_budget % batch for batch in batches):
            parser.error("Every positive batch size must divide --sample-budget exactly")
        if args.rounds <= 0 or args.min_run_time <= 0 or args.threads <= 0 or args.lo_iters < 0:
            parser.error("rounds, minimum run time and thread count must be positive; --lo-iters nonnegative")
    if getattr(args, "pairs", None) is not None and args.pairs <= 0:
        parser.error("--pairs must be positive")
    if args.command == "sweep":
        unknown = set(args.kornia.split(",")) - set(SWEEP_KORNIA) | set(args.opencv.split(",")) - set(SWEEP_OPENCV)
        if unknown - {""}:
            parser.error(f"unknown --kornia or --opencv method: {sorted(unknown - {''})}")
        if args.batch <= 0 or any(b % min(b, args.batch) for b in (int(x) for x in args.budgets.split(","))):
            parser.error("every --budgets entry must be a multiple of --batch or smaller than it")
    commands_by_name = {"prepare": prepare, "run": run, "sweep": sweep, "plot": plot}
    commands_by_name.update(compare=compare, evaluate=evaluate)
    commands_by_name[args.command](args)


if __name__ == "__main__":
    main()

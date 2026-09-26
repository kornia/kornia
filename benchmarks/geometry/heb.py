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

"""Homography RANSAC time versus accuracy on a scene of the Homography Estimation Benchmark (HEB).

HEB (Barath et al., "Large-scale homography benchmark", CVPR 2023, https://github.com/danini/homography-benchmark)
ships SIFT correspondences with second-nearest-neighbour ratios and ground-truth inlier flags per image pair in
one HDF5 file per scene. Download one scene (``NYC_Library_homographies.h5``, 2.5 GB, from the dataset link in
that README) and run from the checkout being measured, with optional h5py:

    python benchmarks/geometry/heb.py run --h5 /data/heb/NYC_Library_homographies.h5 --device cuda \
        --batches auto --budgets 2048,8192,32768 --lo 0,5 --prosac off,on --pairs 200 --json /tmp/heb-cuda.json
    python benchmarks/geometry/heb.py plot --json this=/tmp/heb-cuda.json --json main=/tmp/heb-cuda-main.json \
        --out /tmp/heb-cuda.png

Protocol, as in the benchmark's own ``test_kornia.py``: correspondences with a ratio below ``--snn`` (0.8) are
kept and sorted best-first for PROSAC, the homography is estimated once per configuration, pair and seed on
preloaded device tensors, and its accuracy is the mean reprojection error of the ground-truth inliers, turned
into a mean average accuracy over ten thresholds spaced logarithmically from 1 to 20 px. Estimates with fewer
than four inliers, or non-finite ones, count as failures. Timing covers ``RANSAC.forward`` only, through
``common.time_us`` (median of many synchronized calls after warm-up), on every scored pair unless
``--timing-pairs`` limits it. The metric here is the benchmark's reprojection mAA; its homography-decomposition
pose error is not used, because on this data the ground-truth homography itself scores 16 degrees of
translation error.

The scene is hard: the median pair keeps 485 correspondences of which 3% are ground-truth inliers, so uniform
sampling never stops early and accuracy is bought with the draw budget. Configurations sweep the sampler,
the batch size (integers or ``auto``), the draw budget (``max_samples``) and local optimization; the
inlier threshold, confidence and scoring definition are fixed across configurations. Frontier curves are
empirical envelopes over one seed per pair unless ``--seeds`` lists more.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "benchmarks"))

from common import finish_run, setup_run, start_run, time_us  # noqa: E402

from kornia.geometry import RANSAC  # noqa: E402

MAA_THRESHOLDS_PX = np.logspace(np.log2(1.0), np.log2(20.0), 10, base=2.0)
FAILURE = 1e10


def reprojection_error(points1: np.ndarray, points2: np.ndarray, homography: np.ndarray) -> float:
    """Mean distance, in pixels, between ``H @ points1`` and ``points2``; the benchmark's accuracy measure."""
    projected = np.column_stack((points1, np.ones(len(points1)))) @ homography.T
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = projected[:, :2] / projected[:, 2:3]
    error = np.sqrt(np.sum((projected - points2) ** 2, axis=1))
    return float(np.mean(error)) if np.all(np.isfinite(error)) else FAILURE


def mean_average_accuracy(errors: np.ndarray) -> float:
    """Mean over thresholds of the fraction of pairs whose error is at most the threshold."""
    return float(np.mean([(errors <= threshold).mean() for threshold in MAA_THRESHOLDS_PX]))


def load_pairs(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(f"Reading HEB needs h5py: {exc}") from exc
    with h5py.File(args.h5, "r") as handle:
        names = sorted(key[5:] for key in handle if key.startswith("corr_"))
        rng = np.random.default_rng(args.pair_seed)
        chosen = sorted(rng.choice(len(names), size=min(args.pairs, len(names)), replace=False))
        return {names[index]: {"matches": handle[f"corr_{names[index]}"][()]} for index in chosen}


def parse_batches(spec: str) -> list[int | str]:
    return [item if item == "auto" else int(item) for item in spec.split(",")]


def make_estimator(args: argparse.Namespace, config: dict[str, Any], seed: int) -> RANSAC:
    """Build the estimator; older revisions without ``max_samples`` get the budget as whole batches."""
    common: dict[str, Any] = {
        "inl_th": args.threshold,
        "confidence": args.confidence,
        "max_lo_iters": config["max_lo_iters"],
        "score_type": args.score,
        "prosac_sampling": config["prosac"],
        "seed": seed,
    }
    try:
        return RANSAC("homography", batch_size=config["batch_size"], max_samples=config["budget"], **common)
    except TypeError:
        if config["batch_size"] == "auto":
            raise SystemExit("This revision has no auto batch size; pass integer --batches") from None
        return RANSAC(
            "homography",
            batch_size=config["batch_size"],
            max_iter=-(-config["budget"] // config["batch_size"]),
            **common,
        )


def run(args: argparse.Namespace) -> None:
    device, dtype, sync = setup_run(args, opencv=False)
    meta = start_run(
        "RANSAC on HEB",
        args,
        device,
        units="pairs/s",
        regimes=["Preloaded device tensors sorted best-first; full RANSAC.forward including sampling, scoring and LO"],
    )
    meta.update(
        h5=Path(args.h5).name,
        pairs=args.pairs,
        pair_seed=args.pair_seed,
        snn=args.snn,
        threshold_px=args.threshold,
        confidence=args.confidence,
        score=args.score,
        min_run_time=args.min_run_time,
        maa_thresholds_px=MAA_THRESHOLDS_PX.tolist(),
        accuracy="mean reprojection error of the ground-truth inliers, mAA over 1-20 px; failures count as misses",
    )
    data = load_pairs(args)
    inputs = {}
    for name, item in data.items():
        matches = item["matches"]
        keep = matches[:, 8] < args.snn
        order = np.argsort(matches[keep, 8], kind="stable")
        inputs[name] = tuple(
            torch.as_tensor(matches[keep][order, column : column + 2], device=device, dtype=dtype) for column in (0, 2)
        )
    configs = [
        {"prosac": prosac == "on", "batch_size": batch, "budget": budget, "max_lo_iters": lo}
        for prosac, batch, budget, lo in itertools.product(
            args.prosac.split(","),
            parse_batches(args.batches),
            map(int, args.budgets.split(",")),
            map(int, args.lo.split(",")),
        )
        if batch == "auto" or budget >= batch
    ]
    seeds = [int(seed) for seed in args.seeds.split(",")]
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        errors = []
        for seed in seeds:
            estimator = make_estimator(args, config, seed).to(device)
            for count, (name, (kp1, kp2)) in enumerate(inputs.items()):
                row: dict[str, Any] = {**config, "pair": name, "seed": seed, "correspondences": len(kp1)}
                if len(kp1) < 4:
                    row.update(inliers=0, error_px=FAILURE, median_us=None, iqr_us=None)
                    rows.append(row)
                    errors.append(FAILURE)
                    continue
                batches = []
                sample = estimator.sample
                estimator.sample = lambda m, n, batch, *a, _s=sample, _b=batches, **k: (
                    _b.append(batch) or _s(m, n, batch, *a, **k)
                )
                try:
                    with torch.inference_mode():
                        model, mask = estimator(kp1, kp2)
                finally:
                    estimator.sample = sample
                inliers = int(mask.sum())
                matches = data[name]["matches"]
                gt = matches[:, 9].astype(bool)
                error = FAILURE
                if inliers >= 4 and gt.sum() > 1:
                    error = reprojection_error(
                        matches[gt, :2], matches[gt, 2:4], model.detach().cpu().numpy().astype(np.float64)
                    )
                errors.append(error)
                median = iqr = None
                if args.timing_pairs is None or count < args.timing_pairs:
                    with torch.inference_mode():
                        median, iqr = time_us(partial(estimator, kp1, kp2), args.min_run_time, sync=sync)
                row.update(
                    inliers=inliers,
                    sampled_sets=int(sum(batches)),
                    batches=len(batches),
                    error_px=error,
                    median_us=median,
                    iqr_us=iqr,
                )
                rows.append(row)
        timed = [
            row["median_us"]
            for row in rows
            if row["median_us"] is not None and all(row[k] == config[k] for k in config)
        ]
        maa = mean_average_accuracy(np.array(errors))
        mean_ms = np.mean(timed) / 1000 if timed else float("nan")
        print(
            f"[{index + 1}/{len(configs)}] prosac={config['prosac']!s:5} batch={config['batch_size']!s:5} "
            f"budget={config['budget']:6} lo={config['max_lo_iters']}: mAA={maa:.4f} mean {mean_ms:.2f} ms/pair",
            flush=True,
        )
    finish_run(args, "geometry-heb", meta, rows)


def summarize(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-configuration mAA and mean timed latency from a ``run`` output."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in document["results"]:
        groups[(row["prosac"], str(row["batch_size"]), row["budget"], row["max_lo_iters"])].append(row)
    summaries = []
    for (prosac, batch, budget, lo), items in sorted(groups.items()):
        timed = [row["median_us"] for row in items if row.get("median_us") is not None]
        summaries.append(
            {
                "prosac": prosac,
                "batch_size": batch,
                "budget": budget,
                "max_lo_iters": lo,
                "maa": mean_average_accuracy(np.array([row["error_px"] for row in items])),
                "mean_ms": float(np.mean(timed)) / 1000 if timed else None,
                "mean_sampled_sets": float(np.mean([row.get("sampled_sets", 0) for row in items])),
                "pairs": len(items),
            }
        )
    return summaries


def plot(args: argparse.Namespace) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {False: "#eb6834", True: "#2a78d6"}
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(7, 5))
    for (label, path), style in zip((spec.split("=", 1) for spec in args.json), itertools.cycle(styles)):
        summaries = [s for s in summarize(json.loads(Path(path).read_text())) if s["mean_ms"] is not None]
        for prosac in (False, True):
            points = [(s["mean_ms"], s["maa"], s["max_lo_iters"]) for s in summaries if s["prosac"] == prosac]
            if not points:
                continue
            for ms, maa, lo in points:
                ax.plot(
                    ms,
                    maa,
                    marker="o" if lo == 0 else "s",
                    ms=6,
                    color=colors[prosac],
                    mfc=colors[prosac] if style == "-" else "white",
                    mew=1.5,
                    ls="none",
                )
            best, frontier = -1.0, []
            for ms, maa, _ in sorted(points):
                if maa > best:
                    frontier.append((ms, maa))
                    best = maa
            ax.plot(
                *zip(*frontier), ls=style, color=colors[prosac], label=f"{'PROSAC' if prosac else 'uniform'}, {label}"
            )
    ax.set_xscale("log")
    ax.set_xlabel("mean RANSAC time per pair (ms)")
    ax.set_ylabel("mAA, reprojection error 1-20 px")
    ax.grid(True, color="#e6e5e1")
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], marker="o", ls="none", color="gray", mfc="none"),
        Line2D([], [], marker="s", ls="none", color="gray", mfc="none"),
    ]
    labels += ["no local optimization", "local optimization"]
    ax.legend(handles, labels, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"# wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("run")
    measure.add_argument("--h5", type=Path, required=True, help="one HEB scene, e.g. NYC_Library_homographies.h5")
    measure.add_argument("--pairs", type=int, default=200, help="random pairs of the scene to use")
    measure.add_argument("--pair-seed", type=int, default=0)
    measure.add_argument("--device", default="cpu")
    measure.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    measure.add_argument("--threads", type=int, default=1)
    measure.add_argument("--snn", type=float, default=0.8, help="keep correspondences with a ratio below this")
    measure.add_argument("--threshold", type=float, default=1.0, help="inlier threshold in pixels")
    measure.add_argument("--confidence", type=float, default=0.99)
    measure.add_argument("--score", choices=("ransac", "msac"), default="msac")
    measure.add_argument("--prosac", default="off,on")
    measure.add_argument("--batches", default="auto", help="comma-separated batch sizes, integers or auto")
    measure.add_argument("--budgets", default="2048,8192,32768", help="comma-separated draw budgets (max_samples)")
    measure.add_argument("--lo", default="0,5", help="comma-separated max_lo_iters values")
    measure.add_argument("--seeds", default="0")
    measure.add_argument("--min-run-time", type=float, default=0.05)
    measure.add_argument("--timing-pairs", type=int, help="time only the first N pairs; score all")
    measure.add_argument("--json", type=Path, required=True)
    measure.set_defaults(contribute=None, machine_slug=None)
    plotter = commands.add_parser("plot")
    plotter.add_argument("--json", action="append", required=True, help="label=path (repeatable)")
    plotter.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()

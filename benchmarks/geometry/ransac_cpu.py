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

"""Reproducible CPU RANSAC A/B timings and geometric quality on HEB and cached PhotoTourism pairs.

Run this same harness from each checkout. Only public RANSAC.forward is timed, on preloaded float32 CPU
inputs, with default sampling, scoring, batching and local optimization; only threshold, confidence
(default 0.999), budget and seed are overridden. Construction and pose recovery
are excluded. Each pair/budget is repeatedly timed at seed zero; quality is scored independently at all
requested seeds. Budgets count minimal samples, with confidence stopping allowed to finish earlier.

PhotoTourism NPZ files are prepared by ransac.py. Pair selection follows its pair_keys helper: the first
--pairs entries per scene/feature in sorted export order. Input correspondence order is preserved unless
--sort-matches is given. Essential estimation uses calibrated coordinates and divides the pixel threshold
by the mean focal length. Pose quality is max(rotation error, sign-invariant translation angular error),
with strict 1..10 degree mAA thresholds, averaged equally over scenes and seeds per feature.

HEB selects --h-pairs without replacement using default_rng(--pair-seed), then retains SNN ratios below
0.8 and sorts them best-first. Quality is mean reprojection error on all ground-truth inliers, with mAA
at ten logarithmic thresholds from 1 to 20 pixels (inclusive comparisons, as in heb.py). Failures count
as misses. The default pixel inlier thresholds are H=8, SIFT F/E=0.75, SIFT8k=0.5, ALIKED-LightGlue=1.5;
these are fixed diagnostic choices, not a claim of universal optimality. Override via --h-threshold and
--epi-threshold FEATURE=PX. Optional h5py reads HEB; OpenCV supplies pose recovery for F/E. No downloads.

    python benchmarks/geometry/ransac_cpu.py --npz /data/phototourism-7x10.npz \
        --heb /data/NYC_Library_homographies.h5 --pairs 10 --h-pairs 30 --json /tmp/cpu.json
"""

from __future__ import annotations

import argparse
import hashlib
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

from common import run_metadata, save_json, time_us, warm_up_cpu  # noqa: E402
from geometry.heb import MAA_THRESHOLDS_PX, load_pairs, reprojection_error  # noqa: E402
from geometry.ransac import PairCalibration, digest, measured_ransac_source, pair_keys  # noqa: E402

import kornia  # noqa: E402
from kornia.geometry import RANSAC  # noqa: E402

MODEL_TYPES = {"H": "homography", "F": "fundamental", "E": "essential"}
EPI_THRESHOLDS = {"sift": 0.75, "sift8k": 0.5, "aliked_lightglue": 1.5}


def load_epipolar(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load only the selected exported records, retaining their correspondence order by default."""
    items = []
    with np.load(args.npz, allow_pickle=False) as data:
        for key in pair_keys(data, args):
            scene, feature, _ = key.split("__")
            order = np.arange(len(data[key + "__mkp_a"]))
            field = key + "__score" if key + "__score" in data else key + "__ratio"
            if args.sort_matches and field in data:
                order = np.argsort(data[field].reshape(-1), kind="stable")
            items.append(
                {
                    "pair": key,
                    "scene": scene,
                    "feature": feature,
                    "a": data[key + "__mkp_a"][order].astype(np.float64),
                    "b": data[key + "__mkp_b"][order].astype(np.float64),
                    "calib": [{f: data[f"{key}__{side}_{f}"] for f in ("K", "R", "T")} for side in ("a", "b")],
                }
            )
    return items


def load_homography(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Use the same seeded HEB pair selection, SNN filter and stable ranking as heb.py."""
    selected = load_pairs(argparse.Namespace(h5=args.heb, pairs=args.h_pairs, pair_seed=args.pair_seed))
    items = []
    for name, item in selected.items():
        matches = item["matches"]
        retained = matches[matches[:, 8] < 0.8]
        ranked = retained[np.argsort(retained[:, 8], kind="stable")]
        gt = matches[matches[:, 9].astype(bool)]
        items.append(
            {
                "pair": name,
                "scene": args.heb.stem.removesuffix("_homographies"),
                "feature": "heb",
                "a": ranked[:, :2].astype(np.float64),
                "b": ranked[:, 2:4].astype(np.float64),
                "gt_a": gt[:, :2],
                "gt_b": gt[:, 2:4],
            }
        )
    return items


def pose_error(F: np.ndarray, a: np.ndarray, b: np.ndarray, calib_a: dict[str, Any], calib_b: dict[str, Any]) -> float:
    """IMC angular pose error: max(rotation angle, unsigned translation angle), in degrees.

    This is the same OpenCV recovery and angular convention as imc2021-simple's metrics.pose_error;
    keeping this small metric here avoids importing its reconstruction and dataset dependencies.
    """
    import cv2

    if len(a) < 5 or len(b) < 5:
        return float("inf")
    rotation_gt = calib_b["R"] @ calib_a["R"].T
    translation_gt = calib_b["T"].reshape(-1) - rotation_gt @ calib_a["T"].reshape(-1)
    if np.linalg.norm(translation_gt) < 1e-9:
        return float("inf")
    Ka, Kb = calib_a["K"], calib_b["K"]
    normalized_a = (a - Ka[:2, 2]) / np.array([Ka[0, 0], Ka[1, 1]])
    normalized_b = (b - Kb[:2, 2]) / np.array([Kb[0, 0], Kb[1, 1]])
    try:
        _, rotation, translation, _ = cv2.recoverPose(Kb.T @ F @ Ka, normalized_a, normalized_b)
    except cv2.error:
        return float("inf")
    rotation_cos = np.clip((np.trace(rotation @ rotation_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    translation = translation.reshape(-1)
    translation = translation / (np.linalg.norm(translation) + 1e-15)
    translation_gt = translation_gt / (np.linalg.norm(translation_gt) + 1e-15)
    translation_cos = np.clip(abs(translation @ translation_gt), 0.0, 1.0)
    return float(np.degrees(max(np.arccos(rotation_cos), np.arccos(translation_cos))))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average geometric accuracy equally over scenes and seeds; report mean per-pair median latency."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["problem"], row["feature"], row["budget"]].append(row)
    result = []
    for (problem, feature, budget), group in sorted(groups.items()):
        cells: dict[tuple, list[float]] = defaultdict(list)
        for row in group:
            error = row["err"] if row["err"] is not None else float("inf")
            success = error <= MAA_THRESHOLDS_PX if problem == "H" else error < np.arange(1, 11)
            cells[row["scene"], row["seed"]].append(float(np.mean(success)))
        timings = [row["median_us"] for row in group if row["median_us"] is not None]
        result.append(
            {
                "problem": problem,
                "feature": feature,
                "budget": budget,
                "maa": float(np.mean([np.mean(values) for values in cells.values()])),
                "mean_median_us": float(np.mean(timings)) if timings else None,
                "quality_evaluations": len(group),
                "timed_pairs": len(timings),
            }
        )
    return result


def run(args: argparse.Namespace) -> None:
    """Measure public estimator calls and persist all per-pair geometric errors and timing spreads."""
    expected = ROOT / "kornia" / "__init__.py"
    if Path(kornia.__file__).resolve() != expected:
        raise RuntimeError(f"Imported {kornia.__file__}; expected {expected}")
    print(f"# kornia: {kornia.__file__}\n# interpreter: {sys.executable}", flush=True)
    models = args.models.split(",")
    torch.set_num_threads(args.threads)
    if any(model in models for model in ("F", "E")):
        import cv2

        cv2.setNumThreads(args.threads)
    torch.manual_seed(0)
    warm_up_cpu()
    epipolar = load_epipolar(args) if any(model in models for model in ("F", "E")) else []
    homography = load_homography(args) if "H" in models else []
    thresholds = EPI_THRESHOLDS | dict(args.epi_threshold)
    for item in epipolar:
        if item["feature"] not in thresholds:
            raise ValueError(f"Specify --epi-threshold {item['feature']}=PX for this feature")
    meta = run_metadata(torch.device("cpu"))
    meta.update(
        kornia_module="kornia/__init__.py",
        harness_sha256=digest(Path(__file__)),
        ransac_source_sha256=digest(measured_ransac_source()),
        source_sha256={
            name: digest(ROOT / name)
            for name in (
                "kornia/geometry/ransac.py",
                "kornia/geometry/homography.py",
                "kornia/geometry/epipolar/fundamental.py",
                "kornia/geometry/epipolar/essential.py",
            )
        },
        units="pairs/s",
        dtype="float32",
        models=models,
        budgets=args.budgets,
        seeds=args.seeds,
        pairs_per_scene_feature=args.pairs,
        h_pairs=args.h_pairs,
        pair_seed=args.pair_seed,
        sort_matches=args.sort_matches,
        h_snn=0.8,
        h_threshold_px=args.h_threshold,
        epipolar_thresholds_px=thresholds,
        confidence=args.confidence,
        min_run_time=args.min_run_time,
        timing_seed=0,
        timing_storage="Seed-zero timing is stored on the first requested quality seed; other rows are null",
        timing_enabled=not args.no_timing,
        timing_includes="Repeated public RANSAC.forward on preloaded float32 CPU inputs; seed zero; median and IQR",
        quality="All requested seeds; geometric errors; mAA averaged equally over scenes and seeds per feature",
        h_maa_thresholds_px=MAA_THRESHOLDS_PX.tolist(),
        pose_maa_thresholds_degrees=list(range(1, 11)),
        npz_sha256=digest(args.npz) if epipolar else None,
        heb_file=args.heb.name if homography else None,
    )
    # Identify the selected HEB content without hashing a multi-gigabyte scene or recording local paths.
    hasher = hashlib.sha256()
    for item in homography:
        hasher.update(item["pair"].encode())
        for field in ("a", "b", "gt_a", "gt_b"):
            hasher.update(np.ascontiguousarray(item[field]).tobytes())
    meta["heb_selected_sha256"] = hasher.hexdigest() if homography else None
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for problem in models:
            items = homography if problem == "H" else epipolar
            if not items:
                raise ValueError(f"No pairs selected for {problem}")
            for index, item in enumerate(items):
                calibration = None
                points = item["a"], item["b"]
                threshold = args.h_threshold if problem == "H" else thresholds[item["feature"]]
                if problem == "E":
                    calibration = PairCalibration(*(entry["K"] for entry in item["calib"]))
                    points = calibration.normalized(*points)
                kp1, kp2 = (torch.as_tensor(point, dtype=torch.float32) for point in points)
                for budget in args.budgets:
                    estimator = RANSAC(
                        MODEL_TYPES[problem],
                        inl_th=threshold / calibration.focal if calibration is not None else threshold,
                        max_samples=budget,
                        confidence=args.confidence,
                        seed=0,
                        compile=args.compile,
                    )
                    minimum = {"H": 4, "F": 7, "E": 5}[problem]
                    enough = len(kp1) >= minimum
                    median = iqr = None
                    if enough and not args.no_timing:
                        if args.compile:
                            estimator(kp1, kp2)  # the first call compiles, or loads the saved graph
                        median, iqr = time_us(partial(estimator, kp1, kp2), min_run_time=args.min_run_time)
                    for seed_index, seed in enumerate(args.seeds):
                        estimator.seed = seed
                        error, support = float("inf"), 0
                        if enough:
                            matrix, mask = estimator(kp1, kp2)
                            matrix = matrix.double().numpy()
                            inliers = mask.numpy().reshape(-1).astype(bool)
                            support = int(inliers.sum())
                            usable = np.isfinite(matrix).all() and np.any(matrix != 0)
                            if usable and problem == "H" and support >= 4 and len(item["gt_a"]) > 1:
                                error = reprojection_error(item["gt_a"], item["gt_b"], matrix)
                            elif usable and problem != "H":
                                fundamental = calibration.fundamental(matrix) if calibration is not None else matrix
                                error = pose_error(fundamental, item["a"][inliers], item["b"][inliers], *item["calib"])
                        rows.append(
                            {
                                "op": "RANSAC",
                                "backend": "kornia (eager)",
                                "batch": 1,
                                "problem": problem,
                                "feature": item["feature"],
                                "scene": item["scene"],
                                "pair": item["pair"],
                                "n": len(kp1),
                                "budget": budget,
                                "seed": seed,
                                "threshold_px": threshold,
                                "err": error,
                                "error_units": "pixels" if problem == "H" else "degrees",
                                "support": support,
                                "timing_seed": 0,
                                "median_us": median if seed_index == 0 else None,
                                "iqr_us": iqr if seed_index == 0 else None,
                                "throughput_per_s": 1e6 / median if seed_index == 0 and median else None,
                            }
                        )
                print(f"# {problem} {index + 1}/{len(items)} {item['pair']}", flush=True)
                # Preserve partial results if a long run is interrupted.
                save_json(args.json, meta, rows)
    summaries = summarize(rows)
    meta["summaries"] = summaries
    save_json(args.json, meta, rows)
    for summary in summaries:
        latency = summary["mean_median_us"]
        time_text = f"{latency / 1000:.3f} ms" if latency is not None else "untimed"
        print(
            f"{summary['problem']} {summary['feature']} budget={summary['budget']}: "
            f"mAA={summary['maa']:.4f}, {time_text}",
            flush=True,
        )


def threshold_override(value: str) -> tuple[str, float]:
    """Parse FEATURE=PX and reject nonpositive or nonfinite thresholds."""
    try:
        feature, raw = value.split("=", 1)
        threshold = float(raw)
        if not feature or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError
        return feature, threshold
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected FEATURE=PX with a positive finite threshold") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, help="PhotoTourism export from ransac.py prepare; required for F/E")
    parser.add_argument("--heb", type=Path, help="HEB scene HDF5; required for H")
    parser.add_argument("--models", default="H,F,E")
    parser.add_argument(
        "--pairs", type=int, default=2, help="PhotoTourism pairs per scene/feature; 10 uses the 7x10 export"
    )
    parser.add_argument("--h-pairs", type=int, default=30)
    parser.add_argument("--pair-seed", type=int, default=0, help="HEB selection seed")
    parser.add_argument("--scenes", default="", help="Optional comma-separated PhotoTourism scene filter")
    parser.add_argument("--features", default="", help="Optional comma-separated PhotoTourism feature filter")
    parser.add_argument("--sort-matches", action="store_true", help="Sort PhotoTourism score/ratio ascending")
    parser.add_argument("--budgets", default="256,4096")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--h-threshold", type=float, default=8.0)
    parser.add_argument("--epi-threshold", type=threshold_override, action="append", default=[], metavar="FEATURE=PX")
    parser.add_argument(
        "--min-run-time", type=float, default=0.1, help="Minimum repeated timing duration per pair/budget"
    )
    parser.add_argument("--no-timing", action="store_true", help="Evaluate geometric quality only")
    parser.add_argument(
        "--compile", action="store_true", help="Time RANSAC(compile=True); its first call compiles and is not timed"
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not args.models or any(model not in MODEL_TYPES for model in args.models.split(",")):
        parser.error("--models must be a comma-separated selection of H,F,E")
    if "H" in args.models and args.heb is None:
        parser.error("--heb is required when --models includes H")
    if any(model in args.models for model in ("F", "E")) and args.npz is None:
        parser.error("--npz is required when --models includes F or E")
    try:
        args.budgets = list(dict.fromkeys(int(value) for value in args.budgets.split(",")))
        args.seeds = list(dict.fromkeys(int(value) for value in args.seeds.split(",")))
    except ValueError:
        parser.error("--budgets and --seeds must be comma-separated integers")
    if min(args.budgets) < 1 or min(args.seeds) < 0:
        parser.error("budgets must be positive and seeds nonnegative")
    if min(args.pairs, args.h_pairs, args.threads) < 1:
        parser.error("pair counts and threads must be positive")
    if not np.isfinite(args.min_run_time) or args.min_run_time <= 0:
        parser.error("--min-run-time must be positive and finite")
    if not np.isfinite(args.h_threshold) or args.h_threshold <= 0:
        parser.error("--h-threshold must be positive and finite")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must lie strictly between zero and one")
    run(args)


if __name__ == "__main__":
    main()

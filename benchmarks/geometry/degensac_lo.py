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

"""Compare public fundamental RANSAC with and without DEGENSAC.

The two arms differ only in ``RANSAC(..., degensac=...)``.  Timing covers repeated
public ``RANSAC.forward`` calls on float32 CPU inputs; quality uses independent
calls for every seed.  The synthetic scene is deliberately near-planar: 95% of
its 60% geometric inliers lie on a plane, while the remaining inliers constrain
the camera motion.  Its off-plane metric evaluates the returned F against clean
off-plane correspondences, avoiding the dominated plane's apparent support.

Run the same command from each checkout being compared.  ``--golden-dir`` reads
pydegensac's calibrated Reichstag regression pairs; ``--npz`` optionally reads
the PhotoTourism export accepted by ransac_cpu.py.

    python benchmarks/geometry/degensac_lo.py --json /tmp/degensac.json
    python benchmarks/geometry/degensac_lo.py --golden-dir /data/pydegensac-golden \\
        --json /tmp/degensac-real.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "benchmarks"))

from common import collect_load_metrics, run_metadata, save_json, time_us, warm_up_cpu  # noqa: E402
from geometry.ransac import digest, measured_ransac_source  # noqa: E402
from geometry.ransac_cpu import load_epipolar, pose_error  # noqa: E402

import kornia  # noqa: E402
from kornia.geometry import RANSAC  # noqa: E402
from kornia.geometry.epipolar import sampson_epipolar_distance  # noqa: E402

from testing.geometry.create import create_dominant_plane_scene  # noqa: E402


def synthetic_pair(n: int, inlier_ratio: float, plane_ratio: float) -> dict[str, Any]:
    """Wrap the regression fixture used by RANSAC tests in benchmark input metadata."""
    a, b, _, clean_a, clean_b = create_dominant_plane_scene(
        n, inlier_ratio, plane_ratio, 0, device=torch.device("cpu"), dtype=torch.float32
    )
    return {
        "scene": "synthetic_dominant_plane",
        "pair": f"n{n}_inlier{inlier_ratio:.0%}_plane{plane_ratio:.0%}_noise0.5",
        "a": a,
        "b": b,
        "clean_a": clean_a,
        "clean_b": clean_b,
        "gt_a": clean_a,
        "gt_b": clean_b,
        "F_gt": None,
        "inlier_threshold_px": 1.0,
        "calib": None,
    }


def golden_pairs(directory: Path) -> list[dict[str, Any]]:
    """Load calibrated real Reichstag pairs published by pydegensac's regression suite."""
    items = []
    for path in sorted(directory.glob("golden_f*.npz")):
        with np.load(path, allow_pickle=False) as data:
            calib = [{field: data[f"{field}{side}"] for field in ("K", "R", "T")} for side in ("1", "2")]
            items.append(
                {
                    "scene": "reichstag",
                    "pair": path.stem.removeprefix("golden_f_"),
                    "a": torch.as_tensor(data["pts1"], dtype=torch.float32),
                    "b": torch.as_tensor(data["pts2"], dtype=torch.float32),
                    "clean_a": None,
                    "clean_b": None,
                    "gt_a": None,
                    "gt_b": None,
                    "F_gt": data["F_gt"],
                    "inlier_threshold_px": float(data["px_th"]),
                    "calib": calib,
                    "source_sha256": digest(path),
                }
            )
    if not items:
        raise ValueError(f"No golden_f*.npz files in {directory}")
    return items


def phototourism_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Reuse ransac_cpu's correspondence order and calibrated pose metric input."""
    items = []
    for item in load_epipolar(args):
        items.append(
            {
                "scene": item["scene"],
                "pair": item["pair"],
                "a": torch.as_tensor(item["a"], dtype=torch.float32),
                "b": torch.as_tensor(item["b"], dtype=torch.float32),
                "clean_a": None,
                "clean_b": None,
                "gt_a": None,
                "gt_b": None,
                "F_gt": None,
                "inlier_threshold_px": args.threshold_px,
                "calib": item["calib"],
            }
        )
    return items


def loftr_pair(path: Path, threshold: float) -> dict[str, Any]:
    """Load the checked-in LoFTR fundamental regression data without loading its model."""
    if path.suffix == ".safetensors":
        from kornia.core import load_safetensors

        data = load_safetensors(path)
    else:
        data = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "scene": "loftr_indoor",
        "pair": path.stem,
        "a": data["loftr_indoor_tentatives0"].float(),
        "b": data["loftr_indoor_tentatives1"].float(),
        "clean_a": None,
        "clean_b": None,
        "gt_a": data["pts0"].float(),
        "gt_b": data["pts1"].float(),
        "F_gt": data["F_gt"].numpy(),
        "inlier_threshold_px": threshold,
        "calib": None,
        "source_sha256": digest(path),
    }


def quality(
    model: torch.Tensor, mask: torch.Tensor, item: dict[str, Any]
) -> tuple[float | None, float | None, float | None, int, int]:
    """Return synthetic off-plane error, calibrated pose error, and returned support."""
    support = int(mask.sum())
    usable = bool(torch.isfinite(model).all() and torch.any(model != 0))
    if not usable:
        return (
            float("inf") if item["clean_a"] is not None else None,
            float("inf") if item["calib"] else None,
            float("inf") if item["gt_a"] is not None else None,
            len(item["gt_a"]) if item["gt_a"] is not None else 0,
            support,
        )
    off_plane = None
    if item["clean_a"] is not None:
        errors = sampson_epipolar_distance(
            item["clean_a"].double()[None], item["clean_b"].double()[None], model.double()[None], squared=False
        )[0]
        off_plane = float(errors.median())
    pose = None
    if item["calib"] is not None:
        selected = mask.numpy().astype(bool)
        pose = pose_error(
            model.double().numpy(), item["a"].numpy()[selected], item["b"].numpy()[selected], *item["calib"]
        )
    gt_error, gross = None, 0
    if item["gt_a"] is not None:
        errors = sampson_epipolar_distance(
            item["gt_a"].double()[None], item["gt_b"].double()[None], model.double()[None], squared=False
        )[0]
        gt_error, gross = float(errors.median()), int((errors > 10.0).sum())
    return off_plane, pose, gt_error, gross, support


def run(args: argparse.Namespace) -> None:
    expected = ROOT / "kornia" / "__init__.py"
    if Path(kornia.__file__).resolve() != expected:
        raise RuntimeError(f"Imported {kornia.__file__}; expected {expected}")
    print(f"# kornia: {kornia.__file__}\n# interpreter: {sys.executable}", flush=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    warm_up_cpu()
    items = [synthetic_pair(1000, ratio, plane) for ratio in (0.6, 0.3) for plane in (0.0, 0.95)]
    if args.golden_dir is not None:
        items.extend(golden_pairs(args.golden_dir))
    if args.npz is not None:
        items.extend(phototourism_pairs(args))
    if args.loftr is not None:
        items.append(loftr_pair(args.loftr, args.threshold_px))
    meta = run_metadata(torch.device("cpu"))
    meta.update(
        kornia_module="kornia/__init__.py",
        harness_sha256=digest(Path(__file__)),
        ransac_source_sha256=digest(measured_ransac_source()),
        source_sha256={
            name: digest(ROOT / name) for name in ("kornia/geometry/ransac.py", "kornia/geometry/_degensac.py")
        },
        units="pairs/s",
        dtype="float32",
        modes=args.degensac,
        budgets=args.budgets,
        seeds=args.seeds,
        confidence=args.confidence,
        min_run_time=args.min_run_time,
        timing_seeds=args.timing_seeds,
        load=collect_load_metrics(),
        synthetic={"n": 1000, "inlier_ratios": [0.6, 0.3], "plane_ratios": [0.0, 0.95], "noise_px": 0.5},
        timing="Repeated public RANSAC.forward for selected seeds; identical inputs across estimator seeds",
        quality=(
            "All requested seeds; synthetic median clean off-plane Sampson distance and fraction below 2 px; "
            "calibrated pose error for real pairs"
        ),
        input_sha256={item["pair"]: item["source_sha256"] for item in items if "source_sha256" in item},
        golden_dir_sha256=hashlib.sha256(
            "".join(sorted(x.get("source_sha256", "") for x in items)).encode()
        ).hexdigest()
        if args.golden_dir is not None
        else None,
        npz_sha256=digest(args.npz) if args.npz is not None else None,
    )
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for item in items:
            for mode in args.degensac:
                for budget in args.budgets:
                    for seed in args.seeds:
                        estimator = RANSAC(
                            "fundamental",
                            inl_th=item["inlier_threshold_px"],
                            max_samples=budget,
                            confidence=args.confidence,
                            seed=seed,
                            degensac=mode,
                        )
                        call = partial(estimator, item["a"], item["b"])
                        model, mask = call()
                        off_plane, pose, gt_error, gross, support = quality(model, mask, item)
                        median = iqr = None
                        if seed in args.timing_seeds:
                            median, iqr = time_us(call, min_run_time=args.min_run_time)
                            if not np.isfinite(median) or not np.isfinite(iqr):
                                raise RuntimeError("RANSAC.forward failed during repeated timing")
                        fraction = None
                        if item["clean_a"] is not None and torch.isfinite(model).all() and torch.any(model != 0):
                            errors = sampson_epipolar_distance(
                                item["clean_a"].double()[None],
                                item["clean_b"].double()[None],
                                model.double()[None],
                                squared=False,
                            )[0]
                            fraction = float((errors < 2.0).float().mean())
                        rows.append(
                            {
                                "op": "RANSAC.forward",
                                "backend": "kornia (eager)",
                                "batch": 1,
                                "problem": "F",
                                "scene": item["scene"],
                                "pair": item["pair"],
                                "mode": "degensac" if mode else "plain",
                                "degensac": mode,
                                "budget": budget,
                                "seed": seed,
                                "n": len(item["a"]),
                                "inlier_threshold_px": item["inlier_threshold_px"],
                                "median_us": median,
                                "iqr_us": iqr,
                                "throughput_per_s": 1e6 / median if median else None,
                                "support": support,
                                "off_plane_error_px": off_plane,
                                "off_plane_fraction_lt_2px": fraction,
                                "pose_error_deg": pose,
                                "median_gt_error_px": gt_error,
                                "gross_gt_errors_gt_10px": gross,
                            }
                        )
                    save_json(args.json, meta, rows)
                    print(f"# {item['scene']} {item['pair']} mode={mode} budget={budget}", flush=True)
    save_json(args.json, meta, rows)


def parse_modes(value: str) -> list[bool]:
    values = value.split(",")
    if not values or any(v not in ("false", "true") for v in values):
        raise argparse.ArgumentTypeError("--degensac must be a comma-separated selection of false,true")
    return list(dict.fromkeys(v == "true" for v in values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path)
    parser.add_argument("--npz", type=Path, help="PhotoTourism export prepared by ransac.py")
    parser.add_argument("--loftr", type=Path, help="LoFTR fundamental regression .pt or .safetensors data")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--scenes", default="")
    parser.add_argument("--features", default="")
    parser.add_argument("--sort-matches", action="store_true")
    parser.add_argument("--threshold-px", type=float, default=0.75)
    parser.add_argument("--degensac", type=parse_modes, default=[False, True])
    parser.add_argument("--budgets", default="256,4096")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--timing-seeds", default="0,1,2")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--min-run-time", type=float, default=0.3)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.budgets = list(dict.fromkeys(int(x) for x in args.budgets.split(",")))
        args.seeds = list(dict.fromkeys(int(x) for x in args.seeds.split(",")))
        args.timing_seeds = list(dict.fromkeys(int(x) for x in args.timing_seeds.split(",")))
    except ValueError:
        parser.error("--budgets and --seeds must be comma-separated integers")
    if min(args.budgets) < 1 or min(args.seeds) < 0 or min(args.timing_seeds) < 0 or args.threads < 1 or args.pairs < 1:
        parser.error("budgets, threads and pairs must be positive; seeds must be nonnegative")
    if not 0 < args.confidence <= 1 or not np.isfinite(args.threshold_px) or args.threshold_px <= 0:
        parser.error("confidence must lie in (0, 1], and threshold must be positive")
    if not np.isfinite(args.min_run_time) or args.min_run_time <= 0:
        parser.error("--min-run-time must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()

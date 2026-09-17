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
"""Pyramid-based versus patch-based sparse SIFT extraction on fixed graf detections.

Run from the measured checkout root using ``python -m benchmarks.feature.dense_sift``.
Image loading, detection and matching are excluded from orientation+description
latency. Both methods receive identical upright DoG LAFs, grayscale float32 images,
and use RootSIFT. Dense pyramid construction IS timed, with no cross-call cache.
Matching uses SNN ratio 0.8; precision is forward ground-truth homography transfer
error <= 3 pixels. Separately, fixed-seed RANSAC estimates a homography and reports
its inlier count and mean L1 corner error against ground truth. RANSAC is outside
the descriptor timing and runs on CPU for MPS inputs. Counts and all five
pairs are retained, including failures. These are algorithm alternatives, not
numerically equivalent implementations. CPU uses one thread, MPS is synchronized.
Use --affine to adapt detections with LAFAffineShapeEstimator before the timed
stages. Use --methods patch with this script from a base worktree for an A/B baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import kornia
import kornia.feature as KF
from kornia.geometry import RANSAC, transform_points

if importlib.util.find_spec("PIL") is None:
    raise SystemExit("SKIP: this benchmark requires the optional Pillow package")
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import collect_load_metrics, run_metadata, save_json, time_us, versions_line


def homography_quality(
    source: torch.Tensor, target: torch.Tensor, ground_truth: torch.Tensor, height: int, width: int
) -> dict:
    """Estimate a homography with the existing graf benchmark RANSAC protocol.

    RANSAC consensus is not ground-truth accuracy: a wrong model can have many
    inliers, so retain its corner error even when large. Failed estimates have
    a null error, never a misleading zero.
    """
    result = {"ransac_inliers": 0, "ransac_corner_error_px": None, "ransac_status": "insufficient_matches"}
    if source.shape[0] < 4:
        return result
    ransac = RANSAC("homography", inl_th=2.0, max_iter=10, batch_size=8196, confidence=0.9999, seed=3407)
    estimate, mask = ransac(source, target)
    inliers = int(mask.sum())
    result["ransac_inliers"] = inliers
    result["ransac_status"] = "failed_estimate"
    if inliers < 4 or not torch.isfinite(estimate).all():
        return result
    corners = source.new_tensor([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
    # Use explicit homogeneous division to reject a corner at infinity. This
    # matches the original graf L1 metric for finite homography projections.
    homogeneous = torch.cat([corners, torch.ones_like(corners[:, :1])], dim=-1)
    prediction = homogeneous @ estimate.T
    reference = homogeneous @ ground_truth.reshape(3, 3).to(source).T
    if (prediction[:, 2].abs() <= 1e-8).any() or (reference[:, 2].abs() <= 1e-8).any():
        result["ransac_status"] = "nonfinite_projection"
        return result
    distance = (prediction[:, :2] / prediction[:, 2:] - reference[:, :2] / reference[:, 2:]).abs().sum(-1)
    if not torch.isfinite(distance).all():
        result["ransac_status"] = "nonfinite_projection"
        return result
    result["ransac_corner_error_px"] = distance.mean().item()
    result["ransac_status"] = "estimated"
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--expected-checkout", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--nf", type=int, default=4096)
    parser.add_argument("--affine", action="store_true", help="Adapt fixed detections with LAFAffineShapeEstimator")
    parser.add_argument("--methods", nargs="+", choices=("patch", "dense"), default=["patch", "dense"])
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument(
        "--quality-only", action="store_true", help="Evaluate matching and RANSAC without repeating timings"
    )
    args = parser.parse_args()
    print(f"# interpreter: {sys.executable}\n# kornia: {kornia.__file__}", flush=True)
    if Path(kornia.__file__).resolve().parents[1] != args.expected_checkout.resolve():
        raise RuntimeError("Imported wrong checkout")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    ransac_device = torch.device("cpu") if device.type == "mps" else device
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device.type, lambda: None)
    paths = [args.seq / f"img{i}.ppm" for i in range(1, 7)]
    images = []
    for path in paths:
        # PIL's RGB -> L conversion is identical across both methods and revisions.
        pixels = np.array(Image.open(path).convert("L"), copy=True)
        images.append(torch.from_numpy(pixels).to(device=device, dtype=torch.float32)[None, None] / 255)
    meta = run_metadata(device)
    meta.update(
        load=collect_load_metrics(),
        seed=0,
        num_features=args.nf,
        rootsift=True,
        affine=args.affine,
        timing="orientation + description per image, including dense pyramid; excludes detection, I/O and matching",
        matching_ratio=0.8,
        precision_threshold_px=3.0,
        precision_metric="forward GT transfer Euclidean error",
        min_run_time=args.min_run_time,
        quality_only=args.quality_only,
        ransac_device=str(ransac_device),
        ransac={
            "model": "homography",
            "seed": 3407,
            "inlier_threshold_px": 2.0,
            "max_iter": 10,
            "batch_size": 8196,
            "confidence": 0.9999,
        },
        ransac_corner_metric="mean L1 corner transfer error against GT in pixels",
        detector="SIFTFeatureScaleSpace(upright=True).detector",
        image_conversion="PIL RGB to L",
        compile=False,
        dense_spatial_bin_size=8,
        dense_orientation_samples=9,
        dense_orientation_peak="parabolic refinement",
        algorithm_sources=[
            "https://www.vlfeat.org/api/dsift.html",
            "https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.simd.hpp",
        ],
        input_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths + [args.seq / f"H1to{i}p" for i in range(2, 7)]
        },
    )
    meta["implementation_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("sift_pyramid.py", "siftdesc.py", "integrated.py", "laf.py", "orientation.py")
        if (path := Path(kornia.__file__).parent / "feature" / name).is_file()
    }
    print(versions_line(meta), flush=True)
    detector = KF.SIFTFeatureScaleSpace(num_features=args.nf, upright=True, device=device).detector.eval()
    frames = []
    affine = KF.LAFAffineShapeEstimator().to(device).eval() if args.affine else None
    rows = []
    for i, img in enumerate(images, 1):
        sync()
        start = time.perf_counter()
        lafs, _ = detector(img)
        sync()
        lafs = lafs[:, KF.laf_is_filled(lafs)[0]]
        if affine is not None:
            lafs = affine(lafs, img)
        frames.append(lafs)
        print(f"# img{i}: {lafs.shape[1]} detections ({time.perf_counter() - start:.2f}s, untimed)", flush=True)
    meta["laf_sha256"] = [hashlib.sha256(laf.cpu().numpy().tobytes()).hexdigest() for laf in frames]
    for method in args.methods:
        if method == "patch":
            orienter = KF.LAFOrienter(19).to(device).eval()
            descriptor = (
                KF.LAFDescriptor(KF.SIFTDescriptor(patch_size=41, rootsift=True), patch_size=41).to(device).eval()
            )

            def run(img, laf, orienter=orienter, descriptor=descriptor):
                oriented = orienter(laf, img)
                return oriented, descriptor(img, oriented)
        else:
            model = KF.SIFTDescriptorFromPyramid(rootsift=True).to(device).eval()

            def run(img, laf, model=model):
                return model.orient_and_describe(img, laf)

        outputs = []
        for i, (img, laf) in enumerate(zip(images, frames), 1):
            sync()
            start = time.perf_counter()
            _oriented, desc = run(img, laf)
            sync()
            warm = time.perf_counter() - start
            if not torch.isfinite(desc).all():
                raise RuntimeError(f"Nonfinite descriptors: {method}, img{i}")
            median, iqr = float("nan"), float("nan")
            if not args.quality_only:
                median, iqr = time_us(
                    lambda img=img, laf=laf: run(img, laf),
                    min_run_time=max(args.min_run_time, 5 * warm),
                    sync=sync if device.type == "mps" else None,
                )
                if not math.isfinite(median):
                    raise RuntimeError(f"Timing failed: {method}, img{i}")
            outputs.append(desc[0])
            rows.append(
                {
                    "op": "orientation_description",
                    "backend": method,
                    "batch": 1,
                    "image": i,
                    "height": img.shape[-2],
                    "width": img.shape[-1],
                    "dtype": "float32",
                    "features": laf.shape[1],
                    "median_us": median,
                    "iqr_us": iqr,
                    "throughput_per_s": 1e6 / median,
                }
            )
            if not args.quality_only:
                print(f"{method} img{i}: {median / 1000:.2f} +/- {iqr / 1000:.2f} ms", flush=True)
            save_json(args.json, meta, rows)
        for i in range(2, 7):
            _, matches = KF.match_snn(outputs[0], outputs[i - 1], 0.8)
            H = torch.tensor(np.loadtxt(args.seq / f"H1to{i}p"), device=device, dtype=torch.float32)[None]
            xy1 = KF.get_laf_center(frames[0])[0]
            xy2 = KF.get_laf_center(frames[i - 1])[0]
            projected = transform_points(H, xy1[None])[0]
            error = (projected[matches[:, 0]] - xy2[matches[:, 1]]).norm(dim=-1)
            correct = int((error <= 3.0).sum())
            count = len(matches)
            precision = correct / count if count else 0.0
            geometry = homography_quality(
                xy1[matches[:, 0]].to(ransac_device),
                xy2[matches[:, 1]].to(ransac_device),
                H.to(ransac_device),
                *images[0].shape[-2:],
            )
            rows.append(
                {
                    "op": "matching_quality",
                    "backend": method,
                    "batch": 1,
                    "pair": f"1-{i}",
                    "median_us": None,
                    "throughput_per_s": None,
                    "matches": count,
                    "correct_matches": correct,
                    "precision": precision,
                    "median_transfer_error_px": error.median().item() if count else None,
                    **geometry,
                }
            )
            print(
                f"{method} 1-{i}: {correct}/{count} correct ({100 * precision:.1f}%); "
                f"RANSAC {geometry['ransac_inliers']} inliers, corner L1 {geometry['ransac_corner_error_px']}",
                flush=True,
            )
            save_json(args.json, meta, rows)


if __name__ == "__main__":
    main()

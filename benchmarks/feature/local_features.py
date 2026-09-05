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
"""Repeatable local-feature speed and homography quality on Oxford affine sequences.

Run from the checkout root, using its interpreter::

    .venv/bin/python -m benchmarks.feature.local_features --seq /data/graf --device cuda --json graf.json

Compares revisions of public Kornia pipelines, without a cross-library baseline. All
images retain their original resolution; inputs are grayscale float32, batch one,
with 4096 requested features. SIFT uses the existing scale-space harness defaults
(DoG, AdaptiveQuadInterp3d, doubled image, three levels, sigma 1.6, RootSIFT), with
gradient orientation. SIFT-AffNet-HardNet adds pretrained AffNet and HardNet.
KeyNet-HardNet uses its public preset, including pretrained OriNet.

Timings are warmed median/IQR wall times, including device synchronization, for
extracting BOTH images and SNN matching (ratio 0.85), excluding I/O and RANSAC.
Image 1 is deliberately re-extracted for every pair. Quality is the existing
mean L1 corner reprojection error against the supplied H1toKp homography, with
fixed-seed Kornia RANSAC. Original Oxford PPM images are preferred; a PNG is used
only when no PPM exists (converted copies can decode to different pixels).
CPU timings use one thread, as does common.time_us's Timer. ``--device mps`` times
with ``torch.mps.synchronize`` inside the timed region and evaluates RANSAC on CPU,
because its batched SVD aborts the process on MPS; RANSAC is outside the timed region. Optional --compile uses
the existing factory's selective compilation: scale-space pyramid/response/subpixel
modules, or KeyNet response/NMS, with dynamic=True and the default Inductor mode.
Descriptors, orientation, affine adaptation, matching, and RANSAC remain eager.
Initial calls (including any compilation/cache loading) are recorded separately,
outside warmup and steady-state timing. No autocast or pipeline settings change.
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
from kornia.geometry import RANSAC

if importlib.util.find_spec("cv2") is None:
    raise SystemExit("SKIP: this benchmark requires the optional opencv-python package")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_metadata, save_json, time_us, versions_line

from benchmarks.feature.scale_space_detector import build_extractor, get_MAE_imgcorners, load_gray


def image_path(seq: Path, index: int) -> Path:
    for ext in ("ppm", "png"):
        path = seq / f"img{index}.{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No img{index}.png or img{index}.ppm in {seq}")


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    # NumPy only reads ground-truth files; all stochastic work uses PyTorch.
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    # blocked_autorange synchronizes CUDA itself; MPS needs an explicit sync inside the timed region.
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device.type, lambda: None)
    # Homography RANSAC batches an SVD that aborts the process on MPS (#4201, #4204); it is
    # outside the timed region, so evaluating it on CPU leaves the timings untouched.
    ransac_device = torch.device("cpu") if device.type == "mps" else device
    meta = run_metadata(device)
    meta.update(
        sequence=args.seq.name,
        num_features=args.nf,
        seed=0,
        ransac_seed=3407,
        matching_ratio=0.85,
        timing="extract both images + SNN, excluding RANSAC and I/O",
        corner_error="mean L1 corner distance in pixels",
        timing_pairs=args.timing_pairs,
        min_run_time=args.min_run_time,
        ransac_device=str(ransac_device),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        compile=args.compile,
        compile_scope="scale-space pyramid/response/subpixel; KeyNet response/NMS" if args.compile else None,
        compile_dynamic=True if args.compile else None,
        input_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [image_path(args.seq, i) for i in range(1, 7)] + [args.seq / f"H1to{i}p" for i in range(2, 7)]
        },
    )
    print(f"# {meta['git_commit']} | {meta['platform']} | {meta.get('cuda_device', device)}", flush=True)
    print(versions_line(meta), flush=True)
    print(f"# interpreter: {sys.executable}\n# kornia: {kornia.__file__}", flush=True)
    images = [load_gray(str(image_path(args.seq, i)), device) for i in range(1, 7)]
    rows = []
    for name in args.methods:
        torch.manual_seed(0)
        method, desc, ori, aff = {
            "sift": ("scalespace", "sift", "lap", "none"),
            "sift_affnet_hardnet": ("scalespace", "hardnet", "lap", "affnet"),
            "keynet_hardnet": ("keynet", "hardnet", "orinet", "none"),
        }[name]
        extractor = build_extractor(
            method, "dog", "adaptive", desc, ori, aff, device, args.nf, compile_modules=args.compile
        ).eval()
        print(f"\n# {name}\n# pair   median ms   IQR ms   corner L1 px   inliers   matches", flush=True)

        def extract_match(
            img2: torch.Tensor, extractor: torch.nn.Module = extractor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            kp1, d1, _ = extractor(images[0])
            kp2, d2, _ = extractor(img2)
            _, indices = KF.match_snn(d1, d2, 0.85)
            return kp1, kp2, indices

        for k, img2 in enumerate(images[1:], 2):
            # Never use a potentially minutes-long compilation call to size the
            # steady-state timing budget. Warm every pair before measuring it.
            initial_call_seconds = None
            if args.compile:
                sync()
                start = time.perf_counter()
                extract_match(img2)
                sync()
                initial_call_seconds = time.perf_counter() - start
                print(f"# 1-{k} initial compiled call: {initial_call_seconds:.3f} s", flush=True)
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                memory_before = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            kp1, kp2, indices = extract_match(img2)
            sync()
            peak_bytes = torch.cuda.max_memory_allocated(device) - memory_before if device.type == "cuda" else None
            warmup_seconds = time.perf_counter() - start
            median, iqr = float("nan"), float("nan")
            if k in args.timing_pairs:
                # A CPU call can take longer than the time budget. Extend it using
                # the warmup so blocked_autorange collects several samples there too.
                median, iqr = time_us(
                    lambda img2=img2: extract_match(img2),
                    min_run_time=max(args.min_run_time, 5 * warmup_seconds),
                    sync=sync if device.type == "mps" else None,
                )
                if not math.isfinite(median):
                    raise RuntimeError(f"Timing failed for {name}, pair 1-{k}")
            ransac = RANSAC("homography", inl_th=2.0, max_iter=10, batch_size=8196, confidence=0.9999, seed=3407)
            error, inliers = float("nan"), 0
            if len(indices) >= 4:
                H, mask = ransac(kp1[indices[:, 0]].to(ransac_device), kp2[indices[:, 1]].to(ransac_device))
                inliers = int(mask.sum())
                if inliers >= 4:
                    error = get_MAE_imgcorners(
                        *images[0].shape[-2:], np.loadtxt(args.seq / f"H1to{k}p"), H.cpu().numpy()
                    )
            row = {
                "op": name,
                "backend": "kornia (selectively compiled)" if args.compile else "kornia (eager)",
                "batch": 1,
                "height": img2.shape[-2],
                "width": img2.shape[-1],
                "dtype": "float32",
                "pair": f"1-{k}",
                "median_us": median,
                "iqr_us": iqr,
                "throughput_per_s": 2e6 / median,
                "corner_error_px": error,
                "inliers": inliers,
                "matches": len(indices),
                "features1": len(kp1),
                "features2": len(kp2),
                "peak_extra_cuda_bytes": peak_bytes,
                "initial_call_seconds": initial_call_seconds,
            }
            rows.append(row)
            print(
                f"  1-{k} {median / 1000:11.2f} {iqr / 1000:8.2f} {error:14.3f} {inliers:9d} {len(indices):9d}",
                flush=True,
            )
            if args.json:
                save_json(args.json, meta, rows)
        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                extractor(images[0])
            sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
            print(prof.key_averages().table(sort_by=sort_by, row_limit=25), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--nf", type=int, default=4096)
    parser.add_argument("--min-run-time", type=float, default=3.0)
    parser.add_argument("--timing-pairs", nargs="+", type=int, choices=range(2, 7), default=[2, 3, 4, 5, 6])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Selectively compile detector modules (see above)")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("sift", "sift_affnet_hardnet", "keynet_hardnet"),
        default=["sift", "sift_affnet_hardnet", "keynet_hardnet"],
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

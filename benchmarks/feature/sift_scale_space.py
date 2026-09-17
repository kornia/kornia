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
"""End-to-end SIFT descriptor-backend comparison on Oxford graf.

Run from the measured checkout root::

    python -m benchmarks.feature.sift_scale_space --seq /data/graf \\
        --expected-checkout "$PWD" --device cuda --json graf-sift-scale-space.json

Both public :class:`kornia.feature.SIFTFeatureScaleSpace` backends run their
whole forward pass: DoG detection, orientation, and RootSIFT description. The
patch implementation is constructed without ``descriptor_backend`` so this
harness can also measure a checkout from before that public argument existed.
Timing excludes image I/O, matching, and RANSAC. Matching uses SNN ratio 0.8;
quality is forward ground-truth transfer precision at three pixels and the
existing fixed-seed Graf homography RANSAC corner metric. MPS runs RANSAC on
CPU because MPS batched SVD is unsafe; it remains outside the timed region.
Optional ``--methods opencv --device cpu`` measures native uint8 CPU SIFT with
standard rejection filters plus NumPy RootSIFT; input conversion and conversion
of returned keypoints to Torch tensors are excluded. It shares the grayscale input and matching/RANSAC protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import kornia
import kornia.feature as KF
from kornia.geometry import transform_points

if importlib.util.find_spec("PIL") is None:
    raise SystemExit("SKIP: this benchmark requires the optional Pillow package")
from PIL import Image

from benchmarks.feature.dense_sift import homography_quality

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import collect_load_metrics, run_metadata, save_json, time_us, versions_line


def _image_path(seq: Path, index: int) -> Path:
    for extension in ("ppm", "png"):
        path = seq / f"img{index}.{extension}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No img{index}.ppm or img{index}.png in {seq}")


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    pixels = np.array(Image.open(path).convert("L"), copy=True)
    return torch.from_numpy(pixels).to(device=device, dtype=torch.float32)[None, None].div_(255.0)


def _build(method: str, num_features: int, device: torch.device) -> torch.nn.Module:
    # Deliberately leave this keyword off the patch constructor: the same file is
    # useful when run from a pre-feature base checkout where the new API is absent.
    if method == "patch":
        return KF.SIFTFeatureScaleSpace(num_features=num_features, rootsift=True, device=device).eval()
    return KF.SIFTFeatureScaleSpace(
        num_features=num_features, rootsift=True, device=device, descriptor_backend="pyramid"
    ).eval()


def _profile_cpu_allocations(run) -> dict[str, int]:
    """Measure tensor allocation churn and peak live bytes in a separate forward.

    These are PyTorch CPU allocator events, not RSS or measured DRAM traffic.
    Native-library workspaces outside the allocator are not included.
    """
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        run()
    with tempfile.TemporaryDirectory(prefix="kornia-sift-profile-") as directory:
        path = Path(directory) / "trace.json"
        profile.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    allocations = sorted(
        (event for event in events if event.get("name") == "[memory]" and event["args"]["Device Type"] == 0),
        key=lambda event: event["ts"],
    )
    if not allocations:
        raise RuntimeError("CPU profiler did not record allocation events")
    live = peak = allocated = 0
    for event in allocations:
        size = event["args"]["Bytes"]
        live += size
        peak = max(peak, live)
        allocated += max(0, size)
    return {"cpu_peak_live_tensor_bytes": peak, "cpu_total_allocated_tensor_bytes": allocated}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--expected-checkout", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--nf", type=int, default=4096)
    parser.add_argument("--methods", nargs="+", choices=("patch", "pyramid", "opencv"), default=["patch", "pyramid"])
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--profile-memory", action="store_true", help="Profile CPU tensor allocations outside timing")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.profile_memory and (args.device != "cpu" or "opencv" in args.methods):
        parser.error("--profile-memory supports PyTorch CPU methods only")
    if "opencv" in args.methods:
        if args.device != "cpu":
            parser.error("OpenCV SIFT is a CPU baseline; run --methods opencv --device cpu")
        if importlib.util.find_spec("cv2") is None:
            raise SystemExit("SKIP: the OpenCV baseline requires optional opencv-python")
        import cv2

        cv2.setNumThreads(1)

    imported_root = Path(kornia.__file__).resolve().parents[1]
    print(f"# interpreter: {sys.executable}\n# kornia: {kornia.__file__}", flush=True)
    if imported_root != args.expected_checkout.resolve():
        raise RuntimeError(f"Wrong Kornia checkout: {imported_root}")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device.type, lambda: None)
    ransac_device = torch.device("cpu") if device.type == "mps" else device

    paths = [_image_path(args.seq, index) for index in range(1, 7)]
    homographies = [args.seq / f"H1to{index}p" for index in range(2, 7)]
    images = [_load_image(path, device) for path in paths]
    meta = run_metadata(device)
    meta.update(
        load=collect_load_metrics(),
        sequence=args.seq.name,
        num_features=args.nf,
        seed=0,
        rootsift=True,
        descriptor_backends=args.methods,
        timing=(
            "entire SIFTFeatureScaleSpace forward (detection + orientation + description); "
            "or native uint8 OpenCV detectAndCompute + RootSIFT; I/O, matching and RANSAC excluded"
        ),
        min_run_time=args.min_run_time,
        quality_only=args.quality_only,
        profile_memory=args.profile_memory,
        matching_ratio=0.8,
        precision_threshold_px=3.0,
        precision_metric="forward GT transfer Euclidean error",
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
        image_conversion="Pillow RGB to L",
        compile=False,
        opencv_regime="uint8 CPU detectAndCompute + NumPy RootSIFT; input/keypoint-to-Torch conversion excluded",
        opencv_parameters={"nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        input_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths + homographies},
        implementation_sha256={
            str(path.relative_to(imported_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("dense_sift.py"),
                Path(kornia.__file__).parent / "feature" / "integrated.py",
                Path(kornia.__file__).parent / "feature" / "scale_space_detector.py",
                Path(kornia.__file__).parent / "feature" / "sift" / "scale_space.py",
                Path(kornia.__file__).parent / "feature" / "siftdesc.py",
                Path(kornia.__file__).parent / "geometry" / "transform" / "pyramid.py",
            )
            if path.is_file()
        },
    )
    print(f"# {meta['git_commit']} | {meta['platform']} | {meta.get('cuda_device', device)}", flush=True)
    print(versions_line(meta), flush=True)

    rows: list[dict] = []
    for method in args.methods:
        torch.manual_seed(0)
        model = (
            cv2.SIFT_create(nfeatures=args.nf, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
            if method == "opencv"
            else _build(method, args.nf, device)
        )
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        print(f"# {method}: {model!r}", flush=True)
        for index, image in enumerate(images, 1):
            if method == "opencv":
                pixels = image[0, 0].mul(255).round().to(torch.uint8).numpy()

                def run(pixels=pixels, model=model):
                    keypoints, desc = model.detectAndCompute(pixels, None)
                    if desc is None:
                        desc = np.empty((0, 128), dtype=np.float32)
                    desc = np.sqrt(desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-8))
                    return keypoints, desc

            else:

                def run(image=image, model=model):
                    return model(image)

            sync()
            start = time.perf_counter()
            extracted = run()
            sync()
            warm_seconds = time.perf_counter() - start
            if method == "opencv":
                keypoints, desc = extracted
                points = torch.tensor([kp.pt for kp in keypoints], dtype=torch.float32).reshape(-1, 2)
                descriptors = torch.from_numpy(desc)[None]
                filled = torch.ones(len(keypoints), dtype=torch.bool)
            else:
                lafs, _responses, descriptors = extracted
                filled = KF.laf_is_filled(lafs)[0]
                lafs = lafs[:, filled]
                descriptors = descriptors[:, filled]
                points = KF.get_laf_center(lafs)[0]
            if not torch.isfinite(descriptors).all():
                raise RuntimeError(f"Nonfinite descriptors for {method}, image {index}")
            median, iqr = float("nan"), float("nan")
            if not args.quality_only:
                median, iqr = time_us(
                    run,
                    min_run_time=max(args.min_run_time, 5.0 * warm_seconds),
                    sync=sync if device.type == "mps" else None,
                )
                if not math.isfinite(median):
                    raise RuntimeError(f"Timing failed for {method}, image {index}")
            outputs.append((points, descriptors[0]))
            row = {
                "op": "OpenCVSIFT" if method == "opencv" else "SIFTFeatureScaleSpace",
                "backend": method,
                "batch": 1,
                "image": index,
                "height": image.shape[-2],
                "width": image.shape[-1],
                "dtype": "uint8" if method == "opencv" else "float32",
                "features": int(filled.sum()),
                "median_us": median,
                "iqr_us": iqr,
                "throughput_per_s": 1e6 / median if math.isfinite(median) else float("nan"),
            }
            if args.profile_memory:
                row.update(_profile_cpu_allocations(run))
            rows.append(row)
            if not args.quality_only:
                print(
                    f"  img{index}: {median / 1000:.2f} ms +/- {iqr / 1000:.2f}; {int(filled.sum())} features",
                    flush=True,
                )
            save_json(args.json, meta, rows)

        points1, descriptors1 = outputs[0]
        for index, (points2, descriptors2) in enumerate(outputs[1:], 2):
            _distance, matches = KF.match_snn(descriptors1, descriptors2, 0.8)
            ground_truth = torch.tensor(np.loadtxt(args.seq / f"H1to{index}p"), device=device, dtype=torch.float32)[
                None
            ]
            projected = transform_points(ground_truth, points1[None])[0]
            error = (projected[matches[:, 0]] - points2[matches[:, 1]]).norm(dim=-1)
            correct = int((error <= 3.0).sum())
            count = len(matches)
            geometry = homography_quality(
                points1[matches[:, 0]].to(ransac_device),
                points2[matches[:, 1]].to(ransac_device),
                ground_truth.to(ransac_device),
                *images[0].shape[-2:],
            )
            rows.append(
                {
                    "op": "matching_quality",
                    "backend": method,
                    "batch": 1,
                    "pair": f"1-{index}",
                    "median_us": None,
                    "iqr_us": None,
                    "throughput_per_s": None,
                    "matches": count,
                    "correct_matches": correct,
                    "precision": correct / count if count else 0.0,
                    "median_transfer_error_px": error.median().item() if count else None,
                    **geometry,
                }
            )
            print(
                f"  1-{index}: {correct}/{count} correct; RANSAC {geometry['ransac_inliers']} inliers, "
                f"corner L1 {geometry['ransac_corner_error_px']}",
                flush=True,
            )
            save_json(args.json, meta, rows)


if __name__ == "__main__":
    main()

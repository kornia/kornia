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

"""Benchmark ScaleSpaceDetector quality and speed on an Oxford-Affine-style sequence.

An Oxford-Affine sequence directory contains:
  img1.png  img2.png  …  imgN.png
  H1to2p    H1to3p    …  H1toNp   (3×3 homography ground truth, space-separated)

Two detector configurations are compared:
  baseline  — ConvSoftArgmax3d (dense soft-NMS, main-branch default)
  new       — AdaptiveQuadInterp3d (sparse true-NMS + iterative subpixel, this branch)

Metrics per image pair (img1 ↔ imgK):
  MAE   — mean corner reprojection error of the RANSAC homography vs ground truth (px)
  inl   — number of RANSAC inliers after SNN matching with SIFT descriptors
  ms    — wall-clock time for detect + describe + match (milliseconds)

Usage::

    # single sequence
    python benchmarks/feature/scale_space_detector.py  \\
        --seq  /data/oxford-affine/graf               \\
        --nf   2000  --device cuda

    # multiple sequences (glob)
    python benchmarks/feature/scale_space_detector.py  \\
        --seq  /data/oxford-affine/graf               \\
        --seq  /data/oxford-affine/leuven             \\
        --nf   2000  --device cuda

    # all subdirectories of a root folder
    python benchmarks/feature/scale_space_detector.py  \\
        --root /data/oxford-affine  --nf 2000  --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

import kornia.feature as KF
from kornia.feature.responses import BlobDoG
from kornia.geometry import RANSAC
from kornia.geometry.subpix import AdaptiveQuadInterp3d, ConvSoftArgmax3d
from kornia.geometry.transform import ScalePyramid

# ── metric ────────────────────────────────────────────────────────────────────


def get_MAE_imgcorners(h: int, w: int, H_gt: np.ndarray, H_est: np.ndarray) -> float:
    """Mean corner reprojection error of H_est vs H_gt (pixels).

    Example::

        H_gt = np.loadtxt(Hgt)
        img1 = K.image_to_tensor(cv2.imread(f1, 0), False) / 255.
        img2 = K.image_to_tensor(cv2.imread(f2, 0), False) / 255.
        h = img1.size(2)
        w = img1.size(3)
        H_out = matchImages(img1, img2)
        MAE = get_MAE_imgcorners(h, w, H_gt, H_out.detach().cpu().numpy())
    """
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H_est).squeeze(1)
    dst_GT = cv2.perspectiveTransform(pts, H_gt).squeeze(1)
    return float(np.abs(dst - dst_GT).sum(axis=1).mean())


# ── I/O helpers ───────────────────────────────────────────────────────────────


def load_gray(path: str, device: torch.device) -> torch.Tensor:
    """Load a grayscale image as a (1,1,H,W) float32 tensor."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    t = torch.from_numpy(img).float().div(255.0)
    return t.unsqueeze(0).unsqueeze(0).to(device)


def find_pairs(seq: Path) -> list[tuple[int, Path, Path]]:
    """Return [(k, img_path, H_path), …] for all pairs img1↔imgK."""
    pairs = []
    for k in range(2, 10):
        img_path = seq / f"img{k}.png"
        h_path = seq / f"H1to{k}p"
        if img_path.exists() and h_path.exists():
            pairs.append((k, img_path, h_path))
    return pairs


# ── detector constructors ─────────────────────────────────────────────────────


def build_detector(subpix: nn.Module, num_features: int, device: torch.device) -> KF.ScaleSpaceDetector:
    return KF.ScaleSpaceDetector(
        num_features=num_features,
        resp_module=BlobDoG(),
        subpix_module=subpix,
        scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=True),
        scale_space_response=True,
        minima_are_also_good=True,
    ).to(device)


def make_baseline(num_features: int, device: torch.device) -> KF.ScaleSpaceDetector:
    """Main-branch default: ConvSoftArgmax3d (dense soft-NMS, no true subpixel)."""
    subpix = ConvSoftArgmax3d(
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        normalized_coordinates=False,
        output_value=True,
    )
    return build_detector(subpix, num_features, device)


def make_new(num_features: int, device: torch.device) -> KF.ScaleSpaceDetector:
    """This branch: AdaptiveQuadInterp3d (sparse true-NMS + iterative subpixel)."""
    subpix = AdaptiveQuadInterp3d(strict_maxima_bonus=0.0)
    return build_detector(subpix, num_features, device)


# ── matching pipeline ─────────────────────────────────────────────────────────


@torch.no_grad()
def match_pair(
    img1: torch.Tensor,
    img2: torch.Tensor,
    detector: KF.ScaleSpaceDetector,
    descriptor: KF.SIFTDescriptor,
    ransac: RANSAC,
) -> tuple[np.ndarray | None, float, int]:
    """Detect → describe → SNN match → RANSAC homography.

    Returns:
        H_est    : (3,3) numpy homography, or None if fewer than 4 matches
        ms       : wall-clock time for detect+describe+match (ms)
        n_inliers: RANSAC inlier count
    """
    t0 = time.perf_counter()

    lafs1, _ = detector(img1)
    lafs2, _ = detector(img2)

    patches1 = KF.extract_patches_from_pyramid(img1, lafs1, 32)
    patches2 = KF.extract_patches_from_pyramid(img2, lafs2, 32)
    B1, N1, CH, PH, PW = patches1.shape
    B2, N2 = patches2.shape[:2]
    desc1 = descriptor(patches1.view(B1 * N1, CH, PH, PW)).view(B1, N1, -1)
    desc2 = descriptor(patches2.view(B2 * N2, CH, PH, PW)).view(B2, N2, -1)

    _, idxs = KF.match_snn(desc1[0], desc2[0], 0.9)
    kp1 = KF.get_laf_center(lafs1)[0, idxs[:, 0]]
    kp2 = KF.get_laf_center(lafs2)[0, idxs[:, 1]]

    ms = (time.perf_counter() - t0) * 1000

    if kp1.shape[0] < 4:
        return None, ms, 0

    H_est, mask = ransac(kp1, kp2)
    n_inliers = int(mask.sum().item()) if mask is not None else 0
    return H_est.cpu().numpy(), ms, n_inliers


# ── per-sequence evaluation ───────────────────────────────────────────────────


def eval_sequence(
    seq: Path,
    det_base: KF.ScaleSpaceDetector,
    det_new: KF.ScaleSpaceDetector,
    descriptor: KF.SIFTDescriptor,
    ransac: RANSAC,
    device: torch.device,
) -> dict:
    """Run both detectors on all pairs in *seq* and return aggregated stats."""
    img1 = load_gray(str(seq / "img1.png"), device)
    h, w = img1.shape[2], img1.shape[3]
    pairs = find_pairs(seq)
    if not pairs:
        raise RuntimeError(f"No valid pairs found in {seq}")

    rows: list[dict] = []
    for k, img_path, h_path in pairs:
        img2 = load_gray(str(img_path), device)
        H_gt = np.loadtxt(str(h_path))

        H_b, ms_b, ni_b = match_pair(img1, img2, det_base, descriptor, ransac)
        H_n, ms_n, ni_n = match_pair(img1, img2, det_new, descriptor, ransac)

        mae_b = get_MAE_imgcorners(h, w, H_gt, H_b) if H_b is not None else float("nan")
        mae_n = get_MAE_imgcorners(h, w, H_gt, H_n) if H_n is not None else float("nan")
        rows.append({"k": k, "mae_b": mae_b, "mae_n": mae_n, "ni_b": ni_b, "ni_n": ni_n, "ms_b": ms_b, "ms_n": ms_n})

    return {"seq": seq.name, "h": h, "w": w, "rows": rows}


def print_sequence_table(stats: dict) -> None:
    seq, rows = stats["seq"], stats["rows"]
    print(f"\n── {seq}  ({stats['h']}×{stats['w']}) ──")
    print(
        f"{'pair':<6} {'MAE_base':>10} {'MAE_new':>10} {'Δ MAE':>8} {'inl_b':>7} {'inl_n':>7} {'ms_b':>8} {'ms_n':>8}"
    )
    print("─" * 68)
    for r in rows:
        delta = r["mae_b"] - r["mae_n"]
        print(
            f"1↔{r['k']:<3}  {r['mae_b']:>10.2f} {r['mae_n']:>10.2f} {delta:>+8.2f} "
            f"{r['ni_b']:>7} {r['ni_n']:>7} {r['ms_b']:>8.1f} {r['ms_n']:>8.1f}"
        )
    print("─" * 68)
    mae_b = np.nanmean([r["mae_b"] for r in rows])
    mae_n = np.nanmean([r["mae_n"] for r in rows])
    ni_b = np.mean([r["ni_b"] for r in rows])
    ni_n = np.mean([r["ni_n"] for r in rows])
    ms_b = np.mean([r["ms_b"] for r in rows])
    ms_n = np.mean([r["ms_n"] for r in rows])
    print(
        f"{'mean':<6}  {mae_b:>10.2f} {mae_n:>10.2f} {mae_b - mae_n:>+8.2f} "
        f"{ni_b:>7.1f} {ni_n:>7.1f} {ms_b:>8.1f} {ms_n:>8.1f}"
    )


def print_summary(all_stats: list[dict]) -> None:
    print("\n" + "═" * 68)
    print("OVERALL SUMMARY")
    print("═" * 68)
    all_mae_b = np.nanmean([np.nanmean([r["mae_b"] for r in s["rows"]]) for s in all_stats])
    all_mae_n = np.nanmean([np.nanmean([r["mae_n"] for r in s["rows"]]) for s in all_stats])
    all_ni_b = np.mean([np.mean([r["ni_b"] for r in s["rows"]]) for s in all_stats])
    all_ni_n = np.mean([np.mean([r["ni_n"] for r in s["rows"]]) for s in all_stats])
    all_ms_b = np.mean([np.mean([r["ms_b"] for r in s["rows"]]) for s in all_stats])
    all_ms_n = np.mean([np.mean([r["ms_n"] for r in s["rows"]]) for s in all_stats])

    print(
        f"  Mean MAE  baseline / new     : {all_mae_b:.2f} / {all_mae_n:.2f} px  "
        f"→ {all_mae_b / max(all_mae_n, 1e-9):.2f}× improvement"
    )
    print(
        f"  Mean inliers  baseline / new : {all_ni_b:.1f} / {all_ni_n:.1f}  "
        f"→ {all_ni_n / max(all_ni_b, 1e-9):.2f}× improvement"
    )
    print(
        f"  Mean time  baseline / new    : {all_ms_b:.1f} / {all_ms_n:.1f} ms  "
        f"→ {all_ms_b / max(all_ms_n, 1e-9):.2f}× speedup"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--seq",
        metavar="DIR",
        action="append",
        default=[],
        help="Path to an Oxford-Affine sequence directory (repeatable).",
    )
    p.add_argument(
        "--root", metavar="DIR", default=None, help="Root folder: every subdirectory is treated as a sequence."
    )
    p.add_argument("--nf", metavar="N", type=int, default=2000, help="Number of features per image (default: 2000).")
    p.add_argument(
        "--device",
        metavar="DEV",
        default=None,
        help="'cpu', 'cuda', or 'cuda:N' (default: cuda if available, else cpu).",
    )
    p.add_argument(
        "--warmup", metavar="N", type=int, default=1, help="Warmup passes on img1↔img2 before timing (default: 1)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Collect sequence directories
    seqs: list[Path] = [Path(s) for s in args.seq]
    if args.root:
        root = Path(args.root)
        seqs += sorted(d for d in root.iterdir() if d.is_dir() and (d / "img1.png").exists())
    if not seqs:
        raise SystemExit("No sequences specified. Use --seq DIR or --root DIR.")

    print(f"device: {device}  num_features: {args.nf}  sequences: {len(seqs)}")
    print("  baseline : ConvSoftArgmax3d  (dense soft-NMS)")
    print("  new      : AdaptiveQuadInterp3d  (sparse NMS + iterative subpixel)")

    det_base = make_baseline(args.nf, device)
    det_new = make_new(args.nf, device)
    descriptor = KF.SIFTDescriptor(32).to(device)
    ransac = RANSAC("homography", inl_th=2.0, max_iter=20, confidence=0.9999)

    # Warmup on first sequence
    first_img1 = load_gray(str(seqs[0] / "img1.png"), device)
    pairs0 = find_pairs(seqs[0])
    if pairs0:
        first_img2 = load_gray(str(pairs0[0][1]), device)
        for _ in range(args.warmup):
            match_pair(first_img1, first_img2, det_base, descriptor, ransac)
            match_pair(first_img1, first_img2, det_new, descriptor, ransac)

    all_stats = []
    for seq in seqs:
        stats = eval_sequence(seq, det_base, det_new, descriptor, ransac, device)
        all_stats.append(stats)
        print_sequence_table(stats)

    if len(all_stats) > 1:
        print_summary(all_stats)


if __name__ == "__main__":
    main()

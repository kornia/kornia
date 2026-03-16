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
"""Benchmark two fully-configurable feature pipelines on Oxford-Affine-style sequences.

An Oxford-Affine sequence directory contains::

    img1.png  img2.png  ...  imgN.png
    H1to2p    H1to3p    ...  H1toNp   (3x3 homography ground truth, whitespace-separated)

Two independent pipelines (A and B) are compared side-by-side.  Each can be either a
``ScaleSpaceDetector``-based pipeline (freely composable response function, subpix module,
descriptor, orientation estimator and affine-shape estimator) or one of the end-to-end
learned detectors (ALIKED, DISK, DeDoDe) which replace the entire pipeline.

Metrics per image pair (img1 vs imgK):

* MAE   -- mean corner reprojection error vs ground-truth homography (px, lower is better).
* inl   -- number of RANSAC inliers (higher is better).
* ms    -- wall-clock time for detect + describe + match (ms, lower is better).

Default A = ``scalespace`` with ``subpix=soft`` (ConvSoftArgmax3d — the legacy default).
Default B = ``scalespace`` with ``subpix=adaptive`` (AdaptiveQuadInterp3d — the new default).
All other components default to ``dog`` / ``sift`` / ``none`` / ``none`` on both sides.

Usage examples::

    # default comparison (ConvSoftArgmax3d vs AdaptiveQuadInterp3d, DoG + SIFT)
    python benchmarks/feature/scale_space_detector.py --seq /data/graf

    # compare two descriptors (both sides share detector)
    python benchmarks/feature/scale_space_detector.py --seq /data/graf \\
        --a-subpix adaptive --a-desc sift \\
        --b-subpix adaptive --b-desc hardnet

    # Hessian + AffNet + OriNet vs DoG + SIFT (both adaptive subpix)
    python benchmarks/feature/scale_space_detector.py --seq /data/graf \\
        --a-resp dog   --a-subpix adaptive --a-desc sift \\
        --b-resp hessian --b-subpix adaptive --b-desc hardnet \\
        --b-aff affnet --b-ori orinet

    # compare ALIKED vs DISK
    python benchmarks/feature/scale_space_detector.py --seq /data/graf \\
        --a-method aliked --b-method disk

    # multiple sequences from a root folder
    python benchmarks/feature/scale_space_detector.py --root /data/oxford-affine --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn

import kornia.feature as KF
from kornia.feature.responses import BlobDoG, BlobDoGSingle, BlobHessian, CornerGFTT, CornerHarris
from kornia.geometry import RANSAC
from kornia.geometry.subpix import (
    AdaptiveQuadInterp3d,
    ConvQuadInterp3d,
    ConvSoftArgmax3d,
    IterativeQuadInterp3d,
)
from kornia.geometry.transform import ScalePyramid


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def get_MAE_imgcorners(h: int, w: int, H_gt, H_est) -> float:
    """Mean corner reprojection error of H_est vs H_gt (pixels).

    Example::

        H_gt = np.loadtxt(Hgt)
        img1 = K.image_to_tensor(cv2.imread(f1, 0), False) / 255.
        img2 = K.image_to_tensor(cv2.imread(f2, 0), False) / 255.
        h, w = img1.size(2), img1.size(3)
        H_out = matchImages(img1, img2)
        MAE = get_MAE_imgcorners(h, w, H_gt, H_out.detach().cpu().numpy())
    """
    pts    = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst    = cv2.perspectiveTransform(pts, H_est).squeeze(1)
    dst_GT = cv2.perspectiveTransform(pts, H_gt).squeeze(1)
    return float(np.abs(dst - dst_GT).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_gray(path: str, device: torch.device) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)


def gray_to_rgb(t: torch.Tensor) -> torch.Tensor:
    return t.expand(-1, 3, -1, -1)


def find_pairs(seq: Path):
    pairs = []
    for k in range(2, 10):
        img_p = seq / f"img{k}.png"
        h_p   = seq / f"H1to{k}p"
        if img_p.exists() and h_p.exists():
            pairs.append((k, img_p, h_p))
    return pairs


# ---------------------------------------------------------------------------
# Component registries
# ---------------------------------------------------------------------------

RESP_REGISTRY: dict[str, tuple] = {
    "dog":        (BlobDoG,                              True,  True),
    "dog_single": (lambda: BlobDoGSingle(1.0, 1.6),      True,  True),
    "hessian":    (BlobHessian,                          False, True),
    "harris":     (lambda: CornerHarris(k=0.04),         False, False),
    "gftt":       (CornerGFTT,                           False, False),
}

SUBPIX_REGISTRY: dict[str, Callable] = {
    "adaptive":  lambda: AdaptiveQuadInterp3d(strict_maxima_bonus=0.0),
    "conv":      lambda: ConvQuadInterp3d(strict_maxima_bonus=0.0),
    "iterative": lambda: IterativeQuadInterp3d(strict_maxima_bonus=0.0),
    "soft":      lambda: ConvSoftArgmax3d(
        (3,3,3),(1,1,1),(1,1,1), normalized_coordinates=False, output_value=True),
}

DESC_REGISTRY: dict[str, tuple] = {
    "sift":     (lambda: KF.SIFTDescriptor(32, rootsift=True),  32),
    "hardnet":  (lambda: KF.HardNet(pretrained=True),            32),
    "hardnet8": (lambda: KF.HardNet8(pretrained=True),           32),
    "hynet":    (lambda: KF.HyNet(pretrained=True),              32),
    "sosnet":   (lambda: KF.SOSNet(pretrained=True),             32),
    "tfeat":    (lambda: KF.TFeat(pretrained=True),              32),
    "mkd":      (lambda: KF.MKDDescriptor(32),                   32),
}

ORI_REGISTRY: dict[str, Callable] = {
    "none":   KF.PassLAF,
    "lap":    lambda: KF.LAFOrienter(32, angle_detector=KF.PatchDominantGradientOrientation(32)),
    "orinet": lambda: KF.LAFOrienter(32, angle_detector=KF.OriNet(pretrained=True)),
}

AFF_REGISTRY: dict[str, Callable] = {
    "none":   KF.PassLAF,
    "patch":  lambda: KF.LAFAffineShapeEstimator(32),
    "affnet": lambda: KF.LAFAffNetShapeEstimator(pretrained=True),
}


# ---------------------------------------------------------------------------
# Extractor wrappers  (unified: img -> (kp N x 2, desc N x D))
# ---------------------------------------------------------------------------

class ScaleSpaceExtractor(nn.Module):
    def __init__(self, detector, desc_module, ori_module, aff_module, patch_size: int):
        super().__init__()
        self.detector, self.desc = detector, desc_module
        self.ori, self.aff = ori_module, aff_module
        self.patch_size = patch_size

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        lafs, _ = self.detector(img)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        patches = KF.extract_patches_from_pyramid(img, lafs, self.patch_size)
        B, N, C, H, W = patches.shape
        desc = self.desc(patches.view(B*N, C, H, W)).view(B, N, -1)
        return KF.get_laf_center(lafs)[0], desc[0]


class ALIKEDExtractor(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        f = self.model(gray_to_rgb(img))[0]
        return f.keypoints, f.descriptors


class DISKExtractor(nn.Module):
    def __init__(self, model, n: int): super().__init__(); self.model, self.n = model, n
    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        f = self.model(gray_to_rgb(img), n=self.n, pad_if_not_divisible=True)[0]
        return f.keypoints, f.descriptors


class DeDoDEExtractor(nn.Module):
    def __init__(self, model, n: int): super().__init__(); self.model, self.n = model, n
    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        kp, _, desc = self.model(gray_to_rgb(img), n=self.n)
        return kp[0], desc[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_extractor(method: str, resp: str, subpix: str, desc: str,
                    ori: str, aff: str, device: torch.device, nf: int) -> nn.Module:
    if method == "scalespace":
        resp_factory, ssr, minima = RESP_REGISTRY[resp]
        detector = KF.ScaleSpaceDetector(
            num_features=nf,
            resp_module=resp_factory(),
            subpix_module=SUBPIX_REGISTRY[subpix](),
            scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=True),
            scale_space_response=ssr, minima_are_also_good=minima,
        ).to(device)
        desc_factory, patch_size = DESC_REGISTRY[desc]
        return ScaleSpaceExtractor(detector,
            desc_factory().to(device),
            ORI_REGISTRY[ori]().to(device),
            AFF_REGISTRY[aff]().to(device),
            patch_size=patch_size)
    if method == "aliked":
        return ALIKEDExtractor(
            KF.ALIKED.from_pretrained("aliked-n16rot", max_num_keypoints=nf).to(device))
    if method == "disk":
        return DISKExtractor(KF.DISK.from_pretrained("depth", device=device), n=nf)
    if method == "dedode":
        return DeDoDEExtractor(KF.DeDoDe.from_pretrained(
            detector_weights="L-upright", descriptor_weights="B-upright").to(device), n=nf)
    raise ValueError(f"Unknown method: {method!r}")


def make_label(method: str, resp: str, subpix: str, desc: str, ori: str, aff: str) -> str:
    if method != "scalespace":
        return method
    parts = [f"resp={resp}", f"subpix={subpix}", f"desc={desc}"]
    if ori != "none": parts.append(f"ori={ori}")
    if aff != "none": parts.append(f"aff={aff}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Matching pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def match_pair(img1, img2, extractor, ransac):
    t0 = time.perf_counter()
    kp1, desc1 = extractor(img1)
    kp2, desc2 = extractor(img2)
    _, idxs = KF.match_snn(desc1, desc2, 0.9)
    ms = (time.perf_counter() - t0) * 1000
    if idxs.shape[0] < 4:
        return None, ms, 0
    H_est, mask = ransac(kp1[idxs[:, 0]], kp2[idxs[:, 1]])
    n_inliers = int(mask.sum().item()) if mask is not None else 0
    return H_est.cpu().numpy(), ms, n_inliers


# ---------------------------------------------------------------------------
# Evaluation + printing
# ---------------------------------------------------------------------------

def eval_sequence(seq: Path, ext_a, ext_b, ransac, device) -> dict:
    img1  = load_gray(str(seq / "img1.png"), device)
    h, w  = img1.shape[2], img1.shape[3]
    pairs = find_pairs(seq)
    if not pairs:
        raise RuntimeError(f"No valid pairs in {seq}")
    rows = []
    for k, img_path, h_path in pairs:
        img2 = load_gray(str(img_path), device)
        H_gt = np.loadtxt(str(h_path))
        H_a, ms_a, ni_a = match_pair(img1, img2, ext_a, ransac)
        H_b, ms_b, ni_b = match_pair(img1, img2, ext_b, ransac)
        mae_a = get_MAE_imgcorners(h, w, H_gt, H_a) if H_a is not None else float("nan")
        mae_b = get_MAE_imgcorners(h, w, H_gt, H_b) if H_b is not None else float("nan")
        rows.append({"k": k, "mae_a": mae_a, "mae_b": mae_b,
                     "ni_a": ni_a, "ni_b": ni_b, "ms_a": ms_a, "ms_b": ms_b})
    return {"seq": seq.name, "h": h, "w": w, "rows": rows}


def _col(rows, key):
    return [r[key] for r in rows]


def print_sequence_table(stats: dict, label_a: str, label_b: str) -> None:
    rows = stats["rows"]
    print(f"\n-- {stats['seq']}  ({stats['h']}x{stats['w']}) --")
    print(f"  A : {label_a}")
    print(f"  B : {label_b}")
    print(f"\n{'pair':<6} {'MAE_A':>10} {'MAE_B':>10} {'B-A':>8} "
          f"{'inl_A':>7} {'inl_B':>7} {'ms_A':>8} {'ms_B':>8}")
    print("-" * 70)
    for r in rows:
        delta = r["mae_b"] - r["mae_a"]
        print(f"1<>{r['k']:<3}  {r['mae_a']:>10.2f} {r['mae_b']:>10.2f} {delta:>+8.2f} "
              f"{r['ni_a']:>7} {r['ni_b']:>7} {r['ms_a']:>8.1f} {r['ms_b']:>8.1f}")
    print("-" * 70)
    mae_a = np.nanmean(_col(rows, "mae_a"))
    mae_b = np.nanmean(_col(rows, "mae_b"))
    print(f"{'mean':<6}  {mae_a:>10.2f} {mae_b:>10.2f} {mae_b - mae_a:>+8.2f} "
          f"{np.mean(_col(rows, 'ni_a')):>7.1f} {np.mean(_col(rows, 'ni_b')):>7.1f} "
          f"{np.mean(_col(rows, 'ms_a')):>8.1f} {np.mean(_col(rows, 'ms_b')):>8.1f}")


def print_summary(all_stats, label_a: str, label_b: str) -> None:
    print("\n" + "=" * 70)
    print(f"OVERALL  A={label_a}  B={label_b}")
    print("=" * 70)
    agg = lambda key: np.nanmean([np.nanmean([r[key] for r in s["rows"]]) for s in all_stats])
    mae_a, mae_b = agg("mae_a"), agg("mae_b")
    ni_a,  ni_b  = agg("ni_a"),  agg("ni_b")
    ms_a,  ms_b  = agg("ms_a"),  agg("ms_b")
    print(f"  MAE     A / B : {mae_a:.2f} / {mae_b:.2f} px"
          f"  -> {mae_a / max(mae_b, 1e-9):.2f}x  (>1 = B has lower error)")
    print(f"  inliers B / A : {ni_b:.1f} / {ni_a:.1f}"
          f"  -> {ni_b / max(ni_a, 1e-9):.2f}x  (>1 = B finds more inliers)")
    print(f"  time    A / B : {ms_a:.1f} / {ms_b:.1f} ms"
          f"  -> {ms_a / max(ms_b, 1e-9):.2f}x  (>1 = B is faster)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_side(p: argparse.ArgumentParser, side: str,
              default_method: str, default_subpix: str) -> None:
    s = side.upper()
    g = p.add_argument_group(f"Method {s}")
    g.add_argument(f"--{side}-method",  dest=f"{side}_method",  default=default_method,
                   choices=["scalespace", "aliked", "disk", "dedode"])
    g.add_argument(f"--{side}-resp",    dest=f"{side}_resp",    default="dog",
                   choices=list(RESP_REGISTRY),    metavar="RESP",
                   help=f"[scalespace only] choices: {list(RESP_REGISTRY)}")
    g.add_argument(f"--{side}-subpix",  dest=f"{side}_subpix",  default=default_subpix,
                   choices=list(SUBPIX_REGISTRY),  metavar="SUBPIX",
                   help=f"[scalespace only] choices: {list(SUBPIX_REGISTRY)}")
    g.add_argument(f"--{side}-desc",    dest=f"{side}_desc",    default="sift",
                   choices=list(DESC_REGISTRY),    metavar="DESC",
                   help=f"[scalespace only] choices: {list(DESC_REGISTRY)}")
    g.add_argument(f"--{side}-ori",     dest=f"{side}_ori",     default="none",
                   choices=list(ORI_REGISTRY),     metavar="ORI",
                   help=f"[scalespace only] none=upright; choices: {list(ORI_REGISTRY)}")
    g.add_argument(f"--{side}-aff",     dest=f"{side}_aff",     default="none",
                   choices=list(AFF_REGISTRY),     metavar="AFF",
                   help=f"[scalespace only] none=circular; choices: {list(AFF_REGISTRY)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("Sequences")
    g.add_argument("--seq",  metavar="DIR", action="append", default=[])
    g.add_argument("--root", metavar="DIR", default=None,
                   help="Root folder; every subdirectory with img1.png is used.")

    # A defaults = legacy (ConvSoftArgmax3d), B defaults = new (AdaptiveQuadInterp3d)
    _add_side(p, "a", default_method="scalespace", default_subpix="soft")
    _add_side(p, "b", default_method="scalespace", default_subpix="adaptive")

    g = p.add_argument_group("Shared")
    g.add_argument("--nf",     metavar="N",   type=int, default=2000)
    g.add_argument("--device", metavar="DEV", default=None)
    g.add_argument("--warmup", metavar="N",   type=int, default=1)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seqs   = [Path(s) for s in args.seq]
    if args.root:
        seqs += sorted(d for d in Path(args.root).iterdir()
                       if d.is_dir() and (d / "img1.png").exists())
    if not seqs:
        raise SystemExit("No sequences.  Use --seq DIR or --root DIR.")

    label_a = make_label(args.a_method, args.a_resp, args.a_subpix,
                         args.a_desc, args.a_ori, args.a_aff)
    label_b = make_label(args.b_method, args.b_resp, args.b_subpix,
                         args.b_desc, args.b_ori, args.b_aff)

    print(f"device: {device}  nf: {args.nf}  sequences: {len(seqs)}")
    print(f"  A : {label_a}")
    print(f"  B : {label_b}")

    ext_a  = build_extractor(args.a_method, args.a_resp, args.a_subpix,
                             args.a_desc, args.a_ori, args.a_aff, device, args.nf)
    ext_b  = build_extractor(args.b_method, args.b_resp, args.b_subpix,
                             args.b_desc, args.b_ori, args.b_aff, device, args.nf)
    ransac = RANSAC("homography", inl_th=2.0, max_iter=20, confidence=0.9999)

    first_img1  = load_gray(str(seqs[0] / "img1.png"), device)
    first_pairs = find_pairs(seqs[0])
    if first_pairs:
        first_img2 = load_gray(str(first_pairs[0][1]), device)
        for _ in range(args.warmup):
            match_pair(first_img1, first_img2, ext_a, ransac)
            match_pair(first_img1, first_img2, ext_b, ransac)

    all_stats = []
    for seq in seqs:
        stats = eval_sequence(seq, ext_a, ext_b, ransac, device)
        all_stats.append(stats)
        print_sequence_table(stats, label_a, label_b)

    if len(all_stats) > 1:
        print_summary(all_stats, label_a, label_b)


if __name__ == "__main__":
    main()

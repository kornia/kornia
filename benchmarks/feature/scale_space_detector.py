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
"""Benchmark a fully-configurable feature pipeline on Oxford-Affine-style sequences.

An Oxford-Affine sequence directory contains::

    img1.png  img2.png  ...  imgN.png
    H1to2p    H1to3p    ...  H1toNp   (3x3 homography ground truth, whitespace-separated)

The pipeline can be either a ``ScaleSpaceDetector``-based combination (freely composable
response function, subpix module, descriptor, orientation estimator and affine-shape
estimator) or one of the end-to-end learned detectors (ALIKED, DISK, DeDoDe).

Output columns per image pair (img1 vs imgK):

* ``error [px]``   -- mean corner reprojection error vs ground-truth homography (lower is better).
* ``inliers [#]``  -- RANSAC inlier count (higher is better).
* ``time [ms]``    -- detect + describe + match wall-clock time (lower is better).

Usage examples::

    # default  (AdaptiveQuadInterp3d + DoG + SIFT)
    python benchmarks/feature/scale_space_detector.py --seq /data/graf

    # swap descriptor
    python benchmarks/feature/scale_space_detector.py --seq /data/graf --desc hardnet

    # Hessian + AffNet + OriNet
    python benchmarks/feature/scale_space_detector.py --seq /data/graf \\
        --resp hessian --aff affnet --ori orinet

    # end-to-end ALIKED (ignores --resp/--subpix/--desc/--ori/--aff)
    python benchmarks/feature/scale_space_detector.py --seq /data/graf --method aliked

    # one named sequence inside a root folder
    python benchmarks/feature/scale_space_detector.py --root /data/oxford-affine --seq graf

    # all sequences inside a root folder
    python benchmarks/feature/scale_space_detector.py --root /data/oxford-affine --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn

import kornia.feature as KF
from kornia.feature.integrated import KeyNetAffNetHardNet, KeyNetHardNet
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
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H_est).squeeze(1)
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
        h_p = seq / f"H1to{k}p"
        if img_p.exists() and h_p.exists():
            pairs.append((k, img_p, h_p))
    return pairs


# ---------------------------------------------------------------------------
# Component registries
# ---------------------------------------------------------------------------

RESP_REGISTRY: dict[str, tuple] = {
    "dog": (BlobDoG, True, True),
    "dog_single": (lambda: BlobDoGSingle(1.0, 1.6), False, True),
    "hessian": (BlobHessian, False, True),
    "harris": (lambda: CornerHarris(k=0.04), False, False),
    "gftt": (CornerGFTT, False, False),
}

SUBPIX_REGISTRY: dict[str, Callable] = {
    "adaptive": lambda: AdaptiveQuadInterp3d(strict_maxima_bonus=0.0, allow_scale_steps=True),
    "conv": lambda: ConvQuadInterp3d(strict_maxima_bonus=0.0),
    "iterative": lambda: IterativeQuadInterp3d(strict_maxima_bonus=0.0),
    "soft": lambda: ConvSoftArgmax3d((3, 3, 3), (1, 1, 1), (1, 1, 1), normalized_coordinates=False, output_value=True),
}

DESC_REGISTRY: dict[str, tuple] = {
    "sift": (lambda: KF.SIFTDescriptor(32, rootsift=True), 32),
    "hardnet": (lambda: KF.HardNet(pretrained=True), 32),
    "hardnet8": (lambda: KF.HardNet8(pretrained=True), 32),
    "hynet": (lambda: KF.HyNet(pretrained=True), 32),
    "sosnet": (lambda: KF.SOSNet(pretrained=True), 32),
    "tfeat": (lambda: KF.TFeat(pretrained=True), 32),
    "mkd": (lambda: KF.MKDDescriptor(32), 32),
}

ORI_REGISTRY: dict[str, Callable] = {
    "none": KF.PassLAF,
    "lap": lambda: KF.LAFOrienter(32, angle_detector=KF.PatchDominantGradientOrientation(32)),
    "orinet": lambda: KF.LAFOrienter(32, angle_detector=KF.OriNet(pretrained=True)),
}

AFF_REGISTRY: dict[str, Callable] = {
    "none": KF.PassLAF,
    "patch": lambda: KF.LAFAffineShapeEstimator(32),
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
        t_det0 = time.perf_counter()
        lafs, _ = self.detector(img)
        if img.device.type == "cuda":
            torch.cuda.synchronize()
        det_ms = (time.perf_counter() - t_det0) * 1000
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        patches = KF.extract_patches_from_pyramid(img, lafs, self.patch_size)
        B, N, C, H, W = patches.shape
        desc = self.desc(patches.view(B * N, C, H, W)).view(B, N, -1)
        return KF.get_laf_center(lafs)[0], desc[0], det_ms


class ALIKEDExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        f = self.model(gray_to_rgb(img))[0]
        return f.keypoints, f.descriptors, None


class DISKExtractor(nn.Module):
    def __init__(self, model, n: int):
        super().__init__()
        self.model, self.n = model, n

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        f = self.model(gray_to_rgb(img), n=self.n, pad_if_not_divisible=True)[0]
        return f.keypoints, f.descriptors, None


class DeDoDEExtractor(nn.Module):
    def __init__(self, model, n: int):
        super().__init__()
        self.model, self.n = model, n

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        kp, _, desc = self.model(gray_to_rgb(img), n=self.n)
        return kp[0], desc[0], None


class DoGHardNet(nn.Module):
    def __init__(self, nf: int):
        super().__init__()
        from kornia_moons.feature import OpenCVDetectorWithAffNetKornia

        kornia_cv2dogaffnet = OpenCVDetectorWithAffNetKornia(
            cv2.SIFT_create(nf, edgeThreshold=-1, contrastThreshold=-1), make_upright=True
        )
        self.det = kornia_cv2dogaffnet
        self.desc = KF.HardNet(pretrained=True)
        self.ori = KF.LAFOrienter(32, angle_detector=KF.OriNet(pretrained=True))

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        lafs, _ = self.det(img)
        lafs = self.ori(lafs, img)
        patches = KF.extract_patches_from_pyramid(img, lafs, 32)
        B, N, C, H, W = patches.shape
        desc = self.desc(patches.view(B * N, C, H, W)).view(B, N, -1)
        return KF.get_laf_center(lafs)[0], desc[0], None


class CV2SIFT(nn.Module):
    def __init__(self, nf: int):
        super().__init__()
        self.det = cv2.SIFT_create(nf, edgeThreshold=-1, contrastThreshold=-1)

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        kpts, desc = self.det.detectAndCompute((255 * img.cpu().numpy().squeeze()).astype(np.uint8), None)
        return torch.from_numpy(np.array([(x.pt[0], x.pt[1]) for x in kpts])), torch.from_numpy(desc), None


class KeyNetExtractor(nn.Module):
    def __init__(self, n: int, ori: str, aff: str, device: torch.device, compile_model: bool = False):
        super().__init__()
        if aff != "none":
            self.feat = KeyNetAffNetHardNet(num_features=n, upright=(ori == "none")).to(device)
        else:
            self.feat = KeyNetHardNet(num_features=n, upright=(ori == "none")).to(device)
        if compile_model:
            det = self.feat.detector
            # model and nms run at 6 different image sizes → dynamic=True avoids recompilation
            det.model = torch.compile(det.model, dynamic=True)
            det.nms = torch.compile(det.nms, dynamic=True)
            # aff/ori/descriptor NOT compiled: extract_patches_from_pyramid (laf.py) contains
            # an .item() call inside a while-loop that creates graph breaks, causing extra
            # sub-graphs that need additional warmup and show as a spike on the first eval pair.

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        lafs, _, desc = self.feat(img)
        return KF.get_laf_center(lafs)[0], desc[0], None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_extractor(
    method: str,
    resp: str,
    subpix: str,
    desc: str,
    ori: str,
    aff: str,
    device: torch.device,
    nf: int,
    compile_modules: Union[bool, List[str]] = False,
) -> nn.Module:
    if method == "scalespace":
        resp_factory, ssr, minima = RESP_REGISTRY[resp]
        subpix_mod = SUBPIX_REGISTRY[subpix]()
        detector = KF.ScaleSpaceDetector(
            num_features=nf,
            resp_module=resp_factory(),
            subpix_module=subpix_mod,
            scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=True),
            scale_space_response=ssr,
            minima_are_also_good=minima,
            compile_modules=["subpix", "resp", "scale_pyr"] if compile_modules else [],
        ).to(device)
        desc_factory, patch_size = DESC_REGISTRY[desc]
        return ScaleSpaceExtractor(
            detector,
            desc_factory().to(device),
            ORI_REGISTRY[ori]().to(device),
            AFF_REGISTRY[aff]().to(device),
            patch_size=patch_size,
        )
    if method == "aliked":
        return ALIKEDExtractor(KF.ALIKED.from_pretrained("aliked-n16rot", max_num_keypoints=nf).to(device))
    if method == "disk":
        return DISKExtractor(KF.DISK.from_pretrained("depth", device=device), n=nf)
    if method == "keynet":
        return KeyNetExtractor(n=nf, ori=ori, aff=aff, device=device, compile_model=bool(compile_modules))
    if method == "opencv_sift_affnet":
        return DoGHardNet(nf).to(device)
    if method == "opencv_sift":
        return CV2SIFT(nf).to(device)
    if method == "dedode":
        return DeDoDEExtractor(
            KF.DeDoDe.from_pretrained(detector_weights="L-upright", descriptor_weights="B-upright").to(device), n=nf
        )
    raise ValueError(f"Unknown method: {method!r}")


def make_label(method: str, resp: str, subpix: str, desc: str, ori: str, aff: str) -> str:
    if method != "scalespace":
        return method
    parts = [f"resp={resp}", f"subpix={subpix}", f"desc={desc}"]
    if ori != "none":
        parts.append(f"ori={ori}")
    if aff != "none":
        parts.append(f"aff={aff}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Matching pipeline
# ---------------------------------------------------------------------------


@torch.no_grad()
def match_pair(img1, img2, extractor, ransac):
    t0 = time.perf_counter()
    kp1, desc1, det_ms1 = extractor(img1)
    kp2, desc2, det_ms2 = extractor(img2)
    _, idxs = KF.match_snn(desc1, desc2, 0.85)
    ms = (time.perf_counter() - t0) * 1000
    det_ms = (det_ms1 + det_ms2) if (det_ms1 is not None and det_ms2 is not None) else None
    if idxs.shape[0] < 4:
        return None, ms, det_ms, 0
    H_est, mask = ransac(kp1[idxs[:, 0]], kp2[idxs[:, 1]])
    n_inliers = int(mask.sum().item()) if mask is not None else 0
    return H_est.cpu().numpy(), ms, det_ms, n_inliers


# ---------------------------------------------------------------------------
# Evaluation + printing
# ---------------------------------------------------------------------------


def eval_sequence(seq: Path, extractor, ransac, device) -> dict:
    img1 = load_gray(str(seq / "img1.png"), device)
    h, w = img1.shape[2], img1.shape[3]
    pairs = find_pairs(seq)
    if not pairs:
        raise RuntimeError(f"No valid pairs in {seq}")
    rows = []
    for k, img_path, h_path in pairs:
        img2 = load_gray(str(img_path), device)
        H_gt = np.loadtxt(str(h_path))
        H, ms, det_ms, ni = match_pair(img1, img2, extractor, ransac)
        mae = get_MAE_imgcorners(h, w, H_gt, H) if H is not None else float("nan")
        rows.append({"k": k, "mae": mae, "ni": ni, "ms": ms, "det_ms": det_ms})
    return {"seq": seq.name, "h": h, "w": w, "rows": rows}


def _col(rows, key):
    return [r[key] for r in rows]


def print_sequence_table(stats: dict, label: str) -> None:
    rows = stats["rows"]
    has_det = rows[0]["det_ms"] is not None
    print(f"\n-- {stats['seq']}  ({stats['h']}x{stats['w']}) --")
    print(f"  method : {label}")
    if has_det:
        print(f"\n{'pair':<6} {'error [px]':>12} {'inliers [#]':>12} {'det [ms]':>10} {'time [ms]':>10}")
        print("-" * 56)
        for r in rows:
            print(f"1<>{r['k']:<3}  {r['mae']:>12.1f} {r['ni']:>12} {r['det_ms']:>10.1f} {r['ms']:>10.1f}")
        print("-" * 56)
        print(
            f"{'mean':<6}  {np.nanmean(_col(rows, 'mae')):>12.1f}"
            f" {np.mean(_col(rows, 'ni')):>12.1f}"
            f" {np.nanmean(_col(rows, 'det_ms')):>10.1f}"
            f" {np.mean(_col(rows, 'ms')):>10.1f}"
        )
    else:
        print(f"\n{'pair':<6} {'error [px]':>12} {'inliers [#]':>12} {'time [ms]':>10}")
        print("-" * 44)
        for r in rows:
            print(f"1<>{r['k']:<3}  {r['mae']:>12.1f} {r['ni']:>12} {r['ms']:>10.1f}")
        print("-" * 44)
        print(
            f"{'mean':<6}  {np.nanmean(_col(rows, 'mae')):>12.1f}"
            f" {np.mean(_col(rows, 'ni')):>12.1f}"
            f" {np.mean(_col(rows, 'ms')):>10.1f}"
        )


def print_summary(all_stats, label: str) -> None:
    print("\n" + "=" * 44)
    print(f"OVERALL  method={label}")
    print("=" * 44)

    def agg(key):
        return np.nanmean([np.nanmean([r[key] for r in s["rows"]]) for s in all_stats])

    print(f"  error [px]  : {agg('mae'):.1f}")
    print(f"  inliers [#] : {agg('ni'):.1f}")
    has_det = all_stats[0]["rows"][0]["det_ms"] is not None
    if has_det:
        print(f"  det [ms]    : {agg('det_ms'):.1f}")
    print(f"  time [ms]   : {agg('ms'):.1f}")


# ---------------------------------------------------------------------------
# Detection-only batch speed benchmark
# ---------------------------------------------------------------------------


def _get_bench_module(extractor: nn.Module) -> Tuple[nn.Module, str]:
    """Return (module_to_time, label) for the speed benchmark.

    For pipeline extractors, isolates the detector so aff/ori/desc are excluded.
    For end-to-end models (ALIKED, DISK, DeDoDe) the full forward is timed.
    """
    if isinstance(extractor, ScaleSpaceExtractor):
        return extractor.detector, "detection only"
    if isinstance(extractor, KeyNetExtractor):
        return extractor.feat.detector, "detection only"
    return extractor, "full forward"


def _time_fn(fn: Callable, n_iter: int, dev: torch.device) -> float:
    """Return mean wall-clock time in ms over n_iter calls."""
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def run_speed_benchmark(
    extractor: nn.Module,
    img: torch.Tensor,
    batch_sizes: Tuple[int, ...] = (1, 4, 8),
    n_iter_gpu: int = 20,
    n_iter_cpu: int = 5,
    warmup: int = 3,
) -> None:
    """Print a batch-size x device timing table.

    Args:
        extractor: the extractor built by :func:`build_extractor`.
        img: single-image tensor ``(1, C, H, W)`` on any device (moved internally).
        batch_sizes: batch sizes to benchmark.
        n_iter_gpu: number of timed iterations on GPU.
        n_iter_cpu: number of timed iterations on CPU.
        warmup: warm-up iterations (not timed).
    """
    mod, what = _get_bench_module(extractor)

    try:
        orig_dev = next(mod.parameters()).device
    except StopIteration:
        orig_dev = img.device

    dev_map: Dict[str, torch.device] = {"cpu": torch.device("cpu")}
    if torch.cuda.is_available():
        dev_map["gpu"] = torch.device("cuda")

    col_w = 14
    print(f"\n-- Speed benchmark ({what}, ms / call) --")
    print(f"{'bs':<6}" + "".join(f"{k:>{col_w}}" for k in dev_map))
    print("-" * (6 + col_w * len(dev_map)))

    for bs in batch_sizes:
        row = f"{bs:<6}"
        for _, dev in dev_map.items():
            n_iter = n_iter_gpu if dev.type == "cuda" else n_iter_cpu
            mod.to(dev)
            img_b = img[0:1].expand(bs, -1, -1, -1).contiguous().to(dev)
            try:
                with torch.no_grad():
                    for _ in range(warmup):
                        mod(img_b)
                    ms = _time_fn(lambda: mod(img_b), n_iter, dev)  # noqa: B023
                row += f"{ms:>{col_w - 3}.1f} ms"
            except Exception:
                row += f"{'N/A':>{col_w}}"
        print(row)

    mod.to(orig_dev)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("Sequences")
    g.add_argument("--seq", metavar="DIR", action="append", default=[])
    g.add_argument(
        "--root",
        metavar="DIR",
        default=None,
        help="Base folder. Alone: enumerates all subdirs with img1.png. With --seq name: resolves to root/name.",
    )

    g = p.add_argument_group("Method")
    g.add_argument(
        "--method",
        default="scalespace",
        choices=["scalespace", "aliked", "disk", "dedode", "keynet", "opencv_sift_affnet", "opencv_sift"],
    )
    g.add_argument(
        "--resp",
        default="dog",
        choices=list(RESP_REGISTRY),
        metavar="RESP",
        help=f"[scalespace only] choices: {list(RESP_REGISTRY)}",
    )
    g.add_argument(
        "--subpix",
        default="adaptive",
        choices=list(SUBPIX_REGISTRY),
        metavar="SUBPIX",
        help=f"[scalespace only] choices: {list(SUBPIX_REGISTRY)}",
    )
    g.add_argument(
        "--desc",
        default="sift",
        choices=list(DESC_REGISTRY),
        metavar="DESC",
        help=f"[scalespace only] choices: {list(DESC_REGISTRY)}",
    )
    g.add_argument(
        "--ori",
        default="none",
        choices=list(ORI_REGISTRY),
        metavar="ORI",
        help=f"[scalespace only] none=upright; choices: {list(ORI_REGISTRY)}",
    )
    g.add_argument(
        "--aff",
        default="none",
        choices=list(AFF_REGISTRY),
        metavar="AFF",
        help=f"[scalespace only] none=circular; choices: {list(AFF_REGISTRY)}",
    )

    g = p.add_argument_group("Shared")
    g.add_argument("--nf", metavar="N", type=int, default=4096)
    g.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the extractor (if supported by the method and PyTorch version)",
    )
    g.add_argument("--device", metavar="DEV", default=None)
    g.add_argument("--warmup", metavar="N", type=int, default=1)
    g.add_argument(
        "--speed-bench",
        action="store_true",
        help="Run detection-only speed benchmark (bs=1,4,8 on cpu+gpu) after the matching eval.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(args.root) if args.root else None
    if args.seq:
        # Resolve each --seq name: use as-is if it exists, else try root/name
        seqs = []
        for s in args.seq:
            p = Path(s)
            if not p.exists() and root is not None:
                p = root / s
            seqs.append(p)
    elif root is not None:
        # --root only: enumerate all subdirectories that look like sequences
        seqs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "img1.png").exists())
    else:
        seqs = []
    if not seqs:
        raise SystemExit("No sequences.  Use --seq DIR  or  --root DIR  or  --root DIR --seq name.")

    label = make_label(args.method, args.resp, args.subpix, args.desc, args.ori, args.aff)
    print(f"device: {device}  nf: {args.nf}  sequences: {len(seqs)}")
    print(f"  method : {label}")

    extractor = build_extractor(
        args.method,
        args.resp,
        args.subpix,
        args.desc,
        args.ori,
        args.aff,
        device,
        args.nf,
        compile_modules=args.compile,
    )
    ransac = RANSAC("homography", inl_th=2.0, max_iter=10, batch_size=8196, confidence=0.9999, seed=3407)

    first_img1 = load_gray(str(seqs[0] / "img1.png"), device)
    first_pairs = find_pairs(seqs[0])
    if first_pairs:
        first_img2 = load_gray(str(first_pairs[0][1]), device)
        for _ in range(args.warmup):
            match_pair(first_img1, first_img2, extractor, ransac)

    all_stats = []
    for seq in seqs:
        stats = eval_sequence(seq, extractor, ransac, device)
        all_stats.append(stats)
        print_sequence_table(stats, label)

    if len(all_stats) > 1:
        print_summary(all_stats, label)

    if args.speed_bench:
        bench_img = load_gray(str(seqs[0] / "img1.png"), torch.device("cpu"))
        run_speed_benchmark(extractor, bench_img)


if __name__ == "__main__":
    main()

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

"""Benchmark and quality comparison: kornia ScaleSpaceDetector (BlobHessian) vs pyhesaff.

Metrics:
  - Number of keypoints
  - Scale distribution (in comparable pixel units)
  - Repeatability: fraction of keypoints that survive a homographic transformation
  - Speed: wall-clock on CPU and GPU for 1 image and batch-of-8

EVD dataset layout:
  EVD/1/<name>.png  — reference image (moderate transformation, severity 1)
  EVD/2/<name>.png  — query image     (more severe transformation, severity 2)
  EVD/h/<name>.txt  — 3×3 homography  mapping reference → query pixels

Scale conventions:
  - pyhesaff [x, y, a11, a12, a22, ori]:  scale_px = HESAFF_MRSIZE * (a11*a22)^0.25
  - kornia    LAF (B,N,2,3):               scale_px = laf[b,n,0,0]  (already mr_size*sigma)
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pyhesaff
import torch

import kornia
from kornia.feature import ScaleSpaceDetector, get_laf_scale
from kornia.geometry.subpix import IterativeQuadInterp3d

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVD_ROOT = Path("/home/ducha-aiki/dev/small_datasets/EVD")
REF_DIR = EVD_ROOT / "1"
QRY_DIR = EVD_ROOT / "2"
HOM_DIR = EVD_ROOT / "h"

PIXEL_THRESHOLD = 3.0  # pixels: max reprojection error for "repeatable"
N_FEATS = 2048  # max keypoints to request
HESAFF_MRSIZE = 5.19615  # pyhesaff default mrSize = 3*sqrt(3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_gray_u8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_gray_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """(1,1,H,W) float32 in [0,1]."""
    img = load_gray_u8(path)
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0).to(device)


def load_homography(path: Path) -> np.ndarray:
    return np.loadtxt(str(path))


def project_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project (N,2) [x,y] through homography H."""
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    proj = (H @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


def repeatability(
    kpts1: np.ndarray,  # (N1,2) [x,y]
    kpts2: np.ndarray,  # (N2,2)
    H12: np.ndarray,  # maps img1 → img2
    img2_hw: tuple[int, int],
    threshold: float = PIXEL_THRESHOLD,
) -> float:
    """Position-only repeatability: fraction of kpts1 that project within
    `threshold` pixels of a kpts2 detection.
    """
    if len(kpts1) == 0 or len(kpts2) == 0:
        return 0.0

    proj1 = project_points(kpts1, H12)

    h2, w2 = img2_hw
    inside = (proj1[:, 0] >= 0) & (proj1[:, 0] < w2) & (proj1[:, 1] >= 0) & (proj1[:, 1] < h2)
    proj1_in = proj1[inside]

    if len(proj1_in) == 0:
        return 0.0

    diff = proj1_in[:, None, :] - kpts2[None, :, :]  # (M, N2, 2)
    min_dists = np.linalg.norm(diff, axis=2).min(axis=1)  # (M,)
    n_rep = int((min_dists < threshold).sum())
    return n_rep / max(min(len(kpts1), len(kpts2)), 1)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def make_kornia_hessian(device: torch.device) -> ScaleSpaceDetector:
    return (
        ScaleSpaceDetector(
            N_FEATS,
            resp_module=kornia.feature.BlobHessian(),
            nms_module=IterativeQuadInterp3d(strict_maxima_bonus=0.0),
            scale_pyr_module=kornia.geometry.ScalePyramid(3, 1.6, 32, double_image=True),
            mr_size=HESAFF_MRSIZE,
            minima_are_also_good=True,
        )
        .to(device)
        .eval()
    )


def detect_kornia(det: ScaleSpaceDetector, img_t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,2) [x,y] and (N,) scale_px arrays."""
    with torch.no_grad():
        lafs, _ = det(img_t)
    pts = lafs[0, :, :, 2].cpu().numpy()  # (N,2) centers [x,y]
    scales = get_laf_scale(lafs)[0, :, 0, 0].cpu().numpy()  # (N,) mr_size*sigma
    return pts, scales


def detect_pyhesaff(img_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,2) [x,y] and (N,) scale_px arrays."""
    kpts, _ = pyhesaff.detect_feats(str(img_path))
    if len(kpts) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    pts = kpts[:, :2]  # (N,2) [x,y]
    a11, a22 = kpts[:, 2], kpts[:, 4]
    scale_px = HESAFF_MRSIZE * (np.maximum(a11 * a22, 1e-9) ** 0.25)
    return pts, scale_px


# ---------------------------------------------------------------------------
# Evaluation on EVD
# ---------------------------------------------------------------------------


def evaluate(device: torch.device) -> None:
    det = make_kornia_hessian(device)
    image_names = sorted(p.stem for p in REF_DIR.glob("*.png"))

    print("\nEVD repeatability — kornia BlobHessian vs pyhesaff")
    print(f"Device: {device}  |  threshold: {PIXEL_THRESHOLD}px  |  N_FEATS: {N_FEATS}")
    print()
    hdr = f"{'Image':<12}  {'Rep(Kor)':>9}  {'Rep(Pyhes)':>10}  "
    hdr += f"{'N(Kor)':>7}  {'N(Pyh)':>7}  {'Scl(Kor)':>9}  {'Scl(Pyh)':>9}"
    print(hdr)
    print("-" * 78)

    rep_kor, rep_pyh = [], []
    all_scales_kor, all_scales_pyh = [], []

    for name in image_names:
        img1_p = REF_DIR / f"{name}.png"
        img2_p = QRY_DIR / f"{name}.png"
        hom_p = HOM_DIR / f"{name}.txt"
        if not img2_p.exists() or not hom_p.exists():
            continue

        H12 = load_homography(hom_p)
        img2_np = load_gray_u8(img2_p)

        img1_t = load_gray_tensor(img1_p, device)
        img2_t = load_gray_tensor(img2_p, device)

        pts1_k, sc1_k = detect_kornia(det, img1_t)
        pts2_k, _ = detect_kornia(det, img2_t)
        pts1_p, sc1_p = detect_pyhesaff(img1_p)
        pts2_p, _ = detect_pyhesaff(img2_p)

        r_k = repeatability(pts1_k, pts2_k, H12, img2_np.shape[:2])
        r_p = repeatability(pts1_p, pts2_p, H12, img2_np.shape[:2])

        rep_kor.append(r_k)
        rep_pyh.append(r_p)
        all_scales_kor.extend(sc1_k.tolist())
        all_scales_pyh.extend(sc1_p.tolist())

        med_scl_k = float(np.median(sc1_k)) if len(sc1_k) else 0.0
        med_scl_p = float(np.median(sc1_p)) if len(sc1_p) else 0.0
        print(
            f"{name:<12}  {r_k:>9.3f}  {r_p:>10.3f}  "
            f"{len(pts1_k):>7}  {len(pts1_p):>7}  {med_scl_k:>9.1f}  {med_scl_p:>9.1f}"
        )

    print("-" * 78)
    print(
        f"{'MEAN':<12}  {np.mean(rep_kor):>9.3f}  {np.mean(rep_pyh):>10.3f}  "
        f"{'':>7}  {'':>7}  "
        f"{np.median(all_scales_kor):>9.1f}  {np.median(all_scales_pyh):>9.1f}"
    )
    print()
    q_kor = np.percentile(all_scales_kor, [10, 25, 50, 75, 90])
    q_pyh = np.percentile(all_scales_pyh, [10, 25, 50, 75, 90])
    print("Scale percentiles (px)   [p10  p25  p50  p75  p90]")
    print(f"  Kornia BlobHessian:  {np.round(q_kor, 1)}")
    print(f"  Pyhesaff:            {np.round(q_pyh, 1)}")


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------


def benchmark_speed() -> None:
    print("\n" + "=" * 70)
    print("SPEED BENCHMARK  (640×480 grayscale image)")
    print("=" * 70)

    img_path = next(REF_DIR.glob("*.png"))
    img_np = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_np = cv2.resize(img_np, (640, 480))
    img_t_cpu = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    n_warmup, n_runs = 3, 20

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    def sync(dev):
        if dev.type == "cuda":
            torch.cuda.synchronize()

    for dev in devices:
        det = make_kornia_hessian(dev)
        img1 = img_t_cpu.to(dev)
        img8 = img1.repeat(8, 1, 1, 1)

        for _ in range(n_warmup):
            with torch.no_grad():
                det(img1)
        sync(dev)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                det(img1)
            sync(dev)
        t1 = (time.perf_counter() - t0) / n_runs * 1000

        for _ in range(n_warmup):
            with torch.no_grad():
                det(img8)
        sync(dev)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                det(img8)
            sync(dev)
        t8 = (time.perf_counter() - t0) / n_runs * 1000

        print(f"\n  kornia BlobHessian [{dev}]")
        print(f"    batch=1: {t1:7.1f} ms    batch=8: {t8:7.1f} ms  ({t8 / 8:.1f} ms/img)")

    # pyhesaff (CPU only, single image)
    for _ in range(n_warmup):
        pyhesaff.detect_feats(str(img_path))

    t0 = time.perf_counter()
    for _ in range(n_runs):
        pyhesaff.detect_feats(str(img_path))
    t_pyh = (time.perf_counter() - t0) / n_runs * 1000
    print("\n  pyhesaff [cpu] (single image, no batching)")
    print(f"    batch=1: {t_pyh:7.1f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(device)
    benchmark_speed()

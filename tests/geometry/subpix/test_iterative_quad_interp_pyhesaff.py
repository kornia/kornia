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

"""Reference tests for iterative_quad_interp3d against pyhesaff.

pyhesaff is a Python binding to the C++ HessAff detector.  The C++ function
``HessianDetector::localizeKeypoint`` is the direct reference for the
algorithm implemented in ``iterative_quad_interp3d``.  These tests:

1. Verify that the single-step localizeKeypoint formula (implemented here in
   pure Python/NumPy) matches our batched PyTorch implementation.
2. Verify that subpixel positions recovered from a scale-normalised Hessian
   response map agree with pyhesaff's end-to-end detections on the same
   Gaussian-blob image, to within ~0.05 pixels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kornia.geometry.subpix.nms import nms3d
from kornia.geometry.subpix.spatial_soft_argmax import iterative_quad_interp3d

pyhesaff = pytest.importorskip("pyhesaff", reason="pyhesaff not installed")

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _gaussian_blur_np(img: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Separable Gaussian blur implemented with PyTorch conv2d (reflect padding)."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    kx = torch.from_numpy(k).view(1, 1, 1, -1)
    ky = torch.from_numpy(k).view(1, 1, -1, 1)
    pad = len(k) // 2
    out = F.conv2d(F.pad(t, (pad, pad, 0, 0), "reflect"), kx)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), "reflect"), ky)
    return out.squeeze().numpy()


def _hessian_det_normalized(img: np.ndarray, sigma: float) -> np.ndarray:
    """Scale-normalised Hessian determinant at a single scale.

    Uses 3-point finite differences (matching the C++ reference) and multiplies
    by sigma^4 so that the response is comparable across scales and peaks at the
    characteristic scale of the feature.
    """
    L = _gaussian_blur_np(img, sigma)
    Lxx = np.zeros_like(L)
    Lxx[:, 1:-1] = L[:, 2:] - 2 * L[:, 1:-1] + L[:, :-2]
    Lyy = np.zeros_like(L)
    Lyy[1:-1, :] = L[2:, :] - 2 * L[1:-1, :] + L[:-2, :]
    Lxy = np.zeros_like(L)
    Lxy[1:-1, 1:-1] = 0.25 * (L[2:, 2:] - L[2:, :-2] - L[:-2, 2:] + L[:-2, :-2])
    return (sigma**4 * (Lxx * Lyy - Lxy**2)).astype(np.float32)


def _build_hessian_stack(img: np.ndarray, sigmas: list[float]) -> torch.Tensor:
    """Build a (1, 1, len(sigmas), H, W) scale-normalised Hessian response stack."""
    layers = [_hessian_det_normalized(img, s) for s in sigmas]
    return torch.from_numpy(np.stack(layers)).unsqueeze(0).unsqueeze(0)


def _make_blob_image(cx: float, cy: float, sigma_blob: float, H: int = 128, W: int = 128) -> np.ndarray:
    """Float32 Gaussian blob image (values in [0, 1]) at subpixel position (cx, cy)."""
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_blob**2)).astype(np.float32)


def _pyhesaff_detect_closest(img_u8: np.ndarray, cx: float, cy: float) -> tuple[float, float] | None:
    """Run pyhesaff and return the keypoint (x, y) closest to (cx, cy), or None."""
    kpts, _ = pyhesaff.detect_feats_in_image(img_u8, affine_invariance=False)
    if len(kpts) == 0:
        return None
    dists = np.sqrt((kpts[:, 0] - cx) ** 2 + (kpts[:, 1] - cy) ** 2)
    best = kpts[dists.argmin()]
    return float(best[0]), float(best[1])


# pyhesaff's scale space: first octave inner scales that produce a 3D NMS peak
# for sigma_blob ≈ 2.5 (verified experimentally).
_N_SCALES = 3
_INIT_SIGMA = 1.6
_SIGMAS_ALL = [_INIT_SIGMA * 2 ** (k / _N_SCALES) for k in range(5)]
# Middle three scales create a proper 3D NMS maximum for sigma_blob ≈ 2.5:
# sigmas[1]=2.016, sigmas[2]=2.540, sigmas[3]=3.200
_SIGMAS_TRIPLET = _SIGMAS_ALL[1:4]
_SIGMA_BLOB = 2.5  # blob size chosen so the peak is interior in the triplet


# ---------------------------------------------------------------------------
# Reference formula: Python port of C++ localizeKeypoint (one iteration)
# ---------------------------------------------------------------------------


def _localizeKeypoint_ref(
    low: np.ndarray, cur: np.ndarray, high: np.ndarray, r: int, c: int
) -> tuple[float, float, float, bool]:
    """Single step of the C++ HessianDetector::localizeKeypoint.

    Solves ``H * [shift_x, shift_y, shift_s]^T = -[dx, dy, ds]^T`` and returns
    ``(shift_x, shift_y, shift_s, is_valid)``.  The coordinate convention matches
    the C++ code: x = column direction, y = row direction, s = scale direction.
    """
    c000 = cur[r, c]
    dx = 0.5 * (cur[r, c + 1] - cur[r, c - 1])
    dy = 0.5 * (cur[r + 1, c] - cur[r - 1, c])
    ds = 0.5 * (high[r, c] - low[r, c])

    dxx = cur[r, c - 1] - 2 * c000 + cur[r, c + 1]
    dyy = cur[r - 1, c] - 2 * c000 + cur[r + 1, c]
    dss = low[r, c] - 2 * c000 + high[r, c]
    dxy = 0.25 * (cur[r + 1, c + 1] - cur[r + 1, c - 1] - cur[r - 1, c + 1] + cur[r - 1, c - 1])
    dxs = 0.25 * (high[r, c + 1] - high[r, c - 1] - low[r, c + 1] + low[r, c - 1])
    dys = 0.25 * (high[r + 1, c] - high[r - 1, c] - low[r + 1, c] + low[r - 1, c])

    A = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]], dtype=np.float64)
    b = np.array([-dx, -dy, -ds], dtype=np.float64)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, False
    return float(x[0]), float(x[1]), float(x[2]), True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIterativeQuadInterp3dVsRefFormula:
    """Compare our function against a direct Python port of the C++ formula."""

    def _run(self, cx: float, cy: float, device: torch.device, dtype: torch.dtype) -> None:
        img = _make_blob_image(cx, cy, _SIGMA_BLOB)
        resp_t = _build_hessian_stack(img, _SIGMAS_TRIPLET).to(device=device, dtype=dtype)

        # Make sure there is an NMS peak
        assert nms3d(resp_t, (3, 3, 3), True).sum().item() >= 1, "no NMS peak found"

        # Our function (one iteration so it stays at the integer center)
        coords, _ = iterative_quad_interp3d(resp_t, n_iters=1, strict_maxima_bonus=0)

        # Reference: apply C++ formula to the numpy arrays
        low_np, cur_np, high_np = [_hessian_det_normalized(img, s) for s in _SIGMAS_TRIPLET]
        h_peak, w_peak = np.unravel_index(cur_np.argmax(), cur_np.shape)
        ref_sx, ref_sy, ref_ss, valid = _localizeKeypoint_ref(low_np, cur_np, high_np, h_peak, w_peak)
        assert valid, "reference formula failed to solve"

        d_peak = 1  # middle scale in the triplet
        ours_x = coords[0, 0, 1, d_peak, h_peak, w_peak].item()  # width (x)
        ours_y = coords[0, 0, 2, d_peak, h_peak, w_peak].item()  # height (y)
        ours_s = coords[0, 0, 0, d_peak, h_peak, w_peak].item()  # scale

        tol = 1e-3
        assert abs(ours_x - (w_peak + ref_sx)) < tol, f"x mismatch: ours={ours_x:.5f} ref={w_peak + ref_sx:.5f}"
        assert abs(ours_y - (h_peak + ref_sy)) < tol, f"y mismatch: ours={ours_y:.5f} ref={h_peak + ref_sy:.5f}"
        assert abs(ours_s - (d_peak + ref_ss)) < tol, f"s mismatch: ours={ours_s:.5f} ref={d_peak + ref_ss:.5f}"

    @pytest.mark.parametrize("cx,cy", [(64.4, 63.7), (32.3, 31.8), (64.0, 64.0), (48.7, 50.2)])
    def test_single_step_matches_cpp_formula_float32(self, cx: float, cy: float) -> None:
        self._run(cx, cy, torch.device("cpu"), torch.float32)

    @pytest.mark.parametrize("cx,cy", [(64.4, 63.7), (32.3, 31.8)])
    def test_single_step_matches_cpp_formula_float64(self, cx: float, cy: float) -> None:
        self._run(cx, cy, torch.device("cpu"), torch.float64)


class TestIterativeQuadInterp3dVsPyhesaff:
    """Compare subpixel positions against pyhesaff end-to-end detections.

    pyhesaff uses the same C++ localizeKeypoint function internally.  Both
    operate on the same Gaussian-blob image (different scale-space pipelines)
    so positions should agree to within ~0.05 pixels.
    """

    # Tolerance in pixels for position agreement between our function and pyhesaff.
    _TOL = 0.05

    def _run(self, cx: float, cy: float, device: torch.device, dtype: torch.dtype) -> None:
        img = _make_blob_image(cx, cy, _SIGMA_BLOB)
        resp_t = _build_hessian_stack(img, _SIGMAS_TRIPLET).to(device=device, dtype=dtype)

        nms_mask = nms3d(resp_t, (3, 3, 3), True)
        n_peaks = nms_mask.sum().item()
        assert n_peaks >= 1, "no NMS peak — choose a blob size with a 3D interior maximum"

        coords, _ = iterative_quad_interp3d(resp_t, strict_maxima_bonus=0)

        # Our detected position: read coords at the 3D NMS peak
        d_peak = 1
        low_np, cur_np, _ = [_hessian_det_normalized(img, s) for s in _SIGMAS_TRIPLET]
        h_peak, w_peak = np.unravel_index(cur_np.argmax(), cur_np.shape)
        ours_x = coords[0, 0, 1, d_peak, h_peak, w_peak].item()
        ours_y = coords[0, 0, 2, d_peak, h_peak, w_peak].item()

        # pyhesaff reference
        img_u8 = (img * 200).clip(0, 255).astype(np.uint8)
        result = _pyhesaff_detect_closest(img_u8, cx, cy)
        assert result is not None, "pyhesaff found no keypoints"
        ph_x, ph_y = result

        assert abs(ours_x - ph_x) < self._TOL, (
            f"x mismatch: ours={ours_x:.4f} pyhesaff={ph_x:.4f} true={cx}"
        )
        assert abs(ours_y - ph_y) < self._TOL, (
            f"y mismatch: ours={ours_y:.4f} pyhesaff={ph_y:.4f} true={cy}"
        )

    @pytest.mark.parametrize("cx,cy", [(64.4, 63.7), (32.3, 31.8), (64.0, 64.0)])
    def test_blob_position_float32(self, cx: float, cy: float) -> None:
        self._run(cx, cy, torch.device("cpu"), torch.float32)

    @pytest.mark.parametrize("cx,cy", [(64.4, 63.7), (32.3, 31.8)])
    def test_blob_position_float64(self, cx: float, cy: float) -> None:
        self._run(cx, cy, torch.device("cpu"), torch.float64)

    def test_integer_peak_exact(self) -> None:
        """Blob at integer position — both should return exactly the integer coords."""
        cx, cy = 64.0, 64.0
        img = _make_blob_image(cx, cy, _SIGMA_BLOB)
        resp_t = _build_hessian_stack(img, _SIGMAS_TRIPLET)
        coords, _ = iterative_quad_interp3d(resp_t, strict_maxima_bonus=0)
        d_peak = 1
        cur_np = _hessian_det_normalized(img, _SIGMAS_TRIPLET[1])
        h_peak, w_peak = np.unravel_index(cur_np.argmax(), cur_np.shape)
        ours_x = coords[0, 0, 1, d_peak, h_peak, w_peak].item()
        ours_y = coords[0, 0, 2, d_peak, h_peak, w_peak].item()
        assert abs(ours_x - cx) < 1e-3
        assert abs(ours_y - cy) < 1e-3

        # pyhesaff also returns integer coords
        img_u8 = (img * 200).clip(0, 255).astype(np.uint8)
        result = _pyhesaff_detect_closest(img_u8, cx, cy)
        assert result is not None
        assert abs(result[0] - cx) < 1e-2
        assert abs(result[1] - cy) < 1e-2

    def test_multiple_blobs(self) -> None:
        """Two blobs at different subpixel positions — verify both are recovered."""
        H, W = 256, 256
        blobs = [(64.3, 63.8), (192.6, 190.1)]
        img = np.zeros((H, W), dtype=np.float32)
        for bx, by in blobs:
            img += _make_blob_image(bx, by, _SIGMA_BLOB, H=H, W=W)
        img = img.clip(0, 1)

        resp_t = _build_hessian_stack(img, _SIGMAS_TRIPLET)
        coords, _ = iterative_quad_interp3d(resp_t, strict_maxima_bonus=0)

        cur_np = _hessian_det_normalized(img, _SIGMAS_TRIPLET[1])
        d_peak = 1

        # pyhesaff reference
        img_u8 = (img * 200).clip(0, 255).astype(np.uint8)
        kpts, _ = pyhesaff.detect_feats_in_image(img_u8, affine_invariance=False)

        for bx, by in blobs:
            # Our detected position
            h_int, w_int = int(round(by)), int(round(bx))
            ours_x = coords[0, 0, 1, d_peak, h_int, w_int].item()
            ours_y = coords[0, 0, 2, d_peak, h_int, w_int].item()

            # Closest pyhesaff detection
            if len(kpts) > 0:
                dists = np.sqrt((kpts[:, 0] - bx) ** 2 + (kpts[:, 1] - by) ** 2)
                best = kpts[dists.argmin()]
                ph_x, ph_y = float(best[0]), float(best[1])
                assert abs(ours_x - ph_x) < 0.1, f"blob ({bx},{by}): x ours={ours_x:.3f} ph={ph_x:.3f}"
                assert abs(ours_y - ph_y) < 0.1, f"blob ({bx},{by}): y ours={ours_y:.3f} ph={ph_y:.3f}"

    def test_pyhesaff_accuracy_vs_true(self) -> None:
        """Verify that pyhesaff itself accurately recovers the true blob position.

        This is a sanity-check that pyhesaff's subpixel localisation works on
        our synthetic images, which justifies using it as a reference.
        """
        for cx, cy in [(64.4, 63.7), (32.3, 31.8), (96.2, 95.5)]:
            img = _make_blob_image(cx, cy, _SIGMA_BLOB)
            img_u8 = (img * 200).clip(0, 255).astype(np.uint8)
            result = _pyhesaff_detect_closest(img_u8, cx, cy)
            assert result is not None, f"pyhesaff found no kpts for blob at ({cx},{cy})"
            ph_x, ph_y = result
            assert abs(ph_x - cx) < 0.05, f"pyhesaff x error {abs(ph_x-cx):.4f} for blob at x={cx}"
            assert abs(ph_y - cy) < 0.05, f"pyhesaff y error {abs(ph_y-cy):.4f} for blob at y={cy}"

    def test_our_accuracy_vs_true(self) -> None:
        """Verify that our function accurately recovers the true blob position.

        Complements test_pyhesaff_accuracy_vs_true: both should find the peak
        to within a few hundredths of a pixel.
        """
        for cx, cy in [(64.4, 63.7), (32.3, 31.8), (96.2, 95.5)]:
            img = _make_blob_image(cx, cy, _SIGMA_BLOB)
            resp_t = _build_hessian_stack(img, _SIGMAS_TRIPLET)
            coords, _ = iterative_quad_interp3d(resp_t, strict_maxima_bonus=0)
            d_peak = 1
            cur_np = _hessian_det_normalized(img, _SIGMAS_TRIPLET[1])
            h_peak, w_peak = np.unravel_index(cur_np.argmax(), cur_np.shape)
            ours_x = coords[0, 0, 1, d_peak, h_peak, w_peak].item()
            ours_y = coords[0, 0, 2, d_peak, h_peak, w_peak].item()
            assert abs(ours_x - cx) < 0.05, f"our x error {abs(ours_x-cx):.4f} for blob at x={cx}"
            assert abs(ours_y - cy) < 0.05, f"our y error {abs(ours_y-cy):.4f} for blob at y={cy}"

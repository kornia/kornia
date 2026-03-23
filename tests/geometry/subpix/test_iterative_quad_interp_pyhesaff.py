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

"""Reference tests for conv_quad_interp3d.

These tests verify that the subpixel localisation implemented in
``conv_quad_interp3d`` agrees with the single-step C++ HessAff
``localizeKeypoint`` formula (ported here in pure Python/NumPy).

A synthetic 3-scale Gaussian response map is used as input so that the
expected peak location is known analytically and no external dependencies
are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kornia.geometry.subpix.nms import nms3d
from kornia.geometry.subpix.spatial_soft_argmax import conv_quad_interp3d

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_3d_gauss_response(
    H: int, W: int, cx: float, cy: float, cs: float, sigma_xy: float = 1.5, sigma_s: float = 1.0
) -> torch.Tensor:
    """Build a synthetic (1, 1, 3, H, W) Gaussian response with known peak.

    The analytic peak is at (cx, cy) in x/y and cs in the scale dimension
    (0-indexed, expected value in (0, 2) so that it is interior to the 3 slices).
    """
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    resp = np.zeros((3, H, W), dtype=np.float32)
    for d in range(3):
        resp[d] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_xy**2) - (d - cs) ** 2 / (2 * sigma_s**2))
    return torch.from_numpy(resp).unsqueeze(0).unsqueeze(0)


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

    def _run(self, cx: float, cy: float, cs: float, device: torch.device, dtype: torch.dtype) -> None:
        resp_t = _make_3d_gauss_response(19, 19, cx, cy, cs).to(device=device, dtype=dtype)

        assert nms3d(resp_t, (3, 3, 3), True).sum().item() >= 1, "no NMS peak found"

        # Our function — one iteration keeps the initial integer position
        coords, _ = conv_quad_interp3d(resp_t, n_iters=1, strict_maxima_bonus=0)

        # Reference: apply the C++ formula to the numpy arrays
        resp_np = resp_t.cpu().float().numpy()[0, 0]  # (3, H, W)
        low_np, cur_np, high_np = resp_np[0], resp_np[1], resp_np[2]
        h_peak, w_peak = round(cy), round(cx)
        ref_sx, ref_sy, ref_ss, valid = _localizeKeypoint_ref(low_np, cur_np, high_np, h_peak, w_peak)
        assert valid, "reference formula failed to solve"

        d_peak = 1  # middle scale
        ours_x = coords[0, 0, 1, d_peak, h_peak, w_peak].item()
        ours_y = coords[0, 0, 2, d_peak, h_peak, w_peak].item()
        ours_s = coords[0, 0, 0, d_peak, h_peak, w_peak].item()

        tol = 1e-3
        assert abs(ours_x - (w_peak + ref_sx)) < tol, f"x mismatch: ours={ours_x:.5f} ref={w_peak + ref_sx:.5f}"
        assert abs(ours_y - (h_peak + ref_sy)) < tol, f"y mismatch: ours={ours_y:.5f} ref={h_peak + ref_sy:.5f}"
        assert abs(ours_s - (d_peak + ref_ss)) < tol, f"s mismatch: ours={ours_s:.5f} ref={d_peak + ref_ss:.5f}"

    @pytest.mark.parametrize("cx,cy,cs", [(9.4, 8.7, 1.3), (5.3, 4.8, 1.2), (9.0, 9.0, 1.0), (7.7, 8.2, 1.4)])
    def test_single_step_matches_cpp_formula_float32(self, cx: float, cy: float, cs: float) -> None:
        self._run(cx, cy, cs, torch.device("cpu"), torch.float32)

    @pytest.mark.parametrize("cx,cy,cs", [(9.4, 8.7, 1.3), (5.3, 4.8, 1.2)])
    def test_single_step_matches_cpp_formula_float64(self, cx: float, cy: float, cs: float) -> None:
        self._run(cx, cy, cs, torch.device("cpu"), torch.float64)


class TestIterativeQuadInterp3dAccuracy:
    """Verify that conv_quad_interp3d accurately recovers subpixel positions."""

    def _run(self, cx: float, cy: float, cs: float, device: torch.device, dtype: torch.dtype) -> None:
        resp_t = _make_3d_gauss_response(19, 19, cx, cy, cs).to(device=device, dtype=dtype)

        nms_mask = nms3d(resp_t, (3, 3, 3), True)
        assert nms_mask.sum().item() >= 1, "no NMS peak"

        coords, _ = conv_quad_interp3d(resp_t, strict_maxima_bonus=0)
        d_peak = 1
        h_peak, w_peak = round(cy), round(cx)
        ours_x = coords[0, 0, 1, d_peak, h_peak, w_peak].item()
        ours_y = coords[0, 0, 2, d_peak, h_peak, w_peak].item()

        tol = 0.05
        assert abs(ours_x - cx) < tol, f"x error {abs(ours_x - cx):.4f} for blob at x={cx}"
        assert abs(ours_y - cy) < tol, f"y error {abs(ours_y - cy):.4f} for blob at y={cy}"

    @pytest.mark.parametrize("cx,cy,cs", [(9.4, 8.7, 1.3), (5.3, 4.8, 1.2), (8.4, 9.3, 1.3)])
    def test_subpixel_accuracy_float32(self, cx: float, cy: float, cs: float) -> None:
        self._run(cx, cy, cs, torch.device("cpu"), torch.float32)

    @pytest.mark.parametrize("cx,cy,cs", [(9.4, 8.7, 1.3), (5.3, 4.8, 1.2)])
    def test_subpixel_accuracy_float64(self, cx: float, cy: float, cs: float) -> None:
        self._run(cx, cy, cs, torch.device("cpu"), torch.float64)

    def test_integer_peak_exact(self) -> None:
        """Blob at an integer position — recovered coords should be very close to integer."""
        cx, cy, cs = 9.0, 9.0, 1.0
        resp_t = _make_3d_gauss_response(19, 19, cx, cy, cs)
        coords, _ = conv_quad_interp3d(resp_t, strict_maxima_bonus=0)
        d_peak = 1
        h_peak, w_peak = round(cy), round(cx)
        ours_x = coords[0, 0, 1, d_peak, h_peak, w_peak].item()
        ours_y = coords[0, 0, 2, d_peak, h_peak, w_peak].item()
        assert abs(ours_x - cx) < 1e-3
        assert abs(ours_y - cy) < 1e-3

    def test_multiple_peaks(self) -> None:
        """Two blobs at different subpixel positions — both should be recovered."""
        H, W = 32, 32
        blobs = [(8.3, 7.8, 1.3), (22.6, 21.1, 1.2)]
        resp_np = np.zeros((3, H, W), dtype=np.float32)
        xs = np.arange(W, dtype=np.float64)
        ys = np.arange(H, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        for bx, by, bs in blobs:
            for d in range(3):
                resp_np[d] += np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * 1.5**2) - (d - bs) ** 2 / (2 * 1.0**2))
        resp_t = torch.from_numpy(resp_np).unsqueeze(0).unsqueeze(0)

        coords, _ = conv_quad_interp3d(resp_t, strict_maxima_bonus=0)
        d_peak = 1
        for bx, by, _ in blobs:
            h_int, w_int = round(by), round(bx)
            ours_x = coords[0, 0, 1, d_peak, h_int, w_int].item()
            ours_y = coords[0, 0, 2, d_peak, h_int, w_int].item()
            assert abs(ours_x - bx) < 0.1, f"blob ({bx},{by}): x ours={ours_x:.3f}"
            assert abs(ours_y - by) < 0.1, f"blob ({bx},{by}): y ours={ours_y:.3f}"

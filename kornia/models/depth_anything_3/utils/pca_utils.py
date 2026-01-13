# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PCA utilities for feature visualization and dimensionality reduction (video-friendly).
- Support frame-by-frame: transform_frame / transform_video
- Support one-time global PCA fitting and reuse (mean, V3) for stable colors
- Support Procrustes alignment (solving principal component order/sign/rotation jumps)
- Support global fixed or temporal EMA for percentiles (time dimension only, no spatial)
"""

import numpy as np
import torch


def pca_to_rgb_4d_bf16_percentile(
    x_np: np.ndarray,
    device=None,
    q_oversample: int = 6,
    clip_percent: float = 10.0,  # Percentage to clip from top and bottom (0~49.9)
    return_uint8: bool = False,
    enable_autocast_bf16: bool = True,
):
    """
    Reduce numpy array of shape (49, 27, 36, 3072) to 3D via PCA and visualize as (49, 27, 36, 3).
    - PCA uses torch.pca_lowrank (randomized SVD), defaults to GPU.
    - Uses CUDA bf16 autocast in computation (if available),
      then per-channel percentile clipping and normalization.
    - Default removes 5% outliers from top and bottom (adjustable via clip_percent) to
      improve visualization contrast.

    Parameters
    ----------
    x_np : np.ndarray
        Shape must be (49, 27, 36, 3072). dtype recommended float32/float64.
    device : str | None
        Specify 'cuda' or 'cpu'. Auto-select if None (prefer cuda).
    q_oversample : int
        Oversampling q for pca_lowrank, must be >= 3.
        Slightly larger than target dim (3) is more stable, default 6.
    clip_percent : float
        Percentage to clip from top and bottom (0~49.9),
        e.g. 5.0 means clip lowest 5% and highest 5% per channel.
    return_uint8 : bool
        True returns uint8(0~255), otherwise returns float32(0~1).
    enable_autocast_bf16 : bool
        Enable bf16 autocast on CUDA.

    Returns
    -------
    np.ndarray
        Array of shape (49, 27, 36, 3), float32[0,1] or uint8[0,255].
    """
    assert (
        x_np.ndim == 4
    )  # and x_np.shape[-1] == 3072, f"expect (49,27,36,3072), got {x_np.shape}"
    B1, B2, B3, D = x_np.shape
    N = B1 * B2 * B3

    # Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert input to torch, unified float32
    X = torch.from_numpy(x_np.reshape(N, D)).to(device=device, dtype=torch.float32)

    # Parameter and safety checks
    k = 3
    q = max(int(q_oversample), k)
    clip_percent = float(clip_percent)
    if not (0.0 <= clip_percent < 50.0):
        raise ValueError(
            "clip_percent must be in [0, 50), e.g. 5.0 means clip 5% from top and bottom"
        )
    low = clip_percent / 100.0
    high = 1.0 - low

    with torch.no_grad():
        # Zero mean
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean

        # Main computation: PCA + projection, try to use bf16
        # (auto-fallback if operator not supported)
        device.startswith("cuda") and enable_autocast_bf16
        U, S, V = torch.pca_lowrank(Xc, q=q, center=False)  # V: (D, q)
        V3 = V[:, :k]  # (3072, 3)
        PCs = Xc @ V3  # (N, 3)

        # === Per-channel percentile clipping and normalization to [0,1] ===
        # Vectorized one-time calculation of low/high percentiles for each channel
        qs = torch.tensor([low, high], device=PCs.device, dtype=PCs.dtype)
        qvals = torch.quantile(PCs, q=qs, dim=0)  # Shape (2, 3)
        lo = qvals[0]  # (3,)
        hi = qvals[1]  # (3,)

        # Avoid degenerate case where hi==lo
        denom = torch.clamp(hi - lo, min=1e-8)

        # Broadcast clipping + normalization
        PCs = torch.clamp(PCs, lo, hi)
        PCs = (PCs - lo) / denom  # (N, 3) in [0,1]

        # Restore 4D
        PCs = PCs.reshape(B1, B2, B3, k)

        # Output
        if return_uint8:
            out = (PCs * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        else:
            out = PCs.clamp(0, 1).to(torch.float32).cpu().numpy()

    return out


class PCARGBVisualizer:
    """
    Stable PCA→RGB for video features shaped (T, H, W, D) or a single frame (H, W, D).
    - Global mean/V3 reference for stable colors
    - Per-frame PCA with Procrustes alignment to V3_ref (basis_mode='procrustes')
    - Percentile normalization with global or EMA stats (time-only, no spatial smoothing)
    """

    def __init__(
        self,
        device=None,
        q_oversample: int = 16,
        clip_percent: float = 10.0,
        return_uint8: bool = False,
        enable_autocast_bf16: bool = True,
        basis_mode: str = "procrustes",  # 'fixed' | 'procrustes'
        percentile_mode: str = "ema",  # 'global' | 'ema'
        ema_alpha: float = 0.1,
        denom_eps: float = 1e-4,
    ):
        assert 0.0 <= clip_percent < 50.0
        assert basis_mode in ("fixed", "procrustes")
        assert percentile_mode in ("global", "ema")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = max(int(q_oversample), 6)
        self.clip_percent = float(clip_percent)
        self.return_uint8 = return_uint8
        self.enable_autocast_bf16 = enable_autocast_bf16
        self.basis_mode = basis_mode
        self.percentile_mode = percentile_mode
        self.ema_alpha = float(ema_alpha)
        self.denom_eps = float(denom_eps)

        # reference state
        self.mean_ref = None  # (1, D)
        self.V3_ref = None  # (D, 3)
        self.lo_ref = None  # (3,)
        self.hi_ref = None  # (3,)

    @torch.no_grad()
    def fit_reference(self, frames):
        """
        Fit global mean/V3 and initialize percentiles from a reference set.
        frames: ndarray (T,H,W,D) or list of (H,W,D)
        """
        if isinstance(frames, np.ndarray):
            if frames.ndim != 4:
                raise ValueError("fit_reference expects (T,H,W,D) ndarray.")
            T, H, W, D = frames.shape
            X = torch.from_numpy(frames.reshape(T * H * W, D))
        else:  # list of (H,W,D)
            xs = [torch.from_numpy(x.reshape(-1, x.shape[-1])) for x in frames]
            D = xs[0].shape[-1]
            X = torch.cat(xs, dim=0)

        X = X.to(self.device, dtype=torch.float32)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        mean = X.mean(0, keepdim=True)
        Xc = X - mean

        U, S, V = torch.pca_lowrank(Xc, q=max(self.q, 8), center=False)
        V3 = V[:, :3]  # (D,3)

        PCs = Xc @ V3
        low = self.clip_percent / 100.0
        high = 1.0 - low
        qs = torch.tensor([low, high], device=PCs.device, dtype=PCs.dtype)
        qvals = torch.quantile(PCs, q=qs, dim=0)
        lo, hi = qvals[0], qvals[1]

        self.mean_ref = mean
        self.V3_ref = V3
        if self.percentile_mode == "global":
            self.lo_ref, self.hi_ref = lo, hi
        else:
            self.lo_ref = lo.clone()
            self.hi_ref = hi.clone()

    @torch.no_grad()
    def _project_with_stable_colors(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N,D) where N = H*W
        Returns PCs_raw: (N,3) using stable basis (fixed or Procrustes-aligned)
        """
        assert self.mean_ref is not None and self.V3_ref is not None, "Call fit_reference() first."
        X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        Xc = X - self.mean_ref

        if self.basis_mode == "fixed":
            V3_used = self.V3_ref
        else:
            U, S, V = torch.pca_lowrank(Xc, q=max(self.q, 6), center=False)
            V3 = V[:, :3]  # (D,3)
            M = V3.T @ self.V3_ref
            Uo, So, Vh = torch.linalg.svd(M)
            R = Uo @ Vh
            V3_used = V3 @ R
            # Optional polarity fix via anchor
            a = self.V3_ref.mean(0, keepdim=True)
            sign = torch.sign((V3_used * a).sum(0, keepdim=True)).clamp(min=-1)
            V3_used = V3_used * sign

        return Xc @ V3_used

    @torch.no_grad()
    def _normalize_rgb(self, PCs_raw: torch.Tensor) -> torch.Tensor:
        assert self.lo_ref is not None and self.hi_ref is not None
        if self.percentile_mode == "global":
            lo, hi = self.lo_ref, self.hi_ref
        else:
            low = self.clip_percent / 100.0
            high = 1.0 - low
            qs = torch.tensor([low, high], device=PCs_raw.device, dtype=PCs_raw.dtype)
            qvals = torch.quantile(PCs_raw, q=qs, dim=0)
            lo_now, hi_now = qvals[0], qvals[1]
            a = self.ema_alpha
            self.lo_ref = (1 - a) * self.lo_ref + a * lo_now
            self.hi_ref = (1 - a) * self.hi_ref + a * hi_now
            lo, hi = self.lo_ref, self.hi_ref

        denom = torch.clamp(hi - lo, min=self.denom_eps)
        PCs = torch.clamp(PCs_raw, lo, hi)
        PCs = (PCs - lo) / denom
        return PCs.clamp_(0, 1)

    @torch.no_grad()
    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: (H,W,D) -> (H,W,3)
        """
        if frame.ndim != 3:
            raise ValueError("transform_frame expects (H,W,D).")
        H, W, D = frame.shape
        X = torch.from_numpy(frame.reshape(H * W, D)).to(self.device, dtype=torch.float32)
        PCs_raw = self._project_with_stable_colors(X)
        PCs = self._normalize_rgb(PCs_raw).reshape(H, W, 3)
        if self.return_uint8:
            return (PCs * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return PCs.to(torch.float32).cpu().numpy()

    @torch.no_grad()
    def transform_video(self, frames) -> np.ndarray:
        """
        frames: (T,H,W,D) or list of (H,W,D)
        returns: (T,H,W,3)
        """
        outs = []
        if isinstance(frames, np.ndarray):
            if frames.ndim != 4:
                raise ValueError("transform_video expects (T,H,W,D).")
            T, H, W, D = frames.shape
            for t in range(T):
                outs.append(self.transform_frame(frames[t]))
        else:
            for f in frames:
                outs.append(self.transform_frame(f))
        return np.stack(outs, axis=0)

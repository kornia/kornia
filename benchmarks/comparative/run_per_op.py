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

"""Per-op eager benchmark: kornia (patched) vs torchvision.v2 vs albumentations.

Methodology:
  - Input: pre-resident GPU tensor (B=8, 3, 512, 512, fp32)
  - Timing: CUDA events, 25 warmup + 100 timed iterations
  - Albumentations: CPU-only, numpy uint8 (B, 512, 512, 3), time.perf_counter
  - NO torch.compile -- eager only

Run from /tmp with PYTHONNOUSERSITE=1:
  cd /tmp && PYTHONNOUSERSITE=1 \
    /home/nvidia/bubbaloop-nodes-official/camera-object-detector/.pixi/envs/default/bin/python3.10 \
    /home/nvidia/kornia/benchmarks/comparative/run_per_op.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
import traceback as _traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

# ---------------------------------------------------------------------------
# WORKAROUND 1: _torch_inverse_cast -- analytical closed-form 3x3 inverse
# Must be applied BEFORE importing any kornia geometry module.
# ---------------------------------------------------------------------------


def _analytical_3x3_inv(input: torch.Tensor) -> torch.Tensor:
    """Closed-form 3x3 matrix inverse via adjugate / determinant."""
    dtype = input.dtype
    m = input.to(torch.float32)
    squeeze = m.ndim == 2
    if squeeze:
        m = m.unsqueeze(0)
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    inv_det = 1.0 / det
    inv = torch.empty_like(m)
    inv[..., 0, 0] = (e * i - f * h) * inv_det
    inv[..., 0, 1] = -(b * i - c * h) * inv_det
    inv[..., 0, 2] = (b * f - c * e) * inv_det
    inv[..., 1, 0] = -(d * i - f * g) * inv_det
    inv[..., 1, 1] = (a * i - c * g) * inv_det
    inv[..., 1, 2] = -(a * f - c * d) * inv_det
    inv[..., 2, 0] = (d * h - e * g) * inv_det
    inv[..., 2, 1] = -(a * h - b * g) * inv_det
    inv[..., 2, 2] = (a * e - b * d) * inv_det
    if squeeze:
        inv = inv.squeeze(0)
    return inv.to(dtype)


def _cpu_solve_cast(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Patch for _torch_solve_cast: solve on CPU (avoids CUDA linalg/cusolver)."""
    dev = A.device
    dtype = A.dtype
    out = torch.linalg.solve(A.to("cpu", torch.float64), B.to("cpu", torch.float64))
    return out.to(device=dev, dtype=dtype)


def _patch_kornia_solvers() -> None:
    """Patch both _torch_inverse_cast and _torch_solve_cast in all kornia modules."""
    import kornia.geometry.conversions as _kgc
    import kornia.utils.helpers as _kh

    _kh._torch_inverse_cast = _analytical_3x3_inv
    _kgc._torch_inverse_cast = _analytical_3x3_inv
    _kh._torch_solve_cast = _cpu_solve_cast

    for mod_name, mod in sys.modules.items():
        if mod_name.startswith("kornia"):
            if hasattr(mod, "_torch_inverse_cast"):
                try:
                    mod._torch_inverse_cast = _analytical_3x3_inv
                except (AttributeError, TypeError):
                    pass
            if hasattr(mod, "_torch_solve_cast"):
                try:
                    mod._torch_solve_cast = _cpu_solve_cast
                except (AttributeError, TypeError):
                    pass

    # Explicit patch for imgwarp which imports directly
    try:
        import kornia.geometry.transform.imgwarp as _imgwarp

        _imgwarp._torch_solve_cast = _cpu_solve_cast
    except Exception:
        pass


# Trigger kornia loading so the patch covers geometry.conversions
import kornia.geometry.conversions
import kornia.utils.helpers  # noqa: F401

_patch_kornia_solvers()


# ---------------------------------------------------------------------------
# WORKAROUND 2: Runtime monkey-patches -- all FIVE optimizations from run_v4.py
# ---------------------------------------------------------------------------

_KORNIA_PATCHED = False


def _apply_kornia_optimisation_patches() -> str:
    global _KORNIA_PATCHED
    if _KORNIA_PATCHED:
        return "already patched"

    status_parts: list[str] = []

    # Patch 1: Normalize
    try:
        import kornia.augmentation._2d.intensity.normalize as _norm_mod

        _Normalize = _norm_mod.Normalize
        _orig_norm_init = _Normalize.__init__

        def _patched_norm_init(self, mean, std, p=1.0, keepdim=False, **kw):
            _orig_norm_init(self, mean, std, p=p, keepdim=keepdim)
            _mean = self.flags["mean"]
            _std = self.flags["std"]
            if isinstance(_mean, (int, float)):
                _mean = torch.tensor([float(_mean)])
            if isinstance(_std, (int, float)):
                _std = torch.tensor([float(_std)])
            if isinstance(_mean, (tuple, list)):
                _mean = torch.tensor(_mean, dtype=torch.float32)
            if isinstance(_std, (tuple, list)):
                _std = torch.tensor(_std, dtype=torch.float32)
            if not isinstance(_mean, torch.Tensor):
                _mean = torch.tensor(_mean, dtype=torch.float32)
            if not isinstance(_std, torch.Tensor):
                _std = torch.tensor(_std, dtype=torch.float32)
            _mean_b = _mean.view(1, -1, 1, 1) if _mean.dim() == 1 else _mean
            _std_b = _std.view(1, -1, 1, 1) if _std.dim() == 1 else _std
            self.register_buffer("_mean_b", _mean_b, persistent=False)
            self.register_buffer("_std_b", _std_b, persistent=False)

        def _patched_norm_apply(self, input, params, flags, transform=None):
            mean = self._mean_b
            std = self._std_b
            if mean.dtype != input.dtype or mean.device != input.device:
                mean = mean.to(device=input.device, dtype=input.dtype)
                std = std.to(device=input.device, dtype=input.dtype)
            return (input - mean) / std

        _Normalize.__init__ = _patched_norm_init
        _Normalize.apply_transform = _patched_norm_apply
        status_parts.append("Normalize(buffers)")
    except Exception as exc:
        status_parts.append(f"Normalize FAILED: {exc}")

    # Patch 2: RandomHorizontalFlip cache
    try:
        from typing import Dict as _Dict
        from typing import Tuple as _Tuple

        import kornia.augmentation._2d.geometric.horizontal_flip as _hflip_mod

        _HFLIP_MAT_TEMPLATE = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        _HFLIP_MAT_CACHE: _Dict[_Tuple, torch.Tensor] = {}
        _hflip_mod._HFLIP_MAT_TEMPLATE = _HFLIP_MAT_TEMPLATE
        _hflip_mod._HFLIP_MAT_CACHE = _HFLIP_MAT_CACHE

        _RHF = _hflip_mod.RandomHorizontalFlip

        def _patched_hflip_compute(self, input, params, flags):
            w: int = int(params["forward_input_shape"][-1].item())
            key = (input.device, input.dtype, w)
            cached = _HFLIP_MAT_CACHE.get(key)
            if cached is None:
                flip_mat = _HFLIP_MAT_TEMPLATE.to(device=input.device, dtype=input.dtype).clone()
                flip_mat[0, 2] = w - 1
                cached = flip_mat.unsqueeze(0)
                _HFLIP_MAT_CACHE[key] = cached
            return cached.expand(input.shape[0], 3, 3)

        _RHF.compute_transformation = _patched_hflip_compute
        status_parts.append("HFlip(cache)")
    except Exception as exc:
        status_parts.append(f"HFlip FAILED: {exc}")

    # Patch 3: hflip / vflip -- pure flip()
    try:
        import kornia.augmentation._2d.geometric.horizontal_flip as _hflip_mod2
        import kornia.geometry.transform as _kgt
        import kornia.geometry.transform.flips as _flips_mod

        def _fast_hflip(input: torch.Tensor) -> torch.Tensor:
            return input.flip(-1)

        def _fast_vflip(input: torch.Tensor) -> torch.Tensor:
            return input.flip(-2)

        _flips_mod.hflip = _fast_hflip
        _flips_mod.vflip = _fast_vflip
        _hflip_mod2.hflip = _fast_hflip
        if hasattr(_kgt, "hflip"):
            _kgt.hflip = _fast_hflip
        if hasattr(_kgt, "vflip"):
            _kgt.vflip = _fast_vflip
        for _mn, _m in sys.modules.items():
            if _mn.startswith("kornia"):
                if getattr(_m, "hflip", None) is not _fast_hflip and hasattr(_m, "hflip"):
                    try:
                        _m.hflip = _fast_hflip
                    except (AttributeError, TypeError):
                        pass
                if getattr(_m, "vflip", None) is not _fast_vflip and hasattr(_m, "vflip"):
                    try:
                        _m.vflip = _fast_vflip
                    except (AttributeError, TypeError):
                        pass
        status_parts.append("hflip/vflip(no-contiguous)")
    except Exception as exc:
        status_parts.append(f"hflip/vflip FAILED: {exc}")

    # Patch 4: RandomAffine closed-form
    try:
        import torch.nn.functional as F

        import kornia.augmentation._2d.geometric.affine as _aff_mod

        def _affine_matrix2d_closed(
            translations: torch.Tensor,
            center: torch.Tensor,
            scale: torch.Tensor,
            angle: torch.Tensor,
            shear_x: torch.Tensor,
            shear_y: torch.Tensor,
        ) -> torch.Tensor:
            _PI_OVER_180 = math.pi / 180.0
            ang_rad = angle * _PI_OVER_180
            cos_a = torch.cos(ang_rad)
            sin_a = torch.sin(ang_rad)
            sx = scale[:, 0]
            sy = scale[:, 1]
            cx = center[:, 0]
            cy = center[:, 1]
            tx = translations[:, 0]
            ty = translations[:, 1]
            a = sx * cos_a
            b = -sy * sin_a
            c = cx - a * cx - b * cy + tx
            d = sx * sin_a
            e = sy * cos_a
            f_t = cy - e * cy - d * cx + ty
            sx_t = torch.tan(shear_x * _PI_OVER_180)
            sy_t = torch.tan(shear_y * _PI_OVER_180)
            r00 = a - b * sy_t
            r01 = -a * sx_t + b * (1.0 + sx_t * sy_t)
            r02 = a * sx_t * cy + b * sy_t * (cx - sx_t * cy) + c
            r10 = d - e * sy_t
            r11 = -d * sx_t + e * (1.0 + sx_t * sy_t)
            r12 = d * sx_t * cy + e * sy_t * (cx - sx_t * cy) + f_t
            zeros = torch.zeros_like(r00)
            ones = torch.ones_like(r00)
            return torch.stack([r00, r01, r02, r10, r11, r12, zeros, zeros, ones], dim=-1).reshape(-1, 3, 3)

        def _affine_homography_inv(M: torch.Tensor) -> torch.Tensor:
            a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
            d, e, f_t = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
            det = a * e - b * d
            r00 = e / det
            r01 = -b / det
            r02 = (b * f_t - c * e) / det
            r10 = -d / det
            r11 = a / det
            r12 = (c * d - a * f_t) / det
            zeros = torch.zeros_like(r00)
            ones = torch.ones_like(r00)
            return torch.stack([r00, r01, r02, r10, r11, r12, zeros, zeros, ones], dim=-1).reshape(-1, 3, 3)

        def _make_norm_matrices(height, width, device, dtype):
            eps = 1e-14
            w_denom = float(width - 1) if width > 1 else eps
            h_denom = float(height - 1) if height > 1 else eps
            sw = 2.0 / w_denom
            sh = 2.0 / h_denom
            N = torch.zeros(1, 3, 3, device=device, dtype=dtype)
            N[0, 0, 0] = sw
            N[0, 1, 1] = sh
            N[0, 0, 2] = -1.0
            N[0, 1, 2] = -1.0
            N[0, 2, 2] = 1.0
            N_inv = torch.zeros(1, 3, 3, device=device, dtype=dtype)
            N_inv[0, 0, 0] = 1.0 / sw
            N_inv[0, 1, 1] = 1.0 / sh
            N_inv[0, 0, 2] = 1.0 / sw
            N_inv[0, 1, 2] = 1.0 / sh
            N_inv[0, 2, 2] = 1.0
            return N, N_inv

        def _get_norm_matrices(self, height, width, device, dtype):
            key = (height, width, device, dtype)
            if key not in self._norm_cache:
                self._norm_cache[key] = _make_norm_matrices(height, width, device, dtype)
            return self._norm_cache[key]

        _aff_mod._affine_matrix2d_closed = _affine_matrix2d_closed
        _aff_mod._affine_homography_inv = _affine_homography_inv

        _RA = _aff_mod.RandomAffine
        _orig_ra_init = _RA.__init__

        def _patched_ra_init(
            self,
            degrees,
            translate=None,
            scale=None,
            shear=None,
            resample="BILINEAR",
            same_on_batch=False,
            align_corners=False,
            padding_mode="ZEROS",
            p=0.5,
            keepdim=False,
            **kw,
        ):
            kw.pop("fill_value", None)
            _orig_ra_init(
                self,
                degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                resample=resample,
                same_on_batch=same_on_batch,
                align_corners=align_corners,
                padding_mode=padding_mode,
                p=p,
                keepdim=keepdim,
            )
            self._norm_cache: dict = {}

        def _patched_ra_compute_transform(self, input, params, flags):
            return _affine_matrix2d_closed(
                params["translations"].to(device=input.device, dtype=input.dtype),
                params["center"].to(device=input.device, dtype=input.dtype),
                params["scale"].to(device=input.device, dtype=input.dtype),
                params["angle"].to(device=input.device, dtype=input.dtype),
                params["shear_x"].to(device=input.device, dtype=input.dtype),
                params["shear_y"].to(device=input.device, dtype=input.dtype),
            )

        def _patched_ra_apply(self, input, params, flags, transform=None):
            _, _, height, width = input.shape
            if not isinstance(transform, torch.Tensor):
                raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")
            padding_mode_str = flags["padding_mode"].name.lower()

            if padding_mode_str == "fill":
                from kornia.geometry.transform import warp_affine

                return warp_affine(
                    input,
                    transform[:, :2, :],
                    (height, width),
                    flags["resample"].name.lower(),
                    align_corners=flags["align_corners"],
                    padding_mode=padding_mode_str,
                    fill_value=flags.get("fill_value"),
                )

            align_corners = flags["align_corners"]
            mode = flags["resample"].name.lower()

            if not hasattr(self, "_norm_cache"):
                self._norm_cache = {}

            N, N_inv = _get_norm_matrices(self, height, width, input.device, input.dtype)
            M_inv = _affine_homography_inv(transform)
            theta = (N @ M_inv @ N_inv)[:, :2, :]
            B, C = input.shape[:2]
            grid = F.affine_grid(theta, [B, C, height, width], align_corners=align_corners)
            return F.grid_sample(
                input,
                grid,
                mode=mode,
                padding_mode=padding_mode_str,
                align_corners=align_corners,
            )

        _RA.__init__ = _patched_ra_init
        _RA._make_norm_matrices = _make_norm_matrices
        _RA._get_norm_matrices = _get_norm_matrices
        _RA.compute_transformation = _patched_ra_compute_transform
        _RA.apply_transform = _patched_ra_apply
        status_parts.append("RandomAffine(closed-form+cache)")
    except Exception as exc:
        status_parts.append(f"RandomAffine FAILED: {exc}")

    # Patch 5: ColorJiggle -- fused HSV roundtrip
    try:
        import kornia.augmentation._2d.intensity.color_jiggle as _cj_mod
        from kornia.color import hsv_to_rgb as _hsv_to_rgb
        from kornia.color import rgb_to_hsv as _rgb_to_hsv

        _TWO_PI = 2.0 * math.pi

        def _patched_cj_apply(self, input, params, flags, transform=None):
            brightness_factor: torch.Tensor = params["brightness_factor"]
            contrast_factor: torch.Tensor = params["contrast_factor"]
            saturation_factor: torch.Tensor = params["saturation_factor"]
            hue_factor: torch.Tensor = params["hue_factor"]

            brightness_delta = brightness_factor - 1.0
            hue_shift = hue_factor * _TWO_PI

            do_brightness = bool(brightness_delta.any())
            do_contrast = bool((contrast_factor != 1.0).any())
            do_saturation = bool((saturation_factor != 1.0).any())
            do_hue = bool(hue_shift.any())

            dtype = input.dtype
            device = input.device

            b_vec = brightness_delta.to(dtype=dtype, device=device).view(-1, 1, 1, 1) if do_brightness else None
            c_vec = contrast_factor.to(dtype=dtype, device=device).view(-1, 1, 1, 1) if do_contrast else None
            s_vec = saturation_factor.to(dtype=dtype, device=device).view(-1, 1, 1, 1) if do_saturation else None
            h_vec = hue_shift.to(dtype=dtype, device=device).view(-1, 1, 1, 1) if do_hue else None

            pending_s = None
            pending_h = None

            def flush_hsv(img: torch.Tensor) -> torch.Tensor:
                nonlocal pending_s, pending_h
                if pending_s is None and pending_h is None:
                    return img
                img_hsv = _rgb_to_hsv(img)
                h_ch = img_hsv[:, 0:1, :, :]
                s_ch = img_hsv[:, 1:2, :, :]
                v_ch = img_hsv[:, 2:3, :, :]
                if pending_s is not None:
                    s_ch = (s_ch * pending_s).clamp_(0.0, 1.0)
                if pending_h is not None:
                    h_ch = torch.fmod(h_ch + pending_h, _TWO_PI)
                result = _hsv_to_rgb(torch.cat([h_ch, s_ch, v_ch], dim=1))
                pending_s = None
                pending_h = None
                return result

            cloned = False
            jittered = input

            for idx in params["order"].tolist():
                if idx == 0:
                    if not do_brightness:
                        continue
                    jittered = flush_hsv(jittered)
                    if not cloned:
                        jittered = jittered.clone()
                        cloned = True
                    jittered.add_(b_vec).clamp_(0.0, 1.0)
                elif idx == 1:
                    if not do_contrast:
                        continue
                    jittered = flush_hsv(jittered)
                    if not cloned:
                        jittered = jittered.clone()
                        cloned = True
                    jittered.mul_(c_vec).clamp_(0.0, 1.0)
                elif idx == 2:
                    if not do_saturation:
                        continue
                    pending_s = s_vec if pending_s is None else pending_s * s_vec
                else:
                    if not do_hue:
                        continue
                    pending_h = h_vec if pending_h is None else pending_h + h_vec

            jittered = flush_hsv(jittered)
            return jittered

        _cj_mod.ColorJiggle.apply_transform = _patched_cj_apply
        status_parts.append("ColorJiggle(fused-HSV)")
    except Exception as exc:
        status_parts.append(f"ColorJiggle FAILED: {exc}")

    _KORNIA_PATCHED = True
    return "; ".join(status_parts)


# Apply patches immediately
_patch_status = _apply_kornia_optimisation_patches()
# Re-apply solver patches after augmentation modules loaded
_patch_kornia_solvers()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH = 8
RES = 512
WARMUP = 25
RUNS = 100
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

OUT_DIR = Path("/home/nvidia/kornia/benchmarks/comparative")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _stats(times: list[float]) -> dict:
    s = sorted(times)
    n = len(s)
    return {
        "median_ms": s[n // 2],
        "p25_ms": s[n // 4],
        "p75_ms": s[3 * n // 4],
        "iqr_ms": s[3 * n // 4] - s[n // 4],
        "min_ms": s[0],
        "max_ms": s[-1],
        "mean_ms": statistics.mean(s),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def time_gpu(make_aug_fn, x_gpu: torch.Tensor) -> dict:
    """Time a kornia/torchvision GPU aug with CUDA events."""
    try:
        aug = make_aug_fn()
        if hasattr(aug, "cuda"):
            aug = aug.cuda()
        if hasattr(aug, "eval"):
            aug.eval()
        with torch.no_grad():
            for _ in range(WARMUP):
                _out = aug(x_gpu)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        with torch.no_grad():
            for i in range(RUNS):
                starts[i].record()
                _out = aug(x_gpu)
                ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return {"status": "ok", **_stats(times)}
    except Exception:
        return {"status": "error", "reason": _traceback.format_exc().splitlines()[-1]}


def time_gpu_with_labels(make_aug_fn, x_gpu: torch.Tensor) -> dict:
    """Time a torchvision MixUp/CutMix that requires (x, labels)."""
    try:
        aug = make_aug_fn()
        labels = torch.zeros(BATCH, dtype=torch.long, device=x_gpu.device)
        with torch.no_grad():
            for _ in range(WARMUP):
                _out = aug(x_gpu, labels)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        with torch.no_grad():
            for i in range(RUNS):
                starts[i].record()
                _out = aug(x_gpu, labels)
                ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return {"status": "ok", **_stats(times)}
    except Exception:
        return {"status": "error", "reason": _traceback.format_exc().splitlines()[-1]}


def time_alb(make_aug_fn, x_np: np.ndarray) -> dict:
    """Time albumentations on CPU. x_np is (B, H, W, 3) uint8."""
    try:
        aug = make_aug_fn()
        B = x_np.shape[0]
        for _ in range(WARMUP):
            for i in range(B):
                _out = aug(image=x_np[i])["image"]
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            for i in range(B):
                _out = aug(image=x_np[i])["image"]
            times.append((time.perf_counter() - t0) * 1000.0)
        return {"status": "ok", "cpu_only": True, **_stats(times)}
    except Exception:
        return {"status": "error", "reason": _traceback.format_exc().splitlines()[-1]}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _build_registry():
    import albumentations as A
    import torchvision.transforms.v2 as T

    import kornia.augmentation as K

    MEAN_T = torch.tensor(IMAGENET_MEAN)
    STD_T = torch.tensor(IMAGENET_STD)

    registry = []

    # ---- GEOMETRIC ----

    registry.append(
        {
            "name": "HorizontalFlip",
            "category": "geometric",
            "kornia": lambda: K.RandomHorizontalFlip(p=1.0),
            "tv": lambda: T.RandomHorizontalFlip(p=1.0),
            "alb": lambda: A.HorizontalFlip(p=1.0),
        }
    )

    registry.append(
        {
            "name": "VerticalFlip",
            "category": "geometric",
            "kornia": lambda: K.RandomVerticalFlip(p=1.0),
            "tv": lambda: T.RandomVerticalFlip(p=1.0),
            "alb": lambda: A.VerticalFlip(p=1.0),
        }
    )

    registry.append(
        {
            "name": "Rotation",
            "category": "geometric",
            "kornia": lambda: K.RandomRotation(degrees=15.0, p=1.0),
            "tv": lambda: T.RandomRotation(degrees=15.0),
            "alb": lambda: A.Rotate(limit=15, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Affine",
            "category": "geometric",
            "kornia": lambda: K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0),
            "tv": lambda: T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            "alb": lambda: A.Affine(rotate=(-15, 15), translate_percent=(-0.1, 0.1), scale=(0.8, 1.2), p=1.0),
        }
    )

    registry.append(
        {
            "name": "ResizedCrop",
            "category": "geometric",
            "kornia": lambda: K.RandomResizedCrop(size=(224, 224), p=1.0),
            "tv": lambda: T.RandomResizedCrop(size=224),
            "alb": lambda: A.RandomResizedCrop(size=(224, 224), p=1.0),
        }
    )

    registry.append(
        {
            "name": "CenterCrop",
            "category": "geometric",
            "kornia": lambda: K.CenterCrop(size=224),
            "tv": lambda: T.CenterCrop(size=224),
            "alb": lambda: A.CenterCrop(height=224, width=224, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Resize",
            "category": "geometric",
            "kornia": lambda: K.Resize(size=224),
            "tv": lambda: T.Resize(size=224),
            "alb": lambda: A.Resize(height=224, width=224, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Perspective",
            "category": "geometric",
            "kornia": lambda: K.RandomPerspective(distortion_scale=0.2, p=1.0),
            "tv": lambda: T.RandomPerspective(distortion_scale=0.2, p=1.0),
            "alb": lambda: A.Perspective(scale=(0.05, 0.2), p=1.0),
        }
    )

    # ---- INTENSITY: color/brightness ----

    registry.append(
        {
            "name": "ColorJitter",
            "category": "intensity_color",
            "kornia": lambda: K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            "tv": lambda: T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            "alb": lambda: A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Brightness",
            "category": "intensity_color",
            "kornia": lambda: K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
            "tv": lambda: T.ColorJitter(brightness=0.2),
            "alb": lambda: A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Contrast",
            "category": "intensity_color",
            "kornia": lambda: K.RandomContrast(contrast=(0.8, 1.2), p=1.0),
            "tv": lambda: T.ColorJitter(contrast=0.2),
            "alb": lambda: A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Saturation",
            "category": "intensity_color",
            "kornia": lambda: K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),
            "tv": lambda: T.ColorJitter(saturation=0.2),
            "alb": lambda: A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=51, val_shift_limit=0, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Hue",
            "category": "intensity_color",
            "kornia": lambda: K.RandomHue(hue=(-0.1, 0.1), p=1.0),
            "tv": lambda: T.ColorJitter(hue=0.1),
            "alb": lambda: A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=0, val_shift_limit=0, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Grayscale",
            "category": "intensity_color",
            "kornia": lambda: K.RandomGrayscale(p=1.0),
            "tv": lambda: T.RandomGrayscale(p=1.0),
            "alb": lambda: A.ToGray(p=1.0),
        }
    )

    registry.append(
        {
            "name": "Solarize",
            "category": "intensity_color",
            "kornia": lambda: K.RandomSolarize(thresholds=0.5, p=1.0),
            "tv": lambda: T.RandomSolarize(threshold=0.5, p=1.0),
            "alb": lambda: A.Solarize(p=1.0),
        }
    )

    registry.append(
        {
            "name": "Posterize",
            "category": "intensity_color",
            "kornia": lambda: K.RandomPosterize(bits=4, p=1.0),
            "tv": lambda: T.RandomPosterize(bits=4, p=1.0),
            "alb": lambda: A.Posterize(num_bits=4, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Equalize",
            "category": "intensity_color",
            "kornia": lambda: K.RandomEqualize(p=1.0),
            "tv": lambda: T.RandomEqualize(p=1.0),
            "alb": lambda: A.Equalize(p=1.0),
        }
    )

    registry.append(
        {
            "name": "Invert",
            "category": "intensity_color",
            "kornia": lambda: K.RandomInvert(p=1.0),
            "tv": lambda: T.RandomInvert(p=1.0),
            "alb": lambda: A.InvertImg(p=1.0),
        }
    )

    registry.append(
        {
            "name": "Sharpness",
            "category": "intensity_color",
            "kornia": lambda: K.RandomSharpness(sharpness=0.5, p=1.0),
            "tv": lambda: T.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
            "alb": lambda: A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        }
    )

    # ---- INTENSITY: blur/noise ----

    registry.append(
        {
            "name": "GaussianBlur",
            "category": "intensity_blur",
            "kornia": lambda: K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=1.0),
            "tv": lambda: T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            "alb": lambda: A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.1, 2.0), p=1.0),
        }
    )

    registry.append(
        {
            "name": "GaussianNoise",
            "category": "intensity_blur",
            "kornia": lambda: K.RandomGaussianNoise(std=0.05, p=1.0),
            "tv": lambda: T.GaussianNoise(sigma=0.05),
            "alb": lambda: A.GaussNoise(std_range=(0.05, 0.05), p=1.0),
        }
    )

    registry.append(
        {
            "name": "MotionBlur",
            "category": "intensity_blur",
            "kornia": lambda: K.RandomMotionBlur(kernel_size=5, angle=35.0, direction=0.5, p=1.0),
            "tv": None,
            "alb": lambda: A.MotionBlur(blur_limit=5, p=1.0),
        }
    )

    registry.append(
        {
            "name": "BoxBlur",
            "category": "intensity_blur",
            "kornia": lambda: K.RandomBoxBlur(kernel_size=(3, 3), p=1.0),
            "tv": None,
            "alb": lambda: A.Blur(blur_limit=3, p=1.0),
        }
    )

    registry.append(
        {
            "name": "MedianBlur",
            "category": "intensity_blur",
            "kornia": lambda: K.RandomMedianBlur(kernel_size=(3, 3), p=1.0),
            "tv": None,
            "alb": lambda: A.MedianBlur(blur_limit=3, p=1.0),
        }
    )

    # ---- ERASING ----

    registry.append(
        {
            "name": "RandomErasing",
            "category": "erasing",
            "kornia": lambda: K.RandomErasing(p=1.0),
            "tv": lambda: T.RandomErasing(p=1.0),
            "alb": lambda: A.CoarseDropout(
                num_holes_range=(1, 8), hole_height_range=(32, 64), hole_width_range=(32, 64), p=1.0
            ),
        }
    )

    # ---- NORMALIZE ----

    registry.append(
        {
            "name": "Normalize",
            "category": "normalize",
            "kornia": lambda: K.Normalize(mean=MEAN_T, std=STD_T),
            "tv": lambda: T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            "alb": lambda: A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
        }
    )

    registry.append(
        {
            "name": "Denormalize",
            "category": "normalize",
            "kornia": lambda: K.Denormalize(mean=MEAN_T, std=STD_T),
            "tv": None,
            "alb": None,
        }
    )

    # ---- MIX ----

    registry.append(
        {
            "name": "MixUp",
            "category": "mix",
            "kornia": lambda: K.RandomMixUpV2(p=1.0),
            "tv": lambda: T.MixUp(num_classes=1000),
            "alb": None,
            "tv_timing": "with_labels",
        }
    )

    registry.append(
        {
            "name": "CutMix",
            "category": "mix",
            "kornia": lambda: K.RandomCutMixV2(p=1.0),
            "tv": lambda: T.CutMix(num_classes=1000),
            "alb": None,
            "tv_timing": "with_labels",
        }
    )

    registry.append(
        {
            "name": "Mosaic",
            "category": "mix",
            "kornia": lambda: K.RandomMosaic(output_size=(512, 512), p=1.0),
            "tv": None,
            "alb": None,
        }
    )

    # ---- KORNIA-ONLY ----

    registry.append(
        {
            "name": "RandomRain",
            "category": "kornia_only",
            "kornia": lambda: K.RandomRain(p=1.0),
            "tv": None,
            "alb": None,
        }
    )

    registry.append(
        {
            "name": "RandomSnow",
            "category": "kornia_only",
            "kornia": lambda: K.RandomSnow(p=1.0),
            "tv": None,
            "alb": None,
        }
    )

    registry.append(
        {
            "name": "RandomChannelDropout",
            "category": "kornia_only",
            "kornia": lambda: K.RandomChannelDropout(p=1.0),
            "tv": None,
            "alb": None,
        }
    )

    registry.append(
        {
            "name": "RandomChannelShuffle",
            "category": "kornia_only",
            "kornia": lambda: K.RandomChannelShuffle(p=1.0),
            "tv": None,
            "alb": None,
        }
    )

    registry.append(
        {
            "name": "RandomRGBShift",
            "category": "kornia_only",
            "kornia": lambda: K.RandomRGBShift(p=1.0),
            "tv": None,
            "alb": None,
        }
    )

    registry.append(
        {
            "name": "RandomPlanckianJitter",
            "category": "kornia_only",
            "kornia": lambda: K.RandomPlanckianJitter(p=1.0),
            "tv": None,
            "alb": None,
        }
    )

    registry.append(
        {
            "name": "RandomCLAHE",
            "category": "kornia_only",
            "kornia": lambda: K.RandomClahe(p=1.0),
            "tv": None,
            "alb": None,
        }
    )

    return registry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.perf_counter()

    print("=" * 70)
    print("Per-op eager benchmark: kornia (patched) vs torchvision.v2 vs albumentations")
    print(f"Patch status: {_patch_status}")
    print(f"B={BATCH}, res={RES}x{RES}, fp32, {WARMUP}wu + {RUNS} timed")
    print("=" * 70)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    import albumentations
    import torchvision

    import kornia

    print(f"GPU: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"kornia: {kornia.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"albumentations: {albumentations.__version__}")
    print()

    torch.manual_seed(42)
    x_gpu = torch.rand(BATCH, 3, RES, RES, device="cuda")
    x_np = (np.random.default_rng(42).random((BATCH, RES, RES, 3)) * 255).astype(np.uint8)

    registry = _build_registry()

    results = {}
    total_attempted = 0
    kornia_ok = 0
    tv_ok = 0
    alb_ok = 0

    for entry in registry:
        name = entry["name"]
        category = entry["category"]
        total_attempted += 1
        print(f"[{name}] ({category})", end="", flush=True)

        row: dict = {"name": name, "category": category}

        # Kornia
        if entry.get("kornia") is not None:
            print(" K...", end="", flush=True)
            r = time_gpu(entry["kornia"], x_gpu)
            row["kornia"] = r
            if r["status"] == "ok":
                kornia_ok += 1
                print(f"OK({r['median_ms']:.2f}ms)", end="", flush=True)
            else:
                print(f"FAIL({r['reason'][:40]})", end="", flush=True)
        else:
            row["kornia"] = {"status": "skip", "reason": "not available"}

        # torchvision
        if entry.get("tv") is not None:
            print(" TV...", end="", flush=True)
            tv_timing = entry.get("tv_timing", "standard")
            if tv_timing == "with_labels":
                r = time_gpu_with_labels(entry["tv"], x_gpu)
            else:
                r = time_gpu(entry["tv"], x_gpu)
            row["tv"] = r
            if r["status"] == "ok":
                tv_ok += 1
                print(f"OK({r['median_ms']:.2f}ms)", end="", flush=True)
            else:
                print(f"FAIL({r['reason'][:40]})", end="", flush=True)
        else:
            row["tv"] = {"status": "skip", "reason": "not available"}

        # Albumentations
        if entry.get("alb") is not None:
            print(" ALB...", end="", flush=True)
            r = time_alb(entry["alb"], x_np)
            row["alb"] = r
            if r["status"] == "ok":
                alb_ok += 1
                print(f"OK({r['median_ms']:.2f}ms CPU)", end="", flush=True)
            else:
                print(f"FAIL({r['reason'][:40]})", end="", flush=True)
        else:
            row["alb"] = {"status": "skip", "reason": "not available"}

        print()
        results[name] = row

    t_elapsed = time.perf_counter() - t_start

    # Write JSON
    json_path = OUT_DIR / "results_per_op.json"
    summary = {
        "meta": {
            "date": "2026-04-27",
            "device": device_name,
            "torch": torch.__version__,
            "kornia": kornia.__version__,
            "torchvision": torchvision.__version__,
            "albumentations": albumentations.__version__,
            "batch": BATCH,
            "res": RES,
            "warmup": WARMUP,
            "runs": RUNS,
            "patch_status": _patch_status,
            "elapsed_s": round(t_elapsed, 1),
        },
        "results": results,
        "totals": {
            "attempted": total_attempted,
            "kornia_ok": kornia_ok,
            "tv_ok": tv_ok,
            "alb_ok": alb_ok,
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {json_path}")

    _write_leaderboard(summary)

    print(f"\nTotal elapsed: {t_elapsed:.1f}s")
    print(f"Attempted: {total_attempted} | kornia OK: {kornia_ok} | TV OK: {tv_ok} | Alb OK: {alb_ok}")


def _fmt_ms(r: dict) -> str:
    if r.get("status") != "ok":
        return "SKIP"
    cpu_tag = " *(CPU)*" if r.get("cpu_only") else ""
    return f"{r['median_ms']:.2f}{cpu_tag}"


def _median(r: dict):
    if r.get("status") != "ok":
        return None
    return r["median_ms"]


def _write_leaderboard(summary: dict) -> None:
    meta = summary["meta"]
    results = summary["results"]

    categories: dict = {}
    for name, row in results.items():
        cat = row["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(row)

    lines = []
    lines.append("# Per-op eager benchmark -- kornia (patched) vs torchvision.v2 vs albumentations")
    lines.append("")
    lines.append(f"**Date:** {meta['date']}  ")
    lines.append(f"**GPU:** {meta['device']}  ")
    lines.append(f"**PyTorch:** {meta['torch']}  ")
    lines.append(f"**kornia:** {meta['kornia']} (5 eager patches applied)  ")
    lines.append(f"**torchvision:** {meta['torchvision']}  ")
    lines.append(f"**albumentations:** {meta['albumentations']}  ")
    lines.append(f"**Input:** B={meta['batch']}, {meta['res']}x{meta['res']}, fp32 GPU (kornia/TV); uint8 CPU (Alb)  ")
    lines.append(f"**Timing:** {meta['warmup']} warmup + {meta['runs']} CUDA-event iterations  ")
    lines.append(f"**Total elapsed:** {meta['elapsed_s']}s  ")
    lines.append("")
    lines.append(
        "> Note: albumentations times are CPU-only (per-image loop over uint8 HWC numpy). "
        "GPU vs CPU comparisons are informational only -- not apples-to-apples."
    )
    lines.append("")

    cat_display_names = {
        "geometric": "Geometric",
        "intensity_color": "Intensity (color / brightness)",
        "intensity_blur": "Intensity (blur / noise)",
        "erasing": "Erasing",
        "normalize": "Normalize",
        "mix": "Mix",
        "kornia_only": "Kornia-only ops",
    }

    kornia_wins = []
    tv_wins = []
    tied = []

    for cat_key in ["geometric", "intensity_color", "intensity_blur", "erasing", "normalize", "mix", "kornia_only"]:
        rows = categories.get(cat_key, [])
        if not rows:
            continue

        display = cat_display_names.get(cat_key, cat_key)
        lines.append(f"## {display}")
        lines.append("")

        if cat_key == "kornia_only":
            lines.append("| Op | kornia ms (GPU) |")
            lines.append("|---|---:|")
            for row in rows:
                k_ms = _fmt_ms(row.get("kornia", {}))
                lines.append(f"| {row['name']} | {k_ms} |")
        else:
            lines.append("| Op | kornia ms | torchvision ms | albumentations ms (CPU) | k/tv ratio | winner |")
            lines.append("|---|---:|---:|---:|---:|---|")
            for row in rows:
                name = row["name"]
                k = row.get("kornia", {})
                tv = row.get("tv", {})
                alb = row.get("alb", {})
                k_ms = _fmt_ms(k)
                tv_ms = _fmt_ms(tv)
                alb_ms = _fmt_ms(alb)

                k_v = _median(k)
                tv_v = _median(tv)

                if k_v is not None and tv_v is not None:
                    ratio = k_v / tv_v
                    ratio_str = f"{ratio:.2f}x"
                    if ratio < 0.9:
                        winner = "**kornia**"
                        kornia_wins.append((name, ratio))
                    elif ratio > 1.1:
                        winner = "torchvision"
                        tv_wins.append((name, ratio))
                    else:
                        winner = "tied"
                        tied.append((name, ratio))
                elif k_v is not None and tv_v is None:
                    ratio_str = "--"
                    winner = "kornia-only"
                elif k_v is None and tv_v is not None:
                    ratio_str = "--"
                    winner = "tv-only"
                else:
                    ratio_str = "--"
                    winner = "--"

                lines.append(f"| {name} | {k_ms} | {tv_ms} | {alb_ms} | {ratio_str} | {winner} |")

        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total transforms attempted: {summary['totals']['attempted']}")
    lines.append(f"- kornia OK: {summary['totals']['kornia_ok']}")
    lines.append(f"- torchvision OK: {summary['totals']['tv_ok']}")
    lines.append(f"- albumentations OK: {summary['totals']['alb_ok']}")
    lines.append("")

    kornia_wins_sorted = sorted(kornia_wins, key=lambda x: x[1])
    tv_wins_sorted = sorted(tv_wins, key=lambda x: x[1], reverse=True)

    lines.append("### kornia wins vs torchvision (k/tv < 0.9, lower is better for kornia)")
    if kornia_wins_sorted:
        for name, ratio in kornia_wins_sorted:
            lines.append(f"  - {name}: {ratio:.2f}x (kornia {(1 - ratio) * 100:.0f}% faster)")
    else:
        lines.append("  - (none)")
    lines.append("")

    lines.append("### torchvision wins vs kornia (k/tv > 1.1)")
    if tv_wins_sorted:
        for name, ratio in tv_wins_sorted:
            lines.append(f"  - {name}: {ratio:.2f}x (torchvision {(ratio - 1) * 100:.0f}% faster)")
    else:
        lines.append("  - (none)")
    lines.append("")

    lines.append("### Tied (within 10%)")
    if tied:
        for name, ratio in sorted(tied, key=lambda x: x[1]):
            lines.append(f"  - {name}: {ratio:.2f}x")
    else:
        lines.append("  - (none)")
    lines.append("")

    lines.append("> albumentations runs on CPU; times are not directly comparable to GPU kornia/torchvision times.")
    lines.append("")

    md_path = OUT_DIR / "leaderboard_per_op.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

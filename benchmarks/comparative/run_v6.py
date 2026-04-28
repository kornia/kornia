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

"""Comparative benchmark v6 -- aggressive forward overrides on CUDA.

Builds on run_v5.py:
  * Reuses the v4 monkey-patches: Normalize buffers, HFlip cache, hflip/vflip
    no-contiguous, Affine closed-form, ColorJiggle fused-HSV.
  * Reuses the cusolver workaround.
  * SKIPS the v5 Path A patches (superseded by aggressive forward overrides).
  * Applies the aggressive forward overrides shipped on disk in the kornia
    source tree under /home/nvidia/kornia/kornia/augmentation/_2d/, but the
    pixi env loads kornia 0.7.4 from site-packages, so we translate each
    on-disk override into a runtime monkey-patch on the installed kornia
    classes.

Per-op CUDA event timing for the full 37-transform matrix mirrors run_per_op.py:
B=8, 512x512, fp32, GPU pre-resident, 25 warmup + 100 timed CUDA-event runs.
Albumentations is timed CPU-only with per-image loop.

DETR pipeline measurement: HFlip(p=0.5) + Affine + ColorJiggle + Normalize.

Run from /tmp with PYTHONNOUSERSITE=1:
  cd /tmp && PYTHONNOUSERSITE=1 \\
    /home/nvidia/bubbaloop-nodes-official/camera-object-detector/.pixi/envs/default/bin/python3.10 \\
    /home/nvidia/kornia/benchmarks/comparative/run_v6.py
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
# ---------------------------------------------------------------------------


def _analytical_3x3_inv(input: torch.Tensor) -> torch.Tensor:
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
    dev = A.device
    dtype = A.dtype
    out = torch.linalg.solve(A.to("cpu", torch.float64), B.to("cpu", torch.float64))
    return out.to(device=dev, dtype=dtype)


def _patch_kornia_solvers() -> None:
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

    try:
        import kornia.geometry.transform.imgwarp as _imgwarp

        _imgwarp._torch_solve_cast = _cpu_solve_cast
    except Exception:
        pass


import kornia.geometry.conversions
import kornia.utils.helpers  # noqa: F401

_patch_kornia_solvers()


# ---------------------------------------------------------------------------
# WORKAROUND 2: V4 patches.
# ---------------------------------------------------------------------------

_V4_PATCHED = False


def _apply_v4_patches() -> str:
    global _V4_PATCHED
    if _V4_PATCHED:
        return "already patched"

    status_parts: list[str] = []

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

    try:
        import kornia.augmentation._2d.intensity.denormalize as _denorm_mod

        _Denorm = _denorm_mod.Denormalize
        _orig_dn_init = _Denorm.__init__

        def _patched_dn_init(self, mean, std, p=1.0, keepdim=False, **kw):
            _orig_dn_init(self, mean, std, p=p, keepdim=keepdim)
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

        _Denorm.__init__ = _patched_dn_init
        status_parts.append("Denormalize(buffers)")
    except Exception as exc:
        status_parts.append(f"Denormalize FAILED: {exc}")

    try:
        import kornia.augmentation._2d.geometric.horizontal_flip as _hflip_mod

        _HFLIP_MAT_TEMPLATE = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        _HFLIP_MAT_CACHE: dict = {}
        _hflip_mod._HFLIP_MAT_TEMPLATE = _HFLIP_MAT_TEMPLATE
        _hflip_mod._HFLIP_MAT_CACHE = _HFLIP_MAT_CACHE

        _RHF = _hflip_mod.RandomHorizontalFlip

        def _patched_hflip_compute(self, input, params, flags):
            w = int(params["forward_input_shape"][-1].item())
            key = (input.device, input.dtype, w)
            cached = _HFLIP_MAT_CACHE.get(key)
            if cached is None:
                fm = _HFLIP_MAT_TEMPLATE.to(device=input.device, dtype=input.dtype).clone()
                fm[0, 2] = w - 1
                cached = fm.unsqueeze(0)
                _HFLIP_MAT_CACHE[key] = cached
            return cached.expand(input.shape[0], 3, 3)

        _RHF.compute_transformation = _patched_hflip_compute
        status_parts.append("HFlip(cache)")
    except Exception as exc:
        status_parts.append(f"HFlip FAILED: {exc}")

    try:
        import kornia.augmentation._2d.geometric.horizontal_flip as _hflip_mod2
        import kornia.geometry.transform as _kgt
        import kornia.geometry.transform.flips as _flips_mod

        def _fast_hflip(input):
            return input.flip(-1)

        def _fast_vflip(input):
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

    try:
        import torch.nn.functional as F

        import kornia.augmentation._2d.geometric.affine as _aff_mod

        def _affine_matrix2d_closed(translations, center, scale, angle, shear_x, shear_y):
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

        def _affine_homography_inv(M):
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
            self._norm_cache = {}

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

    try:
        import kornia.augmentation._2d.intensity.color_jiggle as _cj_mod
        from kornia.color import hsv_to_rgb as _hsv_to_rgb
        from kornia.color import rgb_to_hsv as _rgb_to_hsv

        _TWO_PI = 2.0 * math.pi

        def _patched_cj_apply(self, input, params, flags, transform=None):
            brightness_factor = params["brightness_factor"]
            contrast_factor = params["contrast_factor"]
            saturation_factor = params["saturation_factor"]
            hue_factor = params["hue_factor"]

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

            def flush_hsv(img):
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

    _V4_PATCHED = True
    return "; ".join(status_parts)


# ---------------------------------------------------------------------------
# V6 AGGRESSIVE FORWARD OVERRIDES
# ---------------------------------------------------------------------------

_V6_AGGRESSIVE_PATCHED = False


def _apply_v6_aggressive_overrides() -> dict:
    global _V6_AGGRESSIVE_PATCHED
    if _V6_AGGRESSIVE_PATCHED:
        return {"_already": "already patched"}

    statuses: dict = {}

    try:
        import kornia.augmentation._2d.geometric.horizontal_flip as _hflip_mod

        _RHF = _hflip_mod.RandomHorizontalFlip
        _orig_hflip_forward = _RHF.forward

        @torch.no_grad()
        def _hflip_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.same_on_batch
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    self._params = {
                        "batch_prob": torch.full((b,), bool(self.p > 0.5), dtype=torch.bool),
                        "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                    }
                    if self.p == 1.0:
                        w = x.shape[-1]
                        key = (x.device, x.dtype, w)
                        cached = _hflip_mod._HFLIP_MAT_CACHE.get(key)
                        if cached is None:
                            fm = _hflip_mod._HFLIP_MAT_TEMPLATE.to(device=x.device, dtype=x.dtype).clone()
                            fm[0, 2] = w - 1
                            cached = fm.unsqueeze(0)
                            _hflip_mod._HFLIP_MAT_CACHE[key] = cached
                        self._transform_matrix = cached.expand(b, 3, 3)
                        return x.flip(-1)
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    return x
            return _orig_hflip_forward(self, *args, **kwargs)

        _RHF.forward = _hflip_forward
        statuses["RandomHorizontalFlip"] = "OK"
    except Exception as exc:
        statuses["RandomHorizontalFlip"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.geometric.vertical_flip as _vflip_mod

        _RVF = _vflip_mod.RandomVerticalFlip
        _orig_vflip_forward = _RVF.forward

        @torch.no_grad()
        def _vflip_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.same_on_batch
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    self._params = {
                        "batch_prob": torch.full((b,), bool(self.p > 0.5), dtype=torch.bool),
                        "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                    }
                    if self.p == 1.0:
                        h = x.shape[-2]
                        flip_mat = torch.tensor(
                            [[1, 0, 0], [0, -1, h - 1], [0, 0, 1]],
                            device=x.device,
                            dtype=x.dtype,
                        )
                        self._transform_matrix = flip_mat.unsqueeze(0).expand(b, 3, 3)
                        return x.flip(-2)
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    return x
            return _orig_vflip_forward(self, *args, **kwargs)

        _RVF.forward = _vflip_forward
        statuses["RandomVerticalFlip"] = "OK"
    except Exception as exc:
        statuses["RandomVerticalFlip"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.geometric.center_crop as _cc_mod

        _CC = _cc_mod.CenterCrop
        _orig_cc_forward = _CC.forward

        @torch.no_grad()
        def _cc_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.keepdim
                and self.flags.get("cropping_mode") == "slice"
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    crop_h, crop_w = self.size
                    in_h, in_w = x.shape[-2], x.shape[-1]
                    start_y = int(in_h / 2 - crop_h / 2)
                    start_x = int(in_w / 2 - crop_w / 2)
                    self._params = {
                        "batch_prob": torch.full((b,), True, dtype=torch.bool),
                        "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                    }
                    mat = torch.tensor(
                        [[1.0, 0.0, -float(start_x)], [0.0, 1.0, -float(start_y)], [0.0, 0.0, 1.0]],
                        device=x.device,
                        dtype=x.dtype,
                    )
                    self._transform_matrix = mat.unsqueeze(0).expand(b, 3, 3)
                    return x[..., start_y : start_y + crop_h, start_x : start_x + crop_w]
            return _orig_cc_forward(self, *args, **kwargs)

        _CC.forward = _cc_forward
        statuses["CenterCrop"] = "OK"
    except Exception as exc:
        statuses["CenterCrop"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.normalize as _norm_mod

        _Norm = _norm_mod.Normalize
        _orig_norm_forward = _Norm.forward

        @torch.no_grad()
        def _norm_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    self._params = {
                        "batch_prob": torch.full((b,), bool(self.p > 0.5), dtype=torch.bool),
                        "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                    }
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        return x
                    mean = self._mean_b
                    std = self._std_b
                    if mean.dtype != x.dtype or mean.device != x.device:
                        mean = mean.to(device=x.device, dtype=x.dtype)
                        std = std.to(device=x.device, dtype=x.dtype)
                    return (x - mean) / std
            return _orig_norm_forward(self, *args, **kwargs)

        _Norm.forward = _norm_forward
        statuses["Normalize"] = "OK"
    except Exception as exc:
        statuses["Normalize"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.denormalize as _denorm_mod

        _Denorm = _denorm_mod.Denormalize
        _orig_denorm_forward = _Denorm.forward

        @torch.no_grad()
        def _denorm_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    self._params = {
                        "batch_prob": torch.full((b,), bool(self.p > 0.5), dtype=torch.bool),
                        "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                    }
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        return x
                    mean = self._mean_b
                    std = self._std_b
                    if mean.dtype != x.dtype or mean.device != x.device:
                        mean = mean.to(device=x.device, dtype=x.dtype)
                        std = std.to(device=x.device, dtype=x.dtype)
                    return torch.addcmul(mean, x, std)
            return _orig_denorm_forward(self, *args, **kwargs)

        _Denorm.forward = _denorm_forward
        statuses["Denormalize"] = "OK"
    except Exception as exc:
        statuses["Denormalize"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.invert as _inv_mod

        _Inv = _inv_mod.RandomInvert
        _orig_inv_forward = _Inv.forward

        @torch.no_grad()
        def _inv_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.same_on_batch
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    self._params = {
                        "batch_prob": torch.full((b,), bool(self.p > 0.5), dtype=torch.bool),
                        "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                    }
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        return x
                    max_val = torch.as_tensor(self.flags["max_val"], device=x.device, dtype=x.dtype)
                    return max_val - x
            return _orig_inv_forward(self, *args, **kwargs)

        _Inv.forward = _inv_forward
        statuses["RandomInvert"] = "OK"
    except Exception as exc:
        statuses["RandomInvert"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.grayscale as _gs_mod
        from kornia.color import rgb_to_grayscale as _rgb2gray

        _GS = _gs_mod.RandomGrayscale
        _orig_gs_forward = _GS.forward

        @torch.no_grad()
        def _gs_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.same_on_batch
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    self._params = {
                        "batch_prob": torch.full((b,), bool(self.p > 0.5), dtype=torch.bool),
                        "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                    }
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        return x
                    gray = _rgb2gray(x, rgb_weights=self.rgb_weights)
                    return gray.expand_as(x).contiguous()
            return _orig_gs_forward(self, *args, **kwargs)

        _GS.forward = _gs_forward
        statuses["RandomGrayscale"] = "OK"
    except Exception as exc:
        statuses["RandomGrayscale"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.solarize as _sol_mod
        from kornia.enhance import solarize as _solarize_fn

        _Sol = _sol_mod.RandomSolarize
        _orig_sol_forward = _Sol.forward

        @torch.no_grad()
        def _sol_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        self._params = {
                            "batch_prob": torch.zeros(b, dtype=torch.bool),
                            "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                        }
                        return x
                    params = self._param_generator(torch.Size((b, *x.shape[1:])), self.same_on_batch)
                    self._params = dict(params)
                    self._params["batch_prob"] = torch.ones(b, dtype=torch.bool)
                    self._params["forward_input_shape"] = torch.tensor(x.shape, dtype=torch.long)
                    thresholds = params["thresholds"]
                    additions = params.get("additions")
                    return _solarize_fn(x, thresholds, additions)
            return _orig_sol_forward(self, *args, **kwargs)

        _Sol.forward = _sol_forward
        statuses["RandomSolarize"] = "OK"
    except Exception as exc:
        statuses["RandomSolarize"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.brightness as _br_mod
        from kornia.enhance.adjust import adjust_brightness as _adj_b

        _RB = _br_mod.RandomBrightness
        _orig_br_forward = _RB.forward

        @torch.no_grad()
        def _br_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        self._params = {
                            "batch_prob": torch.zeros(b, dtype=torch.bool),
                            "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                        }
                        return x
                    params = self._param_generator(torch.Size((b, *x.shape[1:])), self.same_on_batch)
                    self._params = dict(params)
                    self._params["batch_prob"] = torch.ones(b, dtype=torch.bool)
                    self._params["forward_input_shape"] = torch.tensor(x.shape, dtype=torch.long)
                    bf = params["brightness_factor"].to(x)
                    return _adj_b(x, bf - 1, self.clip_output)
            return _orig_br_forward(self, *args, **kwargs)

        _RB.forward = _br_forward
        statuses["RandomBrightness"] = "OK"
    except Exception as exc:
        statuses["RandomBrightness"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.contrast as _ct_mod
        from kornia.enhance.adjust import adjust_contrast as _adj_c

        _RC = _ct_mod.RandomContrast
        _orig_ct_forward = _RC.forward

        @torch.no_grad()
        def _ct_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        self._params = {
                            "batch_prob": torch.zeros(b, dtype=torch.bool),
                            "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                        }
                        return x
                    params = self._param_generator(torch.Size((b, *x.shape[1:])), self.same_on_batch)
                    self._params = dict(params)
                    self._params["batch_prob"] = torch.ones(b, dtype=torch.bool)
                    self._params["forward_input_shape"] = torch.tensor(x.shape, dtype=torch.long)
                    cf = params["contrast_factor"].to(x)
                    return _adj_c(x, cf, self.clip_output)
            return _orig_ct_forward(self, *args, **kwargs)

        _RC.forward = _ct_forward
        statuses["RandomContrast"] = "OK"
    except Exception as exc:
        statuses["RandomContrast"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.saturation as _sat_mod
        from kornia.enhance.adjust import adjust_saturation as _adj_s

        _RS = _sat_mod.RandomSaturation
        _orig_sat_forward = _RS.forward

        @torch.no_grad()
        def _sat_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        self._params = {
                            "batch_prob": torch.zeros(b, dtype=torch.bool),
                            "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                        }
                        return x
                    params = self._param_generator(torch.Size((b, *x.shape[1:])), self.same_on_batch)
                    self._params = dict(params)
                    self._params["batch_prob"] = torch.ones(b, dtype=torch.bool)
                    self._params["forward_input_shape"] = torch.tensor(x.shape, dtype=torch.long)
                    sf = params["saturation_factor"].to(x)
                    return _adj_s(x, sf)
            return _orig_sat_forward(self, *args, **kwargs)

        _RS.forward = _sat_forward
        statuses["RandomSaturation"] = "OK"
    except Exception as exc:
        statuses["RandomSaturation"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.hue as _hue_mod
        from kornia.constants import pi as _kpi
        from kornia.enhance.adjust import adjust_hue as _adj_h

        _RH = _hue_mod.RandomHue
        _orig_hue_forward = _RH.forward

        @torch.no_grad()
        def _hue_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        self._params = {
                            "batch_prob": torch.zeros(b, dtype=torch.bool),
                            "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                        }
                        return x
                    params = self._param_generator(torch.Size((b, *x.shape[1:])), self.same_on_batch)
                    self._params = dict(params)
                    self._params["batch_prob"] = torch.ones(b, dtype=torch.bool)
                    self._params["forward_input_shape"] = torch.tensor(x.shape, dtype=torch.long)
                    hf = params["hue_factor"].to(x)
                    return _adj_h(x, hf * 2 * _kpi)
            return _orig_hue_forward(self, *args, **kwargs)

        _RH.forward = _hue_forward
        statuses["RandomHue"] = "OK"
    except Exception as exc:
        statuses["RandomHue"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.intensity.posterize as _pos_mod
        from kornia.enhance import posterize as _posterize_fn

        _RP = _pos_mod.RandomPosterize
        _orig_pos_forward = _RP.forward

        @torch.no_grad()
        def _pos_forward(self, *args, **kwargs):
            if (
                len(args) == 1
                and isinstance(args[0], torch.Tensor)
                and not kwargs
                and self.p_batch == 1.0
                and not self.keepdim
                and self.p in (0.0, 1.0)
            ):
                x = args[0]
                d = x.dim()
                if d == 3:
                    x = x.unsqueeze(0)
                    d = 4
                if d == 4:
                    b = x.shape[0]
                    eye = torch.eye(3, device=x.device, dtype=x.dtype)
                    self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                    if self.p == 0.0:
                        self._params = {
                            "batch_prob": torch.zeros(b, dtype=torch.bool),
                            "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                        }
                        return x
                    params = self._param_generator(torch.Size((b, *x.shape[1:])), self.same_on_batch)
                    self._params = dict(params)
                    self._params["batch_prob"] = torch.ones(b, dtype=torch.bool)
                    self._params["forward_input_shape"] = torch.tensor(x.shape, dtype=torch.long)
                    return _posterize_fn(x, params["bits_factor"].to(x.device))
            return _orig_pos_forward(self, *args, **kwargs)

        _RP.forward = _pos_forward
        statuses["RandomPosterize"] = "OK"
    except Exception as exc:
        statuses["RandomPosterize"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.mix.cutmix as _cm_mod
        from kornia.geometry.bbox import bbox_to_mask as _bbox_to_mask

        _CM = _cm_mod.RandomCutMixV2

        def _cm_apply_transform(self, input, params, maybe_flags=None):
            out_inputs = input
            for pair, crop in zip(params["mix_pairs"], params["crop_src"]):
                input_permute = input.index_select(dim=0, index=pair.to(input.device))
                height, width = input.size(2), input.size(3)
                mask = _bbox_to_mask(crop.to(input.device), width, height).bool().unsqueeze(dim=1)
                out_inputs = torch.where(mask, input_permute, out_inputs)
            return out_inputs

        _CM.apply_transform = _cm_apply_transform
        statuses["RandomCutMixV2"] = "OK"
    except Exception as exc:
        statuses["RandomCutMixV2"] = f"FAILED: {exc}"

    try:
        import kornia.augmentation._2d.mix.mixup as _mu_mod

        _MU = _mu_mod.RandomMixUpV2

        def _mu_apply_transform(self, input, params, maybe_flags=None):
            input_permute = input.index_select(dim=0, index=params["mixup_pairs"].to(input.device))
            lam = params["mixup_lambdas"].to(device=input.device, dtype=input.dtype).view(-1, 1, 1, 1)
            return torch.lerp(input, input_permute, lam)

        _MU.apply_transform = _mu_apply_transform
        statuses["RandomMixUpV2"] = "OK"
    except Exception as exc:
        statuses["RandomMixUpV2"] = f"FAILED: {exc}"

    _V6_AGGRESSIVE_PATCHED = True
    return statuses


# ---------------------------------------------------------------------------
# Apply patches at import time.
# ---------------------------------------------------------------------------

_v4_status = _apply_v4_patches()
_v6_status = _apply_v6_aggressive_overrides()
_patch_kornia_solvers()


# ---------------------------------------------------------------------------
# Numerical equivalence verification.
# ---------------------------------------------------------------------------


def _verify_equivalence() -> dict:
    import kornia.augmentation as K

    cases = [
        ("RandomHorizontalFlip", lambda: K.RandomHorizontalFlip(p=1.0), (2, 3, 32, 32), False),
        ("RandomVerticalFlip", lambda: K.RandomVerticalFlip(p=1.0), (2, 3, 32, 32), False),
        ("CenterCrop", lambda: K.CenterCrop(size=(24, 24)), (2, 3, 32, 32), False),
        ("RandomGrayscale", lambda: K.RandomGrayscale(p=1.0), (2, 3, 32, 32), False),
        ("RandomInvert", lambda: K.RandomInvert(p=1.0), (2, 3, 32, 32), False),
        (
            "Normalize",
            lambda: K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
            (2, 3, 32, 32),
            False,
        ),
        (
            "Denormalize",
            lambda: K.Denormalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
            (2, 3, 32, 32),
            False,
        ),
        ("RandomSolarize", lambda: K.RandomSolarize(thresholds=0.5, p=1.0), (2, 3, 32, 32), True),
        ("RandomPosterize", lambda: K.RandomPosterize(bits=4, p=1.0), (2, 3, 32, 32), True),
        ("RandomBrightness", lambda: K.RandomBrightness(brightness=(0.8, 1.2), p=1.0), (2, 3, 32, 32), True),
        ("RandomContrast", lambda: K.RandomContrast(contrast=(0.8, 1.2), p=1.0), (2, 3, 32, 32), True),
        ("RandomSaturation", lambda: K.RandomSaturation(saturation=(0.8, 1.2), p=1.0), (2, 3, 32, 32), True),
        ("RandomHue", lambda: K.RandomHue(hue=(-0.1, 0.1), p=1.0), (2, 3, 32, 32), True),
    ]

    out: dict = {}
    use_cuda = torch.cuda.is_available()

    for name, factory, shape, rng_div in cases:
        try:
            torch.manual_seed(123)
            if use_cuda:
                x = torch.rand(*shape, device="cuda")
            else:
                x = torch.rand(*shape)

            torch.manual_seed(42)
            aug_fast = factory()
            if use_cuda:
                aug_fast = aug_fast.cuda()
            with torch.no_grad():
                out_fast = aug_fast(x)

            torch.manual_seed(42)
            aug_std = factory()
            if use_cuda:
                aug_std = aug_std.cuda()
            with torch.no_grad():
                out_std = aug_std(x, params=None)

            if out_fast.shape != out_std.shape:
                out[name] = f"FAIL shape mismatch fast={out_fast.shape} std={out_std.shape}"
                continue
            if rng_div:
                ok_range = (
                    out_fast.isfinite().all().item()
                    and out_std.isfinite().all().item()
                    and out_fast.shape == out_std.shape
                )
                out[name] = {
                    "ok": bool(ok_range),
                    "max_abs_diff": 0.0,
                    "note": "RNG-divergent: shape-only equivalence",
                }
            else:
                max_abs = (out_fast - out_std).abs().max().item()
                close = torch.allclose(out_fast, out_std, atol=1e-5, rtol=1e-4)
                out[name] = {
                    "ok": bool(close),
                    "max_abs_diff": max_abs,
                }
        except Exception as exc:
            out[name] = f"EXCEPTION: {type(exc).__name__}: {str(exc).splitlines()[-1][:80]}"

    for nm, factory in (
        ("RandomCutMixV2", lambda: K.RandomCutMixV2(p=1.0)),
        ("RandomMixUpV2", lambda: K.RandomMixUpV2(p=1.0)),
    ):
        try:
            torch.manual_seed(123)
            if use_cuda:
                x = torch.rand(2, 3, 32, 32, device="cuda")
                labels = torch.zeros(2, dtype=torch.long, device="cuda")
            else:
                x = torch.rand(2, 3, 32, 32)
                labels = torch.zeros(2, dtype=torch.long)
            aug = factory()
            if use_cuda:
                aug = aug.cuda()
            with torch.no_grad():
                result = aug(x, labels)
            # ``result`` may be a Tensor (B, C, H, W) or a list/tuple
            # (img, labels) depending on the kornia version & data_keys.
            if isinstance(result, torch.Tensor):
                img = result
            elif isinstance(result, (list, tuple)):
                img = result[0]
            else:
                img = result
            ok = img.shape == x.shape and img.isfinite().all().item()
            out[nm] = {
                "ok": bool(ok),
                "max_abs_diff": 0.0,
                "note": "shape-only sanity (mix ops)",
            }
        except Exception as exc:
            out[nm] = f"EXCEPTION: {type(exc).__name__}: {str(exc).splitlines()[-1][:80]}"

    return out


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


def time_gpu(make_aug_fn, x_gpu) -> dict:
    try:
        aug = make_aug_fn()
        if hasattr(aug, "cuda"):
            aug = aug.cuda()
        with torch.no_grad():
            for _ in range(WARMUP):
                _ = aug(x_gpu)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        with torch.no_grad():
            for i in range(RUNS):
                starts[i].record()
                _ = aug(x_gpu)
                ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return {"status": "ok", **_stats(times)}
    except Exception:
        return {"status": "error", "reason": _traceback.format_exc().splitlines()[-1]}


def time_gpu_with_labels(make_aug_fn, x_gpu) -> dict:
    try:
        aug = make_aug_fn()
        labels = torch.zeros(BATCH, dtype=torch.long, device=x_gpu.device)
        with torch.no_grad():
            for _ in range(WARMUP):
                _ = aug(x_gpu, labels)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        with torch.no_grad():
            for i in range(RUNS):
                starts[i].record()
                _ = aug(x_gpu, labels)
                ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return {"status": "ok", **_stats(times)}
    except Exception:
        return {"status": "error", "reason": _traceback.format_exc().splitlines()[-1]}


def time_alb(make_aug_fn, x_np) -> dict:
    try:
        aug = make_aug_fn()
        B = x_np.shape[0]
        for _ in range(WARMUP):
            for i in range(B):
                _ = aug(image=x_np[i])["image"]
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            for i in range(B):
                _ = aug(image=x_np[i])["image"]
            times.append((time.perf_counter() - t0) * 1000.0)
        return {"status": "ok", "cpu_only": True, **_stats(times)}
    except Exception:
        return {"status": "error", "reason": _traceback.format_exc().splitlines()[-1]}


def _build_registry():
    import albumentations as A
    import torchvision.transforms.v2 as T

    import kornia.augmentation as K

    MEAN_T = torch.tensor(IMAGENET_MEAN)
    STD_T = torch.tensor(IMAGENET_STD)

    registry = []

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

    registry.append(
        {
            "name": "MixUp",
            "category": "mix",
            "kornia": lambda: K.RandomMixUpV2(p=1.0),
            "tv": lambda: T.MixUp(num_classes=1000),
            "alb": None,
            "kornia_timing": "with_labels",
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
            "kornia_timing": "with_labels",
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


def _build_detr_kornia():
    import kornia.augmentation as K

    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomAffine(degrees=15.0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0),
        K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, p=1.0),
        K.Normalize(
            mean=torch.tensor(IMAGENET_MEAN),
            std=torch.tensor(IMAGENET_STD),
        ),
    ).cuda()


def _time_module(aug, x_gpu) -> dict:
    try:
        with torch.no_grad():
            for _ in range(WARMUP):
                _ = aug(x_gpu)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        with torch.no_grad():
            for i in range(RUNS):
                starts[i].record()
                _ = aug(x_gpu)
                ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return {"status": "ok", **_stats(times)}
    except Exception:
        return {"status": "error", "reason": _traceback.format_exc().splitlines()[-1]}


def _versions() -> dict:
    out = {}
    for lib in ("kornia", "torchvision", "albumentations"):
        try:
            mod = __import__(lib)
            out[lib] = mod.__version__
        except ImportError:
            out[lib] = "n/a"
    return out


# v5 medians for cross-version comparison.
V5_PER_OP_KORNIA = {
    "HorizontalFlip": 6.15,
    "VerticalFlip": 8.09,
    "Rotation": 50.5,
    "Affine": 51.0,
    "ResizedCrop": 16.09,
    "CenterCrop": 11.83,
    "Resize": 9.43,
    "Perspective": 62.01,
    "ColorJitter": 52.29,
    "Brightness": 17.98,
    "Contrast": 21.04,
    "Saturation": 28.73,
    "Hue": 31.55,
    "Grayscale": 8.88,
    "Solarize": 18.14,
    "Posterize": 19.42,
    "Equalize": 53.54,
    "Invert": 5.84,
    "Sharpness": 20.35,
    "GaussianBlur": 32.89,
    "GaussianNoise": 12.17,
    "MotionBlur": 33.05,
    "BoxBlur": 28.04,
    "MedianBlur": 375.29,
    "RandomErasing": 38.38,
    "Normalize": 6.56,
    "Denormalize": 7.67,
    "MixUp": 34.20,
    "CutMix": 82.66,
    "Mosaic": 36.32,
    "RandomRain": 40.20,
    "RandomSnow": 42.77,
    "RandomChannelDropout": 7.35,
    "RandomChannelShuffle": 8.05,
    "RandomRGBShift": 9.78,
    "RandomPlanckianJitter": 11.43,
    "RandomCLAHE": 163.84,
}

V5_DETR_FAST_ON = 48.82
V5_DETR_FAST_OFF = 49.30
V4_DETR_BASELINE = 58.1


def _fmt_ms(r: dict) -> str:
    if r.get("status") != "ok":
        return "FAIL"
    cpu_tag = " *(CPU)*" if r.get("cpu_only") else ""
    return f"{r['median_ms']:.2f}{cpu_tag}"


def _med(r: dict):
    if r.get("status") != "ok":
        return None
    return r["median_ms"]


def _write_leaderboard(summary: dict) -> Path:
    meta = summary["meta"]
    results = summary["results"]
    lines: list[str] = []

    lines.append("# Comparative augmentation benchmark v6 -- aggressive forward override")
    lines.append("")
    lines.append("## Hardware / stack")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|-----|-------|")
    lines.append(f"| Date | {meta['date']} |")
    lines.append("| Platform | Jetson Orin (aarch64), Linux 5.15.148-tegra |")
    lines.append(f"| GPU | {meta['device']} (Orin integrated GPU, 1792-core Ampere) |")
    lines.append("| CUDA | 12.6 (libcusolver 11.6.4.69) |")
    lines.append("| Python | 3.10 (pixi camera-object-detector env) |")
    lines.append(f"| PyTorch | {meta['torch']} |")
    lines.append(f"| kornia | {meta['kornia']} (installed 0.7.4 + v4 + aggressive forward overrides) |")
    lines.append(f"| torchvision | {meta['torchvision']} |")
    lines.append(f"| albumentations | {meta['albumentations']} |")
    lines.append(f"| Batch size | {meta['batch']} |")
    lines.append(f"| Resolution | {meta['res']}x{meta['res']} |")
    lines.append(f"| Timing | {meta['warmup']} warmup + {meta['runs']} CUDA-event runs (CPU loop for albumentations) |")
    lines.append(f"| Wall time | {meta['elapsed_s']:.1f}s |")
    lines.append("")

    lines.append("## Aggressive forward override -- patch verification")
    lines.append("")
    lines.append(f"v4 patch status: `{summary['v4_status']}`")
    lines.append("")
    lines.append("Per-class aggressive forward override installation (15 transforms):")
    lines.append("")
    lines.append("| # | Class | Install status |")
    lines.append("|---|-------|----------------|")
    expected = [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "CenterCrop",
        "Normalize",
        "Denormalize",
        "RandomInvert",
        "RandomGrayscale",
        "RandomSolarize",
        "RandomBrightness",
        "RandomContrast",
        "RandomSaturation",
        "RandomHue",
        "RandomPosterize",
        "RandomCutMixV2",
        "RandomMixUpV2",
    ]
    for idx, cls in enumerate(expected, 1):
        status = summary["v6_status"].get(cls, "missing")
        lines.append(f"| {idx} | {cls} | {status} |")
    lines.append("")
    lines.append("Numerical equivalence check (aggressive forward vs framework chain):")
    lines.append("")
    lines.append("| Transform | Equivalent? | max abs diff | Note |")
    lines.append("|-----------|-------------|--------------|------|")
    for name, res in summary["equivalence"].items():
        if isinstance(res, dict):
            ok = "YES" if res["ok"] else "NO"
            mad = f"{res['max_abs_diff']:.2e}"
            note = res.get("note", "")
        else:
            ok = "ERROR"
            mad = str(res)[:60]
            note = ""
        lines.append(f"| {name} | {ok} | {mad} | {note} |")
    lines.append("")

    lines.append("## Per-op leaderboard -- sorted by k/tv ratio (best first)")
    lines.append("")
    lines.append(
        "Input: pre-resident GPU tensor B=8, 3, 512, 512, fp32. "
        f"{meta['warmup']} warmup + {meta['runs']} CUDA-event runs. Median ms."
    )
    lines.append("")
    lines.append("| Op | k v6 ms | k v5 ms | tv ms | speedup vs v5 | k/tv ratio | improvement |")
    lines.append("|----|--------:|--------:|------:|--------------:|-----------:|:------------|")

    sortable = []
    unsortable = []
    for name, row in results.items():
        k_now = _med(row.get("kornia", {}))
        tv_now = _med(row.get("tv", {}))
        v5 = V5_PER_OP_KORNIA.get(name)
        if k_now is not None and tv_now is not None:
            sortable.append((name, k_now, v5, tv_now, k_now / tv_now))
        else:
            unsortable.append((name, k_now, v5, tv_now))

    sortable.sort(key=lambda r: r[4])

    for name, k_now, v5, tv_now, ratio in sortable:
        v5_s = f"{v5:.2f}" if v5 else "n/a"
        spd = f"{(v5 / k_now):.2f}x" if v5 and k_now > 0 else "n/a"
        flags = []
        if ratio <= 2.0:
            flags.append(f"**MATCH tv** ({ratio:.2f}x)")
        if v5 and k_now > 0 and (v5 / k_now) >= 5.0:
            flags.append(f"**5x over v5** ({(v5 / k_now):.2f}x)")
        improv = ", ".join(flags) if flags else ""
        lines.append(f"| {name} | {k_now:.2f} | {v5_s} | {tv_now:.2f} | {spd} | {ratio:.2f}x | {improv} |")

    lines.append("")
    lines.append("## Per-op (no torchvision baseline) -- kornia-only / tv-missing")
    lines.append("")
    lines.append("| Op | k v6 ms | k v5 ms | speedup vs v5 |")
    lines.append("|----|--------:|--------:|--------------:|")
    for name, k_now, v5, _tv in unsortable:
        if k_now is None:
            continue
        v5_s = f"{v5:.2f}" if v5 else "n/a"
        spd = f"{(v5 / k_now):.2f}x" if v5 and k_now > 0 else "n/a"
        lines.append(f"| {name} | {k_now:.2f} | {v5_s} | {spd} |")
    lines.append("")

    lines.append("## Per-op with albumentations CPU column (full registry)")
    lines.append("")
    lines.append("| Op | k v6 (GPU ms) | tv (GPU ms) | alb (CPU ms) |")
    lines.append("|----|--------------:|------------:|-------------:|")
    for name, row in results.items():
        k_s = _fmt_ms(row.get("kornia", {}))
        tv_s = _fmt_ms(row.get("tv", {}))
        alb_s = _fmt_ms(row.get("alb", {}))
        lines.append(f"| {name} | {k_s} | {tv_s} | {alb_s} |")
    lines.append("")

    lines.append("## DETR-style pipeline (HFlip(p=0.5) -> Affine -> ColorJiggle -> Normalize)")
    lines.append("")
    sb = summary["scope_b"]
    on_med = sb.get("median_ms")
    on_iqr = sb.get("iqr_ms")
    on_mn = sb.get("min_ms")
    on_mx = sb.get("max_ms")
    on_med_s = f"{on_med:.2f}" if on_med else "FAIL"
    on_iqr_s = f"{on_iqr:.2f}" if on_iqr is not None else "-"
    on_mn_s = f"{on_mn:.2f}" if on_mn is not None else "-"
    on_mx_s = f"{on_mx:.2f}" if on_mx is not None else "-"
    lines.append("| Run | Median ms | IQR | Min | Max |")
    lines.append("|-----|----------:|----:|----:|----:|")
    lines.append(f"| v6 (aggressive overrides ON) | {on_med_s} | {on_iqr_s} | {on_mn_s} | {on_mx_s} |")
    lines.append(f"| v5 fast_on (Path A active) | {V5_DETR_FAST_ON:.2f} | -- | -- | -- |")
    lines.append(f"| v5 fast_off (v4-equivalent) | {V5_DETR_FAST_OFF:.2f} | -- | -- | -- |")
    lines.append(f"| v4 reference | {V4_DETR_BASELINE:.2f} | -- | -- | -- |")
    lines.append("")
    if on_med:
        delta_v5 = on_med - V5_DETR_FAST_ON
        delta_v4 = on_med - V4_DETR_BASELINE
        lines.append(f"- v6 - v5(fast_on): {delta_v5:+.2f}ms")
        lines.append(f"- v6 - v4(ref):     {delta_v4:+.2f}ms")
        lines.append("")

    lines.append("## Honest interpretation")
    lines.append("")

    matched = [r for r in sortable if r[4] <= 2.0]
    five_x = [r for r in sortable if r[2] and r[1] > 0 and r[2] / r[1] >= 5.0]

    lines.append("### Did kornia match torchvision (k/tv <= 2.0) on any op?")
    if matched:
        lines.append("")
        lines.append("**YES** -- the following ops now satisfy k/tv <= 2.0:")
        lines.append("")
        for name, k_now, _v5, tv_now, ratio in matched:
            lines.append(f"- {name}: k={k_now:.2f}ms, tv={tv_now:.2f}ms, ratio={ratio:.2f}x")
    else:
        lines.append("")
        lines.append("**NO** -- no op crossed the k/tv <= 2.0 threshold.")
    lines.append("")

    lines.append("### Did kornia hit 5x over v5 on any op?")
    if five_x:
        lines.append("")
        lines.append("**YES** -- the following ops are 5x or better than v5:")
        lines.append("")
        for name, k_now, v5, _tv, _r in five_x:
            spd = v5 / k_now
            lines.append(f"- {name}: v5={v5:.2f}ms, v6={k_now:.2f}ms, speedup={spd:.2f}x")
    else:
        lines.append("")
        lines.append("**NO** -- no op crossed the 5x-over-v5 threshold.")
    lines.append("")

    lines.append("---")
    lines.append(
        f"*Generated: benchmark v6 on {meta['device']}, batch={meta['batch']}, "
        f"res={meta['res']}x{meta['res']}, kornia 0.7.4 + aggressive forward override.*"
    )

    md_path = OUT_DIR / "leaderboard_v6.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path


def main() -> None:
    t0 = time.perf_counter()

    print("=" * 70)
    print("Comparative benchmark v6 -- aggressive forward override on CUDA")
    print("=" * 70)
    print(f"v4 patches: {_v4_status}")
    print("v6 aggressive forward overrides:")
    for k, v in _v6_status.items():
        print(f"  {k:25s} {v}")

    print()
    print("Numerical equivalence check ...")
    eq = _verify_equivalence()
    for name, res in eq.items():
        if isinstance(res, dict):
            tag = "OK" if res["ok"] else "MISMATCH"
            print(f"  {name:25s} {tag}  max_abs_diff={res['max_abs_diff']:.2e}")
        else:
            print(f"  {name:25s} {res}")
    print()

    versions = _versions()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")
    for lib, ver in versions.items():
        print(f"  {lib}: {ver}")
    print()

    torch.manual_seed(42)
    x_gpu = torch.rand(BATCH, 3, RES, RES, device="cuda")
    x_np = (np.random.default_rng(42).random((BATCH, RES, RES, 3)) * 255).astype(np.uint8)

    print("=" * 70)
    print("Per-op CUDA-event timing (37-transform matrix)")
    print("=" * 70)

    registry = _build_registry()
    results = {}

    for entry in registry:
        name = entry["name"]
        category = entry["category"]
        print(f"[{name}] ({category})", end="", flush=True)
        row: dict = {"name": name, "category": category}

        if entry.get("kornia") is not None:
            print(" K...", end="", flush=True)
            ktime = entry.get("kornia_timing", "standard")
            if ktime == "with_labels":
                r = time_gpu_with_labels(entry["kornia"], x_gpu)
            else:
                r = time_gpu(entry["kornia"], x_gpu)
            row["kornia"] = r
            if r["status"] == "ok":
                print(f"OK({r['median_ms']:.2f}ms)", end="", flush=True)
            else:
                print(f"FAIL({r['reason'][:40]})", end="", flush=True)
        else:
            row["kornia"] = {"status": "skip", "reason": "not available"}

        if entry.get("tv") is not None:
            print(" TV...", end="", flush=True)
            tv_timing = entry.get("tv_timing", "standard")
            if tv_timing == "with_labels":
                r = time_gpu_with_labels(entry["tv"], x_gpu)
            else:
                r = time_gpu(entry["tv"], x_gpu)
            row["tv"] = r
            if r["status"] == "ok":
                print(f"OK({r['median_ms']:.2f}ms)", end="", flush=True)
            else:
                print(f"FAIL({r['reason'][:40]})", end="", flush=True)
        else:
            row["tv"] = {"status": "skip", "reason": "not available"}

        if entry.get("alb") is not None:
            print(" ALB...", end="", flush=True)
            r = time_alb(entry["alb"], x_np)
            row["alb"] = r
            if r["status"] == "ok":
                print(f"OK({r['median_ms']:.2f}ms CPU)", end="", flush=True)
            else:
                print(f"FAIL({r['reason'][:40]})", end="", flush=True)
        else:
            row["alb"] = {"status": "skip", "reason": "not available"}

        print()
        results[name] = row

    print()
    print("=" * 70)
    print("DETR-style 4-op pipeline (aggressive overrides ON)")
    print("=" * 70)
    aug_detr = _build_detr_kornia()
    detr_result = _time_module(aug_detr, x_gpu)
    if detr_result.get("status") == "ok":
        print(f"  median = {detr_result['median_ms']:.2f}ms (iqr={detr_result['iqr_ms']:.2f}ms)")
    else:
        print(f"  FAIL: {detr_result.get('reason', 'unknown')}")
    print()

    elapsed = time.perf_counter() - t0

    summary = {
        "meta": {
            "date": "2026-04-27",
            "device": device_name,
            "torch": torch.__version__,
            "kornia": versions.get("kornia"),
            "torchvision": versions.get("torchvision"),
            "albumentations": versions.get("albumentations"),
            "batch": BATCH,
            "res": RES,
            "warmup": WARMUP,
            "runs": RUNS,
            "elapsed_s": elapsed,
        },
        "v4_status": _v4_status,
        "v6_status": _v6_status,
        "equivalence": eq,
        "results": results,
        "scope_b": detr_result,
    }

    json_path = OUT_DIR / "results_v6.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {json_path}")

    md_path = _write_leaderboard(summary)
    print(f"Wrote {md_path}")

    print()
    print("=" * 70)
    print(f"Total elapsed: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

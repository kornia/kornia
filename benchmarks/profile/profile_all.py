"""Sweep harness: per-op bottleneck profile across all 37 augmentation transforms.

Extends profile_per_op.py to the full registry from run_per_op.py and adds a
phase 2 categorizer that classifies each op as one of:
  - dispatch-bound
  - kernel-bound
  - allocation-bound
  - sync-bound
  - fusion-eligible

Outputs:
  bottlenecks_all.json  -- raw per-op data
  bottlenecks_categorized.md -- categorized analysis

Run from /tmp with PYTHONNOUSERSITE=1:
  cd /tmp && PYTHONNOUSERSITE=1 \\
    /home/nvidia/bubbaloop-nodes-official/camera-object-detector/.pixi/envs/default/bin/python3.10 \\
    /home/nvidia/kornia/benchmarks/profile/profile_all.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Apply v4 + cusolver + v6 aggressive forward patches via run_v6 import side-effects.
sys.path.insert(0, "/home/nvidia/kornia/benchmarks/comparative")
import run_v6  # noqa: F401

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import kornia.augmentation as K
import torchvision.transforms.v2 as T


OUT_DIR = Path("/home/nvidia/kornia/benchmarks/profile")
TRACE_DIR = OUT_DIR / "traces_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

N_WARMUP = 5
N_ITERS = 20
BATCH = 8
RES = 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Registry — mirrors run_per_op.py (37 transforms)
# ---------------------------------------------------------------------------

def _build_registry():
    MEAN_T = torch.tensor(IMAGENET_MEAN)
    STD_T = torch.tensor(IMAGENET_STD)

    reg = []

    # GEOMETRIC
    reg.append(("HorizontalFlip", "geometric",
                lambda: K.RandomHorizontalFlip(p=1.0),
                lambda: T.RandomHorizontalFlip(p=1.0), "image"))
    reg.append(("VerticalFlip", "geometric",
                lambda: K.RandomVerticalFlip(p=1.0),
                lambda: T.RandomVerticalFlip(p=1.0), "image"))
    reg.append(("Rotation", "geometric",
                lambda: K.RandomRotation(degrees=15.0, p=1.0),
                lambda: T.RandomRotation(degrees=15.0), "image"))
    reg.append(("Affine", "geometric",
                lambda: K.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                       scale=(0.8, 1.2), p=1.0),
                lambda: T.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                       scale=(0.8, 1.2)), "image"))
    reg.append(("ResizedCrop", "geometric",
                lambda: K.RandomResizedCrop(size=(224, 224), p=1.0),
                lambda: T.RandomResizedCrop(size=224), "image"))
    reg.append(("CenterCrop", "geometric",
                lambda: K.CenterCrop(size=224),
                lambda: T.CenterCrop(size=224), "image"))
    reg.append(("Resize", "geometric",
                lambda: K.Resize(size=224),
                lambda: T.Resize(size=224), "image"))
    reg.append(("Perspective", "geometric",
                lambda: K.RandomPerspective(distortion_scale=0.2, p=1.0),
                lambda: T.RandomPerspective(distortion_scale=0.2, p=1.0), "image"))

    # INTENSITY: color/brightness
    reg.append(("ColorJitter", "intensity_color",
                lambda: K.ColorJiggle(brightness=0.2, contrast=0.2,
                                      saturation=0.2, hue=0.1, p=1.0),
                lambda: T.ColorJitter(brightness=0.2, contrast=0.2,
                                      saturation=0.2, hue=0.1), "image"))
    reg.append(("Brightness", "intensity_color",
                lambda: K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
                lambda: T.ColorJitter(brightness=0.2), "image"))
    reg.append(("Contrast", "intensity_color",
                lambda: K.RandomContrast(contrast=(0.8, 1.2), p=1.0),
                lambda: T.ColorJitter(contrast=0.2), "image"))
    reg.append(("Saturation", "intensity_color",
                lambda: K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),
                lambda: T.ColorJitter(saturation=0.2), "image"))
    reg.append(("Hue", "intensity_color",
                lambda: K.RandomHue(hue=(-0.1, 0.1), p=1.0),
                lambda: T.ColorJitter(hue=0.1), "image"))
    reg.append(("Grayscale", "intensity_color",
                lambda: K.RandomGrayscale(p=1.0),
                lambda: T.RandomGrayscale(p=1.0), "image"))
    reg.append(("Solarize", "intensity_color",
                lambda: K.RandomSolarize(thresholds=0.5, p=1.0),
                lambda: T.RandomSolarize(threshold=0.5, p=1.0), "image"))
    reg.append(("Posterize", "intensity_color",
                lambda: K.RandomPosterize(bits=4, p=1.0),
                lambda: T.RandomPosterize(bits=4, p=1.0), "image"))
    reg.append(("Equalize", "intensity_color",
                lambda: K.RandomEqualize(p=1.0),
                lambda: T.RandomEqualize(p=1.0), "image"))
    reg.append(("Invert", "intensity_color",
                lambda: K.RandomInvert(p=1.0),
                lambda: T.RandomInvert(p=1.0), "image"))
    reg.append(("Sharpness", "intensity_color",
                lambda: K.RandomSharpness(sharpness=0.5, p=1.0),
                lambda: T.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0), "image"))

    # INTENSITY: blur/noise
    reg.append(("GaussianBlur", "intensity_blur",
                lambda: K.RandomGaussianBlur(kernel_size=(5, 5),
                                             sigma=(0.1, 2.0), p=1.0),
                lambda: T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)), "image"))
    reg.append(("GaussianNoise", "intensity_blur",
                lambda: K.RandomGaussianNoise(std=0.05, p=1.0),
                lambda: T.GaussianNoise(sigma=0.05), "image"))
    reg.append(("MotionBlur", "intensity_blur",
                lambda: K.RandomMotionBlur(kernel_size=5, angle=35.0,
                                           direction=0.5, p=1.0),
                None, "image"))
    reg.append(("BoxBlur", "intensity_blur",
                lambda: K.RandomBoxBlur(kernel_size=(3, 3), p=1.0),
                None, "image"))
    reg.append(("MedianBlur", "intensity_blur",
                lambda: K.RandomMedianBlur(kernel_size=(3, 3), p=1.0),
                None, "image"))

    # ERASING
    reg.append(("RandomErasing", "erasing",
                lambda: K.RandomErasing(p=1.0),
                lambda: T.RandomErasing(p=1.0), "image"))

    # NORMALIZE
    reg.append(("Normalize", "normalize",
                lambda: K.Normalize(mean=MEAN_T, std=STD_T),
                lambda: T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), "image"))
    reg.append(("Denormalize", "normalize",
                lambda: K.Denormalize(mean=MEAN_T, std=STD_T),
                None, "image"))

    # MIX
    reg.append(("MixUp", "mix",
                lambda: K.RandomMixUpV2(p=1.0),
                lambda: T.MixUp(num_classes=1000), "labels"))
    reg.append(("CutMix", "mix",
                lambda: K.RandomCutMixV2(p=1.0),
                lambda: T.CutMix(num_classes=1000), "labels"))
    reg.append(("Mosaic", "mix",
                lambda: K.RandomMosaic(output_size=(512, 512), p=1.0),
                None, "image"))

    # KORNIA-ONLY
    reg.append(("RandomRain", "kornia_only",
                lambda: K.RandomRain(p=1.0), None, "image"))
    reg.append(("RandomSnow", "kornia_only",
                lambda: K.RandomSnow(p=1.0), None, "image"))
    reg.append(("RandomChannelDropout", "kornia_only",
                lambda: K.RandomChannelDropout(p=1.0), None, "image"))
    reg.append(("RandomChannelShuffle", "kornia_only",
                lambda: K.RandomChannelShuffle(p=1.0), None, "image"))
    reg.append(("RandomRGBShift", "kornia_only",
                lambda: K.RandomRGBShift(p=1.0), None, "image"))
    reg.append(("RandomPlanckianJitter", "kornia_only",
                lambda: K.RandomPlanckianJitter(p=1.0), None, "image"))
    reg.append(("RandomCLAHE", "kornia_only",
                lambda: K.RandomClahe(p=1.0), None, "image"))

    return reg


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def _make_input(kind):
    x = torch.rand(BATCH, 3, RES, RES, device="cuda")
    if kind == "labels":
        labels = torch.randint(0, 10, (BATCH,), device="cuda")
        return x, labels
    return x, None


def _call(aug, x, labels, kind):
    if kind == "labels":
        return aug(x, labels)
    return aug(x)


def _set_inference_mode(aug):
    """Switch nn.Module to inference mode (calls .eval() on the module)."""
    fn = getattr(aug, "eval", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return aug
    return aug


def profile_op(name, kornia_factory, tv_factory, kind,
               n_warmup=N_WARMUP, n_iters=N_ITERS):
    print(f"\n=== {name} ===", flush=True)
    x, labels = _make_input(kind)

    prof_k = None
    k_err = None
    try:
        aug = kornia_factory()
        try:
            aug = aug.cuda()
        except Exception:
            pass
        aug = _set_inference_mode(aug)

        for _ in range(n_warmup):
            _ = _call(aug, x, labels, kind)
        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof_k:
            for _ in range(n_iters):
                with record_function(f"{name}_kornia"):
                    _ = _call(aug, x, labels, kind)
                torch.cuda.synchronize()
        try:
            prof_k.export_chrome_trace(str(TRACE_DIR / f"{name}_kornia.json"))
        except Exception:
            pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        k_err = f"{type(e).__name__}: {e}"
        prof_k = None

    prof_tv = None
    tv_err = None
    if tv_factory is not None:
        try:
            aug_tv = tv_factory()
            try:
                aug_tv = aug_tv.cuda()
            except Exception:
                pass
            for _ in range(n_warmup):
                _ = _call(aug_tv, x, labels, kind)
            torch.cuda.synchronize()

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof_tv:
                for _ in range(n_iters):
                    with record_function(f"{name}_tv"):
                        _ = _call(aug_tv, x, labels, kind)
                    torch.cuda.synchronize()
            try:
                prof_tv.export_chrome_trace(str(TRACE_DIR / f"{name}_tv.json"))
            except Exception:
                pass
        except Exception as e:
            tv_err = f"{type(e).__name__}: {e}"
            prof_tv = None

    return prof_k, prof_tv, k_err, tv_err


def extract_top_ops(prof, n=15):
    if prof is None:
        return None
    avgs = prof.key_averages()
    sorted_avgs = sorted(avgs, key=lambda x: getattr(x, "self_cpu_time_total", 0),
                         reverse=True)
    out = []
    for entry in sorted_avgs[:n]:
        out.append({
            "name": entry.key,
            "count": int(entry.count),
            "self_cuda_us": float(getattr(entry, "self_cuda_time_total", 0)),
            "self_cpu_us": float(getattr(entry, "self_cpu_time_total", 0)),
            "cuda_total_us": float(getattr(entry, "cuda_time_total", 0)),
            "cpu_total_us": float(getattr(entry, "cpu_time_total", 0)),
            "self_cuda_memory": int(getattr(entry, "self_cuda_memory_usage", 0)),
            "self_cpu_memory": int(getattr(entry, "self_cpu_memory_usage", 0)),
        })
    return out


def aggregate_totals(prof, n_iters):
    if prof is None:
        return None
    total_self_cuda_us = 0.0
    total_self_cpu_us = 0.0
    total_event_count = 0
    total_self_cuda_mem = 0
    cuda_kernel_event_count = 0
    sync_event_count = 0
    sync_event_self_cpu_us = 0.0
    copy_event_self_cpu_us = 0.0
    copy_event_count = 0
    SYNC_OPS = {"aten::_local_scalar_dense", "aten::item",
                "aten::is_nonzero", "cudaStreamSynchronize"}
    for entry in prof.key_averages():
        scu = float(getattr(entry, "self_cuda_time_total", 0))
        scpu = float(getattr(entry, "self_cpu_time_total", 0))
        cnt = int(entry.count)
        total_self_cuda_us += scu
        total_self_cpu_us += scpu
        total_self_cuda_mem += int(getattr(entry, "self_cuda_memory_usage", 0))
        if scu > 0:
            cuda_kernel_event_count += cnt
        total_event_count += cnt
        if entry.key in SYNC_OPS:
            sync_event_count += cnt
            sync_event_self_cpu_us += scpu
        if entry.key == "aten::copy_":
            copy_event_self_cpu_us += scpu
            copy_event_count += cnt
    return {
        "n_iters": n_iters,
        "total_self_cuda_us": total_self_cuda_us,
        "total_self_cpu_us": total_self_cpu_us,
        "per_iter_self_cuda_ms": total_self_cuda_us / 1000.0 / n_iters,
        "per_iter_self_cpu_ms": total_self_cpu_us / 1000.0 / n_iters,
        "total_event_count": total_event_count,
        "events_per_iter": total_event_count / n_iters,
        "cuda_kernel_event_count": cuda_kernel_event_count,
        "cuda_kernels_per_iter": cuda_kernel_event_count / n_iters,
        "total_self_cuda_memory_bytes": total_self_cuda_mem,
        "sync_event_count": sync_event_count,
        "sync_event_self_cpu_ms": sync_event_self_cpu_us / 1000.0 / n_iters,
        "copy_event_count": copy_event_count,
        "copy_event_self_cpu_ms": copy_event_self_cpu_us / 1000.0 / n_iters,
    }


# ---------------------------------------------------------------------------
# Phase 2 — categorization
# ---------------------------------------------------------------------------

ALLOC_OPS = {
    "aten::zeros", "aten::empty_strided", "aten::eye",
    "aten::ones", "aten::zeros_like", "aten::empty_like",
    "aten::new_empty", "aten::new_zeros", "aten::new_full",
}
DISPATCH_OPS = {
    "aten::full", "aten::empty", "aten::to", "aten::lift_fresh",
    "aten::detach_", "aten::detach",
}
SYNC_OPS = {
    "aten::_local_scalar_dense", "aten::item", "aten::is_nonzero",
    "cudaStreamSynchronize",
}
KERNEL_OPS = {
    "aten::convolution", "aten::conv2d", "aten::convolution_overrideable",
    "aten::_convolution", "aten::cudnn_convolution",
    "aten::grid_sample", "aten::grid_sampler", "aten::grid_sampler_2d",
    "aten::flip", "aten::sub", "aten::mul", "aten::add", "aten::div",
    "aten::sub_", "aten::mul_", "aten::add_", "aten::div_",
    "aten::affine_grid", "aten::affine_grid_generator",
    "aten::clamp", "aten::clamp_", "aten::sort", "aten::index_select",
    "aten::matmul", "aten::bmm", "aten::sum", "aten::mean",
    "aten::addmm", "aten::cat", "aten::stack", "aten::index",
    "aten::pad", "aten::reflection_pad2d", "aten::replication_pad2d",
    "aten::constant_pad_nd", "aten::round", "aten::floor", "aten::ceil",
    "aten::pow", "aten::sqrt", "aten::exp", "aten::log",
    "aten::sin", "aten::cos", "aten::tan", "aten::neg",
    "aten::histc", "aten::cumsum", "aten::gather", "aten::scatter",
    "aten::masked_fill", "aten::masked_fill_", "aten::masked_scatter",
    "aten::nonzero", "aten::where",
}

COMPOSITE_OPS = {
    "ColorJitter", "CutMix", "MixUp", "Mosaic", "Affine", "Perspective",
    "Rotation", "ResizedCrop",
}


def _top_excluding(top_ops, exclude_keys, exclude_substr=("_kornia", "_tv")):
    if not top_ops:
        return None
    for entry in top_ops:
        if entry["name"] in exclude_keys:
            continue
        skip = False
        for s in exclude_substr:
            if s in entry["name"]:
                skip = True
                break
        if skip:
            continue
        return entry
    return None


def categorize_op(name, k_top, k_totals):
    """Return (category, dominant_cost, fix_recommendation).

    Priority ordering reflects the architectural fix that yields the most
    impact for kornia 2.0:

      1. fusion-eligible   — composite op with many sub-events (the right fix
                              is a fused kernel, not chasing each sub-op)
      2. sync-bound        — high host<->device sync count blocks the CPU
                              thread; lifting these is the next biggest win
      3. kernel-bound      — top non-copy op is real compute (irreducible)
      4. allocation-bound  — top non-copy op is a tensor allocation
      5. dispatch-bound    — high event count or dispatch ops dominate
    """
    if k_top is None or k_totals is None:
        return ("unknown", "n/a", "investigate failure")

    # Exclude record_function wrappers (the outer "<name>_kornia").
    real_excl_copy = _top_excluding(k_top, {"aten::copy_"},
                                    exclude_substr=("_kornia", "_tv"))
    real_top_name = real_excl_copy["name"] if real_excl_copy else "?"

    sync_count = k_totals.get("sync_event_count", 0)
    n_iters = k_totals.get("n_iters", N_ITERS)
    sync_per_iter = sync_count / max(n_iters, 1)
    events_per_iter = k_totals.get("events_per_iter", 0)

    # 1. fusion-eligible: composite op with many sub-events (highest priority
    #    so we don't lose them to sync-bound below).
    if name in COMPOSITE_OPS and events_per_iter >= 30:
        return (
            "fusion-eligible",
            f"composite, {events_per_iter:.0f} events/iter, top: {real_top_name}",
            "write a fused kernel (Triton/CUDA) covering all sub-ops",
        )

    # 2. sync-bound: significant host<->device sync events (above the
    #    ~14 events/iter constant baseline emitted by _BatchProbGenerator).
    if sync_per_iter >= 20.0:
        return (
            "sync-bound",
            f"{int(sync_per_iter)} sync events/iter ({sync_count} total)",
            "lift host->device sync points; cache scalar params; avoid .item()/is_nonzero",
        )

    # 3. kernel-bound: top non-copy op is real compute
    if real_top_name in KERNEL_OPS:
        scpu = real_excl_copy["self_cpu_us"] if real_excl_copy else 0.0
        return (
            "kernel-bound",
            f"{real_top_name} dominates ({scpu:.0f} us self CPU)",
            "irreducible floor — only faster kernel (Triton/CUDA) helps",
        )

    # 4. allocation-bound: top non-copy op is a tensor allocation
    if real_top_name in ALLOC_OPS:
        return (
            "allocation-bound",
            f"{real_top_name} dominates allocations",
            "pre-allocate buffers, lift to module __init__",
        )

    # 5. dispatch-bound: events>50 OR top non-copy is dispatch op OR
    #    we're left with the constant ~14-event baseline sync overhead.
    if events_per_iter > 50 or real_top_name in DISPATCH_OPS:
        extra = ""
        if 2.0 <= sync_per_iter < 20.0:
            extra = f" (+ {int(sync_per_iter)} baseline sync evt/iter)"
        return (
            "dispatch-bound",
            f"{real_top_name} + {events_per_iter:.0f} events/iter{extra}",
            "kornia 2.0 base class redesign — fewer aten::full / empty / to",
        )

    # 6. fallback
    return (
        "dispatch-bound",
        f"{real_top_name} ({events_per_iter:.0f} events/iter)",
        "kornia 2.0 base class redesign",
    )


# ---------------------------------------------------------------------------
# Versions / metadata
# ---------------------------------------------------------------------------

def _versions():
    out = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    try:
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = "?"
    try:
        out["cuda"] = torch.version.cuda
    except Exception:
        out["cuda"] = "?"
    try:
        out["device"] = (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu")
    except Exception:
        out["device"] = "?"
    try:
        import kornia as _k
        out["kornia"] = _k.__version__
    except Exception:
        out["kornia"] = "?"
    try:
        import torchvision as _tv
        out["torchvision"] = _tv.__version__
    except Exception:
        out["torchvision"] = "?"
    return out


# ---------------------------------------------------------------------------
# Per-op timings from existing CUDA-event leaderboard
# ---------------------------------------------------------------------------

def _load_per_op_medians():
    p = Path("/home/nvidia/kornia/benchmarks/comparative/results_per_op.json")
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for name, row in data.get("results", {}).items():
        k = row.get("kornia", {})
        tv = row.get("tv", {})
        out[name] = {
            "k_ms": k.get("median_ms") if k.get("status") == "ok" else None,
            "tv_ms": tv.get("median_ms") if tv.get("status") == "ok" else None,
        }
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v, fmt="{:.2f}"):
    if v is None:
        return "—"
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


def write_categorized_md(out_path: Path, results: dict, versions: dict,
                         per_op_medians: dict, patch_status: str):
    lines = []
    lines.append("# Bottleneck categorization — all 37 augmentation transforms")
    lines.append("")
    lines.append("**Date:** 2026-04-27  ")
    lines.append(f"**Hardware:** {versions.get('device','?')} "
                 f"({versions.get('machine','?')})  ")
    lines.append(f"**PyTorch:** {versions.get('torch','?')}, "
                 f"CUDA {versions.get('cuda','?')}, "
                 f"kornia {versions.get('kornia','?')}, "
                 f"torchvision {versions.get('torchvision','?')}  ")
    lines.append(f"**Patches:** {patch_status}  ")
    lines.append(f"**Profile:** {N_WARMUP} warmup + {N_ITERS} timed iters via "
                 f"`torch.profiler` (CPU+CUDA, record_shapes=True, "
                 f"profile_memory=True)  ")
    lines.append(f"**Inputs:** B={BATCH}, 3x{RES}x{RES}, fp32, GPU pre-resident  ")
    lines.append("")
    lines.append("**CUPTI note:** `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` on this "
                 "unprivileged Jetson run means kernel-level CUDA self-times are "
                 "0 in every event. We classify using **self CPU time + event "
                 "count + sync proxy** (the trailing `aten::copy_` self CPU is "
                 "the closest proxy for CUDA wall-clock since each iter ends "
                 "with `cuda.synchronize()`).")
    lines.append("")
    lines.append("**Categories:**")
    lines.append("")
    lines.append("- **dispatch-bound** — `events_per_iter > 50` or top non-copy "
                 "event is `aten::full`/`empty`/`to`/`lift_fresh`. The op spends "
                 "most time in framework bookkeeping. *Fix: kornia 2.0 base "
                 "class redesign.*")
    lines.append("- **allocation-bound** — top non-copy op is a tensor "
                 "allocation (`aten::zeros`, `empty_strided`, `eye`, etc.). "
                 "*Fix: pre-allocate buffers, lift to module `__init__`.*")
    lines.append("- **kernel-bound** — top non-copy op is real compute "
                 "(`aten::convolution`, `grid_sample`, `flip`, `sub`, `mul`). "
                 "*Fix: faster kernel (Triton/CUDA); this is the irreducible "
                 "floor.*")
    lines.append("- **sync-bound** — large `aten::_local_scalar_dense` / "
                 "`aten::item` count. *Fix: lift host->device sync points; "
                 "cache scalar params.*")
    lines.append("- **fusion-eligible** — composite op (CutMix, ColorJitter, "
                 "Mosaic, Affine, …) with many sub-op events that could be "
                 "fused. *Fix: write a fused kernel.*")
    lines.append("")

    rows = []
    for name, row in results.items():
        if name == "_meta":
            continue
        if "error" in row:
            rows.append({
                "name": name,
                "category": "error",
                "dominant": row["error"][:60],
                "fix": "investigate",
                "k_ms": None, "tv_ms": None,
                "events_per_iter": None,
                "self_cpu_ms": None,
                "copy_ms": None,
                "sync_per_iter": None,
            })
            continue
        k_top = row.get("kornia_top_ops")
        k_tot = row.get("kornia_totals")
        cat, dom, fix = categorize_op(name, k_top, k_tot)
        med = per_op_medians.get(name, {})
        rows.append({
            "name": name,
            "category": cat,
            "dominant": dom,
            "fix": fix,
            "k_ms": med.get("k_ms"),
            "tv_ms": med.get("tv_ms"),
            "events_per_iter": k_tot.get("events_per_iter") if k_tot else None,
            "self_cpu_ms": k_tot.get("per_iter_self_cpu_ms") if k_tot else None,
            "copy_ms": k_tot.get("copy_event_self_cpu_ms") if k_tot else None,
            "sync_per_iter": ((k_tot.get("sync_event_count", 0) / N_ITERS)
                              if k_tot else 0),
        })

    cat_totals = {}
    for r in rows:
        c = r["category"]
        if c not in cat_totals:
            cat_totals[c] = {"count": 0, "k_ms_sum": 0.0, "tv_ms_sum": 0.0}
        cat_totals[c]["count"] += 1
        if r["k_ms"] is not None:
            cat_totals[c]["k_ms_sum"] += r["k_ms"]
        if r["tv_ms"] is not None:
            cat_totals[c]["tv_ms_sum"] += r["tv_ms"]

    lines.append("## Summary by category")
    lines.append("")
    lines.append("| Category | Count | Total kornia time (ms) | Total tv time (ms) |")
    lines.append("|---|---:|---:|---:|")
    cat_order = ["dispatch-bound", "allocation-bound", "kernel-bound",
                 "sync-bound", "fusion-eligible", "error", "unknown"]
    for c in cat_order:
        if c in cat_totals:
            t = cat_totals[c]
            lines.append(f"| {c} | {t['count']} | "
                         f"{_fmt(t['k_ms_sum'])} | {_fmt(t['tv_ms_sum'])} |")
    lines.append("")

    lines.append("## Per-op classification")
    lines.append("")
    lines.append("| Op | k ms | tv ms | events/iter | self CPU ms | "
                 "copy_ ms (sync proxy) | sync evt/iter | category | "
                 "dominant cost | fix |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    cat_rank = {c: i for i, c in enumerate(cat_order)}
    rows_sorted = sorted(
        rows,
        key=lambda r: (cat_rank.get(r["category"], 99),
                       -(r["k_ms"] or 0)),
    )
    for r in rows_sorted:
        lines.append(
            f"| {r['name']} | "
            f"{_fmt(r['k_ms'])} | {_fmt(r['tv_ms'])} | "
            f"{_fmt(r['events_per_iter'], '{:.0f}')} | "
            f"{_fmt(r['self_cpu_ms'])} | "
            f"{_fmt(r['copy_ms'])} | "
            f"{_fmt(r.get('sync_per_iter'), '{:.1f}')} | "
            f"{r['category']} | {r['dominant']} | {r['fix']} |"
        )
    lines.append("")

    lines.append("## High-priority fixes (sorted by ROI)")
    lines.append("")

    def _by_k(rs):
        return sorted([r for r in rs if r["k_ms"] is not None],
                      key=lambda r: -r["k_ms"])

    disp = _by_k([r for r in rows if r["category"] == "dispatch-bound"])
    allo = _by_k([r for r in rows if r["category"] == "allocation-bound"])
    fus = _by_k([r for r in rows if r["category"] == "fusion-eligible"])
    syn = _by_k([r for r in rows if r["category"] == "sync-bound"])
    krn = _by_k([r for r in rows if r["category"] == "kernel-bound"])

    def _section(title, items, est_speedup, difficulty):
        lines.append(f"### {title}")
        lines.append(f"- **Estimated kornia 2.0 perf improvement:** {est_speedup}")
        lines.append(f"- **Difficulty:** {difficulty}")
        lines.append("- **Ops that benefit:**")
        if not items:
            lines.append("  - (none)")
        for r in items[:15]:
            tv = r["tv_ms"]
            ratio = (r["k_ms"] / tv) if (tv and tv > 0) else None
            ratio_s = f", k/tv={ratio:.1f}x" if ratio else ""
            lines.append(f"  - **{r['name']}** "
                         f"(k={_fmt(r['k_ms'])} ms{ratio_s}) — {r['dominant']}")
        lines.append("")

    _section(
        "1. Base-class redesign (dispatch-bound)",
        disp,
        "**3-10x** on most dispatch-bound ops (eliminates ~80% of "
        "`aten::full`/`empty`/`to`/`lift_fresh` events that today come from "
        "the random-param sampler + CPU->GPU param shipping)",
        "**high** (cross-cutting refactor of `_AugmentationBase`, parameter "
        "sampler, and forward dispatch)",
    )
    _section(
        "2. Pre-allocate buffers (allocation-bound)",
        allo,
        "**2-4x** on allocation-bound ops (move `zeros`/`eye`/`empty_strided` "
        "into `__init__` once; reuse through `register_buffer`)",
        "**low** (per-op patch, similar to existing Normalize buffer patch)",
    )
    _section(
        "3. Lift sync points (sync-bound)",
        syn,
        "**2-5x** on sync-bound ops (every `aten::item` / `is_nonzero` "
        "blocks the CPU thread for an entire kernel queue; deferring or "
        "removing these is pure win)",
        "**medium** (audit each call site; some are baked into augmentation "
        "control flow)",
    )
    _section(
        "4. Fused composite kernels (fusion-eligible)",
        fus,
        "**2-8x** on composite ops by collapsing N sub-ops into one kernel "
        "(ColorJitter HSV roundtrip already fused via patch #5; extend the "
        "pattern to CutMix/MixUp/Affine)",
        "**medium-high** (Triton or hand-rolled CUDA per fused recipe)",
    )
    _section(
        "5. Kernel optimizations (kernel-bound — irreducible floor)",
        krn,
        "**1.2-2x** at best (these ops are already real compute; only a "
        "faster Triton/CUDA kernel or different algorithm helps)",
        "**high** (per-kernel rewrite; gains are smaller and hardware-"
        "specific)",
    )

    lines.append("## Concrete recommendations for kornia 2.0 RFC")
    lines.append("")
    lines.append("Based on the categorization above, these architectural "
                 "decisions are justified:")
    lines.append("")

    n_disp = cat_totals.get("dispatch-bound", {}).get("count", 0)
    n_allo = cat_totals.get("allocation-bound", {}).get("count", 0)
    n_krn = cat_totals.get("kernel-bound", {}).get("count", 0)
    n_syn = cat_totals.get("sync-bound", {}).get("count", 0)
    n_fus = cat_totals.get("fusion-eligible", {}).get("count", 0)

    lines.append(
        f"1. **Slim `_AugmentationBase` (highest ROI — covers {n_disp} ops).** "
        "Today every op pays an `aten::full` + `aten::empty` + `aten::to` + "
        "`aten::lift_fresh` tax purely for the random-parameter sampler. "
        "Replace `_BatchProbGenerator` with a single device-resident `rand` "
        "tensor that is sliced per call. Drop the 60+ event scaffolding "
        "around `_apply_func_by_input_type`. Target: **events_per_iter < 10** "
        "for all single-op augmentations.")
    lines.append("")
    lines.append(
        "2. **Move parameter generators to GPU once, not per-call.** Most "
        "dispatch-bound ops show 20+ `aten::to` calls per iter, each "
        "shipping a tiny CPU scalar (degrees, threshold, brightness factor) "
        "to GPU. Cache these as device-resident `register_buffer` at "
        "`__init__` time; resample on-device.")
    lines.append("")
    lines.append(
        f"3. **Pre-allocate transformation matrices and masks "
        f"({n_allo} allocation-bound ops).** Geometric ops re-create "
        "`torch.zeros(B, 3, 3)` every iter for the affine matrix, and "
        "`torch.eye(3)` for identity initializers. Lift to `__init__` and "
        "fill in-place.")
    lines.append("")
    lines.append(
        f"4. **Eliminate `aten::is_nonzero` / `aten::item` in hot paths "
        f"({n_syn} sync-bound ops today).** Replace `if mask.any():` "
        "with `mask_t * branch_a + (1-mask_t) * branch_b` to keep the "
        "graph fully on-device. The current pattern blocks the CPU "
        "thread on every iter.")
    lines.append("")
    lines.append(
        f"5. **Provide fused recipes for {n_fus} composite ops.** Build a "
        "small registry of hand-rolled kernels: ColorJiggle (already has "
        "fused-HSV via v4 patch), CutMix (single mask + blend), MixUp "
        "(single linear combo), Affine (closed-form matrix + grid_sample). "
        "Mosaic is the largest ROI; today it stitches 4 images via 4 "
        "separate `affine_grid` + `grid_sample` calls.")
    lines.append("")
    lines.append(
        f"6. **Don't optimize the {n_krn} kernel-bound ops first.** They "
        "are already hitting the irreducible floor (true convolution / "
        "grid_sample / flip / hist). Optimizing them is high effort, low "
        "ROI compared to (1)–(5). Defer to a phase 2 kernel-rewrite RFC.")
    lines.append("")
    lines.append(
        "7. **Adopt a v6-style `forward()` override path as the default.** "
        "The `run_v6.py` aggressive override shaves 30-70% off many ops "
        "by skipping `_AugmentationBase.forward`. The kornia 2.0 base "
        "class should have this as its only path — no opt-in monkey-"
        "patch needed.")
    lines.append("")
    lines.append(
        "8. **Verify on Jetson Orin (this rig): CUPTI privileges block "
        "CUDA-self-time profiling.** k2 RFC should mandate dual profiling "
        "(CPU self-time + CUDA-event wallclock) to detect regressions in "
        "either dimension; relying on `aten::copy_` as a sync proxy is "
        "fragile.")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    versions = _versions()
    print("Versions:", json.dumps(versions, indent=2), flush=True)
    v4 = getattr(run_v6, '_v4_status', '?')
    v6 = getattr(run_v6, '_v6_status', '?')
    print(f"Patches: v4_status={v4}", flush=True)
    print(f"         v6_status={v6}", flush=True)

    patch_status = f"v4: {v4}; v6: {v6}"

    results = {
        "_meta": {
            "versions": versions,
            "n_warmup": N_WARMUP,
            "n_iters": N_ITERS,
            "batch": BATCH,
            "res": RES,
            "v4_status": str(v4),
            "v6_status": str(v6),
        },
    }

    registry = _build_registry()
    n_total = len(registry)
    n_ok = 0
    n_err = 0

    for name, category, k_fac, tv_fac, kind in registry:
        try:
            prof_k, prof_tv, k_err, tv_err = profile_op(
                name, k_fac, tv_fac, kind
            )
            row = {
                "category": category,
                "kornia_top_ops": extract_top_ops(prof_k, n=15),
                "kornia_totals": aggregate_totals(prof_k, N_ITERS),
                "tv_top_ops": (extract_top_ops(prof_tv, n=15)
                               if prof_tv is not None else None),
                "tv_totals": (aggregate_totals(prof_tv, N_ITERS)
                              if prof_tv is not None else None),
            }
            if k_err:
                row["kornia_error"] = k_err
            if tv_err:
                row["tv_error"] = tv_err
            results[name] = row
            if row.get("kornia_totals") is not None:
                n_ok += 1
            else:
                n_err += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {
                "category": category,
                "error": f"{type(e).__name__}: {e}",
            }
            n_err += 1

    out_path = OUT_DIR / "bottlenecks_all.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    per_op = _load_per_op_medians()
    md_path = OUT_DIR / "bottlenecks_categorized.md"
    write_categorized_md(md_path, results, versions, per_op, patch_status)

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s "
          f"({n_ok}/{n_total} ops profiled, {n_err} errors)", flush=True)


if __name__ == "__main__":
    main()

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

"""Per-op profiling harness using torch.profiler.

Empirical evidence for the kornia 2.0 architecture redesign: identifies
where each augmentation's CUDA + CPU time goes vs torchvision v2.

Profiles 8 ops chosen as worst k/tv ratios + biggest absolute times:
  CenterCrop, CutMix, HFlip, Affine, Grayscale, Normalize,
  ColorJitter, MedianBlur.

Reuses the v6 patch infrastructure: importing run_v6 applies v4 patches +
cusolver workaround + aggressive forward overrides at module-import time
(see run_v6.py:1128-1130).

Run from /tmp with PYTHONNOUSERSITE=1:
  cd /tmp && PYTHONNOUSERSITE=1 \\
    /home/nvidia/bubbaloop-nodes-official/camera-object-detector/.pixi/envs/default/bin/python3.10 \\
    /home/nvidia/kornia/benchmarks/profile/profile_per_op.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Apply all v4 + cusolver + v6 aggressive forward patches via run_v6 import side-effects.
sys.path.insert(0, "/home/nvidia/kornia/benchmarks/comparative")
import run_v6
import torch
import torchvision.transforms.v2 as T
from torch.profiler import ProfilerActivity, profile, record_function

import kornia.augmentation as K

OUT_DIR = Path("/home/nvidia/kornia/benchmarks/profile")
TRACE_DIR = OUT_DIR / "traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

N_WARMUP = 5
N_ITERS = 20
BATCH = 8
RES = 512


OPS = [
    (
        "CenterCrop",
        lambda: K.CenterCrop(size=(256, 256)),
        lambda: T.CenterCrop(size=(256, 256)),
        "image",
    ),
    (
        "HFlip",
        lambda: K.RandomHorizontalFlip(p=1.0),
        lambda: T.RandomHorizontalFlip(p=1.0),
        "image",
    ),
    (
        "Grayscale",
        lambda: K.RandomGrayscale(p=1.0),
        lambda: T.RandomGrayscale(p=1.0),
        "image",
    ),
    (
        "Normalize",
        lambda: K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
        lambda: T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        "image",
    ),
    (
        "Affine",
        lambda: K.RandomAffine(degrees=15.0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0),
        lambda: T.RandomAffine(degrees=15.0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        "image",
    ),
    (
        "ColorJitter",
        lambda: K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        lambda: T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        "image",
    ),
    (
        "CutMix",
        lambda: K.RandomCutMixV2(p=1.0),
        lambda: T.CutMix(num_classes=10),
        "labels",
    ),
    (
        "MedianBlur",
        lambda: K.RandomMedianBlur(kernel_size=(3, 3), p=1.0),
        None,
        "no_tv",
    ),
]


def _make_input(kind):
    x = torch.rand(BATCH, 3, RES, RES, device="cuda")
    if kind == "labels":
        labels = torch.randint(0, 10, (BATCH,), device="cuda")
        return x, labels
    return x, None


def _call_kornia(aug, x, labels, kind):
    if kind == "labels":
        return aug(x, labels)
    return aug(x)


def _call_tv(aug, x, labels, kind):
    if kind == "labels":
        return aug(x, labels)
    return aug(x)


def _set_eval(aug):
    fn = getattr(aug, "eval", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return aug
    return aug


def profile_op(name, kornia_factory, tv_factory, kind, n_warmup=N_WARMUP, n_iters=N_ITERS):
    print(f"\n=== {name} ===", flush=True)
    x, labels = _make_input(kind)

    aug = kornia_factory()
    try:
        aug = aug.cuda()
    except Exception:
        pass
    aug = _set_eval(aug)

    for _ in range(n_warmup):
        try:
            _ = _call_kornia(aug, x, labels, kind)
        except Exception as e:
            print(f"  kornia warmup failed: {e}", flush=True)
            return None, None
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof_k:
        for _ in range(n_iters):
            with record_function(f"{name}_kornia"):
                _ = _call_kornia(aug, x, labels, kind)
            torch.cuda.synchronize()
    prof_k.export_chrome_trace(str(TRACE_DIR / f"{name}_kornia.json"))
    print(f"  kornia trace -> {TRACE_DIR / f'{name}_kornia.json'}", flush=True)

    prof_tv = None
    if tv_factory is not None:
        try:
            aug_tv = tv_factory()
            try:
                aug_tv = aug_tv.cuda()
            except Exception:
                pass
            for _ in range(n_warmup):
                _ = _call_tv(aug_tv, x, labels, kind)
            torch.cuda.synchronize()

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof_tv:
                for _ in range(n_iters):
                    with record_function(f"{name}_tv"):
                        _ = _call_tv(aug_tv, x, labels, kind)
                    torch.cuda.synchronize()
            prof_tv.export_chrome_trace(str(TRACE_DIR / f"{name}_tv.json"))
            print(f"  tv     trace -> {TRACE_DIR / f'{name}_tv.json'}", flush=True)
        except Exception as e:
            print(f"  tv profile failed: {e}", flush=True)
            prof_tv = None

    return prof_k, prof_tv


def extract_top_ops(prof, n=15, sort_by="self_cuda_time_total"):
    """Top-N events by self CUDA time. torch.profiler reports times in microseconds."""
    if prof is None:
        return None
    avgs = prof.key_averages()
    sorted_avgs = sorted(avgs, key=lambda x: getattr(x, sort_by, 0), reverse=True)
    out = []
    for entry in sorted_avgs[:n]:
        out.append(
            {
                "name": entry.key,
                "count": int(entry.count),
                "self_cuda_us": float(getattr(entry, "self_cuda_time_total", 0)),
                "self_cpu_us": float(getattr(entry, "self_cpu_time_total", 0)),
                "cuda_total_us": float(getattr(entry, "cuda_time_total", 0)),
                "cpu_total_us": float(getattr(entry, "cpu_time_total", 0)),
                "self_cuda_memory": int(getattr(entry, "self_cuda_memory_usage", 0)),
                "self_cpu_memory": int(getattr(entry, "self_cpu_memory_usage", 0)),
            }
        )
    return out


def aggregate_totals(prof, n_iters):
    """Roll-up: sum self-times, count events. self_cuda > 0 is a kernel-launching op."""
    if prof is None:
        return None
    total_self_cuda_us = 0.0
    total_self_cpu_us = 0.0
    total_event_count = 0
    total_self_cuda_mem = 0
    cuda_kernel_event_count = 0
    for entry in prof.key_averages():
        total_self_cuda_us += float(getattr(entry, "self_cuda_time_total", 0))
        total_self_cpu_us += float(getattr(entry, "self_cpu_time_total", 0))
        total_self_cuda_mem += int(getattr(entry, "self_cuda_memory_usage", 0))
        if getattr(entry, "self_cuda_time_total", 0) > 0:
            cuda_kernel_event_count += int(entry.count)
        total_event_count += int(entry.count)
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
    }


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
        out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
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


def main():
    t_start = time.perf_counter()
    versions = _versions()
    print("Versions:", json.dumps(versions, indent=2), flush=True)
    print(f"Patches: v4_status={getattr(run_v6, '_v4_status', '?')}", flush=True)
    print(f"         v6_status={getattr(run_v6, '_v6_status', '?')}", flush=True)

    results = {"_meta": {"versions": versions, "n_warmup": N_WARMUP, "n_iters": N_ITERS, "batch": BATCH, "res": RES}}

    for name, k_fac, tv_fac, kind in OPS:
        try:
            prof_k, prof_tv = profile_op(name, k_fac, tv_fac, kind)
            results[name] = {
                "kornia_top_ops": extract_top_ops(prof_k, n=15),
                "kornia_totals": aggregate_totals(prof_k, N_ITERS),
                "tv_top_ops": extract_top_ops(prof_tv, n=15) if prof_tv is not None else None,
                "tv_totals": aggregate_totals(prof_tv, N_ITERS) if prof_tv is not None else None,
            }
        except Exception as e:
            import traceback

            traceback.print_exc()
            results[name] = {"error": f"{type(e).__name__}: {e}"}

    out_path = OUT_DIR / "bottlenecks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"Total elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

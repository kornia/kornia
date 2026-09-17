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
"""Regenerate PR #4638 figures from measurements archived in Git history.

Run from any directory: python benchmarks/feature/plot_sift_summary.py
Requires matplotlib and the archive commit (git fetch origin <commit> in a shallow clone).
No benchmark runs or local JSON files are needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from statistics import mean

ARCHIVE = "efb04dbf9c85e4cf71625cc2467bd5243b0c803c"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "benchmarks/feature"


def read_rows(path: str) -> list[dict]:
    """Read measurement rows without restoring raw files into the checkout."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to read the archived measurements")
    document = subprocess.check_output([git, "show", f"{ARCHIVE}:benchmarks/{path}"], cwd=ROOT, text=True)
    return json.loads(document)["results"]


def timing(pattern: str, repeated: bool = True) -> float:
    """Average six image medians, and both reversed-order runs when available, in ms."""
    rows = [row for run in ((1, 2) if repeated else (1,)) for row in read_rows(pattern.format(run=run))]
    timings = [row for row in rows if row["op"] == "SIFTFeatureScaleSpace"]
    assert len(timings) == (12 if repeated else 6)
    assert all((row["batch"], row["height"], row["width"], row["dtype"]) == (1, 640, 800, "float32") for row in timings)
    return mean(row["median_us"] for row in timings) / 1000


def main() -> None:
    """Render device speedups and CUDA matching quality from the archived source data."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    groups = [
        (
            "Apple M1 · memory pass",
            "pyramid · bc790974 → 7547be2c · one run",
            [
                (
                    "CPU · 1 thread",
                    "feature/sift_memory_results/before-cpu.json",
                    "feature/sift_memory_results/after-cpu.json",
                    False,
                ),
                (
                    "MPS",
                    "feature/sift_memory_results/before-mps.json",
                    "feature/sift_memory_results/after-mps.json",
                    False,
                ),
            ],
        ),
        (
            "Intel i7-14700K / RTX 4090 · earlier passes",
            "pyramid · 3172b5f1 → 7547be2c · two runs",
            [
                (
                    label,
                    f"feature/sift_device_results/before-{device}-pyramid-graf-r{{run}}.json",
                    f"feature/sift_device_results/head-{device}-pyramid-graf-r{{run}}.json",
                    True,
                )
                for label, device in [
                    ("CPU · 1 thread", "cpu-t1"),
                    ("CPU · 14 threads", "cpu-t14"),
                    ("CUDA", "cuda-t1"),
                ]
            ],
        ),
        (
            "Intel i7-14700K / RTX 4090 · CUDA pass",
            "pyramid · 7547be2c → 40bcf122 · two runs",
            [
                (
                    label,
                    f"feature/sift_device_results/base-final-{device}-r{{run}}.json",
                    f"feature/sift_device_results/opt-{device}-r{{run}}.json",
                    True,
                )
                for label, device in [
                    ("CPU · 1 thread", "cpu-t1"),
                    ("CPU · 14 threads", "cpu-t14"),
                    ("CUDA", "cuda-t1"),
                ]
            ],
        ),
        (
            "Intel i7-14700K / RTX 4090 · shared interpolation",
            "default patch · 40bcf122 → fa3d732c · two runs",
            [
                (
                    label,
                    f"geometry/quad_interp_results/sift-base-{device}-r{{run}}.json",
                    f"geometry/quad_interp_results/sift-opt-{device}-r{{run}}.json",
                    True,
                )
                for label, device in [
                    ("CPU · 1 thread", "cpu-t1"),
                    ("CPU · 14 threads", "cpu-t14"),
                    ("CUDA", "cuda-t1"),
                ]
            ],
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.76, bottom=0.24, wspace=0.48, hspace=0.85)
    for ax, (title, subtitle, comparisons) in zip(axes.flat, groups):
        before = [timing(old, repeat) for _, old, _, repeat in comparisons]
        after = [timing(new, repeat) for _, _, new, repeat in comparisons]
        ratios = [old / new for old, new in zip(before, after)]
        positions = np.arange(len(comparisons))
        ax.barh(positions, ratios, color=["#197a87" if ratio >= 1 else "#bb573c" for ratio in ratios], height=0.52)
        ax.axvline(1, color="#414952", linestyle="--", linewidth=1)
        ax.set_yticks(positions, [row[0] for row in comparisons])
        ax.invert_yaxis()
        ax.set_xlim(0, 3.6)
        ax.set_xticks([0, 1, 2, 3], ["0\u00d7", "1\u00d7", "2\u00d7", "3\u00d7"])
        ax.set_xlabel("Speedup = before / after · higher is faster", fontsize=10)
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=11, pad=13, linespacing=1.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for y, old, new, ratio in zip(positions, before, after, ratios):
            ax.text(max(ratio, 1) + 0.06, y, f"{ratio:.3f}\u00d7\n{old:.2f} → {new:.2f} ms", va="center", fontsize=9)
            print(f"{title} | {comparisons[y][0]} | {old:.2f} | {new:.2f} | {ratio:.3f}\u00d7")
    fig.text(0.04, 0.96, "Where SIFT got faster", fontsize=24, weight="bold")
    fig.text(0.04, 0.91, "Measured optimization stages within PR #4638 · each panel has its own baseline", fontsize=13)
    fig.text(
        0.04,
        0.855,
        "Graf images 1-6 · 640\u00d7800 · batch 1 · 4,096 features · FP32 RootSIFT · eager full forward",
        fontsize=11,
    )
    fig.text(
        0.04,
        0.12,
        "Mean of six warmed per-image medians per run; two-run panels reverse revision order.\n"
        "PyTorch 2.14.0 (Apple M1) / 2.14.0+cu130 (Intel / NVIDIA). CPU thread counts shown; CUDA host uses one.\n"
        "Below 1\u00d7 means slower. Separate stages must not be multiplied into an unmeasured cumulative speedup.",
        fontsize=10,
        linespacing=1.6,
        va="top",
    )
    fig.savefig(OUTPUT / "sift_speedups.png", dpi=150, facecolor="white")
    plt.close(fig)

    pairs = [f"1-{i}" for i in range(2, 7)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.73, bottom=0.33, wspace=0.24)
    for offset, label, path, color in [
        (-0.19, "Default patch", "geometry/quad_interp_results/sift-opt-cuda-t1-r1.json", "#64748b"),
        (0.19, "Pyramid", "feature/sift_device_results/opt-cuda-t1-r1.json", "#197a87"),
    ]:
        quality = {row["pair"]: row for row in read_rows(path) if row["op"] == "matching_quality"}
        assert set(quality) == set(pairs)
        for ax, field in zip(axes, ["correct_matches", "ransac_corner_error_px"]):
            values = [quality[pair][field] for pair in pairs]
            x = np.arange(5) + offset
            ax.bar(x, values, width=0.36, color=color, label=label)
            for pos, value in zip(x, values):
                ax.text(
                    pos,
                    value * 1.16,
                    f"{value:.0f}" if field == "correct_matches" else f"{value:.2f}",
                    ha="center",
                    fontsize=9,
                )
    for ax in axes:
        ax.set_xticks(range(5), pairs)
        ax.set_xlabel("Graf image pair")
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylim(0.7, 6000)
    axes[0].set_title("Correct matches · higher is better")
    axes[0].set_ylabel("Count (log scale)")
    axes[1].set_ylim(0.7, 2500)
    axes[1].set_title("Homography corner error · lower is better")
    axes[1].set_ylabel("Mean L1 error [px] (log scale)")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.97, 0.87), ncol=2, frameon=False
    )
    fig.text(0.045, 0.94, "CUDA: pyramid SIFT trades runtime for more Graf matches", fontsize=19, weight="bold")
    fig.text(0.045, 0.86, "RTX 4090 · FP32 RootSIFT · 4,096 features per image", fontsize=11)
    fig.text(
        0.045,
        0.20,
        "Pyramid recovers pair 1-5; both backends still fail pair 1-6. Graf informed development and is not held out.\n"
        "SNN ratio 0.8 · correct-match threshold 3 px · RANSAC threshold 2 px, seed 3407.\n"
        "Pyramid 40bcf122; patch fa3d732c. Matching/RANSAC excluded from extraction timing.",
        fontsize=10,
        linespacing=1.6,
        va="top",
    )
    fig.savefig(OUTPUT / "sift_quality.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

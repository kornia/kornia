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
"""Render the Oxford graf SIFT runtime comparison as native PNG and SVG charts.

Usage::

    python benchmarks/feature/plot_sift_runtime.py \\
        --inputs /tmp/sift-runtime-*.json --output /tmp/sift-runtime

Inputs use ``common.save_json``'s metadata/results envelope. Each result records
``op``, ``series``, ``device``, ``batch``, ``median_us`` and ``iqr_us``; timings
are raw batch latencies, normalized here to milliseconds per image. Feature
counts and CUDA memory measurements remain in the source JSON. Only CPU batch
one and CUDA batches one, four and eight belong in this fixed comparison.

The historical versions use their native presets, so quality differs. This is
a runtime comparison, not an assertion of equivalent detections. OpenCV is a
CPU-only baseline, including its dashed reference across the CUDA categories.
Error bars represent median +/- half the IQR, not confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SERIES = ("0.8.2", "0.8.3", "current", "current + compile", "OpenCV")
COLORS = ("#2549e8", "#7972c5", "#ffae00", "#a71920", "#228b32")
GROUPS = (("cpu", 1), ("cuda", 1), ("cuda", 4), ("cuda", 8))
GROUP_LABELS = ("CPU (BS=1)", "CUDA BS=1", "CUDA BS=4", "CUDA BS=8")


def normalize_timing(row: dict[str, Any]) -> tuple[float, float]:
    """Validate raw batch timings and return median and IQR in ms/image."""
    batch = row.get("batch")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError(f"batch must be a positive integer, got {batch!r}")
    values = []
    for name in ("median_us", "iqr_us"):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite numeric data, got {value!r}")
        if value < 0 or (name == "median_us" and value == 0):
            raise ValueError(f"{name} must be {'positive' if name == 'median_us' else 'nonnegative'}, got {value}")
        values.append(value / batch / 1000.0)
    return values[0], values[1]


def load_rows(paths: list[Path]) -> dict[tuple[str, str, int], tuple[float, float]]:
    """Load measurements, rejecting ambiguous duplicates and incompatible rows."""
    timings: dict[tuple[str, str, int], tuple[float, float]] = {}
    for path in paths:
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
            raise ValueError(f"{path}: expected a metadata/results JSON object")
        if not isinstance(document.get("results"), list):
            raise ValueError(f"{path}: results must be a list")
        if document["metadata"].get("num_features") != 4096:
            raise ValueError(f"{path}: this chart requires metadata num_features=4096")
        for index, row in enumerate(document["results"]):
            context = f"{path}: results[{index}]"
            if not isinstance(row, dict):
                raise ValueError(f"{context}: expected an object")
            if (row.get("height"), row.get("width")) != (640, 800):
                raise ValueError(f"{context}: this chart requires height=640, width=800")
            series, device, batch = row.get("series"), row.get("device"), row.get("batch")
            if series not in SERIES:
                raise ValueError(f"{context}: unsupported series {series!r}")
            expected_op = "OpenCV SIFT" if series == "OpenCV" else "SIFTFeatureScaleSpace"
            if row.get("op") != expected_op:
                raise ValueError(f"{context}: expected op={expected_op!r}")
            try:
                timing = normalize_timing(row)
            except ValueError as error:
                raise ValueError(f"{context}: {error}") from error
            if (device, batch) not in GROUPS:
                raise ValueError(f"{context}: unsupported device/batch combination {(device, batch)!r}")
            if series == "OpenCV" and (device, batch) != ("cpu", 1):
                raise ValueError(f"{context}: OpenCV must be CPU batch one")
            key = (series, device, batch)
            if key in timings:
                raise ValueError(f"{context}: duplicate measurement {key!r}")
            timings[key] = timing
    if not timings:
        raise ValueError("No SIFT runtime measurements found")
    return timings


def describe_environment(paths: list[Path], cpu_label: str | None = None) -> str:
    """Build a hardware subtitle and reject mixed GPU or PyTorch environments."""
    metadata = [json.loads(path.read_text())["metadata"] for path in paths]
    for key in ("python", "input_sha256", "torch_num_threads", "opencv_num_threads"):
        values = {str(meta.get(key)) for meta in metadata}
        if len(values) != 1 or "None" in values:
            raise ValueError(f"Input metadata must agree on {key}")
    labels = []
    for key in ("cuda_device", "cpu_model", "torch"):
        values = {str(meta[key]) for meta in metadata if meta.get(key)}
        if len(values) > 1:
            raise ValueError(f"Cannot combine different {key} environments: {sorted(values)}")
        value = next(iter(values), None)
        if key == "cpu_model":
            labels.append(cpu_label or value or "CPU model not recorded")
        elif key == "torch":
            if value is None:
                raise ValueError("Input metadata must record the PyTorch version")
            labels.append(f"PyTorch {value.split('+')[0].split('.dev')[0]}")
        elif value:
            labels.append(value)
    return " · ".join(labels)


def describe_outputs(paths: list[Path]) -> str:
    """Disclose returned work, without calling historical LAF slots valid detections."""
    counts: dict[str, set[int]] = {"Kornia": set(), "OpenCV": set()}
    for path in paths:
        for row in json.loads(path.read_text())["results"]:
            label = "OpenCV" if row["series"] == "OpenCV" else "Kornia"
            values = row.get("features_per_image")
            if not isinstance(values, list) or len(values) != row["batch"]:
                raise ValueError(f"{path}: expected one output count per image")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise ValueError(f"{path}: output counts must be nonnegative integers")
            counts[label].update(values)
    labels = []
    for label, values in counts.items():
        if values:
            count = str(min(values)) if len(values) == 1 else f"{min(values)}\N{EN DASH}{max(values)}"
            unit = "keypoints" if label == "OpenCV" else "nonzero LAF/descriptor slots"
            labels.append(f"{label}: {count} {unit}/image")
    return "; ".join(labels) + ". RootSIFT enabled; native detection/filtering differ."


def plot_runtime(
    timings: dict[tuple[str, str, int], tuple[float, float]],
    output: Path,
    environment: str = "",
    output_note: str = "",
) -> None:
    """Write a grouped logarithmic chart with explicit missing measurements."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    low = min(median for median, _ in timings.values()) / 2.2
    high = max(median + iqr / 2 for median, iqr in timings.values()) * 2.4
    clipped = any(median - iqr / 2 < low for median, iqr in timings.values())
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(12, 7.5))
        fig.subplots_adjust(left=0.085, right=0.975, bottom=0.30, top=0.77)
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        ax.set_xlim(-0.57, 3.57)
        ax.set_axisbelow(True)
        ax.grid(axis="y", which="major", color="#b9bec5", linewidth=0.8)
        ax.grid(axis="y", which="minor", color="#d9dde1", linewidth=0.6, linestyle=":")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        for boundary in (0.5, 1.5, 2.5):
            ax.axvline(boundary, color="#e0e3e6", linewidth=0.8)

        width = 0.158
        for group_index, (device, batch) in enumerate(GROUPS):
            group_series = SERIES if device == "cpu" else SERIES[:-1]
            for series_index, series in enumerate(group_series):
                x = group_index + (series_index - (len(group_series) - 1) / 2) * width
                color = COLORS[series_index]
                timing = timings.get((series, device, batch))
                if timing is None:
                    ax.text(
                        x,
                        0.025,
                        "not measured",
                        transform=ax.get_xaxis_transform(),
                        rotation=90,
                        ha="center",
                        va="bottom",
                        color=color,
                        fontsize=8,
                    )
                    continue
                median, iqr = timing
                ax.bar(x, median, width=width * 0.92, color=color, zorder=3)
                ax.errorbar(
                    x,
                    median,
                    yerr=[[min(iqr / 2, median - low)], [iqr / 2]],
                    fmt="none",
                    ecolor="#30343a",
                    elinewidth=0.9,
                    capsize=2,
                    zorder=4,
                )
                label = f"{median:,.2f}" if median >= 0.01 else f"{median:.2g}"
                ax.annotate(
                    label,
                    (x, median + iqr / 2),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    rotation=35,
                    rotation_mode="anchor",
                    fontsize=8.5,
                    fontweight="medium",
                    zorder=5,
                )

        handles = [
            Patch(facecolor=color, label=series if series != "OpenCV" else "OpenCV (CPU only)")
            for series, color in zip(SERIES, COLORS)
        ]
        opencv = timings.get(("OpenCV", "cpu", 1))
        if opencv is not None:
            ax.axhline(opencv[0], color=COLORS[-1], linestyle="--", linewidth=1.2, zorder=4)
            handles.append(Line2D([0], [0], color=COLORS[-1], linestyle="--", label="OpenCV CPU reference"))
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.835),
            ncol=3,
            frameon=False,
            fontsize=10,
            columnspacing=2.1,
            handlelength=2,
        )
        ax.set_xticks(range(len(GROUPS)), GROUP_LABELS)
        ax.tick_params(axis="x", length=0, pad=10)
        ax.set_ylabel("Median runtime [ms/image] · log scale", labelpad=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            "SIFT (DoG + SIFT descriptor) runtime", x=0.085, y=0.97, ha="left", fontsize=19, fontweight="semibold"
        )
        fig.text(
            0.085,
            0.905,
            "Oxford graf img1 · 640\N{MULTIPLICATION SIGN}800 · 4096 requested features · lower is faster",
            fontsize=11,
            color="#51565d",
        )
        fig.text(0.085, 0.867, environment, fontsize=10, color="#51565d")
        caption = (
            "True batches of a repeated image; batch latency divided by batch size. CPU: one thread. "
            "GPU: transfers excluded.\n"
            "Historical source versions run on the same PyTorch stack; native presets differ in detection quality.\n"
            "Current + compile: detector pyramid, response and refinement only.\n"
            "Whiskers: median ± IQR/2 (spread, not confidence intervals). "
            "OpenCV is measured on CPU only."
        )
        if clipped:
            caption += "\nLower whiskers clipped at the logarithmic axis floor."
        if output_note:
            caption += "\n" + output_note
        fig.text(0.085, 0.19, caption, ha="left", va="top", fontsize=9.2, linespacing=1.65, color="#51565d")
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg"):
            path = output.with_suffix(suffix)
            fig.savefig(path, dpi=200, facecolor="white")
            print(path)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Runtime benchmark JSON files")
    parser.add_argument("--output", type=Path, required=True, help="Output basename; writes .png and .svg")
    parser.add_argument("--cpu-label", help="CPU model label if absent from the benchmark metadata")
    args = parser.parse_args()
    try:
        timings = load_rows(args.inputs)
        environment = describe_environment(args.inputs, args.cpu_label)
        output_note = describe_outputs(args.inputs)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        plot_runtime(timings, args.output, environment, output_note)
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("matplotlib"):
            print("SKIP: Matplotlib is not installed; install the project's plotting dependencies to render charts.")
            return
        raise


if __name__ == "__main__":
    main()

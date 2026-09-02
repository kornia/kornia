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

"""Render docs/source/get-started/performance.rst from benchmarks/results/**.json.

The same result files feed the landing page: ``render_hero_svg`` draws the CPU-vs-accelerator bar
chart in the hero's "GPU-accelerated" tab, so the figures there are read from the committed run
rather than typed in.
"""

from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
OUT = REPO / "docs" / "source" / "get-started" / "performance.rst"
HERO_OUT = REPO / "docs" / "source" / "_generated" / "hero-benchmark.html"
LLMS_FULL = REPO / "docs" / "source" / "_extra" / "llms-full.txt"

# The one comparison the landing page draws: a kornia op at one batch size, CPU against the
# accelerator of the same machine. ``machine`` is a preference; any machine with both a CPU and an
# accelerator result set for the latest version is used when that one is absent.
HERO = {
    "suite": "augmentation",
    "machine": "apple-m1",
    "op": "RandomGaussianBlur",
    "batch": 32,
    "backend": "kornia (eager)",
}

BEGIN, END = "<!-- BENCH:BEGIN -->", "<!-- BENCH:END -->"

INTRO = """\
Performance
===========

Numbers below are generated at docs-build time from the committed benchmark result files in
``benchmarks/results/`` — median wall clock over repeated runs with device synchronization and
recorded hardware/software metadata (see ``benchmarks/README.md`` for methodology). Kornia's
regime is batched float tensors on an accelerator, differentiable end-to-end; OpenCV and
albumentations win the single-image uint8 CPU regime and we publish those losses too.

.. warning::

   Each table is a snapshot of one machine. Before contributing a run: close other applications,
   use mains power, and let the machine cool. Expect run-to-run variance (about ±5% on CPU and
   up to ±30% on MPS). Never compare numbers across machines — only across columns of one table.
   Only aggregate load metrics (load average, memory) are recorded, never process names.

Reproduce or contribute a machine with::

   python benchmarks/<suite>/flagship.py --device <cpu|cuda|mps> --contribute benchmarks/results

"""


def load_results(results_root: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(Path(results_root).rglob("*.json")):
        out.setdefault(path.parent.name, {})[path.name] = json.loads(path.read_text())
    return out


def _version_key(v: str) -> tuple:
    # pad to three numeric parts so digit-less dirs (e.g. "unknown") stay comparable
    nums = ([int(x) for x in re.findall(r"\d+", v)[:3]] + [0, 0, 0])[:3]
    is_final = not re.search(r"(rc|a|b|dev)\d*$", v)
    return (*nums, is_final, v)


def latest_version(versions: list[str]) -> str:
    return max(versions, key=_version_key)


def _table(payload: dict) -> str:
    rows = payload["results"]
    backends = list(dict.fromkeys(r["backend"] for r in rows))
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - op @ batch"]
    lines += [f"     - {b}" for b in backends]
    cells = {(r["op"], r["batch"], r["backend"]): r.get("throughput_per_s") for r in rows}
    for op, batch in dict.fromkeys((r["op"], r["batch"]) for r in rows):
        lines.append(f"   * - {op} @ {batch}")
        for b in backends:
            v = cells.get((op, batch, b))
            lines.append(f"     - {v:.0f}" if isinstance(v, (int, float)) else "     - —")
    return "\n".join(lines) + "\n"


def render_page(results_root: Path) -> str:
    data = load_results(results_root)
    if not data:
        return INTRO + "\nNo benchmark results are committed yet.\n"
    version = latest_version(list(data))
    parts = [INTRO, f"Results for kornia {version}\n{'-' * (20 + len(version))}\n"]
    for fname, payload in sorted(data[version].items()):
        suite, slug, device = fname[:-5].split("--")
        meta = payload["metadata"]
        parts.append(f"\n{suite} — {slug} ({device})\n{'^' * (len(suite) + len(slug) + len(device) + 6)}\n")
        parts.append(
            f"``{meta['platform']}`` — torch {meta['torch']}, kornia {meta['kornia']}, "
            f"commit ``{meta['git_commit']}``, {meta['timestamp_utc'][:10]}, throughput in items/s\n\n"
        )
        parts.append(_table(payload))
    older = sorted(set(data) - {version})
    if older:
        parts.append("\nOlder result sets in git: " + ", ".join(f"``benchmarks/results/{v}/``" for v in older) + "\n")
    return "\n".join(parts)


def _throughput(payload: dict, op: str, batch: int, backend: str) -> float | None:
    for row in payload["results"]:
        if row["op"] == op and row["batch"] == batch and row["backend"] == backend:
            value = row.get("throughput_per_s")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def hero_figures(results_root: Path) -> dict | None:
    """CPU and accelerator throughput of the ``HERO`` op from the latest committed result set.

    Returns ``None`` when no machine has both measurements, so the caller can leave the chart out.
    """
    data = load_results(results_root)
    if not data:
        return None
    version = latest_version(list(data))
    by_machine: dict[str, dict[str, dict]] = {}
    for fname, payload in data[version].items():
        suite, slug, device = fname[:-5].split("--")
        if suite == HERO["suite"]:
            by_machine.setdefault(slug, {})[device] = payload
    machines = sorted(by_machine, key=lambda slug: (slug != HERO["machine"], slug))
    for slug in machines:
        devices = by_machine[slug]
        accelerators = sorted(device for device in devices if device != "cpu")
        if "cpu" not in devices or not accelerators:
            continue
        cpu = _throughput(devices["cpu"], HERO["op"], HERO["batch"], HERO["backend"])
        gpu = _throughput(devices[accelerators[0]], HERO["op"], HERO["batch"], HERO["backend"])
        if cpu is None or gpu is None or cpu <= 0 or gpu <= 0:
            continue
        meta = devices[accelerators[0]]["metadata"]
        return {
            "machine": slug.replace("-", " ").title(),  # "apple-m1" -> "Apple M1"
            "device": accelerators[0],
            "cpu": cpu,
            "gpu": gpu,
            "speedup": gpu / cpu,
            "kornia": meta["kornia"],
            "torch": meta["torch"],
        }
    return None


def render_hero_svg(results_root: Path) -> str:
    """The landing page's CPU-vs-accelerator bar chart, as an HTML fragment; empty when there is no data."""
    fig = hero_figures(results_root)
    if fig is None:
        return ""
    full, left = 516, 80  # the longer bar spans the chart; the other is scaled to it
    longest = max(fig["cpu"], fig["gpu"])

    def bar(y: int, value: float, text: str, fill: str, text_class: str) -> str:
        width = round(full * value / longest)
        rect = f'<rect x="{left}" y="{y}" width="{width}" height="32" rx="6" class="{fill}"/>'
        if width == full:  # no room to the right: print the value inside the bar, right-aligned
            anchor = f'x="{left + width - 12}" y="{y + 23}" class="illo-value {text_class}" text-anchor="end"'
        else:
            anchor = f'x="{left + width + 12}" y="{y + 23}" class="illo-value"'
        return f"{rect}\n    <text {anchor}>{text}</text>"

    faster = "GPU" if fig["gpu"] >= fig["cpu"] else "CPU"
    ratio = max(fig["gpu"], fig["cpu"]) / min(fig["gpu"], fig["cpu"])
    speedup = f" · {ratio:.1f}\N{MULTIPLICATION SIGN} faster"
    cpu_text = f"{fig['cpu']:,.0f} img/s" + (speedup if faster == "CPU" else "")
    gpu_text = f"{fig['gpu']:,.0f} img/s" + (speedup if faster == "GPU" else "")
    title = escape(f"{HERO['op']} · batch {HERO['batch']} · {fig['machine']}")
    label = escape(
        f"{HERO['op']} at batch {HERO['batch']} on an {fig['machine']}: {fig['cpu']:,.0f} images per second on "
        f"the CPU, {fig['gpu']:,.0f} on the GPU"
    )
    note = escape(f"items/s, eager mode — committed benchmark run, kornia {fig['kornia']} / torch {fig['torch']}")
    return f"""\
<div class="kornia-tab-visual">
  <svg class="kornia-illo" viewBox="0 0 640 220" role="img" aria-label="{label}">
    <text x="24" y="36" class="illo-title">{title}</text>
    <text x="24" y="90" class="illo-label">CPU</text>
    {bar(68, fig["cpu"], cpu_text, "illo-muted", "")}
    <text x="24" y="152" class="illo-label">GPU</text>
    {bar(130, fig["gpu"], gpu_text, "illo-primary", "illo-on-primary")}
    <text x="24" y="204" class="illo-note">{note}</text>
  </svg>
</div>
"""


def _digest(results_root: Path) -> str:
    data = load_results(results_root)
    if not data:
        return "- No committed benchmark results yet.\n"
    version = latest_version(list(data))
    lines = [f"- Result set: kornia {version}, committed in `benchmarks/results/{version}/` (per-machine"]
    lines += ["  snapshots; reproduce with `python benchmarks/<suite>/flagship.py --contribute benchmarks/results`)."]
    for fname, payload in sorted(data[version].items()):
        suite, slug, device = fname[:-5].split("--")
        meta = payload["metadata"]
        rows = [r for r in payload["results"] if isinstance(r.get("throughput_per_s"), (int, float))]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["throughput_per_s"])
        kornia_rows = [r for r in rows if r["backend"].startswith("kornia")]
        worst = min(kornia_rows, key=lambda r: r["throughput_per_s"]) if kornia_rows else None
        line = (
            f"- {suite} on {slug}/{device} ({meta['timestamp_utc'][:10]}): fastest overall "
            f"{best['backend']} {best['op']}@{best['batch']} at {best['throughput_per_s']:.0f} items/s"
        )
        if worst is not None:
            line += f"; slowest kornia op {worst['op']}@{worst['batch']} at {worst['throughput_per_s']:.0f} items/s"
        lines.append(line + ".")
    return "\n".join(lines) + "\n"


def refresh_llms(llms_path: Path, results_root: Path) -> None:
    text = Path(llms_path).read_text()
    if BEGIN not in text or END not in text or text.index(BEGIN) > text.index(END):
        raise RuntimeError(f"markers {BEGIN}/{END} not found (in order) in {llms_path}")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    Path(llms_path).write_text(f"{head}{BEGIN}\n{_digest(results_root)}{END}{tail}")


def main() -> None:
    OUT.write_text(render_page(RESULTS))
    HERO_OUT.parent.mkdir(exist_ok=True)
    HERO_OUT.write_text(render_hero_svg(RESULTS))


if __name__ == "__main__":
    if "--refresh-llms" in sys.argv:
        refresh_llms(LLMS_FULL, RESULTS)
    else:
        main()

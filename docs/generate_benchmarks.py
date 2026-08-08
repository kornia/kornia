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

"""Render docs/source/get-started/performance.rst from benchmarks/results/**.json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
OUT = REPO / "docs" / "source" / "get-started" / "performance.rst"
LLMS_FULL = REPO / "docs" / "source" / "_extra" / "llms-full.txt"

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
    nums = [int(x) for x in re.findall(r"\d+", v)[:3]]
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


if __name__ == "__main__":
    if "--refresh-llms" in sys.argv:
        refresh_llms(LLMS_FULL, RESULTS)
    else:
        main()

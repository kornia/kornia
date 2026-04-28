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

"""Generate a leaderboard markdown table from results.jsonl.

Usage:
    python -m benchmarks.cuda_graph.leaderboard [--input PATH] [--output PATH]

Sorting order:
    1. FAILED first (highest priority — these are CI blockers)
    2. OK sorted by speedup descending
    3. SKIPPED last
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_STATUS_ORDER = {"FAILED": 0, "OK": 1, "SKIPPED": 2}


def _sort_key(row: dict) -> tuple[int, float]:
    status_rank = _STATUS_ORDER.get(row.get("capture_status", "SKIPPED"), 2)
    # For OK rows, sort by speedup descending (negate so higher speedup sorts first)
    speedup = row.get("speedup") or 0.0
    return (status_rank, -speedup)


def _fmt(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def generate(input_jsonl: Path, output_md: Path) -> None:
    rows: list[dict] = []
    with input_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rows.sort(key=_sort_key)

    lines: list[str] = [
        "# CUDA Graph Capture Leaderboard",
        "",
        "Generated from `results.jsonl`. FAILED = CI blocker; OK = graph-compatible; SKIPPED = no CUDA.",
        "",
        "| Transform | Capture | Eager ms | Replay ms | Speedup | Launch OH % |",
        "|-----------|---------|----------|-----------|---------|-------------|",
    ]
    for r in rows:
        status = r.get("capture_status", "?")
        name = r.get("transform", "?")
        eager = _fmt(r.get("eager_ms"))
        replay = _fmt(r.get("replay_ms"))
        speedup = _fmt(r.get("speedup"))
        overhead = _fmt(r.get("launch_overhead_pct"))
        lines.append(f"| {name} | {status} | {eager} | {replay} | {speedup} | {overhead} |")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} rows to {output_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CUDA graph capture leaderboard.")
    parser.add_argument(
        "--input",
        default="benchmarks/cuda_graph/results.jsonl",
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/cuda_graph/leaderboard.md",
        help="Path to write leaderboard markdown",
    )
    args = parser.parse_args()
    generate(Path(args.input), Path(args.output))

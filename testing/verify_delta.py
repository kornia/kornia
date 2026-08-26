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

"""Diff failing-test SETS between this branch and ``origin/main`` on every supported surface.

Run as ``pixi run verify-delta`` (``python -m testing.verify_delta``). Counts are never compared —
a branch can add and fix an equal number of tests and look unchanged by count.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

WIDEN_PREFIXES = ("testing/", "tests/conftest.py", "conftest.py", "pyproject.toml", "pixi.toml")


def changed_test_dirs(changed_files: Iterable[str]) -> list[str]:
    """Map changed paths to the test directories that exercise them (``["tests"]`` when everything)."""
    dirs: set[str] = set()
    for f in changed_files:
        if f.startswith(WIDEN_PREFIXES):
            return ["tests"]
        parts = Path(f).parts
        if parts[0] == "kornia":
            if len(parts) == 2:  # kornia/constants.py and friends have no dedicated test dir
                return ["tests"]
            dirs.add(f"tests/{parts[1]}")
        elif parts[0] == "tests" and len(parts) > 2:
            dirs.add(f"tests/{parts[1]}")
    return sorted(dirs)


def failing_ids(junit_xml: str | Path) -> set[str]:
    """Return ``classname::name`` for every testcase with a ``failure`` or ``error`` child."""
    path = Path(junit_xml)
    if not path.exists():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()  # noqa: S314 -- locally generated pytest junit report, not untrusted input
    ids: set[str] = set()
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            ids.add(f"{case.get('classname')}::{case.get('name')}")
    return ids


@dataclass(frozen=True)
class FailureDelta:
    new: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def diff_failures(branch: set[str], main: set[str]) -> FailureDelta:
    return FailureDelta(new=sorted(branch - main), fixed=sorted(main - branch), unchanged=sorted(branch & main))


def render_table(rows: Sequence[tuple[str, FailureDelta | None]]) -> str:
    lines = ["| surface | new | fixed | unchanged |", "|---|---|---|---|"]
    details: list[str] = []
    for name, delta in rows:
        if delta is None:
            lines.append(f"| {name} | skipped | | |")
            continue
        lines.append(f"| {name} | {len(delta.new)} | {len(delta.fixed)} | {len(delta.unchanged)} |")
        details.extend(f"NEW {t}" for t in delta.new)
        details.extend(f"FIXED {t}" for t in delta.fixed)
    return "\n".join(lines + ([""] + details if details else []))

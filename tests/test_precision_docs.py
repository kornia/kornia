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

"""The precision guide's known-failure counts follow the checked-in CPU half-precision manifests.

``docs/source/get-started/precision.rst`` is the only page that states half-precision counts; the READMEs link to it.
Its module table and headline totals are derived from ``testing/half_precision_xfails/cpu_*.txt``, so a change to a
manifest that does not update the table fails here instead of leaving the page stale.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "source" / "get-started" / "precision.rst"
MANIFESTS = ROOT / "testing" / "half_precision_xfails"
DTYPES = ("float16", "bfloat16")
OTHER_PACKAGES = "contrib, core, io, onnx, sensors, tracking, utils"

# Test files whose module row is not their directory's.
_FILE_ROWS = {
    "tests/integration/test_conversions.py": "kornia.geometry.conversions",
    "tests/geometry/test_conversions.py": "kornia.geometry.conversions",
    "tests/geometry/test_homography.py": "kornia.geometry.homography",
    "tests/geometry/test_degensac.py": "kornia.geometry.ransac",
}


def _module_row(test_path: str, rows: set[str]) -> str:
    """Return the module-table row that a manifest entry's test file counts towards."""
    if test_path in _FILE_ROWS:
        return _FILE_ROWS[test_path]
    parts = test_path.split("/")
    if parts[1] == "geometry":
        if len(parts) > 3 and f"kornia.geometry.{parts[2]}" in rows:
            return f"kornia.geometry.{parts[2]}"
        if parts[2].startswith("test_ransac"):
            return "kornia.geometry.ransac"
        return "kornia.geometry (other)"
    if f"kornia.{parts[1]}" in rows:
        return f"kornia.{parts[1]}"
    return OTHER_PACKAGES


def _manifest_paths(dtype: str) -> list[str]:
    lines = (MANIFESTS / f"cpu_{dtype}.txt").read_text(encoding="utf-8").splitlines()
    return [line.split("\t")[2].split("::")[0] for line in lines if line.strip() and not line.startswith("#")]


def _module_table() -> dict[str, tuple[str, str, int, int]]:
    """Parse the guide's known-failure list-table into ``{module: (fp16 mark, bf16 mark, fp16 count, bf16 count)}``."""
    text = GUIDE.read_text(encoding="utf-8")
    # The directive and its indented body, up to the first line that is neither blank nor indented.
    match = re.search(
        r"^\.\. list-table:: Known failures by module \(Linux CPU\)\n((?:[ \t]*\n|[ \t]+\S.*\n)+)", text, re.MULTILINE
    )
    assert match, f"{GUIDE}: the 'Known failures by module (Linux CPU)' list-table is missing"
    rows: dict[str, tuple[str, str, int, int]] = {}
    for block in re.split(r"^   \* - ", match.group(1), flags=re.MULTILINE)[2:]:
        cells = [" ".join(cell.split()) for cell in re.split(r"\n     - ?", block.rstrip())]
        assert len(cells) >= 4, f"{GUIDE}: module row {cells[0]!r} has {len(cells)} cells, expected 5"
        counts = re.fullmatch(r"(\d+) / (\d+)", cells[3])
        assert counts, f"{GUIDE}: module row {cells[0]!r} has known failures {cells[3]!r}, expected 'N / M'"
        rows[cells[0].strip("`")] = (cells[1], cells[2], int(counts.group(1)), int(counts.group(2)))
    assert rows, f"{GUIDE}: the module table has no rows"
    return rows


def test_module_table_counts_match_the_manifests() -> None:
    table = _module_table()
    expected = {name: [0, 0] for name in table}
    for index, dtype in enumerate(DTYPES):
        for path in _manifest_paths(dtype):
            row = _module_row(path, set(table))
            assert row in expected, f"{path} counts towards module row {row!r}, which the guide's table lacks"
            expected[row][index] += 1
    actual = {name: [row[2], row[3]] for name, row in table.items()}
    assert actual == expected


@pytest.mark.parametrize("index", [0, 1], ids=DTYPES)
def test_module_table_marks_follow_the_counts(index: int) -> None:
    for name, row in _module_table().items():
        assert row[index] == ("✅" if row[2 + index] == 0 else "⚠️"), name


def test_headline_totals_match_the_manifests() -> None:
    fp16, bf16 = (len(_manifest_paths(dtype)) for dtype in DTYPES)
    assert f"There are **{fp16} float16 and {bf16} bfloat16 recorded failures**." in GUIDE.read_text(encoding="utf-8")


def test_manifest_pins_match_the_guide() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for dtype in DTYPES:
        header = (MANIFESTS / f"cpu_{dtype}.txt").read_text(encoding="utf-8")
        python = re.search(r"^# python: (\S+)$", header, re.MULTILINE)
        pytorch = re.search(r"^# pytorch: (\S+)$", header, re.MULTILINE)
        assert python
        assert pytorch
        assert f"pinned to Python {python.group(1)} and PyTorch {pytorch.group(1)}" in text

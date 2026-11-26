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

"""kornia's ONNX and torch.compile promises (#5819): which survey cases are promised, which are left out and why, and
which are known to fail on each enforced torch release.

The conformance tests (``testing/conformance.py``) check exactly this, and the docs build renders the support
statement of ``docs/source/get-started/export-support.rst`` from it (``docs/generate_export_support.py``), so the two
cannot drift apart. It imports neither pytest nor the survey's cases: the docs environment has no pytest.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

REPO = Path(__file__).resolve().parents[1]
EXPORT_SUPPORT = REPO / "docs" / "export_support"
SNAPSHOT = REPO / "docs" / "source" / "_data" / "export_support.json"
MANIFEST_DIR = Path(__file__).with_name("known_failure_xfails")


@dataclass(frozen=True)
class Promise:
    """One conformance promise: the snapshot status it promises, how a run selects it, and where its tests live."""

    name: str  # in messages: "ONNX", "torch.compile"
    field: str  # the snapshot's status field, and the stem of the option, marker and manifests
    test_file: str
    case_tests: tuple[str, ...]  # the per-case tests, parametrized by case name

    @property
    def option(self) -> str:
        return f"--{self.field}-conformance"

    @property
    def env_var(self) -> str:
        return f"KORNIA_TEST_{self.field.upper()}_CONFORMANCE"

    @property
    def marker(self) -> str:
        return f"{self.field}_conformance"


ONNX = Promise(
    "ONNX", "onnx", "tests/onnx/test_conformance.py", ("test_export_matches_eager", "test_legacy_export_is_not_wrong")
)
# The root conftest deselects tests whose name or id contains "compile" or "dynamo" unless KORNIA_TEST_OPTIMIZER is set,
# so the compile promise's tests are named after inductor; only the option and the marker say "compile".
COMPILE = Promise("torch.compile", "compile", "tests/test_inductor_conformance.py", ("test_inductor_matches_eager",))
PROMISES = (ONNX, COMPILE)
# What each per-case test checks, as the support statement names it.
CASE_TEST_CHECKS = {
    "test_export_matches_eager": "ONNX export (dynamo exporter)",
    "test_legacy_export_is_not_wrong": "ONNX export (legacy exporter)",
    "test_inductor_matches_eager": "torch.compile",
}
# Where CI checks the promises: the onnx-conformance and compile-conformance jobs of scheduled_test_cpu.yml (on main,
# CI_RUNS_PER_DAY times a day) and of pr_test_cpu.yml (pull requests). tests/test_generate_export_support.py checks
# these against the workflows, and the support statement renders them.
CI_TORCH_SCHEDULED = ("2.9.1", "2.15.0")
CI_TORCH_PULL_REQUEST = ("2.15.0",)
CI_RUNS_PER_DAY = 2
CI_PYTHON = "3.11"
CI_OS = "ubuntu-latest"
CI_DTYPE = "float32"
# The ONNX opset the conformance harness exports at: the one ``ONNXExportMixin.to_onnx`` requests (checked against
# both by tests/test_generate_export_support.py).
ONNX_OPSET = 18


def release(version: str) -> str:
    """The ``<major>.<minor>`` release of a torch version: ``"2.14.0+cpu"`` -> ``"2.14"``."""
    return ".".join(version.split("+", maxsplit=1)[0].split(".")[:2])


# The torch releases whose manifests the scheduled conformance jobs enforce.
ENFORCED_TORCH_RELEASES = tuple(release(version) for version in CI_TORCH_SCHEDULED)

# The stable core of docs/source/get-started/stability.rst, as the ``kornia.<package>`` names the snapshot records.
# tests/scripts/test_conformance_packages.py checks it against that page and against the CI path mapping.
STABLE_CORE = (
    "geometry",
    "augmentation",
    "color",
    "filters",
    "enhance",
    "losses",
    "metrics",
    "morphology",
    "io",
    "image",
    "core",
    "feature",
)

# The survey's case registries (``docs/export_support/cases_*.py``) that may hold promised cases. ``cases_models.py``
# builds the pretrained models of ``kornia.models``, none of which is promised.
REGISTRIES = ("cases_aug", "cases_feature", "cases_geomA", "cases_geomB", "cases_misc")

# Snapshot records a promise leaves out although they meet its filter, with the reason the support statement shows.
# Each must still meet the filter (exclusion_problems): delete an entry once a regenerated snapshot no longer does.
EXCLUDED: dict[str, dict[str, str]] = {
    "onnx": {},
    "compile": {
        # #6194: the survey marked these "ok" by torch._dynamo.explain's break count, but they read tensor values on
        # the host (a loop count, slice bounds), which torch.compile(..., fullgraph=True) cannot be relied on to
        # capture.
        "bbox.nms": (
            "for one box or more, the number of loop iterations and the output size depend on the box values, so"
            " torch.compile cannot capture it in one graph; a default torch.compile runs it eagerly (#6194)"
        ),
        "image.draw_rectangle": (
            "reads the rectangle coordinates back to the host, where a default torch.compile breaks the graph, and"
            " fullgraph=True fails on most inputs (#6194)"
        ),
        "image.draw_rectangle[fill]": (
            "reads the rectangle coordinates back to the host, where a default torch.compile breaks the graph, and"
            " fullgraph=True fails on most inputs (#6194)"
        ),
    },
}

# A manifest comment that opens with an issue reference ("#5037: ...", "pytorch/pytorch#198270 (...): ...") names the
# tracking issue of the entries below it.
_CITATION = re.compile(r"^# (?P<issue>(?:[\w.-]+/[\w.-]+)?#\d+)\b")


@functools.cache
def pretrained_cases() -> frozenset[str]:
    """Names of the registry cases tagged ``pretrained``, read from the registry sources without importing them."""
    names: set[str] = set()
    for registry in REGISTRIES:
        path = EXPORT_SUPPORT / f"{registry}.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            tags = next((kw.value for kw in node.keywords if kw.arg == "tags"), None)
            if not isinstance(tags, (ast.Tuple, ast.List)):
                continue
            if not any(isinstance(tag, ast.Constant) and tag.value == "pretrained" for tag in tags.elts):
                continue
            first = node.args[0] if node.args else None
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                raise ValueError(f"{path}:{node.lineno}: a case tagged pretrained needs a literal name")
            names.add(first.value)
    return frozenset(names)


@functools.cache
def snapshot() -> dict[str, Any]:
    """The committed survey snapshot."""
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@functools.cache
def candidate_cases(promise: Promise) -> dict[str, str]:
    """Map every case that meets ``promise``'s filter to its package, in snapshot order, exclusions included.

    The filter: status ``ok`` in the snapshot field of the promise, a stable-core package, and nothing built with
    pretrained weights (registry tag ``pretrained``; the stable core holds only the classical parts of
    ``kornia.feature``).
    """
    pretrained = pretrained_cases()
    candidates: dict[str, str] = {}
    for record in snapshot()["cases"]:
        package = record["package"].removeprefix("kornia.")
        if record[promise.field] != "ok" or package not in STABLE_CORE or record["name"] in pretrained:
            continue
        if package == "feature" and record["section"] == "models":
            continue
        candidates[record["name"]] = package
    return candidates


@functools.cache
def promised_cases(promise: Promise) -> dict[str, str]:
    """Map every case ``promise`` covers to its package (``geometry``, ``color``, ...), in snapshot order."""
    excluded = EXCLUDED[promise.field]
    return {name: package for name, package in candidate_cases(promise).items() if name not in excluded}


def exclusion_problems(promise: Promise, excluded: Mapping[str, str] | None = None) -> list[str]:
    """The exclusions of ``promise`` (default: its ``EXCLUDED`` entry) that leave nothing out any more.

    An excluded name must be a case the promise would cover without the exclusion. A regenerated snapshot that no
    longer marks it ``ok`` drops it by itself, and the stale entry would then hide it if it ever passed again.
    """
    excluded = EXCLUDED[promise.field] if excluded is None else excluded
    candidates = candidate_cases(promise)
    return [
        f"{name}: the snapshot no longer records it as {promise.field} 'ok' in a stable-core package; delete its"
        f" entry from EXCLUDED[{promise.field!r}] in testing/conformance_promise.py"
        for name in excluded
        if name not in candidates
    ]


def promised_digest(names: Iterable[str]) -> str:
    """SHA-256 of the sorted names, one per line: what ``test_promised_set_is_pinned`` compares."""
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def manifest_path(promise: Promise, version: str | None = None) -> Path:
    """A promise's known-failure manifest of a torch release (default: the running one).

    For example ``onnx_conformance_torch2.14.txt``.
    """
    return MANIFEST_DIR / f"{promise.marker}_torch{release(version or torch.__version__)}.txt"


@dataclass(frozen=True)
class KnownFailure:
    """One entry of a known-failure manifest."""

    line: int
    exception: str  # the exception type's name as the manifest spells it
    nodeid: str
    test: str  # the per-case test
    case: str  # the promised case
    issue: str  # the reference that opens the comment above the entry: "#5037", "pytorch/pytorch#198270"


def known_failures(promise: Promise, path: Path) -> list[KnownFailure]:
    """Read ``<exception type>\\t<node ID>`` lines; every node ID must name a per-case test of a promised case.

    Every entry must sit under a comment that opens with its tracking issue (``# #NNNN: ...``); the comment lines
    right below it continue it, and a comment that names no issue after an entry or a blank line starts a block
    without one. The exception name is checked by ``testing.conformance.load_known_failures``, which maps it to a type.
    """
    node = re.compile(rf"^{re.escape(promise.test_file)}::(?P<test>\w+)\[(?P<case>.+)\]$")
    promised = promised_cases(promise)
    entries: list[KnownFailure] = []
    seen: set[str] = set()
    issue = ""
    block_ended = False  # an entry or a blank line since the last comment line
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            block_ended = True
            continue
        if line.startswith("#"):
            if citation := _CITATION.match(line):
                issue = citation["issue"]
            elif block_ended:  # a comment block of its own that names no issue
                issue = ""
            block_ended = False
            continue
        block_ended = True
        try:
            exception, nodeid = line.split("\t", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: expected '<exception>\\t<node ID>'") from error
        match = node.match(nodeid)
        if match is None or match["test"] not in promise.case_tests:
            raise ValueError(f"{path}:{number}: not a per-case node of {promise.test_file}: {nodeid}")
        if match["case"] not in promised:
            raise ValueError(f"{path}:{number}: {match['case']!r} is not a promised case (renamed or dropped?)")
        if nodeid in seen:
            raise ValueError(f"{path}:{number}: duplicate node ID: {nodeid}")
        if not issue:
            raise ValueError(f"{path}:{number}: no tracking issue; put the entry under a '# #NNNN: why' comment")
        seen.add(nodeid)
        entries.append(KnownFailure(number, exception, nodeid, match["test"], match["case"], issue))
    return entries

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

"""Support for ``tests/onnx/test_conformance.py``: the ONNX promise of #5819, checked with the survey's own cases.

The promised cases are the records of the committed survey snapshot (``docs/source/_data/export_support.json``)
whose ONNX status is ``ok`` and whose package is in the stable core of
``docs/source/get-started/stability.rst``, minus the pretrained wrappers of ``kornia.feature`` (section ``models``).
Each is run through ``docs/export_support/harness.py`` -- the code the survey runs -- by case name, so the test and
the survey page cannot drift apart.

The run is opt-in: ``--onnx-conformance=<package>[,<package>...]`` or ``all`` (env
``KORNIA_TEST_ONNX_CONFORMANCE``); without it the root ``conftest.py`` deselects every test marked
``onnx_conformance``. Known failures are strict xfails listed per torch release in
``testing/known_failure_xfails/onnx_conformance_torch<major>.<minor>.txt``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import io
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Sequence

import pytest
import torch

from testing.known_failures import _EXCEPTION_TYPES, KnownFailureTracker

REPO = Path(__file__).resolve().parents[1]
EXPORT_SUPPORT = REPO / "docs" / "export_support"
SNAPSHOT = REPO / "docs" / "source" / "_data" / "export_support.json"
MANIFEST_DIR = Path(__file__).with_name("known_failure_xfails")
TEST_FILE = "tests/onnx/test_conformance.py"
CASE_TESTS = ("test_export_matches_eager", "test_legacy_export_is_not_wrong")
MARKER = "onnx_conformance"
OPTION = "--onnx-conformance"
ENV_VAR = "KORNIA_TEST_ONNX_CONFORMANCE"

# The registries the promise is joined to. ``cases_models.py`` is never imported: it builds the pretrained models of
# ``kornia.models``, none of which is promised, and downloads their weights on a cold cache.
REGISTRIES = ("cases_aug", "cases_feature", "cases_geomA", "cases_geomB", "cases_misc")

# The stable core of docs/source/get-started/stability.rst, as the ``kornia.<package>`` names the snapshot records.
# tests/scripts/test_onnx_conformance_packages.py checks it against that page and against the CI path mapping.
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

# What a registry case's skip reason says when the download guard below refused a weights download.
NOT_CACHED = "weights are not in the cache"

# Legacy (TorchScript) export statuses that pass: the export worked and matched, or it failed loudly.
LEGACY_PASSES = ("ok", "export-fail", "checker-fail", "ort-load-fail", "ort-run-fail")


def parse_selection(value: str | None) -> tuple[str, ...]:
    """Return the packages an ``--onnx-conformance`` value selects; ``()`` when it is unset or blank.

    Args:
        value: ``all``, or comma-separated package names with or without the ``kornia.`` prefix.

    Returns:
        The selected packages, in ``STABLE_CORE`` order.

    Raises:
        ValueError: when a name is not a stable-core package.
    """
    names = [v.strip().removeprefix("kornia.") for v in (value or "").split(",") if v.strip()]
    if "all" in names:
        return STABLE_CORE
    unknown = sorted(set(names) - set(STABLE_CORE))
    if unknown:
        raise ValueError(f"{OPTION}: unknown package(s) {unknown}; choose from {', '.join(STABLE_CORE)} or all")
    return tuple(p for p in STABLE_CORE if p in names)


@functools.cache
def promised_cases() -> dict[str, str]:
    """Map every promised case name to its package (``geometry``, ``color``, ...), in snapshot order."""
    promised: dict[str, str] = {}
    for record in json.loads(SNAPSHOT.read_text(encoding="utf-8"))["cases"]:
        package = record["package"].removeprefix("kornia.")
        if record["onnx"] != "ok" or package not in STABLE_CORE:
            continue
        if package == "feature" and record["section"] == "models":
            continue
        promised[record["name"]] = package
    return promised


def selected_cases(packages: Sequence[str]) -> list[str]:
    """Return the promised case names of ``packages``, in snapshot order."""
    return [name for name, package in promised_cases().items() if package in packages]


@contextlib.contextmanager
def _no_downloads() -> Iterator[None]:
    """Make a weights download raise instead, so a pretrained case on a cold cache is skipped, not fetched."""
    from testing.doctest_downloads import install_download_guard

    undo: list[tuple[Any, str, Any]] = []

    def set_attribute(obj: Any, name: str, value: Any) -> None:
        undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def refuse(url: str) -> None:
        raise RuntimeError(f"{NOT_CACHED} (the conformance test does not download): {url}")

    try:
        install_download_guard(set_attribute, refuse)
        yield
    finally:
        for obj, name, value in reversed(undo):
            setattr(obj, name, value)


@functools.cache
def import_survey_module(name: str) -> ModuleType:
    """Import ``docs/export_support/<name>.py`` without letting the import touch the process.

    The registries seed the global RNG and build their inputs (and pretrained models) at import, and print the
    traceback of a model they cannot build: the RNG state is restored, downloads refused and the output dropped.
    """
    path = str(EXPORT_SUPPORT)
    sys.path.insert(0, path)
    try:
        sink = io.StringIO()
        with (
            torch.random.fork_rng(devices=[]),
            _no_downloads(),
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            return importlib.import_module(name)
    finally:
        sys.path.remove(path)


@functools.cache
def registry_cases() -> dict[str, dict[str, Any]]:
    """Every case of the five registries by name; a name registered twice is an error."""
    cases: dict[str, dict[str, Any]] = {}
    for registry in REGISTRIES:
        for case in import_survey_module(registry).CASES:
            if case["name"] in cases:
                raise ValueError(f"case {case['name']!r} is registered twice ({registry})")
            cases[case["name"]] = case
    return cases


def run_case(case: dict[str, Any], dynamo: bool) -> dict[str, Any]:
    """Run one case through the survey harness, leaving the global RNG as it was."""
    harness = import_survey_module("harness")
    with torch.random.fork_rng(devices=[]):
        return harness.run_case(case, dynamo=dynamo)


def describe(record: dict[str, Any]) -> str:
    """A failure message naming the status, the error and the kornia frames involved."""
    lines = [f"{record['status']}: {record.get('error', '')}"]
    lines += [f"  at {frame}" for frame in record.get("where") or ()]
    if record.get("error_full"):
        lines.append(record["error_full"][-1500:])
    return "\n".join(lines)


def manifest_path(version: str | None = None) -> Path:
    """The known-failure manifest of a torch release (default: the running one), e.g. ``..._torch2.14.txt``."""
    major, minor = (version or torch.__version__).split("+")[0].split(".")[:2]
    return MANIFEST_DIR / f"onnx_conformance_torch{major}.{minor}.txt"


_NODE = re.compile(rf"^{re.escape(TEST_FILE)}::(?P<test>\w+)\[(?P<case>.+)\]$")


def load_known_failures(path: Path) -> dict[str, type[BaseException]]:
    """Read ``<exception type>\\t<node ID>`` lines; every node ID must name a per-case test of a promised case."""
    failures: dict[str, type[BaseException]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        try:
            exception_name, nodeid = line.split("\t", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: expected '<exception>\\t<node ID>'") from error
        if exception_name not in _EXCEPTION_TYPES:
            raise ValueError(f"{path}:{number}: unknown exception type: {exception_name}")
        match = _NODE.match(nodeid)
        if match is None or match["test"] not in CASE_TESTS:
            raise ValueError(f"{path}:{number}: not a per-case node of {TEST_FILE}: {nodeid}")
        if match["case"] not in promised_cases():
            raise ValueError(f"{path}:{number}: {match['case']!r} is not a promised case (renamed or dropped?)")
        if nodeid in failures:
            raise ValueError(f"{path}:{number}: duplicate node ID: {nodeid}")
        failures[nodeid] = _EXCEPTION_TYPES[exception_name]
    return failures


class KnownFailurePlugin:
    """Apply the running torch's manifest to the final selection, after ``-k`` / ``-m`` / ``--deselect``."""

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[Any]) -> None:
        try:
            tracker = mark_known_failures(items)
        except (OSError, ValueError) as error:
            raise pytest.UsageError(str(error)) from error
        if tracker is not None:
            config.pluginmanager.register(tracker, "onnx-conformance-known-failure-tracker")


def mark_known_failures(items: Sequence[Any], path: Path | None = None) -> KnownFailureTracker | None:
    """Strictly xfail the collected entries of the running torch's manifest; ``None`` when it has none.

    Entries of packages the run did not select are not collected and not checked. A collected entry that passes
    (strict XPASS), skips, or fails with another exception type fails the session through the tracker.
    """
    path = path or manifest_path()
    if not path.exists():
        return None
    failures = load_known_failures(path)
    collected = {item.nodeid for item in items}
    selected = {nodeid: exception for nodeid, exception in failures.items() if nodeid in collected}
    reason = f"Known ONNX conformance failure recorded in {path.name}; see its tracking issue there"
    for item in items:
        if exception := selected.get(item.nodeid):
            item.add_marker(pytest.mark.xfail(reason=reason, raises=exception, strict=True))
    return KnownFailureTracker(selected, path, reason)

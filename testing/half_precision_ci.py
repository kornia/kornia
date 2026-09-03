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

from __future__ import annotations

import builtins
import hashlib
import importlib
import platform
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import numpy as np
import pytest
import torch
from torch.autograd.gradcheck import GradcheckError

from kornia.core.exceptions import BaseError

ISSUE_URL = "https://github.com/kornia/kornia/issues/4153"
_MANIFEST_DIR = Path(__file__).with_name("half_precision_xfails")
_XFAIL_REASON_PREFIX = f"Known Linux CPU half-precision failure tracked in {ISSUE_URL}"
_ALLOWED_EXCEPTION_MODULE_PREFIXES = ("kornia.", "torch.")
_EXCEPTION_TYPES: dict[str, type[BaseException]] = {
    "AssertionError": AssertionError,
    "KeyError": KeyError,
    "NotImplementedError": NotImplementedError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "kornia.core.exceptions.BaseError": BaseError,
    "torch.autograd.gradcheck.GradcheckError": GradcheckError,
}
_SCHEMA_VERSION = "1"
_SEED_SCHEME = "sha256-nodeid-v1"
_ALLOWED_PHASES = {"setup", "call"}


@dataclass(frozen=True)
class ManifestProfile:
    """Stable configuration for one Linux CPU half-precision baseline."""

    name: str
    dtype: str
    manifest_path: Path


@dataclass(frozen=True)
class ManifestEntry:
    """One exact failure outcome accepted by a manifest."""

    phase: Literal["setup", "call"]
    exception: str
    nodeid: str


@dataclass(frozen=True)
class ManifestEnvironment:
    """Compatibility and provenance fields recorded in a manifest header."""

    python: str
    pytorch: str
    os: str
    arch: str
    pytest: str
    numpy: str
    pillow: str


_PROFILES = {
    name: ManifestProfile(name, dtype, _MANIFEST_DIR / f"cpu_{dtype}.txt")
    for name, dtype in (("cpu-float16", "float16"), ("cpu-bfloat16", "bfloat16"))
}


def get_profile(name: str) -> ManifestProfile:
    """Return a supported CPU-half manifest profile."""
    try:
        return _PROFILES[name]
    except KeyError as error:
        known = ", ".join(sorted(_PROFILES))
        raise ValueError(f"unknown profile {name!r}; known profiles are: {known}") from error


def current_environment() -> ManifestEnvironment:
    """Capture enforced compatibility and warning-only provenance metadata."""
    try:
        from PIL import __version__ as pillow_version
    except ImportError:
        pillow_version = "not-installed"
    return ManifestEnvironment(
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        pytorch=torch.__version__.partition("+")[0],
        os=platform.system(),
        arch=platform.machine(),
        pytest=pytest.__version__,
        numpy=np.__version__,
        pillow=pillow_version,
    )


def serialize_manifest(
    profile: ManifestProfile,
    entries: Sequence[ManifestEntry],
    environment: ManifestEnvironment,
    generation_command: str,
) -> str:
    """Serialize a canonical profile manifest without resolving exception classes."""
    header = {
        "known-failure-schema": _SCHEMA_VERSION,
        "profile": profile.name,
        "python": environment.python,
        "pytorch": environment.pytorch,
        "os": environment.os,
        "arch": environment.arch,
        "seed-scheme": _SEED_SCHEME,
        "pytest": environment.pytest,
        "numpy": environment.numpy,
        "pillow": environment.pillow,
        "generated-by": generation_command,
    }
    lines = [*(f"# {key}: {value}" for key, value in header.items()), ""]
    lines.extend(f"{entry.phase}\t{entry.exception}\t{entry.nodeid}" for entry in sorted(entries, key=lambda x: x.nodeid))
    return "\n".join([*lines, ""])


def parse_manifest(
    profile: ManifestProfile,
    text: str,
    environment: ManifestEnvironment,
    *,
    source: Path | None = None,
) -> dict[str, ManifestEntry]:
    """Parse and validate one manifest, enforcing compatibility but not importing exception types."""
    label = str(source) if source is not None else "manifest"
    header: dict[str, str] = {}
    entries: dict[str, ManifestEntry] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        if line.startswith("# ") and ": " in line:
            key, value = line[2:].split(": ", maxsplit=1)
            header[key] = value
            continue
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"{label}:{line_number}: expected '<phase>\\t<exception>\\t<node ID>'")
        phase, exception, nodeid = fields
        if phase not in _ALLOWED_PHASES:
            raise ValueError(f"{label}:{line_number}: unsupported phase: {phase!r}")
        if not exception or not nodeid:
            raise ValueError(f"{label}:{line_number}: exception identity and node ID must be non-empty")
        if nodeid in entries:
            raise ValueError(f"{label}:{line_number}: duplicate node ID: {nodeid}")
        entries[nodeid] = ManifestEntry(cast(Literal["setup", "call"], phase), exception, nodeid)

    required = {
        "known-failure-schema": _SCHEMA_VERSION,
        "profile": profile.name,
        "python": environment.python,
        "pytorch": environment.pytorch,
        "os": environment.os,
        "arch": environment.arch,
        "seed-scheme": _SEED_SCHEME,
    }
    for key, expected in required.items():
        observed = header.get(key)
        if observed != expected:
            display = "Python" if key == "python" else key
            raise ValueError(f"{label}: {display} mismatch: expected {expected!r}, found {observed!r}")

    for key, expected in {
        "pytest": environment.pytest,
        "numpy": environment.numpy,
        "pillow": environment.pillow,
    }.items():
        observed = header.get(key)
        if observed != expected:
            warnings.warn(
                f"{label}: {key} provenance differs: recorded {observed!r}, running {expected!r}", stacklevel=2
            )
    return entries


def seed_test_rng(nodeid: str) -> int:
    """Derive a stable per-node seed without mutating any process-global RNG."""
    return int.from_bytes(hashlib.sha256(f"v1\0{nodeid}".encode()).digest()[:4], "big")


def _exception_name(exception_type: type[BaseException]) -> str:
    if exception_type.__module__ == "builtins":
        return exception_type.__name__
    return f"{exception_type.__module__}.{exception_type.__qualname__}"


def _canonical_exception_identity(exception_type: type[BaseException]) -> str:
    return f"{exception_type.__module__}.{exception_type.__qualname__}"


def _resolve_exception_type(exception_name: str) -> type[BaseException] | None:
    exception_type = _EXCEPTION_TYPES.get(exception_name, getattr(builtins, exception_name, None))
    if isinstance(exception_type, type) and issubclass(exception_type, BaseException):
        return cast(type[BaseException], exception_type)
    if not exception_name.startswith(_ALLOWED_EXCEPTION_MODULE_PREFIXES):
        return None

    parts = exception_name.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:split_at]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[split_at:]:
            value = getattr(value, attribute, None)
            if value is None:
                break
        if isinstance(value, type) and issubclass(value, BaseException):
            return cast(type[BaseException], value)
    return None


class FailureRecorder:
    """Record setup- and call-phase failures as a sorted half-precision manifest."""

    def __init__(self, dtype: str, output_path: Path) -> None:
        self.dtype = dtype
        self.output_path = output_path
        self.failures: dict[str, type[BaseException]] = {}

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
        """Capture failures that are not already handled by an xfail marker."""
        outcome = yield
        report = outcome.get_result()
        if (
            report.when in {"setup", "call"}
            and report.outcome == "failed"
            and getattr(report, "wasxfail", None) is None
            and call.excinfo is not None
        ):
            self.failures.setdefault(report.nodeid, call.excinfo.type)

    def pytest_sessionfinish(self) -> None:
        """Write the complete sorted manifest even though recorded failures make pytest fail."""
        lines = [
            f"# Recorded Linux CPU {self.dtype} failures.",
            "# Generated by --record-half-precision-failures; do not edit counts by hand.",
            *(
                f"{_exception_name(exception_type)}\t{nodeid}"
                for nodeid, exception_type in sorted(self.failures.items())
            ),
            "",
        ]
        self.output_path.write_text("\n".join(lines), encoding="utf-8")


class KnownFailureTracker:
    """Require every manifest entry to reproduce its exact phase and exception identity."""

    def __init__(
        self,
        profile_or_failures: ManifestProfile | dict[str, type[BaseException]],
        manifest_path_or_paths: Path | Sequence[Path],
        *,
        mode: Literal["focus", "complete"] = "focus",
        selectors: Sequence[str] | None = None,
        rootpath: Path | None = None,
    ) -> None:
        self.mode = mode
        self.selectors = tuple(selectors or ())
        self.rootpath = rootpath
        self.reason = _XFAIL_REASON_PREFIX
        self._collection_failed = False
        self._matched: set[str] = set()
        self._invalid: set[str] = set()
        self._finished_with_unexpected_outcome: set[str] = set()
        self._observed: dict[str, list[str]] = {}
        self._legacy_exception_types: dict[str, type[BaseException]] = {}

        if isinstance(profile_or_failures, ManifestProfile):
            self.profile = profile_or_failures
            manifest_path = cast(Path, manifest_path_or_paths)
            self.manifest_paths = (manifest_path,)
            entries = parse_manifest(
                self.profile,
                manifest_path.read_text(encoding="utf-8"),
                current_environment(),
                source=manifest_path,
            )
            if self.mode == "focus" and self.selectors:
                entries = {
                    nodeid: entry
                    for nodeid, entry in entries.items()
                    if any(_matches_selector(nodeid, selector, self.rootpath) for selector in self.selectors)
                }
            self._expected_entries = entries
        else:
            self.profile = None
            self._legacy_exception_types = profile_or_failures.copy()
            self._expected_entries = {
                nodeid: ManifestEntry("call", _canonical_exception_identity(exception_type), nodeid)
                for nodeid, exception_type in profile_or_failures.items()
            }
            self.manifest_paths = tuple(cast(Sequence[Path], manifest_path_or_paths))
        self.pending = set(self._expected_entries)

    @pytest.hookimpl(tryfirst=True)
    def pytest_collectreport(self, report: Any) -> None:
        """Remember collection failure so stale-manifest advice cannot mask it."""
        if report.failed:
            self._collection_failed = True

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, items: Sequence[Any]) -> None:
        """Apply the manifest marker after collection without inferring configuration from node IDs."""
        if self.profile is None:
            return
        for item in items:
            if item.nodeid in self._expected_entries:
                item.add_marker(pytest.mark.xfail(reason=self.reason, strict=True))

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
        """Observe all phases; pytest's xfail result alone is not the exact-outcome authority."""
        outcome = yield
        report = outcome.get_result()
        if report.nodeid not in self.pending:
            return
        wasxfail = getattr(report, "wasxfail", None)
        expected = self._expected_entries[report.nodeid]
        exception_type = call.excinfo.type if call.excinfo is not None else None

        if report.when in {"setup", "call"} and report.outcome == "skipped" and wasxfail == self.reason:
            if exception_type is None:
                observed = f"{report.when} xfail without an exception"
            else:
                identity = _canonical_exception_identity(exception_type)
                observed = f"{report.when} {identity}"
                legacy_match = self.profile is None and exception_type is self._legacy_exception_types[report.nodeid]
                exact_match = report.when == expected.phase and identity == expected.exception
                if legacy_match or exact_match:
                    self._matched.add(report.nodeid)
                    return
            self._observed.setdefault(report.nodeid, []).append(observed)
            self._invalid.add(report.nodeid)
        elif report.when == "teardown" and report.outcome != "passed":
            self._observed.setdefault(report.nodeid, []).append(f"teardown {report.outcome}")
            self._invalid.add(report.nodeid)
        elif report.outcome == "skipped":
            category = "competing xfail" if wasxfail is not None else "skip"
            self._observed.setdefault(report.nodeid, []).append(f"{report.when} {category}")
            self._invalid.add(report.nodeid)
        elif report.outcome == "failed" or (wasxfail is not None and report.outcome != "passed"):
            identity = _canonical_exception_identity(exception_type) if exception_type is not None else "strict XPASS"
            self._observed.setdefault(report.nodeid, []).append(f"{report.when} {identity}")
            self._invalid.add(report.nodeid)

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        """Accept an entry only after all setup, call, and teardown reports are known."""
        if nodeid in self._matched and nodeid not in self._invalid:
            self.pending.remove(nodeid)
        elif nodeid in self.pending:
            self._finished_with_unexpected_outcome.add(nodeid)
        self._matched.discard(nodeid)
        self._invalid.discard(nodeid)

    def pytest_deselected(self, items: Sequence[Any]) -> None:
        """Exclude locally deselected entries from outcome validation."""
        if self.mode == "focus":
            for item in items:
                self.pending.discard(item.nodeid)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Fail the session when a manifest entry was skipped or handled by another xfail."""
        if self.pending and not self._collection_failed and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        """Report manifest entries that did not produce their required xfail outcome."""
        if not self.pending or self._collection_failed:
            return
        count = len(self.pending)
        noun = "failure" if count == 1 else "failures"
        manifests = ", ".join(str(path) for path in self.manifest_paths)
        verb = "needs" if count == 1 else "need"
        terminalreporter.write_line(f"ERROR: {count} known half-precision {noun} {verb} updates in {manifests}:")
        unexpected = self.pending & self._finished_with_unexpected_outcome
        not_run = self.pending - unexpected
        if unexpected:
            terminalreporter.write_line("Ran with an unexpected outcome:")
            for nodeid in sorted(unexpected):
                expected = self._expected_entries[nodeid]
                terminalreporter.write_line(f"{nodeid}: expected {expected.phase} {expected.exception}")
                for observed in self._observed.get(nodeid, ["pass or incomplete lifecycle"]):
                    terminalreporter.write_line(f"  observed {observed}")
                terminalreporter.write_line(
                    f"  remove or update: {expected.phase}\t{expected.exception}\t{expected.nodeid}"
                )
        if not_run:
            terminalreporter.write_line("Did not run; rerun without selection or remove/update a stale manifest line:")
            for nodeid in sorted(not_run):
                terminalreporter.write_line(nodeid)


def load_known_failures(dtype: str, manifest_dir: Path | None = None) -> dict[str, type[BaseException]]:
    """Load exact test node IDs and their expected exception types for a CPU half dtype."""
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError(f"unsupported half-precision dtype: {dtype}")

    path = (manifest_dir or _MANIFEST_DIR) / f"cpu_{dtype}.txt"
    failures: dict[str, type[BaseException]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        try:
            exception_name, nodeid = line.split("\t", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: expected '<exception>\\t<node ID>'") from error
        exception_type = _resolve_exception_type(exception_name)
        if exception_type is None:
            raise ValueError(f"{path}:{line_number}: unknown exception type: {exception_name}")
        other_dtype = "bfloat16" if dtype == "float16" else "float16"
        if re.search(rf"(?<![A-Za-z]){other_dtype}(?![A-Za-z0-9])", nodeid) is not None:
            raise ValueError(f"{path}:{line_number}: node ID does not select {dtype}: {nodeid}")
        if nodeid in failures:
            raise ValueError(f"{path}:{line_number}: duplicate node ID: {nodeid}")
        failures[nodeid] = cast(type[BaseException], exception_type)
    return failures


def _normalize_selector(selector: str, rootpath: Path | None = None) -> str:
    path, separator, node = selector.partition("::")
    selector_path = Path(path)
    if selector_path.is_absolute() and rootpath is not None:
        try:
            selector_path = selector_path.relative_to(rootpath)
        except ValueError:
            pass
    normalized_path = selector_path.as_posix().removeprefix("./").rstrip("/")
    return f"{normalized_path}::{node}" if separator else normalized_path


def _matches_selector(nodeid: str, selector: str, rootpath: Path | None = None) -> bool:
    selector = _normalize_selector(selector, rootpath)
    node_path = nodeid.partition("::")[0]
    if "::" in selector:
        return nodeid == selector or nodeid.startswith((f"{selector}[", f"{selector}::"))
    return node_path == selector or node_path.startswith(f"{selector}/")


def mark_known_failures(
    items: Sequence[Any],
    dtypes: Sequence[str],
    manifest_dir: Path | None = None,
    *,
    selectors: Sequence[str] | None = None,
    rootpath: Path | None = None,
) -> KnownFailureTracker:
    """Strictly xfail the complete known-failure set for selected CPU half dtypes."""
    failures: dict[str, type[BaseException]] = {}
    manifest_paths: list[Path] = []
    for dtype in dtypes:
        dtype_failures = load_known_failures(dtype, manifest_dir)
        manifest_paths.append((manifest_dir or _MANIFEST_DIR) / f"cpu_{dtype}.txt")
        overlap = failures.keys() & dtype_failures.keys()
        if overlap:
            raise ValueError(f"duplicate node IDs across manifests: {sorted(overlap)!r}")
        failures.update(dtype_failures)

    if selectors:
        failures = {
            nodeid: exception_type
            for nodeid, exception_type in failures.items()
            if any(_matches_selector(nodeid, selector, rootpath) for selector in selectors)
        }

    collected = {item.nodeid for item in items}
    missing = failures.keys() - collected
    if missing:
        preview = "\n".join(sorted(missing)[:10])
        count = len(missing)
        noun = "failure was" if count == 1 else "failures were"
        manifests = ", ".join(str(path) for path in manifest_paths)
        raise ValueError(
            f"{count} known half-precision {noun} not collected from {manifests}; "
            f"remove or update stale manifest lines:\n{preview}"
        )

    tracker = KnownFailureTracker(failures, manifest_paths)
    for item in items:
        if exception_type := failures.get(item.nodeid):
            item.add_marker(pytest.mark.xfail(reason=tracker.reason, raises=exception_type, strict=True))
    return tracker

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

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import pytest
import torch

from testing.half_precision_eager_rng import AuditedEagerRngCall, eager_rng_calls_for_node

ISSUE_URL = "https://github.com/kornia/kornia/issues/4153"
_MANIFEST_DIR = Path(__file__).with_name("half_precision_xfails")
_XFAIL_REASON_PREFIX = f"Known Linux CPU half-precision failure tracked in {ISSUE_URL}"
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
    lines.extend(
        f"{entry.phase}\t{entry.exception}\t{entry.nodeid}" for entry in sorted(entries, key=lambda x: x.nodeid)
    )
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


def _canonical_exception_identity(exception_type: type[BaseException]) -> str:
    return f"{exception_type.__module__}.{exception_type.__qualname__}"


def validate_record_destination(output_path: Path, rootpath: Path) -> Path:
    """Resolve a candidate destination and reject checked-in or Git-tracked aliases."""
    root = rootpath.resolve()
    unresolved = output_path if output_path.is_absolute() else root / output_path
    destination = unresolved.resolve()
    manifest_dir = _MANIFEST_DIR.resolve()
    if destination == manifest_dir or destination.is_relative_to(manifest_dir):
        raise ValueError(f"record destination resolves inside the checked-in manifest directory: {destination}")
    if destination.is_relative_to(root):
        git = shutil.which("git")
        if git is None:
            raise OSError("cannot validate the candidate destination because Git is unavailable")
        repository = subprocess.run(  # noqa: S603 - all arguments are fixed or resolved local paths.
            [git, "-C", str(root), "rev-parse", "--show-toplevel"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if repository.returncode == 0 and Path(repository.stdout.strip()).resolve() == root:
            relative = destination.relative_to(root)
            tracked = subprocess.run(  # noqa: S603 - -- terminates options before the resolved relative path.
                [git, "-C", str(root), "ls-files", "--error-unmatch", "--", relative.as_posix()],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if tracked.returncode == 0:
                raise ValueError(f"record destination is Git-tracked: {destination}")
            if tracked.returncode != 1:
                raise OSError(f"could not determine whether record destination is Git-tracked: {destination}")
    if not destination.parent.is_dir():
        raise ValueError(f"record destination parent does not exist: {destination.parent}")
    return destination


def _previous_manifest_entries(profile: ManifestProfile) -> dict[str, ManifestEntry]:
    """Load the checked-in v1 entries used to categorize a candidate delta."""
    text = profile.manifest_path.read_text(encoding="utf-8")
    return parse_manifest(profile, text, current_environment(), source=profile.manifest_path)


class FailureRecorder:
    """Write a candidate only after proving that the full pytest session completed."""

    def __init__(
        self,
        profile: ManifestProfile,
        output_path: Path,
        *,
        rootpath: Path,
        previous_entries: Mapping[str, ManifestEntry] | None = None,
    ) -> None:
        self.profile = profile
        self.output_path = validate_record_destination(output_path, rootpath)
        self.previous_entries = (
            dict(previous_entries) if previous_entries is not None else _previous_manifest_entries(profile)
        )
        self.entries: dict[str, ManifestEntry] = {}
        self.collected: set[str] = set()
        self.finished: set[str] = set()
        self.reports: dict[str, list[tuple[str, str, str | None]]] = defaultdict(list)
        self.collection_finished = False
        self.collection_failed = False
        self.interrupted = False
        self.abort_reasons: list[str] = []
        self.eager_blockers: dict[str, tuple[AuditedEagerRngCall, ...]] = {}
        self.categories: dict[str, str] = {}
        self.written = False

    @pytest.hookimpl(tryfirst=True)
    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.collection_failed = True

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collection_finished = True
        self.collected = {item.nodeid for item in session.items}

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        self.finished.add(nodeid)

    def pytest_keyboard_interrupt(self) -> None:
        self.interrupted = True

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
        """Capture exact unhandled setup/call failures and all lifecycle outcomes."""
        outcome = yield
        report = outcome.get_result()
        wasxfail = getattr(report, "wasxfail", None)
        self.reports[report.nodeid].append((report.when, report.outcome, wasxfail))
        if (
            report.when in _ALLOWED_PHASES
            and report.outcome == "failed"
            and wasxfail is None
            and call.excinfo is not None
        ):
            entry = ManifestEntry(report.when, _canonical_exception_identity(call.excinfo.type), report.nodeid)
            existing = self.entries.setdefault(report.nodeid, entry)
            if existing != entry:
                self.abort_reasons.append(f"multiple representable failures for {report.nodeid}")
            blockers = eager_rng_calls_for_node(report.nodeid)
            if blockers:
                self.eager_blockers[report.nodeid] = blockers
        if report.when == "teardown" and report.outcome != "passed":
            if report.nodeid in self.entries or report.nodeid in self.previous_entries:
                self.abort_reasons.append(f"unrepresentable teardown failure for {report.nodeid}")

    def _check_completeness(self, session: pytest.Session) -> None:
        if session.config.option.collectonly:
            self.abort_reasons.append("collect-only execution cannot produce a candidate")
        if not self.collection_finished or self.collection_failed:
            self.abort_reasons.append("collection did not finish successfully")
        if not self.collected:
            self.abort_reasons.append("the canonical suite collected no tests")
        unfinished = self.collected - self.finished
        if unfinished:
            self.abort_reasons.append(f"{len(unfinished)} collected test(s) never reached log finish")
        if self.interrupted:
            self.abort_reasons.append("execution was interrupted")
        if session.exitstatus not in {pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED}:
            self.abort_reasons.append(f"pytest stopped with incomplete exit status {session.exitstatus}")
        for nodeid, blockers in sorted(self.eager_blockers.items()):
            sites = ", ".join(f"{entry.call.path}:{entry.call.line} ({entry.call.name})" for entry in blockers)
            self.abort_reasons.append(
                f"eager RNG feeds candidate node {nodeid}: {sites}; move the draw into the test or a seeded fixture"
            )

    def _categorize_previous_entries(self) -> None:
        for nodeid, previous in self.previous_entries.items():
            candidate = self.entries.get(nodeid)
            if candidate is not None:
                self.categories[nodeid] = "reproduced" if candidate == previous else "changed-outcome"
            elif nodeid not in self.collected:
                self.categories[nodeid] = "deleted-or-renamed"
            elif nodeid not in self.finished:
                self.categories[nodeid] = "incomplete"
            elif any(wasxfail is not None for _, _, wasxfail in self.reports[nodeid]):
                self.categories[nodeid] = "competing-xfail"
            elif any(outcome == "skipped" for _, outcome, _ in self.reports[nodeid]):
                self.categories[nodeid] = "skipped"
            else:
                self.categories[nodeid] = "passed"

    def _write_candidate(self) -> None:
        generation_command = (
            f"pytest tests/ --record-known-failures=<candidate> --known-failure-profile={self.profile.name}"
        )
        contents = serialize_manifest(
            self.profile, tuple(self.entries.values()), current_environment(), generation_command
        )
        parse_manifest(self.profile, contents, current_environment(), source=self.output_path)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_path.parent,
                prefix=f".{self.output_path.name}.",
                delete=False,
            ) as temporary:
                temporary.write(contents)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, self.output_path)
            self.written = True
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Serialize atomically only when completeness and lifecycle checks succeed."""
        self._check_completeness(session)
        self._categorize_previous_entries()
        if self.abort_reasons:
            return
        try:
            self._write_candidate()
        except Exception as error:  # noqa: BLE001 - every writer failure must preserve the previous candidate.
            self.abort_reasons.append(f"candidate serialization failed: {error}")

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        """Print a receipt or an actionable abort report independently of pytest's test count."""
        if self.abort_reasons:
            terminalreporter.write_line(
                f"known-failure candidate ABORTED for {self.profile.name}; preserved {self.output_path}", red=True
            )
            for reason in dict.fromkeys(self.abort_reasons):
                terminalreporter.write_line(f"  {reason}")
            return
        if not self.written:
            return
        terminalreporter.write_line(
            f"known-failure candidate complete for {self.profile.name}: "
            f"{len(self.entries)} entries at {self.output_path}"
        )
        for nodeid, category in sorted(self.categories.items()):
            if category != "reproduced":
                terminalreporter.write_line(f"OLD {category}: {nodeid}")
        for nodeid in sorted(set(self.entries) - set(self.previous_entries)):
            entry = self.entries[nodeid]
            terminalreporter.write_line(f"NEW: {entry.phase}\t{entry.exception}\t{entry.nodeid}")


class KnownFailureTracker:
    """Require every manifest entry to reproduce its exact phase and exception identity."""

    def __init__(
        self,
        profile: ManifestProfile,
        manifest_path: Path,
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
        self.profile = profile
        self.manifest_paths = (manifest_path,)
        entries = parse_manifest(
            profile,
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
        self.pending = set(self._expected_entries)

    @pytest.hookimpl(tryfirst=True)
    def pytest_collectreport(self, report: Any) -> None:
        """Remember collection failure so stale-manifest advice cannot mask it."""
        if report.failed:
            self._collection_failed = True

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, items: Sequence[Any]) -> None:
        """Apply the manifest marker after collection without inferring configuration from node IDs."""
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
                exact_match = report.when == expected.phase and identity == expected.exception
                if exact_match:
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

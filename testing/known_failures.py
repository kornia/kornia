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

from pathlib import Path
from typing import Any, Sequence

import pytest

ISSUE_URL = "https://github.com/kornia/kornia/issues/4159"
_MANIFEST_DIR = Path(__file__).with_name("known_failure_xfails")
_EXCEPTION_TYPES: dict[str, type[BaseException]] = {
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "pytest.fail.Exception": pytest.fail.Exception,
}


def manifest_path(device: str, dtype: str, manifest_dir: Path | None = None) -> Path:
    """Return the known-failure manifest path for one test configuration."""
    return (manifest_dir or _MANIFEST_DIR) / f"{device}_{dtype}.txt"


class KnownFailureTracker:
    """Require every manifest entry to finish under its exact strict xfail."""

    def __init__(self, failures: dict[str, type[BaseException]], path: Path, reason: str) -> None:
        self.pending = set(failures)
        self._expected_exceptions = failures.copy()
        self.path = path
        self.reason = reason
        self._matched: set[str] = set()
        self._invalid: set[str] = set()
        self._collect_only = False

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
        """Record only call-phase xfails caused by the exact recorded exception type."""
        outcome = yield
        report = outcome.get_result()
        if report.nodeid not in self.pending:
            return
        wasxfail = getattr(report, "wasxfail", None)
        if (
            report.when == "call"
            and report.outcome == "skipped"
            and wasxfail == self.reason
            and call.excinfo is not None
            and type(call.excinfo.value) is self._expected_exceptions[report.nodeid]
        ):
            self._matched.add(report.nodeid)
        elif report.outcome in {"skipped", "failed"} or wasxfail is not None:
            self._invalid.add(report.nodeid)

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        """Accept an entry only after setup, call, and teardown reports are known."""
        if nodeid in self._matched and nodeid not in self._invalid:
            self.pending.remove(nodeid)
        self._matched.discard(nodeid)
        self._invalid.discard(nodeid)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Fail when a recorded test passed, skipped, or failed in a different way."""
        self._collect_only = session.config.getoption("collectonly")
        if self._collect_only:
            return
        if self.pending and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        """Explain how to update entries whose recorded outcome changed."""
        if not self.pending or self._collect_only:
            return
        count = len(self.pending)
        noun = "failure" if count == 1 else "failures"
        terminalreporter.write_line(f"ERROR: {count} known {noun} did not reproduce exactly; update {self.path}:")
        for nodeid in sorted(self.pending):
            terminalreporter.write_line(nodeid)


def load_known_failures(
    device: str, dtype: str, manifest_dir: Path | None = None
) -> tuple[dict[str, type[BaseException]], Path]:
    """Load exact node IDs and exception types for one device/dtype configuration."""
    path = manifest_path(device, dtype, manifest_dir)
    failures: dict[str, type[BaseException]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        try:
            exception_name, nodeid = line.split("\t", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: expected '<exception>\\t<node ID>'") from error
        if exception_name not in _EXCEPTION_TYPES:
            raise ValueError(f"{path}:{line_number}: unknown exception type: {exception_name}")
        parameter_ids = nodeid.rsplit("[", maxsplit=1)[-1].removesuffix("]").split("-")
        if device not in parameter_ids:
            raise ValueError(f"{path}:{line_number}: node ID does not select {device}: {nodeid}")
        if nodeid in failures:
            raise ValueError(f"{path}:{line_number}: duplicate node ID: {nodeid}")
        failures[nodeid] = _EXCEPTION_TYPES[exception_name]
    return failures, path


def mark_known_failures(
    items: Sequence[Any], device: str, dtype: str, manifest_dir: Path | None = None
) -> KnownFailureTracker:
    """Strictly xfail the complete known-failure set for one test configuration."""
    failures, path = load_known_failures(device, dtype, manifest_dir)
    collected = {item.nodeid for item in items}
    missing = failures.keys() - collected
    if missing:
        preview = "\n".join(sorted(missing)[:10])
        raise ValueError(f"{len(missing)} failures from {path} were not collected:\n{preview}")

    reason = f"Known {device}/{dtype} failure tracked in {ISSUE_URL}"
    tracker = KnownFailureTracker(failures, path, reason)
    for item in items:
        if exception_type := failures.get(item.nodeid):
            item.add_marker(pytest.mark.xfail(reason=reason, raises=exception_type, strict=True))
    return tracker

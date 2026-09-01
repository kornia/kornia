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

import secrets
from pathlib import Path
from typing import Any, Sequence

import pytest

from kornia.core.exceptions import BaseError

ISSUE_URL = "https://github.com/kornia/kornia/issues/4153"
_MANIFEST_DIR = Path(__file__).with_name("half_precision_xfails")
_XFAIL_REASON_PREFIX = f"Known Linux CPU half-precision failure tracked in {ISSUE_URL}"
_EXCEPTION_TYPES: dict[str, type[BaseException]] = {
    "AssertionError": AssertionError,
    "KeyError": KeyError,
    "NotImplementedError": NotImplementedError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "kornia.core.exceptions.BaseError": BaseError,
}


class KnownFailureTracker:
    """Require every manifest entry to finish under the manifest-specific xfail marker."""

    def __init__(self, failures: dict[str, type[BaseException]]) -> None:
        self.pending = set(failures)
        self._expected_exceptions = failures.copy()
        self.reason = f"{_XFAIL_REASON_PREFIX} [tracker={secrets.token_hex(8)}]"
        self._matched: set[str] = set()
        self._invalid: set[str] = set()

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
        """Record only call-phase xfails caused by the manifest's expected exception."""
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
            and isinstance(call.excinfo.value, self._expected_exceptions[report.nodeid])
        ):
            self._matched.add(report.nodeid)
        elif report.outcome in {"skipped", "failed"} or wasxfail is not None:
            self._invalid.add(report.nodeid)

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        """Accept an entry only after all setup, call, and teardown reports are known."""
        if nodeid in self._matched and nodeid not in self._invalid:
            self.pending.remove(nodeid)
        self._matched.discard(nodeid)
        self._invalid.discard(nodeid)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Fail the session when a manifest entry was skipped or handled by another xfail."""
        if self.pending and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        """Report manifest entries that did not produce their required xfail outcome."""
        if not self.pending:
            return
        count = len(self.pending)
        noun = "failure" if count == 1 else "failures"
        terminalreporter.write_line(
            f"ERROR: {count} known half-precision {noun} did not finish with the manifest-specific xfail:"
        )
        for nodeid in sorted(self.pending):
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
        if exception_name not in _EXCEPTION_TYPES:
            raise ValueError(f"{path}:{line_number}: unknown exception type: {exception_name}")
        if f"cpu-{dtype}" not in nodeid:
            raise ValueError(f"{path}:{line_number}: node ID does not select cpu-{dtype}: {nodeid}")
        if nodeid in failures:
            raise ValueError(f"{path}:{line_number}: duplicate node ID: {nodeid}")
        failures[nodeid] = _EXCEPTION_TYPES[exception_name]
    return failures


def mark_known_failures(
    items: Sequence[Any], dtypes: Sequence[str], manifest_dir: Path | None = None
) -> KnownFailureTracker:
    """Strictly xfail the complete known-failure set for selected CPU half dtypes."""
    failures: dict[str, type[BaseException]] = {}
    for dtype in dtypes:
        dtype_failures = load_known_failures(dtype, manifest_dir)
        overlap = failures.keys() & dtype_failures.keys()
        if overlap:
            raise ValueError(f"duplicate node IDs across manifests: {sorted(overlap)!r}")
        failures.update(dtype_failures)

    collected = {item.nodeid for item in items}
    missing = failures.keys() - collected
    if missing:
        preview = "\n".join(sorted(missing)[:10])
        raise ValueError(f"{len(missing)} known half-precision failures were not collected:\n{preview}")

    tracker = KnownFailureTracker(failures)
    for item in items:
        if exception_type := failures.get(item.nodeid):
            item.add_marker(pytest.mark.xfail(reason=tracker.reason, raises=exception_type, strict=True))
    return tracker

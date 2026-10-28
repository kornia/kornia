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

"""The support statement of the export-support page is the promise the conformance tests check (#5819)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from testing.conformance import import_survey_module
from testing.conformance_promise import (
    CI_DTYPE,
    CI_OS,
    CI_PYTHON,
    CI_RUNS_PER_DAY,
    CI_TORCH_PULL_REQUEST,
    CI_TORCH_SCHEDULED,
    COMPILE,
    ENFORCED_TORCH_RELEASES,
    EXCLUDED,
    ONNX,
    ONNX_OPSET,
    PROMISES,
    known_failures,
    manifest_path,
    promised_cases,
    release,
)

REPO = Path(__file__).parents[1]

_spec = importlib.util.spec_from_file_location("generate_export_support", REPO / "docs" / "generate_export_support.py")
generate_export_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_export_support)


def _statement() -> tuple[str, str]:
    """The rendered page, and its statement section with the line breaks of the source joined."""
    page = generate_export_support.render_page(generate_export_support.load_data())
    statement = page.split(".. _export-support-promise:", 1)[1].split("\nThe survey\n----------\n", 1)[0]
    return page, " ".join(statement.split())


def _except_known(promise, tests: tuple[str, ...]) -> str:
    counts = []
    for enforced in sorted(ENFORCED_TORCH_RELEASES, key=lambda r: tuple(map(int, r.split(".")))):
        n = sum(entry.test in tests for entry in known_failures(promise, manifest_path(promise, enforced)))
        counts.append(f"{n or 'none'} on torch {enforced}")
    return f"except the known failures listed below ({', '.join(counts)})"


def test_statement_renders_the_promise_the_tests_check() -> None:
    # The generator loads testing/conformance_promise.py from its own checkout: the same file as this import.
    loaded = Path(generate_export_support.promise_module().__file__).resolve()
    assert loaded == REPO / "testing" / "conformance_promise.py"
    page, statement = _statement()
    for promise in PROMISES:
        assert f"The {len(promised_cases(promise))} promised configurations" in statement
    # every headline sentence names the known failures of each enforced release, counted from the manifests
    assert f"on torch 2.9 and later, {_except_known(ONNX, ('test_export_matches_eager',))}." in statement
    assert f"fails loudly, {_except_known(ONNX, ('test_legacy_export_is_not_wrong',))}." in statement
    assert f"tolerance, {_except_known(COMPILE, COMPILE.case_tests)}." in statement
    assert f"at opset {ONNX_OPSET} " in statement
    for promise in PROMISES:
        for reason in EXCLUDED[promise.field].values():
            assert " ".join(generate_export_support._issue_links(reason).split()) in statement
    issues = {
        entry.issue
        for promise in PROMISES
        for enforced in ENFORCED_TORCH_RELEASES
        for entry in known_failures(promise, manifest_path(promise, enforced))
    }
    for issue in issues:
        assert generate_export_support._issue_links(issue) in statement
    # one marker per promised configuration of the survey tables, plus the statement's own and the legend's
    marker = ":kornia-promised:`promised`"
    assert page.count(marker) == sum(len(promised_cases(promise)) for promise in PROMISES) + 2


def test_statement_states_where_ci_checks_it() -> None:
    _, statement = _statement()
    times = {1: "once a day", 2: "twice a day"}[CI_RUNS_PER_DAY]
    assert (
        f"On CPU in {CI_DTYPE}, on Linux with Python {CI_PYTHON}: torch {' and '.join(CI_TORCH_SCHEDULED)} {times} on"
        f" ``main``, and torch {' and '.join(CI_TORCH_PULL_REQUEST)} on every pull request"
    ) in statement
    assert ENFORCED_TORCH_RELEASES == tuple(release(version) for version in CI_TORCH_SCHEDULED)


@pytest.mark.parametrize(
    ("workflow", "torch_versions"),
    [("scheduled_test_cpu.yml", CI_TORCH_SCHEDULED), ("pr_test_cpu.yml", CI_TORCH_PULL_REQUEST)],
    ids=["scheduled", "pull-request"],
)
def test_ci_facts_match_the_conformance_jobs(workflow: str, torch_versions: tuple[str, ...]) -> None:
    """The page's CI facts are those of the workflows' onnx-conformance and compile-conformance jobs."""
    import yaml

    config = yaml.safe_load((REPO / ".github" / "workflows" / workflow).read_text(encoding="utf-8"))
    for promise in PROMISES:
        job = config["jobs"][f"{promise.field}-conformance"]["with"]
        assert job["test-path"] == promise.test_file
        assert json.loads(job["pytorch-version"]) == list(torch_versions)
        assert json.loads(job["python-version"]) == [CI_PYTHON]
        assert job["os"] == CI_OS
        assert job["pytorch-dtype"] == CI_DTYPE
    if workflow == "scheduled_test_cpu.yml":
        triggers = config.get("on", config.get(True))  # YAML 1.1 reads the key `on` as True
        daily = [entry["cron"] for entry in triggers["schedule"] if entry["cron"].split()[2:] == ["*", "*", "*"]]
        assert len(daily) == CI_RUNS_PER_DAY, daily


def test_onnx_opset_is_the_one_ci_and_to_onnx_use() -> None:
    from kornia.core.mixin.onnx import _ONNX_EXPORT_OPSET

    assert import_survey_module("harness").OPSET == ONNX_OPSET == _ONNX_EXPORT_OPSET

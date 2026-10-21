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
from pathlib import Path

from testing.conformance_promise import (
    ENFORCED_TORCH_RELEASES,
    EXCLUDED,
    PROMISES,
    known_failures,
    manifest_path,
    promised_cases,
)

_spec = importlib.util.spec_from_file_location(
    "generate_export_support", Path(__file__).parents[1] / "docs" / "generate_export_support.py"
)
generate_export_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_export_support)


def test_statement_renders_the_promise_the_tests_check() -> None:
    # The generator loads testing/conformance_promise.py from its own checkout: the same file as this import.
    assert Path(generate_export_support.promise_module().__file__).resolve() == (
        Path(__file__).parents[1] / "testing" / "conformance_promise.py"
    )
    page = generate_export_support.render_page(generate_export_support.load_data())
    statement = page.split(".. _export-support-promise:", 1)[1].split("\nThe survey\n----------\n", 1)[0]
    for promise in PROMISES:
        assert f"The {len(promised_cases(promise))} promised configurations" in statement
    assert f"torch {' and '.join(ENFORCED_TORCH_RELEASES)}," in statement
    for promise in PROMISES:
        for reason in EXCLUDED[promise.field].values():
            assert generate_export_support._issue_links(reason) in statement
    issues = {
        entry.issue
        for promise in PROMISES
        for release in ENFORCED_TORCH_RELEASES
        for entry in known_failures(promise, manifest_path(promise, release))
    }
    for issue in issues:
        assert generate_export_support._issue_links(issue) in statement
    # one marker per promised configuration of the survey tables, plus the statement's own and the legend's
    marker = ":kornia-promised:`promised`"
    assert page.count(marker) == sum(len(promised_cases(promise)) for promise in PROMISES) + 2

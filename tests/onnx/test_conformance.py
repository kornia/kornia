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

"""ONNX conformance of the promised operators (#5819).

Every promised case (see ``testing/onnx_conformance.py``) must export with the ``torch.export``-based exporter
(``torch.onnx.export(..., dynamo=True)``) and match eager execution in onnxruntime: the same dtype, and values within
the case's tolerance. A legacy (TorchScript) export that succeeds must match too; one that fails loudly passes.

Selected with ``--onnx-conformance=<package>[,<package>...]`` or ``all`` (env ``KORNIA_TEST_ONNX_CONFORMANCE``);
deselected otherwise. Known failures are strict xfails in
``testing/known_failure_xfails/onnx_conformance_torch<major>.<minor>.txt``: delete an entry when its case passes.
"""

from __future__ import annotations

import logging
import warnings

import pytest
import torch

from testing.onnx_conformance import (
    LEGACY_PASSES,
    NOT_CACHED,
    OPTION,
    describe,
    import_survey_module,
    parse_selection,
    promised_cases,
    registry_cases,
    run_case,
    selected_cases,
)

# Every case runs on CPU tensors and is compared in onnxruntime as numpy, whatever ``--device`` says.
pytestmark = [pytest.mark.device_agnostic, pytest.mark.filterwarnings("ignore")]

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")  # the dynamo exporter hard-requires it


def _selection(config: pytest.Config) -> tuple[str, ...]:
    return parse_selection(config.getoption(OPTION))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    names = selected_cases(_selection(metafunc.config))
    # Without a selection the root conftest deselects the placeholder; with one that has no promised case it skips.
    metafunc.parametrize("case_name", names or [pytest.param(None, id="unselected")])


def _promised_case(case_name: str | None) -> dict:
    if case_name is None:
        pytest.skip("the selected packages have no promised case")
    case = registry_cases().get(case_name)
    if case is not None and case["skip"] and NOT_CACHED in case["skip"]:
        pytest.skip(case["skip"])
    if case is None or case["skip"] or not case["check"]:
        pytest.fail(f"{case_name!r} is promised but no registry case runs it; see test_promised_cases_are_registered")
    return case


@pytest.mark.onnx_conformance
def test_export_matches_eager(case_name: str | None) -> None:
    record = run_case(_promised_case(case_name), dynamo=True)
    assert record["status"] == "ok", describe(record)


@pytest.mark.onnx_conformance
def test_legacy_export_is_not_wrong(case_name: str | None) -> None:
    record = run_case(_promised_case(case_name), dynamo=False)
    assert record["status"] in LEGACY_PASSES, describe(record)


@pytest.mark.onnx_conformance
def test_promised_cases_are_registered(request: pytest.FixtureRequest) -> None:
    """Every promised case of the selected packages has a registry case that runs and is compared with eager."""
    cases = registry_cases()
    problems = []
    for name in selected_cases(_selection(request.config)):
        case = cases.get(name)
        if case is None:
            problems.append(f"{name}: no registry case of that name")
        elif case["skip"] and NOT_CACHED not in case["skip"]:
            problems.append(f"{name}: skipped ({case['skip']})")
        elif not case["check"]:
            problems.append(f"{name}: check=False, the output is not compared with eager")
    assert not problems, "promised cases the registries no longer run:\n" + "\n".join(problems)


def test_promised_set_size_is_pinned() -> None:
    # The promise is read from the snapshot, so a regeneration that measured a case as failing would quietly drop it.
    # Pinning the size makes that a reviewed change: 859 cases (651 operators) on the 2026-09-02 snapshot.
    promised = promised_cases()
    assert len(promised) == 859
    assert not any(name.startswith(("models.", "contrib.")) for name in promised)
    # The root conftest deselects or skips tests whose id contains these words, which would drop the case silently.
    assert not [name for name in promised if any(word in name.lower() for word in ("dynamo", "compile", "autocast"))]


def test_survey_harness_leaves_process_state_alone() -> None:
    """Importing the harness and a registry and running a case keep threads, logging, warnings and the RNG."""
    before = (
        torch.get_num_threads(),
        logging.root.manager.disable,
        list(warnings.filters),
        torch.random.get_rng_state(),
    )
    import_survey_module("harness")
    misc = import_survey_module("cases_misc")
    case = next(c for c in misc.CASES if c["name"] in promised_cases())
    records = [run_case(case, dynamo=dynamo) for dynamo in (True, False)]
    after = (
        torch.get_num_threads(),
        logging.root.manager.disable,
        list(warnings.filters),
        torch.random.get_rng_state(),
    )
    assert after[:3] == before[:3]
    assert torch.equal(after[3], before[3])
    assert all(record["status"] not in ("eager-fail", "skipped") for record in records), records

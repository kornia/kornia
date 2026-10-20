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

"""torch.compile conformance of the promised operators (#5819).

Every promised case (see ``testing/conformance.py``) must compile with ``torch.compile(..., fullgraph=True)`` and the
default inductor backend -- no graph break -- and match eager execution: the same dtype, and values within the case's
tolerance. One call per case, at the registry's input shapes, as the survey measures; a recompile at another shape is
not part of the promise.

Selected with ``--compile-conformance=<package>[,<package>...]`` or ``all`` (env ``KORNIA_TEST_COMPILE_CONFORMANCE``);
deselected otherwise. Known failures are strict xfails in
``testing/known_failure_xfails/compile_conformance_torch<major>.<minor>.txt``: delete an entry when its case passes. A
case that does not pass raises the exception ``testing.conformance.STATUS_EXCEPTIONS`` maps its status to
(``AssertionError`` for a wrong result, ``RuntimeError`` for a failed compile, ``ValueError`` for a failed eager
reference), which its manifest entry names. The test needs neither onnx nor onnxruntime.

The root conftest drops a test whose name or id contains "compile" or "dynamo" unless ``KORNIA_TEST_OPTIMIZER`` is
set, so the tests here are named after inductor; this test does not use ``KORNIA_TEST_OPTIMIZER``.
"""

from __future__ import annotations

import pytest

from testing.conformance import (
    COMPILE,
    describe,
    import_survey_module,
    parametrize_cases,
    pin_problems,
    process_state,
    promised_case,
    promised_cases,
    raise_for_status,
    registration_problems,
    run_compile_case,
    selection,
    state_changes,
)

# Every case runs on CPU tensors, whatever ``--device`` says. The registries and the probe call torch.manual_seed,
# which seeds CUDA and MPS too; testing.conformance forks only the CPU generator, so restore_torch_rng puts the
# accelerator generators back.
pytestmark = [pytest.mark.device_agnostic, pytest.mark.usefixtures("restore_torch_rng")]

# Dynamo and inductor warn about graphs they handle; the per-case tests keep the session output readable. Not
# module-wide: the process-state tests must see a filter the probe would add.
quiet = pytest.mark.filterwarnings("ignore")

# The compile promise of the 2026-09-02 snapshot: 832 cases of 637 operators.
PROMISED_COUNT = 832
PROMISED_DIGEST = "8461e5c612ed2c8aa696633f3dc9f4154bfcb8c1cc01f19a2ddb29610e9b188d"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    parametrize_cases(metafunc, COMPILE)


@pytest.mark.compile_conformance
@quiet
def test_inductor_matches_eager(case_name: str | None) -> None:
    raise_for_status(run_compile_case(promised_case(COMPILE, case_name)), ("ok",), "compile")


@pytest.mark.compile_conformance
@quiet
def test_promised_cases_are_registered(request: pytest.FixtureRequest) -> None:
    """Every promised case of the selected packages has a registry case that runs and is compared with eager."""
    problems = registration_problems(COMPILE, selection(request.config, COMPILE))
    assert not problems, "promised cases the registries no longer run:\n" + "\n".join(problems)


def test_promised_set_is_pinned() -> None:
    problems = pin_problems(COMPILE, PROMISED_COUNT, PROMISED_DIGEST)
    assert not problems, "\n".join(problems)


def test_survey_probe_import_leaves_process_state_alone() -> None:
    """Importing the compile probe (and with it the harness) and a registry changes nothing."""
    before = process_state()
    import_survey_module("probe_compile")
    import_survey_module("cases_misc")
    after = process_state()
    assert not state_changes(before, after), f"importing the survey changed {state_changes(before, after)}"
    assert after["filters"] == before["filters"]


@pytest.mark.compile_conformance
def test_survey_probe_leaves_process_state_alone() -> None:
    """Compiling a case leaves the RNG, torch's hooks and the dynamo and inductor configs; it adds only torch filters.

    Marked like the per-case tests: it needs inductor's C++ toolchain, which the conformance jobs have and not every
    leg of the ordinary suite does.
    """
    before = process_state()
    misc = import_survey_module("cases_misc")
    case = next(c for c in misc.CASES if c["name"] in promised_cases(COMPILE))
    record = run_compile_case(case)
    # A canary: this case compiles on every enforced torch, so an error of the probe itself cannot pass silently.
    assert record["compile"] == "ok", describe(record, "compile")
    changes = state_changes(before, process_state())
    assert not changes, f"compiling a case changed {changes}"

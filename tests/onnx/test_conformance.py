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

Every promised case (see ``testing/conformance.py``) must export with the ``torch.export``-based exporter
(``torch.onnx.export(..., dynamo=True)``) and match eager execution in onnxruntime: the same dtype, and values within
the case's tolerance. A legacy (TorchScript) export that succeeds must match too; one that fails loudly passes.

Selected with ``--onnx-conformance=<package>[,<package>...]`` or ``all`` (env ``KORNIA_TEST_ONNX_CONFORMANCE``);
deselected otherwise. Known failures are strict xfails in
``testing/known_failure_xfails/onnx_conformance_torch<major>.<minor>.txt``: delete an entry when its case passes. A
case that does not pass raises the exception ``testing.conformance.STATUS_EXCEPTIONS`` maps its status to, which its
manifest entry names.
"""

from __future__ import annotations

import pytest

from testing.conformance import (
    ENFORCED_TORCH_RELEASES,
    LEGACY_PASSES,
    ONNX,
    import_survey_module,
    load_known_failures,
    manifest_path,
    parametrize_cases,
    pin_problems,
    process_state,
    promised_case,
    promised_cases,
    raise_for_status,
    registration_problems,
    run_onnx_case,
    selection,
    state_changes,
)

# Every case runs on CPU tensors and is compared in onnxruntime as numpy, whatever ``--device`` says. The registries
# and the harness call torch.manual_seed, which seeds CUDA and MPS too; testing.conformance forks only the CPU
# generator, so restore_torch_rng puts the accelerator generators back.
pytestmark = [pytest.mark.device_agnostic, pytest.mark.usefixtures("restore_torch_rng")]

# The exporters warn at length about every graph; the per-case tests keep the session output readable. Not module-wide:
# test_survey_harness_leaves_process_state_alone must see a filter the harness would add.
quiet = pytest.mark.filterwarnings("ignore")

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")  # the dynamo exporter hard-requires it

# The promised set of the 2026-09-02 snapshot: 848 cases of 643 operators.
PROMISED_COUNT = 848
PROMISED_DIGEST = "7b8f869f2d8812be750269f7bed6d0b2db499f915c8ddf5c34fb76deb785912d"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    parametrize_cases(metafunc, ONNX)


@pytest.mark.onnx_conformance
@quiet
def test_export_matches_eager(case_name: str | None) -> None:
    raise_for_status(run_onnx_case(promised_case(ONNX, case_name), dynamo=True), ("ok",))


@pytest.mark.onnx_conformance
@quiet
def test_legacy_export_is_not_wrong(case_name: str | None) -> None:
    raise_for_status(run_onnx_case(promised_case(ONNX, case_name), dynamo=False), LEGACY_PASSES)


@pytest.mark.onnx_conformance
@quiet
def test_promised_cases_are_registered(request: pytest.FixtureRequest) -> None:
    """Every promised case of the selected packages has a registry case that runs and is compared with eager."""
    problems = registration_problems(ONNX, selection(request.config, ONNX))
    assert not problems, "promised cases the registries no longer run:\n" + "\n".join(problems)


def test_promised_set_is_pinned() -> None:
    problems = pin_problems(ONNX, PROMISED_COUNT, PROMISED_DIGEST)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("version", ENFORCED_TORCH_RELEASES)
def test_known_failure_manifest_names_promised_cases(version: str) -> None:
    """The manifest of every enforced torch release parses, not only the running torch's, which the run reads."""
    load_known_failures(ONNX, manifest_path(ONNX, version))


def test_survey_harness_leaves_process_state_alone() -> None:
    """Importing the harness and a registry changes nothing; running a case leaves only torch's own warning filters."""
    before = process_state()
    import_survey_module("harness")
    misc = import_survey_module("cases_misc")
    after_import = process_state()
    assert not state_changes(before, after_import), (
        f"importing the survey changed {state_changes(before, after_import)}"
    )
    assert after_import["filters"] == before["filters"]

    case = next(c for c in misc.CASES if c["name"] in promised_cases(ONNX))
    records = [run_onnx_case(case, dynamo=dynamo) for dynamo in (True, False)]
    assert all(record["status"] not in ("eager-fail", "skipped") for record in records), records
    # A canary for the legacy test, which passes any export failure: this case exports with the legacy exporter on
    # every supported torch, so an error of the harness itself (a renamed torch constant, say) cannot pass silently.
    assert records[1]["status"] == "ok", records[1]
    changes = state_changes(before, process_state())
    assert not changes, f"running a case changed {changes}"

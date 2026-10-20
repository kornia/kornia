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
    OPTION,
    describe,
    import_survey_module,
    parse_selection,
    pretrained_cases,
    promised_cases,
    promised_digest,
    registration_hooks,
    registries_of,
    registry_cases,
    run_case,
    selected_cases,
)

# Every case runs on CPU tensors and is compared in onnxruntime as numpy, whatever ``--device`` says. The registries
# and the harness call torch.manual_seed, which seeds CUDA and MPS too; testing.onnx_conformance forks only the CPU
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
    case = registry_cases(registries_of([promised_cases()[case_name]])).get(case_name)
    if case is None or case["skip"] or not case["check"]:
        pytest.fail(f"{case_name!r} is promised but no registry case runs it; see test_promised_cases_are_registered")
    return case


@pytest.mark.onnx_conformance
@quiet
def test_export_matches_eager(case_name: str | None) -> None:
    record = run_case(_promised_case(case_name), dynamo=True)
    assert record["status"] == "ok", describe(record)


@pytest.mark.onnx_conformance
@quiet
def test_legacy_export_is_not_wrong(case_name: str | None) -> None:
    record = run_case(_promised_case(case_name), dynamo=False)
    assert record["status"] in LEGACY_PASSES, describe(record)


@pytest.mark.onnx_conformance
@quiet
def test_promised_cases_are_registered(request: pytest.FixtureRequest) -> None:
    """Every promised case of the selected packages has a registry case that runs and is compared with eager."""
    packages = _selection(request.config)
    registries = registries_of(packages)
    cases = registry_cases(registries)
    problems = []
    for name in selected_cases(packages):
        case = cases.get(name)
        if case is None:
            problems.append(f"{name}: no case of that name in {', '.join(registries) or 'no registry'}")
        elif case["skip"]:
            problems.append(f"{name}: skipped ({case['skip']})")
        elif not case["check"]:
            problems.append(f"{name}: check=False, the output is not compared with eager")
    # The promise leaves out pretrained cases by reading the registry sources; the imported registries must agree.
    tagged = {name for name, case in cases.items() if "pretrained" in case["tags"]}
    if disagree := tagged ^ (pretrained_cases() & cases.keys()):
        problems.append(f"registry tags and the static pretrained scan disagree on {sorted(disagree)}")
    assert not problems, "promised cases the registries no longer run:\n" + "\n".join(problems)


def test_promised_set_is_pinned() -> None:
    # The promise is read from the snapshot, so a regeneration that measured a case as failing would quietly drop it.
    # Pinning the names makes a change of the promised set a reviewed change of these two constants.
    promised = promised_cases()
    assert (len(promised), promised_digest(promised)) == (PROMISED_COUNT, PROMISED_DIGEST), (
        "the promised set changed; review the snapshot diff, then update PROMISED_COUNT and PROMISED_DIGEST"
    )
    assert not any(name.startswith(("models.", "contrib.")) for name in promised)
    assert not promised.keys() & pretrained_cases()
    # The root conftest deselects or skips tests whose id contains these words, which would drop the case silently.
    assert not [name for name in promised if any(word in name.lower() for word in ("dynamo", "compile", "autocast"))]


def _process_state() -> dict:
    return {
        "threads": torch.get_num_threads(),
        "logging": logging.root.manager.disable,
        "filters": list(warnings.filters),
        "rng": torch.random.get_rng_state(),
        "hooks": registration_hooks(),
    }


def test_survey_harness_leaves_process_state_alone() -> None:
    """Importing the harness and a registry changes nothing; running a case leaves only torch's own warning filters."""
    before = _process_state()
    import_survey_module("harness")
    misc = import_survey_module("cases_misc")
    after_import = _process_state()
    assert torch.equal(after_import.pop("rng"), before["rng"])
    assert after_import == {key: value for key, value in before.items() if key != "rng"}

    case = next(c for c in misc.CASES if c["name"] in promised_cases())
    records = [run_case(case, dynamo=dynamo) for dynamo in (True, False)]
    after = _process_state()
    assert all(record["status"] not in ("eager-fail", "skipped") for record in records), records
    # A canary for the legacy test, which passes any export failure: this case exports with the legacy exporter on
    # every supported torch, so an error of the harness itself (a renamed torch constant, say) cannot pass silently.
    assert records[1]["status"] == "ok", records[1]
    assert torch.equal(after.pop("rng"), before["rng"])
    # A torch.export trace may add torch's own filters (torch 2.5 adds one for module "torch.distributed"); the
    # harness used to add a blanket "ignore everything" one. Anything new must name a torch module.
    added = [f for f in after.pop("filters") if f not in before["filters"]]
    assert all(f[3] is not None and f[3].pattern.startswith("torch") for f in added), added
    assert after == {key: value for key, value in before.items() if key not in ("rng", "filters")}

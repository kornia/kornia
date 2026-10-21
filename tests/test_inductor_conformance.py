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

import types
from pathlib import Path

import pytest
import torch

from testing.conformance import (
    describe,
    import_survey_module,
    load_known_failures,
    parametrize_cases,
    pin_problems,
    process_state,
    promised_case,
    raise_for_status,
    registration_problems,
    run_compile_case,
    selection,
    state_changes,
)
from testing.conformance_promise import (
    COMPILE,
    ENFORCED_TORCH_RELEASES,
    EXCLUDED,
    exclusion_problems,
    known_failures,
    manifest_path,
    promised_cases,
)

# Every case runs on CPU tensors, whatever ``--device`` says. The registries and the probe call torch.manual_seed,
# which seeds CUDA and MPS too; testing.conformance forks only the CPU generator, so restore_torch_rng puts the
# accelerator generators back.
pytestmark = [pytest.mark.device_agnostic, pytest.mark.usefixtures("restore_torch_rng")]

# Dynamo and inductor warn about graphs they handle; the per-case tests keep the session output readable. Not
# module-wide: the process-state tests must see a filter the probe would add.
quiet = pytest.mark.filterwarnings("ignore")

# The compile promise of the 2026-09-02 snapshot, less its exclusions: 829 cases of 635 operators.
PROMISED_COUNT = 829
PROMISED_DIGEST = "de2f308f80fad99ac1fccf38efaeb07088fe23b2c8dde9e21070e3c0a4e9c263"


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


def test_stale_exclusion_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exclusion must leave out a case the promise would cover without it; one that leaves nothing out is reported.

    A regenerated snapshot that stops marking an excluded case ``ok`` drops it from the promise by itself (#6194).
    """
    assert not exclusion_problems(COMPILE)
    stale = {
        "transform.crop_by_boxes": "recorded with graph breaks",
        "utils.create_meshgrid": "not a stable-core package",
        "feature.OriNet": "built with pretrained weights",
        "no.such.case": "not in the snapshot",
    }
    problems = exclusion_problems(COMPILE, {**EXCLUDED["compile"], **stale})
    assert [problem.split(": ")[0] for problem in problems] == list(stale), problems
    assert all("delete its entry from EXCLUDED['compile']" in problem for problem in problems), problems
    # the pin test runs the guard on the real exclusions; none of these names is promised, so the pin itself holds
    monkeypatch.setitem(EXCLUDED["compile"], "transform.crop_by_boxes", "recorded with graph breaks")
    assert pin_problems(COMPILE, PROMISED_COUNT, PROMISED_DIGEST) == [problems[0]]


@pytest.mark.parametrize("version", ENFORCED_TORCH_RELEASES)
def test_known_failure_manifest_names_promised_cases(version: str) -> None:
    """The manifest of every enforced torch release parses, not only the running torch's, which the run reads."""
    load_known_failures(COMPILE, manifest_path(COMPILE, version))


def test_known_failure_needs_its_tracking_issue(tmp_path: Path) -> None:
    """Every manifest entry sits under a comment that opens with its tracking issue, which the support page cites."""
    entry = "RuntimeError\ttests/test_inductor_conformance.py::test_inductor_matches_eager[{}]"
    path = tmp_path / "manifest.txt"
    path.write_text(f"# header\n{entry.format('LongestMaxSize')}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no tracking issue"):
        load_known_failures(COMPILE, path)
    # a comment block of its own after an entry does not inherit the issue above it
    path.write_text(
        f"# #5037: why\n# more of why\n{entry.format('LongestMaxSize')}\n# a note\n{entry.format('SmallestMaxSize')}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"manifest\.txt:5: no tracking issue"):
        load_known_failures(COMPILE, path)
    # nor does one after a blank line, before any entry
    path.write_text(f"# #5037: why\n\n# an unrelated note\n{entry.format('LongestMaxSize')}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"manifest\.txt:4: no tracking issue"):
        load_known_failures(COMPILE, path)
    path.write_text(
        f"# #5037: why\n{entry.format('LongestMaxSize')}\n# pytorch/pytorch#1 (fixed upstream): why\n"
        f"{entry.format('SmallestMaxSize')}\n",
        encoding="utf-8",
    )
    assert [(e.case, e.issue) for e in known_failures(COMPILE, path)] == [
        ("LongestMaxSize", "#5037"),
        ("SmallestMaxSize", "pytorch/pytorch#1"),
    ]
    # an excluded case is not promised, so no manifest may name it
    path.write_text(f"# #6194: why\n{entry.format('bbox.nms')}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"'bbox\.nms' is not a promised case"):
        load_known_failures(COMPILE, path)


def test_survey_probe_import_leaves_process_state_alone() -> None:
    """Importing the compile probe (and with it the harness) and a registry changes nothing."""
    before = process_state()
    import_survey_module("probe_compile")
    import_survey_module("cases_misc")
    after = process_state()
    assert not state_changes(before, after), f"importing the survey changed {state_changes(before, after)}"
    assert after["filters"] == before["filters"]


def test_survey_probe_records_failures_and_lets_test_outcomes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed export or compile is recorded; a test's own outcome, such as pytest-timeout's expiry, ends the test.

    Recorded, a timeout would become a ``fail`` record, which a known-failure entry of the same type absorbs.
    """
    probe = import_survey_module("probe_compile")
    harness = import_survey_module("harness")
    case = harness.case("neg", "test", torch.neg, [torch.ones(2)])
    wrap, inputs, ref = probe._reference(case)

    def raising(error: BaseException):
        def refuse(*args, **kwargs):
            raise error

        return refuse

    monkeypatch.setattr(torch, "compile", raising(probe.Timeout("case exceeded 420s")))
    record = probe.compile_step(case, wrap, inputs, ref, fullgraph=True)
    assert record["compile"] == "fail", record
    assert "Timeout" in record["compile_error"], record
    monkeypatch.setattr(torch, "compile", raising(pytest.fail.Exception("Timeout (>600.0s) from pytest-timeout.")))
    with pytest.raises(pytest.fail.Exception, match="pytest-timeout"):
        probe.compile_step(case, wrap, inputs, ref, fullgraph=True)
    monkeypatch.setattr(
        torch.export, "export", raising(pytest.fail.Exception("Timeout (>600.0s) from pytest-timeout."))
    )
    with pytest.raises(pytest.fail.Exception, match="pytest-timeout"):
        probe.export_step(wrap, inputs)


@pytest.mark.parametrize(
    ("explained_breaks", "explained_reasons", "status", "reason"),
    [
        (0, [], "ok-breaks:0", "fullgraph=True: RuntimeError"),
        (-1, [], "ok-breaks:0", "fullgraph=True: RuntimeError"),
        (3, ["generic_jump TensorVariable()"], "ok-breaks:3", "generic_jump TensorVariable()"),
    ],
    ids=["no-break", "no-graph", "breaks"],
)
def test_survey_probe_reports_ok_only_for_one_graph(
    monkeypatch: pytest.MonkeyPatch, explained_breaks: int, explained_reasons: list[str], status: str, reason: str
) -> None:
    """The survey's compile ``ok`` means ``torch.compile(..., fullgraph=True)`` captured the case (#6194).

    A case ``fullgraph=True`` refuses gets ``torch._dynamo.explain``'s break count and reasons. Explain can count no
    graph break for a call Dynamo runs in Python: 0, or -1 when it captures no graph at all, as for ``bbox.nms``; the
    survey records such a case as ``ok-breaks:0``, with the refusal as its break reason. The conformance test
    (``fullgraph=True`` alone) records every refused case as ``fail``.
    """
    probe = import_survey_module("probe_compile")
    harness = import_survey_module("harness")
    case = harness.case("neg", "test", torch.neg, [torch.ones(2)])
    wrap, inputs, ref = probe._reference(case)

    def compile_without_one_graph(model, fullgraph=False, **kwargs):
        def refuse(*args):
            raise RuntimeError("Data dependent operator")

        return refuse if fullgraph else model

    def explain(model):
        reasons = [types.SimpleNamespace(reason=text) for text in explained_reasons]
        return lambda *args: types.SimpleNamespace(graph_break_count=explained_breaks, break_reasons=reasons)

    monkeypatch.setattr(torch, "compile", compile_without_one_graph)
    monkeypatch.setattr(torch._dynamo, "explain", explain)
    record = probe.compile_step(case, wrap, inputs, ref)
    assert (record["compile"], record["graph_breaks"]) == (status, max(explained_breaks, 0)), record
    [recorded] = record["break_reasons"]
    assert recorded.startswith(reason), record
    assert ("Data dependent operator" in recorded) == (not explained_reasons), record
    assert probe.compile_step(case, wrap, inputs, ref, fullgraph=True)["compile"] == "fail"
    # a case fullgraph=True captures is ok in both
    monkeypatch.setattr(torch, "compile", lambda model, fullgraph=False, **kwargs: model)
    assert probe.compile_step(case, wrap, inputs, ref)["compile"] == "ok"
    assert probe.compile_step(case, wrap, inputs, ref, fullgraph=True)["compile"] == "ok"


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


@pytest.mark.compile_conformance
@quiet
def test_survey_probe_rejects_conjugated_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real compiled promised case must fail conformance if its imaginary component is corrupted."""
    case = promised_case(COMPILE, "geometry.liegroup.So2.exp.z")
    original_compile = torch.compile

    def conjugating_compile(*args, **kwargs):
        compiled = original_compile(*args, **kwargs)

        def conjugated(*inputs):
            return tuple(output.conj().resolve_conj() for output in compiled(*inputs))

        return conjugated

    monkeypatch.setattr(torch, "compile", conjugating_compile)
    record = run_compile_case(case)
    assert record["graph_breaks"] == 0, record
    assert record["compile"] == "mismatch", describe(record, "compile")
    assert "max abs diff" in record["compile_error"], record

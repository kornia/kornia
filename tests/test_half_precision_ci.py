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

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.autograd.gradcheck import GradcheckError

import conftest as project_conftest
from kornia.core.exceptions import ShapeError

from testing.half_precision_ci import (
    ISSUE_URL,
    load_known_failures,
    mark_known_failures,
    seed_test_rng,
)

pytest_plugins = ["pytester"]


def _write_manifest(directory: Path, dtype: str, contents: str) -> None:
    (directory / f"cpu_{dtype}.txt").write_text(contents)


def test_seed_test_rng_is_stable_and_node_specific() -> None:
    seed_test_rng("tests/a.py::test_a[cpu-float16]")
    first = (random.random(), np.random.random(), torch.rand(()).item())  # noqa: NPY002

    seed_test_rng("tests/a.py::test_b[cpu-float16]")
    other = (random.random(), np.random.random(), torch.rand(()).item())  # noqa: NPY002
    seed_test_rng("tests/a.py::test_a[cpu-float16]")
    repeated = (random.random(), np.random.random(), torch.rand(()).item())  # noqa: NPY002

    assert repeated == first
    assert other != first


def test_failure_recorder_writes_setup_and_call_failures_in_nodeid_order(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(
        """
        from pathlib import Path

        from testing.half_precision_ci import FailureRecorder


        def pytest_configure(config):
            recorder = FailureRecorder("float16", Path(__file__).with_name("recorded.txt"))
            config.pluginmanager.register(recorder, "half-precision-failure-recorder")
        """
    )
    pytester.makepyfile(
        test_sample="""
        import pytest


        def test_z_failure():
            raise ValueError("second after sorting")


        def test_a_failure():
            raise AssertionError("first after sorting")


        @pytest.fixture
        def failed_setup():
            raise RuntimeError("fixture setup failure")


        def test_setup_failure(failed_setup):
            pass


        def test_kornia_failure():
            from kornia.core.exceptions import ShapeError

            raise ShapeError("kornia validation failure")


        def test_skip():
            pytest.skip("not a failure")


        @pytest.mark.xfail(reason="already tracked")
        def test_existing_xfail():
            raise RuntimeError("not a new failure")
        """
    )

    result = pytester.runpytest("-q")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    recorded = (pytester.path / "recorded.txt").read_text(encoding="utf-8")
    assert "Recorded Linux CPU float16 failures" in recorded
    lines = [line for line in recorded.splitlines() if line and not line.startswith("#")]
    assert lines == [
        "AssertionError\ttest_sample.py::test_a_failure",
        "kornia.core.exceptions.ShapeError\ttest_sample.py::test_kornia_failure",
        "RuntimeError\ttest_sample.py::test_setup_failure",
        "ValueError\ttest_sample.py::test_z_failure",
    ]
    _write_manifest(pytester.path, "float16", recorded)
    assert load_known_failures("float16", pytester.path)["test_sample.py::test_kornia_failure"] is ShapeError


class TestLoadKnownFailures:
    def test_loads_nodeids_and_exception_types(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            "float16",
            "# tracked in kornia#4153\n"
            "AssertionError\ttests/a.py::test_a[cpu-float16]\n"
            "RuntimeError\ttests/b.py::TestB::test_b[cpu-float16]\n",
        )

        failures = load_known_failures("float16", tmp_path)

        assert failures == {
            "tests/a.py::test_a[cpu-float16]": AssertionError,
            "tests/b.py::TestB::test_b[cpu-float16]": RuntimeError,
        }

    def test_rejects_duplicate_nodeids(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            "float16",
            "AssertionError\ttests/a.py::test_a[cpu-float16]\nRuntimeError\ttests/a.py::test_a[cpu-float16]\n",
        )

        with pytest.raises(ValueError, match="duplicate node ID"):
            load_known_failures("float16", tmp_path)

    def test_rejects_unknown_exception_types(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "float16", "UnknownError\ttests/a.py::test_a[cpu-float16]\n")

        with pytest.raises(ValueError, match="unknown exception type"):
            load_known_failures("float16", tmp_path)

    def test_loads_any_builtin_exception_type(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "float16", "IndexError\ttests/a.py::test_a[cpu-float16]\n")

        failures = load_known_failures("float16", tmp_path)

        assert failures == {"tests/a.py::test_a[cpu-float16]": IndexError}

    def test_loads_gradcheck_error_recorded_by_the_recorder(self, tmp_path: Path) -> None:
        nodeid = "tests/a.py::test_a[cpu-float16]"
        _write_manifest(tmp_path, "float16", f"torch.autograd.gradcheck.GradcheckError\t{nodeid}\n")

        failures = load_known_failures("float16", tmp_path)

        assert failures == {nodeid: GradcheckError}

    def test_loads_kornia_exception_recorded_by_the_recorder(self, tmp_path: Path) -> None:
        nodeid = "tests/a.py::test_a[cpu-float16]"
        _write_manifest(tmp_path, "float16", f"kornia.core.exceptions.ShapeError\t{nodeid}\n")

        failures = load_known_failures("float16", tmp_path)

        assert failures == {nodeid: ShapeError}

    def test_loads_nodeid_without_dtype_parameter(self, tmp_path: Path) -> None:
        nodeid = "tests/a.py::test_a"
        _write_manifest(tmp_path, "float16", f"AssertionError\t{nodeid}\n")

        failures = load_known_failures("float16", tmp_path)

        assert failures == {nodeid: AssertionError}

    def test_rejects_entries_for_the_wrong_dtype(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "float16", "AssertionError\ttests/a.py::test_a[cpu-bfloat16]\n")

        with pytest.raises(ValueError, match="does not select float16"):
            load_known_failures("float16", tmp_path)

    def test_loads_dtype_only_nodeids_for_cpu_runs(self, tmp_path: Path) -> None:
        nodeid = "tests/a.py::test_a[float16-case]"
        _write_manifest(tmp_path, "float16", f"AssertionError\t{nodeid}\n")

        failures = load_known_failures("float16", tmp_path)

        assert failures == {nodeid: AssertionError}


class _Item:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid
        self.markers: list[pytest.MarkDecorator] = []

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.markers.append(marker)


class TestMarkKnownFailures:
    def test_marks_exact_nodeid_strictly_with_expected_exception(self, tmp_path: Path) -> None:
        nodeid = "tests/a.py::test_a[cpu-float16]"
        _write_manifest(tmp_path, "float16", f"AssertionError\t{nodeid}\n")
        item = _Item(nodeid)

        tracker = mark_known_failures([item], ["float16"], tmp_path)

        assert len(item.markers) == 1
        assert item.markers[0].mark.kwargs == {
            "raises": AssertionError,
            "reason": tracker.reason,
            "strict": True,
        }
        assert tracker.reason == f"Known Linux CPU half-precision failure tracked in {ISSUE_URL}"

    def test_rejects_manifest_entries_that_were_not_collected(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            "float16",
            "AssertionError\ttests/missing.py::test_missing[cpu-float16]\n",
        )

        with pytest.raises(ValueError, match="1 known half-precision failure was not collected") as error:
            mark_known_failures([], ["float16"], tmp_path)

        assert str(tmp_path / "cpu_float16.txt") in str(error.value)
        assert "remove or update" in str(error.value)

    def test_scopes_manifest_to_an_explicit_node_selector(self, tmp_path: Path) -> None:
        selected = "tests/a.py::test_a[cpu-float16]"
        _write_manifest(
            tmp_path,
            "float16",
            f"AssertionError\t{selected}\nAssertionError\ttests/a.py::test_b[cpu-float16]\n",
        )
        item = _Item(selected)

        tracker = mark_known_failures([item], ["float16"], tmp_path, selectors=[selected])

        assert tracker.pending == {selected}
        assert len(item.markers) == 1

    def test_selected_file_still_rejects_missing_entries(self, tmp_path: Path) -> None:
        selected = "tests/a.py::test_a[cpu-float16]"
        _write_manifest(
            tmp_path,
            "float16",
            f"AssertionError\t{selected}\nAssertionError\ttests/a.py::test_renamed[cpu-float16]\n",
        )

        with pytest.raises(ValueError, match="1 known half-precision failure was not collected"):
            mark_known_failures([_Item(selected)], ["float16"], tmp_path, selectors=["tests/a.py"])

    def test_absolute_selector_is_normalized_against_rootpath(self, tmp_path: Path) -> None:
        selected = "tests/a.py::test_a[cpu-float16]"
        _write_manifest(tmp_path, "float16", f"AssertionError\t{selected}\n")
        item = _Item(selected)

        tracker = mark_known_failures(
            [item], ["float16"], tmp_path, selectors=[str(tmp_path / "tests/a.py")], rootpath=tmp_path
        )

        assert tracker.pending == {selected}
        assert len(item.markers) == 1


def _run_manifest_case(
    pytester: pytest.Pytester, test_body: str, extra_marker: str = "", manifest_exception: str = "AssertionError"
) -> pytest.RunResult:
    nodeid = "test_sample.py::test_known_failure[cpu-float16]"
    _write_manifest(pytester.path, "float16", f"{manifest_exception}\t{nodeid}\n")
    pytester.makeconftest(
        """
        from pathlib import Path

        from testing.half_precision_ci import mark_known_failures


        def pytest_collection_modifyitems(config, items):
            tracker = mark_known_failures(items, ["float16"], Path(__file__).parent)
            config.pluginmanager.register(tracker, "known-half-precision-failure-tracker")
        """
    )
    pytester.makepyfile(
        test_sample=f"""
        import pytest


        {extra_marker}
        @pytest.mark.parametrize("unused", [None], ids=["cpu-float16"])
        def test_known_failure(unused, request):
            {test_body}
        """
    )
    return pytester.runpytest("-q")


def test_deselected_manifest_entries_do_not_fail_partial_runs(pytester: pytest.Pytester) -> None:
    nodeid = "test_sample.py::test_known_failure[cpu-float16]"
    _write_manifest(pytester.path, "float16", f"AssertionError\t{nodeid}\n")
    pytester.makeconftest(
        """
        from pathlib import Path

        from testing.half_precision_ci import mark_known_failures


        def pytest_collection_modifyitems(config, items):
            tracker = mark_known_failures(
                items, ["float16"], Path(__file__).parent, selectors=config.args
            )
            config.pluginmanager.register(tracker, "known-half-precision-failure-tracker")
        """
    )
    pytester.makepyfile(
        test_sample="""
        import pytest


        @pytest.mark.parametrize("unused", [None], ids=["cpu-float16"])
        def test_known_failure(unused):
            raise AssertionError("known failure")


        def test_other():
            pass
        """
    )

    result = pytester.runpytest("test_sample.py", "-k", "other", "-q")

    result.assert_outcomes(passed=1, deselected=1)
    assert result.ret == pytest.ExitCode.OK


class TestKnownFailureOutcomes:
    def test_accepts_manifest_specific_xfail(self, pytester: pytest.Pytester) -> None:
        result = _run_manifest_case(pytester, "raise AssertionError('known failure')")

        result.assert_outcomes(xfailed=1)
        assert result.ret == pytest.ExitCode.OK

    def test_accepts_manifest_specific_setup_xfail(self, pytester: pytest.Pytester) -> None:
        result = _run_manifest_case(
            pytester,
            "pass",
            "@pytest.fixture(autouse=True)\n"
            "        def fail_setup():\n"
            "            raise AssertionError('setup failure')",
        )

        result.assert_outcomes(xfailed=1)
        assert result.ret == pytest.ExitCode.OK

    @pytest.mark.parametrize(
        ("test_body", "extra_marker"),
        [
            ("pytest.skip('disabled test')", ""),
            (
                "raise AssertionError('known failure')",
                "@pytest.mark.skip(reason='disabled test')",
            ),
            (
                "raise AssertionError('known failure')",
                "@pytest.mark.xfail(reason='different xfail marker', strict=False)",
            ),
            ("raise ValueError('wrong failure type')", ""),
            (
                "raise ValueError('wrong failure type')",
                f"@pytest.mark.xfail(reason='Known Linux CPU half-precision failure tracked in {ISSUE_URL}')",
            ),
            (
                "request.addfinalizer(lambda: pytest.skip('disabled in teardown')); "
                "raise AssertionError('known failure')",
                "",
            ),
            (
                "request.addfinalizer("
                "lambda: (_ for _ in ()).throw(AssertionError('teardown failure'))"
                "); raise AssertionError('known failure')",
                "",
            ),
            (
                "pytest.xfail(next(request.node.iter_markers('xfail')).kwargs['reason'])",
                "",
            ),
            ("raise NotImplementedError('narrower runtime error')", ""),
        ],
        ids=[
            "runtime-skip",
            "static-skip",
            "prior-xfail",
            "wrong-exception",
            "same-reason-prior-xfail",
            "teardown-skip",
            "teardown-expected-exception",
            "dynamic-xfail-with-manifest-reason",
            "exception-subclass",
        ],
    )
    def test_rejects_manifest_entry_bypassed_by_other_outcome(
        self, pytester: pytest.Pytester, test_body: str, extra_marker: str
    ) -> None:
        manifest_exception = "RuntimeError" if test_body.startswith("raise NotImplementedError") else "AssertionError"
        result = _run_manifest_case(pytester, test_body, extra_marker, manifest_exception)

        assert result.ret == pytest.ExitCode.TESTS_FAILED
        output = result.stdout.str()
        assert "ERROR: 1 known half-precision failure needs updates in" in output
        assert "cpu_float16.txt" in output
        assert "Ran with an unexpected outcome; remove or update the manifest line:" in output
        assert "test_sample.py::test_known_failure[cpu-float16]" in output


class _RecordingConfig:
    def __init__(self, option_name: str, option_value: object, rootpath: Path) -> None:
        options = {
            "keyword": "",
            "markexpr": "",
            "deselect": [],
            "maxfail": 0,
            "lf": False,
            "failedfirst": False,
        }
        options[option_name] = option_value
        self.option = SimpleNamespace(**options)
        self.args = ["tests"]
        self.rootpath = rootpath

    def getoption(self, option: str) -> object:
        return {
            "--xfail-known-half-precision": False,
            "--record-half-precision-failures": "recorded.txt",
            "--device": "cpu",
            "--dtype": "float16",
            "--runslow": False,
        }[option]


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("keyword", "selected"),
        ("markexpr", "unit"),
        ("deselect", ["tests/a.py::test_a"]),
        ("maxfail", 1),
        ("lf", True),
        ("failedfirst", True),
    ],
)
def test_record_mode_rejects_partial_run_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option_name: str, option_value: object
) -> None:
    monkeypatch.delenv("KORNIA_TEST_OPTIMIZER", raising=False)
    config = _RecordingConfig(option_name, option_value, tmp_path)

    with pytest.raises(pytest.UsageError, match="does not allow partial-selection options"):
        project_conftest._configure_half_precision_manifest(config, [])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_recorded_baseline_manifest(dtype: str) -> None:
    failures = load_known_failures(dtype)

    assert failures
    assert all(issubclass(exception, BaseException) for exception in failures.values())

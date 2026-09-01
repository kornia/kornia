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

import pytest

from testing.half_precision_ci import ISSUE_URL, load_known_failures, mark_known_failures

pytest_plugins = ["pytester"]


def _write_manifest(directory: Path, dtype: str, contents: str) -> None:
    (directory / f"cpu_{dtype}.txt").write_text(contents)


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

    def test_rejects_entries_for_the_wrong_dtype(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "float16", "AssertionError\ttests/a.py::test_a[cpu-bfloat16]\n")

        with pytest.raises(ValueError, match="does not select cpu-float16"):
            load_known_failures("float16", tmp_path)


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
        assert tracker.reason.startswith(f"Known Linux CPU half-precision failure tracked in {ISSUE_URL} [tracker=")

    def test_rejects_manifest_entries_that_were_not_collected(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            "float16",
            "AssertionError\ttests/missing.py::test_missing[cpu-float16]\n",
        )

        with pytest.raises(ValueError, match="1 known half-precision failures were not collected"):
            mark_known_failures([], ["float16"], tmp_path)


def _run_manifest_case(pytester: pytest.Pytester, test_body: str, extra_marker: str = "") -> pytest.RunResult:
    nodeid = "test_sample.py::test_known_failure[cpu-float16]"
    _write_manifest(pytester.path, "float16", f"AssertionError\t{nodeid}\n")
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


class TestKnownFailureOutcomes:
    def test_accepts_manifest_specific_xfail(self, pytester: pytest.Pytester) -> None:
        result = _run_manifest_case(pytester, "raise AssertionError('known failure')")

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
                "raise AssertionError('known failure')",
                "@pytest.fixture(autouse=True)\n"
                "        def fail_setup():\n"
                "            raise AssertionError('setup failure')",
            ),
            (
                "pytest.xfail(next(request.node.iter_markers('xfail')).kwargs['reason'])",
                "",
            ),
        ],
        ids=[
            "runtime-skip",
            "static-skip",
            "prior-xfail",
            "wrong-exception",
            "same-reason-prior-xfail",
            "teardown-skip",
            "teardown-expected-exception",
            "setup-expected-exception",
            "dynamic-xfail-with-manifest-reason",
        ],
    )
    def test_rejects_manifest_entry_bypassed_by_other_outcome(
        self, pytester: pytest.Pytester, test_body: str, extra_marker: str
    ) -> None:
        result = _run_manifest_case(pytester, test_body, extra_marker)

        assert result.ret == pytest.ExitCode.TESTS_FAILED
        output = result.stdout.str()
        assert "ERROR: 1 known half-precision failure did not finish with the manifest-specific xfail:" in output
        assert "test_sample.py::test_known_failure[cpu-float16]" in output


@pytest.mark.parametrize(("dtype", "expected_count"), [("float16", 605), ("bfloat16", 591)])
def test_recorded_baseline_manifest(dtype: str, expected_count: int) -> None:
    failures = load_known_failures(dtype)

    assert len(failures) == expected_count
    assert all(issubclass(exception, BaseException) for exception in failures.values())
    assert ISSUE_URL == "https://github.com/kornia/kornia/issues/4153"

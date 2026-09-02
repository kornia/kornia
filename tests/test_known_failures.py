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

from testing.known_failures import ISSUE_URL, load_known_failures, mark_known_failures

pytest_plugins = ["pytester"]


def _write_manifest(directory: Path, contents: str) -> None:
    (directory / "mps_float32.txt").write_text(contents)


class TestLoadKnownFailures:
    def test_loads_nodeids_and_exception_types(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            "# tracked in kornia#4159\n"
            "AssertionError\ttests/a.py::test_a[mps-float32]\n"
            "RuntimeError\ttests/b.py::TestB::test_b[mps]\n",
        )

        failures, path = load_known_failures("mps", "float32", tmp_path)

        assert failures == {
            "tests/a.py::test_a[mps-float32]": AssertionError,
            "tests/b.py::TestB::test_b[mps]": RuntimeError,
        }
        assert path == tmp_path / "mps_float32.txt"

    def test_rejects_duplicate_nodeids(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            "AssertionError\ttests/a.py::test_a[mps-float32]\nRuntimeError\ttests/a.py::test_a[mps-float32]\n",
        )

        with pytest.raises(ValueError, match="duplicate node ID"):
            load_known_failures("mps", "float32", tmp_path)

    def test_rejects_unknown_exception_types(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "UnknownError\ttests/a.py::test_a[mps-float32]\n")

        with pytest.raises(ValueError, match="unknown exception type"):
            load_known_failures("mps", "float32", tmp_path)

    def test_rejects_entries_for_the_wrong_device(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "AssertionError\ttests/a.py::test_a[cpu-float32]\n")

        with pytest.raises(ValueError, match="does not select mps"):
            load_known_failures("mps", "float32", tmp_path)


class _Item:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid
        self.markers: list[pytest.MarkDecorator] = []

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.markers.append(marker)


class TestMarkKnownFailures:
    def test_marks_exact_nodeid_strictly_with_expected_exception(self, tmp_path: Path) -> None:
        nodeid = "tests/a.py::test_a[mps-float32]"
        _write_manifest(tmp_path, f"AssertionError\t{nodeid}\n")
        item = _Item(nodeid)

        tracker = mark_known_failures([item], "mps", "float32", tmp_path)

        assert len(item.markers) == 1
        assert item.markers[0].mark.kwargs == {
            "raises": AssertionError,
            "reason": tracker.reason,
            "strict": True,
        }
        assert tracker.reason == f"Known mps/float32 failure tracked in {ISSUE_URL}"

    def test_rejects_manifest_entries_that_were_not_collected(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "AssertionError\ttests/missing.py::test_missing[mps-float32]\n")

        with pytest.raises(ValueError, match=r"1 failures .* were not collected"):
            mark_known_failures([], "mps", "float32", tmp_path)


def _run_manifest_case(pytester: pytest.Pytester, test_body: str, extra_marker: str = "") -> pytest.RunResult:
    nodeid = "test_sample.py::test_known_failure[mps-float32]"
    _write_manifest(pytester.path, f"RuntimeError\t{nodeid}\n")
    pytester.makeconftest(
        """
        from pathlib import Path

        from testing.known_failures import mark_known_failures


        def pytest_collection_modifyitems(config, items):
            tracker = mark_known_failures(items, "mps", "float32", Path(__file__).parent)
            config.pluginmanager.register(tracker, "known-failure-tracker")
        """
    )
    pytester.makepyfile(
        test_sample=f"""
        import pytest


        {extra_marker}
        @pytest.mark.parametrize("unused", [None], ids=["mps-float32"])
        def test_known_failure(unused, request):
            {test_body}
        """
    )
    return pytester.runpytest("-q")


class TestKnownFailureOutcomes:
    def test_accepts_exact_manifest_xfail(self, pytester: pytest.Pytester) -> None:
        result = _run_manifest_case(pytester, "raise RuntimeError('known failure')")

        result.assert_outcomes(xfailed=1)
        assert result.ret == pytest.ExitCode.OK

    @pytest.mark.parametrize(
        ("test_body", "extra_marker"),
        [
            ("pass", ""),
            ("raise AssertionError('wrong failure type')", ""),
            ("raise NotImplementedError('subclass of RuntimeError')", ""),
            ("pytest.skip('disabled test')", ""),
            (
                "raise RuntimeError('known failure')",
                "@pytest.mark.skip(reason='disabled test')",
            ),
            (
                "raise RuntimeError('known failure')",
                "@pytest.mark.xfail(reason='different xfail marker', strict=False)",
            ),
            (
                "request.addfinalizer(lambda: pytest.skip('disabled in teardown')); "
                "raise RuntimeError('known failure')",
                "",
            ),
            (
                "raise RuntimeError('known failure')",
                "@pytest.fixture(autouse=True)\n"
                "        def fail_setup():\n"
                "            raise RuntimeError('setup failure')",
            ),
        ],
        ids=[
            "fixed",
            "wrong-exception",
            "exception-subclass",
            "runtime-skip",
            "static-skip",
            "prior-xfail",
            "teardown-skip",
            "setup-failure",
        ],
    )
    def test_rejects_changed_or_bypassed_outcome(
        self, pytester: pytest.Pytester, test_body: str, extra_marker: str
    ) -> None:
        result = _run_manifest_case(pytester, test_body, extra_marker)

        assert result.ret == pytest.ExitCode.TESTS_FAILED
        output = result.stdout.str()
        assert "ERROR: 1 known failure did not reproduce exactly" in output
        assert "test_sample.py::test_known_failure[mps-float32]" in output


def test_recorded_mps_float32_baseline() -> None:
    failures, path = load_known_failures("mps", "float32")

    # The exact count changes with every fix that removes a manifest line; the contract is the file, not a number.
    assert len(failures) > 0
    assert path.name == "mps_float32.txt"
    assert all(issubclass(exception, BaseException) for exception in failures.values())
    assert ISSUE_URL == "https://github.com/kornia/kornia/issues/4159"

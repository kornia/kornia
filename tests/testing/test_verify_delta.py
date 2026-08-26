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

import os
from pathlib import Path
from types import SimpleNamespace

from testing import verify_delta
from testing.verify_delta import (
    SURFACES,
    FailureDelta,
    _failures_or_empty,
    changed_test_dirs,
    diff_failures,
    failing_ids,
    parse_args,
    render_table,
    select_surfaces,
)


class TestChangedTestDirs:
    def test_maps_library_modules_to_test_dirs(self):
        assert changed_test_dirs(["kornia/geometry/grid.py", "kornia/geometry/conversions.py"]) == ["tests/geometry"]

    def test_includes_test_dirs_touched_directly(self):
        assert changed_test_dirs(["tests/filters/test_sobel.py", "kornia/color/gray.py"]) == [
            "tests/color",
            "tests/filters",
        ]

    def test_shared_infrastructure_widens_to_everything(self):
        assert changed_test_dirs(["testing/base.py", "kornia/geometry/grid.py"]) == ["tests"]
        assert changed_test_dirs(["tests/conftest.py"]) == ["tests"]
        assert changed_test_dirs(["pyproject.toml"]) == ["tests"]

    def test_ignores_docs_and_other_paths(self):
        assert changed_test_dirs(["docs/source/geometry.rst", "CHANGELOG.md", "AGENTS.md"]) == []

    def test_top_level_module_files(self):
        assert changed_test_dirs(["kornia/constants.py"]) == ["tests"]


JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="4">
<testcase classname="tests.geometry.test_grid" name="test_a[cpu-float32]" time="0.1"/>
<testcase classname="tests.geometry.test_grid" name="test_b[cpu-float32]" time="0.1">
<failure message="x">t</failure></testcase>
<testcase classname="tests.geometry.test_grid" name="test_c[cpu-float32]" time="0.1">
<error message="y">t</error></testcase>
<testcase classname="tests.geometry.test_grid" name="test_d[cpu-float32]" time="0.1"><skipped message="z"/></testcase>
</testsuite></testsuites>
"""


class TestFailingIds:
    def test_collects_failures_and_errors_not_skips(self, tmp_path: Path):
        p = tmp_path / "junit.xml"
        p.write_text(JUNIT)
        assert failing_ids(p) == {
            "tests.geometry.test_grid::test_b[cpu-float32]",
            "tests.geometry.test_grid::test_c[cpu-float32]",
        }

    def test_missing_file_is_an_error(self, tmp_path: Path):
        import pytest

        with pytest.raises(FileNotFoundError):
            failing_ids(tmp_path / "nope.xml")


class TestDiffFailures:
    def test_sets_not_counts(self):
        d = diff_failures(branch={"a", "b", "c"}, main={"b", "c", "d"})
        assert d == FailureDelta(new=["a"], fixed=["d"], unchanged=["b", "c"])


class TestRenderTable:
    def test_lists_new_and_fixed_ids_and_marks_skipped_surfaces(self):
        out = render_table(
            [
                ("cpu float32", FailureDelta(new=["x::t1"], fixed=[], unchanged=["x::t0"])),
                ("mps float32", None),
            ]
        )
        assert "| cpu float32 | 1 | 0 | 1 |" in out
        assert "| mps float32 | skipped |" in out
        assert "NEW x::t1" in out


class TestSurfaces:
    def test_default_surfaces(self):
        assert [s.name for s in SURFACES] == [
            "cpu float32",
            "cpu float16,bfloat16,float64",
            "mps float32",
            "inductor cpu float32",
        ]

    def test_select_skips_mps_when_unavailable(self):
        chosen = select_surfaces(parse_args([]), mps_available=False)
        assert "mps float32" not in [s.name for s in chosen]

    def test_select_honours_only_flag(self):
        chosen = select_surfaces(parse_args(["--only", "cpu float32"]), mps_available=True)
        assert [s.name for s in chosen] == ["cpu float32"]

    def test_tests_override(self):
        assert parse_args(["--tests", "tests/geometry", "tests/filters"]).tests == ["tests/geometry", "tests/filters"]


class TestFailuresOrEmpty:
    def test_missing_junit_reads_as_no_failures(self, tmp_path: Path, capsys):
        assert _failures_or_empty(tmp_path / "nope.xml") == set()
        assert "no junit report" in capsys.readouterr().out

    def test_existing_junit_is_parsed(self, tmp_path: Path):
        p = tmp_path / "junit.xml"
        p.write_text(JUNIT)
        assert _failures_or_empty(p) == failing_ids(p)


class TestOnlyThroughATaskShell:
    """`pixi run verify-delta -- --only "cpu float32"` forwards `--` and drops the quoting."""

    def test_leading_double_dash_separator_is_dropped(self):
        assert parse_args(["--", "--only", "cpu float32"]).only == ["cpu float32"]

    def test_space_split_surface_name_is_rejoined(self):
        chosen = select_surfaces(parse_args(["--only", "cpu", "float32"]), mps_available=True)
        assert [s.name for s in chosen] == ["cpu float32"]

    def test_longest_name_wins_over_a_shorter_prefix_match(self):
        chosen = select_surfaces(parse_args(["--only", "inductor", "cpu", "float32"]), mps_available=True)
        assert [s.name for s in chosen] == ["inductor cpu float32"]

    def test_several_space_split_names_in_one_flag(self):
        chosen = select_surfaces(parse_args(["--only", "cpu", "float32", "mps", "float32"]), mps_available=True)
        assert [s.name for s in chosen] == ["cpu float32", "mps float32"]

    def test_unknown_surface_is_an_error(self):
        import pytest

        with pytest.raises(SystemExit, match="unknown surface"):
            select_surfaces(parse_args(["--only", "cuda", "float32"]), mps_available=True)


class _FakeRun:
    """Stand-in for ``subprocess.run`` that records argv and never launches anything."""

    def __init__(self, stdout: str = ""):
        self.stdout = stdout
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append([str(a) for a in argv])
        return SimpleNamespace(stdout=self.stdout, returncode=0)


class TestRunSurface:
    def test_a_stale_junit_from_a_previous_run_is_not_reused(self, tmp_path: Path, monkeypatch, capsys):
        (tmp_path / "tests").mkdir()
        junit = tmp_path / "j.xml"
        junit.write_text(JUNIT)  # a previous run's failures, which this run must not inherit
        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"))
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        assert verify_delta._run_surface(tmp_path, SURFACES[0], ["tests"], junit) == set()
        assert not junit.exists()
        assert "no junit report" in capsys.readouterr().out

    def test_the_tree_under_test_is_on_pythonpath(self, tmp_path: Path, monkeypatch):
        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"))
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        monkeypatch.delenv("PYTHONPATH", raising=False)
        (tmp_path / "tests").mkdir()
        verify_delta._run_surface(tmp_path, SURFACES[0], ["tests"], tmp_path / "j.xml")
        assert os.environ.get("PYTHONPATH") is None  # the child env is not leaked into this process

    def test_paths_absent_from_the_tree_are_dropped_not_fatal(self, tmp_path: Path, monkeypatch, capsys):
        # pytest aborts collection entirely when any argument path is missing, which would silently
        # empty the base tree's failure set and report every branch failure as new.
        (tmp_path / "tests" / "geometry").mkdir(parents=True)
        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"))
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        verify_delta._run_surface(tmp_path, SURFACES[0], ["tests/geometry", "tests/testing"], tmp_path / "j.xml")
        pytest_argv = fake.calls[-1]
        assert pytest_argv[-1] == "tests/geometry"
        assert "tests/testing" not in pytest_argv
        assert "tests/testing" in capsys.readouterr().out

    def test_no_present_paths_means_no_run_and_no_failures(self, tmp_path: Path, monkeypatch):
        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"))
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        assert verify_delta._run_surface(tmp_path, SURFACES[0], ["tests/testing"], tmp_path / "j.xml") == set()
        assert not any("pytest" in call for call in fake.calls)


class TestEnsureMainWorktree:
    def test_a_local_base_revision_is_not_fetched(self, tmp_path: Path, monkeypatch):
        fake = _FakeRun()
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        verify_delta._ensure_main_worktree(tmp_path, "HEAD~1", tmp_path, fetch=True)
        assert not any("fetch" in call for call in fake.calls)

    def test_a_remote_base_revision_is_fetched(self, tmp_path: Path, monkeypatch):
        fake = _FakeRun()
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        verify_delta._ensure_main_worktree(tmp_path, "origin/main", tmp_path, fetch=True)
        assert ["git", "-C", str(tmp_path), "fetch", "origin", "main"] in fake.calls

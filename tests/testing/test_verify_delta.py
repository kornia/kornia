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

from pathlib import Path

from testing.verify_delta import FailureDelta, changed_test_dirs, diff_failures, failing_ids, render_table


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

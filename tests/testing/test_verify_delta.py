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
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

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

    def test_a_module_without_a_test_dir_of_its_own_widens_to_everything(self, tmp_path: Path, capsys):
        # kornia/transpiler/ is the live example: mapping it to a non-existent tests/transpiler and
        # then dropping the missing path would leave the change unverified behind a clean verdict.
        (tmp_path / "tests" / "geometry").mkdir(parents=True)
        assert changed_test_dirs(["kornia/transpiler/transpiler.py"], tmp_path) == ["tests"]
        assert "kornia/transpiler/transpiler.py maps to tests/transpiler" in capsys.readouterr().out

    def test_a_module_with_a_test_dir_is_not_widened(self, tmp_path: Path):
        (tmp_path / "tests" / "geometry").mkdir(parents=True)
        assert changed_test_dirs(["kornia/geometry/grid.py"], tmp_path) == ["tests/geometry"]


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
        assert "NEW [cpu float32] x::t1" in out

    def test_distinguishes_a_row_that_had_no_baseline_from_a_deselected_one(self):
        out = render_table(
            [
                ("cpu float32", FailureDelta(new=["x::t1"], baseline=False)),
                ("mps float32", None),
            ]
        )
        assert "| cpu float32 | 1* | 0 | 0 |" in out
        assert "| mps float32 | skipped |" in out
        assert "* no baseline on the base revision" in out

    def test_no_legend_when_every_row_had_a_baseline(self):
        assert "no baseline" not in render_table([("cpu float32", FailureDelta(new=["x::t1"]))])


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

    def test_surfaces_are_hashable(self):
        # a frozen dataclass with a `dict` field is unhashable, so `surface in selected` works only
        # because `selected` is a list; a set/dict of surfaces would raise TypeError at the callsite.
        assert len(set(SURFACES)) == len(SURFACES)

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
        with pytest.raises(SystemExit, match="unknown surface"):
            select_surfaces(parse_args(["--only", "cuda", "float32"]), mps_available=True)


class _FakeRun:
    """Stand-in for ``subprocess.run`` that records argv and kwargs and never launches anything."""

    def __init__(
        self,
        stdout: str = "",
        stdout_by_arg: dict[str, str] | None = None,
        junit: str | None = None,
        fails_for: tuple[str, ...] = (),
        stderr: str = "",
        pytest_returncode: int = 0,
    ):
        self.stdout = stdout
        self.stdout_by_arg = stdout_by_arg or {}  # first matching argv entry wins, e.g. keyed by a tree path
        self.junit = junit  # written to the --junitxml path, standing in for what pytest would report
        self.fails_for = fails_for  # argv entries whose command exits non-zero, e.g. "symbolic-ref"
        self.stderr = stderr
        self.pytest_returncode = pytest_returncode  # what the pytest child exits with, e.g. 4 for a usage error
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        self.kwargs.append(kwargs)
        if any(token in argv for token in self.fails_for):
            raise verify_delta.subprocess.CalledProcessError(1, argv, stderr=self.stderr)
        for arg in argv if self.junit is not None else []:
            if arg.startswith("--junitxml="):
                Path(arg.split("=", 1)[1]).write_text(self.junit)
        for key, stdout in self.stdout_by_arg.items():
            if key in argv:
                return SimpleNamespace(stdout=stdout, returncode=0)
        returncode = self.pytest_returncode if "pytest" in argv else 0
        return SimpleNamespace(stdout=self.stdout, returncode=returncode)


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

    def test_the_child_gets_the_tree_on_pythonpath_and_the_surface_env(self, tmp_path: Path, monkeypatch):
        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"))
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        monkeypatch.delenv("PYTHONPATH", raising=False)
        (tmp_path / "tests").mkdir()
        inductor = SURFACES[3]
        verify_delta._run_surface(tmp_path, inductor, ["tests"], tmp_path / "j.xml")
        assert len(fake.kwargs) == 2  # the import guard and pytest itself
        for kwargs in fake.kwargs:
            assert kwargs["cwd"] == tmp_path
            assert kwargs["env"]["PYTHONPATH"] == str(tmp_path)
            assert kwargs["env"]["KORNIA_TEST_DEVICE"] == "cpu"
            assert kwargs["env"]["KORNIA_TEST_OPTIMIZER"] == "inductor"
        assert fake.calls[-1][-3:] == ["-k", "dynamo or compile", "tests"]
        # a fresh dict, so setting PYTHONPATH for the child never mutates this process's environment
        assert all(kwargs["env"] is not os.environ for kwargs in fake.kwargs)

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

    def test_a_pytest_that_could_not_start_is_unverified_not_clean(self, tmp_path: Path, monkeypatch, capsys):
        # exit 4 (usage error: a broken conftest, a bad -k) leaves no junit behind, and reading that
        # as an empty failing set turns a tree where nothing ran into a green row and a zero exit.
        (tmp_path / "tests").mkdir()
        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"), pytest_returncode=4)
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        assert verify_delta._run_surface(tmp_path, SURFACES[0], ["tests"], tmp_path / "j.xml") is None
        assert "pytest exited 4" in capsys.readouterr().out

    @pytest.mark.parametrize("code", [0, 1, 5], ids=["all-passed", "tests-failed", "nothing-collected"])
    def test_the_ordinary_pytest_exit_codes_still_report_failures(self, code, tmp_path: Path, monkeypatch):
        # 0/1/5 are the three codes where an absent or empty report really does mean zero failures.
        (tmp_path / "tests").mkdir()
        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"), junit=JUNIT, pytest_returncode=code)
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        assert verify_delta._run_surface(tmp_path, SURFACES[0], ["tests"], tmp_path / "j.xml") == failing_ids(
            tmp_path / "j.xml"
        )

    def test_no_present_paths_reports_that_it_could_not_run_not_that_it_passed(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"))
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        # `set()` here would read as "this tree has no failures", which is a pass it never measured.
        assert verify_delta._run_surface(tmp_path, SURFACES[0], ["tests/testing"], tmp_path / "j.xml") is None
        assert fake.calls == []
        assert "nothing to run" in capsys.readouterr().out


def _reusable_worktree_run(repo: Path, **kwargs) -> _FakeRun:
    """A fake git in which the reused path is a detached worktree of ``repo`` and ``repo`` is primary."""
    defaults = {
        "stdout": str(repo / ".git"),  # both trees report the same .git common dir
        "stdout_by_arg": {"--porcelain": f"worktree {repo}\nHEAD abc1234\nbranch refs/heads/main\n"},
        # `git symbolic-ref -q HEAD` exits non-zero exactly when HEAD is detached
        "fails_for": ("symbolic-ref",),
    }
    return _FakeRun(**{**defaults, **kwargs})


class TestEnsureMainWorktree:
    def test_a_local_base_revision_is_not_fetched(self, tmp_path: Path, monkeypatch):
        fake = _reusable_worktree_run(tmp_path / "repo")
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        verify_delta._ensure_main_worktree(tmp_path / "repo", "HEAD~1", tmp_path, fetch=True)
        assert not any("fetch" in call for call in fake.calls)

    def test_a_remote_base_revision_is_fetched(self, tmp_path: Path, monkeypatch):
        fake = _reusable_worktree_run(tmp_path / "repo")
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        verify_delta._ensure_main_worktree(tmp_path / "repo", "origin/main", tmp_path, fetch=True)
        assert ["git", "-C", str(tmp_path / "repo"), "fetch", "origin", "main"] in fake.calls


class TestBaseRevisionIsResolvedOnce:
    """A relative ``--base`` must not be re-resolved against the reused worktree's own HEAD."""

    def test_the_reused_worktree_checks_out_the_repo_resolved_sha(self, tmp_path: Path, monkeypatch):
        repo, wt = tmp_path / "repo", tmp_path / "wt"
        wt.mkdir()
        fake = _reusable_worktree_run(
            repo,
            stdout_by_arg={
                "--porcelain": f"worktree {repo}\nHEAD abc1234\n",
                "HEAD~1": "deadbee",  # what the relative revision resolves to *in the repo*
            },
        )
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        assert verify_delta._ensure_main_worktree(repo, "HEAD~1", wt, fetch=False) == "deadbee"
        checkout = [call for call in fake.calls if "checkout" in call][-1]
        # `checkout --detach HEAD~1` inside the worktree would resolve against the PREVIOUS run's
        # base and walk one commit further back on every invocation.
        assert checkout[-1] == "deadbee"
        assert "HEAD~1" not in checkout

    def test_a_fresh_worktree_is_added_at_the_resolved_sha(self, tmp_path: Path, monkeypatch):
        repo = tmp_path / "repo"
        fake = _FakeRun(stdout="deadbee")
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        assert verify_delta._ensure_main_worktree(repo, "origin/main", tmp_path / "absent", fetch=False) == "deadbee"
        assert [call for call in fake.calls if "worktree" in call][-1][-1] == "deadbee"


class TestBaseRevisionCannotBeResolved:
    def test_an_unresolvable_base_is_reported_not_raised_raw(self, tmp_path: Path, monkeypatch):
        fake = _FakeRun(fails_for=("rev-parse",), stderr="unknown revision 'origin/nope'")
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="cannot resolve --base origin/nope"):
            verify_delta._ensure_main_worktree(tmp_path / "repo", "origin/nope", tmp_path / "wt", fetch=False)


class TestAssertImportsFrom:
    def test_accepts_kornia_imported_from_the_tree_under_test(self, tmp_path: Path, monkeypatch):
        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"))
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        verify_delta._assert_imports_from(tmp_path, {"PYTHONPATH": str(tmp_path)})  # does not raise

    def test_rejects_kornia_imported_from_another_checkout(self, tmp_path: Path, monkeypatch):
        # the shared .venv re-points its editable kornia install to whichever tree ran uv last
        fake = _FakeRun(stdout=str(tmp_path.parent / "elsewhere" / "kornia" / "__init__.py"))
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="refusing to test the wrong tree"):
            verify_delta._assert_imports_from(tmp_path, {"PYTHONPATH": str(tmp_path)})


class TestEnsureMainWorktreeReuse:
    def test_reuses_a_detached_scratch_worktree_of_this_repo(self, tmp_path: Path, monkeypatch):
        repo, wt = tmp_path / "repo", tmp_path / "wt"
        repo.mkdir()
        wt.mkdir()
        fake = _reusable_worktree_run(repo)
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        verify_delta._ensure_main_worktree(repo, "origin/main", wt, fetch=False)
        # the checkout target is the sha `origin/main` resolved to in the repo, not the ref name
        resolved = str(repo / ".git")  # this fake answers every rev-parse with the same stdout
        assert ["git", "-C", str(wt), "checkout", "--detach", resolved] in fake.calls

    def test_refuses_a_worktree_whose_head_is_on_a_branch(self, tmp_path: Path, monkeypatch):
        # a linked worktree of this same repo, but with a branch checked out: detaching it at `base`
        # would move somebody off the branch they were working on.
        repo, wt = tmp_path / "repo", tmp_path / "wt"
        repo.mkdir()
        wt.mkdir()
        fake = _FakeRun(stdout=str(repo / ".git"))  # symbolic-ref succeeds -> HEAD is attached
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        with pytest.raises(SystemExit, match=rf"{re.escape(str(wt))}.*looks like a user checkout"):
            verify_delta._ensure_main_worktree(repo, "origin/main", wt, fetch=False)
        assert not any("checkout" in call for call in fake.calls)

    def test_refuses_the_repos_own_main_worktree(self, tmp_path: Path, monkeypatch):
        # detached, same repo, but it is the primary worktree -- the user's own checkout.
        repo = tmp_path / "repo"
        repo.mkdir()
        fake = _reusable_worktree_run(repo)
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        with pytest.raises(SystemExit, match=rf"{re.escape(str(repo))}.*looks like a user checkout"):
            verify_delta._ensure_main_worktree(repo, "origin/main", repo, fetch=False)
        assert not any("checkout" in call for call in fake.calls)

    def test_a_checkout_that_fails_is_reported_not_raised_raw(self, tmp_path: Path, monkeypatch):
        repo, wt = tmp_path / "repo", tmp_path / "wt"
        repo.mkdir()
        wt.mkdir()
        fake = _reusable_worktree_run(repo, fails_for=("symbolic-ref", "checkout"), stderr="local changes")
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        with pytest.raises(
            SystemExit, match=rf"{re.escape(str(wt))} is dirty or cannot check out origin/main: local changes"
        ):
            verify_delta._ensure_main_worktree(repo, "origin/main", wt, fetch=False)

    def test_refuses_a_leftover_directory_that_belongs_to_another_repo(self, tmp_path: Path, monkeypatch):
        repo, wt = tmp_path / "repo", tmp_path / "wt"
        repo.mkdir()
        wt.mkdir()
        # `git -C <plain dir>` walks up to whatever repo encloses it, so the checkout would land there
        fake = _FakeRun(stdout_by_arg={str(repo): str(repo / ".git"), str(wt): str(tmp_path / "other" / ".git")})
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="is not a worktree of"):
            verify_delta._ensure_main_worktree(repo, "origin/main", wt, fetch=False)
        assert not any("checkout" in call for call in fake.calls)

    def test_refuses_a_path_that_is_not_in_any_repository(self, tmp_path: Path, monkeypatch):
        repo, wt = tmp_path / "repo", tmp_path / "wt"
        repo.mkdir()
        wt.mkdir()

        def explode(argv, **kwargs):
            if str(wt) in [str(a) for a in argv]:
                raise verify_delta.subprocess.CalledProcessError(128, argv)
            return SimpleNamespace(stdout=str(repo / ".git"), returncode=0)

        monkeypatch.setattr(verify_delta.subprocess, "run", explode)
        with pytest.raises(SystemExit, match="is not a worktree of"):
            verify_delta._ensure_main_worktree(repo, "origin/main", wt, fetch=False)


class TestMainExitCodes:
    """0 verified clean, 1 something is newly failing, 2 nothing was verified at all."""

    @staticmethod
    def _stub_repo(monkeypatch, tmp_path: Path, junit: str | None = None) -> _FakeRun:
        def fake_git(repo, *cmd):
            if cmd[:2] == ("rev-parse", "--show-toplevel"):
                return str(tmp_path)
            if cmd[0] == "diff":
                return ""
            return "abc1234"

        fake = _FakeRun(stdout=str(tmp_path / "kornia" / "__init__.py"), junit=junit)
        monkeypatch.setattr(verify_delta, "_git", fake_git)
        monkeypatch.setattr(verify_delta, "_ensure_main_worktree", lambda *a, **k: "abc1234")
        monkeypatch.setattr(verify_delta.subprocess, "run", fake)
        return fake

    def _argv(self, tmp_path: Path, *extra: str) -> list[str]:
        return [
            "--no-fetch",
            "--main-worktree",
            str(tmp_path / "wt"),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ]

    def test_a_test_path_on_neither_tree_is_not_a_pass(self, tmp_path: Path, monkeypatch, capsys):
        fake = self._stub_repo(monkeypatch, tmp_path)
        code = verify_delta.main(self._argv(tmp_path, "--only", "cpu float32", "--tests", "tests/gone"))
        assert code == 2
        assert "nothing was verified" in capsys.readouterr().out
        assert fake.calls == []

    def test_a_path_only_the_branch_has_is_an_empty_baseline_not_a_refusal(self, tmp_path: Path, monkeypatch, capsys):
        # tests/ exists here but not in the base worktree: the branch side really ran, so its failures
        # are real and unconditionally new -- that is a verdict, not a failure to measure.
        fake = self._stub_repo(monkeypatch, tmp_path, junit=JUNIT)
        (tmp_path / "tests").mkdir()
        code = verify_delta.main(self._argv(tmp_path, "--only", "cpu float32", "--tests", "tests"))
        out = capsys.readouterr().out
        assert code == 1
        assert "no baseline on origin/main for tests; branch failures there are unconditionally new" in out
        assert "| cpu float32 | 2* | 0 | 0 |" in out
        assert "* no baseline on the base revision" in out
        assert "NEW [cpu float32] tests.geometry.test_grid::test_b[cpu-float32]" in out
        assert not any("| cpu float32 | skipped" in line for line in out.splitlines())
        assert sum("pytest" in call for call in fake.calls) == 1  # only the branch side had anything to run

    def test_an_empty_baseline_with_a_clean_branch_still_passes(self, tmp_path: Path, monkeypatch, capsys):
        self._stub_repo(monkeypatch, tmp_path)  # no junit written, so the branch side has no failures
        (tmp_path / "tests").mkdir()
        code = verify_delta.main(self._argv(tmp_path, "--only", "cpu float32", "--tests", "tests"))
        out = capsys.readouterr().out
        assert code == 0
        assert "| cpu float32 | 0* | 0 | 0 |" in out
        assert "no baseline on origin/main" in out

    def test_a_baseline_that_ran_fewer_paths_than_the_branch_is_marked_partial(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        # The base tree has tests/geometry but not tests/testing, so its failing set is missing a
        # whole path. Without the `*` the row reads as a like-for-like comparison it never was.
        self._stub_repo(monkeypatch, tmp_path, junit=JUNIT)
        for tree in (tmp_path, tmp_path / "wt"):
            (tree / "tests" / "geometry").mkdir(parents=True)
        # the import guard answers from argv alone here, so it cannot tell the two trees apart;
        # it has its own tests in TestAssertImportsFrom
        monkeypatch.setattr(verify_delta, "_assert_imports_from", lambda tree, env: None)
        (tmp_path / "tests" / "testing").mkdir()
        code = verify_delta.main(
            self._argv(tmp_path, "--only", "cpu float32", "--tests", "tests/geometry", "tests/testing")
        )
        out = capsys.readouterr().out
        # both trees report the same failures here, so the delta is clean -- the `*` is the whole
        # point: it says the comparison covered fewer paths on the base side than on the branch.
        assert code == 0
        assert "| cpu float32 | 0* | 0 | 2 |" in out
        assert "* no baseline on the base revision" in out

    def test_a_base_tree_where_pytest_could_not_run_is_not_an_empty_baseline(self, tmp_path: Path, monkeypatch, capsys):
        # The base tree HAS the paths, so an unfinished pytest there is a failure to measure, not a
        # legitimately empty baseline -- reading it as empty would report every branch failure NEW.
        fake = self._stub_repo(monkeypatch, tmp_path, junit=JUNIT)
        for tree in (tmp_path, tmp_path / "wt"):
            (tree / "tests").mkdir(parents=True)
        # the import guard answers from argv alone here, so it cannot tell the two trees apart;
        # it has its own tests in TestAssertImportsFrom
        monkeypatch.setattr(verify_delta, "_assert_imports_from", lambda tree, env: None)

        real_run_surface = verify_delta._run_surface

        def only_the_base_tree_crashes(tree, surface, tests, junit):
            if tree == tmp_path / "wt":
                fake.pytest_returncode = 3  # internal error on the base side only
            result = real_run_surface(tree, surface, tests, junit)
            fake.pytest_returncode = 0
            return result

        monkeypatch.setattr(verify_delta, "_run_surface", only_the_base_tree_crashes)
        code = verify_delta.main(self._argv(tmp_path, "--only", "cpu float32", "--tests", "tests"))
        out = capsys.readouterr().out
        assert code == 2
        assert "the base tree did not finish, so this surface was not verified" in out
        assert "| cpu float32 | skipped |" in out
        assert "no baseline on origin/main" not in out

    def test_selecting_only_an_unavailable_surface_is_not_a_pass(self, tmp_path: Path, monkeypatch, capsys):
        import torch

        fake = self._stub_repo(monkeypatch, tmp_path)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        (tmp_path / "tests").mkdir()
        code = verify_delta.main(self._argv(tmp_path, "--only", "mps float32", "--tests", "tests"))
        assert code == 2
        out = capsys.readouterr().out
        assert "| mps float32 | skipped |" in out
        assert "nothing was verified" in out
        assert fake.calls == []

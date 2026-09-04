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
import sys
from pathlib import Path

import pytest

# `.github/scripts/` is neither a package nor on the path. This deliberately stays a per-module
# insert rather than moving to a `tests/scripts/conftest.py`: `tests/` has no `__init__.py`, so a
# second bare `conftest` module shadows the repository-root one, and `tests/test_half_precision_ci.py`
# locates the repository through `Path(conftest.__file__).parent`. Measured: adding that conftest
# makes `import conftest` resolve here and sends five of its workflow assertions looking for
# `tests/scripts/.github/workflows/`.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".github" / "scripts"))

import check_torch_env
from check_torch_env import (
    SIDECAR_PREFIX,
    VERSION_PREFIX,
    check_environment,
    check_torch_version,
    installed_versions,
    main,
    runtime_requirements,
    scan_installed,
    undeclared_sidecars,
)

INSTALL_STEP = "Install the matrix torch and reconcile the environment"

# Verbatim Requires-Dist of torch-2.6.0+cpu-cp311-cp311-linux_x86_64.whl, the leg that #4199
# broke. The CPU wheel index declares no triton; the PyPI wheel of the same version does.
CPU_WHEEL_REQUIRES = [
    "filelock",
    "typing-extensions (>=4.10.0)",
    "networkx",
    "jinja2",
    "fsspec",
    'setuptools ; python_version >= "3.12"',
    'sympy (==1.13.1) ; python_version >= "3.9"',
    "opt-einsum (>=3.3) ; extra == 'opt-einsum'",
    "optree (>=0.13.0) ; extra == 'optree'",
]
PYPI_WHEEL_REQUIRES = [
    *CPU_WHEEL_REQUIRES,
    'triton (==3.2.0) ; platform_system == "Linux" and platform_machine == "x86_64" and python_version < "3.13"',
]

# A Linux x86_64 CPython 3.11 leg, so the markers above resolve the same way everywhere.
LINUX_PY311 = {"python_version": "3.11", "platform_system": "Linux", "platform_machine": "x86_64"}

# What `uv sync` leaves behind once the leg swaps in torch 2.6.0+cpu: the lockfile's triton,
# built for the locked torch 2.14.0, plus torch 2.6.0's own (correctly reinstalled) closure.
MIS_PROVISIONED = {
    "torch": "2.6.0+cpu",
    "triton": "3.8.0",
    "filelock": "3.20.0",
    "typing-extensions": "4.15.0",
    "networkx": "3.5",
    "jinja2": "3.1.6",
    "fsspec": "2025.9.0",
    "sympy": "1.13.1",
}
HEALTHY = {name: version for name, version in MIS_PROVISIONED.items() if name != "triton"}


class TestRuntimeRequirements:
    def test_drops_extras_and_false_markers(self):
        requirements = runtime_requirements(CPU_WHEEL_REQUIRES, LINUX_PY311)

        names = [item.name for item in requirements]
        assert names == ["filelock", "typing-extensions", "networkx", "jinja2", "fsspec", "sympy"]
        # setuptools is python_version >= "3.12"; opt-einsum and optree are extras-only.
        assert "setuptools" not in names
        assert "optree" not in names

    def test_handles_a_distribution_without_requirements(self):
        assert runtime_requirements(None) == []


class TestUndeclaredSidecars:
    def test_reports_the_lockfile_triton_the_cpu_wheel_does_not_declare(self):
        assert undeclared_sidecars(CPU_WHEEL_REQUIRES, MIS_PROVISIONED, LINUX_PY311) == ["triton"]

    def test_keeps_a_triton_the_installed_torch_declares(self):
        assert undeclared_sidecars(PYPI_WHEEL_REQUIRES, MIS_PROVISIONED, LINUX_PY311) == []

    def test_reports_nothing_when_no_sidecar_is_installed(self):
        assert undeclared_sidecars(CPU_WHEEL_REQUIRES, HEALTHY, LINUX_PY311) == []


class TestCheckTorchVersion:
    @pytest.mark.parametrize("version", ["2.6.0", "2.6.0+cpu"])
    def test_pinned_accepts_the_matrix_version_with_or_without_a_local_segment(self, version):
        assert check_torch_version(version, "pinned", "2.6.0") is None

    def test_pinned_rejects_a_version_that_merely_shares_a_prefix(self):
        # `2.14.0`.startswith(`2.1`) is true, so a prefix test would have let this through.
        assert check_torch_version("2.14.0+cpu", "pinned", "2.1") is not None

    def test_pinned_rejects_another_version(self):
        assert "2.6.0" in check_torch_version("2.9.1+cpu", "pinned", "2.6.0")

    def test_stable_treats_the_matrix_version_as_a_floor(self):
        assert check_torch_version("2.14.0+cpu", "stable", "2.9.1") is None
        assert check_torch_version("2.5.1+cpu", "stable", "2.9.1") is not None

    def test_rejects_an_unknown_channel(self):
        with pytest.raises(ValueError, match="unknown torch channel"):
            check_torch_version("2.6.0", "nightly", "2.6.0")


class TestCheckEnvironment:
    def test_flags_the_orphan_triton_that_broke_the_2_6_0_leg(self):
        problems = check_environment("2.6.0+cpu", CPU_WHEEL_REQUIRES, MIS_PROVISIONED, "pinned", "2.6.0", LINUX_PY311)

        assert len(problems) == 1
        assert "triton 3.8.0 is installed" in problems[0]
        assert "#4199" in problems[0]

    def test_accepts_the_reconciled_environment(self):
        assert check_environment("2.6.0+cpu", CPU_WHEEL_REQUIRES, HEALTHY, "pinned", "2.6.0", LINUX_PY311) == []

    def test_flags_a_missing_requirement(self):
        installed = {name: version for name, version in HEALTHY.items() if name != "sympy"}

        problems = check_environment("2.6.0+cpu", CPU_WHEEL_REQUIRES, installed, "pinned", "2.6.0", LINUX_PY311)

        assert len(problems) == 1
        assert "sympy" in problems[0] and "not installed" in problems[0]

    def test_flags_a_requirement_left_at_the_locked_version(self):
        # The failure mode option 1 of #4199 addresses: a dependency the pin step did not move.
        installed = {**HEALTHY, "sympy": "1.14.0"}

        problems = check_environment("2.6.0+cpu", CPU_WHEEL_REQUIRES, installed, "pinned", "2.6.0", LINUX_PY311)

        assert len(problems) == 1
        assert "sympy 1.14.0 is here" in problems[0]

    def test_reports_the_version_mismatch_alongside_the_environment(self):
        problems = check_environment("2.9.1+cpu", CPU_WHEEL_REQUIRES, MIS_PROVISIONED, "pinned", "2.6.0", LINUX_PY311)

        assert len(problems) == 2
        assert "is not the pinned 2.6.0" in problems[0]

    def test_flags_a_distribution_visible_twice(self):
        # What an interrupted `uv pip install --reinstall-package torch` leaves behind: two
        # dist-info directories in one path entry, where first-wins is filesystem order.
        problems = check_environment(
            "2.6.0+cpu",
            CPU_WHEEL_REQUIRES,
            HEALTHY,
            "pinned",
            "2.6.0",
            LINUX_PY311,
            duplicates={"sympy": ["1.13.1", "1.14.0"]},
        )

        assert len(problems) == 1
        assert "sympy is installed more than once" in problems[0]


def test_installed_versions_reports_this_environment():
    installed = installed_versions()

    # Canonical names, so a `typing_extensions`/`typing-extensions` spelling cannot hide a dist.
    assert installed["torch"]
    assert "typing-extensions" in installed


def test_scan_installed_finds_no_duplicates_in_a_coherent_venv():
    installed, duplicates = scan_installed()

    assert installed["torch"]
    assert duplicates == {}


class TestMain:
    """`main` is the surface the workflow greps and gates on; the helpers above are not.

    Both contracts are load-bearing and neither is visible from the functions themselves:
    the stdout prefixes `tests.yml` parses, and the rule that only a *passing* check prints
    a version. Break either and the workflow steps degrade to no-ops that stay green.
    """

    def test_listing_mode_prints_the_prefix_the_workflow_greps(self, capsys):
        assert main(["--list-undeclared-sidecars"]) == 0

        printed = capsys.readouterr().out.splitlines()
        assert [line for line in printed if line.startswith(SIDECAR_PREFIX)]

    def test_a_passing_check_reports_the_version(self, monkeypatch, capsys):
        monkeypatch.setattr(check_torch_env, "check_environment", lambda *args, **kwargs: [])

        assert main(["--channel", "stable", "--expected", "0.0.1"]) == 0

        printed = capsys.readouterr().out
        assert printed.startswith(f"{VERSION_PREFIX} ")

    def test_a_failing_check_reports_no_version(self, monkeypatch, capsys):
        # The regression that made the workflow's own `-z "$resolved"` guard dead code: with
        # the version printed unconditionally, a mis-provisioned venv still handed the step a
        # valid-looking answer, and only the shell's pipefail default failed the leg.
        monkeypatch.setattr(check_torch_env, "check_environment", lambda *args, **kwargs: ["boom"])

        assert main(["--channel", "pinned", "--expected", "2.6.0"]) == 1

        captured = capsys.readouterr()
        assert VERSION_PREFIX not in captured.out
        assert "error: boom" in captured.err

    def test_a_real_version_mismatch_fails_without_a_version_line(self, capsys):
        # The same contract without a monkeypatch: no installed torch is ever 0.0.1.
        assert main(["--channel", "pinned", "--expected", "0.0.1"]) == 1

        assert VERSION_PREFIX not in capsys.readouterr().out

    def test_a_healthy_run_keeps_the_requirement_table_off_the_log(self, monkeypatch, capsys):
        monkeypatch.setattr(check_torch_env, "check_environment", lambda *args, **kwargs: [])

        main(["--channel", "stable", "--expected", "0.0.1"])

        # The table diagnoses nothing when there is nothing wrong, and every leg pays for it.
        # Matched by its indent rather than by an empty stderr, so an unrelated import warning
        # cannot make this pass or fail for the wrong reason.
        assert [line for line in capsys.readouterr().err.splitlines() if line.startswith("  ")] == []

    def test_missing_arguments_error_before_the_environment_is_scanned(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("the environment was scanned before the arguments were validated")

        monkeypatch.setattr(check_torch_env, "scan_installed", fail)

        with pytest.raises(SystemExit) as excinfo:
            main([])

        assert excinfo.value.code == 2


class TestWorkflowWiring:
    """The reusable workflow has to actually run the check; a script nobody calls fixes nothing."""

    @staticmethod
    def _workflow():
        root = Path(__file__).parent.parent.parent
        return (root / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")

    @staticmethod
    def _step(workflow, name):
        """Return one step's body, bounded at the next step rather than running to end of file.

        An unbounded slice is satisfied by any *later* step, so it would keep passing after the
        step it names had been gutted.
        """
        start = workflow.index(f"- name: {name}")
        end = workflow.find("\n      - name:", start + 1)
        return workflow[start : end if end != -1 else len(workflow)]

    def test_the_reconcile_runs_inside_the_step_that_swaps_torch(self):
        # Ordering is the whole fix. Run the check against the *locked* torch -- which does
        # declare triton -- and it finds nothing stale, so the leg swaps in `+cpu` torch
        # afterwards and #4199 survives under a green run. Keeping the reconcile in the swap's
        # own step makes that ordering unreorderable rather than merely conventional.
        workflow = self._workflow()
        install = self._step(workflow, INSTALL_STEP)

        assert "--reinstall-package torch" in install
        assert "--list-undeclared-sidecars" in install
        assert "uv pip uninstall $stale" in install
        assert install.index("--reinstall-package torch") < install.index("--list-undeclared-sidecars")
        assert workflow.index(f"- name: {INSTALL_STEP}") < workflow.index("- name: Run tests")

    def test_the_reconcile_refuses_an_absent_result_line(self):
        # An empty capture means the parse broke, not that the venv is clean: the script always
        # prints the prefix. Without this guard the step reports success while doing nothing.
        install = self._step(self._workflow(), INSTALL_STEP)

        assert f"grep -q '^{SIDECAR_PREFIX}'" in install
        assert "refusing to assume the venv is clean" in install

    def test_verification_step_checks_the_whole_environment(self):
        # Guards the regression #4199 describes: a verify step that only reads torch.__version__
        # passes a venv whose remaining packages belong to a different torch.
        verify = self._step(self._workflow(), "Verify PyTorch version")

        assert "check_torch_env.py" in verify
        assert '--channel "$TORCH_CHANNEL" --expected "$EXPECTED_PYTORCH_VERSION"' in verify
        assert 'echo "resolved=$resolved" >> "$GITHUB_OUTPUT"' in verify
        assert 'if [[ -z "$resolved" ]]; then' in verify

    def test_the_workflow_parses_the_prefixes_the_script_prints(self):
        # The two ends of the stdout contract. Rename a prefix on one side only and both the
        # uninstall and the version gate degrade to silent no-ops.
        workflow = self._workflow()

        assert f"s/^{SIDECAR_PREFIX}" in workflow
        assert f"s/^{VERSION_PREFIX} //p" in workflow

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".github" / "scripts"))
from check_torch_env import (
    check_environment,
    check_torch_version,
    installed_versions,
    runtime_requirements,
    undeclared_sidecars,
)

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


def test_installed_versions_reports_this_environment():
    installed = installed_versions()

    # Canonical names, so a `typing_extensions`/`typing-extensions` spelling cannot hide a dist.
    assert installed["torch"]
    assert "typing-extensions" in installed


class TestWorkflowWiring:
    """The reusable workflow has to actually run the check; a script nobody calls fixes nothing."""

    @staticmethod
    def _workflow():
        root = Path(__file__).parent.parent.parent
        return (root / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")

    def test_reconcile_step_runs_before_the_tests(self):
        workflow = self._workflow()

        assert "--list-undeclared-sidecars" in workflow
        assert "uv pip uninstall $stale" in workflow
        assert workflow.index("--list-undeclared-sidecars") < workflow.index("- name: Run tests")

    def test_verification_step_checks_the_whole_environment(self):
        # Guards the regression #4199 describes: a verify step that only reads torch.__version__
        # passes a venv whose remaining packages belong to a different torch.
        workflow = self._workflow()

        verify = workflow.partition("- name: Verify PyTorch version")[2]
        assert "check_torch_env.py" in verify
        assert '--channel "$TORCH_CHANNEL" --expected "$EXPECTED_PYTORCH_VERSION"' in verify
        assert 'echo "resolved=$resolved" >> "$GITHUB_OUTPUT"' in verify

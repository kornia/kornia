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
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".github" / "scripts"))
from check_import_surface import (
    _changed_kornia_files,
    _is_experimental,
    _module_name,
    check_file,
    diff_surfaces,
    main,
    parse_module_surface,
)


def test_parse_module_surface_collects_defs_imports_and_assignments():
    source = """
import math
from torch import nn as torch_nn

CONST = 1

def foo():
    pass

class Bar:
    pass
"""
    surface = parse_module_surface(source)
    assert surface.other_names == {"math", "torch_nn", "CONST", "foo", "Bar"}
    assert surface.all_names == set()
    assert surface.has_all is False


def test_parse_module_surface_reads_all_list():
    source = """
def foo():
    pass

def _private():
    pass

__all__ = ["foo"]
"""
    surface = parse_module_surface(source)
    assert surface.has_all is True
    assert surface.all_names == {"foo"}
    # Anything in __all__ is excluded from other_names, so the two sets stay disjoint.
    assert surface.other_names == {"_private"}


def test_parse_module_surface_handles_concatenated_all():
    source = """
__all__ = ["a"] + ["b"]
a = 1
b = 2
"""
    surface = parse_module_surface(source)
    assert surface.all_names == {"a", "b"}


def test_diff_surfaces_flags_all_removal():
    old = parse_module_surface("def foo():\n    pass\n\n__all__ = ['foo']\n")
    new = parse_module_surface("__all__ = []\n")
    removed_from_all, removed_undocumented = diff_surfaces(old, new)
    assert removed_from_all == {"foo"}
    assert removed_undocumented == set()


def test_diff_surfaces_flags_undocumented_removal_separately():
    # This is the pyramid.py/#3986 case: `pad` leaked out via an import that
    # was never in __all__, and a later refactor dropped that import.
    old = parse_module_surface("from torch.nn.functional import pad\n\n__all__ = []\n")
    new = parse_module_surface("__all__ = []\n")
    removed_from_all, removed_undocumented = diff_surfaces(old, new)
    assert removed_from_all == set()
    assert removed_undocumented == {"pad"}


def test_diff_surfaces_no_change_reports_nothing():
    old = parse_module_surface("def foo():\n    pass\n")
    new = parse_module_surface("def foo():\n    pass\n\ndef bar():\n    pass\n")
    removed_from_all, removed_undocumented = diff_surfaces(old, new)
    assert removed_from_all == set()
    assert removed_undocumented == set()


def test_parse_module_surface_recurses_into_try_and_if():
    source = """
try:
    import ujson as jsonlib
except ImportError:
    import json as jsonlib

if True:
    def conditional_fn():
        pass
"""
    surface = parse_module_surface(source)
    assert surface.other_names == {"jsonlib", "conditional_fn"}


def test_parse_module_surface_excludes_type_checking_block():
    source = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
"""
    surface = parse_module_surface(source)
    # TYPE_CHECKING is bound (a real runtime import); Tensor is not (never runs).
    assert surface.other_names == {"TYPE_CHECKING"}


def test_parse_module_surface_detects_module_level_getattr():
    source = """
def __getattr__(name):
    raise AttributeError(name)
"""
    surface = parse_module_surface(source)
    assert surface.has_getattr is True


def test_module_name_from_path():
    assert _module_name("kornia/geometry/transform/pyramid.py") == "kornia.geometry.transform.pyramid"
    assert _module_name("kornia/core/__init__.py") == "kornia.core"


def test_is_experimental_matches_contrib_and_submodules():
    assert _is_experimental("kornia.contrib") is True
    assert _is_experimental("kornia.contrib.foo") is True
    assert _is_experimental("kornia.core") is False


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)  # noqa: S603, S607


def test_check_file_end_to_end_against_a_real_git_repo(tmp_path):
    # Build a throwaway repo so this doesn't depend on kornia's own history.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    mod = repo / "kornia" / "mymodule.py"
    mod.write_text("from torch.nn.functional import pad\n\n__all__ = ['foo']\n\ndef foo():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    # Refactor: drop both the undocumented `pad` import and the documented `foo` export.
    mod.write_text("__all__ = []\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "refactor", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        report = check_file("base", "kornia/mymodule.py")
    finally:
        os.chdir(original_cwd)

    assert report is not None
    assert report.removed_from_all == {"foo"}
    assert report.removed_undocumented == {"pad"}
    assert report.fatal is True


def test_check_file_getattr_shim_is_not_fatal(tmp_path):
    # kornia.utils's own pattern: a name leaves __all__ but is still served
    # dynamically via a module-level __getattr__ deprecation shim.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    mod = repo / "kornia" / "mymodule.py"
    mod.write_text("__all__ = ['old_name']\n\ndef old_name():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    mod.write_text("__all__ = []\n\ndef __getattr__(name):\n    raise AttributeError(name)\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "shim", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        report = check_file("base", "kornia/mymodule.py")
    finally:
        os.chdir(original_cwd)

    assert report is not None
    assert report.removed_from_all == {"old_name"}
    assert report.fatal is False


def test_check_file_contrib_removal_is_not_fatal(tmp_path):
    repo = tmp_path / "repo"
    (repo / "kornia" / "contrib").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    mod = repo / "kornia" / "contrib" / "experimental.py"
    mod.write_text("__all__ = ['thing']\n\ndef thing():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    mod.write_text("__all__ = []\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "remove", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        report = check_file("base", "kornia/contrib/experimental.py")
    finally:
        os.chdir(original_cwd)

    assert report is not None
    assert report.removed_from_all == {"thing"}
    assert report.fatal is False


def test_main_uses_merge_base_not_tip_of_base_ref(tmp_path, capsys):
    # Reproduces the review's repro on #4029: the PR branch never touches __all__, but
    # base_ref (main) moves forward *after* the branch point and adds a name to it. Diffing
    # check_file against the tip of base_ref (instead of the merge-base) would blame the PR for
    # a name it never removed. Non-linear on purpose -- the existing linear-history tests can't
    # exercise this, since merge-base and tip coincide when nothing has moved main forward.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("checkout", "-q", "-b", "main", cwd=repo)

    mod = repo / "kornia" / "mod.py"
    mod.write_text("__all__ = ['a']\n\ndef a():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)

    _git("checkout", "-q", "-b", "pr", cwd=repo)
    mod.write_text("# a docstring, __all__ untouched\n__all__ = ['a']\n\ndef a():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "pr change", cwd=repo)

    _git("checkout", "-q", "main", cwd=repo)
    mod.write_text("__all__ = ['a', 'b']\n\ndef a():\n    pass\n\ndef b():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "main moved on", cwd=repo)

    _git("checkout", "-q", "pr", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        exit_code = main(["--base-ref", "main"])
    finally:
        os.chdir(original_cwd)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "b" not in captured.out


def test_check_file_reports_and_fails_whole_module_deletion(tmp_path):
    # Deleting an entire module is the largest version of the break this check exists to catch;
    # it must not pass silently.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    mod = repo / "kornia" / "mymodule.py"
    mod.write_text("__all__ = ['a', 'b']\n\ndef a():\n    pass\n\ndef b():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    _git("rm", "-q", "kornia/mymodule.py", cwd=repo)
    _git("commit", "-q", "-m", "delete module", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        report = check_file("base", "kornia/mymodule.py")
    finally:
        os.chdir(original_cwd)

    assert report is not None
    assert report.removed_from_all == {"a", "b"}
    assert report.fatal is True


def test_check_file_contrib_module_deletion_is_not_fatal(tmp_path):
    repo = tmp_path / "repo"
    (repo / "kornia" / "contrib").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    mod = repo / "kornia" / "contrib" / "experimental.py"
    mod.write_text("__all__ = ['thing']\n\ndef thing():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    _git("rm", "-q", "kornia/contrib/experimental.py", cwd=repo)
    _git("commit", "-q", "-m", "delete module", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        report = check_file("base", "kornia/contrib/experimental.py")
    finally:
        os.chdir(original_cwd)

    assert report is not None
    assert report.removed_from_all == {"thing"}
    assert report.fatal is False


def test_changed_kornia_files_lists_only_python_files_under_kornia(tmp_path):
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    (repo / "docs").mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    (repo / "kornia" / "a.py").write_text("x = 1\n")
    (repo / "docs" / "b.py").write_text("x = 1\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    (repo / "kornia" / "a.py").write_text("x = 2\n")
    (repo / "docs" / "b.py").write_text("x = 2\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "change", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        files = _changed_kornia_files("base")
    finally:
        os.chdir(original_cwd)

    assert files == ["kornia/a.py"]

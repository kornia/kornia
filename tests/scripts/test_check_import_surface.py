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
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".github" / "scripts"))
from check_import_surface import (
    INVENTORY_PATH,
    REMOVALS_PATH,
    _changed_kornia_files,
    _is_experimental,
    _module_name,
    check_file,
    diff_surfaces,
    export_removals,
    inventory_removals,
    main,
    parse_module_surface,
    untracked_modules,
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
    assert captured.out == ""


def test_main_flags_module_removed_via_rename(tmp_path, capsys):
    # Rename detection is on by default for `git diff`, so a renamed module shows up as only
    # its new path -- the old path (and its __all__) never reaches check_file unless the file
    # list is built with --no-renames. A module removed by being renamed away is exactly how
    # modules get removed in practice (packageification refactors), so this must still fail.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("checkout", "-q", "-b", "main", cwd=repo)

    mod = repo / "kornia" / "oldname.py"
    mod.write_text("__all__ = ['a', 'b']\n\ndef a():\n    pass\n\ndef b():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)

    _git("checkout", "-q", "-b", "pr", cwd=repo)
    _git("mv", "kornia/oldname.py", "kornia/newname.py", cwd=repo)
    _git("commit", "-q", "-m", "rename away, dropping the module", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        exit_code = main(["--base-ref", "main"])
    finally:
        os.chdir(original_cwd)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "kornia/oldname.py" in captured.out
    assert "'a'" in captured.out
    assert "'b'" in captured.out


def test_main_ignores_packageification_that_keeps_all_intact(tmp_path, capsys):
    # x.py -> x/__init__.py is the same importable module name (kornia.mymodule either way);
    # check_file's alternate-path fallback must recognize that and not treat it as a deletion.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("checkout", "-q", "-b", "main", cwd=repo)

    mod = repo / "kornia" / "mymodule.py"
    mod.write_text("__all__ = ['a']\n\ndef a():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)

    _git("checkout", "-q", "-b", "pr", cwd=repo)
    (repo / "kornia" / "mymodule").mkdir()
    _git("mv", "kornia/mymodule.py", "kornia/mymodule/__init__.py", cwd=repo)
    _git("commit", "-q", "-m", "packageify, __all__ untouched", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        exit_code = main(["--base-ref", "main"])
    finally:
        os.chdir(original_cwd)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""


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


def test_main_flags_removal_in_a_path_git_would_quote(tmp_path, capsys):
    # git quotes paths containing non-ASCII bytes ("kornia/f\303\266o.py") unless the diff is
    # asked for -z output. A quoted name doesn't end in ".py", so the file dropped out of the
    # file list and its __all__ removal passed silently -- the check's one job, skipped for a
    # reason invisible in the log.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    mod = repo / "kornia" / "f\u00f6o.py"  # escaped to keep this test file pure ASCII
    mod.write_text("__all__ = ['a']\n\ndef a():\n    pass\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    mod.write_text("__all__ = []\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "drop the export", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        exit_code = main(["--base-ref", "base"])
    finally:
        os.chdir(original_cwd)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "'a'" in captured.out


def test_main_flags_removal_that_is_not_committed_yet(tmp_path, capsys):
    # check_file reads the *working tree*, so the file list has to be the working tree's diff
    # against base. Listing a committed range instead (base...HEAD) skips an edit that isn't
    # committed yet -- silently reporting "clean" for the local `--base-ref origin/main` run
    # the module docstring tells contributors to make before pushing.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    mod = repo / "kornia" / "mymodule.py"
    mod.write_text("__all__ = ['a']\n\ndef a():\n    pass\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    mod.write_text("__all__ = []\n")  # edited, deliberately not committed

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        exit_code = main(["--base-ref", "base"])
    finally:
        os.chdir(original_cwd)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "kornia/mymodule.py" in captured.out
    assert "'a'" in captured.out


def test_check_file_honors_a_declared_source_encoding(tmp_path):
    # PEP 263: a module carrying a coding cookie is valid Python whose bytes aren't UTF-8.
    # Reading either side of the comparison as UTF-8 *text* raised UnicodeDecodeError out of
    # check_file and killed the whole run with a traceback; handing ast.parse the bytes lets
    # it apply the cookie exactly as the interpreter does when importing the same file.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    mod = repo / "kornia" / "mymodule.py"
    mod.write_bytes(b"# -*- coding: latin-1 -*-\n__all__ = ['a']\n\ndef a():\n    pass\n# a latin-1 byte: \xe9\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    mod.write_bytes(b"# -*- coding: latin-1 -*-\n__all__ = []\n# a latin-1 byte: \xe9\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "drop the export", cwd=repo)

    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        report = check_file("base", "kornia/mymodule.py")
    finally:
        os.chdir(original_cwd)

    assert report is not None
    assert report.removed_from_all == {"a"}
    assert report.fatal is True


def _repo_with_inventory(tmp_path, *, module="kornia/mymodule.py", exports=("a", "b"), inventory=None):
    """A throwaway repo whose base commit has `exports` in __all__ and in the inventory.

    Returns the repo path with the base commit made and a `base` branch pointing at it.
    The caller edits the working tree and runs the check against `base`.
    """
    repo = tmp_path / "repo"
    (repo / module).parent.mkdir(parents=True, exist_ok=True)
    (repo / INVENTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    body = "".join(f"def {name}():\n    pass\n\n" for name in exports)
    (repo / module).write_text(f"__all__ = {list(exports)!r}\n\n{body}")
    recorded = inventory if inventory is not None else {_module_name(module): sorted(exports)}
    (repo / INVENTORY_PATH).write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")

    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)
    return repo


def _in_repo(repo, fn):
    original_cwd = Path.cwd()
    try:
        os.chdir(repo)
        return fn()
    finally:
        os.chdir(original_cwd)


def _write_inventory(repo, mapping):
    (repo / INVENTORY_PATH).write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")


def _write_removals(repo, mapping):
    (repo / REMOVALS_PATH).write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")


def _repo_with_package(tmp_path, *, init_body, submodules, inventory):
    """A throwaway repo whose base commit has a `kornia/pkg` package re-exporting submodules."""
    repo = tmp_path / "repo"
    (repo / "kornia" / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    (repo / "kornia" / "pkg" / "__init__.py").write_text(init_body)
    for name, body in submodules.items():
        (repo / "kornia" / "pkg" / f"{name}.py").write_text(body)
    _write_inventory(repo, inventory)

    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)
    return repo


def _exports(*names):
    return f"__all__ = {list(names)!r}\n\n" + "".join(f"def {n}():\n    pass\n\n" for n in names)


def test_check_file_ancestor_entry_does_not_excuse_a_sibling_of_the_same_name(tmp_path):
    # kornia.pkg re-exports `thing` from pkg.a; pkg.b has an unrelated symbol of the same
    # name. Recording the package losing `thing` says nothing about pkg.b's, and matching on
    # the spelling alone let the second removal ride along on the first.
    repo = _repo_with_package(
        tmp_path,
        init_body="from .a import thing\nfrom .b import other\n",
        submodules={"a": _exports("thing"), "b": _exports("thing", "other")},
        inventory={"kornia.pkg": ["other", "thing"]},
    )
    (repo / "kornia" / "pkg" / "__init__.py").write_text("from .b import other\n")
    (repo / "kornia" / "pkg" / "b.py").write_text(_exports("other"))
    _write_inventory(repo, {"kornia.pkg": ["other"]})

    report = _in_repo(repo, lambda: check_file("base", "kornia/pkg/b.py", inventory_removals("base"), {}))

    assert report is not None
    assert report.removed_from_all == {"thing"}
    assert report.acknowledged == set()
    assert report.fatal is True


def test_check_file_explicit_entry_excuses_only_its_exact_module(tmp_path):
    # The other half of the same rule: the module the package actually re-exports from is
    # acknowledged, including through a wildcard, or the hatch would not work for
    # kornia.geometry at all.
    repo = _repo_with_package(
        tmp_path,
        init_body="from .a import *\n",
        submodules={"a": _exports("thing", "other")},
        inventory={"kornia.pkg": ["other", "thing"]},
    )
    (repo / "kornia" / "pkg" / "a.py").write_text(_exports("other"))
    _write_inventory(repo, {"kornia.pkg": ["other"]})

    report = _in_repo(
        repo, lambda: check_file("base", "kornia/pkg/a.py", inventory_removals("base"), {"kornia.pkg.a": {"thing"}})
    )

    assert report is not None
    assert report.acknowledged == {"thing"}
    assert report.fatal is False


def test_check_file_inventory_removal_is_not_fatal(tmp_path):
    # #4190's live case (#4143 dropping AUTUMN): the contributor follows the procedure
    # test_no_public_name_removed's own assertion message spells out, and the check has
    # to see that edit rather than hard-failing on a policy-compliant removal.
    repo = _repo_with_inventory(tmp_path)
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")
    _write_inventory(repo, {"kornia.mymodule": ["b"]})

    report = _in_repo(repo, lambda: check_file("base", "kornia/mymodule.py", inventory_removals("base")))

    assert report is not None
    assert report.removed_from_all == {"a"}
    assert report.acknowledged == {"a"}
    assert report.fatal is False


def test_check_file_unrecorded_removal_is_still_fatal(tmp_path):
    repo = _repo_with_inventory(tmp_path)
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")
    # inventory untouched: the removal is not recorded anywhere

    report = _in_repo(repo, lambda: check_file("base", "kornia/mymodule.py", inventory_removals("base")))

    assert report is not None
    assert report.removed_from_all == {"a"}
    assert report.acknowledged == set()
    assert report.fatal is True


def test_check_file_recording_a_different_name_does_not_excuse_this_one(tmp_path):
    repo = _repo_with_inventory(tmp_path)
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")
    _write_inventory(repo, {"kornia.mymodule": ["a"]})  # records 'b' leaving, not 'a'

    report = _in_repo(repo, lambda: check_file("base", "kornia/mymodule.py", inventory_removals("base")))

    assert report is not None
    assert report.removed_from_all == {"a"}
    assert report.acknowledged == set()
    assert report.fatal is True


def test_check_file_partially_recorded_removal_is_fatal_for_the_rest(tmp_path):
    repo = _repo_with_inventory(tmp_path)
    (repo / "kornia" / "mymodule.py").write_text("__all__ = []\n")
    _write_inventory(repo, {"kornia.mymodule": ["b"]})  # records 'a' only

    report = _in_repo(repo, lambda: check_file("base", "kornia/mymodule.py", inventory_removals("base")))

    assert report is not None
    assert report.removed_from_all == {"a", "b"}
    assert report.acknowledged == {"a"}
    assert report.fatal is True


def test_inventory_removals_ignores_a_corrupt_inventory(tmp_path):
    # Fail closed. An unparsable inventory at the new revision makes every recorded name
    # look removed, which would acknowledge the entire public surface at once.
    repo = _repo_with_inventory(tmp_path)
    (repo / INVENTORY_PATH).write_text("{ this is not json\n")

    assert _in_repo(repo, lambda: inventory_removals("base")) == {}


def test_inventory_removals_ignores_a_deleted_inventory(tmp_path):
    repo = _repo_with_inventory(tmp_path)
    (repo / INVENTORY_PATH).unlink()

    assert _in_repo(repo, lambda: inventory_removals("base")) == {}


def test_inventory_removals_ignores_an_inventory_of_the_wrong_shape(tmp_path):
    repo = _repo_with_inventory(tmp_path)
    (repo / INVENTORY_PATH).write_text('["a", "b"]\n')  # a list, not {module: names}

    assert _in_repo(repo, lambda: inventory_removals("base")) == {}


def test_inventory_removals_rejects_an_entry_whose_value_is_not_a_list(tmp_path):
    # Fail closed on a *partially* wrong-shaped inventory too. Skipping the bad entry would
    # drop the module from the parsed mapping, and a dropped module reads exactly like one
    # that lost every name -- so `"kornia.mymodule": {}` would launder every removal in it
    # while test_no_public_name_removed still passes vacuously on `set({}) == set()`.
    repo = _repo_with_inventory(tmp_path)
    _write_inventory(repo, {"kornia.mymodule": {}})

    assert _in_repo(repo, lambda: inventory_removals("base")) == {}


def test_inventory_removals_rejects_an_entry_holding_a_non_string(tmp_path):
    repo = _repo_with_inventory(tmp_path)
    _write_inventory(repo, {"kornia.mymodule": [1, 2]})

    assert _in_repo(repo, lambda: inventory_removals("base")) == {}


def test_inventory_removals_one_bad_entry_rejects_the_whole_inventory(tmp_path):
    # The rejection is not per-entry: an inventory that cannot be trusted at one key cannot
    # be trusted to speak for another, so an honest edit next to a wrong-shaped entry
    # acknowledges nothing either.
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.mymodule": ["a", "b"], "kornia.other": ["z"]})
    _write_inventory(repo, {"kornia.mymodule": ["b"], "kornia.other": {}})

    assert _in_repo(repo, lambda: inventory_removals("base")) == {}


def test_inventory_removals_ignores_a_deleted_module_entry(tmp_path):
    # Deleting a tracked module's key is not an acknowledgement: it would excuse every name
    # recorded for that module at once, and it silently drops the module from
    # test_no_public_name_removed's parametrization, so nothing guards it afterwards.
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.mymodule": ["a", "b"], "kornia.other": ["z"]})
    _write_inventory(repo, {"kornia.other": ["z"]})

    assert _in_repo(repo, lambda: inventory_removals("base")) == {}


def test_untracked_modules_reports_a_deleted_key(tmp_path):
    # Refusing to *acknowledge* a deleted key is only half the guard. The deletion itself has
    # to fail, because it drops the module from test_no_public_name_removed's parametrization
    # and nothing else notices.
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.mymodule": ["a", "b"], "kornia.other": ["z"]})
    _write_inventory(repo, {"kornia.other": ["z"]})

    assert _in_repo(repo, lambda: untracked_modules("base")) == {"kornia.mymodule"}


def test_untracked_modules_ignores_an_emptied_entry(tmp_path):
    # `[]` is the recorded-removal-of-everything spelling and keeps the module tracked, so it
    # must not read as a dropped key -- the two are exactly the cases the error message
    # contrasts.
    repo = _repo_with_inventory(tmp_path)
    _write_inventory(repo, {"kornia.mymodule": []})

    assert _in_repo(repo, lambda: untracked_modules("base")) == set()


def test_untracked_modules_reports_every_module_for_an_emptied_inventory(tmp_path):
    # `{}` parses and is well-shaped, so nothing else rejects it -- and it parametrizes
    # test_no_public_name_removed with zero modules, which passes.
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.mymodule": ["a", "b"], "kornia.other": ["z"]})
    _write_inventory(repo, {})

    assert _in_repo(repo, lambda: untracked_modules("base")) == {"kornia.mymodule", "kornia.other"}


def test_untracked_modules_reports_every_module_for_an_unreadable_inventory(tmp_path):
    # Absent, unparsable or wrong-shaped: the working tree tracks nothing, so nothing is
    # guarded any more.
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.mymodule": ["a", "b"], "kornia.other": ["z"]})
    (repo / INVENTORY_PATH).unlink()

    assert _in_repo(repo, lambda: untracked_modules("base")) == {"kornia.mymodule", "kornia.other"}

    _write_inventory(repo, {"kornia.mymodule": ["a", "b"], "kornia.other": {}})

    assert _in_repo(repo, lambda: untracked_modules("base")) == {"kornia.mymodule", "kornia.other"}


def test_untracked_modules_ignores_an_inventory_absent_from_the_base(tmp_path):
    # A base revision that predates the inventory tracks nothing, so the working tree cannot
    # have dropped anything -- an inventory that is simply not there yet must not read as
    # every module being removed.
    repo = _repo_with_inventory(tmp_path)

    assert _in_repo(repo, lambda: untracked_modules("base", "tests/not_yet_added.json")) == set()


def test_inventory_removals_reads_an_emptied_entry_as_removing_every_name(tmp_path):
    # The remedy the deleted-key rule points at: an empty list records losing every public
    # name while keeping the module tracked by both this check and the pytest inventory.
    repo = _repo_with_inventory(tmp_path)
    _write_inventory(repo, {"kornia.mymodule": []})

    assert _in_repo(repo, lambda: inventory_removals("base")) == {"kornia.mymodule": {"a", "b"}}


def test_main_wrong_shaped_entry_does_not_acknowledge_a_removal(tmp_path, capsys):
    # End to end on the concrete bypass: swap the module's list for `{}` while dropping a
    # name from __all__, and the check must still hard-fail.
    repo = _repo_with_inventory(tmp_path)
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")
    _write_inventory(repo, {"kornia.mymodule": {}})

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "::error" in captured.out
    assert "'a'" in captured.out


def test_main_deleted_module_entry_does_not_acknowledge_a_removal(tmp_path, capsys):
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.mymodule": ["a", "b"], "kornia.other": ["z"]})
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")
    _write_inventory(repo, {"kornia.other": ["z"]})

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "::error" in captured.out
    assert "'a'" in captured.out


def test_main_fails_on_a_deleted_inventory_key_with_no_kornia_change(tmp_path, capsys):
    # The workflow's path filter lists tests/api_surface.json for this: an inventory-only
    # change reaches no kornia file, so the deletion has to fail on its own rather than only
    # as a side effect of some file report.
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.mymodule": ["a", "b"], "kornia.other": ["z"]})
    _write_inventory(repo, {"kornia.other": ["z"]})

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "::error" in captured.out
    assert "'kornia.mymodule'" in captured.out
    assert INVENTORY_PATH in captured.out


def test_main_fails_when_a_wildcard_reexport_and_its_inventory_key_go_together(tmp_path, capsys):
    # The bypass the deleted-key rule exists for. `from .bbox import *` is deliberately
    # invisible to parse_module_surface, so dropping it from a package __init__ produces no
    # report at all; deleting the package's inventory key in the same change used to remove
    # the runtime guard too, and the pair exited 0 in silence. kornia.geometry and
    # kornia.morphology are shaped exactly like this.
    repo = tmp_path / "repo"
    (repo / "kornia" / "geometry").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    pkg = repo / "kornia" / "geometry" / "__init__.py"
    pkg.write_text("from .bbox import *\nfrom .linalg import *\n")
    (repo / "kornia" / "geometry" / "bbox.py").write_text("__all__ = ['bbox_thing']\n\ndef bbox_thing():\n    pass\n")
    (repo / "kornia" / "geometry" / "linalg.py").write_text(
        "__all__ = ['linalg_thing']\n\ndef linalg_thing():\n    pass\n"
    )
    _write_inventory(repo, {"kornia.geometry": ["bbox_thing", "linalg_thing"]})
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    pkg.write_text("from .linalg import *\n")  # kornia.geometry stops exporting bbox_thing
    _write_inventory(repo, {})

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "::error" in captured.out
    assert "'kornia.geometry'" in captured.out


def test_main_fails_on_an_unreadable_inventory_and_says_so(tmp_path, capsys):
    # Same fatal path, different cause, so the message has to name the real one rather than
    # reading as "somebody deleted all twelve keys".
    repo = _repo_with_inventory(tmp_path)
    (repo / INVENTORY_PATH).write_text("{ this is not json\n")

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unparsable" in captured.out
    assert "'kornia.mymodule'" in captured.out


def test_main_fails_on_an_inventory_removal_this_change_does_not_make(tmp_path, capsys):
    # Staging the acknowledgement in its own PR takes the review moment off the PR that
    # removes the API: record `a` leaving now, and a later PR dropping a `from .sub import *`
    # re-export has nothing left to fail on, because wildcards are invisible to the parser.
    repo = _repo_with_inventory(tmp_path)
    _write_inventory(repo, {"kornia.mymodule": ["b"]})  # no kornia/ file touched

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "::error" in captured.out
    assert "exact resolved export delta" in captured.out
    assert "'a'" in captured.out


def test_main_untracked_name_removal_does_not_prescribe_an_impossible_edit(tmp_path, capsys):
    # The inventory records twelve package aggregates, so 132 public __all__ names under them
    # are in no entry (#4236). Telling their remover to "drop the same name from the
    # inventory" sends them after an edit that cannot be made.
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.other": ["z"]})
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert REMOVALS_PATH in captured.out
    assert "drop these names" not in captured.out


def test_main_exits_zero_and_reports_a_recorded_removal(tmp_path, capsys):
    repo = _repo_with_inventory(tmp_path)
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")
    _write_inventory(repo, {"kornia.mymodule": ["b"]})

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "::notice" in captured.out
    assert INVENTORY_PATH in captured.out
    assert "'a'" in captured.out
    assert "::error" not in captured.out


def test_main_error_message_names_the_escape_hatch(tmp_path, capsys):
    # #4190: the pytest assertion tells contributors the procedure and this check did not,
    # so a contributor could satisfy the test that names it and still be blocked here with
    # no idea what to do about it.
    repo = _repo_with_inventory(tmp_path)
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "::error" in captured.out
    assert INVENTORY_PATH in captured.out
    assert REMOVALS_PATH not in captured.out


def test_main_still_reports_an_undocumented_removal_alongside_a_recorded_one(tmp_path, capsys):
    # An undocumented removal (the pyramid.py/#3986 case) was never fatal and is reported
    # for visibility. The inventory hatch must not swallow that line: it speaks only for
    # names that were in __all__, and the two reports are independent.
    repo = tmp_path / "repo"
    (repo / "kornia").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    mod = repo / "kornia" / "mymodule.py"
    mod.write_text("from torch.nn.functional import pad\n\n__all__ = ['a']\n\ndef a():\n    pass\n")
    _write_inventory(repo, {"kornia.mymodule": ["a"]})
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    mod.write_text("__all__ = []\n")  # drops the documented 'a' AND the undocumented 'pad'
    _write_inventory(repo, {"kornia.mymodule": []})  # records only 'a'
    _write_removals(repo, {"kornia.mymodule": ["a"]})

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "::error" not in captured.out
    assert "'pad'" in captured.out
    assert "No longer bound at module scope" in captured.out


def test_main_rejects_inventory_staging_beside_an_unrelated_package_touch(tmp_path, capsys):
    repo = _repo_with_package(
        tmp_path,
        init_body="from .a import *\n",
        submodules={"a": _exports("thing"), "b": "# unrelated\n"},
        inventory={"kornia.pkg": ["thing"]},
    )
    _write_inventory(repo, {"kornia.pkg": []})
    (repo / "kornia" / "pkg" / "b.py").write_text("# still unrelated\n")

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    assert exit_code == 1
    assert "exact resolved export delta" in capsys.readouterr().out


def test_export_removals_follows_nested_wildcards_without_authorizing_a_sibling(tmp_path):
    repo = tmp_path / "repo"
    (repo / "kornia" / "pkg" / "a").mkdir(parents=True)
    (repo / "tests").mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "kornia" / "pkg" / "__init__.py").write_text("from .a import *\n")
    (repo / "kornia" / "pkg" / "a" / "__init__.py").write_text("from .real import *\n")
    (repo / "kornia" / "pkg" / "a" / "real.py").write_text(_exports("thing"))
    (repo / "kornia" / "pkg" / "a" / "unrelated.py").write_text(_exports("thing"))
    _write_inventory(repo, {"kornia.pkg": ["thing"]})
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)

    (repo / "kornia" / "pkg" / "a" / "real.py").write_text(_exports())
    removed = _in_repo(repo, lambda: export_removals("base", ["kornia/pkg/a/real.py"]))

    assert removed["kornia.pkg"] == {"thing"}
    assert removed["kornia.pkg.a"] == {"thing"}
    assert removed["kornia.pkg.a.real"] == {"thing"}
    assert "kornia.pkg.a.unrelated" not in removed
    (repo / "kornia/pkg/a/unrelated.py").write_text(_exports())
    _write_inventory(repo, {"kornia.pkg": []})
    _write_removals(repo, {"kornia.pkg.a.real": ["thing"]})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 1
    _write_removals(repo, {"kornia.pkg.a.real": ["thing"], "kornia.pkg.a.unrelated": ["thing"]})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 0


def test_explicit_acknowledgement_requires_an_exact_same_module_removal(tmp_path, capsys):
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.other": ["z"]})
    _write_removals(repo, {"kornia.mymodule": ["a"]})

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    assert exit_code == 1
    assert "exact __all__ removal" in capsys.readouterr().out


def test_explicit_acknowledgement_opens_a_route_for_uninventoried_api(tmp_path, capsys):
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.other": ["z"]})
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")
    _write_removals(repo, {"kornia.mymodule": ["a"]})

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    assert exit_code == 0
    assert REMOVALS_PATH in capsys.readouterr().out


def test_malformed_explicit_acknowledgement_fails_closed(tmp_path, capsys):
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.other": ["z"]})
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")
    (repo / REMOVALS_PATH).write_text('{"kornia.mymodule": {}}\n')

    exit_code = _in_repo(repo, lambda: main(["--base-ref", "base"]))

    assert exit_code == 1
    assert "malformed acknowledgement" in capsys.readouterr().out


def test_explicit_acknowledgement_is_new_only_and_can_be_cleaned_up(tmp_path, capsys):
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.other": ["z"]})
    _write_removals(repo, {"kornia.mymodule": ["a"]})
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "old acknowledgement", cwd=repo)
    _git("branch", "-f", "base", "HEAD", cwd=repo)
    (repo / "kornia" / "mymodule.py").write_text("__all__ = ['b']\n\ndef b():\n    pass\n")

    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 1
    assert "cannot be reused" in capsys.readouterr().out
    (repo / "kornia" / "mymodule.py").write_text(_exports("a", "b"))
    _write_removals(repo, {})  # restoring an API clears its old acknowledgement.
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 0
    assert capsys.readouterr().out == ""


def test_mixed_removals_have_separate_actionable_remedies(tmp_path, capsys):
    repo = _repo_with_inventory(tmp_path, inventory={"kornia.mymodule": ["a"]})
    (repo / "kornia/mymodule.py").write_text("__all__ = []\n")
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 1
    errors = [line for line in capsys.readouterr().out.splitlines() if "Removed from __all__" in line]
    assert len(errors) == 2
    assert "['a']" in errors[0] and INVENTORY_PATH in errors[0] and REMOVALS_PATH not in errors[0]
    assert "['b']" in errors[1] and REMOVALS_PATH in errors[1]


def test_explicit_record_does_not_replace_exact_inventory_update(tmp_path, capsys):
    repo = _repo_with_inventory(tmp_path)
    (repo / "kornia/mymodule.py").write_text(_exports("b"))
    _write_removals(repo, {"kornia.mymodule": ["a"]})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 1
    assert "do not replace" in capsys.readouterr().out


def test_explicit_sibling_removal_preserves_unrelated_package_export(tmp_path, capsys):
    repo = _repo_with_package(
        tmp_path,
        init_body="from .a import thing\n",
        submodules={"a": _exports("thing"), "b": _exports("thing")},
        inventory={"kornia.pkg": ["thing"]},
    )
    (repo / "kornia/pkg/b.py").write_text(_exports())
    _write_removals(repo, {"kornia.pkg.b": ["thing"]})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 0
    assert "::error" not in capsys.readouterr().out


def test_inventory_acknowledges_an_unbound_all_entry(tmp_path, capsys):
    repo = _repo_with_package(
        tmp_path,
        init_body="__all__ = ['AUTUMN']\n",
        submodules={},
        inventory={"kornia.pkg": ["AUTUMN"]},
    )
    (repo / "kornia/pkg/__init__.py").write_text("__all__ = []\n")
    _write_inventory(repo, {"kornia.pkg": []})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 0
    assert INVENTORY_PATH in capsys.readouterr().out


def test_dynamic_all_cannot_prove_an_inventory_removal(tmp_path, capsys):
    repo = _repo_with_inventory(tmp_path)
    (repo / "kornia/mymodule.py").write_text("__all__ = ['b']\n__all__.append('a')\n")
    _write_inventory(repo, {"kornia.mymodule": ["b"]})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 1
    assert "exact resolved export delta" in capsys.readouterr().out


@pytest.mark.parametrize(
    "import_line",
    [
        "from .a import *",
        "from .a import thing",
        "from .a import thing as alias",
        "import kornia.pkg.a as alias",
        "from . import a as alias",
    ],
)
def test_inventory_acknowledges_implicit_submodule_removal(tmp_path, capsys, import_line):
    alias = "alias" if "as alias" in import_line else "thing"
    repo = _repo_with_package(
        tmp_path,
        init_body=import_line + "\n",
        submodules={"a": _exports("thing")},
        inventory={"kornia.pkg": ["a", alias]},
    )
    (repo / "kornia/pkg/__init__.py").write_text('"""An empty package."""\npass\n')
    (repo / "kornia/pkg/a.py").unlink()
    _write_inventory(repo, {"kornia.pkg": []})
    _write_removals(repo, {"kornia.pkg.a": ["thing"]})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 0
    assert "::error" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "expression",
    [
        "globals().update(thing=1)",
        "(thing := 1)",
        "sentinel = (thing := 1)",
        "sentinel = globals().update(thing=1)",
        "def f(value=(thing := 1)): pass",
        "@(globals().update(thing=1) or (lambda f: f))\ndef f(): pass",
        "class C:\n    globals().update(thing=1)",
    ],
)
def test_dynamic_binding_cannot_stage_inventory_removal(tmp_path, capsys, expression):
    repo = _repo_with_package(
        tmp_path,
        init_body="thing = 1\n",
        submodules={},
        inventory={"kornia.pkg": ["thing"]},
    )
    (repo / "kornia/pkg/__init__.py").write_text(expression + "\n")
    _write_inventory(repo, {"kornia.pkg": []})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 1
    assert "exact resolved export delta" in capsys.readouterr().out


def test_retained_submodule_binding_cannot_stage_inventory_removal(tmp_path, capsys):
    repo = _repo_with_package(
        tmp_path,
        init_body="from .a import *\n",
        submodules={"a": _exports("thing")},
        inventory={"kornia.pkg": ["a", "thing"]},
    )
    (repo / "kornia/pkg/__init__.py").write_text("from .a import thing\n")
    _write_inventory(repo, {"kornia.pkg": ["thing"]})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 1
    assert "exact resolved export delta" in capsys.readouterr().out


def test_function_body_does_not_make_package_exports_unknown(tmp_path):
    repo = _repo_with_package(
        tmp_path,
        init_body="thing = 1\n",
        submodules={},
        inventory={"kornia.pkg": ["thing"]},
    )
    (repo / "kornia/pkg/__init__.py").write_text("def f():\n    globals().update(thing=1)\n")
    _write_inventory(repo, {"kornia.pkg": []})
    assert _in_repo(repo, lambda: main(["--base-ref", "base"])) == 0

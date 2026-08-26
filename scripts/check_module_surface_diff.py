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
"""Flag module-level names that a change removes from a ``kornia/`` module.

Refactors sometimes drop a name that used to be bound at module scope --
either a documented export (listed in ``__all__``) or a name that was merely
importable because of how the module happened to be written (e.g. ``from
kornia.core import ..., pad, ...`` made ``pad`` importable from that module
too, even though it was never in ``__all__``). Either kind of removal can
silently break third-party code.

This script statically compares the set of module-level names bound in each
changed ``kornia/**/*.py`` file between a base revision and the working tree,
using ``ast`` only -- it never imports kornia, so it works even when the
environment can't build/run the package.

Usage::

    python scripts/check_module_surface_diff.py --base-ref origin/main

Exit status is 1 only when a name is removed from ``__all__`` (a break of
documented public API). Names that disappear from module scope but were
never in ``__all__`` are reported for visibility but do not fail the check.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class ModuleSurface:
    """Names bound at module scope, split into documented and undocumented."""

    all_names: set[str] = field(default_factory=set)
    """Names listed in this module's ``__all__``, if it defines one."""

    other_names: set[str] = field(default_factory=set)
    """Other names bound at module scope (defs, assignments, imports)."""

    has_all: bool = False
    """Whether the module defines ``__all__`` at all."""


def _string_literals(node: ast.AST) -> list[str]:
    """Extract string literals from a list/tuple/set literal AST node."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    return [elt.value for elt in node.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]


def parse_module_surface(source: str) -> ModuleSurface:
    """Parse a module's source and collect the names it binds at module scope."""
    surface = ModuleSurface()
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            surface.other_names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                surface.other_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                surface.other_names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            is_all_assignment = any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets)
            if is_all_assignment:
                surface.has_all = True
                value = node.value
                if isinstance(value, ast.BinOp):
                    # Tolerate `__all__ = [...] + [...]`.
                    for side in (value.left, value.right):
                        surface.all_names.update(_string_literals(side))
                else:
                    surface.all_names.update(_string_literals(value))
                continue
            for t in targets:
                if isinstance(t, ast.Name):
                    surface.other_names.add(t.id)

    # Anything in __all__ is, by definition, also bound at module scope --
    # keep other_names as "everything else" so the two sets are disjoint.
    surface.other_names -= surface.all_names
    return surface


def _git_show(ref: str, path: str) -> str | None:
    """Return the file's content at `ref`, or None if it didn't exist there."""
    # git is the trusted tool this script is built around.
    result = subprocess.run(  # noqa: S603
        ["git", "show", f"{ref}:{path}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _changed_kornia_files(base_ref: str) -> list[str]:
    # Git's "**" glob pathspec only matches when there's at least one directory
    # between "kornia/" and the filename, so "kornia/foo.py" itself wouldn't
    # match "kornia/**/*.py". Scope to the "kornia/" pathspec instead and
    # filter for ".py" ourselves.
    result = subprocess.run(  # noqa: S603
        ["git", "diff", "--name-only", f"{base_ref}...HEAD", "--", "kornia/"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.endswith(".py")]


@dataclass
class FileReport:
    path: str
    removed_from_all: set[str]
    removed_undocumented: set[str]


def diff_surfaces(old: ModuleSurface, new: ModuleSurface) -> tuple[set[str], set[str]]:
    """Return (removed_from_all, removed_undocumented) between two surfaces."""
    removed_from_all = old.all_names - new.all_names
    removed_undocumented = old.other_names - new.other_names - new.all_names
    return removed_from_all, removed_undocumented


def check_file(base_ref: str, path: str) -> FileReport | None:
    """Compare `path`'s module surface between `base_ref` and the working tree.

    Returns None if there's nothing to report (new/deleted file, unparseable
    source, or no names removed).
    """
    old_source = _git_show(base_ref, path)
    if old_source is None:
        return None  # new file, nothing to compare against

    try:
        with open(path, encoding="utf-8") as f:
            new_source = f.read()
    except FileNotFoundError:
        return None  # file was deleted; not this check's concern

    try:
        old_surface = parse_module_surface(old_source)
        new_surface = parse_module_surface(new_source)
    except SyntaxError:
        return None

    removed_from_all, removed_undocumented = diff_surfaces(old_surface, new_surface)

    if not removed_from_all and not removed_undocumented:
        return None
    return FileReport(path=path, removed_from_all=removed_from_all, removed_undocumented=removed_undocumented)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main", help="Git ref to diff against (default: origin/main)")
    args = parser.parse_args(argv)

    files = _changed_kornia_files(args.base_ref)
    reports = [r for r in (check_file(args.base_ref, path) for path in files) if r is not None]

    hard_fail = False
    for report in reports:
        if report.removed_from_all:
            hard_fail = True
            print(f"::error file={report.path}::Removed from __all__: {sorted(report.removed_from_all)}")
        if report.removed_undocumented:
            names = sorted(report.removed_undocumented)
            print(
                f"::notice file={report.path}::No longer bound at module scope (was never in "
                f"__all__, so this is informational): {names}"
            )

    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())

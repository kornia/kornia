#!/usr/bin/env python3

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
silently break third-party code. See #3986.

This script statically compares the set of module-level names bound in each
changed ``kornia/**/*.py`` file between a base revision and the working tree,
using ``ast`` only -- it never imports kornia, so it works even when the
environment can't build/run the package.

Usage::

    python3 .github/scripts/check_import_surface.py --base-ref origin/main

Exit status is 1 only when a name is removed from ``__all__`` (a break of
documented public API, per the stability policy). Names that disappear from
module scope but were never in ``__all__`` are reported for visibility but do
not fail the check -- hard-failing on those would make every incidental
third-party re-export permanent API. Two exemptions from the hard-fail path:

- A module with a module-level ``__getattr__`` in the new revision is
  presumed to be using it as a deprecation shim (the pattern
  ``kornia.utils`` uses) -- its removals are reported, not fatal.
- ``kornia.contrib`` is the Experimental tier per
  ``docs/source/get-started/stability.rst`` ("No stability promise") --
  its removals are reported, not fatal.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass, field


def _is_type_checking(test: ast.expr) -> bool:
    """Whether an `if` test is (a reference to) `typing.TYPE_CHECKING`."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


@dataclass
class ModuleSurface:
    """Names bound at module scope, split into documented and undocumented."""

    all_names: set[str] = field(default_factory=set)
    """Names listed in this module's ``__all__``, if it defines one."""

    other_names: set[str] = field(default_factory=set)
    """Other names bound at module scope (defs, assignments, imports)."""

    has_all: bool = False
    """Whether the module defines ``__all__`` at all."""

    has_getattr: bool = False
    """Whether the module defines a module-level ``__getattr__`` (deprecation shim)."""


def _string_literals(node: ast.AST) -> list[str]:
    """Extract string literals from a list/tuple/set literal AST node."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    return [elt.value for elt in node.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]


def parse_module_surface(source: str | bytes) -> ModuleSurface:
    """Parse a module's source and collect the names it binds at module scope.

    Accepts bytes as well as text, and hands them to `ast.parse` undecoded so
    that a PEP 263 coding cookie is honored exactly as the interpreter honors
    it -- the same source Python can import is the source this can parse.

    Recurses into ``try``/``except``/``else``/``finally`` and plain ``if``
    bodies, since names bound there (e.g. a version-conditional import) are
    still real module-scope bindings. The one deliberate exception is an
    ``if TYPE_CHECKING:`` body: those names are never bound at runtime, so
    they are neither surface nor regression.
    """
    surface = ModuleSurface()

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.If):
                walk(node.orelse if _is_type_checking(node.test) else node.body + node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body + node.orelse + node.finalbody)
                for handler in node.handlers:
                    walk(handler.body)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == "__getattr__":
                    surface.has_getattr = True
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
                    elif value is not None:
                        surface.all_names.update(_string_literals(value))
                    continue
                for t in targets:
                    if isinstance(t, ast.Name):
                        surface.other_names.add(t.id)

    walk(ast.parse(source).body)

    # Anything in __all__ is, by definition, also bound at module scope --
    # keep other_names as "everything else" so the two sets are disjoint.
    surface.other_names -= surface.all_names
    return surface


def _git_show(ref: str, path: str) -> bytes | None:
    """Return the file's bytes at `ref`, or None if it didn't exist there.

    Bytes, not text: decoding is `ast.parse`'s job (see `parse_module_surface`),
    and decoding here with the platform locale (e.g. cp1252 on Windows) would
    raise on non-ASCII source bytes that Python itself reads fine.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "show", f"{ref}:{path}"],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _changed_kornia_files(base_ref: str) -> list[str]:
    """Paths under `kornia/` that differ between `base_ref` and the working tree."""
    # Git's "**" glob pathspec only matches when there's at least one directory
    # between "kornia/" and the filename, so "kornia/foo.py" itself wouldn't
    # match "kornia/**/*.py". Scope to the "kornia/" pathspec instead and
    # filter for ".py" ourselves.
    #
    # --no-renames: with rename detection on (the default), a moved module
    # (`git mv old.py new.py`) shows up as only `new.py` -- the old path never
    # reaches check_file, so a module deleted by being renamed away is never
    # diffed against its old __all__. check_file's own FileNotFoundError
    # fallback (_alternate_path) still tolerates the one legitimate rename
    # shape, `x.py` <-> `x/__init__.py` packageification, so this doesn't
    # trade the rename blind spot for a packageification false positive.
    #
    # No `...HEAD`: check_file reads the *working tree*, so the file list has
    # to be the working tree's diff against base too. Committed-range diffing
    # would silently skip a removal that is edited but not yet committed --
    # invisible for the local runs this script documents, identical to
    # `base...HEAD` on CI's clean checkout (base is already the merge-base).
    #
    # -z: without it, git quotes (and octal-escapes) any path holding
    # non-ASCII or special bytes, so the name arrives wrapped in double
    # quotes, fails the ".py" filter below, and drops out of the check
    # entirely. -z also removes the need to un-escape such a name.
    result = subprocess.run(  # noqa: S603
        ["git", "diff", "--no-renames", "--name-only", "-z", base_ref, "--", "kornia/"],  # noqa: S607
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=True,
    )
    return [name for name in result.stdout.split("\0") if name.endswith(".py")]


# docs/source/get-started/stability.rst puts kornia.contrib in the Experimental
# tier: "No stability promise." Removals there are reported, never fatal.
EXPERIMENTAL = ("kornia.contrib",)


def _module_name(path: str) -> str:
    stem = path[: -len(".py")]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def _is_experimental(module: str) -> bool:
    return any(module == e or module.startswith(e + ".") for e in EXPERIMENTAL)


def _read_source(path: str) -> bytes:
    """Read a working-tree file as bytes, the way `_git_show` reads the base revision.

    Both ends of the comparison must handle the same bytes. Decoding as UTF-8 text
    here would raise on a module Python imports happily under a PEP 263 cookie, and
    abort the whole check with a traceback rather than reporting that one file.
    """
    with open(path, "rb") as f:
        return f.read()


def _alternate_path(path: str) -> str:
    """The other file spelling of the same module name (`x.py` <-> `x/__init__.py`)."""
    stem = path[: -len(".py")]
    if stem.endswith("/__init__"):
        return stem[: -len("/__init__")] + ".py"
    return stem + "/__init__.py"


@dataclass
class FileReport:
    path: str
    removed_from_all: set[str]
    removed_undocumented: set[str]
    fatal: bool
    """Whether removed_from_all should actually fail the check for this file."""


def diff_surfaces(old: ModuleSurface, new: ModuleSurface) -> tuple[set[str], set[str]]:
    """Return (removed_from_all, removed_undocumented) between two surfaces."""
    removed_from_all = old.all_names - new.all_names
    removed_undocumented = old.other_names - new.other_names - new.all_names
    return removed_from_all, removed_undocumented


def check_file(base_ref: str, path: str) -> FileReport | None:
    """Compare `path`'s module surface between `base_ref` and the working tree.

    A deleted file is diffed against an empty surface, so removing a whole
    module is reported (and can be fatal) just like emptying its `__all__`.

    Returns None if there's nothing to report (new file, unparsable source,
    or no names removed).
    """
    old_source = _git_show(base_ref, path)
    if old_source is None:
        return None  # new file, nothing to compare against

    try:
        new_source: bytes | None = _read_source(path)
    except FileNotFoundError:
        # The path is gone, but the *module* may not be: x.py <-> x/__init__.py is the same
        # importable name (packageification), not a removal. Only try the one alternate
        # spelling -- anything else missing here is a real deletion.
        try:
            new_source = _read_source(_alternate_path(path))
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            new_source = None  # module genuinely gone -- diff against an empty surface

    try:
        old_surface = parse_module_surface(old_source)
        new_surface = parse_module_surface(new_source) if new_source is not None else ModuleSurface()
    except (SyntaxError, ValueError):
        # SyntaxError: not valid Python at one end of the diff (a template file, or
        # bytes that don't decode under the module's declared encoding).
        # ValueError: source ast.parse rejects outright, e.g. embedded null bytes.
        return None

    removed_from_all, removed_undocumented = diff_surfaces(old_surface, new_surface)

    if not removed_from_all and not removed_undocumented:
        return None

    # A module-level __getattr__ in the new revision is presumed to be
    # serving these names as a deprecation shim (kornia.utils's pattern) --
    # don't punish the one thing done right.
    fatal = bool(removed_from_all) and not new_surface.has_getattr and not _is_experimental(_module_name(path))
    return FileReport(
        path=path,
        removed_from_all=removed_from_all,
        removed_undocumented=removed_undocumented,
        fatal=fatal,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main", help="Git ref to diff against (default: origin/main)")
    args = parser.parse_args(argv)

    # _changed_kornia_files diffs against the merge-base (triple-dot); check_file must read old
    # content from that same commit, not the tip of base_ref, or the two can disagree whenever
    # base_ref has moved since the branch point (see review discussion on #4029).
    merge_base = subprocess.run(  # noqa: S603
        ["git", "merge-base", args.base_ref, "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if merge_base.returncode != 0:
        print(f"::error::could not resolve merge-base of {args.base_ref!r} and HEAD")
        return 1
    base = merge_base.stdout.strip()

    try:
        files = _changed_kornia_files(base)
    except subprocess.CalledProcessError as exc:
        # `base` came from merge-base, so this is close to unreachable -- but a check that
        # can't list its files must fail loudly and closed, not traceback (or pass empty).
        print(f"::error::could not list changed files against {base}: {exc.stderr.strip() or exc}")
        return 1

    reports = [r for r in (check_file(base, path) for path in files) if r is not None]

    hard_fail = False
    for report in reports:
        if report.removed_from_all:
            names = sorted(report.removed_from_all)
            if report.fatal:
                hard_fail = True
                print(f"::error file={report.path}::Removed from __all__: {names}")
            else:
                print(f"::notice file={report.path}::Removed from __all__ (exempt, see comment above): {names}")
        if report.removed_undocumented:
            names = sorted(report.removed_undocumented)
            print(
                f"::notice file={report.path}::No longer bound at module scope (was never in "
                f"__all__, so this is informational): {names}"
            )

    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())

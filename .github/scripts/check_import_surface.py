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

Exit status is 1 when a name is removed from ``__all__`` (a break of documented
public API, per the stability policy), when the change drops a module that
``tests/api_surface.json`` tracks, or when a removal record does not match an
actual export delta in this change. Names that disappear from module scope but
were never in ``__all__`` are reported for visibility but do not fail the check
-- hard-failing on those would make every incidental third-party re-export
permanent API. Three exemptions from the hard-fail path:

- A module with a module-level ``__getattr__`` in the new revision is
  presumed to be using it as a deprecation shim (the pattern
  ``kornia.utils`` uses) -- its removals are reported, not fatal.
- ``kornia.contrib`` is the Experimental tier per
  ``docs/source/get-started/stability.rst`` ("No stability promise") --
  its removals are reported, not fatal.
- An exact module/name removal recorded in this change is reported, not fatal.
  Drop the name from that module's ``tests/api_surface.json`` entry, or add an
  exact pair to ``tests/api_surface_removals.json`` for a submodule or API the
  inventory does not cover. Ancestor inventory entries never authorize a
  descendant's removal. Existing inventory coverage still needs updating.
  Only newly added explicit pairs count, and each must match a direct
  ``__all__`` removal in this diff. See #4190.

Dropping a *module* from that inventory is the mirror image and is always
fatal: its key is what puts the module in
``test_no_public_name_removed``'s parametrization, so deleting the key ends
that guard silently, and this check cannot stand in for it -- a package whose
public names arrive through ``from .sub import *`` has no static ``__all__``
here to compare.
"""

from __future__ import annotations

import argparse
import ast
import json
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


# tests/api_surface.json is the checked-in inventory of the stable-core modules'
# public names, and tests/test_api_surface.py::test_no_public_name_removed tells a
# contributor to edit it when a deprecation window has passed. Dropping a name from
# it in the same change is therefore the acknowledgement this check honors (#4190).
INVENTORY_PATH = "tests/api_surface.json"
REMOVALS_PATH = "tests/api_surface_removals.json"


def _parse_inventory(source: bytes | None) -> dict[str, set[str]] | None:
    """Parse the api_surface.json inventory into {module: names}.

    Returns ``None`` -- distinct from an empty mapping -- for anything that is not
    exactly ``dict[str, list[str]]``: absent, invalid JSON, a different top-level
    structure, an entry whose key is not a string, an entry whose value is not a list,
    or a list holding anything but strings. The distinction is the whole safety
    property. An unreadable inventory at the new revision makes every recorded name
    look *removed*, which would open the escape hatch for the entire surface; the
    caller turns ``None`` into "acknowledge nothing" instead.

    Rejecting one bad *entry* rather than skipping it matters just as much as rejecting
    a bad top-level value: skipping ``"kornia.color": {}`` would drop that module from
    the parsed mapping, and a dropped module reads downstream exactly like one that lost
    every name -- so a single wrong-shaped entry would launder every removal in it.
    """
    if source is None:
        return None
    try:
        data = json.loads(source)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    inventory: dict[str, set[str]] = {}
    for module, names in data.items():
        if not isinstance(module, str) or not isinstance(names, list):
            return None
        if not all(isinstance(name, str) for name in names):
            return None
        inventory[module] = set(names)
    return inventory


def _parse_removals(source: bytes | None, *, missing_is_empty: bool = False) -> dict[str, set[str]] | None:
    """Parse the explicit removal acknowledgement, rejecting every ambiguous shape."""
    if source is None:
        return {} if missing_is_empty else None
    return _parse_inventory(source)


def _removal_ends(base_ref: str) -> tuple[dict[str, set[str]] | None, dict[str, set[str]] | None]:
    old = _parse_removals(_git_show(base_ref, REMOVALS_PATH), missing_is_empty=True)
    try:
        new_source: bytes | None = _read_source(REMOVALS_PATH)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        new_source = None
    return old, _parse_removals(new_source, missing_is_empty=True)


def removal_acknowledgements(base_ref: str) -> dict[str, set[str]] | None:
    """Return acknowledgement names newly added by this diff, or ``None`` if invalid.

    Only additions grant approval.  Old entries may be removed when an API is
    restored, so that a later removal receives its own review moment.
    """
    old, new = _removal_ends(base_ref)
    if old is None or new is None:
        return None
    return {module: names - old.get(module, set()) for module, names in new.items() if names - old.get(module, set())}


def _inventory_ends(
    base_ref: str, inventory_path: str
) -> tuple[dict[str, set[str]] | None, dict[str, set[str]] | None]:
    """The parsed inventory at `base_ref` and in the working tree (`None` = unusable).

    Working tree, not `HEAD`, for the same reason `_changed_kornia_files` diffs the
    working tree: an uncommitted inventory edit has to count during a local run, and
    on CI's clean checkout the two are identical.
    """
    old = _parse_inventory(_git_show(base_ref, inventory_path))
    try:
        new_source: bytes | None = _read_source(inventory_path)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        new_source = None
    return old, _parse_inventory(new_source)


def untracked_modules(base_ref: str, inventory_path: str = INVENTORY_PATH) -> set[str]:
    """Modules the base inventory tracks that the working tree's inventory no longer does.

    Every key in the inventory is a stable-core module that
    ``tests/test_api_surface.py::test_no_public_name_removed`` imports and compares against
    the live library. Deleting a key does not fail that test -- it removes the module from
    its parametrization, so the module simply stops being guarded, quietly and permanently.
    This check cannot stand in for it either: a package that re-exports through
    ``from .sub import *`` (``kornia.geometry``, ``kornia.morphology``) has no ``__all__``
    for `diff_surfaces` to compare, so dropping such a re-export produces no report at all.
    The two holes line up, which is why the key deletion itself has to be fatal.

    An inventory that is unreadable at the working tree -- absent, unparsable, or
    wrong-shaped anywhere -- stops tracking *every* module, so it returns all of them.
    An inventory unreadable at `base_ref` tracks nothing to begin with, so nothing is lost.
    """
    old, new = _inventory_ends(base_ref, inventory_path)
    if old is None:
        return set()
    if new is None:
        return set(old)
    return set(old) - set(new)


def inventory_removals(base_ref: str, inventory_path: str = INVENTORY_PATH) -> dict[str, set[str]]:
    """Names each inventory module lost between `base_ref` and the working tree.

    Only a name dropped from an entry that is well-shaped at *both* ends counts; every
    other way the inventory can change -- unreadable, wrong-shaped, or a tracked module
    key deleted -- acknowledges nothing, because each of those makes recorded names look
    removed without anyone recording them. The last two are also fatal in their own right
    (`untracked_modules`, and `_parse_inventory` for the shape); acknowledging nothing is
    what keeps them from laundering an `__all__` removal on the way out.
    """
    old, new = _inventory_ends(base_ref, inventory_path)

    if old is None or new is None:
        # Deleting or corrupting the inventory is not a way to acknowledge a removal:
        # without a readable file at *both* ends there is no recorded intent to read.
        return {}

    removals: dict[str, set[str]] = {}
    for module, names in old.items():
        if module not in new:
            # Deleting a tracked module's entry is not an acknowledgement either: it would
            # make every name recorded for that module look removed at once. `main` fails
            # the check on the deletion itself (see `untracked_modules`); refusing to
            # acknowledge here is what stops the same edit from excusing an `__all__`
            # removal in passing. A change that really removes every public name of a module
            # records that with an empty list, which keeps the module tracked by both this
            # check and test_no_public_name_removed.
            continue
        gone = names - new[module]
        if gone:
            removals[module] = gone
    return removals


class _ExportResolver:
    """Resolve static export membership without importing the library.

    This is intentionally a small, conservative interpreter.  It follows only
    ordered module-level imports, assignments, definitions, and literal
    ``__all__`` declarations.  A conditional, dynamic export, missing module,
    or wildcard import it cannot resolve makes that module unknown; an unknown
    surface never proves that an inventory name was removed.
    """

    def __init__(self, base_ref: str | None) -> None:
        self.base_ref = base_ref
        self.cache: dict[str, set[str] | None] = {}
        self.visiting: set[str] = set()

    def _source(self, module: str) -> tuple[bytes, bool] | None:
        stem = module.replace(".", "/")
        paths = ((f"{stem}/__init__.py", True), (f"{stem}.py", False))
        for path, is_package in paths:
            try:
                source = _git_show(self.base_ref, path) if self.base_ref is not None else _read_source(path)
            except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
                source = None
            if source is not None:
                return source, is_package
        return None

    @staticmethod
    def _literal_all(value: ast.expr | None) -> list[str] | None:
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            left = _ExportResolver._literal_all(value.left)
            right = _ExportResolver._literal_all(value.right)
            return left + right if left is not None and right is not None else None
        if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return None
        names = _string_literals(value)
        return names if len(names) == len(value.elts) else None

    def _relative(self, module: str, is_package: bool, node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            return node.module
        package = module if is_package else module.rpartition(".")[0]
        parts = package.split(".") if package else []
        if node.level - 1 > len(parts):
            return None
        base = ".".join(parts[: len(parts) - (node.level - 1)])
        return f"{base}.{node.module}" if node.module else base

    def resolve(self, module: str) -> set[str] | None:
        if module in self.cache:
            return self.cache[module]
        if module in self.visiting:
            return None
        self.visiting.add(module)
        found = self._source(module)
        if found is None:
            result = None
        else:
            source, is_package = found
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                result = None
            else:
                result = self._resolve_tree(tree, module, is_package)
        self.visiting.discard(module)
        self.cache[module] = result
        return result

    def _resolve_tree(self, tree: ast.Module, module: str, is_package: bool) -> set[str] | None:
        # A literal __all__ is the export contract by itself.  Do this
        # before resolving imports: package membership must not become
        # unknowable because an unrelated dependency is dynamic.
        all_values = [
            node.value
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        ]

        def touches_all(node: ast.AST) -> bool:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                return False
            return any(isinstance(child, ast.Name) and child.id == "__all__" for child in ast.walk(node))

        mutates_all = any(
            touches_all(node)
            and not (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                )
            )
            for node in tree.body
        )
        if mutates_all:
            return None
        if all_values:
            names = self._literal_all(all_values[-1])
            result = None if names is None else set(names)
            return result
        return self._bound_exports(tree, module, is_package)

    def _bound_exports(self, tree: ast.Module, module: str, is_package: bool) -> set[str] | None:
        bindings: set[str] = set()
        result = bindings
        for node in tree.body:
            if isinstance(node, ast.If):
                if _is_type_checking(node.test) and not node.orelse:
                    continue
                result = None
                break
            if isinstance(node, (ast.Try, ast.With, ast.For, ast.While, ast.Match)):
                result = None
                break
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    bindings.add(bound)
            elif isinstance(node, ast.ImportFrom):
                target = self._relative(module, is_package, node)
                if target is None:
                    result = None
                    break
                # Third-party imports are terminal bindings.  Following
                # torch/typing/__future__ would make ordinary Kornia
                # modules unknowable for reasons unrelated to their own
                # re-export provenance.
                for alias in node.names:
                    if alias.name == "*":
                        imported = self.resolve(target) if target == "kornia" or target.startswith("kornia.") else None
                        if imported is None:
                            result = None
                            break
                        bindings.update(imported)
                    else:
                        bound = alias.asname or alias.name
                        # Explicit imports bind their alias regardless of
                        # the source module's __all__ or implementation.
                        bindings.add(bound)
                if result is None:
                    break
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        bindings.add(target.id)
                    else:
                        result = None
                        break
            elif isinstance(node, (ast.Delete, ast.Global, ast.Nonlocal, ast.AugAssign)):
                result = None
                break
        if result is not None:
            result = {name for name in bindings if not name.startswith("_")}
        return result


def export_removals(base_ref: str, paths: list[str]) -> dict[str, set[str]]:
    """Exact module/name export deltas caused by this change.

    Include package ancestors of a changed module: removing a leaf from a
    wildcard re-export changes the package's public surface even if its
    ``__init__.py`` text is unchanged.
    """
    modules: set[str] = set()
    for path in paths:
        module = _module_name(path)
        parts = module.split(".")
        modules.update(".".join(parts[:depth]) for depth in range(1, len(parts) + 1))
    old = _ExportResolver(base_ref)
    new = _ExportResolver(None)
    removed: dict[str, set[str]] = {}
    for module in modules:
        old_exports = old.resolve(module)
        new_exports = new.resolve(module)
        if old_exports is None or new_exports is None:
            continue
        names = set(old_exports) - set(new_exports)
        if names:
            removed[module] = names
    return removed


@dataclass
class FileReport:
    path: str
    removed_from_all: set[str]
    removed_undocumented: set[str]
    fatal: bool
    """Whether removed_from_all should actually fail the check for this file."""

    acknowledged: set[str] = field(default_factory=set)
    """Subset of removed_from_all acknowledged by either record file."""


def diff_surfaces(old: ModuleSurface, new: ModuleSurface) -> tuple[set[str], set[str]]:
    """Return (removed_from_all, removed_undocumented) between two surfaces."""
    removed_from_all = old.all_names - new.all_names
    removed_undocumented = old.other_names - new.other_names - new.all_names
    return removed_from_all, removed_undocumented


def check_file(
    base_ref: str,
    path: str,
    removals: dict[str, set[str]] | None = None,
    explicit_removals: dict[str, set[str]] | None = None,
) -> FileReport | None:
    """Compare `path`'s module surface between `base_ref` and the working tree.

    A deleted file is diffed against an empty surface, so removing a whole
    module is reported (and can be fatal) just like emptying its `__all__`.

    `removals` is `inventory_removals`' mapping for the same diff; a name it
    records for this exact module is reported rather than fatal. Explicit records
    supply the same exact-module acknowledgement for APIs outside the inventory.

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

    module = _module_name(path)
    # Only exact module/name pairs authorize removals. An ancestor's inventory
    # entry cannot identify which same-named descendant API was removed.
    acknowledged = removed_from_all & (
        (removals or {}).get(module, set()) | (explicit_removals or {}).get(module, set())
    )

    # A module-level __getattr__ in the new revision is presumed to be
    # serving these names as a deprecation shim (kornia.utils's pattern) --
    # don't punish the one thing done right. Same for a removal this change
    # already recorded in the inventory: only the *unrecorded* ones are fatal.
    fatal = bool(removed_from_all - acknowledged) and not new_surface.has_getattr and not _is_experimental(module)
    return FileReport(
        path=path,
        removed_from_all=removed_from_all,
        removed_undocumented=removed_undocumented,
        fatal=fatal,
        acknowledged=acknowledged,
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

    removals = inventory_removals(base)
    base_inventory = _inventory_ends(base, INVENTORY_PATH)[0] or {}
    explicit_removals = removal_acknowledgements(base)
    old_explicit = _removal_ends(base)[0] or {}
    exports_removed = export_removals(base, files)
    reports = [
        r
        for r in (
            check_file(base, path, removals, explicit_removals if explicit_removals is not None else {})
            for path in files
        )
        if r is not None
    ]

    hard_fail = False

    dropped = untracked_modules(base)
    if dropped:
        hard_fail = True
        if _inventory_ends(base, INVENTORY_PATH)[1] is None:
            # Absent, unparsable, or wrong-shaped anywhere: it records nothing for anyone now.
            print(
                f"::error file={INVENTORY_PATH}::{INVENTORY_PATH} is missing, unparsable, or "
                f"wrong-shaped, so it tracks nothing and every module it recorded "
                f"({sorted(dropped)}) stops being guarded. It has to stay a JSON object mapping "
                f"each stable-core module to a list of its public names."
            )
        else:
            print(
                f"::error file={INVENTORY_PATH}::No longer tracked in {INVENTORY_PATH}: "
                f"{sorted(dropped)}. Each key there is a stable-core module that "
                f"tests/test_api_surface.py::test_no_public_name_removed imports and guards, so "
                f"dropping the key ends that guard instead of failing it -- and this check cannot "
                f"replace it, because a package that re-exports through `from .sub import *` has "
                f"no __all__ here to compare. To record removing every public name of a module, "
                f"set its list to [] -- that keeps the module tracked. Removing a stable-core "
                f"module itself is a policy decision (docs/source/get-started/stability.rst), not "
                f"a routine recorded removal."
            )

    # An inventory row is aggregate package coverage, so it is only meaningful
    # when the package's *own* resolved export surface loses that exact name.
    # A touched sibling is no evidence; unknown static membership is deliberately
    # not treated as a match.
    unmatched_inventory = {
        module: names - exports_removed.get(module, set())
        for module, names in removals.items()
        if names - exports_removed.get(module, set())
    }
    if unmatched_inventory:
        hard_fail = True
        listed = ", ".join(f"{module}: {sorted(names)}" for module, names in sorted(unmatched_inventory.items()))
        print(
            f"::error file={INVENTORY_PATH}::Recorded removals without an exact resolved export delta in "
            f"this change ({listed}). A touched file under the package is insufficient. Keep the inventory "
            f"entry until the package export is statically demonstrably removed; dynamic exports need a "
            f"maintainer-reviewed guard change rather than this acknowledgement."
        )

    if explicit_removals is None:
        hard_fail = True
        print(
            f"::error file={REMOVALS_PATH}::{REMOVALS_PATH} must be a JSON object mapping modules to lists "
            f"of strings. A malformed acknowledgement authorizes nothing."
        )
        explicit_removals = {}

    direct_removed: dict[str, set[str]] = {}
    for report in reports:
        direct_removed.setdefault(_module_name(report.path), set()).update(report.removed_from_all)
    for module, names in direct_removed.items():
        missing_inventory = (names & base_inventory.get(module, set())) - removals.get(module, set())
        if missing_inventory & explicit_removals.get(module, set()):
            hard_fail = True
            print(
                f"::error file={INVENTORY_PATH}::Explicit acknowledgements do not replace this module's "
                f"inventory update: remove {sorted(missing_inventory)} from {module}'s entry in the same change."
            )
    invalid_explicit = {
        module: names - direct_removed.get(module, set())
        for module, names in explicit_removals.items()
        if names - direct_removed.get(module, set())
    }
    if invalid_explicit:
        hard_fail = True
        listed = ", ".join(f"{module}: {sorted(names)}" for module, names in sorted(invalid_explicit.items()))
        print(
            f"::error file={REMOVALS_PATH}::Explicit acknowledgements must match an exact __all__ removal "
            f"from the same module in this diff; unmatched entries: {listed}."
        )

    for report in reports:
        if report.acknowledged:
            explicit = report.acknowledged & explicit_removals.get(_module_name(report.path), set())
            acknowledgement_file = REMOVALS_PATH if explicit else INVENTORY_PATH
            print(
                f"::notice file={report.path}::Removed from __all__ and acknowledged in "
                f"{acknowledgement_file} in this same change: {sorted(report.acknowledged)}"
            )
        unrecorded = sorted(report.removed_from_all - report.acknowledged)
        if unrecorded:
            if report.fatal:
                hard_fail = True
                tracked = set(unrecorded) & base_inventory.get(_module_name(report.path), set())
                untracked = set(unrecorded) - tracked
                if tracked:
                    print(
                        f"::error file={report.path}::Removed from __all__: {sorted(tracked)}. If deliberate, "
                        f"drop these names from this module's {INVENTORY_PATH} entry in this PR. "
                        f"Honor the deprecation window and document the removal in CHANGELOG.md."
                    )
                stale = untracked & old_explicit.get(_module_name(report.path), set())
                if stale:
                    print(
                        f"::error file={REMOVALS_PATH}::These acknowledgements already existed at the merge base "
                        f"and cannot be reused: {sorted(stale)}. Clear obsolete records in a separate cleanup "
                        f"change before recording this removal; restoration of an API should clear its old record."
                    )
                untracked -= stale
                if untracked:
                    print(
                        f"::error file={report.path}::Removed from __all__: {sorted(untracked)}. Add these exact "
                        f"module/name pairs to {REMOVALS_PATH}; they have no exact-module inventory entry. "
                        f"Update affected package inventory entries too, honor the deprecation window, "
                        f"and document the removal in CHANGELOG.md."
                    )
            else:
                print(f"::notice file={report.path}::Removed from __all__ (exempt, see comment above): {unrecorded}")
        if report.removed_undocumented:
            names = sorted(report.removed_undocumented)
            print(
                f"::notice file={report.path}::No longer bound at module scope (was never in "
                f"__all__, so this is informational): {names}"
            )

    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())

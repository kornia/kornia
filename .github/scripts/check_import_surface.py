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

"""Check that a change does not silently remove an importable name.

Written after #3986, where ``kornia.geometry.transform.pyramid.pad`` stopped
existing. It was never in ``__all__`` and never documented -- it was only
``torch.nn.functional.pad`` leaking out of a ``from kornia.core import ...,
pad, ...`` line -- so when the module switched to calling ``F.pad`` directly,
the binding went with it. ComfyUI-LTXVideo was broken for three months.

``pad`` was not special. The same release removed all 32 names from
``kornia.core.__all__`` (#4024), which took 824 module-scope bindings with it
across the tree, and a performance PR deleted the documented ``upscale_double``
(#4025). None of it was visible in review, because nothing compared the export
surface between two revisions.

This script is that comparison. It is stdlib-only and never imports kornia, so
it runs on any interpreter, needs no environment, and can read any git revision
through ``git show``.

Two modes:

``regression``
    Public module-scope names importable at ``--old`` and gone at ``--new``.
    By default only losses of names that were in their module's ``__all__``
    fail the run; undeclared losses are reported but do not fail. That split is
    deliberate and matches the stability policy: what a module does not declare
    is not API. Use ``--fail-on any`` to make every loss fatal.

``inventory``
    The undeclared public surface at one revision, graded:

    ``L1a``  re-exported third-party/stdlib name (the #3986 class)
    ``L1b``  re-exported kornia-internal name
    ``L2``   defined here, module has ``__all__`` and excludes it (#4026)
    ``L3``   module declares no ``__all__`` at all
    ``X``    name in ``__all__`` with no module-scope binding -- a live bug,
             since ``from <module> import *`` raises (#4023)

Names bound only under ``if TYPE_CHECKING:`` are ignored throughout: they are
not importable at runtime, so they are neither surface nor regression. A module
with a module-level ``__getattr__`` is treated as shimmed -- that is the
correct deprecation pattern (``kornia.utils`` uses it) and the check must not
punish it.

Usage::

    python3 .github/scripts/check_import_surface.py regression --old main --new HEAD
    python3 .github/scripts/check_import_surface.py regression --old v0.8.2 --new v0.8.3
    python3 .github/scripts/check_import_surface.py inventory
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Binding:
    """One name bound at module scope, and how it got there."""

    name: str
    kind: str  # import-external | import-kornia | def | class | assign
    origin: str  # module the name was imported from; "" for def/class/assign
    lineno: int


@dataclass
class Surface:
    """The module-scope name surface of a single module."""

    module: str
    path: str
    has_all: bool = False
    has_getattr: bool = False
    all_names: set[str] = field(default_factory=set)
    bindings: dict[str, Binding] = field(default_factory=dict)

    @property
    def public(self) -> dict[str, Binding]:
        return {n: b for n, b in self.bindings.items() if not n.startswith("_")}


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _string_literals(node: ast.expr) -> list[str] | None:
    """Collect the string literals of an ``__all__`` value, or None if not literal."""
    out: list[str] = []
    stack: list[ast.AST] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.List, ast.Tuple, ast.Set)):
            stack.extend(cur.elts)
        elif isinstance(cur, ast.BinOp):
            stack.extend([cur.left, cur.right])
        elif isinstance(cur, ast.Constant) and isinstance(cur.value, str):
            out.append(cur.value)
        else:
            return out or None
    return out


def collect(source: str, module: str, path: str) -> Surface | None:
    """Parse one module and return the names it binds at module scope."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    surf = Surface(module=module, path=path)

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.If):
                # A TYPE_CHECKING guard binds nothing at runtime; its else does.
                walk(node.orelse if _is_type_checking(node.test) else node.body + node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body + node.orelse + node.finalbody)
                for handler in node.handlers:
                    walk(handler.body)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                _record_import(surf, node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "__getattr__":
                    surf.has_getattr = True
                surf.bindings[node.name] = Binding(node.name, "def", "", node.lineno)
            elif isinstance(node, ast.ClassDef):
                surf.bindings[node.name] = Binding(node.name, "class", "", node.lineno)
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                _record_assign(surf, node)

    walk(tree.body)
    return surf


def _record_import(surf: Surface, node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.ImportFrom):
        origin = ("." * node.level) + (node.module or "")
        internal = node.level > 0 or (node.module or "").startswith("kornia")
    else:
        origin = ""
        internal = False
    for alias in node.names:
        if alias.name == "*":
            continue
        bound = alias.asname or alias.name.split(".")[0]
        source_module = origin or alias.name
        kind = "import-kornia" if internal or source_module.startswith("kornia") else "import-external"
        surf.bindings[bound] = Binding(bound, kind, source_module, node.lineno)


def _record_assign(surf: Surface, node: ast.Assign | ast.AugAssign | ast.AnnAssign) -> None:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        if not isinstance(target, ast.Name):
            continue
        if target.id == "__all__":
            names = _string_literals(node.value) if node.value is not None else None
            if names is not None:
                surf.has_all = True
                surf.all_names.update(names)
            continue
        surf.bindings[target.id] = Binding(target.id, "assign", "", node.lineno)


def _module_name(path: str) -> str:
    stem = path[: -len(".py")]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def load(rev: str | None, root: str) -> dict[str, Surface]:
    """Collect every module surface under ``root``, at ``rev`` or in the working tree."""
    surfaces: dict[str, Surface] = {}
    if rev:
        listing = subprocess.run(  # noqa: S603
            ["git", "ls-tree", "-r", "--name-only", rev, root],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        paths = [p for p in listing.splitlines() if p.endswith(".py")]
    else:
        paths = [str(p) for p in sorted(Path(root).rglob("*.py"))]

    for path in paths:
        try:
            source = (
                subprocess.run(  # noqa: S603
                    ["git", "show", f"{rev}:{path}"],  # noqa: S607
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                if rev
                else Path(path).read_text(encoding="utf-8")
            )
        except (subprocess.CalledProcessError, UnicodeDecodeError):
            continue
        surface = collect(source, _module_name(path), path)
        if surface is not None:
            surfaces[surface.module] = surface
    return surfaces


# docs/source/get-started/stability.rst puts kornia.contrib in the Experimental tier:
# "No stability promise." Removals there are reported, never fatal.
EXPERIMENTAL = ("kornia.contrib",)


def _experimental(module: str) -> bool:
    return any(module == e or module.startswith(e + ".") for e in EXPERIMENTAL)


def _grade(surf: Surface, binding: Binding) -> str:
    if surf.has_all and binding.name in surf.all_names:
        return ""
    if binding.kind == "import-external":
        return "L1a"
    if binding.kind == "import-kornia":
        return "L1b"
    return "L2" if surf.has_all else "L3"


def inventory(surfaces: dict[str, Surface]) -> int:
    """Report the undeclared public surface; non-zero when a declared name does not resolve."""
    buckets: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    broken: list[tuple[str, str]] = []
    without_all = 0

    for module, surf in sorted(surfaces.items()):
        without_all += not surf.has_all
        if not surf.has_getattr:
            broken.extend((module, n) for n in sorted(surf.all_names) if n not in surf.bindings)
        for name, binding in sorted(surf.public.items()):
            grade = _grade(surf, binding)
            if grade:
                buckets[grade][module].append(name)

    print(f"modules scanned: {len(surfaces)} · without __all__: {without_all}\n")
    titles = {
        "L1a": "re-exported third-party/stdlib name",
        "L1b": "re-exported kornia-internal name",
        "L2": "defined here, module has __all__ and excludes it",
        "L3": "module declares no __all__",
    }
    for grade, title in titles.items():
        rows = buckets.get(grade, {})
        total = sum(len(v) for v in rows.values())
        print(f"{grade} — {title}: {total} names in {len(rows)} modules")
    print(f"X  — in __all__ with no module-scope binding: {len(broken)}")
    for module, name in broken:
        print(f"     {module}.{name}  ('from {module} import *' would raise AttributeError)")
    return 1 if broken else 0


def regression(old: dict[str, Surface], new: dict[str, Surface], old_rev: str, new_rev: str, fail_on: str) -> int:
    """Report public module-scope names importable at old_rev and gone at new_rev."""
    declared: list[tuple[str, str, str]] = []
    undeclared: list[tuple[str, str, str]] = []
    experimental: list[tuple[str, str, str]] = []

    for module, old_surf in sorted(old.items()):
        new_surf = new.get(module)
        if new_surf is not None and new_surf.has_getattr:
            continue  # a deprecation shim may resolve names the AST cannot see
        current = new_surf.public if new_surf else {}
        for name, binding in sorted(old_surf.public.items()):
            if name in current:
                continue
            row = (module, name, binding.kind)
            if _experimental(module):
                experimental.append(row)
            elif old_surf.has_all and name in old_surf.all_names:
                declared.append(row)
            else:
                undeclared.append(row)

    print(f"import surface: {old_rev} -> {new_rev}\n")
    print(f"declared (in __all__) names removed: {len(declared)}")
    for module, name, kind in declared:
        print(f"  {module}.{name}  (was bound by: {kind})")
    print(f"\nundeclared public names removed: {len(undeclared)}")
    for module, name, kind in undeclared[:40]:
        print(f"  {module}.{name}  (was bound by: {kind})")
    if len(undeclared) > 40:
        print(f"  ... and {len(undeclared) - 40} more")
    if experimental:
        print(f"\nexperimental-tier (kornia.contrib) names removed, not fatal: {len(experimental)}")

    if declared:
        print(
            "\nA name in __all__ was removed. Per the API stability policy "
            "(docs/source/get-started/stability.rst) a public symbol spends at least one minor "
            "release as a deprecated shim before removal. If the deprecation window has passed, "
            "record the removal in the release notes in this PR."
        )
    if undeclared:
        print(
            "\nThe undeclared names above are not API under the stability policy and do not fail "
            "this check. They are listed because dropping one is how #3986 happened: if any is "
            "plausibly relied on downstream, keep it or give it a deprecation shim."
        )
    if fail_on == "any":
        return 1 if declared or undeclared else 0
    return 1 if declared else 0


def main() -> int:
    """Parse arguments and run the requested mode."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["regression", "inventory"])
    parser.add_argument("--root", default="kornia", help="package directory to scan (default: kornia)")
    parser.add_argument("--rev", help="inventory: git revision to scan (default: the working tree)")
    parser.add_argument("--old", help="regression: the revision to compare from")
    parser.add_argument("--new", default="HEAD", help="regression: the revision to compare to (default: HEAD)")
    parser.add_argument(
        "--fail-on",
        choices=["declared", "any"],
        default="declared",
        help="regression: fail on __all__ removals only (default) or on every removal",
    )
    args = parser.parse_args()

    if args.mode == "inventory":
        return inventory(load(args.rev, args.root))
    if not args.old:
        parser.error("regression requires --old")
    return regression(load(args.old, args.root), load(args.new, args.root), args.old, args.new, args.fail_on)


if __name__ == "__main__":
    raise SystemExit(main())

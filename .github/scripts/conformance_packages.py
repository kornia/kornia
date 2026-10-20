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
"""Map a pull request's changed paths to the package selection of its conformance runs (#5819).

``python conformance_packages.py onnx`` prints the ``--onnx-conformance`` value, ``python conformance_packages.py
compile`` the ``--compile-conformance`` one. Reads the changed paths on stdin (one per line, as ``git diff --name-only``
prints them, or separated by any whitespace: no kornia path contains a space) and prints ``all``, a comma-separated list
of stable-core packages, or nothing when no change can affect a promised operator (the conformance job is then
skipped).

A changed module under ``kornia/`` selects the stable-core package it belongs to and every stable-core package with a
module that imports it, directly or through other modules. The import graph is read from the checkout with ``ast``:
``import kornia.x.y``, ``from kornia.x import y``, relative imports, and ``kornia.x.y`` attribute chains after a bare
``import kornia``, at module granularity (importing ``kornia.x.y`` depends on ``kornia/x/y.py`` or on the package
``kornia/x/y/__init__.py``, which depends on what it imports in turn). So a change to ``kornia/metrics/psnr.py`` also
selects ``losses``, whose PSNR losses call it, while a ``kornia/contrib`` module no stable-core module reaches selects
nothing.

``all`` is selected by code every package runs (``kornia/core``, ``kornia/utils``, ``kornia/onnx``, the modules
directly under ``kornia/``), by the sources of the promises and their tests (the survey snapshot, the stability page,
the survey under ``docs/export_support/``, ``testing/``, the promise's own test, the root ``conftest.py``, this script,
the workflows that run it) and by the dependency pins. The two promises differ only in their own test file and
known-failure manifests (``testing/known_failure_xfails/<onnx|compile>_conformance_*``), which select ``all`` for their
promise and nothing for the other.

Standard library only: the ``changes`` job runs it before any environment is installed.
"""

from __future__ import annotations

import argparse
import ast
import functools
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The stable core of docs/source/get-started/stability.rst; the same tuple as testing/conformance.py, which
# tests/scripts/test_conformance_packages.py checks.
STABLE_CORE = (
    "geometry",
    "augmentation",
    "color",
    "filters",
    "enhance",
    "losses",
    "metrics",
    "morphology",
    "io",
    "image",
    "core",
    "feature",
)

# Code every package runs, or the promise and the machinery that checks it.
ALL_PREFIXES = (
    "kornia/core/",
    "kornia/utils/",
    "kornia/onnx/",
    "docs/export_support/",
    "testing/",
)
ALL_FILES = (
    "docs/source/_data/export_support.json",
    "docs/source/get-started/stability.rst",
    ".github/scripts/conformance_packages.py",
    ".github/workflows/tests.yml",
    ".github/workflows/pr_test_cpu.yml",
    ".github/workflows/scheduled_test_cpu.yml",
    "conftest.py",
    "pyproject.toml",
    "uv.lock",
)
# Each promise's own test, by the snapshot field it promises (testing/conformance.py PROMISES).
TEST_FILES = {
    "onnx": "tests/onnx/test_conformance.py",
    "compile": "tests/test_inductor_conformance.py",
}
# Each promise's known-failure manifests: an edit selects all of that promise and nothing of the other.
MANIFEST_PREFIX = "testing/known_failure_xfails/{}_conformance_"


def module_index(root: Path) -> dict[str, Path]:
    """Map every module under ``root/kornia`` to its file: ``kornia.x.y`` for ``x/y.py`` or ``x/y/__init__.py``."""
    index: dict[str, Path] = {}
    for path in sorted((root / "kornia").rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        index[".".join(parts)] = path
    return index


def _resolve(name: str, index: dict[str, Path]) -> str | None:
    """The longest prefix of the dotted ``name`` that is a kornia module other than the root package."""
    parts = name.split(".")
    for end in range(len(parts), 1, -1):
        candidate = ".".join(parts[:end])
        if candidate in index:
            return candidate
    return None


def _attribute_chain(node: ast.Attribute) -> list[str] | None:
    names: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    names.append(current.id)
    return names[::-1]


def imported_modules(module: str, path: Path, index: dict[str, Path]) -> set[str]:
    """The kornia modules ``module`` imports anywhere in its body, function-local imports included."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    root_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "kornia" or alias.name.startswith("kornia."):
                    if alias.asname is None or alias.name == "kornia":
                        root_aliases.add(alias.asname or "kornia")
                    if (target := _resolve(alias.name, index)) is not None:
                        found.add(target)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join(anchor + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base != "kornia" and not base.startswith("kornia."):
                continue
            for alias in node.names:
                # ``from kornia.x import y``: the submodule y when there is one, else the module that defines y
                full = f"{base}.{alias.name}"
                target = full if full in index else _resolve(base, index)
                if target is not None:
                    found.add(target)
    # ``kornia.x.y.f(...)`` after a bare ``import kornia``: the deepest module of the chain, once per chain
    inner = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in inner:
            chain = _attribute_chain(node)
            if chain and chain[0] in root_aliases and (target := _resolve(".".join(["kornia", *chain[1:]]), index)):
                found.add(target)
    found.discard(module)
    return found


def importers(index: dict[str, Path]) -> dict[str, set[str]]:
    """Reverse import graph: module -> the modules that import it."""
    reverse: dict[str, set[str]] = defaultdict(set)
    for module, path in index.items():
        if module == "kornia":  # the root package only re-exports; nothing resolves to it
            continue
        for target in imported_modules(module, path, index):
            reverse[target].add(module)
    return reverse


def _package(module: str) -> str:
    return module.split(".")[1]


@functools.cache
def _graph(root: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    index = module_index(root)
    return {path.relative_to(root).as_posix(): module for module, path in index.items()}, importers(index)


def select(paths: Iterable[str], root: Path = REPO, *, promise: str) -> str:
    """Return the ``--<promise>-conformance`` value for a set of changed paths, ``""`` when nothing is selected."""
    changed: set[str] = set()
    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        manifest_of = [other for other in TEST_FILES if path.startswith(MANIFEST_PREFIX.format(other))]
        if manifest_of:
            if manifest_of == [promise]:
                return "all"
            continue
        if (
            path.startswith(ALL_PREFIXES)
            or path in ALL_FILES
            or path == TEST_FILES[promise]
            or path.startswith("pixi.")
        ):
            return "all"
        parts = path.split("/")
        if parts[0] != "kornia":
            continue
        if len(parts) == 2:  # kornia/__init__.py, kornia/constants.py, ...: imported by every package
            return "all"
        changed.add(path)
    if not changed:
        return ""
    by_path, reverse = _graph(root)
    index = {module: root / path for path, module in by_path.items()}
    affected: set[str] = set()
    for path in changed:
        if (module := by_path.get(path)) is not None:
            affected.add(module)
            continue
        # a deleted file, or data next to the code, stands for its package's ``__init__``; a deleted module also for
        # the nearest module that survives it, where the imports of its remaining importers now resolve
        affected.add(f"kornia.{path.split('/')[1]}")
        if path.endswith(".py") and (survivor := _resolve(path.removesuffix(".py").replace("/", "."), index)):
            affected.add(survivor)
    pending = list(affected)
    while pending:
        for importer in reverse.get(pending.pop(), set()) - affected:
            affected.add(importer)
            pending.append(importer)
    packages = {_package(module) for module in affected}
    return ",".join(p for p in STABLE_CORE if p in packages)


def main(argv: list[str] | None = None) -> int:
    """Print the selection of a promise for the changed paths read from stdin."""
    parser = argparse.ArgumentParser(description="Print the conformance package selection for changed paths on stdin.")
    parser.add_argument("promise", choices=sorted(TEST_FILES), help="the conformance test the selection is for")
    args = parser.parse_args(argv)
    print(select(sys.stdin.read().split(), promise=args.promise))
    return 0


if __name__ == "__main__":
    sys.exit(main())

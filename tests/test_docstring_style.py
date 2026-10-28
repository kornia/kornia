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

"""Converted kornia modules keep the docstring style the rollout set out to give them.

``testing/docstring_rollout.py`` lists the modules that are converted. For those, this file enforces what a reader,
an IDE or a coding agent relies on and what a refactor silently breaks:

* every public function and class has a ``See Also:`` section whose entries name a sibling and say how it differs;
* every ``:func:``, ``:class:`` and ``:meth:`` role resolves to a name that Sphinx registers from ``docs/source``,
  or to a torch, Python or NumPy object in the cached intersphinx inventories, so it renders as a link and not as
  plain text (a short import alias is importable but is not a target); See Also entries name kornia objects only;
* the source writes a tensor shape as a literal such as ``(B, 1, H, W)``, not as ``:math:``;
* the module docstring of a namespace names every export of its ``__all__`` in a role or a double-backtick literal.

The test ids are qualified names, so a failure names the object to fix. Run with ``-s`` to see the progress table.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

import kornia

from testing.docstring_rollout import (
    CATALOG_NAMESPACES,
    DOCSTRING_MODULES,
    EXCLUDED_PACKAGES,
    IN_SCOPE_PACKAGES,
    SEE_ALSO_EXCEPTIONS,
    SHAPE_CHECK_MODULES,
)
from testing.docstring_style import (
    catalog_missing,
    catalog_required,
    catalog_unimportable,
    covered,
    docstring_roles,
    intersphinx_names,
    is_documented,
    math_shape_roles,
    modules_declaring_all,
    own_docstring,
    public_objects,
    reference_resolves,
    resolve_reference,
    see_also_entries,
    sphinx_registry,
)

KORNIA_ROOT = Path(kornia.__file__).resolve().parent
DOCS = KORNIA_ROOT.parent / "docs" / "source"
REGISTRY = sphinx_registry(DOCS)
INTERSPHINX = intersphinx_names(DOCS)

CONVERTED = sorted(
    ((qualified, obj) for obj, qualified in public_objects().values() if covered(obj.__module__, DOCSTRING_MODULES)),
    key=lambda pair: pair[0],
)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(KORNIA_ROOT.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


SHAPE_CHECK_FILES = sorted(
    path
    for path in KORNIA_ROOT.rglob("*.py")
    if SHAPE_CHECK_MODULES is None or covered(_module_name(path), SHAPE_CHECK_MODULES)
)


def _suggest(target: str) -> str:
    """Name the one registered full name that shares the last component of ``target``, if exactly one does."""
    last = target.removesuffix("()").rpartition(".")[2]
    matches = [name for name in REGISTRY if name.rpartition(".")[2] == last]
    return f" (registered as {matches[0]!r})" if len(matches) == 1 else ""


def test_rollout_selects_something() -> None:
    """The parametrized checks below are not vacuous: the rollout lists select objects and files."""
    assert CONVERTED, f"DOCSTRING_MODULES {DOCSTRING_MODULES} select no public object"
    assert SHAPE_CHECK_FILES, f"SHAPE_CHECK_MODULES {SHAPE_CHECK_MODULES} select no source file"
    assert CATALOG_NAMESPACES, "CATALOG_NAMESPACES is empty, so no module catalog is checked"
    assert REGISTRY, f"sphinx_registry({DOCS}) registered nothing"
    assert INTERSPHINX, f"intersphinx_names({DOCS}) found no inventory names"
    for prefix in DOCSTRING_MODULES:
        assert any(covered(obj.__module__, [prefix]) for _, obj in CONVERTED), (
            f"DOCSTRING_MODULES entry {prefix!r} selects no public object with a docstring"
        )
    for prefix in SHAPE_CHECK_MODULES or ():
        assert any(covered(_module_name(path), [prefix]) for path in SHAPE_CHECK_FILES), (
            f"SHAPE_CHECK_MODULES entry {prefix!r} selects no source file"
        )


def test_every_package_is_assigned() -> None:
    present = {path.name for path in KORNIA_ROOT.iterdir() if path.is_dir() and (path / "__init__.py").exists()}
    present |= {path.stem for path in KORNIA_ROOT.glob("*.py") if not path.name.startswith("_")}
    assigned = set(IN_SCOPE_PACKAGES) | set(EXCLUDED_PACKAGES)
    assert present == assigned, (
        f"add {sorted(present - assigned)} to IN_SCOPE_PACKAGES or EXCLUDED_PACKAGES in testing/docstring_rollout.py; "
        f"remove {sorted(assigned - present)}, which no longer exist"
    )


def test_rollout_lists_are_real() -> None:
    lists = {
        "DOCSTRING_MODULES": DOCSTRING_MODULES,
        "CATALOG_NAMESPACES": CATALOG_NAMESPACES,
        "SHAPE_CHECK_MODULES": SHAPE_CHECK_MODULES or (),
    }
    for list_name, entries in lists.items():
        for entry in entries:
            parts = entry.split(".")
            package = parts[1] if parts[0] == "kornia" and len(parts) > 1 else None
            assert package in IN_SCOPE_PACKAGES, (
                f"{list_name} entry {entry!r} is not under kornia.<pkg> for an in-scope package"
            )
            importlib.import_module(entry)
    for entry in CATALOG_NAMESPACES:
        assert hasattr(importlib.import_module(entry), "__all__"), f"CATALOG_NAMESPACES entry {entry!r} has no __all__"
    converted = {qualified for qualified, _ in CONVERTED}
    for key, reason in SEE_ALSO_EXCEPTIONS.items():
        assert key in converted, f"SEE_ALSO_EXCEPTIONS key {key!r} is not a converted object"
        assert reason.strip(), f"SEE_ALSO_EXCEPTIONS entry {key!r} needs a reason"


@pytest.mark.parametrize(("qualified", "obj"), CONVERTED, ids=[qualified for qualified, _ in CONVERTED])
def test_converted_object_has_see_also(qualified: str, obj: object) -> None:
    if qualified in SEE_ALSO_EXCEPTIONS:
        pytest.skip(f"excepted: {SEE_ALSO_EXCEPTIONS[qualified]}")
    entries = see_also_entries(own_docstring(obj))
    assert entries, (
        f"{qualified} needs a 'See Also:' section with at least one bullet like "
        "'- :func:`name`: how it differs.' (or an entry in SEE_ALSO_EXCEPTIONS with a reason)"
    )
    problems = []
    for target, explanation in entries:
        if not explanation:
            problems.append(f"{target!r} has no explanation after the colon")
        if target.rpartition(".")[2].startswith("_"):
            problems.append(f"{target!r} is private")
        resolved = resolve_reference(obj, target, REGISTRY)
        if resolved is obj:
            problems.append(f"{target!r} links the object to itself")
        if resolved is None and target.removesuffix("()") in INTERSPHINX:
            problems.append(
                f"{target!r} is not a kornia object; See Also lists kornia objects, so mention torch and Python "
                "functions in the contract paragraph instead"
            )
    assert not problems, f"{qualified} See Also: " + "; ".join(problems)


@pytest.mark.parametrize(("qualified", "obj"), CONVERTED, ids=[qualified for qualified, _ in CONVERTED])
def test_converted_object_references_resolve(qualified: str, obj: object) -> None:
    unresolved = [
        f"{target!r}{_suggest(target)}"
        for _, target in docstring_roles(own_docstring(obj))
        if not reference_resolves(obj, target, REGISTRY, INTERSPHINX)
    ]
    hint = "write the full name the docs register"
    if not is_documented(obj, REGISTRY):
        hint = f"{qualified} is not documented on any docs/source page, so short names have no scope; {hint}"
    assert not unresolved, f"{qualified} has references Sphinx cannot link ({hint}): " + ", ".join(unresolved)


@pytest.mark.parametrize(
    "path", SHAPE_CHECK_FILES, ids=[str(path.relative_to(KORNIA_ROOT.parent)) for path in SHAPE_CHECK_FILES]
)
def test_no_math_shape_roles(path: Path) -> None:
    found = math_shape_roles(path.read_text(encoding="utf-8"))
    assert found == [], (
        f"{path.relative_to(KORNIA_ROOT.parent)} writes shapes as :math: {found}; use literals like ``(B, 1, H, W)``"
    )


@pytest.mark.parametrize("namespace", CATALOG_NAMESPACES)
def test_catalog_names_every_export(namespace: str) -> None:
    module = importlib.import_module(namespace)
    doc = inspect.getdoc(module) or ""
    missing = catalog_missing(doc, sorted(module.__all__))
    assert missing == [], (
        f"the module docstring of {namespace} does not mention {missing}; list every export as a role such as "
        ":func:`name` or a literal such as ``name``"
    )
    broken = catalog_unimportable(module, doc)
    assert not broken, f"the module docstring of {namespace} has references that do not import: {broken}"


def test_rollout_progress() -> None:
    """Print, per in-scope package, the converted objects and catalogued namespaces (read it with ``-s``)."""
    objects = [obj for obj, _ in public_objects().values()]
    namespaces = modules_declaring_all()
    print("\npackage        converted objects   catalogued namespaces")
    for package in IN_SCOPE_PACKAGES:
        prefix = f"kornia.{package}"
        in_package = [obj for obj in objects if covered(obj.__module__, [prefix])]
        done = [obj for obj in in_package if covered(obj.__module__, DOCSTRING_MODULES)]
        required = [ns for ns in namespaces if covered(ns, [prefix]) and catalog_required(ns, CATALOG_NAMESPACES)]
        catalogued = [ns for ns in required if ns in CATALOG_NAMESPACES]
        print(f"{package:<14} {len(done):>4} / {len(in_package):<12} {len(catalogued):>4} / {len(required)}")

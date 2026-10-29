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
an IDE or a coding agent relies on and what a refactor silently breaks; the shape rule applies everywhere:

* every public function and class has a docstring, so none escapes the checks below;
* every public function and class has a ``See Also:`` section whose entries name a sibling and say how it differs,
  unless its title line is a status title, ``Alias of :func:`target`.`` or ``Deprecated: use :func:`target`.``,
  which points to another public kornia object and needs none (``Deprecated: use`` only on an object that
  ``kornia.core._compat.deprecated`` wraps);
* every Python-domain object role (``:func:``, ``:class:``, ``:meth:``, ``:attr:``, ``:data:``, ``:const:``,
  ``:exc:``, ``:obj:``) resolves to a name that Sphinx registers from ``docs/source``, or to a torch, Python or NumPy
  object of a type that role links in the cached intersphinx inventories, so it renders as a link and not as plain
  text (a short import alias is importable but is not a target); See Also entries name kornia objects only;
* every ``kornia/**/*.py`` file and every page Sphinx builds from ``docs/source`` writes a tensor shape as a literal
  such as ``(B, 1, H, W)``, not as ``:math:``, except for the values listed in ``MATH_NON_SHAPES``;
* the module docstring of a namespace names every export of its ``__all__`` in a role or a double-backtick literal.

The test ids are qualified names, or repository paths for the shape check, so a failure names what to fix. Run with
``-s`` to see the progress table.
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
    MATH_NON_SHAPES,
    SEE_ALSO_EXCEPTIONS,
)
from testing.docstring_style import (
    catalog_missing,
    catalog_required,
    catalog_unimportable,
    covered,
    docstring_roles,
    intersphinx_names,
    is_deprecated_wrapper,
    is_documented,
    is_private_target,
    math_shape_roles,
    modules_declaring_all,
    own_docstring,
    public_objects,
    reference_resolves,
    resolve_reference,
    rst_pages,
    see_also_entries,
    sphinx_registry,
    status_target,
    undocumented_public_objects,
)

KORNIA_ROOT = Path(kornia.__file__).resolve().parent
REPO_ROOT = KORNIA_ROOT.parent
DOCS = REPO_ROOT / "docs" / "source"
REGISTRY = sphinx_registry(DOCS)
INTERSPHINX = intersphinx_names(DOCS)

CONVERTED = sorted(
    ((qualified, obj) for obj, qualified in public_objects().values() if covered(obj.__module__, DOCSTRING_MODULES)),
    key=lambda pair: pair[0],
)
UNDOCUMENTED = sorted(
    (
        qualified
        for obj, qualified in undocumented_public_objects().values()
        if covered(obj.__module__, DOCSTRING_MODULES)
    ),
)


SHAPE_CHECK_FILES = sorted([*KORNIA_ROOT.rglob("*.py"), *rst_pages(DOCS)])


def _suggest(target: str) -> str:
    """Name the one registered full name that shares the last component of ``target``, if exactly one does."""
    last = target.removesuffix("()").rpartition(".")[2]
    matches = [name for name in REGISTRY if name.rpartition(".")[2] == last]
    return f" (registered as {matches[0]!r})" if len(matches) == 1 else ""


def _wrong_role(role: str, target: str) -> str:
    """Name the intersphinx types of ``target`` when it is there under a type that ``role`` does not link."""
    types = INTERSPHINX.get(target.removesuffix("()") if role in ("func", "meth") else target)
    return f" (an intersphinx {', '.join(sorted(types))}, which :{role}: does not link)" if types else ""


def test_rollout_selects_something() -> None:
    """The parametrized checks below are not vacuous: the rollout lists select objects and files."""
    assert CONVERTED, f"DOCSTRING_MODULES {DOCSTRING_MODULES} select no public object"
    assert SHAPE_CHECK_FILES, f"no source file or docs page found under {KORNIA_ROOT} and {DOCS}"
    assert CATALOG_NAMESPACES, "CATALOG_NAMESPACES is empty, so no module catalog is checked"
    assert REGISTRY, f"sphinx_registry({DOCS}) registered nothing"
    assert INTERSPHINX, f"intersphinx_names({DOCS}) found no inventory names"
    for prefix in DOCSTRING_MODULES:
        assert any(covered(obj.__module__, [prefix]) for _, obj in CONVERTED), (
            f"DOCSTRING_MODULES entry {prefix!r} selects no public object with a docstring"
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


@pytest.mark.parametrize("qualified", UNDOCUMENTED, ids=UNDOCUMENTED)
def test_converted_object_has_docstring(qualified: str) -> None:
    pytest.fail(
        f"{qualified} is exported without a docstring, so the checks cannot see it; write one. A thin wrapper "
        "that only forwards to another object starts with 'Alias of :func:`target`.', a deprecated one with "
        "'Deprecated: use :func:`target`.'"
    )


@pytest.mark.parametrize(("qualified", "obj"), CONVERTED, ids=[qualified for qualified, _ in CONVERTED])
def test_converted_object_has_see_also(qualified: str, obj: object) -> None:
    if qualified in SEE_ALSO_EXCEPTIONS:
        pytest.skip(f"excepted: {SEE_ALSO_EXCEPTIONS[qualified]}")
    doc = own_docstring(obj)
    entries = see_also_entries(doc)
    target = status_target(doc)
    if entries is None and target:
        # an "Alias of" or "Deprecated: use" title points to the target, so a See Also would only repeat it; the
        # target is another public kornia object, and only a deprecated object says "Deprecated: use"
        resolved = resolve_reference(obj, target, REGISTRY)
        message = f"{qualified}: a status title points to another public kornia object, not to {target!r}"
        assert resolved is not None, message
        assert resolved is not obj, message
        assert not is_private_target(target), message
        assert not doc.lstrip().startswith("Deprecated: use") or is_deprecated_wrapper(obj), (
            f"{qualified} has a 'Deprecated: use' title but is not wrapped by kornia.core._compat.deprecated"
        )
        return
    assert entries, (
        f"{qualified} needs a 'See Also:' section with at least one bullet like "
        "'- :func:`name`: how it differs.' (or an entry in SEE_ALSO_EXCEPTIONS with a reason)"
    )
    problems = []
    for target, explanation in entries:
        if not target:
            problems.append(f"{explanation!r} does not start with a role such as :func:`name`, so it is no link")
            continue
        if not explanation:
            problems.append(f"{target!r} has no explanation after the colon")
        if is_private_target(target):
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
        f":{role}:`{target}`{_suggest(target)}{_wrong_role(role, target)}"
        for role, target in docstring_roles(own_docstring(obj))
        if not reference_resolves(obj, target, REGISTRY, INTERSPHINX, role)
    ]
    hint = "write the full name the docs register"
    if not is_documented(obj, REGISTRY):
        hint = f"{qualified} is not documented on any docs/source page, so short names have no scope; {hint}"
    assert not unresolved, f"{qualified} has references Sphinx cannot link ({hint}): " + ", ".join(unresolved)


@pytest.mark.parametrize(
    "path", SHAPE_CHECK_FILES, ids=[path.relative_to(REPO_ROOT).as_posix() for path in SHAPE_CHECK_FILES]
)
def test_no_math_shape_roles(path: Path) -> None:
    relative = path.relative_to(REPO_ROOT).as_posix()
    values = MATH_NON_SHAPES.get(relative, {})
    found = [content for content in math_shape_roles(path.read_text(encoding="utf-8")) if content not in values]
    assert found == [], (
        f"{relative} writes shapes as :math: {found}; use literals like ``(B, 1, H, W)``, or list a tuple that is "
        "a value, not a shape, in MATH_NON_SHAPES with what it is"
    )


def test_math_non_shapes_are_current() -> None:
    """Every ``MATH_NON_SHAPES`` entry names a checked file that still writes that tuple as ``:math:``."""
    checked = {path.relative_to(REPO_ROOT).as_posix(): path for path in SHAPE_CHECK_FILES}
    for relative, values in MATH_NON_SHAPES.items():
        assert relative in checked, f"MATH_NON_SHAPES key {relative!r} is not a checked source file or docs page"
        found = set(math_shape_roles(checked[relative].read_text(encoding="utf-8")))
        for content, reason in values.items():
            assert content in found, f"MATH_NON_SHAPES entry {content!r} no longer occurs in {relative}; remove it"
            assert reason.strip(), f"MATH_NON_SHAPES entry {content!r} in {relative} needs a reason"


@pytest.mark.parametrize("namespace", CATALOG_NAMESPACES)
def test_catalog_names_every_export(namespace: str) -> None:
    module = importlib.import_module(namespace)
    doc = inspect.getdoc(module) or ""
    missing = catalog_missing(doc, sorted(module.__all__), module)
    assert missing == [], (
        f"the module docstring of {namespace} does not mention {missing}; list every export as a role such as "
        ":func:`name` or a literal such as ``name``, and spell a qualified name as a path to that export"
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

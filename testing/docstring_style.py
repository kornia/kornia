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

"""Helpers shared by the tests that check kornia's docstring style.

They find the public API, read the structure of a docstring (``See Also`` entries, cross-reference roles, shape
tuples) and resolve a cross-reference the way Sphinx does, through the names that the ``.rst`` pages under
``docs/source`` register and, for torch, Python and NumPy objects, the cached intersphinx inventories.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import warnings
import zlib
from functools import cache
from pathlib import Path
from typing import Iterable

import kornia

_KORNIA_ROOT = Path(kornia.__file__).resolve().parent


def modules_declaring_all() -> list[str]:
    """Name every non-private kornia module that assigns ``__all__``, found from source without importing it.

    The same parse as ``_declared_inventory_modules`` in ``tests/test_api_surface.py``, over the whole package
    rather than the inventory roots: the inventory leaves out ``kornia.feature``, whose matchers and LAF estimators
    are among the symbols the docstring tests exist for.
    """
    found = []
    for path in sorted(_KORNIA_ROOT.rglob("*.py")):
        parts = list(path.relative_to(_KORNIA_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if any(part.startswith("_") for part in parts):
            continue
        tree = ast.parse(path.read_bytes(), filename=str(path))
        if any(
            (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
            )
            or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__")
            for node in tree.body
        ):
            found.append(".".join(["kornia", *parts]))
    return found


def own_docstring(obj: object) -> str | None:
    """Return the cleaned docstring ``obj`` defines itself, or ``None``.

    A class does not inherit one from its base here, unlike with ``inspect.getdoc``.
    """
    doc = obj.__dict__.get("__doc__") if inspect.isclass(obj) else getattr(obj, "__doc__", None)
    cleaned = inspect.cleandoc(doc) if doc else ""
    return cleaned or None


def _import_exported(module_name: str) -> object | None:
    """Import ``module_name`` quietly; ``None`` when it needs an optional package that is not installed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if (error.name or "").partition(".")[0] == "kornia":
                raise
            return None


def _is_kornia_function_or_class(obj: object) -> bool:
    if not (inspect.isclass(obj) or inspect.isfunction(obj)):
        return False
    module_of_obj = obj.__module__ or ""
    return module_of_obj == "kornia" or module_of_obj.startswith("kornia.")


@cache
def public_objects() -> dict[int, tuple[object, str]]:
    """Map ``id(obj)`` to ``(obj, "module.qualname")`` for every exported kornia function and class with a docstring.

    Covers every name in the ``__all__`` of a non-private kornia module that is a function or class defined in
    kornia and has a docstring of its own, deduplicated by object identity. Names starting with an underscore are
    out of scope. The dict is cached and shared between callers, who must not mutate it.
    """
    collected: dict[int, tuple[object, str]] = {}
    for module_name in modules_declaring_all():
        module = _import_exported(module_name)
        if module is None:
            continue
        with warnings.catch_warnings():
            # ``getattr`` runs module-level ``__getattr__`` shims, which warn about deprecated names.
            warnings.simplefilter("ignore")
            exported = [getattr(module, name, None) for name in module.__all__ if not name.startswith("_")]
        for obj in exported:
            if _is_kornia_function_or_class(obj) and own_docstring(obj):
                collected[id(obj)] = (obj, f"{obj.__module__}.{obj.__qualname__}")
    return collected


def covered(module_name: str, prefixes: Iterable[str]) -> bool:
    """Whether ``module_name`` equals one of ``prefixes`` or lies below it (``prefix + "."``)."""
    return any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in prefixes)


_SHAPE_OPERAND = r"(?:[A-Z]\w*|\d+)"
_SHAPE_ELEMENT = re.compile(rf"[A-Z]\w*|\d+|\*|\.\.\.|{_SHAPE_OPERAND}\s*(?://|[/*+-])\s*{_SHAPE_OPERAND}")


def is_shape_tuple(content: str) -> bool:
    """Whether ``content`` is a tensor shape such as ``(B, 3, H, W)``, ``(*, H, W)`` or ``(B, H/2, W/2)``.

    The elements must be uppercase identifiers, non-negative integers, ``*``, ``...`` or one binary
    ``//``, ``/``, ``*``, ``+`` or ``-`` between two of the first two. Coordinate pairs such as ``(x, y)``, LaTeX
    tuples, ranges with signs or decimals and the empty tuple are not shapes.
    """
    text = content.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return False
    elements = [element.strip() for element in text[1:-1].split(",")]
    if len(elements) > 1 and elements[-1] == "":
        elements.pop()  # a trailing comma, as in ``(B,)``
    return all(_SHAPE_ELEMENT.fullmatch(element) for element in elements)


_MATH_ROLE = re.compile(r":math:`([^`]*)`")


def math_shape_roles(text: str) -> list[str]:
    """Return the contents of every ``:math:`` role in ``text`` that is a shape tuple, in order."""
    return [content for content in _MATH_ROLE.findall(text) if is_shape_tuple(content)]


_XREF_ROLE = re.compile(r":(?:py:)?(func|class|meth):`~?([^`]+)`")


def docstring_roles(doc: str) -> list[tuple[str, str]]:
    """Return ``(role, target)`` for every ``func``, ``class`` and ``meth`` role in ``doc``, without a leading ``~``."""
    return [(match.group(1), match.group(2)) for match in _XREF_ROLE.finditer(doc)]


_SEE_ALSO = re.compile(r"^(\s*)See Also:\s*$")


_LEADING_ROLES = re.compile(r"(?::(?:py:)?(?:func|class|meth):`~?[^`]+`[\s,]*)+")


def see_also_entries(doc: str) -> list[tuple[str, str]] | None:
    """Parse the ``See Also:`` section of ``doc`` into ``(target, explanation)`` pairs.

    Returns ``None`` when there is no such section, and ``[]`` when it has no bullets. Each ``- `` bullet gives one
    pair per role in its leading role list (one or more roles separated by commas or whitespace); they share the
    explanation, which is the text after the colon that follows that list, joined across continuation lines
    (``""`` when missing). Roles inside the explanation are not targets. The section ends at the first non-blank
    line indented at or below the ``See Also:`` header, or at the end of the docstring; text before the first
    bullet is ignored.
    """
    lines = doc.splitlines()
    start = next((i for i, line in enumerate(lines) if _SEE_ALSO.match(line)), None)
    if start is None:
        return None
    indent = len(lines[start]) - len(lines[start].lstrip())
    bullets: list[list[str]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        if stripped.startswith("- "):
            bullets.append([stripped[2:].strip()])
        elif bullets:
            bullets[-1].append(stripped)
    entries: list[tuple[str, str]] = []
    for parts in bullets:
        text = " ".join(parts)
        leading = _LEADING_ROLES.match(text)
        if leading is None:
            continue
        tail = text[leading.end() :].strip()
        explanation = tail[1:].strip() if tail.startswith(":") else ""
        entries.extend((match.group(2), explanation) for match in _XREF_ROLE.finditer(leading.group(0)))
    return entries


_DIRECTIVE_KINDS = "autofunction|autoclass|automethod|autoexception|automodule|autosummary|currentmodule|module"
_DIRECTIVE = re.compile(rf"^(\s*)\.\. ({_DIRECTIVE_KINDS})::\s*(.*?)\s*$")


def _import_dotted(name: str, as_module: bool = False) -> object | None:
    """Resolve ``name`` the way autodoc does; ``None`` on any failure.

    Imports the longest importable parent module and walks the rest with ``getattr``, so a function that shares its
    name with a submodule (``kornia.filters.sobel``) is the function. The whole name is tried as a module first only
    for ``as_module`` (``automodule``) and as a last fallback otherwise.
    """
    parts = name.split(".")
    ends = list(range(len(parts) - 1, 0, -1))
    ends = [len(parts), *ends] if as_module else [*ends, len(parts)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for end in ends:
            try:
                obj: object = importlib.import_module(".".join(parts[:end]))
                for attr in parts[end:]:
                    obj = getattr(obj, attr)
            except Exception:  # noqa: BLE001, S112 - an optional dependency may fail in any way
                continue
            return obj
    return None


def _autodoc_documents(owner: object, name: str, member: object, undoc_members: bool) -> bool:
    """Whether autodoc's ``:members:`` documents ``member``, found as ``name`` on the class or module ``owner``.

    Mirrors ``Documenter.filter_members`` with the default ``autodoc_inherit_docstrings = True``: a member is kept
    with ``:undoc-members:`` or when it has a docstring, its own or, for a class member, the first one found under
    the same name up the MRO, or else what ``inspect.getdoc`` finds (a class inherits its base's docstring there).
    """
    if undoc_members:
        return True
    doc = getattr(member, "__doc__", None)
    if doc is None and inspect.isclass(owner):
        for base in owner.__mro__:
            inherited = getattr(base, name, None)
            # ``None.__doc__`` is the docstring of ``NoneType`` from Python 3.13 on, so skip a missing name.
            doc = None if inherited is None else getattr(inherited, "__doc__", None)
            if doc is not None:
                break
    if doc is None:
        doc = inspect.getdoc(member)
    # autodoc drops a docstring that is only the one of the member's type, such as ``property.__doc__``.
    return isinstance(doc, str) and bool(doc) and doc != getattr(type(member), "__doc__", None)


def _public_methods(cls: type, undoc_members: bool, special: list[str] | None = None) -> dict[str, object]:
    """Public methods and properties defined on ``cls`` itself that ``autoclass :members:`` documents.

    Static and class methods are unwrapped. Inherited members are left out, as autodoc does without
    ``:inherited-members:``; an inherited docstring still documents a method that ``cls`` overrides. ``special`` is
    the ``:special-members:`` option: ``None`` keeps out every ``__name__``, ``[]`` admits all, a list the listed.
    """
    methods: dict[str, object] = {}
    for name, value in vars(cls).items():
        is_special = name.startswith("__") and name.endswith("__") and name != "__doc__"
        if name.startswith("_") and not (is_special and special is not None and (not special or name in special)):
            continue
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if (inspect.isfunction(value) or isinstance(value, property)) and _autodoc_documents(
            cls, name, value, undoc_members
        ):
            methods[name] = value
    return methods


def _module_members(module: object, undoc_members: bool) -> dict[str, object]:
    """What ``automodule :members:`` documents: ``__all__`` when declared, else the public names defined in it."""
    declared = getattr(module, "__all__", None)
    if declared is not None:
        names = [name for name in declared if not name.startswith("_")]
    else:
        names = [name for name, value in vars(module).items() if not name.startswith("_")]
    members = {}
    for name in names:
        value = getattr(module, name, None)
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            continue
        if declared is None and value.__module__ != module.__name__:
            continue
        if _autodoc_documents(module, name, value, undoc_members):
            members[name] = value
    return members


def _directive_options(lines: list[str], index: int, indent: int) -> dict[str, str]:
    """Read the option block of the directive at ``index``: the ``:name: value`` lines right below it."""
    options: dict[str, str] = {}
    for line in lines[index + 1 :]:
        stripped = line.strip()
        if not stripped or len(line) - len(line.lstrip()) <= indent or not stripped.startswith(":"):
            break
        name, _, value = stripped[1:].partition(":")
        options[name] = value.strip()
    return options


def _members_option(options: dict[str, str], option: str = "members") -> list[str] | None:
    """Read ``:members:`` or another member option: ``None`` when absent, ``[]`` for all, else the listed names."""
    if option not in options:
        return None
    return [name.strip() for name in options[option].split(",") if name.strip()]


def _autosummary_names(lines: list[str], index: int, indent: int) -> list[str]:
    names = []
    for line in lines[index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        if stripped.startswith((":", "..")):
            continue
        names.append(stripped.removeprefix("~"))
    return names


def _add(registry: dict[str, object], full_name: str, as_module: bool = False) -> object | None:
    obj = _import_dotted(full_name, as_module)
    if obj is not None:
        registry[full_name] = obj
    return obj


def _add_methods(
    registry: dict[str, object],
    cls_name: str,
    cls: object,
    listed: list[str],
    undoc_members: bool,
    special: list[str] | None = None,
) -> None:
    """Register the methods ``:members:`` documents: the listed names, or the public ones defined on ``cls``.

    Either way autodoc keeps only members with a docstring, unless the directive has ``:undoc-members:``. With all
    members, ``special`` (``:special-members:``) selects the ``__name__`` methods as well.
    """
    if not inspect.isclass(cls):
        return
    if listed:
        for name in listed:
            member = _import_dotted(f"{cls_name}.{name}")
            if member is not None and _autodoc_documents(cls, name, member, undoc_members):
                registry[f"{cls_name}.{name}"] = member
    else:
        for name, method in _public_methods(cls, undoc_members, special).items():
            registry[f"{cls_name}.{name}"] = method


def _add_module_members(
    registry: dict[str, object],
    module_name: str,
    module: object,
    listed: list[str],
    undoc_members: bool,
    special: list[str] | None = None,
) -> None:
    if listed:
        members = {name: getattr(module, name, None) for name in listed}
        members = {
            name: member
            for name, member in members.items()
            if member is not None and _autodoc_documents(module, name, member, undoc_members)
        }
    else:
        members = _module_members(module, undoc_members)
    for name, member in members.items():
        registry[f"{module_name}.{name}"] = member
        _add_methods(registry, f"{module_name}.{name}", member, [], undoc_members, special)


# Directories of ``docs/source`` that Sphinx does not read pages from: ``conf.py``'s ``exclude_patterns``,
# ``templates_path``, ``html_static_path`` and ``html_extra_path``.
_NOT_PAGES = frozenset({"_build", ".ipynb_checkpoints", "_templates", "_static", "_extra"})


def _rst_pages(docs_source: Path) -> list[Path]:
    """Every ``.rst`` page under ``docs_source``, subdirectories included, except those Sphinx does not build."""
    root = Path(docs_source)
    return sorted(page for page in root.rglob("*.rst") if _NOT_PAGES.isdisjoint(page.relative_to(root).parts[:-1]))


def sphinx_registry(docs_source: Path) -> dict[str, object]:
    """Map every full name that the ``.rst`` pages under ``docs_source`` register to the imported object.

    Reads ``autofunction``, ``autoclass``, ``automethod``, ``autoexception``, ``automodule`` and ``autosummary``
    entries, resolving relative names against the latest ``currentmodule`` or ``module`` and a nested ``automethod``
    against the enclosing class. ``autoclass`` and ``automodule`` with ``:members:`` register the methods and
    properties defined on the class and the functions and classes of the module that autodoc documents (public, with
    a docstring or under ``:undoc-members:``), and the ``__name__`` methods that ``:special-members:`` selects. An
    ``autosummary`` registers its entries only with ``:toctree:``, which generates their pages; without it, it only
    links to them. Names that do not import are left out.
    """
    registry: dict[str, object] = {}

    for page in _rst_pages(docs_source):
        lines = page.read_text(encoding="utf-8").splitlines()
        current = ""
        enclosing_class = ""
        for index, line in enumerate(lines):
            match = _DIRECTIVE.match(line)
            if match is None:
                continue
            indent, kind, argument = len(match.group(1)), match.group(2), match.group(3)
            if kind in ("currentmodule", "module"):
                current = argument
                continue
            qualified = argument if argument.startswith("kornia") or not current else f"{current}.{argument}"
            if kind == "automethod" and indent > 0 and enclosing_class:
                qualified = f"{enclosing_class}.{argument}"
            options = _directive_options(lines, index, indent)
            members = _members_option(options)
            special = _members_option(options, "special-members")
            # Like autodoc's ``merge_members_option``, a ``:special-members:`` list joins a ``:members:`` that does
            # not already document all members, and stands in for an absent one.
            if special and members != []:
                members = [*(members or []), *(name for name in special if name not in (members or []))]
            undoc_members = "undoc-members" in options
            if kind == "autosummary":
                if "toctree" in options:
                    for name in _autosummary_names(lines, index, indent):
                        _add(registry, name if name.startswith("kornia") or not current else f"{current}.{name}")
            elif kind == "automodule":
                module = _add(registry, qualified, as_module=True)
                if module is not None and members is not None:
                    _add_module_members(registry, qualified, module, members, undoc_members, special)
            else:
                obj = _add(registry, qualified)
                if kind == "autoclass":
                    enclosing_class = qualified
                    if members is not None:
                        _add_methods(registry, qualified, obj, members, undoc_members, special)
    return registry


_INVENTORY_LINE = re.compile(r"(.+?)\s+(\S+)\s+(-?\d+)\s+?(\S*)\s+(.*)")
# The Python object types a ``:func:``, ``:class:`` or ``:meth:`` role can link through intersphinx.
_LINKABLE_INVENTORY_TYPES = frozenset(
    f"py:{kind}" for kind in ("function", "class", "exception", "method", "classmethod", "staticmethod")
)


def intersphinx_names(docs_source: Path) -> frozenset[str]:
    """Return the Python object names of the cached intersphinx inventories in ``docs_source/_intersphinx``.

    ``conf.py`` falls back to these ``python``, ``numpy`` and ``torch`` inventories when it cannot fetch the live
    ones. Each file is a Sphinx inventory version 2: four header lines, then zlib-compressed
    ``name domain:role priority uri dispname`` lines. Only functions, classes, exceptions and methods are kept, the
    object types that ``:func:``, ``:class:`` and ``:meth:`` can link; the set does not say which role fits which.
    """
    names: set[str] = set()
    for path in sorted((Path(docs_source) / "_intersphinx").glob("*.inv")):
        with path.open("rb") as stream:
            header = [stream.readline() for _ in range(4)]
            if not header[0].startswith(b"# Sphinx inventory version 2") or b"zlib" not in header[3]:
                raise ValueError(f"{path} is not a zlib-compressed Sphinx inventory version 2")
            body = zlib.decompress(stream.read()).decode("utf-8")
        for line in body.splitlines():
            match = _INVENTORY_LINE.match(line.rstrip())
            if match is not None and match.group(2) in _LINKABLE_INVENTORY_TYPES:
                names.add(match.group(1))
    return frozenset(names)


def _reference_scopes(obj: object, registry: dict[str, object]) -> list[str]:
    """The prefixes Sphinx tries a reference in the docstring of ``obj`` under: its module, then its class."""
    scopes: list[str] = []
    for name, registered in registry.items():
        if registered is not obj:
            continue
        parent = name.rpartition(".")[0]
        if inspect.isclass(obj):
            scopes += [parent, name]
        elif parent in registry and inspect.isclass(registry[parent]):
            scopes += [parent.rpartition(".")[0], parent]
        else:
            scopes.append(parent)
    return [scope for scope in scopes if scope]


def is_documented(obj: object, registry: dict[str, object]) -> bool:
    """Whether a docs page registers ``obj`` itself, so that short names in its docstring have a scope."""
    return any(registered is obj for registered in registry.values())


def resolve_reference(obj: object, target: str, registry: dict[str, object]) -> object | None:
    """Return the registered object that ``target``, written in the docstring of ``obj``, resolves to, or ``None``.

    As Sphinx does for a ``:func:``, ``:class:`` or ``:meth:`` role: a trailing ``()`` is dropped, then the target
    is tried as written and under each scope of ``obj``, its module (the page it is registered on) and, for a class
    or a method, the class, so ``to_tensor`` and ``Boxes.to_tensor`` both resolve inside the ``Boxes`` page. A short
    import alias is not registered, so it does not resolve; neither does a short name in the docstring of an object
    that no docs page registers. Only kornia objects resolve here; see ``intersphinx_names`` for the rest.
    """
    target = target.removesuffix("()")
    candidates = [target, *(f"{scope}.{target}" for scope in _reference_scopes(obj, registry))]
    for candidate in candidates:
        if candidate in registry:
            return registry[candidate]
    return None


def reference_resolves(
    obj: object, target: str, registry: dict[str, object], external: frozenset[str] = frozenset()
) -> bool:
    """Whether ``target`` in the docstring of ``obj`` renders as a link.

    True when it resolves to a registered kornia object (see ``resolve_reference``) or, without a trailing ``()``,
    is one of the ``external`` names, such as ``intersphinx_names(docs_source)``.
    """
    return resolve_reference(obj, target, registry) is not None or target.removesuffix("()") in external


def catalog_missing(doc: str, names: list[str]) -> list[str]:
    """Return the ``names`` that ``doc`` does not mention inside backticks.

    A name counts only as the whole target of a role, such as :func:`name` or :func:`~kornia.x.name`, or as a
    double-backtick literal, such as ``name`` or ``name()``. The same word in plain prose (``normalize``,
    ``threshold``) does not count, and ``depth_to_3d_v2`` does not count as mentioning ``depth_to_3d``.
    """
    return [name for name in names if not re.search(rf"`~?(?:[\w.]*\.)?{re.escape(name)}(?:\(\))?`", doc)]


def _import_role_target(module: object, target: str) -> object:
    """Import a role target: longest importable dotted prefix, then ``getattr``; a bare name is read off ``module``.

    A trailing ``()`` is dropped first, as Sphinx drops it from a role target such as ``name()``.
    """
    target = target.removesuffix("()")
    if "." not in target:
        return getattr(module, target)
    parts = target.split(".")
    for end in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:end]))
        except ImportError:
            continue
        for attr in parts[end:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(target)


def catalog_unimportable(module: object, doc: str) -> list[str]:
    """Return the role targets in the module catalog ``doc`` of ``module`` that do not import.

    Reads ``:func:``, ``:class:`` and ``:meth:`` roles. A bare target is looked up on ``module``, a dotted one from
    the longest importable module prefix.
    """
    broken = []
    for _, target in docstring_roles(doc):
        try:
            _import_role_target(module, target)
        except (ImportError, AttributeError):
            broken.append(target)
    return broken


def catalog_required(namespace: str, catalogued: Iterable[str]) -> bool:
    """Whether ``namespace`` needs a catalog of its own.

    False when every object in its ``__all__`` is also exported, as the same object, by an ancestor namespace listed
    in ``catalogued``; true otherwise.
    """
    catalogued = set(catalogued)
    module = importlib.import_module(namespace)
    parts = namespace.split(".")
    ancestors = [importlib.import_module(".".join(parts[:end])) for end in range(1, len(parts))]
    ancestors = [ancestor for ancestor in ancestors if ancestor.__name__ in catalogued]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name, None)
            if not any(
                name in getattr(ancestor, "__all__", []) and getattr(ancestor, name, None) is obj
                for ancestor in ancestors
            ):
                return True
    return False

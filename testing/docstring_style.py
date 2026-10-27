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
tuples) and resolve a cross-reference the way Sphinx does, through the names that ``docs/source/*.rst`` register.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import warnings
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
    if not doc:
        return None
    return inspect.cleandoc(doc)


def _import_exported(module_name: str) -> object | None:
    """Import ``module_name`` quietly; ``None`` when it needs an optional package that is not installed."""
    with warnings.catch_warnings():
        # ``getattr`` runs module-level ``__getattr__`` shims, which warn about deprecated names.
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
    out of scope.
    """
    collected: dict[int, tuple[object, str]] = {}
    for module_name in modules_declaring_all():
        module = _import_exported(module_name)
        if module is None:
            continue
        with warnings.catch_warnings():
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


def see_also_entries(doc: str) -> list[tuple[str, str]] | None:
    """Parse the ``See Also:`` section of ``doc`` into ``(target, explanation)`` pairs.

    Returns ``None`` when there is no such section. Each ``- `` bullet gives one pair per role it contains; they
    share the explanation, which is the text after the colon that follows the last role, joined across continuation
    lines (``""`` when missing). The section ends at the next line with the header's indentation that ends with
    ``:``, or at the end of the docstring.
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
        if len(line) - len(line.lstrip()) <= indent and stripped.endswith(":"):
            break
        if stripped.startswith("- "):
            bullets.append([stripped[2:].strip()])
        elif bullets:
            bullets[-1].append(stripped)
    entries: list[tuple[str, str]] = []
    for parts in bullets:
        text = " ".join(parts)
        matches = list(_XREF_ROLE.finditer(text))
        if not matches:
            continue
        tail = text[matches[-1].end() :].strip()
        explanation = tail[1:].strip() if tail.startswith(":") else ""
        entries.extend((match.group(2), explanation) for match in matches)
    return entries


_DIRECTIVE_KINDS = "autofunction|autoclass|automethod|autoexception|automodule|autosummary|currentmodule|module"
_DIRECTIVE = re.compile(rf"^(\s*)\.\. ({_DIRECTIVE_KINDS})::\s*(.*?)\s*$")


def _import_dotted(name: str) -> object | None:
    """Import the longest module prefix of ``name`` and walk the rest with ``getattr``; ``None`` on any failure."""
    parts = name.split(".")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for end in range(len(parts), 0, -1):
            try:
                obj: object = importlib.import_module(".".join(parts[:end]))
                for attr in parts[end:]:
                    obj = getattr(obj, attr)
            except Exception:  # noqa: BLE001 - an optional dependency may fail in any way
                if end == 1:
                    return None
                continue
            return obj
    return None


def _public_methods(cls: type) -> dict[str, object]:
    """Public methods and properties defined on ``cls`` itself, with static and class methods unwrapped."""
    methods: dict[str, object] = {}
    for name, value in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if inspect.isfunction(value) or isinstance(value, property):
            methods[name] = value
    return methods


def _module_members(module: object) -> dict[str, object]:
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
        members[name] = value
    return members


def _has_members_option(lines: list[str], index: int, indent: int) -> bool:
    for line in lines[index + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            return False
        if line.strip().startswith(":members:"):
            return True
    return False


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


def sphinx_registry(docs_source: Path) -> dict[str, object]:
    """Map every full name that ``docs/source/*.rst`` registers to the imported object.

    Reads ``autofunction``, ``autoclass``, ``automethod``, ``autoexception``, ``automodule`` and ``autosummary``
    entries, resolving relative names against the latest ``currentmodule`` or ``module`` and a nested ``automethod``
    against the enclosing class. ``autoclass`` and ``automodule`` with ``:members:`` register the public methods
    defined on the class and the public functions and classes of the module. Names that do not import are left out.
    """
    registry: dict[str, object] = {}

    def add(full_name: str) -> object | None:
        obj = _import_dotted(full_name)
        if obj is not None:
            registry[full_name] = obj
        return obj

    for page in sorted(Path(docs_source).glob("*.rst")):
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
            if kind == "autosummary":
                for name in _autosummary_names(lines, index, indent):
                    add(name if name.startswith("kornia") or not current else f"{current}.{name}")
            elif kind == "automodule":
                module = add(qualified)
                if module is not None and _has_members_option(lines, index, indent):
                    for name in _module_members(module):
                        add(f"{qualified}.{name}")
            else:
                obj = add(qualified)
                if kind == "autoclass":
                    enclosing_class = qualified
                    if inspect.isclass(obj) and _has_members_option(lines, index, indent):
                        for name, method in _public_methods(obj).items():
                            registry[f"{qualified}.{name}"] = method
    return registry


def resolve_reference(obj: object, target: str, registry: dict[str, object]) -> object | None:
    """Return the registered object that ``target``, written in the docstring of ``obj``, resolves to, or ``None``.

    A dotted target must be a key of ``registry``. A bare one is tried in the page of ``obj`` (its registered name
    without the last component) and, for a class, inside the class, as Sphinx resolves it. A method also finds the
    functions of its module. Short import aliases are not registered, so they do not resolve.
    """
    scopes: list[str] = []
    for name, registered in registry.items():
        if registered is obj:
            parent, _, _ = name.rpartition(".")
            scopes.append(parent)
            if inspect.isclass(obj):
                scopes.append(name)
            elif parent in registry and inspect.isclass(registry[parent]):
                scopes.append(parent.rpartition(".")[0])
    candidates = [target] if "." in target else [f"{scope}.{target}" for scope in scopes if scope]
    for candidate in candidates:
        if candidate in registry:
            return registry[candidate]
    return None


def reference_resolves(obj: object, target: str, registry: dict[str, object]) -> bool:
    """Whether ``target`` resolves from the docstring of ``obj`` to a registered object (see ``resolve_reference``)."""
    return resolve_reference(obj, target, registry) is not None


def catalog_missing(doc: str, names: list[str]) -> list[str]:
    """Return the ``names`` that ``doc`` does not mention as a whole token.

    A name is present only when no word character or dot comes before it and no word character comes after it, so
    ``depth_to_3d_v2`` does not count as mentioning ``depth_to_3d``.
    """
    return [name for name in names if not re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", doc)]


def catalog_required(namespace: str, catalogued: Iterable[str]) -> bool:
    """Whether ``namespace`` needs a catalog of its own.

    False when every object in its ``__all__`` is also exported, as the same object, by an ancestor namespace listed
    in ``catalogued``; true otherwise.
    """
    module = importlib.import_module(namespace)
    parts = namespace.split(".")
    ancestors = [importlib.import_module(".".join(parts[:end])) for end in range(1, len(parts))]
    ancestors = [ancestor for ancestor in ancestors if ancestor.__name__ in set(catalogued)]
    for name in getattr(module, "__all__", []):
        obj = getattr(module, name, None)
        if not any(
            name in getattr(ancestor, "__all__", []) and getattr(ancestor, name, None) is obj for ancestor in ancestors
        ):
            return True
    return False

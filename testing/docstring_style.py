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
import contextlib
import importlib
import inspect
import io
import re
import sys
import tokenize
import warnings
import zlib
from functools import cache
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple

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
def _exported_objects() -> tuple[tuple[object, str], ...]:
    """List ``(obj, "module.qualname")`` for every exported kornia function and class, deduplicated by identity."""
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
            if _is_kornia_function_or_class(obj):
                collected.setdefault(id(obj), (obj, f"{obj.__module__}.{obj.__qualname__}"))
    return tuple(collected.values())


@cache
def public_objects() -> dict[int, tuple[object, str]]:
    """Map ``id(obj)`` to ``(obj, "module.qualname")`` for every exported kornia function and class with a docstring.

    Covers every name in the ``__all__`` of a non-private kornia module that is a function or class defined in
    kornia and has a docstring of its own, deduplicated by object identity. Names starting with an underscore are
    out of scope. The dict is cached and shared between callers, who must not mutate it.
    """
    return {id(obj): (obj, qualified) for obj, qualified in _exported_objects() if own_docstring(obj)}


@cache
def undocumented_public_objects() -> dict[int, tuple[object, str]]:
    """Map ``id(obj)`` to ``(obj, "module.qualname")`` for every exported kornia function and class without a docstring.

    The complement of ``public_objects()`` over the same walk: ``public_objects()`` cannot see these, so a converted
    module could otherwise keep one forever. The dict is cached and shared between callers, who must not mutate it.
    """
    return {id(obj): (obj, qualified) for obj, qualified in _exported_objects() if not own_docstring(obj)}


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


# The Python-domain roles that link to an object. ``:mod:`` links to a module page and is not checked here.
_PY_ROLES = "func|class|meth|attr|data|const|exc|obj"
_XREF_ROLE = re.compile(rf":(?:py:)?({_PY_ROLES}):`~?([^`]+)`")


_STATUS_TITLE = re.compile(rf"(?:Alias of|Deprecated: use) :(?:py:)?(?:{_PY_ROLES}):`~?([^`\s]+)`\.")


def status_target(doc: str) -> str | None:
    """Return the target of a status title line, or ``None`` when ``doc`` has another title.

    The status titles are ``Alias of :role:`target`.`` and ``Deprecated: use :role:`target`.``. Only the first line
    counts: it must read exactly ``Alias of`` or ``Deprecated: use``, one Python-domain role with an optional ``py:``
    prefix and an optional leading ``~``, and a closing period. The returned target has no ``~``.
    """
    match = _STATUS_TITLE.fullmatch(doc.strip().split("\n", 1)[0].strip())
    return match.group(1) if match else None


def is_deprecated_wrapper(obj: object) -> bool:
    """Return whether ``obj`` is a function or class that ``kornia.core._compat.deprecated`` wraps.

    The decorator replaces a function by a wrapper, and the ``__init__`` of a class, with code of its own module.
    """
    import kornia.core._compat as compat

    code = getattr(obj.__init__ if isinstance(obj, type) else obj, "__code__", None)
    return code is not None and Path(code.co_filename).resolve() == Path(compat.__file__).resolve()


def docstring_roles(doc: str) -> list[tuple[str, str]]:
    """Return ``(role, target)`` for every Python-domain object role in ``doc``, without a leading ``~``.

    The roles are ``func``, ``class``, ``meth``, ``attr``, ``data``, ``const``, ``exc`` and ``obj``, with or without
    the ``py:`` prefix; ``:mod:`` and ``:math:`` are not object references.
    """
    return [(match.group(1), match.group(2)) for match in _XREF_ROLE.finditer(doc)]


_SEE_ALSO = re.compile(r"^(\s*)See Also:\s*$")


_LEADING_ROLES = re.compile(rf"(?::(?:py:)?(?:{_PY_ROLES}):`~?[^`]+`[\s,]*)+")


def see_also_entries(doc: str) -> list[tuple[str, str]] | None:
    """Parse the ``See Also:`` section of ``doc`` into ``(target, explanation)`` pairs.

    Returns ``None`` when there is no such section, and ``[]`` when it has no bullets. Each ``- `` bullet gives one
    pair per role in its leading role list (one or more roles separated by commas or whitespace); they share the
    explanation, which is the text after the colon that follows that list, joined across continuation lines
    (``""`` when missing). Roles inside the explanation are not targets. A bullet without a leading role renders
    without a link; it gives one pair ``("", text)`` with its whole text, so a caller can report it. The section
    ends at the first non-blank line indented at or below the ``See Also:`` header, or at the end of the docstring;
    text before the first bullet is ignored.
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
            entries.append(("", text))
            continue
        tail = text[leading.end() :].strip()
        explanation = tail[1:].strip() if tail.startswith(":") else ""
        entries.extend((match.group(2), explanation) for match in _XREF_ROLE.finditer(leading.group(0)))
    return entries


def is_private_target(target: str) -> bool:
    """Whether the last component of ``target`` is private: a leading underscore, unless it is a ``__name__``.

    A special method such as ``__mul__`` is public, and ``:special-members:`` documents it.
    """
    name = target.removesuffix("()").rpartition(".")[2]
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


_DIRECTIVE_KINDS = "autofunction|autoclass|automethod|autoexception|automodule|autosummary|currentmodule|module"
_DIRECTIVE = re.compile(rf"^(\s*)\.\. (?:py:)?({_DIRECTIVE_KINDS})::\s*(.*?)\s*$")
# The directive options the registry models, and the ones that do not change which names autodoc registers. Any
# other option, such as ``:no-index:``, ``:exclude-members:`` or ``:inherited-members:``, raises, so that a page that
# adds one fails the tests instead of silently making the registry disagree with Sphinx.
_KNOWN_OPTIONS = frozenset(
    {"members", "undoc-members", "special-members", "toctree", "show-inheritance", "member-order", "nosignatures"}
)


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


class _Attribute(NamedTuple):
    """What the registry maps a documented data attribute to, in place of its value, which may be ``None``."""

    owner: object
    name: str


_ATTRIBUTE_COMMENT = re.compile(r"^\s*#: ?(.*)$")


class _AttributeDocPicker(ast.NodeVisitor):
    """Collect ``(namespace, name)`` of the documented variables of a module, as ``VariableCommentPicker`` does.

    An assignment documents its targets with a ``#:`` comment after it, with ``#:`` lines right above it, or with
    a string literal right after it. The namespace is the enclosing class path, ``""`` at module level. Inside a
    function only ``self.name`` assignments in a class's ``__init__`` count, under that class; nested functions and
    ``except`` and ``finally`` blocks are not read.
    """

    def __init__(self, lines: list[str], comment_columns: dict[int, int]) -> None:
        self.lines = lines
        self.comment_columns = comment_columns
        self.context: list[str] = []
        self.classes = 0
        self.function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        self.previous: ast.AST | None = None
        self.found: set[tuple[str, str]] = set()

    def visit(self, node: ast.AST) -> None:
        super().visit(node)
        self.previous = node

    def _self_name(self) -> str | None:
        if self.function is None:
            return None
        arguments = self.function.args.args or self.function.args.posonlyargs
        return arguments[0].arg if arguments else None

    def _names(self, target: ast.AST) -> list[str]:
        """The variables an assignment target creates; ``TypeError`` when it creates none (``a[0] = ...``)."""
        self_name = self._self_name()
        if isinstance(target, ast.Name):
            if self_name is None or target.id == self_name:
                return [target.id]
        elif isinstance(target, (ast.Tuple, ast.List)):
            names = []
            for element in target.elts:
                with contextlib.suppress(TypeError):
                    names += self._names(element)
            return names
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if self_name is not None and target.value.id == self_name:
                return [target.attr]
        elif isinstance(target, ast.Starred):
            return self._names(target.value)
        raise TypeError(target)

    def _namespace(self) -> str | None:
        if self.function is None:
            return ".".join(self.context)
        if self.classes and self.context[-1] == "__init__":
            return ".".join(self.context[:-1])
        return None

    def _record(self, names: list[str]) -> None:
        namespace = self._namespace()
        if namespace is not None:
            self.found.update((namespace, name) for name in names)

    def visit_Assign(self, node: ast.Assign | ast.AnnAssign) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        try:
            names = [name for target in targets for name in self._names(target)]
        except TypeError:
            return
        column = self.comment_columns.get(node.end_lineno or node.lineno)
        if (column is not None and column >= (node.end_col_offset or 0)) or (
            node.lineno > 1
            and not self.lines[node.lineno - 1][: node.col_offset].strip()
            and _ATTRIBUTE_COMMENT.match(self.lines[node.lineno - 2])
        ):
            self._record(names)

    visit_AnnAssign = visit_Assign

    def visit_Expr(self, node: ast.Expr) -> None:
        previous = self.previous
        if (
            isinstance(previous, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            target = previous.targets[0] if isinstance(previous, ast.Assign) else previous.target
            with contextlib.suppress(TypeError):
                self._record(self._names(target))

    def visit_Try(self, node: ast.Try) -> None:
        for child in [*node.body, *node.orelse]:
            self.visit(child)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes += 1
        self.context.append(node.name)
        self.previous = node
        for child in node.body:
            self.visit(child)
        self.context.pop()
        self.classes -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.function is not None:
            return
        self.context.append(node.name)
        self.function = node
        for child in node.body:
            self.visit(child)
        self.context.pop()
        self.function = None

    visit_AsyncFunctionDef = visit_FunctionDef


@cache
def _attribute_docs(module_name: str) -> frozenset[tuple[str, str]]:
    """``(namespace, name)`` of the variables the source of ``module_name`` documents (see ``_AttributeDocPicker``)."""
    path = getattr(sys.modules.get(module_name), "__file__", None)
    if not path or not path.endswith(".py"):
        return frozenset()
    source = Path(path).read_text(encoding="utf-8")
    comment_columns = {
        token.start[0]: token.start[1]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and _ATTRIBUTE_COMMENT.match(token.string)
    }
    picker = _AttributeDocPicker(source.splitlines(), comment_columns)
    picker.visit(ast.parse(source))
    return frozenset(picker.found)


def _is_data(value: object) -> bool:
    """Whether autodoc documents ``value`` as data rather than as a function, method, property, class or module."""
    if isinstance(value, (staticmethod, classmethod)):
        return False
    return not (
        inspect.isroutine(value) or isinstance(value, property) or inspect.isclass(value) or inspect.ismodule(value)
    )


def _documented_attributes(owner: object, undoc_members: bool) -> dict[str, _Attribute]:
    """The public data attributes of the class or module ``owner`` that autodoc's ``:members:`` documents.

    Mirrors ``get_class_members``, ``get_module_members`` and ``filter_members``. A class contributes the data in
    its own ``__dict__``, enum members included, its own annotations (``x: int``, dataclass fields) and the instance
    attributes its ``__init__`` documents; such an attribute is documented when the source documents it (a ``#:``
    comment or a string after the assignment), when its value has a docstring other than its type's (a
    ``NamedTuple`` field), or under ``:undoc-members:``. A module's data is documented only when the source
    documents it, or, under ``:undoc-members:``, when it is a bare annotation, as ``DataDocumenter`` takes nothing
    else; with ``__all__``, only the names in it.
    """
    if inspect.isclass(owner):
        module_name, namespace = owner.__module__, owner.__qualname__
        values = {name: value for name, value in vars(owner).items() if _is_data(value)}
        annotated = list(inspect.get_annotations(owner))
        documented = {name for scope, name in _attribute_docs(module_name) if scope == namespace}
    else:
        module_name, namespace = getattr(owner, "__name__", ""), ""
        declared = getattr(owner, "__all__", None)
        annotations = inspect.get_annotations(owner)
        present = set(vars(owner)) | set(annotations) if declared is None else set(declared)
        values = {}
        annotated = [name for name in annotations if name in present and name not in vars(owner)]
        documented = {name for scope, name in _attribute_docs(module_name) if scope == namespace and name in present}

    def kept(name: str) -> bool:
        if name.startswith("_"):
            return False
        return (
            name in documented
            or undoc_members
            or (name in values and _autodoc_documents(owner, name, values[name], False))
        )

    candidates = [*values, *annotated, *sorted(documented)]
    return {name: _Attribute(owner, name) for name in dict.fromkeys(candidates) if kept(name)}


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
    """Register the members ``:members:`` documents: the listed names, or the public ones defined on ``cls``.

    Either way autodoc keeps only members with a docstring, unless the directive has ``:undoc-members:``; a data
    attribute counts as documented as ``_documented_attributes`` says. With all members, ``special``
    (``:special-members:``) selects the ``__name__`` methods as well.
    """
    if not inspect.isclass(cls):
        return
    attributes = _documented_attributes(cls, undoc_members)
    if listed:
        for name in listed:
            if name in attributes:
                registry[f"{cls_name}.{name}"] = attributes[name]
                continue
            member = _import_dotted(f"{cls_name}.{name}")
            if member is not None and _autodoc_documents(cls, name, member, undoc_members):
                registry[f"{cls_name}.{name}"] = member
    else:
        for name, method in _public_methods(cls, undoc_members, special).items():
            registry[f"{cls_name}.{name}"] = method
        for name, attribute in attributes.items():
            registry[f"{cls_name}.{name}"] = attribute


def _add_module_members(
    registry: dict[str, object],
    module_name: str,
    module: object,
    listed: list[str],
    undoc_members: bool,
    special: list[str] | None = None,
) -> None:
    attributes = _documented_attributes(module, undoc_members)
    if listed:
        attributes = {name: attribute for name, attribute in attributes.items() if name in listed}
        members = {name: getattr(module, name, None) for name in listed if name not in attributes}
        members = {
            name: member
            for name, member in members.items()
            if member is not None and not _is_data(member) and _autodoc_documents(module, name, member, undoc_members)
        }
    else:
        members = _module_members(module, undoc_members)
    for name, attribute in attributes.items():
        registry[f"{module_name}.{name}"] = attribute
    for name, member in members.items():
        registry[f"{module_name}.{name}"] = member
        _add_methods(registry, f"{module_name}.{name}", member, [], undoc_members, special)


# Directories of ``docs/source`` that Sphinx does not read pages from: ``conf.py``'s ``exclude_patterns``,
# ``templates_path``, ``html_static_path`` and ``html_extra_path``.
_NOT_PAGES = frozenset({"_build", ".ipynb_checkpoints", "_templates", "_static", "_extra"})


def rst_pages(docs_source: Path) -> list[Path]:
    """Every ``.rst`` page under ``docs_source``, subdirectories included, except those Sphinx does not build."""
    root = Path(docs_source)
    return sorted(page for page in root.rglob("*.rst") if _NOT_PAGES.isdisjoint(page.relative_to(root).parts[:-1]))


def sphinx_registry(docs_source: Path) -> dict[str, object]:
    """Map every full name that the ``.rst`` pages under ``docs_source`` register to the imported object.

    Reads ``autofunction``, ``autoclass``, ``automethod``, ``autoexception``, ``automodule`` and ``autosummary``
    entries, resolving relative names against the latest ``currentmodule`` or ``module`` and a nested ``automethod``
    against the enclosing class. ``autoclass`` and ``automodule`` with ``:members:`` register the methods and
    properties defined on the class and the functions and classes of the module that autodoc documents (public, with
    a docstring or under ``:undoc-members:``), the data attributes it documents (enum members, dataclass fields and
    commented variables; see ``_documented_attributes``), each mapped to an ``_Attribute`` rather than its value, and
    the ``__name__`` methods that ``:special-members:`` selects. An ``autosummary`` registers its entries only with
    ``:toctree:``, which generates their pages; without it, it only links to them. Names that do not import are left
    out. A directive option outside the ones modelled here raises ``ValueError``.
    """
    registry: dict[str, object] = {}

    for page in rst_pages(docs_source):
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
            unknown = sorted(set(options) - _KNOWN_OPTIONS)
            if unknown:
                raise ValueError(
                    f"{page}:{index + 1}: sphinx_registry does not model the {kind} option(s) {unknown}; "
                    "teach it what autodoc registers with them"
                )
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
_ALL_OBJECT_TYPES = frozenset(
    "function data class exception method classmethod staticmethod attribute property type module".split()
)
# The Python object types that intersphinx links each role to: the roles of the Python domain's ``object_types``,
# plus ``method`` for ``:attr:``, where inventories from before Sphinx 2.1 store properties. ``:const:`` is in no
# object type's roles, so intersphinx links it to nothing.
_ROLE_OBJECT_TYPES: dict[str, frozenset[str]] = {
    "func": frozenset({"function"}),
    "data": frozenset({"data"}),
    "class": frozenset({"class", "exception"}),
    "exc": frozenset({"class", "exception"}),
    "meth": frozenset({"method", "classmethod", "staticmethod"}),
    "attr": frozenset({"attribute", "property", "method"}),
    "const": frozenset(),
    "obj": _ALL_OBJECT_TYPES,
}


def intersphinx_names(docs_source: Path) -> dict[str, frozenset[str]]:
    """Map each Python object name of the cached intersphinx inventories in ``docs_source/_intersphinx`` to its types.

    ``conf.py`` falls back to these ``python``, ``numpy`` and ``torch`` inventories when it cannot fetch the live
    ones. Each file is a Sphinx inventory version 2: four header lines, then zlib-compressed
    ``name domain:role priority uri dispname`` lines. The types are the Python domain's object types without the
    ``py:`` prefix, such as ``function`` or ``attribute``; ``reference_resolves`` matches them against the role.
    """
    types: dict[str, set[str]] = {}
    for path in sorted((Path(docs_source) / "_intersphinx").glob("*.inv")):
        with path.open("rb") as stream:
            header = [stream.readline() for _ in range(4)]
            if not header[0].startswith(b"# Sphinx inventory version 2") or b"zlib" not in header[3]:
                raise ValueError(f"{path} is not a zlib-compressed Sphinx inventory version 2")
            body = zlib.decompress(stream.read()).decode("utf-8")
        for line in body.splitlines():
            match = _INVENTORY_LINE.match(line.rstrip())
            if match is not None and match.group(2).startswith("py:"):
                types.setdefault(match.group(1), set()).add(match.group(2).removeprefix("py:"))
    return {name: frozenset(kinds) for name, kinds in types.items()}


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


def _role_target(target: str, role: str | None) -> str:
    """``target`` as Sphinx looks it up: ``:func:`` and ``:meth:`` drop a trailing ``()``, other roles keep it."""
    return target.removesuffix("()") if role in (None, "func", "meth") else target


def resolve_reference(obj: object, target: str, registry: dict[str, object], role: str | None = None) -> object | None:
    """Return the registered object that ``target``, written in the docstring of ``obj``, resolves to, or ``None``.

    As Sphinx does for a Python-domain role: a trailing ``()`` is dropped for ``:func:`` and ``:meth:`` (and when
    ``role`` is ``None``), then the target is tried as written and under each scope of ``obj``, the module it is
    documented under and, for a class or a method, the class, so ``to_tensor`` and ``Boxes.to_tensor`` both resolve
    inside the ``Boxes`` page. The role does not restrict the object type: Sphinx links a name it registers whatever
    its type. A short import alias is not registered, so it does not resolve; neither does a short name in the
    docstring of an object that no docs page registers. Only kornia objects resolve here; see
    ``intersphinx_names`` for the rest.
    """
    target = _role_target(target, role)
    candidates = [target, *(f"{scope}.{target}" for scope in _reference_scopes(obj, registry))]
    for candidate in candidates:
        if candidate in registry:
            return registry[candidate]
    return None


def reference_resolves(
    obj: object,
    target: str,
    registry: dict[str, object],
    external: Mapping[str, frozenset[str]] | None = None,
    role: str | None = None,
) -> bool:
    """Whether ``target``, in a ``role`` role in the docstring of ``obj``, renders as a link.

    True when it resolves to a registered kornia object (see ``resolve_reference``) or is an ``external`` name, such
    as one from ``intersphinx_names(docs_source)``, of an object type that intersphinx links ``role`` to. Unlike the
    local lookup, intersphinx matches the type: ``:meth:`torch.svd``` (a function) and every ``:const:`` stay plain
    text. A ``role`` of ``None`` accepts any type.
    """
    if resolve_reference(obj, target, registry, role) is not None:
        return True
    types = (external or {}).get(_role_target(target, role), frozenset())
    return bool(types if role is None else types & _ROLE_OBJECT_TYPES.get(role, frozenset()))


def catalog_missing(doc: str, names: list[str], namespace: object) -> list[str]:
    """Return the ``names`` exported by ``namespace`` that ``doc`` mentions in no Python role or literal.

    A name counts only as the whole target of a Python-domain role, such as :func:`name` or :func:`~kornia.x.name`,
    or as a double-backtick literal, such as ``name`` or ``name()``. A qualified target counts only when it imports
    as the object ``namespace`` exports under that name, so ``torch.nn.functional.normalize`` does not mention
    ``kornia.enhance.normalize``. The same word in plain prose (``normalize``, ``threshold``), in single backticks
    (the default role, which renders as italics) or in a ``:math:`` role does not count, and ``depth_to_3d_v2`` does
    not count as mentioning ``depth_to_3d``.
    """

    def mentioned(name: str) -> bool:
        target = rf"((?:[\w.]*\.)?{re.escape(name)})(?:\(\))?"
        exported = getattr(namespace, name, None)
        for match in re.finditer(rf"``{target}``|:(?:py:)?(?:{_PY_ROLES}):`~?{target}`", doc):
            written = match.group(1) or match.group(2)
            if written == name or (exported is not None and _import_dotted(written) is exported):
                return True
        return False

    return [name for name in names if not mentioned(name)]


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

    Reads every Python-domain object role (see ``docstring_roles``). A bare target is looked up on ``module``, a
    dotted one from the longest importable module prefix.
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

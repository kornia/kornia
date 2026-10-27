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

"""No two public kornia functions or classes share a first docstring line.

``help()``, IDE hovers, the Sphinx autosummary tables and coding agents often read only the first docstring line.
When two different symbols share it -- ``match_nn``, ``match_snn`` and ``match_fginn`` once all said "Find nearest
neighbors in desc2 for each vector in desc1" -- that reader cannot tell which one it needs. Each first line therefore
names what sets its symbol apart from its siblings: the dimension, variant, criterion, input form or direction.

The one exception is a class and the function it wraps, whose names match once underscores are removed and case is
ignored (``AdjustHue`` / ``adjust_hue``, ``PyrDown`` / ``pyrdown``) and which live in the same top-level subpackage
(``kornia.enhance``): they do the same thing and may say so in the same words. A cross-package pair such as
``kornia.augmentation.Resize`` and ``kornia.geometry.transform.resize`` is not exempt.

The scan covers every name in the ``__all__`` of a non-private kornia module that is a function or a class defined
in kornia, deduplicated by object identity. Names that start with an underscore are out of scope, and so is a class
without a docstring of its own, which inherits one from its base. Lines are compared exactly after casefolding and
collapsing whitespace, so this catches copies, not near-duplicates: two lines that differ by a single word pass.
"""

import ast
import importlib
import inspect
import warnings
from functools import cache
from pathlib import Path

import kornia

_KORNIA_ROOT = Path(kornia.__file__).resolve().parent


def _modules_declaring_all() -> list[str]:
    """Name every non-private kornia module that assigns ``__all__``, found from source without importing it.

    The same parse as ``_declared_inventory_modules`` in ``tests/test_api_surface.py``, over the whole package
    rather than the inventory roots: the inventory leaves out ``kornia.feature``, whose matchers and LAF estimators
    are among the symbols this test exists for.
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


def _first_line(doc: str) -> str:
    return next((line.strip() for line in inspect.cleandoc(doc).splitlines() if line.strip()), "")


def _qualified_name(obj: object) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _squash(name: str) -> str:
    return name.replace("_", "").casefold()


def _subpackage(obj: object) -> str:
    """Return the top-level kornia subpackage that defines ``obj``, such as ``kornia.enhance``."""
    return ".".join(obj.__module__.split(".")[:2])


def _is_wrapper_pair(objects: list) -> bool:
    """Whether ``objects`` is one class and one function of one subpackage whose names match without underscores."""
    if len(objects) != 2:
        return False
    classes = [obj for obj in objects if inspect.isclass(obj)]
    functions = [obj for obj in objects if inspect.isfunction(obj)]
    return (
        len(classes) == 1
        and len(functions) == 1
        and _squash(classes[0].__name__) == _squash(functions[0].__name__)
        and _subpackage(classes[0]) == _subpackage(functions[0])
    )


@cache
def _public_first_lines() -> dict[int, tuple[object, str]]:
    """Map ``id(obj)`` to ``(obj, first docstring line)`` for every public kornia function and class."""
    collected: dict[int, tuple[object, str]] = {}
    for module_name in _modules_declaring_all():
        with warnings.catch_warnings():
            # ``getattr`` runs module-level ``__getattr__`` shims, which warn about deprecated names.
            warnings.simplefilter("ignore")
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as error:
                # A module that needs an optional package which is not installed is out of scope here.
                if (error.name or "").partition(".")[0] == "kornia":
                    raise
                continue
            exported = [(name, getattr(module, name, None)) for name in module.__all__ if not name.startswith("_")]
        for _, obj in exported:
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            module_of_obj = obj.__module__ or ""
            if not (module_of_obj == "kornia" or module_of_obj.startswith("kornia.")):
                continue
            # ``inspect.getdoc`` would hand a class without a docstring its base class's one.
            doc = obj.__dict__.get("__doc__") if inspect.isclass(obj) else obj.__doc__
            if doc:
                collected[id(obj)] = (obj, _first_line(doc))
    return collected


def test_the_scan_covers_the_public_api():
    # Guards against a vacuous pass: a discovery that silently stopped importing modules would find no duplicates.
    collected = _public_first_lines()
    assert len(collected) >= 800, len(collected)
    # Looked up through public names and matched by identity, so moving a definition to another file keeps this.
    for obj in (
        kornia.augmentation.ColorJitter,
        kornia.feature.match_snn,
        kornia.geometry.subpix.nms2d,
        kornia.losses.FocalLoss,
    ):
        assert id(obj) in collected, _qualified_name(obj)
    # A known wrapper pair is scanned, shares its line, and is allowed to.
    assert collected[id(kornia.enhance.AdjustHue)][1] == collected[id(kornia.enhance.adjust_hue)][1]
    assert _is_wrapper_pair([kornia.enhance.AdjustHue, kornia.enhance.adjust_hue])


def test_the_wrapper_pair_rule():
    from kornia.enhance import adjust_hue, adjust_hue_raw
    from kornia.geometry.subpix import NonMaximaSuppression2d, nms2d
    from kornia.geometry.transform import PyrDown, pyrdown

    assert _is_wrapper_pair([PyrDown, pyrdown])
    assert not _is_wrapper_pair([NonMaximaSuppression2d, nms2d])
    assert not _is_wrapper_pair([adjust_hue, adjust_hue_raw])
    assert not _is_wrapper_pair([kornia.enhance.AdjustHue, adjust_hue, adjust_hue_raw])
    # Same name after squashing, but different subpackages.
    assert not _is_wrapper_pair([kornia.augmentation.Resize, kornia.geometry.transform.resize])


def test_first_docstring_lines_are_distinct():
    groups: dict[str, list] = {}
    for obj, line in _public_first_lines().values():
        # Case and spacing changes do not make two lines distinguishable to a reader.
        groups.setdefault(" ".join(line.casefold().split()), []).append((obj, line))

    offending = sorted(
        (sorted(_qualified_name(obj) for obj, _ in members), members[0][1])
        for members in groups.values()
        if len(members) > 1 and not _is_wrapper_pair([obj for obj, _ in members])
    )
    report = "\n".join(f"  {line!r}\n" + "\n".join(f"      {name}" for name in names) for names, line in offending)
    assert not offending, (
        f"{len(offending)} first docstring line(s) are shared by different public symbols, so help(), IDE hovers and "
        "the API tables cannot tell them apart. Give each symbol a first line that names what sets it apart from the "
        "others: dimension, variant, criterion, input form or direction. Only a class and the function it wraps, "
        "defined in the same subpackage with names that match without underscores (AdjustHue / adjust_hue), may "
        "share one.\n" + report
    )

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

"""Guard aggregate and public-module API surfaces within the recorded package roots.

The 0.8.3 release removed 20 of the 34 public ``kornia.utils`` names with no
deprecation window — nobody noticed until an audit months later, because
nothing in CI compared the public surface against the previous state. This
test is that comparison: ``tests/api_surface.json`` is the checked-in
inventory, and removing a public name fails here until the inventory is
edited explicitly in the same PR — which is exactly the deliberate review moment the
stability policy (docs/source/get-started/stability.rst) requires.

Additions are fine and do not fail the test; run this file with
``--update-api-surface`` semantics by regenerating the JSON (see
``regenerate()`` below) whenever new public API lands.
"""

import ast
import contextlib
import importlib
import json
import pkgutil
import sys
import types
import warnings
from pathlib import Path

import pytest

INVENTORY = Path(__file__).parent / "api_surface.json"
INVENTORY_ROOTS = (
    "kornia.augmentation",
    "kornia.color",
    "kornia.core",
    "kornia.enhance",
    "kornia.feature.sift",
    "kornia.filters",
    "kornia.geometry",
    "kornia.image",
    "kornia.io",
    "kornia.losses",
    "kornia.metrics",
    "kornia.morphology",
    "kornia.utils",
)


def _current_surface(module_name: str) -> set:
    mod = importlib.import_module(module_name)
    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n in dir(mod) if not n.startswith("_")]
    return set(names)


def _declared_inventory_modules(package_root: Path) -> list[str]:
    """Find explicit export declarations from source without importing optional dependencies."""
    found = []
    for path in package_root.rglob("*.py"):
        parts = list(path.relative_to(package_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
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
    return sorted(found)


def _inventory_modules(recorded: dict) -> list[str]:
    """Retain existing rows and discover public modules within fixed package roots.

    The SIFT descriptor module is classical feature API. Pretrained wrappers,
    private paths and packages outside these roots keep their existing policy.
    """
    roots = set(INVENTORY_ROOTS)
    discovered = {
        name
        for name in _declared_inventory_modules(Path(__file__).resolve().parents[1] / "kornia")
        if not any(part.startswith("_") for part in name.split(".")[1:])
        and any(name == root or name.startswith(root + ".") for root in roots)
    }
    return sorted(set(recorded) | roots | discovered)


def regenerate() -> None:
    """Add live public exports to the inventory; deliberate removals must be edited explicitly."""
    recorded = json.loads(INVENTORY.read_text())
    surface = {
        name: sorted(
            set(recorded.get(name, [])) | {item for item in _current_surface(name) if not item.startswith("_")}
        )
        for name in _inventory_modules(recorded)
    }
    INVENTORY.write_text(json.dumps(surface, indent=2, sort_keys=True) + "\n")


def test_inventory_includes_public_modules_declaring_all():
    recorded = json.loads(INVENTORY.read_text())
    missing = sorted(set(_inventory_modules(recorded)) - set(recorded))
    assert not missing, f"Public modules absent from the inventory: {missing}. Run tests.test_api_surface.regenerate()."


def test_inventory_module_discovery_preserves_scope(monkeypatch):
    module = sys.modules[__name__]
    recorded = {"kornia.geometry": [], "kornia.utils": []}
    monkeypatch.setattr(module, "INVENTORY_ROOTS", ("kornia.geometry", "kornia.feature.sift"))
    monkeypatch.setattr(
        module,
        "_declared_inventory_modules",
        lambda path: [
            "kornia.geometry.boxes",
            "kornia.geometry._private",
            "kornia.geometry._private.child",
            "kornia.feature.sift",
            "kornia.contrib",
            "kornia.models",
        ],
    )
    assert _inventory_modules(recorded) == [
        "kornia.feature.sift",
        "kornia.geometry",
        "kornia.geometry.boxes",
        "kornia.utils",
    ]


def test_inventory_discovery_does_not_import_optional_dependencies(tmp_path):
    package = tmp_path / "kornia"
    package.mkdir()
    (package / "__init__.py").write_text("__all__ = []")
    (package / "optional.py").write_text("import unavailable_dependency\n__all__: list[str] = ['Export']")
    (package / "unrelated.py").write_text("value = 1")
    assert _declared_inventory_modules(package) == ["kornia", "kornia.optional"]


def test_inventory_roots_do_not_depend_on_existing_rows(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_declared_inventory_modules", lambda path: [])
    assert _inventory_modules({}) == sorted(INVENTORY_ROOTS)


@pytest.mark.parametrize(
    "module_name,name",
    [
        ("kornia.geometry.boxes", "Boxes"),
        ("kornia.geometry.boxes", "VideoBoxes"),
        ("kornia.augmentation.auto", "AutoAugment"),
    ],
)
def test_removal_guard_covers_previously_unrecorded_modules(monkeypatch, module_name, name):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "__all__", [item for item in module.__all__ if item != name])
    with pytest.raises(AssertionError, match=f"Public names removed from {module_name}"):
        test_no_public_name_removed(module_name)


def test_regenerate_discovers_modules_without_losing_aggregate_rows(monkeypatch, tmp_path):
    module = sys.modules[__name__]
    inventory = tmp_path / "api_surface.json"
    inventory.write_text(json.dumps({"kornia.geometry": ["old"]}))
    monkeypatch.setattr(module, "INVENTORY", inventory)
    monkeypatch.setattr(module, "INVENTORY_ROOTS", ("kornia.geometry",))
    monkeypatch.setattr(module, "_declared_inventory_modules", lambda path: ["kornia.geometry.boxes"])
    monkeypatch.setattr(module, "_current_surface", lambda name: {name.rsplit(".", 1)[-1], "_private"})
    regenerate()
    assert json.loads(inventory.read_text()) == {
        "kornia.geometry": ["geometry", "old"],
        "kornia.geometry.boxes": ["boxes"],
    }
    first = inventory.read_bytes()
    regenerate()
    assert inventory.read_bytes() == first


def test_regenerate_import_failure_preserves_inventory(monkeypatch, tmp_path):
    module = sys.modules[__name__]
    inventory = tmp_path / "api_surface.json"
    inventory.write_text(json.dumps({"kornia.geometry": ["old"]}))
    original = inventory.read_bytes()
    monkeypatch.setattr(module, "INVENTORY", inventory)
    monkeypatch.setattr(module, "INVENTORY_ROOTS", ("kornia.geometry",))
    monkeypatch.setattr(module, "_declared_inventory_modules", lambda path: ["kornia.geometry.optional"])

    def unavailable(name):
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(module, "_current_surface", unavailable)
    with pytest.raises(ImportError, match="missing optional dependency"):
        regenerate()
    assert inventory.read_bytes() == original


def test_inventory_is_a_mapping_of_module_to_public_names():
    """Guard the inventory's shape, so the removal test below cannot pass vacuously.

    ``set(recorded) - current`` is empty for an entry mangled into ``{}``, so that entry
    silently stops guarding its module while still parametrizing this file. A list
    holding non-strings fails the removal test instead, but with a message about
    "removed" names that names nothing a reader can act on. Asserting the shape here
    reports the real problem in both cases -- and the ``Import Surface`` check reads the
    same file, where a wrong-shaped entry acknowledges nothing at all.
    """
    inventory = json.loads(INVENTORY.read_text())
    assert isinstance(inventory, dict)
    for module_name, names in inventory.items():
        assert isinstance(module_name, str), f"non-string inventory key: {module_name!r}"
        assert isinstance(names, list), f"{module_name} must record a list of names, got {type(names).__name__}"
        assert all(isinstance(name, str) for name in names), f"{module_name} records a non-string name"


@pytest.mark.parametrize("module_name", sorted(json.loads(INVENTORY.read_text())))
def test_no_public_name_removed(module_name):
    recorded = set(json.loads(INVENTORY.read_text())[module_name])
    current = _current_surface(module_name)
    removed = sorted(recorded - current)
    assert not removed, (
        f"Public names removed from {module_name}: {removed}. Per the API stability policy "
        "(docs/source/get-started/stability.rst), a public symbol must spend at least one minor "
        "release as a deprecated shim before removal. If this removal is deliberate and the "
        "deprecation window has passed, edit the affected tests/api_surface.json row in this PR "
        "and list the removal in the release notes."
    )


@pytest.mark.parametrize("alias", ["np", "ort"])
def test_lightglue_onnx_dependencies_are_private(alias):
    # #4026 Group 2: lazy imports must not reintroduce accidental public globals.
    from kornia.core import external
    from kornia.feature import OnnxLightGlue
    from kornia.feature.lightglue_onnx import lightglue

    assert lightglue.__all__ == ["OnnxLightGlue"]
    assert lightglue.OnnxLightGlue is OnnxLightGlue
    assert not hasattr(lightglue, alias)
    dependency = {"np": external.numpy, "ort": external.onnxruntime}[alias]
    assert getattr(lightglue, f"_{alias}") is dependency


def _foreign_names(module_name: str) -> set:
    """Public names of ``module_name`` bound to an import rather than to its own API.

    That is another package's module, a ``typing`` object, or an object defined in a kornia
    module outside ``module_name``'s own subtree. A module without ``__all__`` exposes whatever
    its star-imported submodules happen to import, so ``import sys`` in one of them makes
    ``kornia.geometry.sys`` public, and ``from kornia.core.utils import is_compiling`` makes
    ``kornia.geometry.is_compiling`` public (#4709).
    """
    mod = importlib.import_module(module_name)
    found = set()
    for name in _current_surface(module_name):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if isinstance(obj, types.ModuleType):
            if obj.__name__ != "kornia" and not obj.__name__.startswith("kornia."):
                found.add(name)
        else:
            owner = getattr(obj, "__module__", None)
            if owner == "typing" or (
                isinstance(owner, str)
                and owner.startswith("kornia.")
                and owner != module_name
                and not owner.startswith(module_name + ".")
            ):
                found.add(name)
    return found


@pytest.mark.parametrize("module_name", sorted(json.loads(INVENTORY.read_text())))
def test_no_foreign_name_leaks_into_the_surface(module_name):
    """An import inside a submodule must not become public API by accident.

    Additions do not fail ``test_no_public_name_removed``, and ``regenerate()`` would
    record a leaked ``sys`` as a guarded public name. Names already in the inventory are
    exempt: removing one is a deliberate API change, not something this test should force.
    """
    recorded = set(json.loads(INVENTORY.read_text())[module_name])
    leaked = sorted(_foreign_names(module_name) - recorded)
    assert not leaked, (
        f"{module_name} exposes {leaked}, which are imports rather than kornia API. Give the "
        "submodule that imports them an __all__ (or import them under a private name). Record a "
        "name in tests/api_surface.json only for a deliberate cross-package re-export."
    )


def _modules_declaring_all() -> list:
    """Every importable ``kornia`` submodule that declares ``__all__``.

    An unbound name is a problem even outside the inventory: the breakage that
    motivated this guard (#4026, the ComfyUI-LTXVideo report in #3986) was in
    ``kornia.geometry.transform.pyramid``. Walking the package finds every module that
    declares a public surface, costs a fraction of a second, and needs no upkeep as
    modules come and go.
    """
    import kornia

    found = []
    for info in pkgutil.walk_packages(kornia.__path__, "kornia."):
        # Optional extras and backends are out of scope for this guard, so a module that
        # cannot be imported here is skipped rather than failing the sweep.
        with contextlib.suppress(Exception):
            mod = importlib.import_module(info.name)
            if getattr(mod, "__all__", None) is not None:
                found.append(info.name)
    return sorted(found)


@pytest.mark.parametrize("module_name", _modules_declaring_all())
def test_public_names_resolve(module_name):
    """
    Every name in ``__all__`` has to be bound in its module.

    An unbound name breaks ``from <module> import *`` for everyone, and
    ``test_no_public_name_removed`` cannot see it: the surface that test compares is
    ``__all__`` itself, so a name whose binding was dropped still reads as present.
    """
    mod = importlib.import_module(module_name)
    declared = mod.__all__

    with warnings.catch_warnings():
        # ``hasattr`` runs module-level ``__getattr__``, which is how the deprecated
        # ``kornia.utils`` re-exports announce themselves. Keep the probe quiet so it
        # does not emit a DeprecationWarning per run just by looking.
        warnings.simplefilter("ignore")
        unbound = sorted(name for name in declared if not hasattr(mod, name))

    assert not unbound, (
        f"{module_name}.__all__ lists names that are not bound in the module: "
        f"{unbound}. `from {module_name} import *` raises AttributeError on the "
        "first of them. Either restore the binding or drop the name from __all__ "
        "and from tests/api_surface.json in the same PR."
    )


if __name__ == "__main__":
    regenerate()
    print(f"regenerated {INVENTORY}")

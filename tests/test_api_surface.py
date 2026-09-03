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

"""Guard the public API surface of the stable-core modules.

The 0.8.3 release removed 20 of the 34 public ``kornia.utils`` names with no
deprecation window — nobody noticed until an audit months later, because
nothing in CI compared the public surface against the previous state. This
test is that comparison: ``tests/api_surface.json`` is the checked-in
inventory, and removing a public name fails here until the inventory is
edited in the same PR — which is exactly the deliberate review moment the
stability policy (docs/source/get-started/stability.rst) requires.

Additions are fine and do not fail the test; run this file with
``--update-api-surface`` semantics by regenerating the JSON (see
``regenerate()`` below) whenever new public API lands.
"""

import contextlib
import importlib
import json
import pkgutil
import warnings
from pathlib import Path

import pytest

INVENTORY = Path(__file__).parent / "api_surface.json"


def _current_surface(module_name: str) -> set:
    mod = importlib.import_module(module_name)
    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n in dir(mod) if not n.startswith("_")]
    return set(names)


def regenerate() -> None:
    """Rewrite the inventory from the live library (run manually after adding public API)."""
    recorded = json.loads(INVENTORY.read_text())
    surface = {name: sorted(_current_surface(name)) for name in recorded}
    INVENTORY.write_text(json.dumps(surface, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("module_name", sorted(json.loads(INVENTORY.read_text())))
def test_no_public_name_removed(module_name):
    recorded = set(json.loads(INVENTORY.read_text())[module_name])
    current = _current_surface(module_name)
    removed = sorted(recorded - current)
    assert not removed, (
        f"Public names removed from {module_name}: {removed}. Per the API stability policy "
        "(docs/source/get-started/stability.rst), a public symbol must spend at least one minor "
        "release as a deprecated shim before removal. If this removal is deliberate and the "
        "deprecation window has passed, update tests/api_surface.json in this PR (run "
        "tests.test_api_surface.regenerate()) and list the removal in the release notes."
    )


def _modules_declaring_all() -> list:
    """Every importable ``kornia`` submodule that declares ``__all__``.

    The inventory tracks the stable-core modules, but an unbound name is a problem
    wherever it sits: the breakage that motivated this guard (#4026, the
    ComfyUI-LTXVideo report in #3986) was in ``kornia.geometry.transform.pyramid``,
    which the inventory does not cover. Walking the package finds every module that
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

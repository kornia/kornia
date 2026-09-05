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

import importlib
import json
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
        "deprecation window has passed, update tests/api_surface.json in this PR (run "
        "tests.test_api_surface.regenerate()) and list the removal in the release notes."
    )


if __name__ == "__main__":
    regenerate()
    print(f"regenerated {INVENTORY}")

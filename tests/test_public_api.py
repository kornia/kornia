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
"""Public API surface freeze test for kornia.augmentation.

Snapshot every public symbol in kornia.augmentation.* and verify none disappear.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import sys
import types
import warnings
from pathlib import Path


def _walk() -> dict[str, list[str]]:
    """Walk kornia.augmentation recursively and collect public symbol names.

    Returns a dict mapping module name -> sorted list of public symbol names
    (i.e. names that do not start with ``_``).  Names from *any* namespace
    (classes, functions, submodules, enum members, …) are included.

    Modules that fail to import are silently skipped with a warning so that
    optional-dependency failures do not break snapshot generation.
    """
    root_name = "kornia.augmentation"

    # Make sure the root package is importable.
    try:
        root_pkg = importlib.import_module(root_name)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not import {root_name}: {exc}", stacklevel=2)
        return {}

    result: dict[str, list[str]] = {}

    def _collect_module(mod_name: str, mod: types.ModuleType) -> None:
        """Record public names for a single already-imported module."""
        public = sorted(name for name in dir(mod) if not name.startswith("_"))
        if public:
            result[mod_name] = public

    # Collect the root package itself.
    _collect_module(root_name, root_pkg)

    # Walk every sub-module / sub-package.
    root_path = getattr(root_pkg, "__path__", None)
    if root_path is None:
        return result

    for finder, mod_name, _is_pkg in pkgutil.walk_packages(
        path=root_path,
        prefix=root_name + ".",
        onerror=lambda name: warnings.warn(f"Error walking {name}", stacklevel=2),
    ):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            try:
                mod = importlib.import_module(mod_name)
            except Exception as exc:
                warnings.warn(f"Skipping {mod_name} (import failed): {exc}", stacklevel=2)
                continue
        _collect_module(mod_name, mod)

    return result


def test_no_public_symbols_removed() -> None:
    """Fail if any symbol present in the snapshot has been removed from the live package."""
    snapshot_path = Path(__file__).parent / "api_surface.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {snapshot_path}\n"
            "Generate it with:\n"
            "  python -c \"from tests.test_public_api import _walk; "
            "import json; print(json.dumps(_walk(), indent=2, sort_keys=True))\" "
            "> tests/api_surface.json"
        )

    with snapshot_path.open() as fh:
        expected: dict[str, list[str]] = json.load(fh)

    actual = _walk()

    missing: dict[str, list[str]] = {}
    for mod_name, syms in expected.items():
        live = set(actual.get(mod_name, []))
        gone = [s for s in syms if s not in live]
        if gone:
            missing[mod_name] = gone

    assert not missing, f"Public symbols disappeared:\n{json.dumps(missing, indent=2)}"

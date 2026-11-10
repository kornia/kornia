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

"""Every place ``kornia/`` unpickles a file asks for ``weights_only=True``.

A behavioural test of a direct ``torch.load`` call cannot guard it on a current torch: from
torch 2.6 ``torch.load`` defaults to ``weights_only=True`` by itself, so dropping the keyword
changes nothing there and only the torch 2.5 leg would notice. Reading the source does not
depend on the torch version, and it also covers call sites added later.

The rules:

* every ``torch.load`` / ``torch.serialization.load`` call passes ``weights_only=True`` as a
  literal keyword;
* ``torch.hub.load_state_dict_from_url`` is called only by kornia's own wrapper in
  ``kornia/core/download.py``, whose ``weights_only`` default ``TestWeightsOnly`` pins;
* neither loader is imported by bare name, which would hide a call from the first two rules.
"""

from __future__ import annotations

import ast
from pathlib import Path

import kornia

_KORNIA_ROOT = Path(kornia.__file__).resolve().parent
_WRAPPER_MODULE = "core/download.py"
_LOAD_CHAINS = {("torch", "load"), ("torch", "serialization", "load")}
_HUB_CHAIN = ("torch", "hub", "load_state_dict_from_url")
_BARE_IMPORTS = {("torch", "load"), ("torch.serialization", "load"), ("torch.hub", "load_state_dict_from_url")}


def _chain(node: ast.AST, torch_names: set[str]) -> tuple[str, ...] | None:
    """Return ``("torch", "hub", "x")`` for ``<torch alias>.hub.x``, or ``None`` for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id in torch_names:
        return ("torch", *reversed(parts))
    return None


def _scan() -> tuple[list[str], list[str], list[str]]:
    """Return ``(torch.load sites, violations, hub sites)`` over every module under ``kornia/``."""
    load_sites: list[str] = []
    violations: list[str] = []
    hub_sites: list[str] = []
    for path in sorted(_KORNIA_ROOT.rglob("*.py")):
        rel = path.relative_to(_KORNIA_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        torch_names = {"torch"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                torch_names |= {alias.asname for alias in node.names if alias.name == "torch" and alias.asname}
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                for alias in node.names:
                    if (node.module, alias.name) in _BARE_IMPORTS:
                        violations.append(f"{rel}:{node.lineno} imports {node.module}.{alias.name} by bare name")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _chain(node.func, torch_names)
            where = f"{rel}:{node.lineno}"
            if chain in _LOAD_CHAINS:
                load_sites.append(where)
                weights_only = next((kw.value for kw in node.keywords if kw.arg == "weights_only"), None)
                if not (isinstance(weights_only, ast.Constant) and weights_only.value is True):
                    violations.append(f"{where} calls {'.'.join(chain)} without weights_only=True")
            elif chain == _HUB_CHAIN:
                hub_sites.append(where)
                if rel != _WRAPPER_MODULE:
                    violations.append(f"{where} calls torch.hub.load_state_dict_from_url; use kornia's wrapper")
    return load_sites, violations, hub_sites


def test_every_unpickling_call_in_kornia_passes_weights_only() -> None:
    load_sites, violations, hub_sites = _scan()
    # The scan must see the calls it polices, or an empty result proves nothing.
    assert load_sites, "found no torch.load call under kornia/; the scan no longer matches how kornia calls it"
    assert hub_sites, f"found no torch.hub.load_state_dict_from_url call; the wrapper in {_WRAPPER_MODULE} moved"
    assert not violations, "\n".join(violations)

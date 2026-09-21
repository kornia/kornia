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

"""Refresh the map of kornia APIs that have an interactive page on the playground.

The kornia.org playground publishes ``playground/registry.json`` (operators) and
``playground/models/index.json`` (browser models). Each entry carries the fully-qualified kornia
name and the slug of its page. This script turns those into a small
``docs/source/_playground_links.json`` mapping ``<fully-qualified name> -> <page URL>`` that the
docs build reads to add a "Try in browser" badge to the matching function, module and model.

Run it whenever the playground gains or loses operators::

    # from a local checkout of kornia.github.io
    KORNIA_SITE=~/git/kornia.github.io python docs/generate_playground_links.py

    # or straight from the published site
    python docs/generate_playground_links.py

The bundled snapshot keeps the docs build offline and reproducible; refreshing it is a deliberate,
reviewable change rather than a network call at build time.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://kornia.org"
HERE = Path(__file__).resolve().parent
OUT = HERE / "source" / "_playground_links.json"

# operators whose page only records that they are not exportable have nothing to run, so no badge
RUNNABLE_STATUS = {"live", "frames"}


def _load(name: str) -> Any:
    """Read a playground JSON file from ``KORNIA_SITE`` if set, otherwise from the published site."""
    site = os.environ.get("KORNIA_SITE")
    if site:
        path = Path(site).expanduser() / "playground" / name
        return json.loads(path.read_text())
    with urllib.request.urlopen(f"{BASE_URL}/playground/{name}", timeout=60) as response:  # noqa: S310 - fixed https URL
        return json.loads(response.read())


def build_map() -> dict[str, str]:
    links: dict[str, str] = {}
    registry = _load("registry.json")
    for op in registry.get("ops", []):
        if op.get("status") not in RUNNABLE_STATUS or op.get("alias_of"):
            continue
        url = f"{BASE_URL}/playground/ops/{op['slug']}/"
        links[op["id"]] = url
        module_name = op.get("module_name")  # the nn.Module counterpart shares the page
        if module_name:
            links[module_name] = url
    models = _load("models/index.json")
    for model in models.get("models", models):
        links[model["id"]] = f"{BASE_URL}/playground/models/{model['slug']}/"
    return dict(sorted(links.items()))


def main() -> None:
    links = build_map()
    if not links:
        sys.exit("no playground links were produced; check KORNIA_SITE or the network")
    OUT.write_text(json.dumps(links, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(HERE.parent)} with {len(links)} entries")


if __name__ == "__main__":
    main()

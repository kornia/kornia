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

"""Render docs/source/community/adoption.rst from docs/source/_data/dependents.json.

The JSON is refreshed by running ``fetch_dependents.py`` by hand and committing the result; this
module turns it into the Adoption page at docs-build time, the same way ``generate_benchmarks.py``
renders the Performance page. ``load_counts()`` also feeds the numbers into ``rst_epilog``
substitutions (``|count-dependents|`` and friends) so other pages can quote them without a second copy.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "source" / "_data" / "dependents.json"
OUT = REPO / "docs" / "source" / "community" / "adoption.rst"

TOP_REPOSITORIES = 30
TOP_PACKAGES = 30


def load_data() -> dict:
    return json.loads(DATA.read_text())


def load_counts(data: dict | None = None) -> dict[str, int | str]:
    """Numbers other pages quote: exact counts, a rounded headline figure, and the snapshot date."""
    data = data or load_data()
    repositories = int(data.get("repositories", {}).get("count", 0))
    packages = int(data.get("packages", {}).get("count", 0))
    return {
        "repositories": repositories,
        "packages": packages,
        # rows GitHub's listing exposes; it stops paginating before its own total (see fetch_dependents.merge)
        "repositories_listed": int(data.get("repositories", {}).get("listed", 0)),
        "packages_listed": int(data.get("packages", {}).get("listed", 0)),
        "repositories_rounded": _round_headline(repositories),
        "generated_at": str(data.get("generated_at", "")),
    }


def _round_headline(count: int) -> str:
    """17,565 -> '17K'; 1,234 -> '1.2K'; 987 -> '987'. Never rounds up past the real figure."""
    if count >= 10_000:
        return f"{count // 1000}K"
    if count >= 1_000:
        # Truncate rather than round: 1,290 is "1.2K", and 9,999 must not become "10K".
        return f"{count // 100 / 10:.1f}K".replace(".0K", "K")
    return str(count)


def _table(items: list[dict], limit: int) -> str:
    rows = ["   * - Repository", "     - Stars", "     - Forks"]
    for item in items[:limit]:
        name = item["name"]
        rows += [
            f"   * - `{name} <https://github.com/{name}>`__",
            f"     - {item['stars']:,}",
            f"     - {item['forks']:,}",
        ]
    return "\n".join(rows)


def render_page(data: dict) -> str:
    counts = load_counts(data)
    repos = data.get("repositories", {})
    packages = data.get("packages", {})
    repo_items = repos.get("items", [])
    package_items = packages.get("items", [])
    date = counts["generated_at"]
    description = (
        f"Who uses Kornia: {counts['repositories']:,} public GitHub repositories and {counts['packages']:,} "
        "packages depend on it, according to GitHub's dependency graph."
    )

    return f"""\
.. This page is generated at build time by docs/generate_adoption.py from
   docs/source/_data/dependents.json. Edit those, not this file.

Adoption
========

.. meta::
   :description: {description}

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: kornia-cards kornia-adoption

   .. grid-item-card::
      :link: adoption-packages
      :link-type: ref
      :class-card: kornia-adoption-card

      .. rst-class:: kornia-adoption-count

      {counts["packages"]:,}

      **packages** built on Kornia

   .. grid-item-card::
      :link: adoption-repositories
      :link-type: ref
      :class-card: kornia-adoption-card

      .. rst-class:: kornia-adoption-count

      {counts["repositories"]:,}

      **repositories** depend on Kornia

**{counts["repositories"]:,} public repositories** and **{counts["packages"]:,} packages** on GitHub
declare kornia as a dependency, according to GitHub's dependency graph on {date}.

The figures come from the `Used by <{data.get("source", "https://github.com/kornia/kornia/network/dependents")}>`__
listing of the kornia repository. They count projects that name ``kornia`` in a manifest GitHub can
read -- ``requirements.txt``, ``pyproject.toml``, ``setup.py``, ``environment.yml`` and the like --
so private repositories, Docker images, notebooks and copied source are not included. A script in
the repository, ``docs/fetch_dependents.py``, walks that listing and the snapshot it takes is
committed; this page is rendered from it whenever the documentation is built.

If kornia is part of your research, please also :doc:`cite the paper </get-started/about>`.

.. _adoption-packages:

Packages built on kornia
------------------------

Libraries and tools that build on kornia and are themselves published as packages -- the
{min(TOP_PACKAGES, len(package_items))} most-starred:

.. list-table::
   :header-rows: 1
   :widths: 70 15 15
   :class: kornia-adoption-table

{_table(package_items, TOP_PACKAGES)}

.. _adoption-repositories:

Most-starred dependent repositories
-----------------------------------

Public repositories that declare kornia as a dependency -- the {min(TOP_REPOSITORIES, len(repo_items))}
most-starred:

.. list-table::
   :header-rows: 1
   :widths: 70 15 15
   :class: kornia-adoption-table

{_table(repo_items, TOP_REPOSITORIES)}

Add your project
----------------

These lists are generated from GitHub's dependency graph and ranked by stars -- nothing to submit; a
project appears once it declares kornia as a dependency and the snapshot is next refreshed. If your
project depends on kornia and you would like it featured here
or elsewhere in the documentation, write to `hello@kornia.org <mailto:hello@kornia.org>`__ with a link
and a line about what it does.
"""


def main() -> dict[str, int | str]:
    data = load_data()
    OUT.write_text(render_page(data))
    return load_counts(data)


if __name__ == "__main__":
    print(main())

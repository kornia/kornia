#!/usr/bin/env python3

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
"""Map a pull request's changed paths to the ``--onnx-conformance`` selection of its CI run (#5819).

Reads one changed path per line on stdin and prints ``all``, a comma-separated list of stable-core packages, or
nothing when no change can affect a promised operator (the conformance job is then skipped):

- ``kornia/<package>/...`` selects ``<package>`` when it is in the stable core;
- ``kornia/core``, ``kornia/utils``, ``kornia/onnx``, a module directly under ``kornia/``, the survey harness and
  registries, the test and its support, the root ``conftest.py``, this script and the dependency pins select ``all``.

Standard library only: the ``changes`` job runs it before any environment is installed.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

# The stable core of docs/source/get-started/stability.rst; the same tuple as testing/onnx_conformance.py, which
# tests/scripts/test_onnx_conformance_packages.py checks.
STABLE_CORE = (
    "geometry",
    "augmentation",
    "color",
    "filters",
    "enhance",
    "losses",
    "metrics",
    "morphology",
    "io",
    "image",
    "core",
    "feature",
)

# Code every package runs, or the machinery of the test itself.
ALL_PREFIXES = (
    "kornia/core/",
    "kornia/utils/",
    "kornia/onnx/",
    "docs/export_support/",
    "testing/",
    "tests/onnx/test_conformance.py",
    ".github/scripts/onnx_conformance_packages.py",
)
ALL_FILES = ("conftest.py", "pyproject.toml", "uv.lock", "pixi.toml", "pixi.lock")


def select(paths: Iterable[str]) -> str:
    """Return the ``--onnx-conformance`` value for a set of changed paths, ``""`` when nothing is selected."""
    packages: set[str] = set()
    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        if path.startswith(ALL_PREFIXES) or path in ALL_FILES or path.startswith("pixi."):
            return "all"
        parts = path.split("/")
        if parts[0] != "kornia":
            continue
        if len(parts) == 2:  # kornia/__init__.py, kornia/constants.py, ...: imported by every package
            return "all"
        if parts[1] in STABLE_CORE:
            packages.add(parts[1])
    return ",".join(p for p in STABLE_CORE if p in packages)


def main() -> int:
    """Print the selection for the changed paths read from stdin."""
    print(select(sys.stdin.read().splitlines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

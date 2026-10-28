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

"""Which parts of kornia the docstring-style checks enforce, widened one batch at a time.

``tests/test_docstring_style.py`` reads these values. A rollout batch converts a module and then adds it to the
lists below, so that the style it was converted to cannot regress.
"""

from __future__ import annotations

IN_SCOPE_PACKAGES: tuple[str, ...] = (
    "augmentation",
    "color",
    "config",
    "constants",
    "contrib",
    "core",
    "enhance",
    "feature",
    "filters",
    "geometry",
    "image",
    "io",
    "losses",
    "metrics",
    "models",
    "morphology",
    "onnx",
    "sensors",
    "tracking",
)
"""Top-level ``kornia`` packages and modules the rollout covers."""

EXCLUDED_PACKAGES: dict[str, str] = {
    "utils": "deprecated shims that keep their 'Deprecated: use X' first lines",
    "transpiler": "a removal stub",
}
"""Top-level packages left out of the rollout, with the reason."""

SHAPE_CHECK_MODULES: tuple[str, ...] | None = ("kornia.geometry.depth",)
"""Module prefixes whose source may not write a shape as ``:math:``; ``None`` means all of ``kornia/``."""

CATALOG_NAMESPACES: tuple[str, ...] = ("kornia.geometry.depth",)
"""Namespaces whose module docstring must name every export in ``__all__``."""

DOCSTRING_MODULES: tuple[str, ...] = ("kornia.geometry.depth",)
"""Prefixes of the defining modules whose public objects carry the converted docstring style."""

SEE_ALSO_EXCEPTIONS: dict[str, str] = {}
"""Qualified name of a converted object that needs no ``See Also:`` section, with the reason."""

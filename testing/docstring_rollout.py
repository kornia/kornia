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

"""Which parts of kornia the docstring-style checks enforce, and the exceptions they allow.

``tests/test_docstring_style.py`` reads these values. The checks cover every public object of the packages in
``IN_SCOPE_PACKAGES``, and the module catalogs that ``testing.docstring_style.catalog_namespaces`` selects there.
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
    "morphology",
    "onnx",
    "sensors",
    "tracking",
)
"""Top-level ``kornia`` packages and modules the docstring checks cover."""

EXCLUDED_PACKAGES: dict[str, str] = {
    "utils": "deprecated shims that keep their 'Deprecated: use X' first lines",
    "models": "not converted to the docstring style; its docstrings keep their earlier form",
    "transpiler": "a removal stub",
}
"""Top-level packages the docstring checks leave out, with the reason."""

CATALOG_EXCEPTIONS: dict[str, str] = {
    "kornia.augmentation.utils": (
        "internal augmentation helpers; __all__ supports implementation imports, not a public API catalog"
    ),
}
"""Namespaces intentionally omitted from public catalogs, with a reason; exports remain unchanged."""

SEE_ALSO_EXCEPTIONS: dict[str, str] = {
    "kornia.contrib.kmeans.KMeans": "the only clustering in kornia; no other object fits or assigns centres",
    "kornia.feature.defmo.DeFMO": "the only deblurring model in kornia; no other object serves a nearby task",
    "kornia.core.exceptions.BaseError": (
        "the base of every kornia check error; the kornia.core.exceptions catalog lists its subclasses"
    ),
    "kornia.constants.DType": "the only dtype-code enum in kornia; no other object names a dtype by an integer",
    "kornia.config.InstallationMode": "the only configuration enum in kornia; no other object sets a nearby policy",
    "kornia.core.utils.batched_forward": (
        "the only micro-batching helper, apart from its deprecated alias in kornia.utils"
    ),
    "kornia.core.utils.is_autocast_enabled": "the only autocast query, apart from its deprecated alias in kornia.utils",
}
"""Qualified name of a checked object that needs no ``See Also:`` section, with the reason."""

MATH_NON_SHAPES: dict[str, dict[str, str]] = {
    "docs/source/geometry.calibration.rst": {"(X, Y, Z)": "the coordinates of one 3D point"},
    "docs/source/geometry.camera.pinhole.rst": {"(X,Y,Z)": "the coordinates of one 3D point"},
    "docs/source/geometry.camera.stereo.rst": {"(X/W, Y/W, Z/W)": "the Euclidean coordinates of one 3D point"},
    "kornia/contrib/visual_prompter.py": {"(0, 0, W, H)": "a box, the whole image before the resize"},
    "kornia/feature/affine_shape.py": {"(1, 0, 1)": "the ellipse coefficients (a, b, c) of a circle"},
    "kornia/geometry/pointcloud.py": {"(X, Y, Z)": "the coordinates of one point"},
    "kornia/metrics/pose.py": {"(0, 0)": "the origin of the recall curve"},
}
"""``:math:`` tuples that read as a shape but are values, by repository path, each with what the value is.

The shape check covers every ``kornia/**/*.py`` file and every page under ``docs/source``; it skips these.
"""

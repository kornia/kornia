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
    "morphology",
    "onnx",
    "sensors",
    "tracking",
)
"""Top-level ``kornia`` packages and modules the rollout covers."""

EXCLUDED_PACKAGES: dict[str, str] = {
    "utils": "deprecated shims that keep their 'Deprecated: use X' first lines",
    "models": "deferred: left out of the rollout for now by maintainer decision (2026-11-10)",
    "transpiler": "a removal stub",
}
"""Top-level packages left out of the rollout, with the reason."""

CATALOG_EXCEPTIONS: dict[str, str] = {
    "kornia.augmentation.utils": (
        "internal augmentation helpers; __all__ supports implementation imports, not a public API catalog"
    ),
}
"""Namespaces intentionally omitted from public catalogs, with a reason; exports remain unchanged."""

CATALOG_NAMESPACES: tuple[str, ...] = (
    "kornia.augmentation",
    "kornia.augmentation.auto",
    "kornia.augmentation.auto.operations.ops",
    "kornia.augmentation.container.base",
    "kornia.augmentation.container.params",
    "kornia.color",
    "kornia.contrib",
    "kornia.contrib.augmentation_audit",
    "kornia.contrib.edge_detection",
    "kornia.contrib.face_detection",
    "kornia.contrib.object_detection",
    "kornia.contrib.super_resolution",
    "kornia.enhance",
    "kornia.feature",
    "kornia.feature.dedode.transformer",
    "kornia.filters",
    "kornia.filters.blur_pool",
    "kornia.geometry.bbox",
    "kornia.geometry.boxes",
    "kornia.geometry.calibration",
    "kornia.geometry.camera",
    "kornia.geometry.camera.stereo",
    "kornia.geometry.conversions",
    "kornia.geometry.depth",
    "kornia.geometry.epipolar",
    "kornia.geometry.grid",
    "kornia.geometry.homography",
    "kornia.geometry.keypoints",
    "kornia.geometry.liegroup",
    "kornia.geometry.linalg",
    "kornia.geometry.line",
    "kornia.geometry.plane",
    "kornia.geometry.pointcloud",
    "kornia.geometry.pose",
    "kornia.geometry.ransac",
    "kornia.geometry.solvers",
    "kornia.geometry.subpix",
    "kornia.geometry.transform.affwarp",
    "kornia.geometry.transform.crop2d",
    "kornia.geometry.transform.crop3d",
    "kornia.geometry.transform.elastic_transform",
    "kornia.geometry.transform.flips",
    "kornia.geometry.transform.homography_warper",
    "kornia.geometry.transform.image_registrator",
    "kornia.geometry.transform.imgwarp",
    "kornia.geometry.transform.pyramid",
    "kornia.geometry.transform.thin_plate_spline",
    "kornia.geometry.vector",
    "kornia.losses",
    "kornia.metrics",
    "kornia.onnx.download",
    "kornia.onnx.module",
    "kornia.onnx.sequential",
    "kornia.onnx.utils",
)
"""Namespaces whose module docstring must name every export in ``__all__``."""

DOCSTRING_MODULES: tuple[str, ...] = (
    "kornia.augmentation",
    "kornia.color",
    "kornia.contrib",
    "kornia.enhance",
    "kornia.feature.adalam",
    "kornia.feature.affine_shape",
    "kornia.feature.aliked",
    "kornia.feature.dedode",
    "kornia.feature.defmo",
    "kornia.feature.disk",
    "kornia.feature.hardnet",
    "kornia.feature.hynet",
    "kornia.feature.integrated",
    "kornia.feature.keynet",
    "kornia.feature.laf",
    "kornia.feature.lightglue",
    "kornia.feature.lightglue_onnx",
    "kornia.feature.loftr",
    "kornia.feature.matching",
    "kornia.feature.mkd",
    "kornia.feature.orientation",
    "kornia.feature.responses",
    "kornia.feature.scale_space_detector",
    "kornia.feature.sift",
    "kornia.feature.siftdesc",
    "kornia.feature.sold2.sold2",
    "kornia.feature.sold2.sold2_detector",
    "kornia.feature.sosnet",
    "kornia.feature.tfeat",
    "kornia.feature.xfeat",
    "kornia.filters",
    "kornia.geometry.bbox",
    "kornia.geometry.boxes",
    "kornia.geometry.calibration",
    "kornia.geometry.camera",
    "kornia.geometry.conversions",
    "kornia.geometry.depth",
    "kornia.geometry.epipolar",
    "kornia.geometry.grid",
    "kornia.geometry.homography",
    "kornia.geometry.keypoints",
    "kornia.geometry.liegroup",
    "kornia.geometry.linalg",
    "kornia.geometry.line",
    "kornia.geometry.plane",
    "kornia.geometry.pointcloud",
    "kornia.geometry.pose",
    "kornia.geometry.quaternion",
    "kornia.geometry.ransac",
    "kornia.geometry.solvers",
    "kornia.geometry.subpix",
    "kornia.geometry.transform",
    "kornia.geometry.vector",
    "kornia.losses",
    "kornia.metrics",
    "kornia.onnx",
)
"""Prefixes of the defining modules whose public objects carry the converted docstring style."""

SEE_ALSO_EXCEPTIONS: dict[str, str] = {
    "kornia.contrib.kmeans.KMeans": "the only clustering in kornia; no other object fits or assigns centres",
    "kornia.feature.defmo.DeFMO": "the only deblurring model in kornia; no other object serves a nearby task",
}
"""Qualified name of a converted object that needs no ``See Also:`` section, with the reason."""

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

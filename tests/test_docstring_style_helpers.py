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

"""Tests for the shared docstring-style helpers in ``testing/docstring_style.py``."""

from functools import cache
from pathlib import Path

import kornia
from kornia.geometry.depth import DepthWarper, depth_to_3d, depth_to_3d_v2

from testing.docstring_style import (
    catalog_missing,
    catalog_required,
    covered,
    docstring_roles,
    is_shape_tuple,
    math_shape_roles,
    own_docstring,
    public_objects,
    reference_resolves,
    resolve_reference,
    see_also_entries,
    sphinx_registry,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "source"


@cache
def _registry():
    return sphinx_registry(DOCS)


def test_shape_predicate():
    for s in ["(B, 3, H, W)", "(*, H, W)", "(B,)", "(B, N, 2)", "(B, H/2, W/2)", "(..., 3)", "(H_out, W_out)"]:
        assert is_shape_tuple(s), s
    for s in [
        "(x, y)",
        "(u, v)",
        "(B, C_{in}, H, W)",
        r"(H \times W)",
        "()",
        "(B, C, h, w)",
        "(-0.436, 0.436)",
        "(1, kH, kW)",
    ]:
        assert not is_shape_tuple(s), s


def test_math_shape_roles():
    assert math_shape_roles("a :math:`(B, 3, H, W)` b :math:`(x, y)` c ``(B, N)``") == ["(B, 3, H, W)"]


def test_docstring_roles():
    doc = "Use :func:`~kornia.a.b` or :py:class:`C`, :meth:`forward`; not :cite:`X` or :math:`y`."
    assert docstring_roles(doc) == [("func", "kornia.a.b"), ("class", "C"), ("meth", "forward")]


def test_see_also_entries():
    doc = (
        "Do a thing.\n\nArgs:\n    x: input.\n\nSee Also:\n"
        "    - :func:`depth_to_3d_v2`: the same points channels last.\n"
        "    - :func:`depth_to_3d_v2`, :func:`~kornia.geometry.camera.perspective.project_points`: the two halves\n"
        "      of the warp.\n\nExample:\n    >>> 1\n"
    )
    assert see_also_entries(doc) == [
        ("depth_to_3d_v2", "the same points channels last."),
        ("depth_to_3d_v2", "the two halves of the warp."),
        ("kornia.geometry.camera.perspective.project_points", "the two halves of the warp."),
    ]
    assert see_also_entries(
        "Do.\n\nSee Also:\n    - :func:`depth_to_3d_v2`: like :func:`depth_to_3d` but channels last.\n"
    ) == [("depth_to_3d_v2", "like :func:`depth_to_3d` but channels last.")]
    assert see_also_entries("Do.\n\nSee Also:\n    Prose only.\n") == []
    assert see_also_entries("Do.\n\nSee Also:\n    - :func:`a`: b.\n.. image:: foo.png\n") == [("a", "b.")]
    assert see_also_entries("Do.\n\nArgs:\n    x: y.\n") is None
    assert see_also_entries("Do.\n\nSee Also:\n    - :func:`a`\n") == [("a", "")]


def test_registry_uses_documented_paths():
    reg = _registry()
    assert "kornia.geometry.camera.perspective.unproject_points" in reg
    assert "kornia.geometry.camera.unproject_points" not in reg
    assert "kornia.geometry.homography.find_homography_dlt" in reg  # automodule :members:
    assert "kornia.geometry.depth.DepthWarper.compute_projection_matrix" in reg  # autoclass :members:
    assert "kornia.color.yuv_to_rgb" in reg  # autosummary


def test_reference_resolution():
    reg = _registry()
    assert reference_resolves(depth_to_3d, "depth_to_3d_v2", reg)
    assert reference_resolves(depth_to_3d, "kornia.geometry.camera.perspective.unproject_points", reg)
    assert not reference_resolves(depth_to_3d, "kornia.geometry.camera.unproject_points", reg)
    assert not reference_resolves(depth_to_3d, "kornia.geometry.depth.no_such_function", reg)
    assert reference_resolves(DepthWarper, "compute_projection_matrix", reg)


def test_resolve_reference_returns_the_target_object():
    reg = _registry()
    assert resolve_reference(depth_to_3d, "depth_to_3d_v2", reg) is depth_to_3d_v2
    assert resolve_reference(depth_to_3d, "depth_to_3d", reg) is depth_to_3d
    assert resolve_reference(depth_to_3d, "no_such_function", reg) is None
    assert resolve_reference(depth_to_3d, "kornia.geometry.camera.unproject_points", reg) is None


def test_covered_prefix():
    assert covered("kornia.color.rgb", ["kornia.color"])
    assert not covered("kornia.colorx", ["kornia.color"])


def test_catalog_missing():
    assert catalog_missing("use depth_to_3d_v2 here", ["depth_to_3d", "depth_to_3d_v2"]) == ["depth_to_3d"]
    assert catalog_missing("- ``depth_to_3d``: points.", ["depth_to_3d"]) == []


def test_catalog_required():
    assert catalog_required("kornia.geometry.depth", [])
    # kornia.enhance re-exports everything kornia.enhance.normalize declares, as the same objects.
    assert not catalog_required("kornia.enhance.normalize", ["kornia.enhance"])
    assert catalog_required("kornia.enhance.normalize", [])


def test_own_docstring_and_public_objects():
    class Base:
        """Base doc."""

    class Child(Base):
        pass

    assert own_docstring(Base) == "Base doc."
    assert own_docstring(Child) is None
    assert id(depth_to_3d) in public_objects()
    assert public_objects()[id(depth_to_3d)][1] == "kornia.geometry.depth.depth_to_3d"


def test_function_sharing_a_submodule_name_is_the_function():
    reg = _registry()
    assert reg["kornia.filters.sobel"] is kornia.filters.sobel
    assert resolve_reference(kornia.filters.sobel, "sobel", reg) is kornia.filters.sobel


def test_explicit_members_list_registers_only_the_listed_names():
    reg = _registry()
    assert "kornia.feature.SOLD2.match" not in reg
    assert "kornia.feature.SOLD2.forward" in reg


def test_automodule_members_register_class_methods():
    reg = _registry()
    assert "kornia.geometry.boxes.Boxes.to_tensor" in reg


def test_nested_automethod_registers_under_its_class():
    assert "kornia.geometry.camera.stereo.StereoCamera.__init__" in _registry()


def test_own_docstring_ignores_whitespace_only():
    def blank():
        """ """

    assert own_docstring(blank) is None

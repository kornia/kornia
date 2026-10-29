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

import functools
import inspect
from functools import cache
from pathlib import Path

import pytest

import kornia
from kornia.augmentation.auto.operations.ops import AutoContrast
from kornia.geometry.boxes import Boxes, Boxes3D
from kornia.geometry.depth import DepthWarper, depth_to_3d, depth_to_3d_v2

from testing import docstring_style
from testing.docstring_style import (
    catalog_missing,
    catalog_required,
    catalog_unimportable,
    covered,
    docstring_roles,
    forwards_to,
    intersphinx_names,
    is_deprecated_wrapper,
    is_documented,
    is_private_target,
    is_shape_tuple,
    math_shape_roles,
    own_docstring,
    public_objects,
    reference_resolves,
    resolve_reference,
    see_also_entries,
    sphinx_registry,
    status_target,
    undocumented_public_objects,
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
    doc = "Read :attr:`data`, :py:exc:`ValueError`, :data:`sys.stdout`, :const:`LIMIT`, :obj:`x`; not :mod:`torch`."
    assert docstring_roles(doc) == [
        ("attr", "data"),
        ("exc", "ValueError"),
        ("data", "sys.stdout"),
        ("const", "LIMIT"),
        ("obj", "x"),
    ]


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
    # A bullet without a leading role renders without a link; it is an entry with an empty target, not skipped.
    assert see_also_entries("Do.\n\nSee Also:\n    - :func:`a`: b.\n    - ``c``: d.\n    - kornia.e: f.\n") == [
        ("a", "b."),
        ("", "``c``: d."),
        ("", "kornia.e: f."),
    ]
    # Any Python-domain object role links, so it leads an entry; :mod: does not name an object.
    doc = "Do.\n\nSee Also:\n    - :attr:`~kornia.x.Y.z`: an attribute.\n    - :exc:`E`: raised.\n    - :mod:`m`: no.\n"
    assert see_also_entries(doc) == [("kornia.x.Y.z", "an attribute."), ("E", "raised."), ("", ":mod:`m`: no.")]


def test_registry_uses_documented_paths():
    reg = _registry()
    assert "kornia.geometry.camera.perspective.unproject_points" in reg
    assert "kornia.geometry.camera.unproject_points" not in reg
    assert "kornia.geometry.homography.find_homography_dlt" in reg  # automodule :members:
    assert "kornia.geometry.depth.DepthWarper.compute_projection_matrix" in reg  # autoclass :members:


def test_reference_resolution():
    reg = _registry()
    assert reference_resolves(depth_to_3d, "depth_to_3d_v2", reg)
    assert reference_resolves(depth_to_3d, "kornia.geometry.camera.perspective.unproject_points", reg)
    assert not reference_resolves(depth_to_3d, "kornia.geometry.camera.unproject_points", reg)
    assert not reference_resolves(depth_to_3d, "kornia.geometry.depth.no_such_function", reg)
    assert reference_resolves(DepthWarper, "compute_projection_matrix", reg)
    # Sphinx drops a trailing ``()`` and tries a dotted target under the page's module too.
    assert resolve_reference(depth_to_3d, "depth_to_3d_v2()", reg) is depth_to_3d_v2
    assert resolve_reference(Boxes3D, "Boxes.to_tensor", reg) is Boxes.to_tensor
    assert not reference_resolves(depth_to_3d, "kornia.geometry.camera.unproject_points()", reg)
    assert not reference_resolves(Boxes3D, "geometry.boxes.Boxes.to_tensor", reg)


def test_resolve_reference_returns_the_target_object():
    reg = _registry()
    assert resolve_reference(depth_to_3d, "depth_to_3d_v2", reg) is depth_to_3d_v2
    assert resolve_reference(depth_to_3d, "depth_to_3d", reg) is depth_to_3d
    assert resolve_reference(depth_to_3d, "no_such_function", reg) is None
    assert resolve_reference(depth_to_3d, "kornia.geometry.camera.unproject_points", reg) is None


def test_intersphinx_names():
    names = intersphinx_names(DOCS)
    assert names["torch.nn.functional.grid_sample"] == {"function"}
    assert names["torch.Tensor"] == {"class"}
    assert names["ValueError"] == {"exception"}
    assert names["random"] == {"module"}
    assert "torch.nn.functional.no_such_function" not in names
    reg = _registry()
    assert resolve_reference(depth_to_3d, "torch.nn.functional.grid_sample", reg) is None
    assert reference_resolves(depth_to_3d, "torch.nn.functional.grid_sample", reg, names)
    assert not reference_resolves(depth_to_3d, "torch.nn.functional.no_such_function", reg, names)


def test_intersphinx_links_a_role_only_to_its_object_types():
    """Intersphinx matches the role to the object type, unlike the local lookup (Sphinx ``objtypes_for_role``)."""
    names, reg = intersphinx_names(DOCS), _registry()

    cases = [
        ("func", "torch.svd", True),
        ("obj", "torch.svd", True),
        ("meth", "torch.svd", False),
        ("attr", "torch.svd", False),
        ("meth", "torch.Tensor.add", True),
        ("func", "torch.Tensor.add", False),
        # Inventories from before Sphinx 2.1 store properties as methods, so intersphinx lets :attr: find a method.
        ("attr", "torch.Tensor.add", True),
        ("attr", "torch.Tensor.shape", True),
        ("meth", "torch.Tensor.shape", False),
        ("exc", "ValueError", True),
        ("class", "ValueError", True),
        ("func", "ValueError", False),
        ("class", "torch.Tensor", True),
        ("exc", "torch.Tensor", True),
        ("data", "sys.stdout", True),
        ("attr", "sys.stdout", False),
        # :const: is in no object type's roles, so intersphinx never links it.
        ("const", "sys.stdout", False),
        ("const", "math.pi", False),
        ("obj", "random", True),
        ("func", "random", False),
    ]
    for role, target, expected in cases:
        assert reference_resolves(depth_to_3d, target, reg, names, role) is expected, (role, target)


def test_only_func_and_meth_drop_parentheses():
    names, reg = intersphinx_names(DOCS), _registry()
    assert resolve_reference(depth_to_3d, "depth_to_3d_v2()", reg, "func") is depth_to_3d_v2
    assert resolve_reference(DepthWarper, "compute_projection_matrix()", reg, "meth") is not None
    assert resolve_reference(depth_to_3d, "DepthWarper()", reg, "class") is None
    assert resolve_reference(depth_to_3d, "DepthWarper", reg, "class") is DepthWarper
    assert reference_resolves(depth_to_3d, "torch.svd()", reg, names, "func")
    assert not reference_resolves(depth_to_3d, "torch.Tensor()", reg, names, "class")


def test_short_names_need_a_documented_object():
    reg = _registry()
    assert is_documented(depth_to_3d, reg)
    assert not is_documented(AutoContrast, reg)
    # With no page, and so no scope, only a full name resolves from its docstring.
    assert resolve_reference(AutoContrast, "kornia.geometry.depth.depth_to_3d", reg) is depth_to_3d
    assert resolve_reference(AutoContrast, "depth_to_3d", reg) is None


def _write_page(root: Path, relative: str, text: str) -> None:
    page = root / relative
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(text, encoding="utf-8")


def test_autosummary_registers_only_with_toctree(tmp_path):
    _write_page(tmp_path, "links.rst", ".. currentmodule:: kornia.color\n\n.. autosummary::\n\n   yuv_to_rgb\n")
    assert sphinx_registry(tmp_path) == {}
    _write_page(
        tmp_path,
        "links.rst",
        ".. currentmodule:: kornia.color\n\n.. autosummary::\n   :toctree: api\n\n   yuv_to_rgb\n",
    )
    assert sphinx_registry(tmp_path) == {"kornia.color.yuv_to_rgb": kornia.color.yuv_to_rgb}


def test_registry_reads_pages_in_subdirectories(tmp_path):
    _write_page(tmp_path, "models/page.rst", ".. autofunction:: kornia.color.yuv_to_rgb\n")
    _write_page(tmp_path, "_build/page.rst", ".. autofunction:: kornia.color.rgb_to_yuv\n")
    _write_page(tmp_path, "_templates/page.rst", ".. autofunction:: kornia.color.rgb_to_hsv\n")
    assert sphinx_registry(tmp_path) == {"kornia.color.yuv_to_rgb": kornia.color.yuv_to_rgb}


def test_members_follow_autodoc_docstring_rule(tmp_path):
    """``:members:`` documents an inherited docstring; a method with none only under ``:undoc-members:``."""
    page = ".. currentmodule:: kornia.augmentation\n\n.. autoclass:: {}\n   :members:\n{}"
    intensity = "kornia.augmentation.IntensityAugmentationBase2D"
    mix = "kornia.augmentation.MixAugmentationBaseV2"
    assert own_docstring(kornia.augmentation.IntensityAugmentationBase2D.apply_transform_mask) is None
    _write_page(tmp_path, "page.rst", page.format("IntensityAugmentationBase2D", ""))
    assert f"{intensity}.apply_transform_mask" in sphinx_registry(tmp_path)
    assert own_docstring(kornia.augmentation.MixAugmentationBaseV2.apply_non_transform) is None
    _write_page(tmp_path, "page.rst", page.format("MixAugmentationBaseV2", ""))
    assert f"{mix}.apply_non_transform" not in sphinx_registry(tmp_path)
    _write_page(tmp_path, "page.rst", page.format("MixAugmentationBaseV2", "   :undoc-members:\n"))
    assert f"{mix}.apply_non_transform" in sphinx_registry(tmp_path)


def test_registry_registers_documented_data_attributes():
    """Enum members and dataclass fields under ``:undoc-members:`` link, even when the value is ``None``."""
    reg = _registry()
    assert kornia.models.sam.SamConfig.checkpoint is None
    for target in ("kornia.color.CFA.BG", "kornia.image.ImageSize.height", "kornia.models.sam.SamConfig.checkpoint"):
        assert resolve_reference(depth_to_3d, target, reg) is not None, target
        assert reference_resolves(depth_to_3d, target, reg, role="attr"), target


_ATTRIBUTE_SOURCES = {
    "docstyle_attrs.py": '''
import enum
from dataclasses import dataclass
from typing import NamedTuple


class Commented:
    """A class."""

    #: before
    before = 1
    after = 2  #: after
    annotated: int
    """string after"""
    plain = 3
    hashed = 4  # an ordinary comment

    #: two lines
    #: of comment
    two = 5

    #: separated by a blank line

    separated = 6
    multi = (
        1,
        2,
    )  #: after a multi-line value
    try:
        in_try = 7  #: in try
    except Exception:
        in_except = 8  #: in except
    _private = 9  #: private

    def __init__(self):
        self.instance = 10  #: instance
        self.undocumented = 11
        local = 12  #: a local

        def nested():
            self.nested = 13  #: in a nested function

        self.after_nested = 15  #: after a nested function

    def other(self):
        self.elsewhere = 14  #: in another method


class Undoc:
    """Same kinds, under undoc-members."""

    plain = 1
    annotated_only: int
    hashed = 2  # ordinary

    def __init__(self):
        self.instance = 3

    @property
    def prop(self):
        return 1


class Point(NamedTuple):
    """A point."""

    x: float
    y: float


class Color(enum.Enum):
    """Colors."""

    RED = 1
    GREEN = 2  #: green


@dataclass
class Config:
    """A config."""

    size: int = 1
    name: str | None = None
''',
    "docstyle_module.py": (
        '"""Data with __all__."""\n\n#: documented\nLIMIT = 3\nOTHER = 4\n__all__ = ["LIMIT", "OTHER"]\n'
    ),
    "docstyle_noall.py": (
        '"""Data without __all__."""\n\nfrom math import pi  #: imported here\n\n#: documented\nLIMIT = 3\nOTHER = 4\n'
    ),
    "docstyle_undoc.py": (
        '"""Data with __all__, under undoc-members."""\n\n__all__ = ["BARE", "PLAIN", "COMMENTED"]\nBARE: int\n'
        "PLAIN = 1\n#: doc\nCOMMENTED = 2\n"
    ),
}

_ATTRIBUTE_PAGE = """
.. autoclass:: docstyle_attrs.Commented
   :members:

.. autoclass:: docstyle_attrs.Undoc
   :members:
   :undoc-members:

.. autoclass:: docstyle_attrs.Point
   :members:

.. autoclass:: docstyle_attrs.Color
   :members:

.. autoclass:: docstyle_attrs.Config
   :members:
   :undoc-members:

.. automodule:: docstyle_module
   :members:

.. automodule:: docstyle_noall
   :members:
   :undoc-members:

.. automodule:: docstyle_undoc
   :members:
   :undoc-members:
"""


def test_registry_matches_sphinx_on_documented_attributes(tmp_path, monkeypatch):
    """The registry holds the names a Sphinx 8.2.3 build of the same page puts into ``objects.inv``.

    The expected set is that build's inventory, modules left out. To regenerate it, write these sources and the page
    (as ``index.rst``, with a ``conf.py`` that adds ``sphinx.ext.autodoc`` and puts the sources on ``sys.path``), run
    ``uv run --no-project --with sphinx==8.2.3 python -m sphinx -b html src out``, then
    ``python -m sphinx.ext.intersphinx out/objects.inv``.
    """
    for name, source in _ATTRIBUTE_SOURCES.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_page(tmp_path / "docs", "index.rst", _ATTRIBUTE_PAGE)
    registered = {name for name, obj in sphinx_registry(tmp_path / "docs").items() if not inspect.ismodule(obj)}
    a = "docstyle_attrs"
    assert registered == {
        *(f"{a}.{name}" for name in ("Commented", "Undoc", "Point", "Color", "Config")),
        *(f"{a}.Commented.{name}" for name in ("before", "after", "annotated", "two", "multi", "in_try")),
        f"{a}.Commented.instance",
        f"{a}.Commented.after_nested",
        *(f"{a}.Undoc.{name}" for name in ("plain", "annotated_only", "hashed", "prop")),
        f"{a}.Point.x",
        f"{a}.Point.y",
        f"{a}.Color.GREEN",
        f"{a}.Config.size",
        f"{a}.Config.name",
        "docstyle_module.LIMIT",
        "docstyle_noall.LIMIT",
        "docstyle_undoc.BARE",
        "docstyle_undoc.COMMENTED",
    }


def test_covered_prefix():
    assert covered("kornia.color.rgb", ["kornia.color"])
    assert not covered("kornia.colorx", ["kornia.color"])


def test_catalog_missing():
    depth, enhance = kornia.geometry.depth, kornia.enhance
    names = ["depth_to_3d", "depth_to_3d_v2"]
    assert catalog_missing("use :func:`depth_to_3d_v2` here", names, depth) == ["depth_to_3d"]
    assert catalog_missing("use ``depth_to_3d_v2()`` here", names, depth) == ["depth_to_3d"]
    assert catalog_missing("- ``depth_to_3d``: points.", ["depth_to_3d"], depth) == []
    assert catalog_missing("- ``depth_to_3d()``: points.", ["depth_to_3d"], depth) == []
    assert catalog_missing("- :func:`~kornia.geometry.depth.depth_to_3d`: points.", ["depth_to_3d"], depth) == []
    assert catalog_missing("- :func:`kornia.geometry.depth.depth_to_3d`: points.", ["depth_to_3d"], depth) == []
    # A plain word is not a mention, and neither is a longer name that ends with it.
    assert catalog_missing("Use the morphological opening and closing.", ["opening", "closing"], kornia.morphology) == [
        "opening",
        "closing",
    ]
    assert catalog_missing("use depth_to_3d here", ["depth_to_3d"], depth) == ["depth_to_3d"]
    assert catalog_missing(":func:`my_depth_to_3d`", ["depth_to_3d"], depth) == ["depth_to_3d"]
    # Single backticks are the default role, which renders as italics, and :math: is no reference: neither counts.
    assert catalog_missing("use `depth_to_3d` here", ["depth_to_3d"], depth) == ["depth_to_3d"]
    assert catalog_missing("the :math:`depth_to_3d` term", ["depth_to_3d"], depth) == ["depth_to_3d"]
    assert catalog_missing("use :py:func:`depth_to_3d` here", ["depth_to_3d"], depth) == []
    # A qualified target counts only as the exported object: torch's normalize is not kornia.enhance.normalize, which
    # is reached through the package or its defining module even though a submodule shares its name.
    for other in (":func:`torch.nn.functional.normalize`", "``torch.nn.functional.normalize``"):
        assert catalog_missing(f"for L2, use {other}", ["normalize"], enhance) == ["normalize"]
    for same in (
        ":func:`~kornia.enhance.normalize`",
        "``kornia.enhance.normalize``",
        ":func:`kornia.enhance.normalize.normalize`",
    ):
        assert catalog_missing(f"use {same}", ["normalize"], enhance) == []


def test_catalog_unimportable():
    depth = kornia.geometry.depth
    doc = (
        ":func:`depth_to_3d`, :func:`depth_to_3d()`, :func:`~kornia.geometry.depth.depth_to_3d_v2()` and "
        ":class:`kornia.geometry.depth.DepthWarper`"
    )
    assert catalog_unimportable(depth, doc) == []
    assert catalog_unimportable(depth, ":func:`depth_to_4d()` and :func:`kornia.geometry.depth.nope`") == [
        "depth_to_4d()",
        "kornia.geometry.depth.nope",
    ]


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


def test_special_members_register_dunder_methods(tmp_path):
    """``:special-members:`` alone selects every documented ``__name__`` method, a list only the listed ones."""
    reg = _registry()
    assert reg["kornia.geometry.quaternion.Quaternion.__init__"] is kornia.geometry.quaternion.Quaternion.__init__
    assert "kornia.geometry.quaternion.Quaternion.__add__" in reg
    # No docstring, and none to inherit: a one-page Sphinx 8.2.3 build leaves it out too.
    assert "kornia.geometry.quaternion.Quaternion.__mul__" not in reg
    assert "kornia.geometry.pose.NamedPose.__mul__" in reg  # :special-members: __init__, __mul__
    assert "kornia.geometry.pose.NamedPose.__repr__" not in reg
    quaternion = "kornia.geometry.quaternion.Quaternion"
    page = ".. currentmodule:: kornia.geometry.quaternion\n\n.. autoclass:: Quaternion\n{}"
    _write_page(tmp_path, "page.rst", page.format("   :members:\n"))
    assert f"{quaternion}.__init__" not in sphinx_registry(tmp_path)
    # Without :members:, all special members document nothing; a list documents the listed names.
    _write_page(tmp_path, "page.rst", page.format("   :special-members:\n"))
    assert sphinx_registry(tmp_path) == {quaternion: kornia.geometry.quaternion.Quaternion}
    _write_page(tmp_path, "page.rst", page.format("   :special-members: __add__\n"))
    assert set(sphinx_registry(tmp_path)) == {quaternion, f"{quaternion}.__add__"}
    _write_page(tmp_path, "page.rst", page.format("   :members: w\n   :special-members: __add__\n"))
    assert set(sphinx_registry(tmp_path)) == {quaternion, f"{quaternion}.w", f"{quaternion}.__add__"}
    # A module's special members apply to its classes.
    _write_page(
        tmp_path, "page.rst", ".. automodule:: kornia.geometry.quaternion\n   :members:\n   :special-members:\n"
    )
    assert f"{quaternion}.__add__" in sphinx_registry(tmp_path)


def test_nested_automethod_registers_under_its_class():
    assert "kornia.geometry.camera.stereo.StereoCamera.__init__" in _registry()


def test_registry_rejects_options_it_does_not_model(tmp_path):
    """An option that changes what autodoc registers fails loudly instead of skewing the registry."""
    for option in ("no-index", "exclude-members", "inherited-members", "private-members"):
        _write_page(
            tmp_path, "page.rst", f".. autoclass:: kornia.geometry.depth.DepthWarper\n   :members:\n   :{option}:\n"
        )
        with pytest.raises(ValueError, match=option):
            sphinx_registry(tmp_path)


def test_py_prefixed_currentmodule_sets_the_scope(tmp_path):
    _write_page(tmp_path, "page.rst", ".. py:currentmodule:: kornia.color\n\n.. autofunction:: yuv_to_rgb\n")
    assert sphinx_registry(tmp_path) == {"kornia.color.yuv_to_rgb": kornia.color.yuv_to_rgb}


def test_private_target():
    assert is_private_target("_helper")
    assert is_private_target("kornia.geometry.depth._depth_grid_dtype")
    assert is_private_target("kornia.x.Class.__mangled")
    # A special method that ``:special-members:`` documents is public.
    assert not is_private_target("kornia.geometry.pose.NamedPose.__mul__")
    assert not is_private_target("depth_to_3d")


def test_own_docstring_ignores_whitespace_only():
    def blank():
        """ """

    assert own_docstring(blank) is None


@pytest.mark.parametrize(
    ("doc", "target"),
    [
        ("Alias of :func:`axis_angle_to_rotation_matrix`.", "axis_angle_to_rotation_matrix"),
        ("Alias of :class:`kornia.geometry.depth.DepthWarper`.", "kornia.geometry.depth.DepthWarper"),
        (
            "Alias of :func:`~kornia.geometry.conversions.rotation_matrix_to_axis_angle`.",
            "kornia.geometry.conversions.rotation_matrix_to_axis_angle",
        ),
        ("Alias of :py:func:`foo`.\n\nMore text.", "foo"),
        (
            "Deprecated: use :func:`~kornia.geometry.conversions.axis_angle_to_quaternion`.",
            "kornia.geometry.conversions.axis_angle_to_quaternion",
        ),
        ("Deprecated: use :class:`kornia.geometry.boxes.Boxes`.", "kornia.geometry.boxes.Boxes"),
        ("Deprecated: use :py:meth:`Foo.bar`.\n\nIt warns on every call.", "Foo.bar"),
    ],
)
def test_status_target_accepts_a_status_title(doc, target):
    assert status_target(doc) == target


def test_is_deprecated_wrapper_recognises_the_decorator():
    from kornia.core._compat import deprecated

    def plain():
        return 1

    class Plain:
        def __init__(self) -> None:
            pass

    wrapped = deprecated(replace_with="plain", version="0.1")(plain)
    wrapped_class = deprecated(replace_with="Plain", version="0.1")(type("Old", (Plain,), {}))
    assert is_deprecated_wrapper(wrapped)
    assert is_deprecated_wrapper(wrapped_class)
    assert not is_deprecated_wrapper(plain)
    assert not is_deprecated_wrapper(Plain)


@pytest.mark.parametrize(
    "doc",
    [
        "Alias of :func:`foo`",
        "Alias of :func:`foo` with a twist.",
        "Alias of :func:`foo` and :func:`bar`.",
        "Alias for :func:`foo`.",
        "Return the thing.\nAlias of :func:`foo`.",
        "Alias of foo.",
        "",
        "Deprecated, use :func:`foo`.",
        "Deprecated: use foo.",
        "Deprecated: use :func:`foo`",
        "Deprecated: use :func:`foo` instead.",
        "Deprecated: use :func:`foo` or :func:`bar`.",
        "Deprecated: Use :func:`foo`.",
        "Deprecated since 0.7.0: use :func:`foo`.",
        "Return the thing.\nDeprecated: use :func:`foo`.",
        "Deprecated: use :math:`foo`.",
    ],
)
def test_status_target_rejects_other_titles(doc):
    assert status_target(doc) is None


def test_undocumented_public_objects_complements_public_objects():
    documented, undocumented = public_objects(), undocumented_public_objects()
    assert not documented.keys() & undocumented.keys()
    assert all(own_docstring(obj) is None for obj, _ in undocumented.values())


def test_public_and_undocumented_objects_split_the_export_walk(monkeypatch):
    # kornia may export nothing undocumented, so the split is checked on a synthetic walk. ``__wrapped__`` calls the
    # functions past their own ``functools.cache``, which the patched walk would otherwise never reach.
    def undocumented():
        pass

    def documented():
        """Do the thing."""

    walk = ((undocumented, "m.undocumented"), (documented, "m.documented"))
    monkeypatch.setattr(docstring_style, "_exported_objects", lambda: walk)
    assert [qualified for _, qualified in undocumented_public_objects.__wrapped__().values()] == ["m.undocumented"]
    assert [qualified for _, qualified in public_objects.__wrapped__().values()] == ["m.documented"]


def _alias_target(a, b=1, *args, c=2, **kwargs):
    return a


def _forwarding_alias(a, b=1, *args, c=2, **kwargs):
    """Alias of :func:`_alias_target`."""
    return _alias_target(a, b, *args, c=c, **kwargs)


def _keyword_alias(a, b=1):
    """Alias of :func:`_alias_target`."""
    return _alias_target(a, b=b)


class _AliasTargetClass:
    pass


class _AliasClass(_AliasTargetClass):
    """Alias of :class:`_AliasTargetClass`."""


class _SubclassWithBody(_AliasTargetClass):
    """Alias of :class:`_AliasTargetClass`."""

    def extra(self):
        return 1


class _TwoBases(_AliasTargetClass, Exception):
    """Alias of :class:`_AliasTargetClass`."""


def _swapped(a, b):
    return _alias_target(b, a)


def _constant_argument(a):
    return _alias_target(a, 2)


def _extra_statement(a):
    a = a + 1
    return _alias_target(a)


def _dropped_argument(a, b=1):
    return _alias_target(a)


def _renamed_keyword(a, b=1):
    return _alias_target(a, c=b)


def _other_target(a):
    return _keyword_alias(a)


@functools.lru_cache
def _decorated(a):
    return _alias_target(a)


@pytest.mark.parametrize(
    ("obj", "target"),
    [(_forwarding_alias, _alias_target), (_keyword_alias, _alias_target), (_AliasClass, _AliasTargetClass)],
)
def test_forwards_to_accepts_a_forwarding_alias(obj, target):
    assert forwards_to(obj, target)


def test_forwards_to_resolves_a_target_from_the_closure():
    def local_target(x):
        return x

    def local_alias(x):
        """Alias of the local target."""
        return local_target(x)

    assert forwards_to(local_alias, local_target)


@pytest.mark.parametrize(
    ("obj", "target"),
    [
        (_swapped, _alias_target),
        (_constant_argument, _alias_target),
        (_extra_statement, _alias_target),
        (_dropped_argument, _alias_target),
        (_renamed_keyword, _alias_target),
        (_other_target, _alias_target),
        (_decorated, _alias_target),
        (_SubclassWithBody, _AliasTargetClass),
        (_TwoBases, _AliasTargetClass),
        (_AliasClass, object),
        (len, _alias_target),
    ],
)
def test_forwards_to_rejects_anything_else(obj, target):
    assert not forwards_to(obj, target)


def test_forwards_to_rejects_a_live_function_titled_alias_of_another():
    # The mutant the status-title check used to accept: give a live converted function the title
    # "Alias of :func:`~kornia.geometry.conversions.rotation_matrix_to_quaternion`." and it skipped its See Also,
    # although it computes something else entirely.
    from kornia.geometry.conversions import axis_angle_to_quaternion, rotation_matrix_to_quaternion

    title = "Alias of :func:`~kornia.geometry.conversions.rotation_matrix_to_quaternion`."
    assert status_target(title) == "kornia.geometry.conversions.rotation_matrix_to_quaternion"
    assert not forwards_to(axis_angle_to_quaternion, rotation_matrix_to_quaternion)

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

# TEST OFFICIAL SUPPORT
import importlib.util
import sys
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, NamedTuple, Optional, Tuple, Union

import pytest
import torch

from kornia.core.exceptions import TypeCheckError
from kornia.core.utils import dataclass_to_dict, dict_to_dataclass
from kornia.feature.sold2.structures import DetectorCfg, LineDetectorCfg


class Point(NamedTuple):
    x: int
    y: int


class Pair(NamedTuple):
    a: Any
    b: Any


@dataclass
class Inner:
    v: int


@dataclass
class Outer:
    inner: Inner
    points: list = field(default_factory=list)


def test_namedtuple_of_scalars():
    assert dataclass_to_dict(Point(1, 2)) == Point(1, 2)


def test_dataclass_field_holding_namedtuple_does_not_raise():
    got = dataclass_to_dict(Outer(Inner(3), [Point(4, 5)]))
    assert got == {"inner": {"v": 3}, "points": [Point(4, 5)]}


def test_plain_nested_dataclass_still_converts():
    assert dataclass_to_dict(Outer(Inner(7), [])) == {"inner": {"v": 7}, "points": []}


def test_tuple_and_list_paths_are_unchanged():
    assert dataclass_to_dict((Point(1, 2),)) == (Point(1, 2),)
    assert dataclass_to_dict([Inner(1)]) == [{"v": 1}]


def test_scalar_passthrough():
    assert dataclass_to_dict(5) == 5


def test_namedtuple_items_are_converted():
    # a namedtuple reached outside `asdict` (top level, in a list or a dict) still has its items converted
    assert dataclass_to_dict(Pair(Inner(1), [Inner(2)])) == Pair({"v": 1}, [{"v": 2}])
    assert dataclass_to_dict({"k": [Pair(Inner(3), 4)]}) == {"k": [Pair({"v": 3}, 4)]}


def test_values_reached_through_a_dataclass_are_deep_copied():
    t = torch.zeros(2)
    got = dataclass_to_dict(Outer(Inner(1), [t]))
    assert got["points"][0] is not t
    assert got["points"][0].data_ptr() != t.data_ptr()
    # outside any dataclass the containers are rebuilt and their items returned as they are
    assert dataclass_to_dict({"k": [t]})["k"][0] is t


@dataclass
class Holder:
    inner: Inner = field(default_factory=lambda: Inner(0))
    maybe: Optional[Inner] = None
    either: Union[Inner, str, None] = None
    pep604: Inner | None = None
    many: List[Inner] = field(default_factory=list)
    builtin_many: list[Inner] = field(default_factory=list)
    nested: List[List[Inner]] = field(default_factory=list)
    maybe_many: Optional[List[Inner]] = None
    maybe_variadic: Optional[Tuple[Inner, ...]] = None
    text_or_many: Union[str, List[Inner]] = ""
    variadic: Tuple[Inner, ...] = ()
    fixed: Tuple[Inner, int] = (Inner(0), 0)
    mapping: Dict[str, Inner] = field(default_factory=dict)
    quoted: "Inner" = field(default_factory=lambda: Inner(0))
    quoted_optional: "Optional[Inner]" = None
    plain: Any = None
    ambiguous: Union[Inner, dict] = None
    fwd_many: List["Inner"] = field(default_factory=list)
    annotated: Annotated[Inner, "meta"] = field(default_factory=lambda: Inner(0))


def test_dict_to_dataclass_converts_optional_list_and_string_annotations():
    # The repro of #5201: only the plain ``inner: Inner`` field used to be converted.
    got = dict_to_dataclass({"inner": {"v": 1}, "maybe": {"v": 2}, "many": [{"v": 3}], "quoted": {"v": 4}}, Holder)
    assert got == Holder(inner=Inner(1), maybe=Inner(2), many=[Inner(3)], quoted=Inner(4))


def test_dict_to_dataclass_converts_through_unions_and_containers():
    got = dict_to_dataclass(
        {
            "either": {"v": 1},
            "pep604": {"v": 2},
            "builtin_many": [{"v": 3}],
            "nested": [[{"v": 4}], []],
            "maybe_many": [{"v": 5}],
            "maybe_variadic": [{"v": 12}],  # a list, as JSON or YAML gives for a tuple
            "text_or_many": [{"v": 13}],
            "variadic": ({"v": 6}, {"v": 7}),
            "fixed": [{"v": 8}, 9],
            "mapping": {"a": {"v": 10}},
            "quoted_optional": {"v": 11},
        },
        Holder,
    )
    assert got == Holder(
        either=Inner(1),
        pep604=Inner(2),
        builtin_many=[Inner(3)],
        nested=[[Inner(4)], []],
        maybe_many=[Inner(5)],
        maybe_variadic=[Inner(12)],
        text_or_many=[Inner(13)],
        variadic=(Inner(6), Inner(7)),
        fixed=[Inner(8), 9],
        mapping={"a": Inner(10)},
        quoted_optional=Inner(11),
    )


def test_dict_to_dataclass_keeps_dataclass_instances_and_other_values():
    inner = Inner(5)
    got = dict_to_dataclass(
        {"inner": inner, "maybe": Inner(6), "many": [Inner(7), {"v": 8}], "plain": {"v": 9}, "pep604": None}, Holder
    )
    assert got.inner is inner
    assert got == Holder(inner=inner, maybe=Inner(6), many=[Inner(7), Inner(8)], plain={"v": 9})
    # a value that does not match a fixed-length tuple annotation is kept whole
    assert dict_to_dataclass({"fixed": [{"v": 1}, 2, 3]}, Holder).fixed == [{"v": 1}, 2, 3]


def test_dict_to_dataclass_follows_a_union_only_when_one_member_fits():
    got = dict_to_dataclass({"ambiguous": {"v": 1}, "maybe": 5}, Holder)
    assert got.ambiguous == {"v": 1}
    assert got.maybe == 5


def test_dict_to_dataclass_rejects_other_values_for_a_dataclass_field():
    # A position annotated with a dataclass takes a dict or an instance, whatever the annotation style or container.
    for value in ({"inner": 5}, {"many": [5]}, {"fixed": ["x", 2]}, {"mapping": {"a": 5}}, {"quoted": None}):
        with pytest.raises(TypeCheckError):
            dict_to_dataclass(value, Holder)


def test_dict_to_dataclass_resolves_nested_forward_refs_and_annotated():
    got = dict_to_dataclass({"fwd_many": [{"v": 1}], "annotated": {"v": 2}}, Holder)
    assert got.fwd_many == [Inner(1)]
    assert got.annotated == Inner(2)


def test_dict_to_dataclass_keeps_a_namedtuple_value_whole():
    # only a plain list or tuple is rebuilt: a namedtuple's constructor takes one argument per field
    assert dict_to_dataclass({"fixed": Pair({"v": 1}, 2)}, Holder).fixed == Pair({"v": 1}, 2)
    assert dict_to_dataclass({"variadic": Point(1, 2)}, Holder).variadic == Point(1, 2)


def test_dict_to_dataclass_round_trips_dataclass_to_dict():
    holder = Holder(
        inner=Inner(1),
        maybe=Inner(2),
        pep604=Inner(3),
        many=[Inner(4)],
        nested=[[Inner(5)]],
        variadic=(Inner(6),),
        fixed=(Inner(7), 8),
        mapping={"a": Inner(9)},
        quoted=Inner(10),
        quoted_optional=Inner(11),
    )
    assert dict_to_dataclass(dataclass_to_dict(holder), Holder) == holder


_POSTPONED_SOURCE = """
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Leaf:
    v: int = 0


@dataclass
class Tree:
    leaf: Leaf = field(default_factory=Leaf)
    maybe: Optional[Leaf] = None
    leaves: list[Leaf] = field(default_factory=list)


@dataclass
class NoneDefault:
    sub: Leaf = None
"""


def _load_postponed_module(tmp_path, monkeypatch):
    path = tmp_path / "_kornia_test_postponed_dataclasses.py"
    path.write_text(_POSTPONED_SOURCE)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, path.stem, module)
    spec.loader.exec_module(module)
    return module


def test_dict_to_dataclass_resolves_postponed_annotations(tmp_path, monkeypatch):
    # With ``from __future__ import annotations`` every annotation is a string, including a plain ``Leaf``.
    module = _load_postponed_module(tmp_path, monkeypatch)
    tree = dict_to_dataclass({"leaf": {"v": 1}, "maybe": {"v": 2}, "leaves": [{"v": 3}]}, module.Tree)
    assert tree == module.Tree(module.Leaf(1), module.Leaf(2), [module.Leaf(3)])


def test_dict_to_dataclass_postponed_none_for_a_dataclass_field_raises(tmp_path, monkeypatch):
    # ``sub: Leaf = None`` resolves to ``Leaf``, which takes a dict or an instance: None used to pass through while
    # the annotation stayed a string, and now raises like it does for a field annotated with the class itself.
    module = _load_postponed_module(tmp_path, monkeypatch)
    with pytest.raises(TypeCheckError):
        dict_to_dataclass(dataclass_to_dict(module.NoneDefault()), module.NoneDefault)


def test_dict_to_dataclass_unresolvable_annotation_is_used_as_written():
    @dataclass
    class Local:
        v: int

    @dataclass
    class WithLocal:
        local: "Local"
        inner: Inner

    got = dict_to_dataclass({"local": {"v": 1}, "inner": {"v": 2}}, WithLocal)
    assert got.local == {"v": 1}
    assert got.inner == Inner(2)


def test_sold2_detector_config_round_trips():
    cfg = DetectorCfg(grid_size=4, line_detector_cfg=LineDetectorCfg(detect_thresh=0.7, num_samples=32))
    cfg.line_detector_cfg.heatmap_refine_cfg.ratio = 0.3
    assert dict_to_dataclass(dataclass_to_dict(cfg), DetectorCfg) == cfg
    partial = dict_to_dataclass({"line_detector_cfg": LineDetectorCfg(detect_thresh=0.7)}, DetectorCfg)
    assert partial == DetectorCfg(line_detector_cfg=LineDetectorCfg(detect_thresh=0.7))

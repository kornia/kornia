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
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from kornia.core.utils import dataclass_to_dict


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

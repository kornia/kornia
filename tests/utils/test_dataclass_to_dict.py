# TEST OFFICIAL SUPPORT
from collections import namedtuple

from dataclasses import dataclass, field

import pytest

from kornia.core.utils import dataclass_to_dict

Point = namedtuple("Point", "x y")


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

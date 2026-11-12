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

"""Tests for ``kornia.core._compat``: the ``deprecated`` decorator and the attention helper."""

from __future__ import annotations

import copy
import dataclasses
import enum
import functools
import importlib.util
import inspect
import pickle
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Self

import pytest
import torch

import kornia
import kornia.core._compat as compat
import kornia.utils
from kornia.core._compat import deprecated, scaled_dot_product_attention
from kornia.feature.dedode.transformer.layers.attention import Attention as DeDoDeAttention
from kornia.models.qwen25.vision import Qwen25VisionAttention

from testing.base import BaseTester

_HEAD = "Since kornia 0.8.3 the `old` is deprecated in favor of `new`."


def _record(fn, *args: Any, **kwargs: Any) -> list[warnings.WarningMessage]:
    """Call ``fn`` and return every warning it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args, **kwargs)
    return caught


@contextmanager
def _caught() -> Iterator[list[warnings.WarningMessage]]:
    """Record the warnings raised inside the ``with`` block.

    The call under test stays in the test body, so ``stacklevel`` is judged against the test's own frame.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _line_below() -> int:
    """Return the line number of the statement that follows the call to this helper."""
    return inspect.stack()[1].lineno + 1


@deprecated(replace_with="NewCls", version="0.9.0")
class _OldCls:
    """Old class docstring."""

    X = 1

    def __init__(self, value: int = 0) -> None:
        self.value = value

    @classmethod
    def build(cls) -> _OldCls:
        return cls(7)

    @staticmethod
    def twice(x: int) -> int:
        return 2 * x


@deprecated(replace_with="NewBare", version="0.9.0")
class _OldBare:
    """A class that defines no ``__init__`` and no ``__new__``."""


@deprecated(version="0.9.0")
class _OldNewOnly(tuple):
    """A class that customises ``__new__`` and relies on ``object.__init__``."""

    __slots__ = ()

    def __new__(cls, a: int, b: int) -> Self:
        return super().__new__(cls, (a, b))


@deprecated(replace_with="NewModule", version="0.9.0")
class _OldModule(torch.nn.Module):
    """A class that inherits ``__init__`` from a base class."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _Base:
    def __init__(self) -> None:
        self.b = 1


class _CooperativeLeft:
    def __init__(self) -> None:
        self.left = 1
        super().__init__()


class _CooperativeRight:
    def __init__(self) -> None:
        self.right = 1
        super().__init__()


@deprecated(version="0.9.0")
class _Mixin:
    """A mixin that defines no ``__init__``."""


class _CallableWithoutGet:
    """A callable with no ``__get__``: ``type.__call__`` calls it as stored, without the instance."""

    def __init__(self, seen: list[Any]) -> None:
        self.seen = seen

    def __call__(self, x: int) -> None:
        self.seen.append(("callable", x))


def _classes_with_an_unusual_init(seen: list[Any]) -> dict[str, type]:
    """Return fresh classes whose own ``__init__`` is not a plain function; each records what it receives."""

    class StaticInit:
        @staticmethod
        def __init__(x: int) -> None:
            seen.append(("static", x))

    class ClassInit:
        @classmethod
        def __init__(cls, x: int) -> None:
            seen.append(("class", cls.__name__, x))

    class CallableInit:
        __init__ = _CallableWithoutGet(seen)

    return {"staticmethod": StaticInit, "classmethod": ClassInit, "callable": CallableInit}


@deprecated(version="0.9.0")
class _MixinWithInit:
    """A mixin that defines an ``__init__`` and continues along the MRO."""

    def __init__(self) -> None:
        self.m = 1
        super().__init__()


def _own_init() -> type:
    class Cls:
        def __init__(self, a: int, b: str = "x", /, c: float = 1.0, *, d: bool = False) -> None:
            self.args = (a, b, c, d)

    return Cls


def _no_init() -> type:
    class Cls:
        pass

    return Cls


def _inherited_init() -> type:
    class Base:
        def __init__(self, n: int = 2) -> None:
            self.n = n

    class Cls(Base):
        pass

    return Cls


def _new_only() -> type:
    class Cls(tuple):
        __slots__ = ()

        def __new__(cls, a: int, b: int) -> Self:
            return super().__new__(cls, (a, b))

    return Cls


def _inherited_new() -> type:
    class Base(tuple):
        __slots__ = ()

        def __new__(cls, a: int, b: int) -> Self:
            return super().__new__(cls, (a, b))

    class Cls(Base):
        __slots__ = ()

    return Cls


# What ``inspect`` reports for the ``__init__`` a decorated class gets when it has none of its own.
_OPEN_SIGNATURE = "(*args: Any, **kwargs: Any) -> None"


def _mixin_in_front_of_base(decorate: bool) -> type:
    class Base:
        def __init__(self, n: int = 2) -> None:
            self.n = n

    class Mixin:
        pass

    if decorate:
        Mixin = deprecated(version="0.9.0")(Mixin)

    class C(Mixin, Base):
        pass

    return C


def _mixin_in_front_of_new(decorate: bool) -> type:
    class Base(tuple):
        __slots__ = ()

        def __new__(cls, a: int, b: int) -> Self:
            return super().__new__(cls, (a, b))

    class Mixin:
        pass

    if decorate:
        Mixin = deprecated(version="0.9.0")(Mixin)

    class C(Mixin, Base):
        __slots__ = ()

    return C


def _mixin_in_a_diamond(decorate: bool) -> type:
    class Root:
        def __init__(self, n: int = 2) -> None:
            self.n = n

    class Mid(Root):
        pass

    if decorate:
        Mid = deprecated(version="0.9.0")(Mid)

    class Side(Root):
        def __init__(self, n: int = 2, *, side: int = 1) -> None:
            super().__init__(n)
            self.side = side

    class Diamond(Mid, Side):
        pass

    return Diamond


class TestDeprecatedMessage:
    def test_plain_function(self):
        @deprecated(replace_with="new", version="0.8.3")
        def old(x):
            return x + 1

        (w,) = _record(old, 1)
        assert issubclass(w.category, DeprecationWarning)
        assert str(w.message) == _HEAD

    @pytest.mark.parametrize(
        "extra_reason",
        [
            " Previously available as `kornia.utils.old`.",
            "Previously available as `kornia.utils.old`.",
            "  Previously available as `kornia.utils.old`.  ",
        ],
    )
    def test_one_space_before_extra_reason(self, extra_reason):
        @deprecated(replace_with="new", version="0.8.3", extra_reason=extra_reason)
        def old():
            return None

        (w,) = _record(old)
        assert str(w.message) == f"{_HEAD} Previously available as `kornia.utils.old`."

    @pytest.mark.parametrize("extra_reason", [None, "", "   "])
    def test_no_trailing_space_without_extra_reason(self, extra_reason):
        @deprecated(replace_with="new", version="0.8.3", extra_reason=extra_reason)
        def old():
            return None

        (w,) = _record(old)
        assert str(w.message) == _HEAD

    def test_without_replacement(self):
        @deprecated(version="0.8.3", extra_reason=" Use something else.")
        def old():
            return None

        (w,) = _record(old)
        assert str(w.message) == (
            "Since kornia 0.8.3 the `old` is deprecated and will be removed in the future versions. Use something else."
        )

    def test_without_version(self):
        @deprecated(replace_with="new")
        def old():
            return None

        (w,) = _record(old)
        assert str(w.message) == "`old` is deprecated in favor of `new`."

    def test_kornia_decorated_shim_has_no_double_space(self):
        (w,) = _record(kornia.utils.torch_meshgrid, [torch.arange(2), torch.arange(3)], indexing="ij")
        assert "`torch_meshgrid` is deprecated in favor of `torch.meshgrid`." in str(w.message)
        assert "  " not in str(w.message)

    def test_kornia_direct_emit_has_no_double_space(self):
        (w,) = _record(kornia.utils.ImageToTensor)
        assert "kornia.image.ImageToTensor" in str(w.message)
        assert "  " not in str(w.message)

    def test_kornia_lazy_attribute_has_no_double_space(self):
        (w,) = _record(getattr, kornia.utils, "CachedDownloader")
        assert "kornia.onnx.download.CachedDownloader" in str(w.message)
        assert "  " not in str(w.message)


class TestDeprecatedFunction:
    def test_wraps_the_function(self):
        def old(x: int, y: int = 2) -> int:
            """Old docstring."""
            return x + y

        new = deprecated(replace_with="new", version="0.8.3")(old)
        assert new.__wrapped__ is old
        assert new.__name__ == "old"
        assert new.__doc__ == "Old docstring."
        assert inspect.signature(new) == inspect.signature(old)
        with pytest.warns(DeprecationWarning, match="`old` is deprecated"):
            assert new(1, y=5) == 6

    def test_warning_points_at_the_caller_line(self):
        @deprecated(replace_with="new", version="0.8.3")
        def old():
            return None

        with _caught() as caught:
            line = _line_below()
            old()
        (w,) = caught
        assert (w.filename, w.lineno) == (__file__, line)

    def test_method(self):
        class Holder:
            @deprecated(replace_with="Holder.new", version="0.9.0")
            def old(self, x):
                return x * 3

        holder = Holder()
        with _caught() as caught:
            line = _line_below()
            assert holder.old(2) == 6
        (w,) = caught
        assert str(w.message) == "Since kornia 0.9.0 the `old` is deprecated in favor of `Holder.new`."
        assert (w.filename, w.lineno) == (__file__, line)

    def test_compile_fullgraph_of_code_calling_it(self, torch_optimizer):
        # Dynamo cannot trace ``warnings.warn``; the warning is skipped while it captures the graph (#5901).
        @deprecated(replace_with="new", version="0.8.3")
        def old(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.rand(3)
        compiled = torch_optimizer(lambda t: old(t) * 2, fullgraph=True)
        with _caught() as caught:
            # Not kornia's: Triton 3.9 warns once per process, on Inductor's first compilation, that
            # TRITON_DEFAULT_FP_FUSION is unset, which fails this test when it compiles first.
            warnings.filterwarnings(
                "ignore", message="TRITON_DEFAULT_FP_FUSION is unset", category=UserWarning, module=r"triton\."
            )
            out = compiled(x)
        assert torch.equal(out, (x + 1) * 2)
        assert caught == []
        with _caught() as caught:
            line = _line_below()
            old(x)
        (w,) = caught
        assert (w.filename, w.lineno) == (__file__, line)


class TestDeprecatedClass:
    """The decorated name has to stay a class: the issue's repro turned it into a function."""

    def test_stays_a_class(self):
        assert isinstance(_OldCls, type)
        assert _OldCls.__name__ == "_OldCls"
        assert _OldCls.__qualname__ == "_OldCls"
        assert _OldCls.__module__ == __name__
        assert _OldCls.__doc__ == "Old class docstring."

    def test_isinstance(self):
        with pytest.warns(DeprecationWarning, match="`_OldCls` is deprecated"):
            obj = _OldCls()
        assert isinstance(obj, _OldCls)
        assert not isinstance(object(), _OldCls)

    def test_subclassing(self):
        class Sub(_OldCls):
            def extra(self) -> int:
                return self.value + 1

        assert issubclass(Sub, _OldCls)
        with pytest.warns(DeprecationWarning, match="`_OldCls` is deprecated"):
            sub = Sub(4)
        assert isinstance(sub, _OldCls)
        assert sub.extra() == 5

    def test_subclass_calling_super_init_warns_once_at_the_super_call(self):
        class Sub(_OldCls):
            def __init__(self) -> None:
                self.line = _line_below()
                super().__init__(3)

        with _caught() as caught:
            sub = Sub()
        (w,) = caught
        assert issubclass(w.category, DeprecationWarning)
        assert (w.filename, w.lineno) == (__file__, sub.line)
        assert sub.value == 3

    def test_subclass_bypassing_the_decorated_init_does_not_warn(self):
        class Sub(_OldCls):
            def __init__(self) -> None:
                self.value = 5

        with _caught() as caught:
            sub = Sub()
        assert caught == []
        assert sub.value == 5

    def test_subclass_without_init_warns_once_at_the_caller(self):
        class Sub(_OldCls):
            pass

        with _caught() as caught:
            line = _line_below()
            Sub(2)
        (w,) = caught
        assert (w.filename, w.lineno) == (__file__, line)

    def test_class_attributes_and_methods(self):
        assert _OldCls.X == 1
        assert _OldCls.twice(4) == 8
        with pytest.warns(DeprecationWarning, match="`_OldCls` is deprecated"):
            built = _OldCls.build()
        assert isinstance(built, _OldCls)
        assert built.value == 7

    def test_instantiation_warns_once_with_the_message(self):
        (w,) = _record(_OldCls, 5)
        assert issubclass(w.category, DeprecationWarning)
        assert str(w.message) == "Since kornia 0.9.0 the `_OldCls` is deprecated in favor of `NewCls`."

    def test_warning_points_at_the_caller_line(self):
        with _caught() as caught:
            line = _line_below()
            _OldCls()
        (w,) = caught
        assert (w.filename, w.lineno) == (__file__, line)

    def test_constructor_arguments_are_forwarded(self):
        with pytest.warns(DeprecationWarning, match="`_OldCls` is deprecated"):
            assert _OldCls(9).value == 9
        with pytest.warns(DeprecationWarning, match="`_OldCls` is deprecated"):
            assert _OldCls(value=11).value == 11
        with pytest.warns(DeprecationWarning, match="`_OldCls` is deprecated"), pytest.raises(TypeError):
            _OldCls(1, 2)

    @pytest.mark.parametrize("make", [_own_init, _new_only], ids=["own_init", "own_new"])
    def test_signature_of_a_class_with_its_own_init_or_new_is_unchanged(self, make):
        """``inspect`` follows the wrapper's ``__wrapped__``, or prefers the class's own ``__new__``."""
        decorated = deprecated(version="0.9.0")(make())
        assert inspect.signature(decorated) == inspect.signature(make())

    @pytest.mark.parametrize(
        "make", [_no_init, _inherited_init, _inherited_new], ids=["no_init", "inherited_init", "inherited_new"]
    )
    def test_signature_of_a_class_without_its_own_init_is_open(self, make):
        """Which ``__init__`` runs depends on the MRO of the instance, so the wrapper claims no signature."""
        decorated = deprecated(version="0.9.0")(make())
        assert str(inspect.signature(decorated)) == _OPEN_SIGNATURE

    def test_signature_of_a_builtin_base_is_open(self):
        """``inspect`` raises for the undecorated ``class Cls(dict)``; the decorated one reports the open signature."""

        @deprecated(version="0.9.0")
        class Cls(dict):
            pass

        assert str(inspect.signature(Cls)) == _OPEN_SIGNATURE
        with _caught() as caught:
            obj = Cls(a=1)
        assert len(caught) == 1
        assert obj == {"a": 1}
        assert isinstance(obj, Cls)

    def test_signature_of_a_subclass_of_a_class_with_its_own_init_is_unchanged(self):
        class Sub(_OldCls):
            pass

        class Plain:
            def __init__(self, value: int = 0) -> None:
                self.value = value

        assert inspect.signature(Sub) == inspect.signature(Plain)

    def test_signature_of_a_subclass_of_a_class_without_its_own_init_is_open(self):
        @deprecated(version="0.9.0")
        class Decorated(_Base):
            pass

        class Sub(Decorated):
            pass

        class SubSub(Sub):
            pass

        assert str(inspect.signature(Decorated)) == _OPEN_SIGNATURE
        assert str(inspect.signature(Sub)) == _OPEN_SIGNATURE
        assert str(inspect.signature(SubSub)) == _OPEN_SIGNATURE

    @pytest.mark.parametrize(
        ("make", "args", "kwargs"),
        [
            pytest.param(_mixin_in_front_of_base, (5,), {}, id="mixin_in_front_of_base"),
            pytest.param(_mixin_in_front_of_new, (1, 2), {}, id="mixin_in_front_of_new"),
            pytest.param(_mixin_in_a_diamond, (5,), {"side": 3}, id="mixin_in_a_diamond"),
        ],
    )
    def test_signature_of_a_class_mixed_in_ahead_of_a_base_is_open(self, make, args, kwargs):
        """The ``__init__`` that runs is the next one along the MRO, which the wrapper cannot know in advance."""
        decorated = make(True)
        assert str(inspect.signature(decorated)) == _OPEN_SIGNATURE
        with _caught():
            decorated(*args, **kwargs)

    def test_init_wrapper_of_a_class_with_its_own_init_looks_like_that_init(self):
        init = _OldCls.__init__
        assert (init.__name__, init.__qualname__, init.__module__) == ("__init__", "_OldCls.__init__", __name__)
        assert init.__wrapped__.__qualname__ == "_OldCls.__init__"

    def test_init_wrapper_of_a_class_without_init_is_named_after_the_class(self):
        init = _OldBare.__init__
        assert (init.__name__, init.__qualname__, init.__module__) == ("__init__", "_OldBare.__init__", __name__)

    def test_init_wrapper_of_an_inherited_init_is_named_like_it_but_does_not_wrap_it(self):
        # A plain Python base with a known ``__init__``: ``nn.Module.__init__`` is replaced by dynamo's tagging
        # ``__init__`` once any compiled test has run, which would make this depend on the order of the tests.
        @deprecated(version="0.9.0")
        class Cls(_Base):
            pass

        init = Cls.__init__
        assert (init.__name__, init.__qualname__, init.__module__) == ("__init__", "_Base.__init__", __name__)
        # Under multiple inheritance another ``__init__`` may run, so ``inspect`` must not follow a fixed one.
        assert not hasattr(init, "__wrapped__")

    def test_init_wrapper_of_a_c_level_init_is_named_after_the_class(self):
        @deprecated(version="0.9.0")
        class Cls(dict):
            pass

        init = Cls.__init__
        assert (init.__name__, init.__qualname__, init.__module__) == (
            "__init__",
            f"{Cls.__qualname__}.__init__",
            __name__,
        )

    def test_without_init(self):
        (w,) = _record(_OldBare)
        assert str(w.message) == "Since kornia 0.9.0 the `_OldBare` is deprecated in favor of `NewBare`."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert isinstance(_OldBare(), _OldBare)

    def test_without_init_still_rejects_arguments(self):
        """``object()`` rejects arguments; the decorated class must not start to swallow them."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(TypeError, match="takes no arguments"):
                _OldBare(1)

    def test_custom_new_without_init(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pair = _OldNewOnly(1, 2)
        assert len(caught) == 1
        assert tuple(pair) == (1, 2)
        assert isinstance(pair, _OldNewOnly)

    def test_inherited_init(self):
        with pytest.warns(DeprecationWarning, match="`_OldModule` is deprecated in favor of `NewModule`"):
            module = _OldModule()
        assert isinstance(module, _OldModule)
        assert isinstance(module, torch.nn.Module)
        assert module(torch.zeros(1)).item() == 1.0

    @pytest.mark.parametrize("kind", ["staticmethod", "classmethod", "callable"])
    def test_an_init_that_is_not_a_plain_function_gets_the_arguments_it_gets_undecorated(self, kind):
        undecorated_seen: list[Any] = []
        decorated_seen: list[Any] = []
        _classes_with_an_unusual_init(undecorated_seen)[kind](1)
        old = deprecated(version="0.9.0")(_classes_with_an_unusual_init(decorated_seen)[kind])
        with _caught() as caught:
            obj = old(1)
        assert isinstance(obj, old)
        assert decorated_seen == undecorated_seen
        assert len(caught) == 1

    def test_a_classmethod_init_is_bound_to_the_subclass_being_instantiated(self):
        seen: list[Any] = []
        old = deprecated(version="0.9.0")(_classes_with_an_unusual_init(seen)["classmethod"])

        class Sub(old):
            pass

        with _caught() as caught:
            Sub(2)
        assert seen == [("class", "Sub", 2)]
        assert len(caught) == 1

    def test_copy_and_pickle_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            obj = _OldCls(3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert copy.copy(obj).value == 3
            assert copy.deepcopy(obj).value == 3
            assert pickle.loads(pickle.dumps(obj)).value == 3  # noqa: S301

    def test_extra_reason_is_separated_by_one_space(self):
        @deprecated(replace_with="New", version="0.9.0", extra_reason=" Moved.")
        class Old:
            pass

        (w,) = _record(Old)
        assert str(w.message) == "Since kornia 0.9.0 the `Old` is deprecated in favor of `New`. Moved."


class TestDeprecatedClassMultipleInheritance:
    """A decorated class must hand ``__init__`` on along the instance's MRO, as the undecorated class does."""

    def test_no_own_init_continues_to_the_next_base(self):
        class C(_Mixin, _Base):
            pass

        with _caught() as caught:
            line = _line_below()
            obj = C()
        (w,) = caught
        assert obj.b == 1
        assert (w.filename, w.lineno) == (__file__, line)
        assert "`_Mixin` is deprecated" in str(w.message)

    def test_no_own_init_follows_the_mro_of_the_instance(self):
        class C(_Mixin, _CooperativeLeft, _CooperativeRight):
            pass

        with _caught() as caught:
            obj = C()
        assert len(caught) == 1
        assert (obj.left, obj.right) == (1, 1)

    def test_own_init_under_multiple_inheritance(self):
        class C(_MixinWithInit, _Base):
            pass

        with _caught() as caught:
            obj = C()
        assert len(caught) == 1
        assert (obj.m, obj.b) == (1, 1)

    def test_no_own_init_still_rejects_arguments_when_object_is_next(self):
        class C(_Mixin):
            pass

        with _caught(), pytest.raises(TypeError, match=r"C\(\) takes no arguments"):
            C(1)

    def test_arguments_reach_the_next_base(self):
        class Takes:
            def __init__(self, n: int) -> None:
                self.n = n

        class C(_Mixin, Takes):
            pass

        with _caught():
            assert C(4).n == 4

    def test_a_twice_decorated_mixin_still_reaches_the_next_base(self):
        @deprecated(version="0.9.0")
        @deprecated(version="0.9.0")
        class Mixin:
            pass

        class C(Mixin, _Base):
            pass

        with _caught() as caught:
            obj = C()
        assert obj.b == 1
        assert len(caught) == 2  # one per decoration

    def test_a_mixin_whose_init_is_rewrapped_above_deprecated_still_reaches_the_next_base(self):
        def rewrap(cls: type) -> type:
            inner = cls.__init__

            @functools.wraps(inner)
            def init(self: Any, *args: Any, **kwargs: Any) -> None:
                inner(self, *args, **kwargs)

            cls.__init__ = init
            return cls

        @rewrap
        @deprecated(version="0.9.0")
        class Mixin:
            pass

        class C(Mixin, _Base):
            pass

        with _caught() as caught:
            obj = C()
        assert obj.b == 1
        assert len(caught) == 1

    def test_the_wrapper_called_on_a_foreign_object_runs_the_init_the_class_had(self):
        class Foreign:
            pass

        with _caught() as caught:
            _Mixin.__init__(Foreign())
        assert len(caught) == 1


class TestDeprecatedClassLimits:
    """The sentences of the ``deprecated`` docstring about what a decorated class does and does not do."""

    def test_the_class_is_modified_in_place(self):
        class New:
            pass

        old = deprecated(version="0.9.0")(New)
        assert old is New
        with pytest.warns(DeprecationWarning, match="`New` is deprecated"):
            New()

    def test_an_enum_never_warns(self):
        @deprecated(version="0.9.0")
        class Color(enum.Enum):
            RED = 1

        with _caught() as caught:
            assert Color(1) is Color.RED
            assert Color["RED"] is Color.RED
        assert caught == []

    def test_a_memberless_enum_base_warns_once_per_member_when_a_subclass_defines_members(self):
        @deprecated(version="0.9.0")
        class Base(enum.Enum):
            pass

        with _caught() as caught:

            class Sub(Base):
                A = 1
                B = 2

        assert len(caught) == 2
        assert all("`Base` is deprecated" in str(w.message) for w in caught)
        with _caught() as caught:
            assert Sub(1) is Sub.A
            assert Sub["A"] is Sub.A
        assert caught == []

    def test_deprecated_above_dataclass_warns_and_keeps_the_fields(self):
        @deprecated(version="0.9.0")
        @dataclasses.dataclass
        class Point:
            x: int = 0
            y: int = 0

        with _caught() as caught:
            point = Point(1, y=2)
        assert len(caught) == 1
        assert (point.x, point.y) == (1, 2)
        assert dataclasses.asdict(point) == {"x": 1, "y": 2}

    @pytest.mark.parametrize(
        ("options", "plain_default_is_readable"), [({}, True), ({"slots": True}, False)], ids=["plain", "slots"]
    )
    def test_dataclass_above_deprecated_keeps_the_decorators_init(self, options, plain_default_is_readable):
        """``@dataclass`` does not replace an ``__init__`` the class already has; ``slots=True`` copies it over."""

        @dataclasses.dataclass(**options)
        @deprecated(version="0.9.0")
        class Point:
            x: int = 0
            ys: list = dataclasses.field(default_factory=list)

        with _caught() as caught:
            point = Point()
        assert len(caught) == 1
        assert hasattr(point, "x") is plain_default_is_readable
        assert not hasattr(point, "ys")
        with _caught() as caught, pytest.raises(TypeError, match=r"Point\(\) takes no arguments"):
            Point(1)
        assert len(caught) == 1


class TestScaledDotProductAttention(BaseTester):
    """The helper that keeps kornia off the faulty float64 CPU kernel of torch 2.5 on Windows (#5992)."""

    @staticmethod
    def _simulate_the_fault(monkeypatch, faulty_platform: bool) -> list[int]:
        # Reproduce the kernel's fault on any host: a 4-D float64 CPU query gives NaN, as it selects the fused kernel.
        defaults = scaled_dot_product_attention.__defaults__
        monkeypatch.setattr(scaled_dot_product_attention, "__defaults__", (*defaults[:-1], faulty_platform))
        real = torch.nn.functional.scaled_dot_product_attention
        ndims = []

        def faulty(query, key, value, **kwargs):
            ndims.append(query.ndim)
            if query.ndim == 4 and query.dtype == torch.float64 and query.device.type == "cpu":
                return torch.full_like(query, float("nan"))
            return real(query, key, value, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", faulty)
        return ndims

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("scale", [None, 0.25])
    def test_faulty_kernel_is_avoided(self, device, monkeypatch, masked, scale):
        if device.type != "cpu":
            pytest.skip("the kernel is CPU-only")
        dtype = torch.float64
        q, k, v = (torch.rand(2, 3, 5, 8, device=device, dtype=dtype) for _ in range(3))
        mask = None
        scores = q @ k.transpose(-2, -1) * (8**-0.5 if scale is None else scale)
        if masked:
            mask = torch.rand(2, 3, 5, 5, device=device) > 0.5
            mask[..., 0] = True
            scores = scores.masked_fill(~mask, -torch.inf)
        expected = scores.softmax(-1) @ v
        self._simulate_the_fault(monkeypatch, faulty_platform=True)
        self.assert_close(scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale), expected)

    @pytest.mark.parametrize(
        "faulty_platform, shape, dtype, expected_ndim",
        [
            (True, (1, 2, 3, 4), torch.float64, 5),
            (True, (1, 2, 3, 4), torch.float32, 4),
            (True, (2, 3, 4), torch.float64, 3),
            (False, (1, 2, 3, 4), torch.float64, 4),
        ],
    )
    def test_only_the_faulty_input_is_rerouted(self, device, monkeypatch, faulty_platform, shape, dtype, expected_ndim):
        if device.type != "cpu":
            pytest.skip("the kernel is CPU-only")
        ndims = self._simulate_the_fault(monkeypatch, faulty_platform)
        q = torch.rand(*shape, device=device, dtype=dtype)
        assert scaled_dot_product_attention(q, q, q).shape == q.shape
        assert ndims == [expected_ndim]

    def test_other_devices_are_not_rerouted(self, monkeypatch):
        ndims = self._simulate_the_fault(monkeypatch, faulty_platform=True)
        q = torch.empty(1, 2, 3, 4, device="meta", dtype=torch.float64)
        scaled_dot_product_attention(q, q, q)
        assert ndims == [4]

    @pytest.mark.parametrize("faulty_platform", [False, True])
    def test_options_reach_torch(self, monkeypatch, faulty_platform):
        self._simulate_the_fault(monkeypatch, faulty_platform)
        seen = []

        def spy(query, key, value, **kwargs):
            seen.append(kwargs)
            return query

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
        q = torch.rand(1, 2, 3, 4, dtype=torch.float64)
        mask = torch.ones(3, 3, dtype=torch.bool)
        scaled_dot_product_attention(q, q, q, attn_mask=mask, dropout_p=0.5, is_causal=True, scale=0.25)
        assert seen == [{"attn_mask": mask, "dropout_p": 0.5, "is_causal": True, "scale": 0.25}]

    @pytest.mark.parametrize(
        ("platform", "torch_version", "expected_version_lt", "expected_flag"),
        [
            pytest.param("win32", "2.5.1", True, True, id="windows-2.5.1"),
            pytest.param("win32", "2.5.1+cpu", True, True, id="windows-2.5.1-local-suffix"),
            pytest.param("win32", "2.6.0", False, False, id="windows-2.6.0"),
            pytest.param("win32", "2.15.0+cu132", False, False, id="windows-2.15.0"),
            pytest.param("linux", "2.5.1", True, False, id="linux-2.5.1"),
            pytest.param("linux", "2.6.0", False, False, id="linux-2.6.0"),
            pytest.param("darwin", "2.5.1", True, False, id="macos-2.5.1"),
        ],
    )
    def test_platform_flag_is_set_only_for_windows_below_2_6(
        self, monkeypatch, platform, torch_version, expected_version_lt, expected_flag
    ):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(torch, "__version__", torch_version)

        spec = importlib.util.spec_from_file_location("kornia.core._compat_copy", compat.__file__)
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)

        assert fresh._TORCH_VERSION == fresh._parse_torch_version(torch_version)
        assert fresh.torch_version_lt(2, 6, 0) is expected_version_lt
        assert fresh._SDPA_FLOAT64_CPU_KERNEL_IS_FAULTY is expected_flag

        param = inspect.signature(fresh.scaled_dot_product_attention).parameters["kernel_is_faulty"]
        assert param.default is expected_flag

    @pytest.mark.parametrize(
        "make_module",
        [
            lambda: DeDoDeAttention(dim=16, num_heads=4),
            lambda: Qwen25VisionAttention(16, 4),
        ],
        ids=["dedode", "qwen25"],
    )
    def test_scripted_callers_are_scriptable(self, make_module):
        # TorchScript cannot read a Python global, whatever its value: the helper takes the flag as a default argument.
        torch.jit.script(make_module().eval())

    @pytest.mark.parametrize("faulty_platform", [False, True])
    def test_scripted_matches_eager(self, faulty_platform):
        # Pass the flag: TorchScript keeps the first compile of a function, so patching the default would not reach it.
        q, k, v = (torch.rand(2, 3, 5, 8, dtype=torch.float64) for _ in range(3))
        actual = torch.jit.script(scaled_dot_product_attention)(q, k, v, kernel_is_faulty=faulty_platform)
        self.assert_close(actual, scaled_dot_product_attention(q, k, v, kernel_is_faulty=faulty_platform))

    def test_gradcheck(self, device, monkeypatch):
        if device.type != "cpu":
            pytest.skip("the kernel is CPU-only")
        self._simulate_the_fault(monkeypatch, faulty_platform=True)
        q, k, v = (torch.rand(1, 1, 2, 4, device=device, dtype=torch.float64, requires_grad=True) for _ in range(3))
        self.gradcheck(scaled_dot_product_attention, (q, k, v))

    def test_kornia_calls_the_helper_rather_than_the_torch_function(self):
        # DeDoDe gained a direct call after LightGlue got the workaround, and the models had always had one: a CI leg
        # that needs Windows and torch 2.5, and runs only on the schedule, is too late and too far to catch the next.
        root = Path(kornia.__file__).parent
        pattern = re.compile(r"\b(F|functional)\.scaled_dot_product_attention\(")
        callers = [
            str(path.relative_to(root))
            for path in sorted(root.rglob("*.py"))
            if path.name != "_compat.py" and pattern.search(path.read_text(encoding="utf-8"))
        ]
        assert not callers, f"use kornia.core._compat.scaled_dot_product_attention in {callers}"

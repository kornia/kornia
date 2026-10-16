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

import copy
import pickle

import pytest
import torch

from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_DM_DESC,
    KORNIA_CHECK_IS_COLOR,
    KORNIA_CHECK_IS_COLOR_OR_GRAY,
    KORNIA_CHECK_IS_GRAY,
    KORNIA_CHECK_IS_IMAGE,
    KORNIA_CHECK_IS_LIST_OF_TENSOR,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_SAME_DEVICE,
    KORNIA_CHECK_SAME_DEVICES,
    KORNIA_CHECK_SAME_SHAPE,
    KORNIA_CHECK_SHAPE,
    KORNIA_CHECK_TYPE,
)
from kornia.core.exceptions import (
    BaseError,
    DeviceError,
    ImageError,
    ShapeError,
    TypeCheckError,
    ValueCheckError,
)

from testing.base import BaseTester

# One real check failure per helper, with the class it raises and the builtin that class also derives from (#5193).
_CHECK_FAILURES = [
    pytest.param(
        lambda device: KORNIA_CHECK_SHAPE(torch.zeros(2, 3, device=device), ["1", "H", "W"]),
        ShapeError,
        ValueError,
        id="KORNIA_CHECK_SHAPE",
    ),
    pytest.param(
        lambda device: KORNIA_CHECK_SAME_SHAPE(torch.zeros(2, 3, device=device), torch.zeros(3, 2, device=device)),
        ShapeError,
        ValueError,
        id="KORNIA_CHECK_SAME_SHAPE",
    ),
    pytest.param(
        lambda device: KORNIA_CHECK_DM_DESC(
            torch.zeros(4, device=device), torch.zeros(8, device=device), torch.zeros(4, 7, device=device)
        ),
        ShapeError,
        ValueError,
        id="KORNIA_CHECK_DM_DESC",
    ),
    pytest.param(lambda device: KORNIA_CHECK_TYPE(1, str), TypeCheckError, TypeError, id="KORNIA_CHECK_TYPE"),
    pytest.param(lambda device: KORNIA_CHECK_IS_TENSOR([1.0]), TypeCheckError, TypeError, id="KORNIA_CHECK_IS_TENSOR"),
    pytest.param(
        lambda device: KORNIA_CHECK_IS_LIST_OF_TENSOR([torch.zeros(1, device=device), 1]),
        TypeCheckError,
        TypeError,
        id="KORNIA_CHECK_IS_LIST_OF_TENSOR",
    ),
    pytest.param(
        lambda device: KORNIA_CHECK_IS_IMAGE(torch.full((1, 2, 2), 2.0, device=device)),
        ValueCheckError,
        ValueError,
        id="KORNIA_CHECK_IS_IMAGE",
    ),
    pytest.param(
        lambda device: KORNIA_CHECK_SAME_DEVICE(torch.zeros(1, device=device), torch.zeros(1, device="meta")),
        DeviceError,
        ValueError,
        id="KORNIA_CHECK_SAME_DEVICE",
    ),
    pytest.param(
        lambda device: KORNIA_CHECK_SAME_DEVICES([torch.zeros(1, device=device), torch.zeros(1, device="meta")]),
        DeviceError,
        ValueError,
        id="KORNIA_CHECK_SAME_DEVICES",
    ),
    pytest.param(
        lambda device: KORNIA_CHECK_IS_COLOR(torch.zeros(1, 4, 4, device=device)),
        ImageError,
        ValueError,
        id="KORNIA_CHECK_IS_COLOR",
    ),
    pytest.param(
        lambda device: KORNIA_CHECK_IS_GRAY(torch.zeros(3, 4, 4, device=device)),
        ImageError,
        ValueError,
        id="KORNIA_CHECK_IS_GRAY",
    ),
    pytest.param(
        lambda device: KORNIA_CHECK_IS_COLOR_OR_GRAY(torch.zeros(2, 4, 4, device=device)),
        ImageError,
        ValueError,
        id="KORNIA_CHECK_IS_COLOR_OR_GRAY",
    ),
]


class TestExceptionBases(BaseTester):
    """The builtin bases of the check exceptions (#5193)."""

    @pytest.mark.parametrize("check, error, builtin", _CHECK_FAILURES)
    def test_convention_kornia_errors_are_catchable_as_builtin(self, device, check, error, builtin):
        # The ``except`` a caller writes around a kornia call catches the check's class through its builtin base.
        with pytest.raises(builtin) as excinfo:
            check(device)
        assert type(excinfo.value) is error
        # ``except BaseError`` keeps catching every check failure.
        assert isinstance(excinfo.value, BaseError)
        # Each class has one builtin base: a shape error is not a ``TypeError``, a type error not a ``ValueError``.
        other = TypeError if builtin is ValueError else ValueError
        assert not isinstance(excinfo.value, other)

    def test_kornia_check_raises_the_bare_base_error(self):
        # ``KORNIA_CHECK`` carries no semantic of its own, so its ``BaseError`` stays a plain ``Exception``.
        with pytest.raises(BaseError) as excinfo:
            KORNIA_CHECK(False, "a failed condition")
        assert type(excinfo.value) is BaseError
        assert not isinstance(excinfo.value, (ValueError, TypeError))
        assert BaseError.__bases__ == (Exception,)

    @pytest.mark.parametrize("check, error, builtin", _CHECK_FAILURES)
    def test_pickle_and_copy_keep_class_message_and_attributes(self, device, check, error, builtin):
        with pytest.raises(error) as excinfo:
            check(device)
        raised = excinfo.value
        protocols = range(2, pickle.HIGHEST_PROTOCOL + 1)
        clones = [pickle.loads(pickle.dumps(raised, protocol=p)) for p in protocols]  # noqa: S301 - our own bytes
        clones += [copy.copy(raised), copy.deepcopy(raised)]
        for clone in clones:
            assert type(clone) is error
            assert isinstance(clone, builtin)
            assert clone.args == raised.args
            assert str(clone) == str(raised)
            assert vars(clone) == vars(raised)

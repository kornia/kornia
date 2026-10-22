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

"""The removed Ivy-powered entry points explain their removal whichever way 0.8.2 code reaches them (#5906)."""

import re

import pytest

import kornia
import kornia.transpiler

_REMOVED = ("to_jax", "to_numpy", "to_tensorflow")


def _message(qualified_name: str) -> str:
    return re.escape(f"{qualified_name}() was removed: testing found the Ivy-powered multi-framework transpiler")


@pytest.mark.parametrize("name", _REMOVED)
@pytest.mark.parametrize("module", ["kornia", "kornia.transpiler"])
class TestRemovedEntryPoints:
    def test_from_import_then_call_explains_the_removal(self, module, name):
        namespace: dict = {}
        exec(f"from {module} import {name}", namespace)  # noqa: S102  a fixed import statement
        with pytest.raises(RuntimeError, match=_message(f"{module}.{name}")) as info:
            namespace[name]()
        assert "multi-framework-support.html" in str(info.value)

    def test_attribute_call_explains_the_removal(self, module, name):
        fn = getattr(kornia if module == "kornia" else kornia.transpiler, name)
        with pytest.raises(RuntimeError, match=_message(f"{module}.{name}")):
            fn(kornia, source="torch")

    def test_looking_the_name_up_does_not_raise(self, module, name):
        assert hasattr(kornia if module == "kornia" else kornia.transpiler, name)


def test_names_stay_out_of_the_transpiler_public_surface():
    assert kornia.transpiler.__all__ == []


def test_star_imports_do_not_raise():
    for module in ("kornia", "kornia.transpiler"):
        exec(f"from {module} import *", {})  # noqa: S102  a fixed import statement

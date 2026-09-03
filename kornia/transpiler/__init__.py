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

"""Removed: ``kornia.to_tensorflow()``, ``kornia.to_jax()`` and ``kornia.to_numpy()``.

Testing in September 2026 found the Ivy-powered transpiler these functions used unreliable
across all three target frameworks, so it was removed. See
``docs/source/get-started/multi-framework-support.rst`` for what was tested and why.
"""

from typing import Any

_REMOVED = ("to_jax", "to_numpy", "to_tensorflow")

_DOCS_URL = "https://kornia.readthedocs.io/en/latest/get-started/multi-framework-support.html"


def _removed_message(qualified_name: str) -> str:
    """Explain the removal of one entry point, named as the caller reached for it.

    ``qualified_name`` is the full dotted path the user wrote -- ``kornia.to_jax`` from the
    top level, ``kornia.transpiler.to_jax`` from this module -- so the error quotes back the
    call that failed rather than a canonical spelling the user never used.
    """
    return (
        f"{qualified_name}() was removed: testing found the Ivy-powered multi-framework "
        f"transpiler unreliable, so it is no longer part of kornia. See {_DOCS_URL}"
    )


def __getattr__(name: str) -> Any:
    if name in _REMOVED:
        raise AttributeError(_removed_message(f"kornia.transpiler.{name}"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

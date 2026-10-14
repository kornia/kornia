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

The three names stay bound, here and at the top level, to functions that raise :class:`RuntimeError`
explaining the removal when called, and stay out of ``__all__``.
"""

from typing import Any, Callable, List, NoReturn

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


def _removed_entry_point(module: str, name: str) -> Callable[..., NoReturn]:
    """Return the stand-in bound as ``<module>.<name>``, which raises :class:`RuntimeError` explaining the removal.

    The name has to stay bound: for ``from kornia import to_jax`` Python replaces the ``AttributeError`` of a missing
    name with a bare ``ImportError``, which drops the explanation. Looking the name up therefore succeeds, and calling
    it explains.
    """
    qualified_name = f"{module}.{name}"

    def removed(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(_removed_message(qualified_name))

    removed.__name__ = removed.__qualname__ = name
    removed.__module__ = module
    removed.__doc__ = f"Removed: calling ``{qualified_name}()`` raises :class:`RuntimeError` explaining why."
    return removed


to_jax = _removed_entry_point(__name__, "to_jax")
to_numpy = _removed_entry_point(__name__, "to_numpy")
to_tensorflow = _removed_entry_point(__name__, "to_tensorflow")

# The stand-ins above are not API.
__all__: List[str] = []

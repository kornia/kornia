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

"""The removed multi-framework entry points must explain themselves where users reach for them.

`kornia.to_tensorflow()`, `kornia.to_jax()` and `kornia.to_numpy()` were removed in #4196. The
docs only ever showed them as attributes of the top-level package (``import kornia`` then
``kornia.to_tensorflow()``), so the removal has to be explained there: a message that lives only
under ``kornia.transpiler`` is reached by nobody, since existing users had no reason to import
that submodule by name.

These tests pin the formerly documented access paths -- the top-level attribute, the submodule
attribute, and the fact that ``kornia.transpiler`` still resolves after a plain ``import kornia``
-- plus the guarantee that adding a module-level ``__getattr__`` to ``kornia/__init__.py`` left
ordinary attribute lookup alone.

One path stays generic and cannot be pinned here: ``from kornia import to_tensorflow`` reports
``ImportError: cannot import name ...``. CPython's ``from X import Y`` suppresses the
``AttributeError`` a module ``__getattr__`` raises and substitutes that message, so the only way
to reach the guidance there would be to keep the name bound to something -- which would un-remove
it. NumPy 2.0's removed attributes behave the same way, and this form never appeared in kornia's
docs, which always showed ``import kornia`` followed by ``kornia.to_tensorflow()``.
"""

import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import pytest

import kornia
import kornia.transpiler

REMOVED = ("to_jax", "to_numpy", "to_tensorflow")

# The page the error message links, resolved back to the source file that renders it.
_LINKED_PAGE = PurePosixPath(urlsplit(kornia.transpiler._DOCS_URL).path)
DOCS_PAGE = (
    Path(__file__).resolve().parents[1] / "docs" / "source" / _LINKED_PAGE.parent.name / f"{_LINKED_PAGE.stem}.rst"
)


@pytest.mark.parametrize("name", REMOVED)
def test_top_level_attribute_explains_the_removal(name):
    """``kornia.to_tensorflow`` — the form every doc and README example used."""
    with pytest.raises(AttributeError) as excinfo:
        getattr(kornia, name)
    message = str(excinfo.value)
    assert f"kornia.{name}() was removed" in message
    assert "multi-framework-support" in message


@pytest.mark.parametrize("name", REMOVED)
def test_submodule_attribute_explains_the_removal(name):
    """``kornia.transpiler.to_tensorflow`` — the path code that imported the submodule used."""
    with pytest.raises(AttributeError) as excinfo:
        getattr(kornia.transpiler, name)
    message = str(excinfo.value)
    assert f"kornia.transpiler.{name}() was removed" in message
    assert "multi-framework-support" in message


def test_documented_paths_from_a_bare_import():
    """A fresh interpreter that only runs ``import kornia`` sees both messages.

    In-process this would prove nothing: any other test importing ``kornia.transpiler`` binds the
    attribute on the parent package as a side effect, so ``kornia.transpiler`` would resolve even
    if ``kornia/__init__.py`` had stopped importing it. A subprocess is the honest check.
    """
    script = textwrap.dedent(
        """
        import kornia

        kornia.transpiler  # must resolve after a plain `import kornia`, with no explicit submodule import

        for name in ("to_jax", "to_numpy", "to_tensorflow"):
            for owner, prefix in ((kornia, "kornia"), (kornia.transpiler, "kornia.transpiler")):
                try:
                    getattr(owner, name)
                except AttributeError as exc:
                    assert f"{prefix}.{name}() was removed" in str(exc), str(exc)
                    assert "multi-framework-support" in str(exc), str(exc)
                else:
                    raise AssertionError(f"{prefix}.{name} still resolves")
        """
    )
    # Trusted, fixed command (the current interpreter running a literal script); no external input.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, f"the documented access paths did not explain the removal:\n{result.stderr}"


@pytest.mark.parametrize("name", REMOVED)
def test_removed_names_are_absent_not_shimmed(name):
    """The names are gone, not kept alive: the message is guidance, not a working entry point."""
    assert not hasattr(kornia, name)
    assert getattr(kornia, name, "absent") == "absent"
    assert name not in dir(kornia)


def test_unknown_attribute_keeps_the_standard_message():
    """The new ``__getattr__`` must not change lookup for anything but the removed names."""
    with pytest.raises(AttributeError, match=r"module 'kornia' has no attribute 'not_a_kornia_name'"):
        getattr(kornia, "not_a_kornia_name")  # noqa: B009 - the raise is the point; a plain attribute is B018


def test_docs_page_the_message_points_at_exists():
    """The error links a docs page by URL; renaming the page would leave a dead link behind.

    The path is derived from the URL in the message rather than typed out again, so this fails if
    either half of the pair moves.
    """
    assert DOCS_PAGE.is_file(), f"the removal message links a page with no source file: {DOCS_PAGE}"

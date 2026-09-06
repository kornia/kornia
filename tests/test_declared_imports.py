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

"""Pin the declared-dependency contract of the ``kornia`` package (kornia#4259).

An audit found seven third-party packages that ``kornia/`` imported without any of them
appearing in ``pyproject.toml``: ``flash_attn``, ``xformers``, ``basicsr``, ``huggingface_hub``,
``safetensors``, ``requests`` and ``transformers``. Nothing in CI compared the imports against
the declaration, so each one only surfaced as an ``ImportError`` in a user's environment.

The four tests here are that comparison:

1. every third-party module imported anywhere under ``kornia/`` is declared -- as a runtime
   dependency, in some user-facing optional-dependency extra, or as a :class:`LazyLoader`;
2. every :class:`LazyLoader` names a module that a dependency a *user* can install actually
   provides, so a new lazy optional dependency cannot be added without also giving users a way to
   install it;
3. every ``LazyLoader(extra=...)`` hint names a user-facing extra that installs that module, so
   the ``pip install "kornia[<extra>]"`` line in the ``ImportError`` is one that works;
4. a bare ``import kornia`` loads none of the optional packages, so declaring a dependency as
   optional stays true at runtime.

"User-facing" excludes :data:`CONTRIBUTOR_EXTRAS` (``dev`` and ``docs``): those install what it
takes to develop kornia, and something a user runs must not need them.

The allowed set is *derived* from ``pyproject.toml`` and from the ``LazyLoader`` registry rather
than hardcoded: adding a dependency in the usual place is all it takes to satisfy these tests.
The only literals are the handful of distribution names whose import name differs
(:data:`DIST_TO_IMPORT`), :data:`CONTRIBUTOR_EXTRAS` and :data:`IMPLICIT_ALLOWED`.

Deliberately no ``packaging`` import: it is not a kornia dependency, and this file must pass on a
bare ``pip install -e ".[dev]"``.
"""

import ast
import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import kornia
from kornia.core import external
from kornia.core.external import LazyLoader

REPO_ROOT = Path(kornia.__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_DIR = REPO_ROOT / "kornia"

# Distribution name (PEP 503-normalised) -> top-level import name, for the few packages whose
# import name is not simply the normalised distribution name.
DIST_TO_IMPORT = {
    "pillow": "PIL",
    "pyyaml": "yaml",
    "opencv_python": "cv2",
    "opencv_python_headless": "cv2",
    "kornia_rs": "kornia_rs",
    "segmentation_models_pytorch": "segmentation_models_pytorch",
}

# Third-party modules that are allowed without appearing in ``pyproject.toml``: torch, which is a
# declared runtime dependency, installs them, so anything that can import torch has them too.
IMPLICIT_ALLOWED = {"typing_extensions"}

# Extras that exist for contributors, not for users of the library: nothing a user runs may depend
# on them, so they do not make a package "declared". Every other extra is user-facing and does.
CONTRIBUTOR_EXTRAS = frozenset({"dev", "docs"})

# ``LazyLoader`` modules that no user-facing dependency installs, mapped to the reason they are
# tolerated. Every entry is a package a user can hit at runtime with no documented way to install
# it, so each one is a bug waiting on a decision rather than a design.
LAZY_LOADERS_WITHOUT_DECLARED_DEP: dict[str, str] = {
    "PIL.Image": (
        "pillow is declared only in the dev/docs extras; ImageModule output_type='pil' and "
        "kornia.io.sample need it at runtime; which user-facing extra should carry it is an open "
        "decision (#4261)"
    ),
}

# Optional packages that ``import kornia`` must not pull in: everything a user-facing extra or a
# ``LazyLoader`` provides beyond the runtime dependencies (the import test checks this tuple stays a
# superset of that derived set), plus the packages the kornia#4259 audit found imported eagerly.
MUST_NOT_LOAD_ON_IMPORT = (
    "onnxruntime",
    "onnx",
    "onnxscript",
    "PIL",
    "requests",
    "diffusers",
    "transformers",
    "boxmot",
    "segmentation_models_pytorch",
    "huggingface_hub",
    "safetensors",
    "cv2",
    "yaml",
)

# ``name[extra1,extra2] >= 1.0`` -> ("name", "extra1,extra2"); markers are stripped beforehand.
_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?")


def _normalise(name: str) -> str:
    """Normalise a distribution or extra name the way PEP 503 does, with ``_`` as separator."""
    return re.sub(r"[-_.]+", "_", name.strip()).lower()


def _requirement(spec: str) -> tuple[str, tuple[str, ...]]:
    """Return the normalised distribution name and extras of a requirement string."""
    head = spec.split(";", 1)[0].split("#", 1)[0].strip()
    match = _REQUIREMENT_RE.match(head)
    if match is None:
        return "", ()
    extras = tuple(_normalise(e) for e in (match.group(2) or "").split(",") if e.strip())
    return _normalise(match.group(1)), extras


def _pyproject_import_names() -> tuple[set[str], dict[str, set[str]]]:
    """Return the import names of the runtime dependencies and of every user-facing extra.

    The extras are every ``[project.optional-dependencies]`` entry except :data:`CONTRIBUTOR_EXTRAS`,
    which exist for developing kornia, not for running it. ``kornia[<extra>]`` self-references
    expand to that extra's own requirements.
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    optional = {_normalise(name): reqs for name, reqs in project.get("optional-dependencies", {}).items()}

    def collect(specs: list[str], seen: frozenset[str]) -> set[str]:
        names: set[str] = set()
        for spec in specs:
            dist, extras = _requirement(spec)
            if not dist:
                continue
            if dist == "kornia":  # self-reference: expand the extras it pulls in
                for extra in extras:
                    if extra not in seen:
                        names |= collect(optional.get(extra, []), seen | {extra})
                continue
            names.add(DIST_TO_IMPORT.get(dist, dist))
        return names

    runtime = collect(project.get("dependencies", []), frozenset())
    user_facing = {
        extra: collect(reqs, frozenset({extra})) for extra, reqs in optional.items() if extra not in CONTRIBUTOR_EXTRAS
    }
    return runtime, user_facing


def _declared_import_names() -> set[str]:
    """Return the import names of every distribution a *user* of kornia can install."""
    runtime, user_facing = _pyproject_import_names()
    return runtime.union(*user_facing.values())


def _lazy_loaders() -> dict[str, LazyLoader]:
    """Return ``{module name: loader}`` for the ``LazyLoader`` registry."""
    return {loader.module_name: loader for loader in vars(external).values() if isinstance(loader, LazyLoader)}


def _lazy_loader_modules() -> dict[str, str]:
    """Return ``{module name: top-level module name}`` for the ``LazyLoader`` registry."""
    return {name: name.split(".")[0] for name in _lazy_loaders()}


def _imported_modules() -> list[tuple[str, int, str]]:
    """Return ``(path, line, top-level module)`` for every absolute import under ``kornia/``.

    Nested imports -- inside functions, ``try`` blocks or ``if TYPE_CHECKING`` -- count exactly
    like module-level ones: the audit that motivated this test found undeclared packages in all
    three positions.
    """
    imports: list[tuple[str, int, str]] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((rel, node.lineno, alias.name.split(".")[0]))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.append((rel, node.lineno, node.module.split(".")[0]))
    return imports


def _is_installed(module: str) -> bool:
    """Return whether ``module`` is importable in this environment, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def test_every_third_party_import_is_declared():
    """Every third-party module imported under ``kornia/`` must be a declared dependency."""
    allowed = _declared_import_names() | set(_lazy_loader_modules().values()) | IMPLICIT_ALLOWED
    ignored = set(sys.stdlib_module_names) | {"kornia"}

    undeclared = [
        f"{path}:{line} -> {module}"
        for path, line, module in sorted(_imported_modules())
        if module not in ignored and module not in allowed
    ]

    assert not undeclared, (
        "kornia/ imports modules that no pyproject.toml runtime dependency, user-facing "
        "optional-dependency extra or LazyLoader declares. Declare them, make them optional "
        "through kornia.core.external, or drop the import:\n  " + "\n  ".join(undeclared)
    )


def test_every_lazy_loader_module_is_installable():
    """Every ``LazyLoader`` module must be installable from a user-facing declared dependency."""
    declared = _declared_import_names() | IMPLICIT_ALLOWED
    orphans = {name: top for name, top in _lazy_loader_modules().items() if top not in declared}

    unexpected = sorted(set(orphans) - set(LAZY_LOADERS_WITHOUT_DECLARED_DEP))
    assert not unexpected, (
        "kornia.core.external declares LazyLoader(s) whose module no runtime dependency and no "
        "user-facing optional-dependency extra installs, so users have no documented way to get "
        "them: "
        + ", ".join(f"{name} -> {orphans[name]}" for name in unexpected)
        + ". Add the package to a user-facing optional-dependencies extra (and pass extra= to the "
        "LazyLoader), or record it in LAZY_LOADERS_WITHOUT_DECLARED_DEP with a reason."
    )

    stale = sorted(set(LAZY_LOADERS_WITHOUT_DECLARED_DEP) - set(orphans))
    assert not stale, (
        "LAZY_LOADERS_WITHOUT_DECLARED_DEP lists LazyLoader(s) that are now declared (or gone); "
        "drop the entries: " + ", ".join(stale)
    )


def test_every_lazy_loader_extra_hint_provides_its_module():
    """A ``LazyLoader(extra=...)`` hint must name a user-facing extra that installs the module.

    The hint ends up in the ``ImportError`` a user reads (``pip install "kornia[<extra>]"``), so an
    extra that does not exist, or exists but does not carry the package, sends them down a dead end.
    """
    _, extras = _pyproject_import_names()
    wrong = {
        name: loader.extra
        for name, loader in _lazy_loaders().items()
        if loader.extra is not None and name.split(".")[0] not in extras.get(_normalise(loader.extra), set())
    }

    assert not wrong, (
        "kornia.core.external declares LazyLoader(s) whose extra= hint names no user-facing "
        "optional-dependency extra that installs the module: "
        + ", ".join(f"{name} -> kornia[{extra}]" for name, extra in sorted(wrong.items()))
        + f". User-facing extras: {sorted(extras)}."
    )


def test_import_kornia_does_not_load_optional_dependencies():
    """A bare ``import kornia`` must not load any optional dependency (kornia#4260)."""
    runtime, extras = _pyproject_import_names()
    optional = (set().union(*extras.values()) | set(_lazy_loader_modules().values())) - runtime
    unlisted = sorted(optional - set(MUST_NOT_LOAD_ON_IMPORT))
    assert not unlisted, f"add these optional packages to MUST_NOT_LOAD_ON_IMPORT so this test covers them: {unlisted}"

    checked = [name for name in MUST_NOT_LOAD_ON_IMPORT if _is_installed(name)]
    if not checked:
        pytest.skip(f"none of {list(MUST_NOT_LOAD_ON_IMPORT)} is installed")

    code = (
        "import sys\n"
        "import kornia\n"
        "names = sys.argv[1:]\n"
        "print(' '.join(n for n in names if any(m == n or m.startswith(n + '.') for m in sys.modules)))\n"
    )
    # S603: no untrusted input -- this interpreter, a literal program, and module names from
    # MUST_NOT_LOAD_ON_IMPORT above. check=False so a crashed probe reports its stderr instead of
    # an opaque CalledProcessError; the return-code assertion keeps the test from passing vacuously.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, *checked],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"`import kornia` failed in the probe subprocess:\n{result.stderr}"
    loaded = result.stdout.split()

    assert not loaded, (
        f"`import kornia` eagerly loaded optional dependencies: {loaded}. They must stay behind a "
        f"kornia.core.external.LazyLoader or a function-local import. Checked: {checked}."
    )

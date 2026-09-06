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

The three tests here are that comparison:

1. every third-party module imported anywhere under ``kornia/`` is declared -- as a runtime
   dependency, in some optional-dependency extra, or as a :class:`LazyLoader`;
2. every :class:`LazyLoader` names a module that some declared dependency actually installs, so
   a new lazy optional dependency cannot be added without also giving users a way to install it;
3. a bare ``import kornia`` loads none of the optional packages, so declaring a dependency as
   optional stays true at runtime.

The allowed set is *derived* from ``pyproject.toml`` and from the ``LazyLoader`` registry rather
than hardcoded: adding a dependency in the usual place is all it takes to satisfy these tests.
The only literals are the handful of distribution names whose import name differs
(:data:`DIST_TO_IMPORT`) and :data:`IMPLICIT_ALLOWED`.

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

# ``LazyLoader`` entries whose module is provided by no declared dependency. Each needs a reason,
# and the list is empty on purpose: a lazily imported optional package the user has no documented
# way to install is a bug, not a design.
LAZY_LOADERS_WITHOUT_DECLARED_DEP: dict[str, str] = {}

# Optional packages that ``import kornia`` must not pull in.
MUST_NOT_LOAD_ON_IMPORT = (
    "onnxruntime",
    "onnx",
    "PIL",
    "requests",
    "diffusers",
    "transformers",
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


def _declared_import_names() -> set[str]:
    """Return the import names of every distribution declared in ``pyproject.toml``.

    Covers the runtime ``dependencies`` and every ``[project.optional-dependencies]`` extra;
    ``kornia[<extra>]`` self-references expand to that extra's own requirements.
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    optional = {_normalise(name): reqs for name, reqs in project.get("optional-dependencies", {}).items()}
    names: set[str] = set()

    def collect(specs: list[str], seen: frozenset[str]) -> None:
        for spec in specs:
            dist, extras = _requirement(spec)
            if not dist:
                continue
            if dist == "kornia":  # self-reference: expand the extras it pulls in
                for extra in extras:
                    if extra not in seen:
                        collect(optional.get(extra, []), seen | {extra})
                continue
            names.add(DIST_TO_IMPORT.get(dist, dist))

    collect(project.get("dependencies", []), frozenset())
    for extra, specs in optional.items():
        collect(specs, frozenset({extra}))
    return names


def _lazy_loader_modules() -> dict[str, str]:
    """Return ``{attribute name: top-level module name}`` for the ``LazyLoader`` registry."""
    return {
        attr: loader.module_name.split(".")[0]
        for attr, loader in vars(external).items()
        if isinstance(loader, LazyLoader)
    }


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
        for path, line, module in _imported_modules()
        if module not in ignored and module not in allowed
    ]

    assert not undeclared, (
        "kornia/ imports modules that no pyproject.toml dependency, optional-dependency extra or "
        "LazyLoader declares. Declare them, make them optional through kornia.core.external, or "
        "drop the import:\n  " + "\n  ".join(undeclared)
    )


def test_every_lazy_loader_module_is_installable():
    """Every ``LazyLoader`` module must be installable from a declared dependency."""
    declared = _declared_import_names() | IMPLICIT_ALLOWED
    orphans = {attr: module for attr, module in _lazy_loader_modules().items() if module not in declared}

    unexpected = sorted(set(orphans) - set(LAZY_LOADERS_WITHOUT_DECLARED_DEP))
    assert not unexpected, (
        "kornia.core.external declares LazyLoader(s) whose module no pyproject.toml dependency "
        "installs, so users have no documented way to get them: "
        + ", ".join(f"{attr} -> {orphans[attr]}" for attr in unexpected)
        + ". Add the package to an optional-dependencies extra (and pass extra= to the LazyLoader),"
        " or record it in LAZY_LOADERS_WITHOUT_DECLARED_DEP with a reason."
    )

    stale = sorted(set(LAZY_LOADERS_WITHOUT_DECLARED_DEP) - set(orphans))
    assert not stale, (
        "LAZY_LOADERS_WITHOUT_DECLARED_DEP lists LazyLoader(s) that are now declared (or gone); "
        "drop the entries: " + ", ".join(stale)
    )


def test_import_kornia_does_not_load_optional_dependencies():
    """A bare ``import kornia`` must not load any optional dependency (kornia#4260)."""
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
    # MUST_NOT_LOAD_ON_IMPORT above.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, *checked],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = result.stdout.split()

    assert not loaded, (
        f"`import kornia` eagerly loaded optional dependencies: {loaded}. They must stay behind a "
        f"kornia.core.external.LazyLoader or a function-local import. Checked: {checked}." + result.stderr
    )

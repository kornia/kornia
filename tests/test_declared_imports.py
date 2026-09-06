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
The literals are the handful of distribution names whose import name differs
(:data:`DIST_TO_IMPORT`), :data:`CONTRIBUTOR_EXTRAS`, :data:`IMPLICIT_ALLOWED`, the tolerated
orphans in :data:`LAZY_LOADERS_WITHOUT_DECLARED_DEP` (each with a reason, and checked for
staleness) and the :data:`AUDIT_EAGER_IMPORTS` canaries. The ``LazyLoader`` registry means the
module-level instances in ``kornia.core.external``; a loader constructed anywhere else is not seen.

Deliberately no ``packaging`` import: it is not a kornia dependency, and this file must pass on a
bare ``pip install -e ".[dev]"``.
"""

import ast
import importlib.metadata
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from kornia.core import external
from kornia.core.external import LazyLoader

# From this file, not from ``kornia.__file__``: under the editable install the latter can name the primary
# checkout while this test runs from a worktree, and the scan must cover the tree the test belongs to.
REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_DIR = REPO_ROOT / "kornia"

# Distribution name (PEP 503-normalised) -> top-level import name, for the few packages whose
# import name is not simply the normalised distribution name.
DIST_TO_IMPORT = {
    "pillow": "PIL",
    "pyyaml": "yaml",
    "opencv_python": "cv2",
    "opencv_python_headless": "cv2",
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

# Packages the kornia#4259 audit found ``import kornia`` loading eagerly. None is declared any more, so
# test 1 forbids importing them from ``kornia/``; the import probe still watches for them as a canary
# for a transitive load, next to every package a user-facing extra or a ``LazyLoader`` provides.
AUDIT_EAGER_IMPORTS = frozenset({"requests", "huggingface_hub", "safetensors", "cv2", "yaml"})

# ``name[extra1,extra2] >= 1.0`` -> ("name", "extra1,extra2"); markers are stripped beforehand.
_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?")


def _normalise(name: str) -> str:
    """Normalise a distribution or extra name the way PEP 503 does, with ``_`` as separator."""
    return re.sub(r"[-_.]+", "_", name.strip()).lower()


def _requirement(spec: str) -> tuple[str, tuple[str, ...]]:
    """Return the normalised distribution name and extras of a requirement string."""
    head = spec.split(";", 1)[0].strip()
    match = _REQUIREMENT_RE.match(head)
    if match is None:  # a URL or ``-e`` line: fail loudly rather than drop it from the allowed set
        raise ValueError(f"unsupported requirement in pyproject.toml: {spec!r}")
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
    """Return ``{attribute name: loader}`` for the ``LazyLoader`` registry.

    Keyed by the attribute, not by ``module_name``, so two handles for the same module are both seen.
    """
    return {name: loader for name, loader in vars(external).items() if isinstance(loader, LazyLoader)}


def _lazy_loader_modules() -> dict[str, str]:
    """Return ``{module name: top-level module name}`` for the ``LazyLoader`` registry."""
    return {loader.module_name: loader.module_name.split(".")[0] for loader in _lazy_loaders().values()}


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


def _optional_import_names() -> set[str]:
    """Return the top-level import names of every optional package: extras and ``LazyLoader`` modules."""
    runtime, extras = _pyproject_import_names()
    return (set().union(*extras.values()) | set(_lazy_loader_modules().values())) - runtime


def test_implicit_allowed_packages_are_installed_by_torch():
    """The premise behind :data:`IMPLICIT_ALLOWED` is executed, not narrated: torch must require them."""
    torch_requires = {_requirement(spec)[0] for spec in importlib.metadata.requires("torch") or ()}
    missing = sorted(name for name in IMPLICIT_ALLOWED if _normalise(name) not in torch_requires)
    assert not missing, f"torch no longer requires {missing}; declare them in pyproject.toml instead"


def test_every_third_party_import_is_declared():
    """Every third-party module imported under ``kornia/`` must be a declared dependency.

    Optional packages count as declared, but importing one directly is a defect of its own: it must
    go through the ``LazyLoader`` registry, or a user without the extra gets a bare
    ``ModuleNotFoundError`` instead of the ``pip install "kornia[<extra>]"`` line.
    """
    runtime, _ = _pyproject_import_names()
    optional = _optional_import_names()
    allowed = runtime | IMPLICIT_ALLOWED
    ignored = set(sys.stdlib_module_names) | {"kornia"}

    undeclared: list[str] = []
    direct_optional: list[str] = []
    for path, line, module in sorted(_imported_modules()):
        if module in ignored or module in allowed:
            continue
        (direct_optional if module in optional else undeclared).append(f"{path}:{line} -> {module}")

    assert not undeclared, (
        "kornia/ imports modules that no pyproject.toml runtime dependency, user-facing "
        "optional-dependency extra or LazyLoader declares. Declare them, make them optional "
        "through kornia.core.external, or drop the import:\n  " + "\n  ".join(undeclared)
    )
    assert not direct_optional, (
        "kornia/ imports optional packages directly; use the kornia.core.external LazyLoader handle "
        "instead, so a missing package fails with the extra to install:\n  " + "\n  ".join(direct_optional)
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
        if loader.extra is not None
        and loader.module_name.split(".")[0] not in extras.get(_normalise(loader.extra), set())
    }

    assert not wrong, (
        "kornia.core.external declares LazyLoader(s) whose extra= hint names no user-facing "
        "optional-dependency extra that installs the module: "
        + ", ".join(f"external.{name} -> kornia[{extra}]" for name, extra in sorted(wrong.items()))
        + f". User-facing extras: {sorted(extras)}."
    )


def test_import_kornia_does_not_load_optional_dependencies():
    """A bare ``import kornia`` must not import any optional dependency (kornia#4260).

    kornia#4260 was ``kornia.feature.lightglue_onnx`` importing onnxruntime at module level; every
    ONNX consumer now goes through the ``LazyLoader`` handles, and this probe keeps it that way for
    every optional package. The child interpreter records each import *attempt* of an optional
    top-level package made while ``import kornia`` runs, through a ``sys.meta_path`` finder installed
    after ``import torch`` (torch's own optional-package probes are not kornia's), so an eager
    ``try: import transformers / except ImportError: pass`` is caught whether or not
    ``transformers`` is installed here: what the check covers does not depend on which extras the
    environment happens to have. Packages that end up in ``sys.modules`` without going through the
    finder are reported too. ``cwd`` is the repo root so the probe imports this tree's ``kornia`` and
    not another checkout's editable install.
    """
    runtime, _ = _pyproject_import_names()
    checked = sorted((_optional_import_names() | AUDIT_EAGER_IMPORTS) - runtime)

    code = (
        "import sys\n"
        "import torch\n"
        "names = set(sys.argv[1:])\n"
        "attempted = set()\n"
        "class Recorder:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        top = fullname.partition('.')[0]\n"
        "        if top in names:\n"
        "            attempted.add(top)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Recorder())\n"
        "before = {m.partition('.')[0] for m in sys.modules}\n"
        "import kornia\n"
        "after = {m.partition('.')[0] for m in sys.modules}\n"
        "print('OPTIONAL_LOADED:' + ' '.join(sorted(attempted | ((after - before) & names))))\n"
    )
    # S603: no untrusted input -- this interpreter, a literal program, and module names derived from
    # pyproject.toml and the registry above. check=False so a crashed probe reports its stderr instead
    # of an opaque CalledProcessError; the return-code assertion keeps the test from passing vacuously.
    # stdin is closed and the call bounded so a LazyLoader that prompts during import (the default
    # InstallationMode.ASK) fails with an EOFError report instead of waiting on a terminal.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, *checked],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=300,
    )
    assert result.returncode == 0, f"`import kornia` failed in the probe subprocess:\n{result.stderr}"
    reports = [line for line in result.stdout.splitlines() if line.startswith("OPTIONAL_LOADED:")]
    assert len(reports) == 1, f"the probe printed no single report line; stdout was:\n{result.stdout}"
    loaded = reports[0].removeprefix("OPTIONAL_LOADED:").split()

    assert not loaded, (
        f"`import kornia` eagerly loaded optional dependencies: {loaded}. They must stay behind a "
        f"kornia.core.external.LazyLoader. Checked: {checked}."
    )

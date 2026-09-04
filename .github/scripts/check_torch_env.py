#!/usr/bin/env python3

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
"""Check the CI virtualenv against the torch build a matrix leg just installed.

``tests.yml`` builds its environment in two steps: ``uv sync`` installs the lockfile,
whose torch comes from PyPI, and the matrix leg then replaces *torch alone* with a build
from the CPU wheel index. Everything ``uv sync`` installed for the locked torch stays
behind, and the ``+cpu`` wheels declare a narrower dependency set than the PyPI ones --
no ``triton`` at all (checked for 2.5.1, 2.6.0, 2.9.1 and 2.14.0). The lockfile's triton
therefore survives the swap as an orphan belonging to a torch that is no longer
installed, and torch dispatches onto it: ``_is_triton_available()`` tests whether
``import triton`` succeeds, not whether that triton matches. torch 2.6 then runs
``from triton.backends.compiler import AttrsDescriptor``, a name triton 3.8 removed, and
32 dynamo and export tests die at import. See #4199.

This module is the single home of that explanation; the workflow comments point here.

Two modes:

``--list-undeclared-sidecars``
    Print the installed sidecar packages the installed torch does not declare, for the
    workflow to uninstall. These legs are CPU-only, so torch taking its non-triton path
    deliberately is the right outcome; an incompatible triton is worse than no triton.

``--channel CHANNEL --expected VERSION``
    Verify that the torch version satisfies the leg's channel, that every runtime
    requirement of *that* torch is installed and satisfied, and that no undeclared
    sidecar survived. A mis-provisioned venv fails here, at setup, instead of surfacing
    later as a wall of unrelated-looking test failures.

Machine-readable results go to stdout as ``key: value`` lines; diagnostics go to stderr.
A ``torch-version:`` line is printed only by a *passing* check, so the workflow can treat
its absence as failure rather than depending on the shell's pipefail setting.

Usage::

    python3 .github/scripts/check_torch_env.py --list-undeclared-sidecars
    python3 .github/scripts/check_torch_env.py --channel pinned --expected 2.6.0
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, distribution, distributions

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

# Packages torch selects at runtime by import availability rather than by version, so an
# orphan left over from a different torch is silently picked up instead of ignored. Both
# names install the same ``triton`` import; ``pytorch-triton`` is the nightly channel's.
#
# Deliberately an allowlist rather than a general "installed but outside the installed
# torch's closure" diff: most of the venv is legitimately outside that closure -- kornia's
# own dependencies, the test tooling -- and installed metadata records no provenance, so a
# diff cannot tell an orphan from a root. The swap's other survivors, the lockfile's
# ``nvidia-*`` and ``cuda-*`` wheels, cost runner disk but are inert on these legs because
# a ``+cpu`` torch never loads them. Add a name here when a new sidecar turns out to be
# dispatched on availability.
SIDECAR_PACKAGES = ("triton", "pytorch-triton")

# The stdout contract the workflow greps. Renaming either of these without updating
# ``tests.yml`` silently turns its steps into no-ops, so ``tests/scripts`` pins both.
SIDECAR_PREFIX = "undeclared-sidecars:"
VERSION_PREFIX = "torch-version:"


def runtime_requirements(
    requires: Iterable[str] | None, environment: Mapping[str, str] | None = None
) -> list[Requirement]:
    """Return the requirements of a distribution that apply to a plain, no-extra install.

    Args:
        requires: ``Requires-Dist`` entries, as ``importlib.metadata`` reports them.
        environment: Marker variables overriding the running interpreter's, for tests.

    Returns:
        The requirements whose markers hold, with the extras-only ones dropped.

    """
    markers = {"extra": ""}
    markers.update(environment or {})
    kept = []
    for entry in requires or ():
        requirement = Requirement(entry)
        if requirement.marker is not None and not requirement.marker.evaluate(markers):
            continue
        kept.append(requirement)
    return kept


def scan_installed() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return every installed distribution's version, and the names that resolve ambiguously.

    ``distributions()`` walks ``sys.path`` in order and the first match wins, which is what
    ``importlib.metadata.version`` and a plain ``import`` resolve to, so first-wins is the
    honest answer across path entries. Two ``.dist-info`` directories for one name *inside a
    single* path entry have no such order -- that is a half-written venv, exactly what an
    interrupted ``uv pip install --reinstall-package torch`` leaves behind -- so those are
    reported rather than settled by whichever the filesystem happened to yield first.

    Returns:
        The canonical-name-to-version map, and the names seen with disagreeing versions.

    """
    found: dict[str, str] = {}
    seen: dict[str, set[str]] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        canonical = canonicalize_name(name)
        found.setdefault(canonical, dist.version)
        seen.setdefault(canonical, set()).add(dist.version)
    return found, {name: sorted(versions) for name, versions in seen.items() if len(versions) > 1}


def installed_versions() -> dict[str, str]:
    """Map the canonical name of every installed distribution to its version."""
    return scan_installed()[0]


def undeclared_sidecars(
    requires: Iterable[str] | None,
    installed: Mapping[str, str],
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the installed sidecar packages that the installed torch does not declare."""
    declared = {canonicalize_name(item.name) for item in runtime_requirements(requires, environment)}
    return [name for name in map(canonicalize_name, SIDECAR_PACKAGES) if name in installed and name not in declared]


def check_torch_version(version: str, channel: str, expected: str) -> str | None:
    """Return a problem description when ``version`` does not satisfy the leg's channel.

    ``pinned`` demands exactly ``expected`` (the local ``+cpu`` segment is ignored);
    ``stable`` treats ``expected`` as a floor, since the upstream probe installs whatever
    the stable index resolves to today.
    """
    release = Version(version.partition("+")[0])
    if channel == "pinned":
        if release != Version(expected):
            return f"torch {version} is not the pinned {expected}"
    elif channel == "stable":
        if release < Version(expected):
            return f"torch {version} is below the {expected} floor of the stable channel"
    else:
        raise ValueError(f"unknown torch channel: {channel!r}")
    return None


def check_environment(
    torch_version: str,
    requires: Iterable[str] | None,
    installed: Mapping[str, str],
    channel: str,
    expected: str,
    environment: Mapping[str, str] | None = None,
    duplicates: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Return every way the environment disagrees with the torch that is installed.

    Args:
        torch_version: Version of the imported torch, ``+cpu`` local segment included.
        requires: ``Requires-Dist`` entries of the installed torch distribution.
        installed: Canonical distribution name to version, as ``installed_versions``.
        channel: ``pinned`` or ``stable``.
        expected: The matrix version -- exact under ``pinned``, a floor under ``stable``.
        environment: Marker variables overriding the running interpreter's, for tests.
        duplicates: Names visible with disagreeing versions, as ``scan_installed``.

    Returns:
        Human-readable problem descriptions; empty when the environment is coherent.

    """
    problems = []
    version_problem = check_torch_version(torch_version, channel, expected)
    if version_problem is not None:
        problems.append(version_problem)

    for requirement in runtime_requirements(requires, environment):
        name = canonicalize_name(requirement.name)
        version = installed.get(name)
        if version is None:
            problems.append(f"torch {torch_version} requires {requirement}, which is not installed")
        elif not requirement.specifier.contains(version, prereleases=True):
            problems.append(f"torch {torch_version} requires {requirement}, but {requirement.name} {version} is here")

    for name in undeclared_sidecars(requires, installed, environment):
        problems.append(
            f"{name} {installed[name]} is installed, but torch {torch_version} declares no dependency on it: "
            "torch dispatches onto an importable sidecar without checking its version (#4199)"
        )

    for name, versions in sorted((duplicates or {}).items()):
        problems.append(
            f"{name} is installed more than once, as {' and '.join(versions)}: the venv is half-written, "
            "and which one importlib.metadata reports is filesystem order rather than a decision"
        )
    return problems


def _torch_distribution() -> tuple[str, list[str] | None]:
    """Return the installed torch's runtime version and its ``Requires-Dist`` entries."""
    import torch

    dist = distribution("torch")
    try:
        recorded = Version(dist.version.partition("+")[0])
        running = Version(torch.__version__.partition("+")[0])
    except InvalidVersion:  # A source or nightly build; the metadata check is best-effort.
        return torch.__version__, dist.requires
    if recorded != running:
        raise SystemExit(f"torch imports as {torch.__version__} but its distribution metadata says {dist.version}")
    return torch.__version__, dist.requires


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested mode and return the process exit status."""
    parser = argparse.ArgumentParser(description="Check the CI venv against the installed torch.")
    parser.add_argument("--channel", choices=("pinned", "stable"), help="torch channel the leg installed with")
    parser.add_argument("--expected", help="matrix torch version: exact under pinned, a floor under stable")
    parser.add_argument(
        "--list-undeclared-sidecars",
        action="store_true",
        help="print the sidecar packages the installed torch does not declare, and exit",
    )
    args = parser.parse_args(argv)

    # Validated before any environment work: enumerating every distribution and importing
    # torch takes seconds and tells a caller who forgot a flag nothing they need.
    if not args.list_undeclared_sidecars and (args.channel is None or args.expected is None):
        parser.error("--channel and --expected are required unless --list-undeclared-sidecars is given")

    try:
        installed, duplicates = scan_installed()
        if args.list_undeclared_sidecars:
            stale = undeclared_sidecars(distribution("torch").requires, installed)
            print(f"{SIDECAR_PREFIX} {' '.join(stale)}".rstrip())
            return 0
        torch_version, requires = _torch_distribution()
    except PackageNotFoundError:
        print("torch is not installed in this environment", file=sys.stderr)
        return 1

    problems = check_environment(torch_version, requires, installed, args.channel, args.expected, duplicates=duplicates)
    if problems:
        # The requirement table is diagnostic, so it is printed where it diagnoses something
        # rather than as ~15 lines of noise on every healthy leg.
        for requirement in runtime_requirements(requires):
            name = canonicalize_name(requirement.name)
            print(f"  {name} {installed.get(name, '(missing)')}", file=sys.stderr)
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    # Only a passing check reports a version. Printing it on the failure path too would hand
    # the workflow a valid-looking answer from a mis-provisioned venv and leave its
    # "did the check report a version?" guard unable to ever fire.
    print(f"{VERSION_PREFIX} {torch_version.partition('+')[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

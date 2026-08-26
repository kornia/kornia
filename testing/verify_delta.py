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

"""Diff failing-test SETS between this branch and ``origin/main`` on every supported surface.

Run as ``pixi run verify-delta`` (``python -m testing.verify_delta``). Counts are never compared —
a branch can add and fix an equal number of tests and look unchanged by count.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

WIDEN_PREFIXES = ("testing/", "tests/conftest.py", "conftest.py", "pyproject.toml", "pixi.toml")


def changed_test_dirs(changed_files: Iterable[str], repo: Path | None = None) -> list[str]:
    """Map changed paths to the test directories that exercise them (``["tests"]`` when everything).

    Pass ``repo`` to check the mapping against the tree: a library module with no ``tests/<mod>``
    directory of its own (``kornia/transpiler/`` today) widens the run to the whole suite instead of
    mapping to a path that does not exist, which would otherwise leave the change unverified while
    the tool still reported a pass.
    """
    dirs: set[str] = set()
    for f in changed_files:
        if f.startswith(WIDEN_PREFIXES):
            return ["tests"]
        parts = Path(f).parts
        if parts[0] == "kornia":
            if len(parts) == 2:  # kornia/constants.py and friends have no dedicated test dir
                return ["tests"]
            mapped = f"tests/{parts[1]}"
            if repo is not None and not (repo / mapped).exists():
                print(f"{f} maps to {mapped}, which does not exist; widening to the whole suite")
                return ["tests"]
            dirs.add(mapped)
        elif parts[0] == "tests" and len(parts) > 2:
            dirs.add(f"tests/{parts[1]}")
    return sorted(dirs)


def failing_ids(junit_xml: str | Path) -> set[str]:
    """Return ``classname::name`` for every testcase with a ``failure`` or ``error`` child."""
    path = Path(junit_xml)
    if not path.exists():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()  # noqa: S314 -- locally generated pytest junit report, not untrusted input
    ids: set[str] = set()
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            ids.add(f"{case.get('classname')}::{case.get('name')}")
    return ids


@dataclass(frozen=True)
class FailureDelta:
    new: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    baseline: bool = True  # False when the base revision had no such tests, so every failure counts as new


def diff_failures(branch: set[str], main: set[str], *, baseline: bool = True) -> FailureDelta:
    """Split the two failing-test sets into the new, the fixed, and the already-failing ones."""
    return FailureDelta(
        new=sorted(branch - main),
        fixed=sorted(main - branch),
        unchanged=sorted(branch & main),
        baseline=baseline,
    )


def render_table(rows: Sequence[tuple[str, FailureDelta | None]]) -> str:
    """Render one markdown row per surface (``None`` means the surface was skipped) plus the NEW/FIXED ids."""
    lines = ["| surface | new | fixed | unchanged |", "|---|---|---|---|"]
    details: list[str] = []
    no_baseline = False
    for name, delta in rows:
        if delta is None:
            lines.append(f"| {name} | skipped | | |")
            continue
        # a run with no baseline is a real measurement, so it must not share the `skipped` label,
        # but its `new` count is unconditional and should not be read as a regression
        new_cell = f"{len(delta.new)}*" if not delta.baseline else str(len(delta.new))
        no_baseline |= not delta.baseline
        lines.append(f"| {name} | {new_cell} | {len(delta.fixed)} | {len(delta.unchanged)} |")
        details.extend(f"NEW {t}" for t in delta.new)
        details.extend(f"FIXED {t}" for t in delta.fixed)
    legend = ["", "* no baseline on the base revision for these paths; every failure there counts as new"]
    return "\n".join(lines + (legend if no_baseline else []) + ([""] + details if details else []))


@dataclass(frozen=True)
class Surface:
    """One test surface: a display name, the ``KORNIA_TEST_*`` environment it needs, and extra pytest args."""

    name: str
    env: dict[str, str]
    extra_args: tuple[str, ...] = ()


SURFACES: list[Surface] = [
    Surface("cpu float32", {"KORNIA_TEST_DEVICE": "cpu", "KORNIA_TEST_DTYPE": "float32"}),
    Surface(
        "cpu float16,bfloat16,float64",
        {"KORNIA_TEST_DEVICE": "cpu", "KORNIA_TEST_DTYPE": "float16,bfloat16,float64"},
    ),
    Surface("mps float32", {"KORNIA_TEST_DEVICE": "mps", "KORNIA_TEST_DTYPE": "float32"}),
    Surface(
        "inductor cpu float32",
        {"KORNIA_TEST_DEVICE": "cpu", "KORNIA_TEST_DTYPE": "float32", "KORNIA_TEST_OPTIMIZER": "inductor"},
        ("-k", "dynamo or compile"),
    ),
]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line of ``python -m testing.verify_delta``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--":  # `pixi run verify-delta -- ...` forwards the separator itself
        args = args[1:]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="origin/main", help="revision to compare against")
    p.add_argument("--tests", nargs="+", help="test dirs to run (default: derived from the diff)")
    p.add_argument("--only", nargs="+", help="surface names to run")
    p.add_argument("--main-worktree", type=Path, help="where to check out --base (default: a sibling dir)")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--out", type=Path, help="directory for junit files (default: <repo>/../.<repo>-verify-delta)")
    return p.parse_args(args)


def _resolve_only(tokens: Sequence[str]) -> list[str]:
    """Rejoin ``--only`` tokens into surface names, longest first, since a task shell drops the quoting."""
    names = [s.name for s in SURFACES]
    wanted: list[str] = []
    i = 0
    while i < len(tokens):
        for width in range(len(tokens) - i, 0, -1):
            candidate = " ".join(tokens[i : i + width])
            if candidate in names:
                wanted.append(candidate)
                i += width
                break
        else:
            raise SystemExit(f"unknown surface {tokens[i]!r}; choose from {names}")
    return wanted


def select_surfaces(args: argparse.Namespace, *, mps_available: bool) -> list[Surface]:
    """Keep the surfaces named by ``--only`` (all of them by default), dropping MPS when it is unavailable."""
    wanted = _resolve_only(args.only) if args.only else []
    chosen = [s for s in SURFACES if not wanted or s.name in wanted]
    return [s for s in chosen if s.name != "mps float32" or mps_available]


def _failures_or_empty(junit: Path) -> set[str]:
    """Read a junit report, treating a missing one (collection error, no tests) as zero failures."""
    if not Path(junit).exists():
        print(f"no junit report at {junit}; reading it as zero failures")
        return set()
    return failing_ids(junit)


def _git(repo: Path, *cmd: str) -> str:
    """Run a git command inside ``repo`` and return its stripped stdout."""
    return subprocess.run(  # noqa: S603 -- trusted: fixed argv, no shell
        ["git", "-C", str(repo), *cmd],  # noqa: S607 -- trusted: git resolved from PATH
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_common_dir(tree: Path) -> Path:
    """Return the shared ``.git`` directory backing ``tree``, which every worktree of a repo has in common."""
    return (tree / _git(tree, "rev-parse", "--git-common-dir")).resolve()


def _is_detached(tree: Path) -> bool:
    """Report whether ``tree`` has a detached HEAD; ``symbolic-ref`` exits non-zero exactly then."""
    try:
        _git(tree, "symbolic-ref", "-q", "HEAD")
    except subprocess.CalledProcessError:
        return True
    return False


def _primary_worktree(repo: Path) -> Path | None:
    """Return the repo's main (non-linked) worktree, the first ``git worktree list --porcelain`` entry."""
    lines = _git(repo, "worktree", "list", "--porcelain").splitlines()
    if not lines or not lines[0].startswith("worktree "):
        return None  # unrecognised output: leave the verdict to the detached-HEAD check
    return Path(lines[0][len("worktree ") :]).resolve()


def _ensure_main_worktree(repo: Path, base: str, path: Path, fetch: bool) -> None:
    """Create (or re-point) a detached worktree of ``base`` at ``path``."""
    remote, _, ref = base.partition("/")
    if fetch and ref:  # a local revision such as `--base HEAD~1` has nothing to fetch
        subprocess.run(  # noqa: S603 -- trusted: fixed argv, no shell
            ["git", "-C", str(repo), "fetch", remote, ref],  # noqa: S607 -- trusted: git resolved from PATH
            check=True,
        )
    if path.exists():
        # `git -C` walks up out of a plain directory, so a leftover non-worktree here would silently
        # check `base` out in whatever repository encloses it -- possibly the user's own checkout.
        try:
            same_repo = _git_common_dir(path) == _git_common_dir(repo)
        except subprocess.CalledProcessError:
            same_repo = False
        if not same_repo:
            raise SystemExit(f"{path} exists but is not a worktree of {repo}; remove it or pass --main-worktree")
        # a worktree of this repo is still not ours to detach: an attached HEAD or the repo's own
        # primary worktree is somebody's working copy, and checking `base` out there would move the
        # branch they are on out from under them.
        if not _is_detached(path) or path.resolve() == _primary_worktree(repo):
            raise SystemExit(
                f"{path} looks like a user checkout (its HEAD is on a branch, or it is the repo's main "
                f"worktree), not a scratch worktree; remove it or pass --main-worktree"
            )
        try:
            _git(path, "checkout", "--detach", base)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"{path} is dirty or cannot check out {base}: {(exc.stderr or '').strip()}") from exc
    else:
        _git(repo, "worktree", "add", "--detach", str(path), base)


def _assert_imports_from(tree: Path, env: dict[str, str]) -> None:
    """Refuse to run when ``import kornia`` in ``tree`` resolves to a checkout other than ``tree``."""
    out = subprocess.run(
        [sys.executable, "-c", "import kornia, pathlib; print(pathlib.Path(kornia.__file__).resolve())"],
        cwd=tree,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not Path(out).is_relative_to(tree.resolve()):
        raise SystemExit(f"{tree}: `import kornia` resolved to {out}; refusing to test the wrong tree")


def _run_surface(tree: Path, surface: Surface, tests: Sequence[str], junit: Path) -> set[str] | None:
    """Run one surface inside ``tree`` and return its failing-test ids, or ``None`` if pytest could not run."""
    # pytest aborts collection outright when any argument path is missing, so a directory that the base
    # revision does not have yet would empty its whole failure set and make every branch failure look new.
    present = [t for t in tests if (tree / t).exists()]
    missing = [t for t in tests if t not in present]
    if not present:
        print(f"{tree}: none of {' '.join(tests)} exist here; nothing to run")
        junit.unlink(missing_ok=True)
        return None
    if missing:
        print(f"{tree}: {' '.join(missing)} does not exist here; running {' '.join(present)}")
    env = {**os.environ, **surface.env, "PYTHONPATH": str(tree)}
    _assert_imports_from(tree, env)
    junit.unlink(missing_ok=True)  # a stale report from a previous run must never be read as this run's result
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--continue-on-collection-errors",
        f"--junitxml={junit}",
        *surface.extra_args,
        *present,
    ]
    subprocess.run(cmd, cwd=tree, env=env, check=False)  # noqa: S603 -- trusted: fixed argv, no shell
    return _failures_or_empty(junit)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every selected surface on both trees and print the failing-set delta."""
    args = parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    main_wt = args.main_worktree or repo.parent / f".{repo.name}-verify-main"
    out = args.out or repo.parent / f".{repo.name}-verify-delta"
    out.mkdir(parents=True, exist_ok=True)
    _ensure_main_worktree(repo, args.base, main_wt, fetch=not args.no_fetch)

    changed = _git(repo, "diff", "--name-only", f"{args.base}...HEAD").splitlines()
    tests = args.tests or changed_test_dirs(changed, repo)
    if not tests:
        print("no test directories map to the changed files; nothing to verify")
        return 0
    branch_rev = _git(repo, "rev-parse", "--short", "HEAD")
    base_rev = _git(main_wt, "rev-parse", "--short", "HEAD")
    print(f"branch {branch_rev} vs {args.base} {base_rev}")
    print(f"tests: {' '.join(tests)}")

    import torch

    selected = select_surfaces(args, mps_available=torch.backends.mps.is_available())
    rows: list[tuple[str, FailureDelta | None]] = []
    for surface in SURFACES:
        if surface not in selected:
            rows.append((surface.name, None))
            continue
        slug = surface.name.replace(" ", "_").replace(",", "-")
        branch_fail = _run_surface(repo, surface, tests, out / f"{slug}-branch.xml")
        if branch_fail is None:  # nothing to verify on this surface at all
            rows.append((surface.name, None))
            continue
        main_fail = _run_surface(main_wt, surface, tests, out / f"{slug}-main.xml")
        if main_fail is None:
            # a path the base revision does not have yet is a legitimately empty baseline, not a failure
            # to measure: the branch side really ran, and everything failing on it really is new.
            print(f"no baseline on {args.base} for {' '.join(tests)}; branch failures there are unconditionally new")
            rows.append((surface.name, diff_failures(branch_fail, set(), baseline=False)))
            continue
        rows.append((surface.name, diff_failures(branch_fail, main_fail)))
    table = render_table(rows)
    print("\n" + table)
    (out / "summary.md").write_text(table + "\n")
    if all(delta is None for _, delta in rows):
        print("nothing was verified: no surface ran pytest on the branch; this is not a pass")
        return 2
    return 1 if any(d is not None and d.new for _, d in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

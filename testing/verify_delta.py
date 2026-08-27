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

"""Diff failing-test SETS between this branch and ``--base`` (``origin/main``) on every supported surface.

Run as ``pixi run verify-delta`` (``python -m testing.verify_delta``). Counts are never compared —
a branch can add and fix an equal number of tests and look unchanged by count.

Exit codes: ``0`` no new failures on any surface that ran; ``1`` at least one ``new`` failure;
``2`` at least one selected, available surface could not be measured, so the gate is partial. A
surface excluded by ``--only`` (``not selected``) or absent from the machine (``unavailable``) is
not a gap and does not affect the code. The branch side is refused when the checkout is dirty —
the scope is ``base...HEAD`` but pytest runs the working tree, so the two must agree.
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
    """Map changed paths to the test targets that exercise them (``["tests"]`` when everything).

    A test file directly under ``tests/`` maps to itself: it has no package directory to stand for
    it, and a shared one there (``tests/__init__.py``, ``tests/benchmark.py``) widens to the suite.

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
        elif parts[0] == "tests":
            if len(parts) > 2:
                dirs.add(f"tests/{parts[1]}")
            elif Path(f).name.startswith("test_") or Path(f).name.endswith("_test.py"):
                # tests/test_import.py, tests/smoke_test.py: no package directory to map to, but the
                # file is itself a runnable pytest target. Dropping it left such a change unverified,
                # and a diff containing nothing else printed "nothing to verify" and exited 0.
                dirs.add(f)
            else:
                # tests/__init__.py, tests/benchmark.py: shared by the whole tree, like tests/conftest.py
                return ["tests"]
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


#: statuses a surface can carry instead of a measurement. Only ``UNVERIFIED`` is a failure: the other
#: two say the surface was never meant to run here, while ``UNVERIFIED`` says it was and did not.
NOT_SELECTED = "not selected"
UNAVAILABLE = "unavailable"
UNVERIFIED = "unverified"


def render_table(rows: Sequence[tuple[str, FailureDelta | str]]) -> str:
    """Render one markdown row per surface (a ``str`` in place of a delta is a status) plus the NEW/FIXED ids."""
    lines = ["| surface | new | fixed | unchanged |", "|---|---|---|---|"]
    details: list[str] = []
    no_baseline = False
    for name, delta in rows:
        if isinstance(delta, str):
            lines.append(f"| {name} | {delta} | | |")
            continue
        # a run with no baseline is a real measurement, so it must not share a status label,
        # but its `new` count is unconditional and should not be read as a regression
        new_cell = f"{len(delta.new)}*" if not delta.baseline else str(len(delta.new))
        no_baseline |= not delta.baseline
        lines.append(f"| {name} | {new_cell} | {len(delta.fixed)} | {len(delta.unchanged)} |")
        details.extend(f"NEW [{name}] {t}" for t in delta.new)
        details.extend(f"FIXED [{name}] {t}" for t in delta.fixed)
    legend: list[str] = []
    if no_baseline:
        legend.append("* no baseline on the base revision for these paths; every failure there counts as new")
    if any(delta == UNVERIFIED for _, delta in rows):
        legend.append(f"{UNVERIFIED}: selected and available, but could not be measured -- not a pass")
    return "\n".join(lines + ([""] + legend if legend else []) + ([""] + details if details else []))


# eq=False keeps these hashable: a `dict` field makes a frozen dataclass's generated __eq__/__hash__
# unusable (`unhashable type: 'dict'`), and identity is the right comparison for the fixed registry below.
@dataclass(frozen=True, eq=False)
class Surface:
    """One test surface: a display name, the ``KORNIA_TEST_*`` environment it needs, and extra pytest args."""

    name: str
    env: dict[str, str]
    extra_args: tuple[str, ...] = ()


MPS_SURFACE = "mps float32"

SURFACES: list[Surface] = [
    Surface("cpu float32", {"KORNIA_TEST_DEVICE": "cpu", "KORNIA_TEST_DTYPE": "float32"}),
    Surface(
        "cpu float16,bfloat16,float64",
        {"KORNIA_TEST_DEVICE": "cpu", "KORNIA_TEST_DTYPE": "float16,bfloat16,float64"},
    ),
    Surface(MPS_SURFACE, {"KORNIA_TEST_DEVICE": "mps", "KORNIA_TEST_DTYPE": "float32"}),
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
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="gate the working tree instead of HEAD (the default refuses an uncommitted change)",
    )
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
    return [s for s in chosen if s.name != MPS_SURFACE or mps_available]


def _present_paths(tree: Path, tests: Sequence[str]) -> list[str]:
    """Return the subset of ``tests`` that exists in ``tree``; pytest aborts collection on a missing path."""
    return [t for t in tests if (tree / t).exists()]


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


def _dirty(tree: Path) -> list[str]:
    """Return the ``git status --porcelain`` lines of ``tree``; empty when it has nothing uncommitted."""
    return [line for line in _git(tree, "status", "--porcelain").splitlines() if line.strip()]


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


def _ensure_main_worktree(repo: Path, base: str, path: Path, fetch: bool) -> str:
    """Create (or re-point) a detached worktree of ``base`` at ``path``; return the sha it resolved to."""
    remote, _, ref = base.partition("/")
    if fetch and ref:  # a local revision such as `--base HEAD~1` has nothing to fetch
        subprocess.run(  # noqa: S603 -- trusted: fixed argv, no shell
            ["git", "-C", str(repo), "fetch", remote, ref],  # noqa: S607 -- trusted: git resolved from PATH
            check=True,
        )
    # resolve against the repo, once. On a reused worktree `git -C <path> checkout --detach <base>`
    # would resolve a relative revision (`HEAD~1`) against the worktree's own HEAD -- the previous
    # run's base -- so it walks back one commit per run while `git diff <base>...HEAD` in the repo
    # does not, and the two trees quietly stop being the pair the table claims to compare.
    try:
        rev = _git(repo, "rev-parse", base)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{repo}: cannot resolve --base {base}: {(exc.stderr or '').strip()}") from exc
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
        # `checkout --detach` carries non-conflicting modifications across instead of refusing, so a
        # leftover edit here would be measured as part of the baseline and make a real branch failure
        # read as `unchanged`. Refuse rather than reset: the edit may be somebody's work.
        if dirty := _dirty(path):
            raise SystemExit(
                f"{path} has uncommitted changes and would contaminate the baseline:\n"
                + "\n".join(f"  {line}" for line in dirty[:10])
                + f"\nclean it (git -C {path} reset --hard && git -C {path} clean -fd) or pass --main-worktree"
            )
        try:
            _git(path, "checkout", "--detach", rev)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"{path} cannot check out {base}: {(exc.stderr or '').strip()}") from exc
    else:
        _git(repo, "worktree", "add", "--detach", str(path), rev)
    return rev


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
    present = _present_paths(tree, tests)
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
    proc = subprocess.run(cmd, cwd=tree, env=env, check=False)  # noqa: S603 -- trusted: fixed argv, no shell
    # 0 all passed, 1 tests failed, 5 nothing collected are the only codes where an absent or partial
    # report really does mean "no failures here". 2 interrupted, 3 internal error, 4 usage error (a
    # broken conftest, a bad -k) leave no junit at all, which would otherwise read as a clean surface.
    if proc.returncode not in (0, 1, 5):
        print(f"{tree}: pytest exited {proc.returncode}; this surface was not verified")
        junit.unlink(missing_ok=True)
        return None
    return _failures_or_empty(junit)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every selected surface on both trees, print the failing-set delta, and return an exit code.

    ``0`` clean, ``1`` a ``new`` failure somewhere, ``2`` a selected and available surface was never
    measured (see the module docstring).
    """
    args = parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    # the scope comes from `base...HEAD` but pytest runs the checkout, so an uncommitted change is
    # measured while the table names two commits: an unpushed fix can make a broken HEAD read green,
    # and an unpushed break can fail a HEAD that is fine.
    if (dirty := _dirty(repo)) and not args.allow_dirty:
        raise SystemExit(
            f"{repo} has uncommitted changes, so the run would not describe HEAD:\n"
            + "\n".join(f"  {line}" for line in dirty[:10])
            + "\ncommit or stash them, or pass --allow-dirty to gate the working tree instead"
        )
    main_wt = args.main_worktree or repo.parent / f".{repo.name}-verify-main"
    out = args.out or repo.parent / f".{repo.name}-verify-delta"
    out.mkdir(parents=True, exist_ok=True)
    base = _ensure_main_worktree(repo, args.base, main_wt, fetch=not args.no_fetch)

    changed = _git(repo, "diff", "--name-only", f"{base}...HEAD").splitlines()
    tests = args.tests or changed_test_dirs(changed, repo)
    if not tests:
        print("no test directories map to the changed files; nothing to verify")
        return 0
    branch_rev = _git(repo, "rev-parse", "--short", "HEAD")
    base_rev = _git(repo, "rev-parse", "--short", base)
    suffix = " + uncommitted changes (--allow-dirty)" if dirty else ""
    print(f"branch {branch_rev}{suffix} vs {args.base} {base_rev}")
    print(f"tests: {' '.join(tests)}")

    import torch

    mps_available = torch.backends.mps.is_available()
    selected = select_surfaces(args, mps_available=mps_available)
    branch_paths = _present_paths(repo, tests)
    base_paths = _present_paths(main_wt, tests)
    # a base tree holding only some of the branch's test paths is a *partial* baseline: failures under
    # a path it never ran are unconditionally new, so such a row earns the `*` an absent baseline gets.
    complete_baseline = set(branch_paths) <= set(base_paths)
    rows: list[tuple[str, FailureDelta | str]] = []
    for surface in SURFACES:
        if surface not in selected:
            # a surface nobody asked for, or one this machine does not have, is not a gap in the gate;
            # a surface that *was* asked for and did not run is, so the two must not share a label.
            absent = surface.name == MPS_SURFACE and not mps_available
            rows.append((surface.name, UNAVAILABLE if absent else NOT_SELECTED))
            continue
        slug = surface.name.replace(" ", "_").replace(",", "-")
        branch_fail = _run_surface(repo, surface, tests, out / f"{slug}-branch.xml")
        if branch_fail is None:  # nothing to verify on this surface at all
            rows.append((surface.name, UNVERIFIED))
            continue
        main_fail = _run_surface(main_wt, surface, tests, out / f"{slug}-main.xml")
        if main_fail is None and not base_paths:
            # a path the base revision does not have yet is a legitimately empty baseline, not a failure
            # to measure: the branch side really ran, and everything failing on it really is new.
            print(f"no baseline on {args.base} for {' '.join(tests)}; branch failures there are unconditionally new")
            rows.append((surface.name, diff_failures(branch_fail, set(), baseline=False)))
            continue
        if main_fail is None:
            # the base tree has the paths but pytest did not finish there; reading that as an empty
            # baseline would report every branch failure as new, the false positive `*` exists for.
            print(f"{surface.name}: the base tree did not finish, so this surface was not verified")
            rows.append((surface.name, UNVERIFIED))
            continue
        rows.append((surface.name, diff_failures(branch_fail, main_fail, baseline=complete_baseline)))
    table = render_table(rows)
    print("\n" + table)
    (out / "summary.md").write_text(table + "\n")
    measured = [name for name, delta in rows if isinstance(delta, FailureDelta)]
    unverified = [name for name, delta in rows if delta == UNVERIFIED]
    if any(isinstance(d, FailureDelta) and d.new for _, d in rows):
        return 1  # named by the NEW lines above; the loudest signal, so it wins over an unverified row
    if not measured:
        print("nothing was verified: no surface ran pytest on the branch; this is not a pass")
        return 2
    if unverified:
        # one measured surface does not cover for another: a gate with a hole is not a pass, and the
        # holes have to be nonzero or a crash on half/MPS reads exactly like a clean run.
        print(f"not verified on {', '.join(unverified)}; the gate is partial, which is not a pass")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
"""Check changelog fragment names and require one for each pull request.

Towncrier builds the fragments in a separate CI step. This check keeps its
input predictable and makes the per-PR policy explicit: a normal PR must add
or update a fragment named for its own number and leave the assembled changelog
to release PRs. The ``no-changelog`` label is reserved for internal work and
release PRs; it waives those two rules. It never permits malformed fragments.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

FRAGMENT_DIR = Path("changelog.d")
README = "README.md"
IGNORED_FILES = {".DS_Store", README}
FRAGMENT_NAME = re.compile(r"(?:[0-9]+|\+migration-[0-9]+)\.(?:added|fixed|breaking)\.md\Z")


def fragment_paths(root: Path = Path(".")) -> list[Path]:
    """Return every fragment candidate, excluding the contributor guide."""
    directory = root / FRAGMENT_DIR
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.name not in IGNORED_FILES)


def validate_fragments(root: Path = Path(".")) -> list[str]:
    """Return policy violations for the complete fragment directory."""
    errors: list[str] = []
    directory = root / FRAGMENT_DIR
    if not directory.is_dir():
        return [f"{FRAGMENT_DIR} is missing"]

    for path in fragment_paths(root):
        relative = path.relative_to(root)
        if not path.is_file() or not FRAGMENT_NAME.fullmatch(path.name):
            errors.append(
                f"{relative}: fragment names must be <PR>.(added|fixed|breaking).md; "
                "only reserved +migration-<number> files may use an orphan prefix"
            )
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{relative}: fragment must be UTF-8 text")
            continue
        if not content.strip():
            errors.append(f"{relative}: fragment must not be empty")
    return errors


def changed_paths(base_ref: str) -> set[Path]:
    """Return all paths changed since the PR merge base, including deletions."""
    merge_base = subprocess.run(  # noqa: S603
        ["git", "merge-base", base_ref, "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(  # noqa: S603
        ["git", "diff", "-M", "--name-only", "-z", f"{merge_base}..HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {Path(path) for path in diff.split("\0") if path}


def has_current_pr_fragment(paths: set[Path], pr_number: int) -> bool:
    """Whether this PR changed a valid fragment that credits itself."""
    prefix = f"{pr_number}."
    return any(
        path.parent == FRAGMENT_DIR
        and path.is_file()
        and path.name.startswith(prefix)
        and FRAGMENT_NAME.fullmatch(path.name)
        for path in paths
    )


def main(argv: list[str] | None = None) -> int:
    """Run the changelog policy check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True, help="Base branch ref, such as origin/main")
    parser.add_argument("--pr-number", required=True, type=int, help="Current pull request number")
    parser.add_argument(
        "--skip-presence",
        action="store_true",
        help="Allow an internal or release PR to omit its fragment and edit CHANGELOG.md",
    )
    args = parser.parse_args(argv)

    errors = validate_fragments()
    if not args.skip_presence:
        changed = changed_paths(args.base_ref)
        if not has_current_pr_fragment(changed, args.pr_number):
            errors.append(
                f"no changed changelog.d/{args.pr_number}.(added|fixed|breaking).md fragment found; "
                "add one for a user-visible change or apply the no-changelog label for an internal or release PR"
            )
        if Path("CHANGELOG.md") in changed:
            errors.append("ordinary PRs must not edit CHANGELOG.md; assemble it only in a labeled release PR")

    for error in errors:
        print(f"::error::{error}")
    if errors:
        return 1
    print("Changelog fragments are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

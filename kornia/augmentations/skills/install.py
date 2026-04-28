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

"""kornia-augmentations skill installer.

Run:
    python -m kornia.augmentations.skills.install
    python -m kornia.augmentations.skills.install --target /custom/path

Default target: ~/.claude/skills/kornia-augmentations/SKILL.md (Claude Code).
The skill activates automatically once copied; no further configuration.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

SKILL_FILENAME_SRC = "kornia-augmentations.md"
SKILL_FILENAME_DST = "SKILL.md"


def _default_target() -> Path:
    """Resolve the default Claude Code skills path."""
    home = Path(os.path.expanduser("~"))
    return home / ".claude" / "skills" / "kornia-augmentations" / SKILL_FILENAME_DST


def _src_path() -> Path:
    return Path(__file__).parent / SKILL_FILENAME_SRC


def install(target: Path | None = None, *, force: bool = False, quiet: bool = False) -> Path:
    """Copy the skill markdown to the target path. Returns the final path."""
    src = _src_path()
    if not src.is_file():
        raise FileNotFoundError(f"skill source not found at {src}")

    dst = target or _default_target()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        if not quiet:
            print(f"skill already installed at {dst} (use --force to overwrite)")
        return dst
    shutil.copy2(src, dst)
    if not quiet:
        print(f"installed kornia-augmentations skill -> {dst}")
        print("the skill activates automatically in any Claude Code / Cursor / Copilot CLI session")
    return dst


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m kornia.augmentations.skills.install")
    p.add_argument(
        "--target",
        type=Path,
        default=None,
        help="destination path; defaults to ~/.claude/skills/kornia-augmentations/SKILL.md",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing installation")
    p.add_argument("--quiet", action="store_true", help="suppress output")
    args = p.parse_args()
    try:
        install(target=args.target, force=args.force, quiet=args.quiet)
    except Exception as e:
        print(f"install failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Script to fix docstring indentation warnings from griffe.

Fixes continuation lines that have 6 spaces when they should have 8 spaces.
The pattern is: continuation lines in docstrings should be indented 8 spaces
from the docstring start, but some have only 6 spaces.
"""

import sys
from pathlib import Path


def fix_file_docstrings(file_path: Path) -> int:
    """Fix docstring indentation in a Python file.

    Returns the number of fixes made.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0

    lines = content.split("\n")
    fixed_lines = []
    fixes_count = 0
    in_triple_quotes = False
    quote_type = None
    docstring_base_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        original_line = line

        # Detect start/end of docstring
        if '"""' in line:
            if not in_triple_quotes:
                in_triple_quotes = True
                quote_type = '"""'
                # Find the base indentation of the docstring
                stripped = line.lstrip()
                docstring_base_indent = len(line) - len(stripped)
            else:
                in_triple_quotes = False
                quote_type = None
                docstring_base_indent = 0
            fixed_lines.append(line)
            i += 1
            continue

        if "'''" in line:
            if not in_triple_quotes:
                in_triple_quotes = True
                quote_type = "'''"
                stripped = line.lstrip()
                docstring_base_indent = len(line) - len(stripped)
            else:
                in_triple_quotes = False
                quote_type = None
                docstring_base_indent = 0
            fixed_lines.append(line)
            i += 1
            continue

        if in_triple_quotes:
            stripped = line.lstrip()

            # Skip empty lines and lines that are just quotes
            if not stripped or stripped.startswith(('"""', "'''")):
                fixed_lines.append(line)
                i += 1
                continue

            current_indent = len(line) - len(stripped)
            expected_indent = docstring_base_indent

            # Check if this is a continuation line
            # Continuation lines should be indented 8 spaces from docstring start
            # But we're seeing 6 spaces instead
            if current_indent == docstring_base_indent + 6:
                # Always fix 6-space indentation to 8-space for continuation lines
                # This matches the griffe warning pattern
                new_indent = docstring_base_indent + 8
                new_line = " " * new_indent + stripped
                fixed_lines.append(new_line)
                fixes_count += 1
            elif current_indent == docstring_base_indent + 5:
                # Sometimes we see 5 spaces instead of 8
                new_indent = docstring_base_indent + 8
                new_line = " " * new_indent + stripped
                fixed_lines.append(new_line)
                fixes_count += 1
            elif current_indent == docstring_base_indent + 7:
                # Sometimes we see 7 spaces instead of 8
                new_indent = docstring_base_indent + 8
                new_line = " " * new_indent + stripped
                fixed_lines.append(new_line)
                fixes_count += 1
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

        i += 1

    if fixes_count > 0:
        new_content = "\n".join(fixed_lines)
        file_path.write_text(new_content, encoding="utf-8")
        print(f"Fixed {file_path}: {fixes_count} indentation issues")

    return fixes_count


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path(__file__).parent.parent / "kornia"

    if not target.exists():
        print(f"Error: {target} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all Python files
    if target.is_file():
        files = [target]
    else:
        files = list(target.rglob("*.py"))

    # Exclude test files and other directories
    files = [
        f
        for f in files
        if "test" not in str(f)
        and "__pycache__" not in str(f)
        and ".pixi" not in str(f)
        and "venv" not in str(f)
        and "benchmarks" not in str(f)
        and "testing" not in str(f)
    ]

    total_fixes = 0

    print(f"Processing {len(files)} files...")
    for file_path in sorted(files):
        fixes = fix_file_docstrings(file_path)
        total_fixes += fixes

    print(f"\nSummary: Fixed {total_fixes} indentation issues across {len(files)} files")


if __name__ == "__main__":
    main()

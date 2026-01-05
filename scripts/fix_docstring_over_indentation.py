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

"""Script to fix over-indented docstrings in the kornia library.

Fixes docstrings where content is over-indented by 4 spaces:
- Content lines at 8 spaces should be at 4 spaces
- Section headers (Args, Returns, etc.) at 12 spaces should be at 8 spaces
- Nested content at 16 spaces should be at 12 spaces
"""

import sys
from pathlib import Path


def find_docstring_ranges(lines):
    """Find all docstring ranges in the file.

    Returns list of (start_line, end_line, base_indent, quote_type) tuples.
    """
    docstrings = []
    in_triple_quotes = False
    quote_type = None
    start_line = None
    base_indent = 0

    for i, line in enumerate(lines):
        quote_count = line.count('"""') + line.count("'''")
        if quote_count > 0:
            if not in_triple_quotes:
                # Starting a docstring
                in_triple_quotes = True
                if '"""' in line:
                    quote_type = '"""'
                elif "'''" in line:
                    quote_type = "'''"
                stripped = line.lstrip()
                base_indent = len(line) - len(stripped)
                start_line = i
            # Check if this line closes the docstring
            elif quote_count % 2 == 1:
                docstrings.append((start_line, i, base_indent, quote_type))
                in_triple_quotes = False
                quote_type = None
                start_line = None
                base_indent = 0

    return docstrings


def fix_docstring_indentation(lines, start_line, end_line, base_indent):
    """Fix indentation for a single docstring.

    Returns number of fixes made.
    """
    fixes_count = 0
    content_indents = []

    # First pass: find the first non-empty content line to determine if docstring is over-indented
    first_content_indent = None
    content_indents = []

    for i in range(start_line + 1, end_line):
        line = lines[i]
        stripped = line.lstrip()

        # Skip empty lines and quote-only lines
        if not stripped or stripped.strip() in ('"""', "'''"):
            continue

        current_indent = len(line) - len(stripped)
        if current_indent > base_indent:
            if first_content_indent is None:
                first_content_indent = current_indent
            content_indents.append(current_indent)

    if not content_indents:
        return 0

    # Only fix if the first content line is over-indented (at base+4 or more)
    # This prevents breaking correctly formatted docstrings
    if first_content_indent is None or first_content_indent < base_indent + 4:
        return 0

    # Find minimum content indentation
    min_content_indent = min(content_indents)

    # Content should start at base_indent (same as opening quote)
    # If first content is at base+4 or more, reduce everything by 4
    if min_content_indent >= base_indent + 4:
        reduction = 4
        # Second pass: fix indentation
        for i in range(start_line + 1, end_line):
            line = lines[i]
            stripped = line.lstrip()

            # Skip empty lines
            if not stripped:
                continue

            # Skip lines that are just quotes
            if stripped.strip() in ('"""', "'''"):
                continue

            current_indent = len(line) - len(stripped)
            # Reduce all content lines by the reduction amount
            # This preserves relative indentation structure
            if current_indent > base_indent:
                new_indent = max(base_indent, current_indent - reduction)
                if new_indent != current_indent:
                    lines[i] = " " * new_indent + stripped
                    fixes_count += 1

    return fixes_count


def fix_file_docstrings(file_path: Path) -> int:
    """Fix over-indented docstrings in a Python file.

    Returns the number of fixes made.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0

    lines = content.split("\n")
    docstrings = find_docstring_ranges(lines)

    total_fixes = 0
    # Process each docstring and fix iteratively until no more fixes
    max_passes = 10
    for pass_num in range(max_passes):
        pass_fixes = 0
        # Re-find docstrings in case line numbers changed (they shouldn't, but be safe)
        docstrings = find_docstring_ranges(lines)
        for start_line, end_line, base_indent, quote_type in docstrings:
            fixes = fix_docstring_indentation(lines, start_line, end_line, base_indent)
            pass_fixes += fixes

        if pass_fixes == 0:
            break

        total_fixes += pass_fixes

    if total_fixes > 0:
        new_content = "\n".join(lines)
        file_path.write_text(new_content, encoding="utf-8")
        print(
            f"Fixed {file_path}: {total_fixes} indentation issues ({pass_num + 1} pass{'es' if pass_num > 0 else ''})"
        )

    return total_fixes


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
    files_fixed = 0

    print(f"Processing {len(files)} files...")
    for file_path in sorted(files):
        fixes = fix_file_docstrings(file_path)
        if fixes > 0:
            files_fixed += 1
        total_fixes += fixes

    print(f"\nSummary: Fixed {total_fixes} indentation issues across {files_fixed} files")


if __name__ == "__main__":
    main()

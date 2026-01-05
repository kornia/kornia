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

"""Script to automatically fix griffe warnings about docstring formatting.

Fixes:
1. Indentation issues (6 spaces -> 8 spaces for continuation lines)
2. Missing return type annotations (adds :rtype: if missing)
"""

import sys
from pathlib import Path
from typing import Tuple


def fix_docstring_indentation(content: str) -> Tuple[str, int]:
    """Fix indentation issues in docstrings.

    Changes continuation lines from 6 spaces to 8 spaces where appropriate.
    """
    lines = content.split("\n")
    fixed_lines = []
    fixes_count = 0
    in_docstring = False
    docstring_indent = 0

    for i, line in enumerate(lines):
        # Detect start of docstring
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                # Calculate base indentation of docstring
                docstring_indent = len(line) - len(line.lstrip())
            else:
                in_docstring = False
                docstring_indent = 0
            fixed_lines.append(line)
            continue

        if in_docstring:
            # Check if this is a continuation line that should have 8 spaces
            # but has 6 spaces instead
            stripped = line.lstrip()
            if stripped and not stripped.startswith(('"""', "'''")):
                current_indent = len(line) - len(stripped)
                # If we have 6 spaces of indentation relative to docstring start
                # and it's a continuation line, fix it
                if current_indent == docstring_indent + 6:
                    # Check if previous line suggests this is a continuation
                    if i > 0 and lines[i - 1].strip() and not lines[i - 1].strip().startswith(('"""', "'''")):
                        # Replace 6 spaces with 8 spaces
                        new_line = " " * (docstring_indent + 8) + stripped
                        fixed_lines.append(new_line)
                        fixes_count += 1
                        continue

        fixed_lines.append(line)

    return "\n".join(fixed_lines), fixes_count


def fix_missing_return_annotations(content: str) -> Tuple[str, int]:
    """Add missing return type annotations to docstrings.

    This is a placeholder - actual implementation would need to parse
    function signatures and docstrings more carefully.
    """
    # This is complex and would require AST parsing
    # For now, we'll focus on indentation fixes
    return content, 0


def process_file(file_path: Path) -> Tuple[int, int]:
    """Process a single Python file and fix docstring issues."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Fix indentation
        content, indent_fixes = fix_docstring_indentation(content)

        # Fix return annotations (placeholder)
        content, return_fixes = fix_missing_return_annotations(content)

        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            print(f"Fixed {file_path}: {indent_fixes} indentation fixes")
            return indent_fixes, return_fixes
        return 0, 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return 0, 0


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
        if "test" not in str(f) and "__pycache__" not in str(f) and ".pixi" not in str(f) and "venv" not in str(f)
    ]

    total_indent_fixes = 0
    total_return_fixes = 0

    print(f"Processing {len(files)} files...")
    for file_path in files:
        indent_fixes, return_fixes = process_file(file_path)
        total_indent_fixes += indent_fixes
        total_return_fixes += return_fixes

    print("\nSummary:")
    print(f"  Indentation fixes: {total_indent_fixes}")
    print(f"  Return annotation fixes: {total_return_fixes}")
    print(f"  Total fixes: {total_indent_fixes + total_return_fixes}")


if __name__ == "__main__":
    main()

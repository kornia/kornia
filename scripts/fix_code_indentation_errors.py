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

"""Script to fix indentation errors introduced by the docstring fix script.

The docstring fix script incorrectly reduced indentation of code blocks.
This script fixes those errors by restoring proper indentation for code
inside if/for/while/function/class blocks.
"""

import ast
import sys
from pathlib import Path


def fix_file_indentation(file_path: Path) -> int:
    """Fix indentation errors in a Python file.

    Returns number of fixes made.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0

    # Try to parse the file to find syntax errors
    try:
        ast.parse(content)
        # No syntax errors, skip
        return 0
    except SyntaxError:
        # Has syntax errors, need to fix
        pass

    lines = content.split("\n")
    fixed_lines = []
    fixes_count = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Check if this is a control structure that needs a body
        if stripped.startswith(("if ", "elif ", "else:", "for ", "while ", "try:", "except ", "finally:")):
            indent = len(line) - len(stripped)
            # Check if next non-empty line is incorrectly indented
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            if j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.lstrip()
                next_indent = len(next_line) - len(next_stripped)

                # If next line should be indented but isn't (or is at same level as control)
                if next_stripped and not next_stripped.startswith("#"):
                    expected_indent = indent + 4
                    if next_indent <= indent:
                        # Fix the indentation
                        fixed_lines.append(line)
                        # Fix all following lines until we hit a dedent
                        while j < len(lines):
                            fix_line = lines[j]
                            fix_stripped = fix_line.lstrip()

                            if not fix_stripped:
                                fixed_lines.append(fix_line)
                                j += 1
                                continue

                            if fix_stripped.startswith("#"):
                                fixed_lines.append(fix_line)
                                j += 1
                                continue

                            fix_indent = len(fix_line) - len(fix_stripped)

                            # If this line is at the wrong indent level, fix it
                            if fix_indent <= indent:
                                # This should be indented
                                new_indent = expected_indent
                                fixed_lines.append(" " * new_indent + fix_stripped)
                                fixes_count += 1
                                j += 1
                            else:
                                # This line is properly indented or dedented, we're done
                                break

                        i = j
                        continue

        # Check for function/class definitions
        if stripped.startswith(("def ", "class ")) and ":" in stripped:
            indent = len(line) - len(stripped)
            # Check if next non-empty line is incorrectly indented
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            if j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.lstrip()

                if next_stripped and not next_stripped.startswith(('"""', "'''", "#")):
                    next_indent = len(next_line) - len(next_stripped)
                    expected_indent = indent + 4

                    if next_indent <= indent:
                        # Fix the indentation
                        fixed_lines.append(line)
                        # Fix all following lines until we hit a dedent
                        while j < len(lines):
                            fix_line = lines[j]
                            fix_stripped = fix_line.lstrip()

                            if not fix_stripped:
                                fixed_lines.append(fix_line)
                                j += 1
                                continue

                            if fix_stripped.startswith(('"""', "'''", "#")):
                                fixed_lines.append(fix_line)
                                j += 1
                                continue

                            fix_indent = len(fix_line) - len(fix_stripped)

                            # If this line is at the wrong indent level, fix it
                            if fix_indent <= indent:
                                # This should be indented
                                new_indent = expected_indent
                                fixed_lines.append(" " * new_indent + fix_stripped)
                                fixes_count += 1
                                j += 1
                            else:
                                # This line is properly indented or dedented, we're done
                                break

                        i = j
                        continue

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
    files_fixed = 0

    print(f"Processing {len(files)} files...")
    for file_path in sorted(files):
        fixes = fix_file_indentation(file_path)
        if fixes > 0:
            files_fixed += 1
        total_fixes += fixes

    print(f"\nSummary: Fixed {total_fixes} indentation issues across {files_fixed} files")


if __name__ == "__main__":
    main()

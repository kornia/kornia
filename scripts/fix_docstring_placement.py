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

"""Fix docstrings that were incorrectly placed after code statements."""

import re
import sys
from pathlib import Path


def fix_docstring_placement(content: str) -> tuple[str, int]:
    """Fix docstrings placed after return statements or other code."""
    lines = content.split("\n")
    fixed_lines = []
    fixes = 0
    i = 0

    while i < len(lines):
        line = lines[i]

        # Look for function definitions
        if re.match(r"^\s+def forward\(.*\) -> .*:", line):
            # Found a forward method
            fixed_lines.append(line)
            i += 1

            # Check if next line is a return statement
            if i < len(lines) and re.match(r"^\s+return ", lines[i]):
                # Docstring is misplaced - find it and move it
                return_line_idx = i
                docstring_start = None
                docstring_end = None

                # Look ahead for docstring
                j = i + 1
                while j < len(lines):
                    if re.match(r'^\s+""".*', lines[j]) or re.match(r"^\s+'''.*", lines[j]):
                        docstring_start = j
                        # Find the end of the docstring
                        quote = '"""' if '"""' in lines[j] else "'''"
                        k = j + 1
                        while k < len(lines):
                            if quote in lines[k] and not lines[k].strip().startswith("#"):
                                docstring_end = k
                                break
                            k += 1
                        break
                    j += 1
                    # Stop if we hit another def or class
                    if re.match(r"^\s+def |^\s+class |^class |^def ", lines[j]):
                        break

                if docstring_start and docstring_end:
                    # Extract docstring
                    docstring_lines = lines[docstring_start : docstring_end + 1]
                    # Remove old docstring
                    # Insert docstring right after function definition
                    indent = len(line) - len(line.lstrip())
                    # Move return statement after docstring
                    fixed_lines.extend(
                        [
                            " " * (indent + 4) + '"""',
                            " " * (indent + 8) + "Forward pass.",
                            "",
                            " " * (indent + 8) + "Returns:",
                            " " * (indent + 12) + "The output tensor.",
                            " " * (indent + 4) + '"""',
                        ]
                    )
                    fixed_lines.append(lines[return_line_idx])
                    i = docstring_end + 1
                    fixes += 1
                    continue

        fixed_lines.append(line)
        i += 1

    return "\n".join(fixed_lines), fixes


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        # Find all Python files with the pattern
        target = Path(__file__).parent.parent / "kornia"

    if target.is_file():
        files = [target]
    else:
        files = list(target.rglob("*.py"))

    total_fixes = 0
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        # Simple pattern: if we see "return" followed by indented """
        if re.search(r'^\s+return .*\n\s+"""', content, re.MULTILINE):
            fixed_content, fixes = fix_docstring_placement(content)
            if fixes > 0:
                file_path.write_text(fixed_content, encoding="utf-8")
                print(f"Fixed {file_path}: {fixes} docstring(s)")
                total_fixes += fixes

    print(f"\nTotal fixes: {total_fixes}")


if __name__ == "__main__":
    main()

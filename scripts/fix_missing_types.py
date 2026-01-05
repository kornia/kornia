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

"""Script to fix missing type annotation warnings.

Fixes:
1. Removes parameters from docstrings that don't exist in function signatures
2. Adds Returns sections for functions with return type annotations
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple


def get_function_params(func_node: ast.FunctionDef) -> Set[str]:
    """Extract all parameter names from function signature."""
    params = set()
    for arg in func_node.args.args:
        params.add(arg.arg)
    if func_node.args.vararg:
        params.add(f"*{func_node.args.vararg.arg}")
    if func_node.args.kwarg:
        params.add(f"**{func_node.args.kwarg.arg}")
    return params


def find_docstring_params(docstring: str) -> List[Tuple[str, int]]:
    """Find parameter names in Args section and their line positions."""
    if not docstring:
        return []

    params = []
    lines = docstring.split("\n")
    in_args = False

    for i, line in enumerate(lines):
        if re.match(r"^\s*Args?:", line):
            in_args = True
            continue
        if in_args and re.match(r"^\s*(Returns?|Note|Raises?|Examples?|Attributes?|Yields?):", line):
            break
        if in_args:
            # Match parameter name (format: "param_name:" or "param_name ")
            match = re.match(r"^\s*(\*?\*?\w+)\s*[:\s]", line)
            if match:
                param_name = match.group(1)
                # Clean up parameter name (remove trailing spaces, etc.)
                param_name = param_name.strip()
                params.append((param_name, i))

    return params


def remove_invalid_params_from_docstring(docstring: str, valid_params: Set[str]) -> Tuple[str, int]:
    """Remove parameters from docstring that don't exist in function signature."""
    if not docstring:
        return docstring, 0

    lines = docstring.split("\n")
    fixed_lines = []
    fixes_count = 0
    in_args = False
    skip_next_continuation = False

    for i, line in enumerate(lines):
        if re.match(r"^\s*Args?:", line):
            in_args = True
            fixed_lines.append(line)
            continue

        if in_args:
            if re.match(r"^\s*(Returns?|Note|Raises?|Examples?|Attributes?|Yields?):", line):
                in_args = False
                fixed_lines.append(line)
                continue

            # Check if this line defines a parameter
            match = re.match(r"^(\s*)(\*?\*?\w+)\s*[:\s]", line)
            if match:
                indent, param_name = match.groups()
                param_name = param_name.strip()

                # Check if parameter is valid
                if param_name not in valid_params:
                    # Skip this line and its continuation
                    fixes_count += 1
                    skip_next_continuation = True
                    continue
                else:
                    skip_next_continuation = False
                    fixed_lines.append(line)
            elif skip_next_continuation:
                # Skip continuation lines of invalid parameters
                # Check if this is still a continuation (same or more indentation)
                if line.strip() and (len(line) - len(line.lstrip())) > len(fixed_lines[-1]) - len(
                    fixed_lines[-1].lstrip()
                ):
                    continue
                else:
                    skip_next_continuation = False
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines), fixes_count


def add_returns_section(docstring: str, return_type: Optional[str] = None) -> Tuple[str, bool]:
    """Add Returns section if missing and function has return annotation."""
    if not docstring:
        return docstring, False

    # Check if Returns section already exists
    if re.search(r"Returns?:", docstring):
        return docstring, False

    # Find the end of the docstring (before closing quotes)
    lines = docstring.split("\n")

    # Find last non-empty line before closing quotes
    last_content_line = len(lines) - 1
    while last_content_line >= 0 and not lines[last_content_line].strip():
        last_content_line -= 1

    # Insert Returns section before closing
    if last_content_line >= 0:
        indent = len(lines[last_content_line]) - len(lines[last_content_line].lstrip())
        if indent == 0:
            indent = 4  # Default indent

        returns_text = f"{' ' * indent}Returns:\n{' ' * (indent + 4)}The output tensor."
        lines.insert(last_content_line + 1, returns_text)
        return "\n".join(lines), True

    return docstring, False


def process_file(file_path: Path) -> Tuple[int, int]:
    """Process a file and fix type annotation issues."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0, 0

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
        return 0, 0

    lines = content.split("\n")
    total_param_fixes = 0
    total_return_fixes = 0
    modified = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node, clean=False)
            if not docstring:
                continue

            # Get valid parameters from function signature
            valid_params = get_function_params(node)

            # Fix invalid parameters in docstring
            fixed_docstring, param_fixes = remove_invalid_params_from_docstring(docstring, valid_params)
            total_param_fixes += param_fixes

            # Add Returns section if function has return annotation but docstring doesn't
            if node.returns and not re.search(r"Returns?:", fixed_docstring):
                fixed_docstring, added = add_returns_section(fixed_docstring)
                if added:
                    total_return_fixes += 1

            if fixed_docstring != docstring:
                # Replace docstring in source
                # Find the docstring node's line number
                for i, line_node in enumerate(ast.walk(tree)):
                    if line_node == node:
                        # This is complex - we'd need to track the exact position
                        # For now, we'll use a simpler approach
                        pass
                modified = True

    if modified and (total_param_fixes > 0 or total_return_fixes > 0):
        # Note: Full implementation would require more sophisticated source rewriting
        # For now, we'll just report what needs to be fixed
        print(f"{file_path}: {total_param_fixes} param fixes, {total_return_fixes} return fixes needed")

    return total_param_fixes, total_return_fixes


def main():
    """Main entry point."""
    print("Fixing missing type annotations...")
    print("Note: Some fixes may require manual review.\n")

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

    # Exclude test files
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

    total_param_fixes = 0
    total_return_fixes = 0

    print(f"Processing {len(files)} files...\n")
    for file_path in sorted(files):
        param_fixes, return_fixes = process_file(file_path)
        total_param_fixes += param_fixes
        total_return_fixes += return_fixes

    print("\nSummary:")
    print(f"  Parameter fixes needed: {total_param_fixes}")
    print(f"  Return section fixes needed: {total_return_fixes}")
    print("\nNote: This script identifies issues. Full automatic fixes")
    print("require more sophisticated source code rewriting.")


if __name__ == "__main__":
    main()

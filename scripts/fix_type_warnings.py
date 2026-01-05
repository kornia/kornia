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

"""Script to automatically fix type annotation warnings.

Fixes:
1. Removes parameters from docstrings that don't exist in function signatures
2. Adds Returns sections for functions with return type annotations
"""

import ast
import re
import sys
from pathlib import Path
from typing import Set, Tuple


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


def remove_invalid_params_from_docstring(docstring: str, valid_params: Set[str]) -> Tuple[str, int]:
    """Remove parameters from docstring that don't exist in function signature."""
    if not docstring:
        return docstring, 0

    lines = docstring.split("\n")
    fixed_lines = []
    fixes_count = 0
    in_args = False
    skip_lines = 0

    for i, line in enumerate(lines):
        if re.match(r"^\s*Args?:", line):
            in_args = True
            fixed_lines.append(line)
            continue

        if in_args:
            # Check if we've moved to another section
            if re.match(r"^\s*(Returns?|Note|Raises?|Examples?|Attributes?|Yields?|Shape|Example):", line):
                in_args = False
                fixed_lines.append(line)
                skip_lines = 0
                continue

            if skip_lines > 0:
                skip_lines -= 1
                continue

            # Check if this line defines a parameter
            # Match patterns like "param_name:", "param_name ", "*args ", "param (type) ", etc.
            # Also handle edge cases like trailing spaces, parentheses, etc.
            match = re.match(r"^(\s*)(\*?\*?\w+)(\s*\([^)]+\))?\s*[:\s\)]", line)
            if match:
                indent, param_name, type_hint = match.groups()
                param_name = param_name.strip()

                # Clean up parameter name - remove trailing spaces, parentheses, type hints, etc.
                clean_param = param_name.rstrip(" )")

                # Handle special cases like "*args " with trailing space
                clean_param = clean_param.rstrip()

                # Skip if it's not a valid parameter name (e.g., "Args", "Returns")
                if clean_param.lower() in (
                    "args",
                    "returns",
                    "raises",
                    "note",
                    "example",
                    "examples",
                    "attributes",
                    "yields",
                    "shape",
                ):
                    fixed_lines.append(line)
                    continue

                # Also check for malformed parameter names like "percentage of the spatial dimensions "
                # These are likely description text, not parameter names
                if len(clean_param.split()) > 3 or not clean_param.replace("*", "").replace("_", "").isalnum():
                    # Likely not a parameter name, keep the line
                    fixed_lines.append(line)
                    continue

                # Check if parameter is valid
                if param_name not in valid_params:
                    # Skip this line
                    fixes_count += 1
                    # Also skip continuation lines (lines with more indentation)
                    j = i + 1
                    while j < len(lines) and lines[j].strip():
                        next_indent = len(lines[j]) - len(lines[j].lstrip())
                        current_indent = len(line) - len(line.lstrip())
                        if next_indent > current_indent:
                            skip_lines += 1
                            j += 1
                        else:
                            break
                    continue
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines), fixes_count


def add_returns_section(docstring: str) -> Tuple[str, bool]:
    """Add Returns section if missing."""
    if not docstring:
        return docstring, False

    # Check if Returns section already exists
    if re.search(r"Returns?:", docstring):
        return docstring, False

    lines = docstring.rstrip().split("\n")

    # Find appropriate indent (usually 4 spaces)
    indent = 4
    for line in lines:
        if line.strip() and not line.strip().startswith(('"""', "'''")):
            indent = len(line) - len(line.lstrip())
            break

    # Add Returns section before closing quotes
    returns_line = f"{' ' * indent}Returns:\n{' ' * (indent + 4)}The output tensor."

    # Find where to insert (before last line if it's just quotes, or append)
    if lines and lines[-1].strip() in ('"""', "'''"):
        lines.insert(-1, returns_line)
    else:
        lines.append(returns_line)

    return "\n".join(lines), True


def replace_docstring_in_source(content: str, func_node: ast.FunctionDef, new_docstring: str) -> str:
    """Replace docstring in source code."""
    lines = content.split("\n")

    # Find the function's docstring
    # The docstring is usually the first statement in the function body
    if not func_node.body:
        return content

    # Find docstring node
    docstring_node = None
    if isinstance(func_node.body[0], ast.Expr) and isinstance(func_node.body[0].value, (ast.Str, ast.Constant)):
        docstring_node = func_node.body[0]
    elif isinstance(func_node.body[0], ast.Expr) and hasattr(ast, "Constant"):
        if isinstance(func_node.body[0].value, ast.Constant) and isinstance(func_node.body[0].value.value, str):
            docstring_node = func_node.body[0]

    if not docstring_node:
        return content

    # Get docstring line range
    docstring_start_line = docstring_node.lineno - 1
    docstring_end_line = (
        docstring_node.end_lineno - 1 if hasattr(docstring_node, "end_lineno") else docstring_start_line
    )

    # Extract the original docstring to find its exact format
    original_docstring = ast.get_docstring(func_node, clean=False)
    if not original_docstring:
        return content

    # Determine quote style and indentation
    quote_match = re.search(r'("""|\'\'\')', lines[docstring_start_line])
    if not quote_match:
        return content

    quote = quote_match.group(1)
    base_indent = len(lines[docstring_start_line]) - len(lines[docstring_start_line].lstrip())

    # Format new docstring with proper indentation
    new_docstring_lines = new_docstring.split("\n")
    formatted_lines = []
    formatted_lines.append(" " * base_indent + quote)

    for line in new_docstring_lines:
        if line.strip():
            formatted_lines.append(" " * (base_indent + 4) + line)
        else:
            formatted_lines.append("")

    formatted_lines.append(" " * base_indent + quote)

    # Replace lines
    new_lines = lines[:docstring_start_line] + formatted_lines + lines[docstring_end_line + 1 :]
    return "\n".join(new_lines)


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

    total_param_fixes = 0
    total_return_fixes = 0
    modified_content = content

    # Process functions in reverse order to maintain line numbers
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    functions.sort(key=lambda x: x.lineno, reverse=True)

    for func_node in functions:
        docstring = ast.get_docstring(func_node, clean=False)
        if not docstring:
            continue

        # Get valid parameters from function signature
        valid_params = get_function_params(func_node)

        # Fix invalid parameters in docstring
        fixed_docstring, param_fixes = remove_invalid_params_from_docstring(docstring, valid_params)
        total_param_fixes += param_fixes

        # Add Returns section if function has return annotation but docstring doesn't
        if func_node.returns and not re.search(r"Returns?:", fixed_docstring):
            fixed_docstring, added = add_returns_section(fixed_docstring)
            if added:
                total_return_fixes += 1

        if fixed_docstring != docstring:
            # Replace in source
            modified_content = replace_docstring_in_source(modified_content, func_node, fixed_docstring)

    if modified_content != content and (total_param_fixes > 0 or total_return_fixes > 0):
        file_path.write_text(modified_content, encoding="utf-8")
        print(f"Fixed {file_path}: {total_param_fixes} param fixes, {total_return_fixes} return fixes")

    return total_param_fixes, total_return_fixes


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
    print(f"  Parameter fixes: {total_param_fixes}")
    print(f"  Return section fixes: {total_return_fixes}")
    print(f"  Total fixes: {total_param_fixes + total_return_fixes}")


if __name__ == "__main__":
    main()

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

"""Script to add Returns sections to docstrings for functions with return type annotations.

Fixes missing return value annotations in docstrings.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Tuple


def add_returns_section(docstring: str, return_type_str: str = "") -> Tuple[str, bool]:
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

    # Generate return description based on return type
    if "Tensor" in return_type_str or "torch.Tensor" in return_type_str:
        returns_desc = "The output tensor."
    elif "Dict" in return_type_str or "dict" in return_type_str:
        returns_desc = "A dictionary containing the output parameters."
    elif "Tuple" in return_type_str or "tuple" in return_type_str:
        returns_desc = "A tuple containing the output values."
    elif "List" in return_type_str or "list" in return_type_str:
        returns_desc = "A list containing the output values."
    else:
        returns_desc = "The output value."

    returns_line = f"{' ' * indent}Returns:\n{' ' * (indent + 4)}{returns_desc}"

    # Find where to insert (before last line if it's just quotes, or append)
    if lines and lines[-1].strip() in ('"""', "'''"):
        lines.insert(-1, returns_line)
    else:
        lines.append(returns_line)

    return "\n".join(lines), True


def get_return_type_string(node: ast.FunctionDef) -> str:
    """Get string representation of return type."""
    if not node.returns:
        return ""

    # Try to get a readable string representation
    if isinstance(node.returns, ast.Name):
        return node.returns.id
    elif isinstance(node.returns, ast.Attribute):
        return f"{node.returns.value.id}.{node.returns.attr}"
    elif isinstance(node.returns, ast.Subscript):
        # Handle generics like Dict[str, Tensor]
        return "Dict"  # Simplified
    else:
        return str(node.returns)


def replace_docstring_in_source(content: str, func_node: ast.FunctionDef, new_docstring: str) -> str:
    """Replace docstring in source code."""
    lines = content.split("\n")

    # Find the function's docstring
    if not func_node.body:
        return content

    # Find docstring node
    docstring_node = None
    if isinstance(func_node.body[0], ast.Expr):
        if isinstance(func_node.body[0].value, (ast.Str, ast.Constant)):
            docstring_node = func_node.body[0]
        elif hasattr(ast, "Constant") and isinstance(func_node.body[0].value, ast.Constant):
            if isinstance(func_node.body[0].value.value, str):
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


def process_file(file_path: Path) -> int:
    """Process a file and add missing Returns sections."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
        return 0

    fixes_count = 0
    modified_content = content

    # Process functions in reverse order to maintain line numbers
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    functions.sort(key=lambda x: x.lineno, reverse=True)

    for func_node in functions:
        docstring = ast.get_docstring(func_node, clean=False)
        if not docstring:
            continue

        # Check if function has return annotation but docstring lacks Returns section
        if func_node.returns and not re.search(r"Returns?:", docstring):
            return_type_str = get_return_type_string(func_node)
            fixed_docstring, added = add_returns_section(docstring, return_type_str)

            if added:
                modified_content = replace_docstring_in_source(modified_content, func_node, fixed_docstring)
                fixes_count += 1

    if modified_content != content and fixes_count > 0:
        file_path.write_text(modified_content, encoding="utf-8")
        print(f"Fixed {file_path}: {fixes_count} return sections added")

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

    total_fixes = 0

    print(f"Processing {len(files)} files...\n")
    for file_path in sorted(files):
        fixes = process_file(file_path)
        total_fixes += fixes

    print(f"\nSummary: Added {total_fixes} Returns sections")


if __name__ == "__main__":
    main()

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

"""Script to automatically fix nn.Module documentation violations.

Fixes:
1. Moves Returns sections from class docstrings to forward method docstrings
2. Adds missing Returns sections to forward method docstrings when forward has return type
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


def is_nn_module_class(node: ast.ClassDef) -> bool:
    """Check if a class inherits from nn.Module."""
    for base in node.bases:
        if isinstance(base, ast.Name):
            if base.id == "Module":
                return True
        elif isinstance(base, ast.Attribute):
            if base.attr == "Module":
                if isinstance(base.value, ast.Name) and base.value.id == "nn":
                    return True
                elif isinstance(base.value, ast.Attribute):
                    if base.value.attr == "nn" and isinstance(base.value.value, ast.Name):
                        if base.value.value.id in ("torch", "kornia"):
                            return True
    return False


def find_forward_method(class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    """Find the forward method in a class."""
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            return node
    return None


def has_return_type_annotation(func_node: ast.FunctionDef) -> bool:
    """Check if a function has a return type annotation."""
    return func_node.returns is not None


def get_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from an AST node."""
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        return None

    if not node.body:
        return None

    first_stmt = node.body[0]
    if isinstance(first_stmt, ast.Expr):
        if isinstance(first_stmt.value, (ast.Str, ast.Constant)):
            if isinstance(first_stmt.value, ast.Constant):
                if isinstance(first_stmt.value.value, str):
                    return first_stmt.value.value
            elif isinstance(first_stmt.value, ast.Str):
                return first_stmt.value.s
    return None


def extract_returns_section(docstring: str) -> Tuple[Optional[str], str]:
    """Extract the Returns section from a docstring.

    Returns:
        Tuple of (returns_section, docstring_without_returns)
    """
    if not docstring:
        return None, docstring

    # Find Returns section (case insensitive)
    # Match from "Returns:" to the next section or end
    pattern = r"(?i)(Returns?:.*?)(?=\n\s*(?:Args?|Note|Example|Attributes?|Raises?|Yields?|Shape|$))"
    match = re.search(pattern, docstring, re.DOTALL | re.MULTILINE)

    if match:
        returns_section = match.group(1).strip()
        # Remove the Returns section from the docstring
        docstring_without_returns = docstring[: match.start()] + docstring[match.end() :]
        # Clean up extra newlines
        docstring_without_returns = re.sub(r"\n\n\n+", "\n\n", docstring_without_returns).strip()
        return returns_section, docstring_without_returns

    return None, docstring


def remove_returns_from_class_docstring(docstring: str) -> str:
    """Remove Returns section from class docstring."""
    _, docstring_without_returns = extract_returns_section(docstring)
    return docstring_without_returns


def add_returns_to_forward_docstring(docstring: str, returns_section: Optional[str], return_type_str: str = "") -> str:
    """Add Returns section to forward method docstring."""
    if not docstring:
        # Create a new docstring
        if returns_section:
            return f"Forward pass.\n\n{returns_section}"
        else:
            # Generate basic Returns section
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
            return f"Forward pass.\n\nReturns:\n    {returns_desc}"

    # Check if Returns already exists
    if re.search(r"(?i)Returns?:", docstring):
        return docstring

    # Add Returns section
    if returns_section:
        # Use the extracted Returns section
        return f"{docstring.rstrip()}\n\n{returns_section}"
    else:
        # Generate basic Returns section
        lines = docstring.rstrip().split("\n")
        indent = 4
        for line in lines:
            if line.strip() and not line.strip().startswith(('"""', "'''")):
                indent = len(line) - len(line.lstrip())
                break

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
        return f"{docstring.rstrip()}\n\n{returns_line}"


def get_return_type_string(node: ast.FunctionDef) -> str:
    """Get string representation of return type."""
    if not node.returns:
        return ""

    try:
        # Try to unparse if available (Python 3.9+)
        if hasattr(ast, "unparse"):
            return ast.unparse(node.returns)
    except:
        pass

    # Fallback to simple string representation
    if isinstance(node.returns, ast.Name):
        return node.returns.id
    elif isinstance(node.returns, ast.Attribute):
        if isinstance(node.returns.value, ast.Name):
            return f"{node.returns.value.id}.{node.returns.attr}"
    elif isinstance(node.returns, ast.Subscript):
        if isinstance(node.returns.value, ast.Name):
            return node.returns.value.id

    return str(node.returns)


def replace_docstring_in_source(content: str, node: ast.AST, new_docstring: str) -> str:
    """Replace docstring in source code."""
    lines = content.split("\n")

    if not node.body:
        return content

    # Find docstring node
    docstring_node = None
    if isinstance(node.body[0], ast.Expr):
        if isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            docstring_node = node.body[0]
        elif isinstance(node.body[0].value, ast.Constant):
            if isinstance(node.body[0].value.value, str):
                docstring_node = node.body[0]

    if not docstring_node:
        # No docstring exists, need to create one
        # Find the line after the definition
        if isinstance(node, ast.ClassDef):
            def_line = node.lineno - 1
        elif isinstance(node, ast.FunctionDef):
            def_line = node.lineno - 1
        else:
            return content

        # Determine indentation
        if def_line < len(lines):
            base_indent = len(lines[def_line]) - len(lines[def_line].lstrip())
        else:
            base_indent = 4

        # Determine quote style (prefer triple double quotes)
        quote = '"""'

        # Format new docstring
        new_docstring_lines = new_docstring.split("\n")
        formatted_lines = []
        formatted_lines.append(" " * (base_indent + 4) + quote)

        for line in new_docstring_lines:
            if line.strip():
                formatted_lines.append(" " * (base_indent + 8) + line)
            else:
                formatted_lines.append("")

        formatted_lines.append(" " * (base_indent + 4) + quote)

        # Insert after definition line
        new_lines = lines[: def_line + 1] + formatted_lines + lines[def_line + 1 :]
        return "\n".join(new_lines)

    # Get docstring line range
    docstring_start_line = docstring_node.lineno - 1
    docstring_end_line = (
        docstring_node.end_lineno - 1 if hasattr(docstring_node, "end_lineno") else docstring_start_line
    )

    # Extract the original docstring to find its exact format
    original_docstring = get_docstring(node)
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


def process_file(file_path: Path, violations: list) -> Tuple[int, int]:
    """Process a file and fix violations."""
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

    # Filter violations for this file
    file_violations = [v for v in violations if v["file_path"] == str(file_path)]
    if not file_violations:
        return 0, 0

    fixes_count = 0
    moves_count = 0
    modified_content = content

    # Process classes in reverse order to maintain line numbers
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    classes.sort(key=lambda x: x.lineno, reverse=True)

    for class_node in classes:
        if not is_nn_module_class(class_node):
            continue

        # Check if this class has violations
        class_violations = [
            v for v in file_violations if v["class_name"] == class_node.name and v["class_line"] == class_node.lineno
        ]
        if not class_violations:
            continue

        class_docstring = get_docstring(class_node)
        forward_method = find_forward_method(class_node)

        if forward_method is None:
            continue

        if not has_return_type_annotation(forward_method):
            continue

        forward_docstring = get_docstring(forward_method)
        return_type_str = get_return_type_string(forward_method)

        # Fix violation type 1: Returns in class docstring
        if any(v["violation_type"] == "returns_in_class_docstring" for v in class_violations):
            if class_docstring and re.search(r"(?i)Returns?:", class_docstring):
                returns_section, class_docstring_fixed = extract_returns_section(class_docstring)

                if returns_section:
                    # Update class docstring (remove Returns)
                    modified_content = replace_docstring_in_source(modified_content, class_node, class_docstring_fixed)

                    # Update forward docstring (add Returns)
                    forward_docstring_new = add_returns_to_forward_docstring(
                        forward_docstring or "", returns_section, return_type_str
                    )
                    modified_content = replace_docstring_in_source(
                        modified_content, forward_method, forward_docstring_new
                    )

                    moves_count += 1
                    fixes_count += 1

        # Fix violation type 2: Missing Returns in forward docstring
        if any(v["violation_type"] == "missing_returns_in_forward" for v in class_violations):
            if not (forward_docstring and re.search(r"(?i)Returns?:", forward_docstring)):
                forward_docstring_new = add_returns_to_forward_docstring(forward_docstring or "", None, return_type_str)
                modified_content = replace_docstring_in_source(modified_content, forward_method, forward_docstring_new)
                fixes_count += 1

    if modified_content != content and fixes_count > 0:
        file_path.write_text(modified_content, encoding="utf-8")
        if moves_count > 0:
            print(f"Fixed {file_path}: {fixes_count} fix(es) ({moves_count} Returns section(s) moved)")
        else:
            print(f"Fixed {file_path}: {fixes_count} fix(es)")

    return fixes_count, moves_count


def main():
    """Main entry point."""
    report_file = Path(__file__).parent.parent / "module_docs_violations.json"

    if not report_file.exists():
        print(f"Error: Violation report not found: {report_file}", file=sys.stderr)
        print("Please run scripts/verify_nnmodule_docs.py first.", file=sys.stderr)
        sys.exit(1)

    with open(report_file) as f:
        report = json.load(f)

    violations = report.get("violations", [])
    if not violations:
        print("No violations to fix!")
        sys.exit(0)

    # Group violations by file
    files_to_fix = set(v["file_path"] for v in violations)

    print(f"Fixing violations in {len(files_to_fix)} file(s)...\n")

    total_fixes = 0
    total_moves = 0

    for file_path_str in sorted(files_to_fix):
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue

        fixes, moves = process_file(file_path, violations)
        total_fixes += fixes
        total_moves += moves

    print(f"\n{'=' * 60}")
    print("FIX SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total fixes: {total_fixes}")
    print(f"  - Returns sections moved: {total_moves}")
    print(f"  - Missing Returns added: {total_fixes - total_moves}")
    print(f"\n✓ Fixed violations in {len(files_to_fix)} file(s)")

    if total_fixes > 0:
        print("\nPlease run scripts/verify_nnmodule_docs.py again to verify fixes.")


if __name__ == "__main__":
    main()

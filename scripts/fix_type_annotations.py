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

"""Script to fix missing type annotation warnings from griffe.

Fixes:
1. Parameters in docstrings that don't match function signatures
2. Missing return type annotations in docstrings
"""

import ast
import re
import sys
from pathlib import Path
from typing import Set, Tuple


def extract_function_signature(func_node: ast.FunctionDef) -> Tuple[Set[str], bool]:
    """Extract parameter names and whether function has return annotation."""
    params = set()
    for arg in func_node.args.args:
        params.add(arg.arg)
    if func_node.args.vararg:
        params.add(f"*{func_node.args.vararg.arg}")
    if func_node.args.kwarg:
        params.add(f"**{func_node.args.kwarg.arg}")

    has_return_annotation = func_node.returns is not None
    return params, has_return_annotation


def parse_docstring_params(docstring: str) -> Tuple[Set[str], bool]:
    """Extract parameter names and check for Returns section in docstring."""
    params = set()
    has_returns = False

    if not docstring:
        return params, has_returns

    # Look for Args: section
    args_match = re.search(r"Args?:", docstring)
    if args_match:
        args_start = args_match.end()
        # Find the next section (Returns, Note, Raises, etc.)
        next_section = re.search(r"\n\s*(Returns?|Note|Raises?|Examples?|Attributes?):", docstring[args_start:])
        if next_section:
            args_text = docstring[args_start : args_start + next_section.start()]
        else:
            args_text = docstring[args_start:]

        # Extract parameter names (format: "param_name: description")
        param_matches = re.findall(r"^\s*(\*?\*?\w+)\s*[:\s]", args_text, re.MULTILINE)
        params.update(param_matches)

    # Check for Returns section
    has_returns = bool(re.search(r"Returns?:", docstring))

    return params, has_returns


def fix_docstring_parameters(content: str, file_path: Path) -> Tuple[str, int]:
    """Fix docstring parameter mismatches.

    Removes parameters from docstrings that don't exist in the function signature.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content, 0

    lines = content.split("\n")
    fixes_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not ast.get_docstring(node):
                continue

            # Get function signature parameters
            sig_params, _ = extract_function_signature(node)

            # Get docstring parameters
            docstring = ast.get_docstring(node)
            doc_params, _ = parse_docstring_params(docstring)

            # Find parameters in docstring that don't exist in signature
            extra_params = doc_params - sig_params

            if extra_params:
                # Find the line numbers for this function's docstring
                func_start = node.lineno - 1
                # Find docstring end (approximate)
                docstring_lines = docstring.split("\n")

                # Try to fix by removing extra parameters from docstring
                # This is complex, so we'll just note it for now
                fixes_count += len(extra_params)

    return content, fixes_count


def add_missing_return_annotations(content: str) -> Tuple[str, int]:
    """Add missing Returns sections to docstrings when function has return annotation.

    This is a simplified version - full implementation would need more context.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content, 0

    lines = content.split("\n")
    fixes_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not ast.get_docstring(node):
                continue

            # Check if function has return annotation but docstring lacks Returns section
            _, has_return_annotation = extract_function_signature(node)
            docstring = ast.get_docstring(node)
            _, has_returns_section = parse_docstring_params(docstring)

            if has_return_annotation and not has_returns_section:
                # This would require inserting Returns section
                # For now, just count it
                fixes_count += 1

    return content, fixes_count


def process_file(file_path: Path) -> Tuple[int, int]:
    """Process a single Python file and fix type annotation issues."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0, 0

    # For now, just analyze and report
    # Full implementation would require more sophisticated parsing
    param_fixes, _ = fix_docstring_parameters(content, file_path)
    return_fixes, _ = add_missing_return_annotations(content)

    # Note: This is a placeholder - actual fixes would require more work
    return param_fixes, return_fixes


def main():
    """Main entry point."""
    print("Note: Type annotation fixes require manual review.")
    print("This script currently only analyzes issues.")
    print("For now, focus on fixing indentation issues which are automated.")
    print("\nTo fix type annotations, you'll need to:")
    print("1. Review each warning")
    print("2. Remove parameters from docstrings that don't exist in signatures")
    print("3. Add Returns sections where return types are annotated but not documented")

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

    print(f"\nAnalyzing {len(files)} files...")
    for file_path in sorted(files):
        param_fixes, return_fixes = process_file(file_path)
        total_param_fixes += param_fixes
        total_return_fixes += return_fixes

    print("\nSummary:")
    print(f"  Parameter mismatches found: {total_param_fixes}")
    print(f"  Missing Returns sections: {total_return_fixes}")


if __name__ == "__main__":
    main()

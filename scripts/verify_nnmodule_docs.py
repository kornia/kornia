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

"""Verify that all nn.Module classes follow the documentation pattern.

The key rule: Return values must be documented in the forward method's docstring,
NOT in the class docstring.

This script identifies violations and generates a report.
"""

import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Violation:
    """Represents a documentation violation."""

    file_path: str
    class_name: str
    class_line: int
    violation_type: str
    description: str
    forward_line: Optional[int] = None
    suggested_fix: Optional[str] = None


def is_nn_module_class(node: ast.ClassDef) -> bool:
    """Check if a class inherits from nn.Module."""
    for base in node.bases:
        if isinstance(base, ast.Name):
            if base.id == "Module":
                # Check if it's from torch.nn
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


def extract_returns_section(docstring: str) -> Optional[str]:
    """Extract the Returns section from a docstring."""
    if not docstring:
        return None

    # Match Returns: section (case insensitive)
    pattern = r"(?i)(Returns?:.*?)(?=\n\s*(?:Args?|Note|Example|Attributes?|Raises?|Yields?|Shape|$))"
    match = re.search(pattern, docstring, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def check_class_docstring_returns(docstring: str) -> bool:
    """Check if class docstring contains Returns section."""
    if not docstring:
        return False
    return bool(re.search(r"(?i)Returns?:", docstring))


def check_forward_docstring_returns(docstring: str) -> bool:
    """Check if forward docstring contains Returns section."""
    if not docstring:
        return False
    return bool(re.search(r"(?i)Returns?:", docstring))


def get_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from an AST node."""
    if not isinstance(node, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef):
        return None

    if not node.body:
        return None

    first_stmt = node.body[0]
    if isinstance(first_stmt, ast.Expr):
        if isinstance(first_stmt.value, ast.Str | ast.Constant):
            if isinstance(first_stmt.value, ast.Constant):
                if isinstance(first_stmt.value.value, str):
                    return first_stmt.value.value
            elif isinstance(first_stmt.value, ast.Str):
                return first_stmt.value.s
    return None


def process_file(file_path: Path) -> List[Violation]:
    """Process a single Python file and find violations."""
    violations = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return violations

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not is_nn_module_class(node):
                continue

            class_docstring = get_docstring(node)
            forward_method = find_forward_method(node)

            if forward_method is None:
                continue

            if not has_return_type_annotation(forward_method):
                continue

            forward_docstring = get_docstring(forward_method)

            # Violation Type 1: Returns in class docstring
            if check_class_docstring_returns(class_docstring or ""):
                violations.append(
                    Violation(
                        file_path=str(file_path),
                        class_name=node.name,
                        class_line=node.lineno,
                        violation_type="returns_in_class_docstring",
                        description="Class docstring contains Returns section (should be in forward method)",
                        forward_line=forward_method.lineno,
                        suggested_fix="Move Returns section to forward method docstring",
                    )
                )

            # Violation Type 2: Missing Returns in forward docstring
            if not check_forward_docstring_returns(forward_docstring or ""):
                return_type = ast.unparse(forward_method.returns) if forward_method.returns else "Unknown"
                violations.append(
                    Violation(
                        file_path=str(file_path),
                        class_name=node.name,
                        class_line=node.lineno,
                        violation_type="missing_returns_in_forward",
                        description=(
                            f"Forward method has return type annotation ({return_type}) "
                            "but no Returns section in docstring"
                        ),
                        forward_line=forward_method.lineno,
                        suggested_fix="Add Returns section to forward method docstring",
                    )
                )

    return violations


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
        and "docs" not in str(f)
        and "site" not in str(f)
    ]

    all_violations = []
    print(f"Scanning {len(files)} files...\n")

    for file_path in sorted(files):
        violations = process_file(file_path)
        all_violations.extend(violations)
        if violations:
            print(f"Found {len(violations)} violation(s) in {file_path}")

    # Generate report
    report = {
        "summary": {
            "total_files_scanned": len(files),
            "total_violations": len(all_violations),
            "returns_in_class_docstring": sum(
                1 for v in all_violations if v.violation_type == "returns_in_class_docstring"
            ),
            "missing_returns_in_forward": sum(
                1 for v in all_violations if v.violation_type == "missing_returns_in_forward"
            ),
        },
        "violations": [asdict(v) for v in all_violations],
    }

    # Save report
    report_file = Path(__file__).parent.parent / "module_docs_violations.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Files scanned: {report['summary']['total_files_scanned']}")
    print(f"Total violations: {report['summary']['total_violations']}")
    print(f"  - Returns in class docstring: {report['summary']['returns_in_class_docstring']}")
    print(f"  - Missing Returns in forward: {report['summary']['missing_returns_in_forward']}")
    print(f"\nReport saved to: {report_file}")

    if all_violations:
        print("\nViolations by file:")
        violations_by_file = {}
        for v in all_violations:
            if v.file_path not in violations_by_file:
                violations_by_file[v.file_path] = []
            violations_by_file[v.file_path].append(v)

        for file_path, file_violations in sorted(violations_by_file.items()):
            print(f"  {file_path}: {len(file_violations)} violation(s)")

        sys.exit(1)
    else:
        print("\n✓ All nn.Module classes follow the documentation pattern!")
        sys.exit(0)


if __name__ == "__main__":
    main()

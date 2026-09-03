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

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_PYTHON_RANDOM_CALLS = {
    "betavariate",
    "choice",
    "choices",
    "expovariate",
    "gammavariate",
    "gauss",
    "getrandbits",
    "lognormvariate",
    "normalvariate",
    "paretovariate",
    "randbytes",
    "randint",
    "random",
    "randrange",
    "sample",
    "seed",
    "shuffle",
    "triangular",
    "uniform",
    "vonmisesvariate",
    "weibullvariate",
}
_TORCH_RANDOM_CALLS = {
    "bernoulli",
    "multinomial",
    "normal",
    "poisson",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
}


@dataclass(frozen=True, order=True)
class EagerRngCall:
    path: str
    line: int
    name: str


@dataclass(frozen=True)
class AuditedEagerRngCall:
    call: EagerRngCall
    node_prefixes: tuple[str, ...]
    classification: Literal["value-independent", "value-dependent", "not-collected"]


def _attribute_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_rng_call(name: str) -> bool:
    parts = name.split(".")
    if len(parts) == 2 and parts[0] == "random":
        return parts[1] in _PYTHON_RANDOM_CALLS
    if len(parts) >= 3 and parts[:2] in (["np", "random"], ["numpy", "random"], ["torch", "random"]):
        return True
    return len(parts) == 2 and parts[0] == "torch" and parts[1] in _TORCH_RANDOM_CALLS


class _EagerRngVisitor(ast.NodeVisitor):
    """Visit expressions evaluated while importing a module, but not lazy bodies."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.calls: list[EagerRngCall] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for expression in (*node.decorator_list, *node.args.defaults, *node.args.kw_defaults):
            if expression is not None:
                self.visit(expression)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for expression in (*node.args.defaults, *node.args.kw_defaults):
            if expression is not None:
                self.visit(expression)

    def visit_Call(self, node: ast.Call) -> None:
        name = _attribute_name(node.func)
        if name is not None and _is_rng_call(name):
            self.calls.append(EagerRngCall(self.path, node.lineno, name))
        self.generic_visit(node)


def find_eager_rng_calls(root: Path) -> tuple[EagerRngCall, ...]:
    """Return eager Python, NumPy, and Torch RNG calls below a repository or test root."""
    scan_root = root / "tests" if (root / "tests").is_dir() else root
    calls: list[EagerRngCall] = []
    for path in sorted(scan_root.rglob("*.py")):
        visitor = _EagerRngVisitor(path.relative_to(root).as_posix())
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        calls.extend(visitor.calls)
    return tuple(sorted(calls))


def _audited(
    path: str,
    line: int,
    name: str,
    count: int,
    node_prefixes: tuple[str, ...],
    classification: Literal["value-independent", "value-dependent", "not-collected"],
) -> tuple[AuditedEagerRngCall, ...]:
    return tuple(
        AuditedEagerRngCall(EagerRngCall(path, line, name), node_prefixes, classification) for _ in range(count)
    )


AUDITED_EAGER_RNG_CALLS = tuple(
    sorted(
        (
            *_audited(
                "tests/augmentation/container/test_augmentation_sequential.py",
                47,
                "torch.randn",
                2,
                (
                    "tests/augmentation/container/test_augmentation_sequential.py::TestAugmentationSequential::test_mixup",
                ),
                "value-dependent",
            ),
            *_audited(
                "tests/augmentation/test_param_validation.py",
                110,
                "torch.rand",
                1,
                ("tests/augmentation/test_param_validation.py::TestParamValidation::test_tuple_range_reader_errors",),
                "value-independent",
            ),
            *_audited("tests/benchmark.py", 32, "torch.rand", 1, (), "not-collected"),
            *(
                AuditedEagerRngCall(
                    EagerRngCall("tests/core/test_check.py", line, "torch.rand"),
                    (f"tests/core/test_check.py::TestCheckShape::{test_name}",),
                    "value-independent",
                )
                for test_name, lines in (("test_valid", range(71, 75)), ("test_invalid", range(83, 87)))
                for line in lines
            ),
            *(
                _audited(
                    "tests/enhance/test_core.py",
                    line,
                    "torch.randn",
                    1,
                    ("tests/enhance/test_core.py::TestAddWeighted::test_shape",),
                    "value-dependent",
                )[0]
                for line in range(91, 94)
            ),
            *_audited(
                "tests/geometry/subpix/test_dsnt.py",
                73,
                "torch.randn",
                1,
                (
                    "tests/geometry/subpix/test_dsnt.py::TestSpatialSoftmax2d::test_forward",
                    "tests/geometry/subpix/test_dsnt.py::TestSpatialSoftmax2d::test_dynamo",
                ),
                "value-dependent",
            ),
            *_audited(
                "tests/image/test_draw.py",
                305,
                "torch.rand",
                4,
                ("tests/image/test_draw.py::TestDrawLine::test_point_size",),
                "value-independent",
            ),
            *_audited(
                "tests/losses/test_total_variation.py",
                127,
                "torch.rand",
                3,
                ("tests/losses/test_total_variation.py::TestTotalVariation::test_tv_shapes",),
                "value-dependent",
            ),
        ),
        key=lambda entry: entry.call,
    )
)


def node_matches_prefix(nodeid: str, prefix: str) -> bool:
    """Match one unparameterized node exactly or at its parameter boundary."""
    return nodeid == prefix or nodeid.startswith(f"{prefix}[")


def eager_rng_calls_for_node(nodeid: str) -> tuple[AuditedEagerRngCall, ...]:
    """Return audited eager RNG calls whose values feed a collected node."""
    return tuple(
        entry
        for entry in AUDITED_EAGER_RNG_CALLS
        if entry.classification == "value-dependent"
        and any(node_matches_prefix(nodeid, prefix) for prefix in entry.node_prefixes)
    )

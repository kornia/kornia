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

"""No ``KORNIA_CHECK*`` message in ``kornia/`` formats a value that ``torch.compile`` can treat as symbolic.

Python evaluates the arguments of ``KORNIA_CHECK(condition, f"... {x.shape[0]} ...")`` on every
call, so the message is formatted even when the check passes. Under a dynamic-shape
``torch.compile``, formatting a symbolic integer makes Dynamo specialize it, so every new value
compiles a new graph (and after ``torch._dynamo.config.recompile_limit`` graphs the function runs
eagerly); ``str()`` of one does not trace at all. Such a value is a single tensor size --
``x.shape[i]``, ``x.size(i)``, ``x.numel()`` -- or an ``int`` argument of the compiled function, and
anything computed from them. A value read from tensor data (``.item()``, ``.tolist()``) is worse:
it breaks the graph, and in eager mode it synchronizes with the device on every call. Whole shapes
(``x.shape``, ``tuple(x.shape)``, slices), dtypes, devices and strings do not specialize, so their
messages stay as they are.

Such a check uses a helper that compares the sizes itself and formats them only on failure
(``KORNIA_CHECK_SAME_SIZE``, ``KORNIA_CHECK_SIZES``, ``KORNIA_CHECK_SHAPE``), or passes the message
as a function, ``KORNIA_CHECK(condition, lambda: f"...")``, which the check calls only on failure.

The rules, for every call to a function named ``KORNIA_CHECK*`` that has a ``msg`` parameter, outside
constructors (``torch.compile`` compiles ``forward``, not module construction):

* a message that is not a ``lambda`` formats no single size, no ``int``-annotated argument of the
  function, no local assigned from those, nothing computed from them, and no ``.item()`` or
  ``.tolist()``;
* except in ``_TORCHSCRIPT_FUNCTIONS``, which TorchScript compiles: TorchScript rejects a ``lambda``
  anywhere in their source, so a message there may format such a value but may not be a ``lambda``.
  It formats whole shapes where it can, which neither specializes nor needs deferring. Every entry
  must exist and hold a formatted message.

A message assigned to a local first counts as formatting what that assignment formats.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import kornia

_KORNIA_ROOT = Path(kornia.__file__).resolve().parent
_SIZE_METHODS = {"size", "numel"}
_DATA_METHODS = {"item", "tolist"}
_PASSTHROUGH = {"int", "min", "max", "abs"}

# Functions with formatted check messages that TorchScript compiles: those that ``torch.jit.script`` compiles directly
# or through a caller in the tests, with or without ``KORNIA_TEST_OPTIMIZER=jit``. TorchScript rejects a ``lambda``
# anywhere in a scripted function, even under ``if not torch.jit.is_scripting()``. Their messages that format a
# single size or an ``int`` stay eager: measured under ``torch.compile(dynamic=True)`` on torch 2.5.1 and 2.14, only
# ``_check_n``'s ``{n}`` adds graphs, when ``n`` is itself an input of the compiled function (``one_hot`` specializes
# ``num_classes`` regardless, and ``reconstruction`` breaks the graph on its data-dependent loop).
_TORCHSCRIPT_FUNCTIONS = {
    "color/raw.py::raw_to_rgb_2x2_downscaled",
    "core/ops.py::_check_n",
    "core/ops.py::eye_like",
    "core/ops.py::vec_like",
    "enhance/shift_rgb.py::shift_rgb",
    "feature/dedode/transformer/layers/patch_embed.py::PatchEmbed.forward",
    "geometry/depth.py::warp_frame_depth",
    "losses/focal.py::binary_focal_loss_with_logits",
    "losses/focal.py::focal_loss",
    "losses/lovasz_softmax.py::lovasz_softmax_loss",
    "losses/one_hot.py::one_hot",
    "morphology/morphology.py::reconstruction",
}


def _int_params(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return the parameters annotated with an ``int`` type, alone or in a union, tuple or ``Optional``."""
    names = set()
    for arg in func.args.posonlyargs + func.args.args + func.args.kwonlyargs:
        if arg.annotation is not None and any(
            isinstance(n, ast.Name) and n.id == "int" for n in ast.walk(arg.annotation)
        ):
            names.add(arg.arg)
    return names


def _is_symbolic(node: ast.AST, names: set[str]) -> bool:
    """Return whether ``node`` is a value that Dynamo can treat as symbolic, or one read from tensor data."""
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            return False
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            return True
        return isinstance(value, ast.Name) and value.id in names
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _DATA_METHODS:
            return True
        if isinstance(func, ast.Attribute) and func.attr in _SIZE_METHODS:
            return func.attr == "numel" or bool(node.args)
        if isinstance(func, ast.Name) and func.id in _PASSTHROUGH:
            return any(_is_symbolic(a, names) for a in node.args)
        return False
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.BinOp):
        return _is_symbolic(node.left, names) or _is_symbolic(node.right, names)
    if isinstance(node, ast.UnaryOp):
        return _is_symbolic(node.operand, names)
    if isinstance(node, ast.IfExp):
        return _is_symbolic(node.body, names) or _is_symbolic(node.orelse, names)
    return False


def _is_shape(node: ast.AST) -> bool:
    """Return whether ``node`` is a whole or sliced shape, whose unpacking gives sizes."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        node = node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "size":
        return not node.args
    return isinstance(node, ast.Attribute) and node.attr == "shape"


def _symbolic_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return a function's ``int`` parameters and the locals it assigns sizes, values computed from them or strings
    that format them."""
    names = _int_params(func)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            for target in targets:
                # a size, or a message built from one
                if isinstance(target, ast.Name) and (
                    _is_symbolic(value, names) or any(_is_symbolic(v, names) for v in _formatted_values(value))
                ):
                    new = {target.id}
                elif isinstance(target, ast.Tuple) and (
                    _is_shape(value)
                    or (isinstance(value, ast.Tuple) and any(_is_symbolic(v, names) for v in value.elts))
                ):
                    new = {t.id for t in target.elts if isinstance(t, ast.Name)}
                else:
                    continue
                if not new <= names:
                    names |= new
                    changed = True
    return names


def _formatted_values(msg: ast.AST) -> list[ast.AST]:
    """Return the values an eagerly built message formats: f-string fields, ``str.format``, ``%`` and ``str()``."""
    values: list[ast.AST] = []
    for node in ast.walk(msg):
        if isinstance(node, ast.FormattedValue):
            values.append(node.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            values += list(node.args) + [k.value for k in node.keywords]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"str", "repr"}:
            values += list(node.args)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            right = node.right
            values += list(right.elts) if isinstance(right, ast.Tuple) else [right]
    return values


def _msg_positions(trees: dict[str, ast.Module]) -> dict[str, int]:
    """Map each ``KORNIA_CHECK*`` function defined in ``kornia/`` to the position of its ``msg`` parameter."""
    positions: dict[str, int] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("KORNIA_CHECK"):
                params = [a.arg for a in node.args.posonlyargs + node.args.args]
                if "msg" in params:
                    positions[node.name] = params.index("msg")
    return positions


def _qualified_names(tree: ast.Module) -> dict[ast.AST, str]:
    """Map every function of a module to its name, as ``Class.method`` inside a class."""
    names: dict[ast.AST, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names[child] = f"{prefix}{child.name}"
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return names


def _violations(
    rel: str, tree: ast.Module, positions: dict[str, int], torchscript: set[str]
) -> tuple[int, list[str], set[str], set[str]]:
    """Return ``(checked calls, violations, functions defined, TorchScript functions with a formatted message)``."""
    qualified = _qualified_names(tree)
    defined = {f"{rel}::{name}" for name in qualified.values()}
    n_calls = 0
    violations: list[str] = []
    formatted_in: set[str] = set()
    seen: set[int] = set()
    # outer functions start before the functions nested in them; in reverse order, each call is judged in the
    # innermost function that contains it
    for func in sorted(qualified, key=lambda f: (f.lineno, f.col_offset), reverse=True):
        key = f"{rel}::{qualified[func]}"
        names = _symbolic_names(func)
        for node in ast.walk(func):
            if not isinstance(node, ast.Call) or id(node) in seen:
                continue
            callee = node.func
            name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
            if name not in positions:
                continue
            seen.add(id(node))
            n_calls += 1
            if func.name == "__init__":
                continue
            pos = positions[name]
            msg = node.args[pos] if len(node.args) > pos else None
            msg = next((k.value for k in node.keywords if k.arg == "msg"), msg)
            if msg is None:
                continue
            if isinstance(msg, ast.Lambda):
                if key in torchscript:
                    violations.append(f"{rel}:{node.lineno} {name} has a lambda message, which TorchScript rejects")
                continue
            values = _formatted_values(msg)
            symbolic = [ast.unparse(v) for v in values if _is_symbolic(v, names)]
            if isinstance(msg, ast.Name) and msg.id in names:
                symbolic.append(msg.id)
            if key in torchscript:
                if values or symbolic:
                    formatted_in.add(key)
            elif symbolic:
                violations.append(f"{rel}:{node.lineno} {name} formats {', '.join(symbolic)}")
    return n_calls, violations, defined, formatted_in


def _scan() -> tuple[int, list[str], set[str], set[str]]:
    """Return ``(checked calls, violations, functions defined, exempt functions used)`` over ``kornia/``."""
    trees = {
        path.relative_to(_KORNIA_ROOT).as_posix(): ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(_KORNIA_ROOT.rglob("*.py"))
    }
    positions = _msg_positions(trees)
    n_calls = 0
    violations: list[str] = []
    defined: set[str] = set()
    exempt_used: set[str] = set()
    for rel, tree in trees.items():
        n, found, names, used = _violations(rel, tree, positions, _TORCHSCRIPT_FUNCTIONS)
        n_calls += n
        violations += found
        defined |= names
        exempt_used |= used
    return n_calls, sorted(violations), defined, exempt_used


class TestCheckMessageCallSites:
    def test_no_message_formats_a_symbolic_value(self) -> None:
        n_calls, violations, _, _ = _scan()
        assert n_calls > 500, f"the scan found only {n_calls} KORNIA_CHECK calls; is it reading kornia/?"
        assert not violations, (
            "These KORNIA_CHECK messages format a tensor size or an int argument on every call. Under a "
            "dynamic-shape torch.compile that specializes the value, so each new value compiles a new graph. Compare "
            "sizes with KORNIA_CHECK_SAME_SIZE / KORNIA_CHECK_SIZES / KORNIA_CHECK_SHAPE, or pass the message as a "
            "function, which the check calls only on failure: `KORNIA_CHECK(condition, lambda: f'...')`.\n"
            + "\n".join(violations)
        )

    def test_torchscript_functions_are_current(self) -> None:
        """Every listed function exists and holds a formatted message, so the list cannot outgrow its reason."""
        _, _, defined, formatted_in = _scan()
        assert _TORCHSCRIPT_FUNCTIONS <= defined, f"no such function: {sorted(_TORCHSCRIPT_FUNCTIONS - defined)}"
        assert formatted_in == _TORCHSCRIPT_FUNCTIONS, (
            f"no formatted message left, drop it from the list: {sorted(_TORCHSCRIPT_FUNCTIONS - formatted_in)}"
        )

    def test_scan_flags_symbolic_values_only(self) -> None:
        """The scanner flags every form of a symbolic value, and leaves whole shapes, dtypes and strings alone."""
        source = """
def KORNIA_CHECK(condition, msg=None, raises=True): ...
def KORNIA_CHECK_SHAPE(x, shape, msg=None, raises=True): ...

def f(x, y, n: int, k: tuple[int, int] | int, name: str, h: float):
    B, C, H, W = x.shape
    rows = x.shape[0] * 2
    side = int(max(H, W))
    KORNIA_CHECK(x.dim() == 4, f"got {x.shape[0]}")                  # flagged: a single size
    KORNIA_CHECK(x.dim() == 4, f"got {x.size(1)} and {x.numel()}")   # flagged: size(i), numel()
    KORNIA_CHECK(n > 0, f"got {n}")                                  # flagged: an int argument
    KORNIA_CHECK(k[0] > 0, f"got {k[0]}")                            # flagged: an element of an int argument
    KORNIA_CHECK(H > 0, f"got {H} x {W}")                            # flagged: unpacked from the shape
    KORNIA_CHECK(rows > 0, f"got {rows}")                            # flagged: computed from a size
    KORNIA_CHECK(side > 0, "got {}".format(side))                    # flagged: through int/max, by str.format
    KORNIA_CHECK(n > 0, "got %d" % (n + 1))                          # flagged: %, arithmetic
    KORNIA_CHECK(x.min() >= 0, f"got {x.min().item()}")              # flagged: a value read from the data
    KORNIA_CHECK_SHAPE(x, ["B"], msg="got " + str(B))                # flagged: str(), keyword msg
    if x.dim() != 4:
        KORNIA_CHECK(False, f"got {x.shape[0]}")                     # flagged: still formatted eagerly
    built = f"got {n}"
    KORNIA_CHECK(n > 0, built)                                       # flagged: a message built in a local
    KORNIA_CHECK(n > 0, f"{built}, so no")                           # flagged: placed in another message
    KORNIA_CHECK(x.dim() == 4, f"got {x.shape} {tuple(x.shape)} {x.shape[-2:]} {(B, C)}")
    KORNIA_CHECK(x.dim() == 4, f"got {x.dtype} on {x.device}, {x.ndim} dims, {type(n).__name__}")
    KORNIA_CHECK(h > 0, f"{name} needs a positive h, got {h}")
    KORNIA_CHECK(n > 0, lambda: f"got {n} and {x.shape[0]}")

    def deferred() -> str:
        return f"got {n}"

    KORNIA_CHECK(n > 0, deferred)

def scripted(n: int):
    KORNIA_CHECK(n > 0, f"got {n}")                                  # TorchScript compiles it: stays eager
    KORNIA_CHECK(n > 0, lambda: f"got {n}")                          # flagged: TorchScript rejects a lambda

class M:
    def __init__(self, n: int):
        KORNIA_CHECK(n > 0, f"got {n}")                              # a constructor

    def forward(self, x):
        KORNIA_CHECK(x.dim() == 4, f"got {x.shape}")                 # TorchScript compiles it
"""
        tree = ast.parse(source)
        positions = _msg_positions({"m.py": tree})
        assert positions == {"KORNIA_CHECK": 1, "KORNIA_CHECK_SHAPE": 2}
        torchscript = {"m.py::scripted", "m.py::M.forward"}
        n_calls, violations, defined, formatted_in = _violations("m.py", tree, positions, torchscript)
        flagged = [i + 1 for i, line in enumerate(source.splitlines()) if "# flagged" in line]
        assert n_calls == 22
        assert sorted(int(v.split(":")[1].split()[0]) for v in violations) == flagged
        assert {"m.py::f", "m.py::f.deferred", "m.py::scripted", "m.py::M.__init__", "m.py::M.forward"} <= defined
        assert formatted_in == torchscript


def _rand(*shape: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(*shape, device=device, dtype=dtype)


def _pnp_inputs(n: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    world = _rand(2, n, 3, device=device, dtype=dtype) + torch.tensor([0.0, 0.0, 4.0], device=device, dtype=dtype)
    intrinsics = torch.eye(3, device=device, dtype=dtype).expand(2, 3, 3)
    return world, _rand(2, n, 2, device=device, dtype=dtype), intrinsics


# Entry points whose messages formatted a size or an int on every call, which recompiled for each new value under a
# dynamic-shape torch.compile: name -> (op, inputs for one value of the formatted size or int)
_COMPILE_CASES = {
    "get_motion_kernel2d (angle batch)": (
        lambda angle, direction: kornia.filters.get_motion_kernel2d(5, angle, direction),
        lambda b, device, dtype: (_rand(b, device=device, dtype=dtype) * 90, _rand(b, device=device, dtype=dtype)),
    ),
    "motion_blur (angle batch)": (
        lambda x, angle, direction: kornia.filters.motion_blur(x, 5, angle, direction),
        lambda b, device, dtype: (
            _rand(b, 3, 9, 11, device=device, dtype=dtype),
            _rand(b, device=device, dtype=dtype) * 90,
            _rand(b, device=device, dtype=dtype),
        ),
    ),
    "fft_conv valid (image size)": (
        lambda x, kernel: kornia.filters.fft_conv(x, kernel, padding="valid"),
        lambda h, device, dtype: (
            _rand(2, 3, h, h + 3, device=device, dtype=dtype),
            _rand(1, 3, 3, device=device, dtype=dtype),
        ),
    ),
    "gaussian_blur2d (kernel size)": (
        lambda x, k: kornia.filters.gaussian_blur2d(x, (k, k), (1.5, 1.5)),
        lambda k, device, dtype: (_rand(2, 3, 16, 19, device=device, dtype=dtype), k),
    ),
    "solve_pnp_dlt (number of points)": (
        kornia.geometry.calibration.solve_pnp_dlt,
        _pnp_inputs,
    ),
    "jpeg_codec_differentiable (batch)": (
        kornia.enhance.jpeg_codec_differentiable,
        lambda b, device, dtype: (
            _rand(b, 3, 16, 16, device=device, dtype=dtype),
            torch.full((b,), 50.0, device=device, dtype=dtype),
        ),
    ),
    "dedode PatchEmbed (image size)": (
        None,
        lambda h, device, dtype: (_rand(2, 3, h, h + 4, device=device, dtype=dtype),),
    ),
}
# the formatted size or int takes these values; none equals another size of the same call, so Dynamo's duck sizing
# cannot tie it to one
_COMPILE_VALUES = {
    "get_motion_kernel2d (angle batch)": (2, 3, 4),
    "motion_blur (angle batch)": (2, 4, 6),
    "fft_conv valid (image size)": (6, 8, 10),
    # 3 is avoided: on macOS arm64, F.conv2d picks another backend for a 3 x 3 kernel, which adds a graph by itself
    "gaussian_blur2d (kernel size)": (5, 7, 9),
    "solve_pnp_dlt (number of points)": (7, 9, 11),
    "jpeg_codec_differentiable (batch)": (2, 3, 4),
    "dedode PatchEmbed (image size)": (16, 20, 24),
}


class TestDeferredMessagesUnderCompile:
    @pytest.mark.parametrize("case", sorted(_COMPILE_CASES))
    def test_dynamo_new_values_compile_no_new_graph(self, case, device, dtype, torch_optimizer, optimizer_backend):
        """A new value of a size or int that a check message formats compiles no new graph."""
        from torch._dynamo.testing import CompileCounter

        if optimizer_backend == "jit":
            pytest.skip("counts the graphs of torch.compile, which torch.jit.script does not build")
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("the graph count does not depend on the floating dtype")
        op, make_inputs = _COMPILE_CASES[case]
        if op is None:
            from kornia.feature.dedode.transformer.layers.patch_embed import PatchEmbed

            op = PatchEmbed(img_size=224, patch_size=4, in_chans=3, embed_dim=8).to(device, dtype)
        counter = CompileCounter()
        compiled = torch_optimizer(op, backend=counter, dynamic=True)
        values = _COMPILE_VALUES[case]
        compiled(*make_inputs(values[0], device, dtype))
        first = counter.frame_count
        for value in values[1:]:
            compiled(*make_inputs(value, device, dtype))
        assert counter.frame_count == first, f"{counter.frame_count - first} new graphs for new values of {case}"

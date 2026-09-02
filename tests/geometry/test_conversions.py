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

import dis
import inspect
import platform
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache, partial
from types import TracebackType

import numpy as np
import pytest
import torch

import kornia
from kornia.core._compat import torch_version, torch_version_lt
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.core.exceptions import BaseError, ShapeError
from kornia.core.ops import eye_like
from kornia.geometry.conversions import (
    ARKitQTVecs_to_ColmapQTVecs,
    Rt_to_matrix4x4,
    axis_angle_to_rotation_matrix,
    camtoworld_graphics_to_vision_4x4,
    camtoworld_graphics_to_vision_Rt,
    camtoworld_to_worldtocam_Rt,
    camtoworld_vision_to_graphics_4x4,
    camtoworld_vision_to_graphics_Rt,
    euler_from_quaternion,
    matrix4x4_to_Rt,
    quaternion_from_euler,
    worldtocam_to_camtoworld_Rt,
)
from kornia.geometry.quaternion import Quaternion

from testing.base import (
    DYNAMIC_EXPORT_UNAVAILABLE_REASON,
    DYNAMO_UNAVAILABLE_REASON,
    BaseTester,
    assert_close,
    dynamic_export_is_available,
    dynamo_is_available,
)


@pytest.fixture(autouse=True)
def seed_rng() -> None:
    torch.manual_seed(0)


@pytest.fixture()
def atol(device, dtype):
    """Lower tolerance for cuda-float16 only."""
    if "cuda" in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


@pytest.fixture()
def rtol(device, dtype):
    """Lower tolerance for cuda-float16 only."""
    if "cuda" in device.type and dtype == torch.float16:
        return 1.0e-3
    return 1.0e-4


def _runs_without_raising(func, *args, **kwargs) -> bool:
    # Shared boolean adapter for the kornia#3955 strict-xfail call sites below, whose CURRENT
    # behavior is a raise. That mark carries raises=AssertionError so that an unrelated
    # environment error cannot silently *satisfy* the mark and stop the pin from testing anything;
    # that in turn means the xfail body must fail as an assertion and must never let the exception
    # escape. Asserting on this boolean is how each call site does it. Retire this helper together
    # with the #3955 pins.
    try:
        func(*args, **kwargs)
    except Exception:
        return False
    return True


def _issue_msg(text: str):
    # torch.testing.assert_close accepts msg as a callable that receives its own diff report; this
    # wrapper prefixes the issue number so that the assert_close-based bug pins below name their
    # issue in the failure text, the way their bare-assert siblings do. The test names carry the
    # number too, but the failure text is what a future XPASS or wart flip is read from.
    # The pins using it call the module-level assert_close rather than BaseTester.assert_close,
    # which does not forward msg; the two apply the same dtype-aware default tolerances.
    return lambda default_message: f"{text}\n{default_message}"


def _innermost_frame(err: BaseException) -> TracebackType | None:
    # The frame an exception actually died in: the last link of its traceback chain. Two helpers
    # below read that frame for different questions -- which routine failed, and which bytecode
    # instruction raised -- so the walk itself is written once and cannot drift between them.
    frame = err.__traceback__
    while frame is not None and frame.tb_next is not None:
        frame = frame.tb_next
    return frame


@cache
def _dtype_allocation_error(device: torch.device, dtype: torch.dtype) -> str | None:
    # "Can this backend hold this dtype at all?", as ONE probe with ONE exception set, shared by
    # _skip_if_dtype_unavailable and _cross_is_unavailable below. Two copies of it with different
    # exception tuples would mean a backend that starts rejecting an allocation with a new
    # exception type makes one of them skip while the other errors, for the same fact. Cached
    # because the answer is a property of the build, not of the caller, and the pins below ask it
    # once per test; the message is returned rather than the exception so no traceback is kept
    # alive between tests.
    try:
        torch.zeros(1, device=device, dtype=dtype)
    except (TypeError, RuntimeError, NotImplementedError) as err:
        return str(err)
    return None


def _skip_if_dtype_unavailable(device: torch.device, dtype: torch.dtype) -> None:
    # Visible skip (never a silent guard) for the pins below, in both directions. The
    # dtype-hardcoded pins drop the dtype fixture on purpose so they run in every test
    # configuration, which means they would otherwise also run on a device that cannot represent
    # the dtype at all -- MPS has no float64 -- and the resulting TypeError would satisfy a
    # raises=AssertionError xfail mark instead of the assertion the pin documents. The
    # fixture-parametrized pins call it for the same reason read the other way round: a
    # --device=mps --dtype=all run would otherwise report them as failures indistinguishable from
    # a real one, when the only fact being reported is that MPS has no float64. Probed at runtime
    # rather than hardcoded per backend so the skip retires itself once a backend gains the
    # dtype.
    allocation_error = _dtype_allocation_error(device, dtype)
    if allocation_error is not None:
        pytest.skip(f"{dtype} is unavailable on device {device}: {allocation_error}")


def _agreement_gap_at_2_28(device: torch.device, dtype: torch.dtype) -> float:
    # max|normalize_pixel_coordinates(grid) - matrix @ grid| over the full pixel grid of a
    # (2, 28) image -- the size pair behind normal_transform_pixel's agreement bullet, chosen
    # because 2/(28 - 1) is not exactly representable in reduced precision. Shared by the two pins
    # that read it (the dtype-scaled bound and the exact kernel measurement) so they cannot measure
    # subtly different things.
    # The grid is built from an int64 arange cast down afterwards, not a typed arange: `arange_mps`
    # has no bfloat16 kernel on torch 2.5.1 (executed; torch 2.9.1 has one) and would raise
    # instead of measuring.
    rows, columns = 2, 28
    y_grid, x_grid = torch.meshgrid(
        torch.arange(rows, device=device).to(dtype),
        torch.arange(columns, device=device).to(dtype),
        indexing="ij",
    )
    grid = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)[None]

    via_helper = kornia.geometry.conversions.normalize_pixel_coordinates(grid, rows, columns)[0]
    matrix = kornia.geometry.conversions.normal_transform_pixel(rows, columns, device=device, dtype=dtype)
    homogeneous = torch.cat([grid[0], torch.ones_like(grid[0][:, :1])], dim=-1)
    via_matrix = (matrix[0] @ homogeneous.transpose(0, 1)).transpose(0, 1)[:, :2]

    # .float() because the difference of two bfloat16 values is not representable in bfloat16.
    return (via_helper.float() - via_matrix.float()).abs().max().item()


def _matmul_input_eps(device: torch.device, dtype: torch.dtype) -> float:
    # eps of the precision a matmul rounds its INPUTS to, which is not always the working dtype's:
    # on cuda a float32 matmul can be configured to round its inputs to TF32, 10 explicit mantissa
    # bits instead of 23. This suite can be run that way -- conftest.py's --tf32 / KORNIA_TEST_TF32
    # calls torch.set_float32_matmul_precision("high"), and executed here, both "high" and "medium"
    # leave torch.backends.cuda.matmul.allow_tf32 True (torch 2.9.1). The bound below is scaled by
    # this rather than by finfo(dtype).eps so that such a run widens it instead of reporting a red
    # test for a configuration change that touched no kornia code: the matrix route is a matmul
    # while the helper route stays in the working dtype, so it is exactly the leg a
    # reduced-precision mode rounds more coarsely.
    # Scoped to cuda because that is where the mode applies: executed on cpu, the (2, 28) gap is
    # identical under "highest", "high" and "medium" at all four dtypes, so widening the cpu bound
    # would only lose resolution. TF32's mantissa width is the documented format rather than a
    # measurement -- no CUDA device was available in this branch -- so this arm is a guard against
    # an unmeasured configuration, not a claim about one; a measured cuda figure belongs in the
    # wart pin's table below.
    if device.type == "cuda" and dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
        return 2.0**-10
    return torch.finfo(dtype).eps


_healthy_closed_form_inverse_routes: set[tuple[torch.device, torch.dtype]] = set()


def _skip_if_closed_form_inverse_unavailable(device: torch.device, dtype: torch.dtype) -> None:
    # Visible skip for the pins that route through normalize_homography, one layer deeper than
    # _skip_if_dtype_unavailable: a backend can REPRESENT a dtype and still have no kernel for an
    # operation the route needs. kornia's cusolver-free 3x3 inverse
    # (_inverse_3x3_closed_form in kornia/core/utils.py) is three torch.linalg.cross calls, and
    # MPS lacks a bfloat16 `cross` kernel in SOME builds -- executed: torch 2.5.1 raises
    # `RuntimeError: Failed to create function state object for: cross_bfloat` there while torch
    # 2.9.1 runs it, and torch.zeros in that dtype succeeds on both, so the allocation probe alone
    # lets the pin fail with a message about torch's kernel coverage rather than about kornia's
    # convention. Probed rather than version-gated: two builds are two data points, not a history.
    # What is attempted is the PUBLIC operation, on a throwaway input, and a failure becomes a skip
    # only when BOTH halves of the identification hold: the call died INSIDE the closed-form
    # inverse (innermost frame, matched by code object rather than by name or message, so a rename
    # is a re-raise and not a silent skip) AND the primitive that routine is built from raises on
    # the same device and dtype. Every other failure is re-raised, so a kornia-side regression
    # still fails the pin here rather than being skipped over. The frame half is what keeps an
    # UNRELATED RuntimeError from riding the skip: on a backend that genuinely lacks the kernel the
    # primitive probe always fails, so on its own it would turn any new failure raised before the
    # inverse -- in the guard, in normal_transform_pixel, in the chain matmul -- into a skip, and
    # the regression would be invisible exactly where the skip is live. The primitive half is what
    # keeps the skip from outliving the limitation: the day kornia's inverse stops needing `cross`,
    # these pins must run again on backends that lack it instead of skipping forever.
    # Residual, stated rather than hidden: this identifies the failing ROUTINE, not the individual
    # `cross` line inside it. _inverse_3x3_closed_form is three `cross` calls, a multiply-sum and a
    # divide, so a failure at one of the latter two on a backend whose `cross` is also missing
    # would still skip. Narrowing further would mean pinning line numbers in another module.
    # Only the HEALTHY verdict is memoized, keyed by (device, dtype): a route that works is a
    # property of the build, and four pins ask this question in every test configuration, each
    # paying a matmul and a 3x3 inverse for the answer. A failing route is deliberately never
    # memoized -- it has to be re-raised with its own traceback every time, and it is the branch
    # this helper exists for.
    # Imported here rather than at module scope on purpose: this is a PRIVATE kornia helper, and a
    # module-level import of it would make the WHOLE file uncollectable if it is ever renamed --
    # every test in it erroring over a rename that concerns the four pins routed through here.
    # Inside the probe, the same rename is an ImportError on those four and nothing else.
    from kornia.core.utils import _inverse_3x3_closed_form

    route = (device, dtype)
    if route in _healthy_closed_form_inverse_routes:
        return
    try:
        kornia.geometry.conversions.normalize_homography(torch.eye(3, device=device, dtype=dtype)[None], (2, 2), (2, 2))
    except (RuntimeError, NotImplementedError) as err:
        innermost = _innermost_frame(err)
        died_in_the_closed_form_inverse = (
            innermost is not None and innermost.tb_frame.f_code is _inverse_3x3_closed_form.__code__
        )
        if died_in_the_closed_form_inverse and _cross_is_unavailable(device, dtype):
            pytest.skip(f"torch.linalg.cross has no {dtype} kernel on device {device}: {err}")
        raise
    _healthy_closed_form_inverse_routes.add(route)


@cache
def _cross_is_unavailable(device: torch.device, dtype: torch.dtype) -> bool:
    # "Can this build run torch.linalg.cross here?", for the helper above and the pin below.
    # A dtype the backend cannot even allocate (mps rejects float8 outright) counts as unavailable
    # rather than propagating as an error, and that half of the question is answered by the shared
    # probe rather than by a second copy of it, so both helpers classify such a backend the same
    # way. Cached for the same reason the probe is: it is a property of the build.
    if _dtype_allocation_error(device, dtype) is not None:
        return True
    try:
        probe = torch.ones(1, 3, device=device, dtype=dtype)
        torch.linalg.cross(probe, probe, dim=-1)
    except (RuntimeError, NotImplementedError, TypeError):
        return True
    return False


def test_skip_probe_re_raises_everything_it_cannot_identify(monkeypatch):
    # Direct pin for _skip_if_closed_form_inverse_unavailable above, for the same reason
    # test_guard_classifier_reads_the_raising_instruction pins the guard classifier: four pins
    # route their "does normalize_homography work here at all" question through that helper, and if
    # it starts skipping too readily they go quiet instead of failing, which is the mode a skip
    # helper fails in. Nothing else in this file would notice.
    # Case B needs a REAL kernel gap rather than a patched `cross`: patching it with a Python
    # function puts that function in the innermost frame, which is precisely what the helper reads,
    # so a patch cannot reproduce the branch it is meant to exercise. torch has no `cross` kernel
    # for bool or float8 on cpu (executed, torch 2.9.1: NotImplementedError, and the route dies
    # inside _inverse_3x3_closed_form), which is the same shape as the mps bfloat16 gap on torch
    # 2.5.1 that the helper exists for, reachable on the default device without that build. The
    # candidate list is searched rather than asserted: mps DOES have a bool `cross`, so a build
    # that grows the missing kernels must make this case skip visibly, not fail.
    # Cases C and D patch normal_transform_pixel, which normalize_homography calls BEFORE the
    # inverse, so the route dies without ever reaching `cross` and the patch stays out of the
    # frame the helper reads. D is the one that matters: it is exactly C on a backend that also
    # lacks the kernel, which a probe of the primitive alone cannot tell apart from a real gap --
    # it would skip there, and a kornia-side regression would be invisible on exactly the backends
    # where the skip is live.
    # The helper's memo and the two cached probes are cleared first: they are performance
    # shortcuts, and a pin whose whole subject is which branch the helper takes has to run the
    # branches rather than a remembered verdict from an earlier test.
    # The two globals that cases C and D patch go through monkeypatch rather than by hand: this is
    # the one pin in the file that writes to torch and to the kornia module itself, and a leak from
    # here would follow every later test in the process, so restoration belongs to pytest rather
    # than to a nest of try/finally blocks that has to be read to be trusted.
    cpu = torch.device("cpu")
    conversions = kornia.geometry.conversions
    _healthy_closed_form_inverse_routes.clear()
    _cross_is_unavailable.cache_clear()
    _dtype_allocation_error.cache_clear()

    _skip_if_closed_form_inverse_unavailable(cpu, torch.float32)  # A: healthy route, must not skip

    # torch.float8_e4m3fn was added in torch 2.1; kornia declares torch>=2.0.0, so it is probed
    # through getattr rather than referenced unconditionally.
    candidate_dtypes = (torch.bool, getattr(torch, "float8_e4m3fn", None))
    unsupported = next(
        (dtype for dtype in candidate_dtypes if dtype is not None and _cross_is_unavailable(cpu, dtype)),
        None,
    )
    if unsupported is None:
        pytest.skip("no dtype without a torch.linalg.cross cpu kernel on this build")
    with pytest.raises(pytest.skip.Exception):  # B: the gap the helper exists for
        _skip_if_closed_form_inverse_unavailable(cpu, unsupported)

    def regression(*args, **kwargs):
        raise RuntimeError("kornia-side regression")

    def assert_the_injected_failure_propagates(case: str) -> None:
        # NOT pytest.raises: a skip is a BaseException that pytest.raises(RuntimeError) lets
        # through, so the regression this pin exists to catch would turn the pin itself yellow
        # instead of red -- the same going-quiet the helper's own over-eager skip causes, one level
        # up.
        try:
            _skip_if_closed_form_inverse_unavailable(cpu, torch.float32)
        except pytest.skip.Exception as skipped:
            raise AssertionError(f"{case}: an unrelated failure was skipped over: {skipped}") from skipped
        except RuntimeError as err:
            assert "kornia-side regression" in str(err), f"{case}: wrong error re-raised: {err}"
        else:
            raise AssertionError(f"{case}: the injected failure did not propagate at all")

    # Cleared again: case A memoized (cpu, float32) as healthy, and C/D have to reach the body.
    _healthy_closed_form_inverse_routes.clear()
    monkeypatch.setattr(conversions, "normal_transform_pixel", regression)
    assert_the_injected_failure_propagates("C, cross available")

    monkeypatch.setattr(torch.linalg, "cross", regression)
    assert_the_injected_failure_propagates("D, cross unavailable too")


# The four deprecated aliases of this module, as (deprecated name, replacement name, call input).
# One table serves all three pins that iterate them -- the alias-forwarding pin in
# TestAngleAxisToQuaternion and the two module-level kornia#3956 pins at the end of this file -- so
# that a fifth alias, or a retired one, is a one-line edit in one place. The third column is a call
# INPUT, not a pinned expected value: each pin computes its own expectation from it, so sharing the
# table shares no literal.
_DEPRECATED_ALIASES = [
    ("angle_axis_to_rotation_matrix", "axis_angle_to_rotation_matrix", [[0.1, 0.2, 0.3]]),
    (
        "rotation_matrix_to_angle_axis",
        "rotation_matrix_to_axis_angle",
        [
            [0.5357142857142858, -0.6229365034008422, 0.5700529070291328],
            [0.765793646257985, 0.6428571428571429, -0.01716931065742361],
            [-0.3557671927434186, 0.4457407392288521, 0.8214285714285714],
        ],
    ),
    ("quaternion_to_angle_axis", "quaternion_to_axis_angle", [1.0, 2.0, 3.0, 4.0]),
    ("angle_axis_to_quaternion", "axis_angle_to_quaternion", [0.1, 0.2, 0.3]),
]

# The projection the two module-level kornia#3956 pins parametrize over -- both read the alias name
# and the call input and neither reads the replacement -- plus the ids both label their cells with.
# Written once here rather than at each decorator: the table above being maintained in one place is
# only true if what is derived from it is too, and an id suffix added to one copy of a duplicated
# comprehension drifts silently from the other.
_DEPRECATED_ALIAS_NAMES_AND_ARGS = [(alias_name, arg) for alias_name, _, arg in _DEPRECATED_ALIASES]
_DEPRECATED_ALIAS_IDS = [row[0] for row in _DEPRECATED_ALIASES]


# Exception types that mean "kornia's OWN guard rejected the input", for the kornia#3959 and
# kornia#3960 pins below: kornia's guards raise BaseError subclasses (KORNIA_CHECK_SHAPE's ShapeError, plain
# KORNIA_CHECK's bare BaseError, TypeCheckError -- BaseError is not a ValueError subclass, so both
# entries are needed) or the hand-rolled ValueError the three homography functions raise today,
# while every torch shape failure in these call chains (matmul, linalg.inv) raises RuntimeError.
# Deliberately type-only -- no message text, which may be reworded on either side. This tuple is
# one input to _raised_by_a_kornia_guard below, which every #3959/#3960 pin classifies through, so
# the strict xfail and its companion warts cannot drift apart.
_KORNIA_GUARD_EXCEPTIONS = (BaseError, ValueError)


def _raised_by_a_kornia_guard(err: BaseException) -> bool:
    # "Did kornia reject this input itself, or did it reach downstream arithmetic and die there?"
    # -- the question every wart pin below has to answer, and the reason a type test alone is not
    # enough: a guard written as a literal `raise RuntimeError(...)` is indistinguishable BY TYPE
    # from the torch matmul/linalg failures these warts pin, so such a fix would land with the
    # strict xfail still XFAIL and the companion warts still green.
    # The discriminator is the BYTECODE INSTRUCTION the innermost traceback frame died on, read
    # through tb_lasti. A Python `raise` statement compiles to RAISE_VARARGS (RERAISE when an
    # exception is re-raised) whatever its source formatting -- `raise X`, `raise(X)`, a raise
    # split across lines, or a KORNIA_CHECK* helper raising inside kornia/core/check.py -- while
    # an exception surfacing out of a C call lands on the call site itself: BINARY_OP for
    # `... @ ...` and `H[..., -1, -1] += 1.0`, CALL for `torch.linalg.inv(...)`. Matching the
    # source line for "raise " instead of the bytecode would be formatting-sensitive rather than
    # semantic -- it would miss `raise(RuntimeError(...))`, whose missing space before the
    # parenthesis is not a semantic difference.
    # Both directions are exercised: test_guard_classifier_reads_the_raising_instruction below
    # covers the classifier itself, the #3960 convention pin asserts this is True (in any guard
    # style), and the #3959 warts assert it is False.
    # The instruction DECIDES; the type tuple is only the fallback for a frame whose instruction
    # cannot be read. ORing the two instead of falling back would make the instruction check
    # unreachable for exactly the types kornia's guards raise, so it would score any downstream
    # ValueError as a guard -- `normalize_homography(eye(3)[None], (4, 5, 6), (8, 9))` dies at the
    # UNPACK_SEQUENCE of `src_h, src_w = dsize_src` with `too many values to unpack`, which is
    # torch-free but is not a guard either. ANDing them instead would be worse, not better: it
    # would reject the literal `raise RuntimeError(...)` guard that is the whole reason this
    # function exists (executed -- three of the six positives below are RuntimeError raised
    # through `raise`).
    innermost = _innermost_frame(err)
    if innermost is None:
        return isinstance(err, _KORNIA_GUARD_EXCEPTIONS)
    raising_opname = next(
        (
            instruction.opname
            for instruction in dis.get_instructions(innermost.tb_frame.f_code)
            if instruction.offset == innermost.tb_lasti
        ),
        None,
    )
    if raising_opname is None:
        return isinstance(err, _KORNIA_GUARD_EXCEPTIONS)
    return raising_opname in ("RAISE_VARARGS", "RERAISE")


def test_guard_classifier_reads_the_raising_instruction():
    # Direct pin for _raised_by_a_kornia_guard above. Every #3959/#3960 pin's "a guard fix cannot
    # land unnoticed" guarantee rests on that one function, and nothing else in this file would
    # notice it regressing -- the pins would simply go quiet, which is the failure mode the
    # guarantee exists to prevent. `raise (X)` is in the positive set because it is semantically
    # identical to `raise X` -- only the formatting differs, and a classifier keyed on source text
    # rather than bytecode would not see that.
    # The negative set is the two shapes of downstream failure the warts actually pin, a mixed
    # matmul and a singular inverse, plus a ValueError raised at a non-`raise` instruction: those
    # two are RuntimeError and so can never reach the type tuple, which would leave the ONE
    # combination that breaks -- a guard TYPE at a downstream instruction -- uncovered. It is not
    # hypothetical for these functions: normalize_homography opens with `src_h, src_w = dsize_src`,
    # so a 3-tuple dsize raises ValueError from UNPACK_SEQUENCE inside kornia and must still
    # classify as NOT a guard. Device-independent by construction: default-device tensors, no
    # fixture, and only the raising instruction is read.
    def spaced_raise():
        raise RuntimeError("guard")

    def parenthesised_raise():
        raise (RuntimeError("guard"))

    def raise_a_bound_name():
        error = RuntimeError("guard")
        raise error

    def kornia_shape_check():
        KORNIA_CHECK_SHAPE(torch.eye(4)[None], ["B", "3", "3"])

    def kornia_value_check():
        KORNIA_CHECK(False, "guard")

    def hand_rolled_value_error():
        raise ValueError("Input dst_pix_trans_src_pix must be a Bx3x3 tensor")

    def mixed_matmul_failure():
        return torch.eye(3, dtype=torch.int64)[None] @ torch.eye(3, dtype=torch.float32)[None]

    def singular_inverse_failure():
        return torch.linalg.inv(torch.zeros(1, 3, 3))

    def downstream_value_error():
        # ValueError, but raised by UNPACK_SEQUENCE rather than by a `raise`.
        first, second, third = [1, 2]  # noqa: F841

    for guard in (
        spaced_raise,
        parenthesised_raise,
        raise_a_bound_name,
        kornia_shape_check,
        kornia_value_check,
        hand_rolled_value_error,
    ):
        with pytest.raises(Exception) as excinfo:
            guard()
        assert _raised_by_a_kornia_guard(excinfo.value), (
            f"{guard.__name__} rejects the input on kornia's side and must classify as a guard"
        )

    for downstream in (mixed_matmul_failure, singular_inverse_failure, downstream_value_error):
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            downstream()
        assert not _raised_by_a_kornia_guard(excinfo.value), (
            f"{downstream.__name__} does not reject the input at a kornia guard and must not classify as one"
        )


# The wrong-sized-matrix cells for the kornia#3960 convention pin: (op name, wrong square size).
# Module-level rather than inline because a fourth op (a denormalize_homography3d per kornia#3962,
# say) becomes a one-line edit that reaches the wrong-size pin and the rank pin together instead of
# landing in one and not the other. The sizes helper lives here for the same reason.
# These are call INPUTS, not pinned expected values: the pin computes its own verdict from them.
_WRONG_SIZE_CASES = [
    ("normalize_homography", 4),
    ("denormalize_homography", 4),
    ("normalize_homography3d", 3),
]
# Derived, not a second hand-maintained list: a length mismatch would fail at collection, but a
# reorder of the table above would silently relabel the cells (a `[denormalize]` id reporting a
# normalize_homography failure). Yields ["normalize", "denormalize", "normalize3d"] today.
_WRONG_SIZE_IDS = [op_name.replace("_homography", "") for op_name, _ in _WRONG_SIZE_CASES]
# The same three ops as _WRONG_SIZE_CASES, without the per-op wrong size: the rank pin below
# derives its own sizes from the op name and only needs the names. Derived from the table above
# rather than written out again so a fourth op reaches both pins from one edit.
_HOMOGRAPHY_OP_NAMES = [op_name for op_name, _ in _WRONG_SIZE_CASES]


def _homography_sizes(op_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # (dsize_src, dsize_dst) for the kornia#3960 cells: 3-D ops take (depth, height, width).
    return ((2, 4, 5), (3, 8, 9)) if op_name.endswith("3d") else ((4, 5), (8, 9))


# The asymmetric camera pose shared by the camera-frame pins: a proper rotation (det = +1) that is
# the 120-degree turn about (1, 1, 1), so it is NOT equal to its own transpose and NOT symmetric --
# an identity or symmetric pose is invariant under the very flips and transposes these pins exist to
# catch. Every entry is 0 or 1 and the translation is small integers, so every product below is
# exact at float16, bfloat16, float32 and float64 alike, which is what lets those pins compare at
# atol=rtol=0 under the dtype fixture. Materialised per test through _asymmetric_pose below, which
# builds FRESH tensors on every call so no tensor is shared between tests; the values are kept as
# plain nested lists for the same reason. Used by TestRt2Extrinsics, TestCamtoworldGraphicsToVision
# and TestCamtoworldRtToPoseRt -- one definition and one materialisation site instead of nine
# copies that would have to be edited in lockstep, and it is what makes those classes'
# "same asymmetric pose as TestRt2Extrinsics" cross-references true by construction.
# The kornia#3961 wart deliberately uses a DIFFERENT, non-orthogonal rotation and stays out of this.
_ASYMMETRIC_R = [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
_ASYMMETRIC_T = [[[1.0], [2.0], [3.0]]]


def _asymmetric_pose(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    # (rotation, translation) of the asymmetric pose above, as fresh tensors on every call.
    return (
        torch.tensor(_ASYMMETRIC_R, device=device, dtype=dtype),
        torch.tensor(_ASYMMETRIC_T, device=device, dtype=dtype),
    )


# Shared inputs for the same reason as _ASYMMETRIC_R above: pins cross-reference each other as
# using "the same input", and a shared definition makes that true by construction instead of by
# inspection of copied literals. Plain nested lists, materialised per test, no tensor shared.
# _DIRECTION_H feeds TestNormalizeHomography's composition/direction/per-sample pins (affine, zero
# projective row, asymmetric); _ROUND_TRIP_H feeds its round-trip and per-sample pins (projective,
# chosen so that at dyadic sizes every intermediate of the round trip is exactly representable);
# _ARKIT_WORKED_QVEC and _ARKIT_WORKED_TVEC are the ARKit worked-example (q, t) pair that three
# TestCARKitToColmap pins anchor to one another -- both halves are shared so that a change to the
# worked example cannot leave one pin silently testing a different pose.
_DIRECTION_H = [[[2.0, 0.5, 2.0], [-0.25, 1.0, 1.0], [0.0, 0.0, 1.0]]]
_ROUND_TRIP_H = [[[1.25, 0.25, 4.0], [-0.5, 0.75, 2.0], [0.0625, 0.125, 1.0]]]
_ARKIT_WORKED_QVEC = [[0.0, 1.0, 0.0, 1.0]]
_ARKIT_WORKED_TVEC = [[[1.0], [1.0], [1.0]]]


@contextmanager
def _ambient_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    # Swap the PROCESS-WIDE torch default dtype for the duration of a with-block, restoring it even
    # if the body raises. The finally-restore is the safety-critical part -- a leaked float64
    # default would silently change every later test's tolerances across the whole suite -- so it
    # lives in one place instead of being hand-rolled at each ambient-default pin (the two #3958
    # pins today; any future one should use this too).
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _assert_proper_rotation(rotation: torch.Tensor) -> None:
    # det(R) == +1 backs the handedness-is-preserved Convention bullets, so it is asserted even
    # where an earlier bitwise pin on a 0/+-1 literal already implies it -- the det claim is
    # documented on its own and deserves its own loud flip. Computed after upcasting to float32:
    # torch.linalg.det is not implemented for float16/bfloat16 on all backends, and the upcast of
    # an exactly-representable matrix is lossless. atol=1e-5 covers float32 det round-off on
    # near-0/+-1 entries while a reflection (det = -1) misses by 2.
    assert_close(
        torch.linalg.det(rotation.to(torch.float32)),
        torch.ones(rotation.shape[0], device=rotation.device, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )


def _assert_strictly_batched(op_name: str, shapes: tuple[tuple[int, ...], ...], device: torch.device) -> None:
    # Shared body for the three test_convention_shapes_are_strictly_batched pins (TestRt2Extrinsics,
    # TestCamtoworldGraphicsToVision, TestCamtoworldRtToPoseRt), whose executable bodies were
    # line-for-line identical -- only their parametrize tables differ. One definition means a change
    # to the assertion policy is one edit, not three synchronized ones; each class keeps its own
    # parametrized test, so collect IDs and per-class failure reporting are unchanged.
    # ShapeError (kornia's own) is asserted rather than the message text: within the call chains
    # these three pins exercise, KORNIA_CHECK_SHAPE is the only thing that raises it (kornia
    # raises ShapeError elsewhere too, e.g. directly in kornia/color/yuv.py, so this is a scoped
    # claim, not a global one -- re-scope it before copying this helper to a new surface), so the
    # type is the evidence that kornia's guard fired and not some downstream arithmetic, and the
    # wording stays free to change.
    # float32 is hardcoded and the dtype fixture dropped: a shape guard runs before any arithmetic,
    # so which shapes are rejected cannot depend on the dtype and the fixture only multiplied cells.
    # TestCARKitToColmap's variant is deliberately NOT routed through here: it is unparametrized and
    # classifies a ValueError branch as well, so it is a different assertion, not a copy of this one.
    op = getattr(kornia.geometry.conversions, op_name)
    args = [torch.zeros(shape, device=device, dtype=torch.float32) for shape in shapes]

    with pytest.raises(ShapeError):
        op(*args)


class TestAngleAxisToQuaternion(BaseTester):
    # based on:
    # https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/rotation_test.cc#L271

    def test_smoke(self, device, dtype):
        axis_angle = torch.zeros(3, dtype=dtype, device=device)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.shape == (4,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        axis_angle = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.shape == (batch_size, 4)

    def test_zero_angle(self, device, dtype, atol, rtol):
        axis_angle = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        axis_angle = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        axis_angle = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        axis_angle = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        expected = torch.tensor((np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        axis_angle = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, half_sqrt2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        axis_angle = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, half_sqrt2, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        half_sqrt2 = 0.5 * np.sqrt(2.0)
        axis_angle = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        expected = torch.tensor((half_sqrt2, 0.0, 0.0, half_sqrt2), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("input_dtype", (torch.int16, torch.int32, torch.int64, torch.uint8))
    def test_convention_integer_input_is_promoted_to_float_3948(self, input_dtype, device):
        # Convention, and the answer kornia#3948 settled: an integer axis-angle is PROMOTED, not
        # rejected. The output buffer used to be allocated with dtype=axis_angle.dtype, so an
        # integer input got an integer buffer and every component was truncated on the way in --
        # the function returned tensor([0, 0, 0, 0]), not merely a wrong quaternion but a zero-norm
        # one, with no error and no warning. The buffer now takes its dtype from the computed
        # values (sqrt already promotes them), so the result is the float quaternion below.
        # The wart this replaces said the intended behavior was undecided -- promote, or raise a
        # TypeError the way a dtype guard would. Promotion is the choice; the sibling wart
        # TestNormalTransformPixel.test_wart_integer_dtype_truncates_the_scale_to_zero_3959 pins
        # the same family of defect and is still open, so this pin is also the precedent it should
        # follow.
        # Four cells because the promotion happens through torch's own type rules rather than an
        # explicit .float(): a signed/unsigned or narrow/wide difference would show up here.
        # The dtype fixture is dropped because the claim is about the input dtype itself.
        # Snippet used to generate expected (torch + stdlib, executed on cpu):
        #   axis_angle_to_quaternion(torch.tensor([1., 0., 0.])) ->
        #     [0.8775825500488281, 0.4794255495071411, 0.0, 0.0]      (float32 reference)
        #   math.cos(0.5), math.sin(0.5) -> (0.8775825618903728, 0.479425538604203)
        axis_angle = torch.tensor((1, 0, 0), device=device, dtype=input_dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.is_floating_point()
        expected = torch.tensor((np.cos(0.5), np.sin(0.5), 0.0, 0.0), device=device, dtype=quaternion.dtype)
        self.assert_close(quaternion, expected, atol=1.0e-4, rtol=1.0e-4)

    @pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
    def test_convention_float_input_keeps_its_dtype_3948(self, input_dtype, device):
        # The other side of the #3948 buffer change, and the regression it could have introduced:
        # taking the buffer dtype from the computed values must NOT widen a float input. float16
        # and bfloat16 are the cells that matter -- a fix written as `.float()` or as an explicit
        # torch.float32 buffer would silently upcast them, which is exactly the half-precision
        # surface kornia treats as its own. The tolerance is set for float16, the widest of the
        # four; the float32/float64 values are pinned exactly by the numerical tests above.
        if device.type == "mps" and input_dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        axis_angle = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=input_dtype)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert quaternion.dtype == input_dtype
        expected = torch.tensor((np.cos(0.5), np.sin(0.5), 0.0, 0.0), device=device, dtype=input_dtype)
        self.assert_close(quaternion, expected, atol=1.0e-2, rtol=1.0e-2)

    def test_convention_the_guarded_sqrt_moves_no_forward_value_3949(self, device, dtype):
        # The #3949 fix is a BACKWARD-pass fix: away from theta == 0 the guarded sqrt must return
        # the very same BITS as the plain `sqrt(theta_squared)` formula it replaced, so no existing
        # forward value moves. torch.equal, not assert_close, for exactly that reason -- a
        # tolerance would hide the reassociation this pin exists to rule out.
        # The fourth row is the near-singular one, and it is derived from the dtype rather than
        # written as a literal: a fixed 1e-9 UNDERFLOWS to 0 at float16 (its smallest subnormal is
        # 5.96e-8), which puts the row on the guarded side of the branch, makes the reference
        # formula's own sin(0)/0 a NaN, and fails this pin under `pixi run test-half` while passing
        # everywhere else. finfo(dtype).eps squares to something representable in all four dtypes,
        # so the row stays near-singular AND stays on the unguarded side in each.
        # Companion: test_convention_gradients_at_the_identity_are_finite_3949 covers theta == 0
        # itself, where the two formulas deliberately DISAGREE.
        near_singular = torch.finfo(dtype).eps
        axis_angle = torch.tensor(
            ((0.1, 0.2, 0.3), (1.0, 0.0, 0.0), (3.0, -2.0, 0.5), (near_singular, 0.0, 0.0)),
            device=device,
            dtype=dtype,
        )
        a0, a1, a2 = axis_angle[..., 0:1], axis_angle[..., 1:2], axis_angle[..., 2:3]
        theta = torch.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        half_theta = theta * 0.5
        k = torch.sin(half_theta) / theta
        expected = torch.cat((torch.cos(half_theta), a0 * k, a1 * k, a2 * k), dim=-1)
        quaternion = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        assert torch.equal(quaternion, expected)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        axis_angle = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.axis_angle_to_quaternion), (axis_angle,))

    def test_convention_theta_beyond_pi_returns_the_w_negative_half(self, device, dtype):
        # Convention pin: axis_angle_to_quaternion applies w = cos(theta/2) and
        # (x, y, z) = sin(theta/2) * axis verbatim, with NO canonicalisation to w >= 0, so any
        # theta > pi comes back in the w < 0 half of the double cover. A full turn about +x gives
        # (-1, 0, 0, 0) -- the same rotation as the identity quaternion (1, 0, 0, 0) that a
        # canonicalising implementation would return, but not the same four numbers. The second
        # case is off-axis (theta = 4 rad about (1, 2, 3)/sqrt(14)) so that a sign flip on any
        # single component is caught as well.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.cos(math.pi), math.sin(math.pi) -> (-1.0, 1.2246467991473532e-16)
        #   n = math.sqrt(14.0); axis = (1 / n, 2 / n, 3 / n); theta = 4.0
        #   [theta * a for a in axis]
        #     -> [1.0690449676496976, 2.138089935299395, 3.2071349029490928]
        #   [math.cos(theta / 2)] + [math.sin(theta / 2) * a for a in axis]
        #     -> [-0.4161468365471424, 0.24301995956120354, 0.48603991912240707, 0.7290598786836107]
        full_turn = kornia.geometry.conversions.axis_angle_to_quaternion(
            torch.tensor([6.283185307179586, 0.0, 0.0], device=device, dtype=dtype)
        )
        # Asserted structurally rather than through assert_close against [-1, 0, 0, 0]: the vector
        # part is sin(theta/2) at theta = 2*pi, whose error is ~1 ulp of the dtype, and at float16
        # that is 9.675e-04 against the shared float16 atol of 1e-3 -- 96.75% of the budget, so one
        # extra ulp (a backend doing native half-precision sin rather than upcasting, e.g. CUDA)
        # turns this green cell red. Default CI runs float32/float64 only, so it is unexercised
        # today. What the pin is actually about is the *convention* -- w = cos(theta/2) with no
        # canonicalisation to w >= 0 -- and that part is exact at every dtype, so w is compared
        # exactly and the vector part is bounded by a few ulp instead.
        assert full_turn[0].item() == -1.0, (
            f"axis_angle_to_quaternion no longer returns the w < 0 half at theta = 2*pi "
            f"(got w = {full_turn[0].item()!r}); a canonicalising implementation would give +1"
        )
        vector_tol = 4 * torch.finfo(dtype).eps
        assert full_turn[1:].abs().max().item() <= vector_tol, (
            f"axis_angle_to_quaternion vector part at theta = 2*pi is no longer zero to a few ulp "
            f"(got {full_turn[1:].tolist()}, tolerance {vector_tol})"
        )

        off_axis = kornia.geometry.conversions.axis_angle_to_quaternion(
            torch.tensor([1.0690449676496976, 2.138089935299395, 3.2071349029490928], device=device, dtype=dtype)
        )
        self.assert_close(
            off_axis,
            torch.tensor(
                [-0.4161468365471424, 0.24301995956120354, 0.48603991912240707, 0.7290598786836107],
                device=device,
                dtype=dtype,
            ),
        )

    def test_convention_axis_angle_quaternion_roundtrip_is_exact_in_float64(self, device):
        # Convention pin: axis_angle_to_quaternion and quaternion_to_axis_angle are exact inverses
        # in float64 over the whole [0, pi] range, including the two singular points theta = 0 and
        # theta = pi. This is what separates the quaternion leg from the matrix leg: the same
        # round-trip through axis_angle_to_rotation_matrix is only accurate to ~1e-6 even in
        # float64 (see TestRotationMatrixToAngleAxis.test_convention_axis_angle_roundtrip_
        # tolerance_is_1e_6_in_float64), so "the round-trip is exact" is a statement about this
        # pair only. float64 is hardcoded and the dtype fixture dropped because exactness is a
        # float64-only claim; MPS is skipped visibly because it has no float64 at all.
        # Snippet used to generate the inputs (stdlib only):
        #   import math
        #   n = math.sqrt(14.0); axis = (1 / n, 2 / n, 3 / n)
        #   [[theta * a for a in axis] for theta in (0.0, 1e-3, 0.7, math.pi)]
        # Measured max |roundtrip - input| at those four thetas (torch 2.9.1, cpu float64):
        #   0.0, 2.168404344971009e-19, 0.0, 0.0 -- so atol 1e-12 is ~7 orders above the worst
        #   observed error and ~6 orders below the 1e-6 the matrix leg would give. 1e-12 rather
        #   than something tighter because this composes two atan2/sqrt chains and no CUDA job
        #   exists to tell us what they cost there.
        _skip_if_dtype_unavailable(device, torch.float64)

        axis_angle = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0002672612419124244, 0.0005345224838248488, 0.0008017837257372733],
                [0.18708286933869706, 0.3741657386773941, 0.5612486080160912],
                [0.839625954181357, 1.679251908362714, 2.518877862544071],
            ],
            device=device,
            dtype=torch.float64,
        )

        roundtrip = kornia.geometry.conversions.quaternion_to_axis_angle(
            kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
        )

        self.assert_close(roundtrip, axis_angle, atol=1e-12, rtol=0.0)

    # Convention pin for the four deprecated aliases (they have no test class and no docstring of
    # their own; this class is named after one of them). Each still works and is a thin wrapper:
    # it emits a DeprecationWarning naming both the old and the new symbol, and returns output
    # that is bit-identical to the replacement's. The replacement, not the alias, is where the
    # Convention block lives.
    # The call is wrapped in warnings.catch_warnings() because invoking a kornia deprecated symbol
    # rewrites the process-global DeprecationWarning filters; pytest.warns alone does not restore
    # them, so without the wrapper this pin would leak filter state into every later test.
    # Snippet used to generate expected (torch only):
    #   import warnings, kornia.geometry.conversions as C
    #   with warnings.catch_warnings(record=True) as w:
    #       warnings.simplefilter("always")
    #       out = C.angle_axis_to_quaternion(torch.tensor([0.1, 0.2, 0.3]))
    #   w[0].category, str(w[0].message) -> DeprecationWarning, 'Since kornia 0.7.0 the
    #     `angle_axis_to_quaternion` is deprecated in favor of `axis_angle_to_quaternion`.'
    #   torch.equal(out, C.axis_angle_to_quaternion(torch.tensor([0.1, 0.2, 0.3]))) -> True
    # The cases come from the module-level _DEPRECATED_ALIASES table, shared with the two #3956 pins
    # at the end of this file. The dtype fixture stays: alias == replacement is a contract about
    # whatever the two implementations are, not about today's one-line forward -- a future rewrite
    # that kept a frozen copy behind the alias could diverge at a single dtype, and only a leg at
    # that dtype would see it. The comparison is
    # torch.testing.assert_close at rtol=atol=0 (exact, and it checks shape/dtype/device itself)
    # rather than torch.equal, which reports no per-element diff on a mismatch. NaN is excluded
    # outright by the assertion above the comparison rather than tolerated through equal_nan=True:
    # all four rows are finite at every dtype today, so equal_nan could only ever matter for an
    # alias and a replacement that BOTH regressed to NaN, which is a pass this pin should not
    # hand out.
    @pytest.mark.parametrize(("deprecated_name", "replacement_name", "arg"), _DEPRECATED_ALIASES)
    def test_convention_deprecated_alias_warns_and_matches_replacement(
        self, device, dtype, deprecated_name, replacement_name, arg
    ):
        deprecated = getattr(kornia.geometry.conversions, deprecated_name)
        replacement = getattr(kornia.geometry.conversions, replacement_name)
        tensor = torch.tensor(arg, device=device, dtype=dtype)

        expected = replacement(tensor)

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with pytest.warns(
                DeprecationWarning, match=f"`{deprecated_name}` is deprecated in favor of `{replacement_name}`"
            ):
                actual = deprecated(tensor)

        assert not actual.isnan().any(), (
            f"`{deprecated_name}` returned NaN; an exact comparison of two NaNs is not a match this pin should accept"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # (op name, input at the identity rotation, expected gradient of the output sum) for the #3949
    # pin below. The gradients are analytic, not measured: around theta == 0 axis_angle_to_quaternion
    # is v -> (1, v/2) so d(sum)/dv = (0.5, 0.5, 0.5), and around the identity quaternion_to_axis_angle
    # is q -> 2*(x, y, z) so d(sum)/dq = (0, 2, 2, 2) -- the w component does not reach the output.
    # Pinning the VALUES and not just finiteness is what separates "the NaN is gone" from "the NaN
    # was replaced by the right number": a guard that clamped the radicand to 1e-12 the way
    # axis_angle_to_rotation_matrix does would also be finite here, and wrong by a factor of the
    # clamp. One table, two cells, because the two functions have independent sqrt guards.
    _IDENTITY_GRADIENT_CASES = [
        ("quaternion_to_axis_angle", [1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 2.0, 2.0]),
        ("axis_angle_to_quaternion", [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
    ]

    @pytest.mark.parametrize(("op_name", "arg", "expected_grad"), _IDENTITY_GRADIENT_CASES)
    def test_convention_gradients_at_the_identity_are_finite_3949(self, device, op_name, arg, expected_grad):
        # Convention: both directions of the axis-angle/quaternion pair are differentiable at the
        # identity rotation -- the point every optimiser initialises at and converges to. Until
        # kornia#3949 was fixed they were not: each took an unguarded sqrt of a quantity that is
        # exactly 0 there, so the backward pass divided by 0 and the gradient was NaN for the whole
        # input (quaternion_to_axis_angle at q = (1,0,0,0), axis_angle_to_quaternion at aa = (0,0,0)).
        # The fix keeps the singular point away from the sqrt with a `where` on the radicand rather
        # than clamping it, so the forward value at the identity is unchanged -- pinned separately
        # by test_convention_the_guarded_sqrt_moves_no_forward_value_3949 in each class.
        # Three claims per cell, in order:
        #   (1) the gradient is finite -- the #3949 headline;
        #   (2) it equals the analytic value from _IDENTITY_GRADIENT_CASES, so a merely-finite
        #       wrong number (a clamped radicand) does not pass;
        #   (3) the guard is ELEMENTWISE: in a batch holding an identity row and an ordinary one,
        #       the identity row gets the identity gradient AND the ordinary row gets exactly the
        #       gradient it gets on its own. Not a NaN-containment claim -- NaN never crossed rows,
        #       measured on the unfixed code, since the ops are elementwise and .sum() sends an
        #       independent gradient to each. What it rules out is a guard written as a PYTHON
        #       branch (`if theta_squared > 0: ...`), which takes one branch for the whole tensor:
        #       that is the shape a naive fix takes, it passes both scalar cells, and it is wrong
        #       for every mixed batch. The ordinary row's own gradient is measured in the test
        #       rather than pinned as a literal, so this cell stays a statement about
        #       batched-vs-unbatched agreement and not a second copy of the numerical pins.
        # Both legs live in this class so the pair stays one edit even though quaternion_to_axis_angle
        # is exercised by TestQuaternionToAngleAxis. float64 is hardcoded and the dtype fixture
        # dropped because gradient claims in this file are float64 claims (see test_rad2deg_gradcheck);
        # the NaN was not float64-specific -- float32 gave NaN too.
        _skip_if_dtype_unavailable(device, torch.float64)

        op = getattr(kornia.geometry.conversions, op_name)
        x = torch.tensor(arg, device=device, dtype=torch.float64, requires_grad=True)

        op(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), f"kornia#3949: {op_name} has a non-finite gradient at the identity"
        self.assert_close(x.grad, torch.tensor(expected_grad, device=device, dtype=torch.float64))

        # The batch leg: the identity row first, an ordinary rotation second.
        regular = [0.9, 0.1, 0.2, 0.3] if op_name == "quaternion_to_axis_angle" else [0.1, 0.2, 0.3]

        alone = torch.tensor(regular, device=device, dtype=torch.float64, requires_grad=True)
        op(alone).sum().backward()

        batch = torch.tensor([arg, regular], device=device, dtype=torch.float64, requires_grad=True)
        op(batch).sum().backward()

        assert batch.grad is not None
        assert torch.isfinite(batch.grad).all(), f"kornia#3949: {op_name} has a non-finite gradient in a batch"
        self.assert_close(batch.grad[0], torch.tensor(expected_grad, device=device, dtype=torch.float64))
        self.assert_close(batch.grad[1], alone.grad, atol=0.0, rtol=0.0)


class TestQuaternionToAngleAxis(BaseTester):
    def test_smoke(self, device, dtype):
        quaternion = torch.zeros(4, device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        assert axis_angle.shape == (3,)

    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        assert axis_angle.shape == (batch_size, 3)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi, 0.0, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((np.sqrt(3.0) / 2.0, 0.0, 0.0, 0.5), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 3.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_small_angle_x(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((theta, 0.0, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_small_angle_y(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, theta, 0.0), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_small_angle_z(self, device, dtype, atol, rtol):
        theta = 1.0e-2
        quaternion = torch.tensor((np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, theta), device=device, dtype=dtype)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        self.assert_close(axis_angle, expected, atol=atol, rtol=rtol)

    def test_convention_the_guarded_sqrt_moves_no_forward_value_3949(self, device, dtype):
        # The quaternion_to_axis_angle half of the pin documented on
        # TestAngleAxisToQuaternion.test_convention_the_guarded_sqrt_moves_no_forward_value_3949 --
        # same claim, same reason the near-singular row is derived from finfo(dtype).eps rather
        # than written as a fixed 1e-9, which underflows to 0 at float16. Kept in this class rather
        # than as a third cell of that one because the reference formula is a different expression.
        quaternion = torch.tensor(
            (
                (0.9, 0.1, 0.2, 0.3),
                (0.0, 1.0, 0.0, 0.0),
                (0.5, 3.0, -2.0, 0.5),
                (1.0, torch.finfo(dtype).eps, 0.0, 0.0),
            ),
            device=device,
            dtype=dtype,
        )
        cos_theta = quaternion[..., 0]
        q1, q2, q3 = quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
        sin_theta = torch.sqrt(q1 * q1 + q2 * q2 + q3 * q3)
        two_theta = 2.0 * torch.where(
            cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
        )
        k = two_theta / sin_theta
        expected = torch.stack((q1 * k, q2 * k, q3 * k), dim=-1)
        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)
        assert torch.equal(axis_angle, expected)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype) + eps
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_to_axis_angle), (quaternion,))

    def test_convention_double_cover_q_and_minus_q_give_the_same_axis_angle(self, device, dtype):
        # Convention pin: q and -q are the same rotation (the unit quaternions double-cover SO(3)),
        # and quaternion_to_axis_angle collapses the two onto one bit-identical vector -- it picks
        # the representative with |theta| <= pi rather than propagating the input's sign. torch.equal
        # rather than assert_close because the agreement is exact, not approximate: measured max
        # difference is 0.0 over 500 random float64 quaternions (seeded torch.Generator(6)),
        # bit-identical in 500/500 of them, and at float32/float16/bfloat16 for the pinned input.
        # Pinned on a non-unit, non-axis-aligned quaternion so no symmetry can carry the assertion.
        # Snippet used to generate expected (stdlib only, q = (1, 2, 3, 4) normalised):
        #   import math
        #   u = [v / math.sqrt(30.0) for v in (1.0, 2.0, 3.0, 4.0)]
        #   nv = math.sqrt(u[1] ** 2 + u[2] ** 2 + u[3] ** 2)
        #   theta = 2 * math.atan2(nv, u[0])
        #   [theta * u[i + 1] / nv for i in range(3)]
        #     -> [1.03038058532817, 1.5455708779922552, 2.06076117065634]
        quaternion = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)

        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)

        self.assert_close(
            axis_angle,
            torch.tensor([1.03038058532817, 1.5455708779922552, 2.06076117065634], device=device, dtype=dtype),
        )
        assert torch.equal(kornia.geometry.conversions.quaternion_to_axis_angle(-quaternion), axis_angle)

        # Second cell: at exactly w = 0 the collapse does NOT happen, so the docstring's "w != 0"
        # qualifier is falsifiable here rather than merely asserted. The branch test is
        # `cos_theta < 0.0` and `-0.0 < 0.0` is False, so q and -q take the same branch and the
        # sign of the vector part passes straight through, making the two outputs exact negations.
        # The first cell cannot catch this: its input has w > 0, and no random sweep can sample
        # w = 0. Both outputs describe the same rotation (a half turn about +x and about -x):
        # axis_angle_to_rotation_matrix of [pi, 0, 0] and of [-pi, 0, 0] differ by 2.45e-16 in
        # float64. The exact negation holds at every dtype: over 400000 random quaternions with the
        # real part zeroed, cast to float64/float32/float16/bfloat16 and restricted to inputs and
        # outputs that are finite in that dtype, there are 0 mismatches each.
        # Snippet used to generate expected (stdlib only, v = (0.6, 0.8, 0.0), |v| = 1, w = 0):
        #   import math
        #   theta = 2 * math.atan2(1.0, 0.0)   # pi
        #   [theta * a for a in (0.6, 0.8, 0.0)]
        #     -> [1.8849555921538759, 2.5132741228718345, 0.0]
        half_turn = torch.tensor([0.0, 0.6, 0.8, 0.0], device=device, dtype=dtype)

        half_turn_axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(half_turn)

        self.assert_close(
            half_turn_axis_angle,
            torch.tensor([1.8849555921538759, 2.5132741228718345, 0.0], device=device, dtype=dtype),
        )
        assert torch.equal(kornia.geometry.conversions.quaternion_to_axis_angle(-half_turn), -half_turn_axis_angle), (
            "quaternion_to_axis_angle no longer returns exact negations at w = 0"
        )

    def test_convention_quaternion_to_axis_angle_is_scale_invariant(self, device, dtype):
        # Convention pin (quaternion_to_axis_angle has no test class under its own name; this class
        # is the one that exercises it): the function does not require -- and does not check -- a
        # unit quaternion. It is homogeneous in its input, so scaling the whole quaternion leaves
        # the axis-angle vector bit-identical. The scale factors are powers of two so that the
        # scaling itself is exact in binary floating point at every dtype; the invariance is a
        # property of atan2 plus the 2*theta/||v|| factor, not of the particular numbers (verified
        # bit-identical in 500/500 random float64 quaternions, seeded torch.Generator(7), for both
        # factors -- a non-power-of-two factor such as 3 is invariant to rounding only, 82/500).
        # Contrast quaternion_exp_to_log, which does NOT normalise and is silently wrong on a
        # non-unit input.
        quaternion = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)

        axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quaternion)

        assert torch.equal(kornia.geometry.conversions.quaternion_to_axis_angle(2.0 * quaternion), axis_angle)
        assert torch.equal(kornia.geometry.conversions.quaternion_to_axis_angle(0.5 * quaternion), axis_angle)


class TestRotationMatrixToQuaternion(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix)
        assert quaternion.shape == (batch_size, 4)

    def test_identity(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_rot_x_45(self, device, dtype, atol, rtol):
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        pi_half2 = torch.cos(kornia.pi / 4.0).to(device=device, dtype=dtype)
        expected = torch.tensor((pi_half2, pi_half2, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix)
        self.assert_close(quaternion, expected, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        matrix_hat = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, matrix_hat, atol=atol, rtol=rtol)

    def test_corner_case(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(
            (
                (-0.7799533010, -0.5432914495, 0.3106555045),
                (0.0492402576, -0.5481169224, -0.8349509239),
                (0.6238971353, -0.6359263659, 0.4542570710),
            ),
            device=device,
            dtype=dtype,
        )
        quaternion_true = torch.tensor(
            (0.177614107728004, 0.280136495828629, -0.440902262926102, 0.834015488624573), device=device, dtype=dtype
        )
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        torch.set_printoptions(precision=10)
        self.assert_close(quaternion_true, quaternion, atol=atol, rtol=rtol)

    def test_cond1_180_rot_x(self, device, dtype, atol, rtol):
        # 180° rotation around X: trace < 0, m00 > m11 and m00 > m22 → activates cond_1 branch.
        # R_x(π) = diag(1, -1, -1); expected quaternion (w,x,y,z) = (0, 1, 0, 0).
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        self.assert_close(quaternion.abs(), expected.abs(), atol=atol, rtol=rtol)
        # Round-trip: convert back and verify the rotation matrix is recovered.
        mat_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(mat_back, matrix, atol=atol, rtol=rtol)

    def test_cond2_180_rot_y(self, device, dtype, atol, rtol):
        # 180° rotation around Y: trace < 0, m11 > m22 and m00 not dominant → activates cond_2 branch.
        # R_y(π) = diag(-1, 1, -1); expected quaternion (w,x,y,z) = (0, 0, 1, 0).
        eps = torch.finfo(dtype).eps
        matrix = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(matrix, eps=eps)
        self.assert_close(quaternion.abs(), expected.abs(), atol=atol, rtol=rtol)
        mat_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(mat_back, matrix, atol=atol, rtol=rtol)

    def test_all_four_branches_in_batch(self, device, dtype, atol, rtol):
        # Batch of 4 rotation matrices that each activate a different internal branch.
        # Verify consistency via round-trip: R → q → R must recover the original rotation.
        eps = torch.finfo(dtype).eps
        identity = torch.eye(3, device=device, dtype=dtype)  # trace > 0 → trace_positive_cond
        rot_x_180 = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        rot_y_180 = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        rot_z_180 = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        batch = torch.stack([identity, rot_x_180, rot_y_180, rot_z_180])  # (4, 3, 3)
        quaternions = kornia.geometry.conversions.rotation_matrix_to_quaternion(batch, eps=eps)
        mats_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternions)
        self.assert_close(mats_back, batch, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        matrix = torch.eye(3, device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.rotation_matrix_to_quaternion, eps=eps), (matrix,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)

    def test_convention_w_is_not_canonicalised_to_non_negative(self, device, dtype):
        # Convention pin: rotation_matrix_to_quaternion picks ONE of the two quaternions that
        # represent the input rotation, and the rule is NOT "return w >= 0". The branch is selected
        # by the sign of the trace:
        #   trace > 0  -> the returned w is >= 0 (0/325 negative over random rotations);
        #   trace <= 0 -> the *dominant* of x, y, z is forced >= 0 and w may come back NEGATIVE
        #                 (6042 of the 12158 such cases in a sweep of 20000 random float64
        #                 rotations -- about half of them, not all).
        # trace = 1 + 2 * cos(theta), so trace <= 0 is exactly theta >= 120 degrees: a caller that
        # assumes a non-negative real part is safe below 120 degrees and wrong for about half of the
        # rotations above it. Both branches are pinned here, each with a rotation whose "natural"
        # quaternion has the opposite sign of w, which is exactly what a canonicalising
        # implementation would not reproduce.
        # Expected values are the true unit quaternions (computed with stdlib below), not the
        # function's own output: the returned components carry an extra ~1e-9 from the default
        # eps added inside the sqrt, which bare assert_close absorbs and no pin here asserts.
        # Snippet used to generate the matrices and the expected quaternions (stdlib only):
        #   import math
        #   n = math.sqrt(14.0)
        #   R = I + sin(theta) * K + (1 - cos(theta)) * K @ K   # Rodrigues, K = skew(axis)
        #   axis, theta = (1 / n, 2 / n, -3 / n), math.radians(170.0)   # trace -0.9696155060244165
        #     R -> [[-0.8430357706541933,  0.4227722475733091, -0.3324970918358584],
        #           [ 0.14431568185875046, -0.4177198235801487, -0.8970413217671823],
        #           [-0.5181348023122309, -0.8042224665289962,  0.29114008820992554]]
        #     the two representatives are +-[0.08715574274765814, 0.2662442321985726,
        #       0.5324884643971451, -0.7987326965957178]; |z| dominates, so the one with z >= 0
        #       is returned and its w is negative
        #   axis, theta = (1 / n, 2 / n, 3 / n), math.radians(60.0)     # trace 2.0
        #     R -> [[ 0.5357142857142858, -0.6229365034008422,  0.5700529070291328],
        #           [ 0.765793646257985,   0.6428571428571429, -0.01716931065742361],
        #           [-0.3557671927434186,  0.4457407392288521,  0.8214285714285714]]
        #     representatives +-[0.8660254037844387, 0.13363062095621217, 0.26726124191242434,
        #       0.40089186286863654]; the w >= 0 one is returned
        rot_trace_negative = torch.tensor(
            [
                [-0.8430357706541933, 0.4227722475733091, -0.3324970918358584],
                [0.14431568185875046, -0.4177198235801487, -0.8970413217671823],
                [-0.5181348023122309, -0.8042224665289962, 0.29114008820992554],
            ],
            device=device,
            dtype=dtype,
        )
        quaternion_trace_negative = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot_trace_negative)
        self.assert_close(
            quaternion_trace_negative,
            torch.tensor(
                [-0.08715574274765814, -0.2662442321985726, -0.5324884643971451, 0.7987326965957178],
                device=device,
                dtype=dtype,
            ),
        )
        assert quaternion_trace_negative[0] < 0.0
        assert quaternion_trace_negative[3] > 0.0

        rot_trace_positive = torch.tensor(
            [
                [0.5357142857142858, -0.6229365034008422, 0.5700529070291328],
                [0.765793646257985, 0.6428571428571429, -0.01716931065742361],
                [-0.3557671927434186, 0.4457407392288521, 0.8214285714285714],
            ],
            device=device,
            dtype=dtype,
        )
        quaternion_trace_positive = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot_trace_positive)
        self.assert_close(
            quaternion_trace_positive,
            torch.tensor(
                [0.8660254037844387, 0.13363062095621217, 0.26726124191242434, 0.40089186286863654],
                device=device,
                dtype=dtype,
            ),
        )
        assert quaternion_trace_positive[0] > 0.0

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="rotation_matrix_to_quaternion adds eps inside the sqrt, so the result is never a "
        "unit quaternion — kornia#3951",
        strict=True,
    )
    def test_convention_returns_a_unit_quaternion_3951(self, device):
        # Intended behavior: the quaternion returned for an exact rotation matrix is a unit
        # quaternion to the precision of the input dtype. It never is in float64: the default
        # eps = 1e-8 is added *inside* the sqrt that builds the components, so every returned
        # quaternion is inflated -- over 20000 random float64 rotations not one comes back exactly
        # unit. Measured on the identity in float64 (torch 2.9.1, cpu):
        #   rotation_matrix_to_quaternion(eye(3))          -> [1.0000000012499999, 0.0, 0.0, 0.0]
        #   ||q|| - 1                                      ->  1.2499998813808588e-09
        #   rotation_matrix_to_quaternion(eye(3), eps=0.0) -> [1.0, 0.0, 0.0, 0.0]  (exactly unit)
        # and the worst |‖q‖ - 1| over 200 random exact rotations is 2.212899197218121e-09, so
        # atol 1e-12 sits three orders below the deviation and eight above the float64 noise floor.
        # float64 is hardcoded and the dtype fixture dropped because a 1.25e-09 inflation is
        # invisible at every other dtype -- float32, float16 and bfloat16 all return exactly
        # [1, 0, 0, 0] for the identity -- so a dtype-fixture version would XPASS three quarters of
        # the time and blow up the strict mark for the wrong reason. MPS is skipped visibly since
        # it has no float64 at all. Marked xfail(strict=True) so fixing #3951 makes this XPASS and
        # forces the mark out. Companion wart: test_wart_eps_inside_the_sqrt_inflates_the_quaternion_3951.
        _skip_if_dtype_unavailable(device, torch.float64)

        quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
            torch.eye(3, device=device, dtype=torch.float64)
        )

        assert abs(quaternion.norm().item() - 1.0) < 1e-12, (
            "kornia#3951: rotation_matrix_to_quaternion did not return a unit quaternion"
        )

    def test_wart_eps_inside_the_sqrt_inflates_the_quaternion_3951(self, device):
        # Wart pin for kornia#3951, companion to the strict xfail above: assert the CURRENT
        # inflated components. Three cells, each discriminating a different fix shape:
        #   (0) the eps default itself is still 1e-8 -- the other two cells pass eps explicitly
        #       (house rule) so they cannot see a re-tuned default, which would otherwise be an
        #       invisible way to half-fix this;
        #   (1) eps=1e-8 passed explicitly still inflates the identity to 1.0000000012499999 --
        #       flips when eps moves out of the sqrt, or when the output is normalised at the end,
        #       but NOT when only the default is changed;
        #   (2) eps=0.0 returns exactly [1, 0, 0, 0] -- the control that proves eps is the cause;
        #       flips if the formula is restructured so that eps=0 no longer gives the exact
        #       answer (e.g. a clamp- or branch-shaped rewrite).
        # If any cell fails, #3951 was (partly) fixed -- flip/remove the strict xfail above. NOT a
        # contract that rotation_matrix_to_quaternion must keep returning a non-unit quaternion.
        # float64 is hardcoded for the same reason as the xfail: at float32 and below the identity
        # already comes back as exactly [1, 0, 0, 0] and there is nothing to pin.
        # Snippet used to generate expected (torch only, executed on cpu float64):
        #   rotation_matrix_to_quaternion(torch.eye(3, dtype=torch.float64), eps=1e-8)
        #     -> [1.0000000012499999, 0.0, 0.0, 0.0]
        #   rotation_matrix_to_quaternion(torch.eye(3, dtype=torch.float64), eps=0.0)
        #     -> [1.0, 0.0, 0.0, 0.0]
        # atol 1e-11 on the inflated cell sits two orders below the 1.25e-9 inflation being
        # discriminated (a fix still flips it red) and four above the 2.2e-16 ulp of 1.0, so a
        # one-ulp reassociation of the trace+1+eps sum (refactor, fusion, fma) cannot. The eps=0.0
        # cell stays exact: 3+1+0 is 4 in every association.
        _skip_if_dtype_unavailable(device, torch.float64)

        rotation_matrix_to_quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion
        assert inspect.signature(rotation_matrix_to_quaternion).parameters["eps"].default == 1e-8, (
            "kornia#3951: the eps default moved, so the literals pinned here no longer describe the default call"
        )

        identity = torch.eye(3, device=device, dtype=torch.float64)

        inflated = rotation_matrix_to_quaternion(identity, eps=1e-8)
        exact = rotation_matrix_to_quaternion(identity, eps=0.0)

        assert_close(
            inflated,
            torch.tensor([1.0000000012499999, 0.0, 0.0, 0.0], device=device, dtype=torch.float64),
            atol=1e-11,
            rtol=0.0,
            msg=_issue_msg("kornia#3951: eps=1e-8 no longer inflates the identity quaternion"),
        )
        assert_close(
            exact,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64),
            atol=0.0,
            rtol=0.0,
            msg=_issue_msg("kornia#3951: eps=0.0 no longer returns the exact unit quaternion"),
        )


class TestQuaternionToRotationMatrix(BaseTester):
    @pytest.mark.parametrize("batch_dims", ((), (1,), (3,), (8,), (1, 1), (5, 6)))
    def test_smoke_batch(self, batch_dims, device, dtype):
        quaternion = torch.zeros(*batch_dims, 4, device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        assert matrix.shape == (*batch_dims, 3, 3)

    @pytest.mark.parametrize("half_dtype", [torch.float16, torch.bfloat16])
    def test_convention_output_dtype_equals_input_dtype_3954(self, device, half_dtype):
        # Convention: the output dtype follows the input at every shape, like the rest of the
        # rotation-representation family. Until kornia#3954 was fixed that held for batched input
        # only. The function built its matrix around `one = torch.tensor(1.0)`, a 0-dim float32
        # tensor, and type promotion ranks a dimensioned tensor above a 0-dim one: the components
        # of a batched input therefore outranked the literal and kept their dtype, while the 0-dim
        # components of an unbatched (4,) input tied with it and float32 won the category. The fix
        # makes `one` a Python float, which is a wrapped scalar and does not participate in
        # promotion at all. That mechanism is why this pin is UNBATCHED -- the batched shape passed
        # both before and after and cannot detect the defect. It is carried here as the third
        # assertion anyway, so a fix that special-cased one shape would not satisfy this pin.
        # Two dtype cells because a fix could plausibly handle float16, the common half dtype, and
        # leave bfloat16 upcast. The dtypes are hardcoded and the dtype fixture dropped so both
        # cells run in every test configuration; the skip is explicit so a backend without the
        # dtype reports as skipped rather than failing on a RuntimeError.
        _skip_if_dtype_unavailable(device, half_dtype)

        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=half_dtype)

        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)

        assert matrix.dtype == half_dtype, f"kornia#3954: quaternion_to_rotation_matrix returned {matrix.dtype}"
        # torch.eye has no CPU bfloat16 kernel before PyTorch 2.3, while direct tensor construction does (#4051).
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=half_dtype)
        self.assert_close(matrix, expected)
        assert kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion[None]).dtype == half_dtype

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_x_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_y_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_z_rotation(self, device, dtype, atol, rtol):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), device=device, dtype=dtype)
        matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)
        self.assert_close(matrix, expected, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=torch.float64)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_to_rotation_matrix), (quaternion,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_to_rotation_matrix
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)

    def test_convention_quaternion_component_order_is_w_x_y_z(self, device, dtype):
        # Convention pin -- THE trap of this module: quaternions are (w, x, y, z), real part FIRST.
        # (1, 0, 0, 0) is the identity. The (x, y, z, w) misreading of the same four numbers,
        # (0, 0, 0, 1), does not raise and does not return anything obviously wrong -- it returns
        # diag(-1, -1, 1), a perfectly valid 180-degree rotation about z. That is why the
        # counter-literal is pinned alongside the identity: an order swap is silent, and only the
        # second assertion catches it.
        # Snippet used to generate expected (stdlib only, R = I + 2*w*K + 2*K@K with K = skew(v)):
        #   q = (1, 0, 0, 0) -> v = 0, K = 0 -> R = I
        #   q = (0, 0, 0, 1) -> w = 0, v = (0, 0, 1)
        #     K @ K = diag(-1, -1, 0) -> R = I + 2 * diag(-1, -1, 0) = diag(-1, -1, 1)
        real_part_first = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        )
        self.assert_close(real_part_first, torch.eye(3, device=device, dtype=dtype))

        read_as_xyzw = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
        )
        self.assert_close(read_as_xyzw, torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device, dtype=dtype)))

    def test_convention_double_cover_q_and_minus_q_give_identical_matrices(self, device, dtype):
        # Convention pin: the unit quaternions double-cover SO(3), and every term of the rotation
        # matrix is a product of two quaternion components, so negating the whole quaternion leaves
        # the matrix BIT-identical -- not merely close. torch.equal, not assert_close: the measured
        # max difference is exactly 0.0, and the identity held in 500/500 random float64 draws
        # (seeded torch.Generator(6)) as well as at float32/float16/bfloat16 for the pinned input.
        # The input is non-unit and non-axis-aligned so the pin cannot pass by symmetry.
        quaternion = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)

        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)

        assert torch.equal(kornia.geometry.conversions.quaternion_to_rotation_matrix(-quaternion), rot)

    def test_convention_non_unit_quaternion_is_normalized_internally(self, device, dtype):
        # Convention pin: quaternion_to_rotation_matrix calls normalize_quaternion on its input
        # first, so a non-unit quaternion is accepted silently and yields the same rotation as its
        # normalised form -- the docstring never says so. Pinned two ways: the returned matrix
        # equals the one built from the unit quaternion (stdlib literal below), and rescaling the
        # input leaves the output bit-identical. The scale factors are powers of two so that the
        # scaling is exact in binary floating point at every dtype (0.001 is not: at bfloat16
        # 0.001 * q rounds differently and the matrices then differ by 1.6e-2).
        # Snippet used to generate expected (stdlib only, q = (1, 2, 3, 4) normalised):
        #   import math
        #   u = [v / math.sqrt(30.0) for v in (1.0, 2.0, 3.0, 4.0)]
        #   nv = math.sqrt(u[1] ** 2 + u[2] ** 2 + u[3] ** 2); theta = 2 * math.atan2(nv, u[0])
        #   Rodrigues([u[i + 1] / nv for i in range(3)], theta) ->
        #     [[-0.666666666666667,   0.13333333333333341, 0.7333333333333334],
        #      [ 0.6666666666666667, -0.3333333333333337,  0.6666666666666669],
        #      [ 0.3333333333333335,  0.9333333333333335,  0.1333333333333333]]
        quaternion = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)

        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)

        expected = torch.tensor(
            [
                [-0.666666666666667, 0.13333333333333341, 0.7333333333333334],
                [0.6666666666666667, -0.3333333333333337, 0.6666666666666669],
                [0.3333333333333335, 0.9333333333333335, 0.1333333333333333],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(rot, expected)

        assert torch.equal(kornia.geometry.conversions.quaternion_to_rotation_matrix(2.0 * quaternion), rot)
        assert torch.equal(kornia.geometry.conversions.quaternion_to_rotation_matrix(0.0009765625 * quaternion), rot)

    def test_convention_normalize_quaternion_is_l2_over_the_last_axis(self, device, dtype):
        # Convention pin (normalize_quaternion has no test class of its own; this is its nearest
        # sibling -- quaternion_to_rotation_matrix calls it on every input): it is a plain L2
        # normalisation of the last axis and nothing more. It does NOT reorder, and it does NOT
        # canonicalise the sign, so the whole vector keeps its sign. That makes it the one symbol
        # in this file whose "(x, y, z, w) or (w, x, y, z)" docstring phrasing is actually true:
        # the same four numbers in the other order come back scaled by the same factor, which the
        # third assertion pins.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   [v / math.sqrt(30.0) for v in (1.0, 2.0, 3.0, 4.0)]
        #     -> [0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214]
        expected = torch.tensor(
            [0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214],
            device=device,
            dtype=dtype,
        )

        out = kornia.geometry.conversions.normalize_quaternion(
            torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)
        )
        self.assert_close(out, expected)

        out_negated = kornia.geometry.conversions.normalize_quaternion(
            torch.tensor([-1.0, -2.0, -3.0, -4.0], device=device, dtype=dtype)
        )
        self.assert_close(out_negated, -expected)

        out_reversed = kornia.geometry.conversions.normalize_quaternion(
            torch.tensor([4.0, 3.0, 2.0, 1.0], device=device, dtype=dtype)
        )
        self.assert_close(out_reversed, expected.flip(0))

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="normalize_quaternion divides by max(‖q‖, eps), so below eps it returns a non-unit "
        "quaternion — kornia#3952",
        strict=True,
    )
    def test_convention_normalize_quaternion_is_unit_below_eps_3952(self, device, dtype):
        # Intended behavior: normalize_quaternion returns a unit quaternion, or refuses. It does
        # neither for a small input: it is F.normalize(q, p=2, dim=-1, eps=eps), which divides by
        # max(‖q‖, eps), so any quaternion with ‖q‖ < eps comes back scaled by ‖q‖/eps instead of
        # to unit length -- silently, with the docstring saying only "small value to avoid division
        # by zero". At ‖q‖ = 1e-13 with the default eps = 1e-12 the output has norm exactly 0.1.
        # This pin is placed in this class because normalize_quaternion has no test class of its
        # own and quaternion_to_rotation_matrix is the function that calls it on every input (the
        # same placement as test_convention_normalize_quaternion_is_l2_over_the_last_axis above);
        # the consequence for that caller is pinned by
        # test_wart_zero_quaternion_becomes_the_identity_matrix_3952 below. Marked xfail(strict=True)
        # so fixing #3952 makes this XPASS and forces the mark out. Companion wart:
        # test_wart_normalize_quaternion_scales_by_norm_over_eps_3952.
        if dtype == torch.float16:
            pytest.skip(
                "float16 cannot represent either 1e-13 or the default eps=1e-12 -- both round to 0, so the input is "
                "the zero quaternion and the output is NaN, which is a different claim (same underflow class as "
                "kornia#3966)"
            )

        quaternion = torch.tensor([1e-13, 0.0, 0.0, 0.0], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_quaternion(quaternion, eps=1e-12)

        assert abs(out.norm().item() - 1.0) < 1e-2, (
            "kornia#3952: normalize_quaternion returned a non-unit quaternion below eps"
        )

    def test_wart_normalize_quaternion_scales_by_norm_over_eps_3952(self, device, dtype):
        # Wart pin for kornia#3952, companion to the strict xfail above: assert the CURRENT
        # sub-eps outputs. Two cells here plus a third in the eps=0 control below, each
        # discriminating a different fix shape:
        #   (1) ‖q‖ = 1e-13 with eps = 1e-12 comes back as [0.1, 0, 0, 0] -- flips under every fix
        #       (identity fallback, smaller eps, raise);
        #   (2) the exactly-zero quaternion comes back as zeros -- does NOT flip under a
        #       "leave the input alone when ‖q‖ < eps" fix, which is exactly the shape that would
        #       leave cell (1) fixed and this one still broken.
        # If either fails, #3952 was (partly) fixed -- flip/remove the strict xfail above. NOT a
        # contract that sub-eps quaternions must keep these values. eps is passed explicitly at
        # every call site so the literals do not silently track a later change of the default.
        # Snippet used to generate expected (torch only, executed on cpu at float64):
        #   normalize_quaternion(torch.tensor([1e-13, 0., 0., 0.], dtype=torch.float64), eps=1e-12)
        #     -> [0.1, 0.0, 0.0, 0.0]           (float32: [0.10000000149011612, 0, 0, 0];
        #                                        bfloat16: [0.099609375, 0, 0, 0])
        #   normalize_quaternion(torch.zeros(4, dtype=torch.float64), eps=1e-12) -> [0, 0, 0, 0]
        if dtype == torch.float16:
            pytest.skip(
                "float16 cannot represent either 1e-13 or the default eps=1e-12 -- both round to 0, so both "
                "cells collapse to NaN (same underflow class as kornia#3966)"
            )
        _skip_if_mps_clamp_caching(device)

        normalize_quaternion = kornia.geometry.conversions.normalize_quaternion

        sub_eps = normalize_quaternion(torch.tensor([1e-13, 0.0, 0.0, 0.0], device=device, dtype=dtype), eps=1e-12)
        zero = normalize_quaternion(torch.zeros(4, device=device, dtype=dtype), eps=1e-12)

        assert_close(
            sub_eps,
            torch.tensor([0.1, 0.0, 0.0, 0.0], device=device, dtype=dtype),
            msg=_issue_msg("kornia#3952: a sub-eps quaternion is no longer scaled by ||q|| / eps"),
        )
        assert_close(
            zero,
            torch.zeros(4, device=device, dtype=dtype),
            msg=_issue_msg("kornia#3952: the zero quaternion no longer passes through unchanged"),
        )

    def test_wart_normalize_quaternion_without_eps_divides_by_zero_3952(self, device, dtype):
        # Wart pin for kornia#3952, cell (3) and the control for the two cells above: with eps=0
        # the zero quaternion produces NaN, which pins the *mechanism* (normalize_quaternion is a
        # plain division by max(‖q‖, eps), nothing more) rather than the symptom. It flips only if
        # that division is replaced by a clamp- or branch-shaped rewrite, which is how an eps-only
        # fix is told apart from a restructuring one. If it fails, #3952 was (partly) fixed --
        # flip/remove the strict xfail above. NOT a contract that eps=0 must keep producing NaN.
        # It lives in its own test rather than as a third cell of the wart above because on MPS the
        # two cannot share a process: this torch build caches clamp's scalar min per shape/dtype on
        # MPS, so an earlier normalize_quaternion(..., eps=1e-12) on a (4,) float32 tensor makes
        # the later eps=0.0 call reuse 1e-12 and return zeros instead of NaN (executed: cpu gives
        # [nan, nan, nan, nan], mps gives [0, 0, 0, 0] for the same three calls in sequence, while
        # the eps=0 call *alone* gives NaN on both). That is the torch defect already probed by
        # _skip_if_mps_clamp_caching further down this file, which is what guards this pin.
        # Snippet used to generate expected (torch only, executed on cpu at float64/float32/bfloat16):
        #   normalize_quaternion(torch.zeros(4, dtype=torch.float64), eps=0.0) -> [nan] * 4
        if dtype == torch.float16:
            pytest.skip(
                "float16 cannot represent the default eps=1e-12 either, so the zero quaternion is already NaN with "
                "the default and there is no eps=0 contrast to draw (same underflow class as kornia#3966)"
            )
        _skip_if_mps_clamp_caching(device)

        out = kornia.geometry.conversions.normalize_quaternion(torch.zeros(4, device=device, dtype=dtype), eps=0.0)

        assert torch.isnan(out).all(), "kornia#3952: the eps=0 division by the zero norm is no longer NaN"

    def test_wart_zero_quaternion_becomes_the_identity_matrix_3952(self, device, dtype):
        # Wart pin for the downstream consequence of kornia#3952: because normalize_quaternion
        # returns the zero vector unchanged for a zero input (cell 2 of the wart above),
        # quaternion_to_rotation_matrix(zeros(4)) evaluates 1 - 0 on the diagonal and returns the
        # IDENTITY -- the zero quaternion, which is not a rotation at all, silently becomes "no
        # rotation". Pinned separately from the normalize_quaternion cells because it flips under
        # two independent fixes: fixing #3952 in normalize_quaternion, or giving
        # quaternion_to_rotation_matrix its own guard while normalize_quaternion stays as it is.
        # If it fails, one of those happened -- flip/remove the #3952 strict xfail above and check
        # which. NOT a contract that the zero quaternion must map to the identity.
        # Snippet used to generate expected (torch only, executed on cpu at float64/float32/bfloat16):
        #   quaternion_to_rotation_matrix(torch.zeros(4, dtype=torch.float64))
        #     -> [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        if dtype == torch.float16:
            pytest.skip(
                "at float16 the default eps=1e-12 inside normalize_quaternion underflows to 0, so the zero "
                "quaternion yields an all-NaN matrix rather than the identity (same underflow class as kornia#3966)"
            )

        out = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.zeros(4, device=device, dtype=dtype))

        assert_close(
            out,
            torch.eye(3, device=device, dtype=dtype),
            msg=_issue_msg("kornia#3952: the zero quaternion no longer maps to the identity matrix"),
        )


class TestQuaternionLogToExp(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        quaternion_log = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log)
        assert quaternion_exp.shape == (batch_size, 4)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), torch.sin(one), 0.0, 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, torch.sin(one), 0.0), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        one = torch.tensor(1.0, device=device, dtype=dtype)
        quaternion_log = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((torch.cos(one), 0.0, 0.0, torch.sin(one)), device=device, dtype=dtype)
        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, expected, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_log = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)

        quaternion_exp = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        quaternion_log_hat = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, quaternion_log_hat, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_log_to_exp, eps=eps), (quaternion,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_log_to_exp
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)

    def test_convention_exp_of_v_equals_axis_angle_to_quaternion_of_twice_v(self, device, dtype):
        # Convention pin: the log quaternion is (theta / 2) * axis, NOT the axis-angle vector, so
        # the exponential map is exactly axis_angle_to_quaternion applied to 2 * v. A caller that
        # feeds an axis-angle vector straight into quaternion_log_to_exp gets a rotation of half
        # the intended angle, silently. The size contract of the pair is pinned alongside:
        # (*, 3) -> (*, 4) here, (*, 4) -> (*, 3) in quaternion_exp_to_log.
        # Snippet used to generate expected (stdlib only, v = (0.15, 0.2, 0.25), theta = 2 * |v|):
        #   import math
        #   th = math.sqrt(0.15 ** 2 + 0.2 ** 2 + 0.25 ** 2) * 2   # 0.7071067811865476
        #   ax = [x / (th / 2) for x in (0.15, 0.2, 0.25)]
        #   [math.cos(th / 2)] + [math.sin(th / 2) * a for a in ax]
        #     -> [0.9381483350397287, 0.14689447322208307, 0.19585929762944412, 0.24482412203680515]
        # assert_close and not torch.equal: the two routes agree bit-for-bit at float64, float32
        # and float16 for this input, but that is an accident of the input -- over 500 random
        # float64 vectors (seeded torch.Generator(4)) only 142/500 are bit-identical (worst
        # difference 4.440892098500626e-16), and at bfloat16 the pinned input already differs by
        # 9.765625e-04 because ||2v|| / 2 and ||v|| round apart.
        log_quaternion = torch.tensor([0.15, 0.2, 0.25], device=device, dtype=dtype)

        out = kornia.geometry.conversions.quaternion_log_to_exp(log_quaternion)

        self.assert_close(
            out,
            torch.tensor(
                [0.9381483350397287, 0.14689447322208307, 0.19585929762944412, 0.24482412203680515],
                device=device,
                dtype=dtype,
            ),
        )
        self.assert_close(out, kornia.geometry.conversions.axis_angle_to_quaternion(2.0 * log_quaternion))

        assert kornia.geometry.conversions.quaternion_log_to_exp(
            torch.zeros(2, 5, 3, device=device, dtype=dtype)
        ).shape == (2, 5, 4)

    def test_convention_exp_real_part_is_cosine_of_the_norm(self, device, dtype):
        # Convention pin: the exponential map returns w = cos(||v||) and (x, y, z) = sin(||v||) * v
        # / ||v||, so it is NOT restricted to the w >= 0 half of the double cover: any ||v|| > pi/2
        # (i.e. any rotation past 180 degrees, since theta = 2 * ||v||) lands in the w < 0 half.
        # Pinned at ||v|| = 2 rad, where w is clearly negative; the output is still a unit
        # quaternion, which the norm assertion states.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.cos(2.0), math.sin(2.0) -> (-0.4161468365471424, 0.9092974268256817)
        out = kornia.geometry.conversions.quaternion_log_to_exp(
            torch.tensor([0.0, 0.0, 2.0], device=device, dtype=dtype)
        )

        self.assert_close(
            out,
            torch.tensor([-0.4161468365471424, 0.0, 0.0, 0.9092974268256817], device=device, dtype=dtype),
        )
        self.assert_close(out.norm(), torch.tensor(1.0, device=device, dtype=dtype))

    def test_convention_exp_of_log_is_the_identity_except_at_minus_one(self, device, dtype):
        # Convention pin (domain fact of the map, not a defect): quaternion_log_to_exp composed
        # with quaternion_exp_to_log is the identity, with exactly one exception -- the pure-real
        # quaternion (-1, 0, 0, 0), whose log is genuinely the origin in this parametrisation, so
        # the round-trip returns the OTHER half of the double cover, (1, 0, 0, 0). The sign of a
        # non-zero vector part is preserved, which is what the third case pins: (-1, 0, 0, -1)/
        # sqrt(2) comes back as itself and is not flipped to its positive-w twin.
        # Snippet used to generate expected (stdlib only):
        #   exp(log(q)) = q for every unit q except q = (-1, 0, 0, 0)
        #   1 / math.sqrt(2.0) -> 0.7071067811865476
        # float16 is skipped: quaternion_exp_to_log((-1, 0, 0, 0)) returns NaN there, so the
        # exception case cannot be evaluated at all (float64/float32/bfloat16 all return the
        # origin as documented above).
        if dtype == torch.float16:
            pytest.skip("quaternion_exp_to_log((-1, 0, 0, 0)) is NaN at float16, so exp(log(.)) is undefined")

        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        exp_to_log = kornia.geometry.conversions.quaternion_exp_to_log
        log_to_exp = kornia.geometry.conversions.quaternion_log_to_exp

        self.assert_close(log_to_exp(exp_to_log(identity)), identity)
        self.assert_close(log_to_exp(exp_to_log(-identity)), identity)

        half_turn = torch.tensor([-0.7071067811865476, 0.0, 0.0, -0.7071067811865476], device=device, dtype=dtype)
        self.assert_close(log_to_exp(exp_to_log(half_turn)), half_turn)

    def test_convention_log_to_exp_of_the_origin_is_the_identity_in_float16_3966(self, device):
        # Intended behavior: the exponential map of the zero vector is the identity quaternion, at
        # every dtype -- float64, float32 and bfloat16 all return [1, 0, 0, 0]. float16 returns
        # [1, nan, nan, nan]: the default eps = 1e-8 is below float16's smallest subnormal
        # (5.960464477539063e-08), so torch.tensor(1e-8, dtype=float16) is exactly 0.0, the
        # .clamp(min=eps) on the norm is a no-op, and the vector part is sin(0) * 0 / 0. The real
        # part survives because it is cos(0). This is the same eps-underflow class as the one
        # kornia#3966 was filed for on quaternion_exp_to_log (pinned in TestQuaternionExpToLog);
        # bfloat16 escapes both because its exponent range is float32's. float16 is hardcoded and
        # the dtype fixture dropped so this pin runs in every configuration, with a visible skip
        # where the device lacks the dtype. Marked xfail(strict=True) so fixing #3966 makes this
        # XPASS and forces the mark out. Companion wart:
        # test_wart_float16_eps_underflow_makes_log_to_exp_nan_3966.
        _skip_if_dtype_unavailable(device, torch.float16)

        out = kornia.geometry.conversions.quaternion_log_to_exp(torch.zeros(3, device=device, dtype=torch.float16))

        assert_close(
            out,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float16),
            msg=_issue_msg("kornia#3966: quaternion_log_to_exp of the float16 origin is not the identity"),
        )


class TestQuaternionExpToLog(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 3, 8))
    def test_smoke_batch(self, batch_size, device, dtype):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        assert quaternion_log.shape == (batch_size, 3)

    def test_unit_quaternion(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_x(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((kornia.pi / 2.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_y(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, kornia.pi / 2.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_pi_quaternion_z(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device, dtype=dtype)
        expected = torch.tensor((0.0, 0.0, kornia.pi / 2.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        self.assert_close(quaternion_log, expected, atol=atol, rtol=rtol)

    def test_back_and_forth(self, device, dtype, atol, rtol):
        eps = torch.finfo(dtype).eps
        quaternion_exp = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        quaternion_log = kornia.geometry.conversions.quaternion_exp_to_log(quaternion_exp, eps=eps)
        quaternion_exp_hat = kornia.geometry.conversions.quaternion_log_to_exp(quaternion_log, eps=eps)
        self.assert_close(quaternion_exp, quaternion_exp_hat, atol=atol, rtol=rtol)

    def test_gradcheck(self, device):
        dtype = torch.float64
        eps = torch.finfo(dtype).eps
        quaternion = torch.tensor((0.0, 1.0, 0.0, 0.0), device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(partial(kornia.geometry.conversions.quaternion_exp_to_log, eps=eps), (quaternion,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        quaternion = torch.tensor((0.0, 0.0, 1.0, 0.0), device=device, dtype=dtype)
        op = kornia.geometry.conversions.quaternion_exp_to_log
        op_optimized = torch_optimizer(op)

        actual = op_optimized(quaternion)
        expected = op(quaternion)

        self.assert_close(actual, expected)

    def test_convention_log_is_half_the_axis_angle_on_the_w_positive_half(self, device, dtype):
        # Convention pin: the log quaternion is (theta / 2) * axis, i.e. exactly half the
        # axis-angle vector -- so quaternion_exp_to_log(q) == quaternion_to_axis_angle(q) / 2, but
        # ONLY on the w >= 0 half. The two functions treat the double cover differently:
        # quaternion_to_axis_angle collapses q and -q onto the same |theta| <= pi vector, while
        # quaternion_exp_to_log takes acos(w) at face value and returns (pi - theta/2) along the
        # negated axis for w < 0. The second half of this pin states that divergence explicitly,
        # because "log is half the axis-angle" is false without the restriction: over 500 random
        # float64 unit quaternions (seeded torch.Generator(3)) the two agree to 4.44e-16 on the
        # 243 with w >= 0 and disagree by up to 3.1367975802888637 on the 257 with w < 0.
        # The size contract (*, 4) -> (*, 3) is pinned alongside.
        # Snippet used to generate expected (stdlib only, axis_angle = (0.3, 0.4, 0.5)):
        #   import math
        #   v = [0.15, 0.2, 0.25]                       # = axis_angle / 2, the expected log
        #   nv = math.sqrt(sum(x * x for x in v))       # 0.3535533905932738
        #   q = [math.cos(nv)] + [math.sin(nv) * x / nv for x in v]
        #     -> [0.9381483350397287, 0.14689447322208307, 0.19585929762944412, 0.24482412203680515]
        #   for -q the log is (nv - pi) * axis:
        #   [-(math.pi - nv) * (x / nv) for x in v]
        #     -> [-1.1828648814475096, -1.5771531752633463, -1.9714414690791828]
        quaternion = torch.tensor(
            [0.9381483350397287, 0.14689447322208307, 0.19585929762944412, 0.24482412203680515],
            device=device,
            dtype=dtype,
        )

        out = kornia.geometry.conversions.quaternion_exp_to_log(quaternion)

        self.assert_close(out, torch.tensor([0.15, 0.2, 0.25], device=device, dtype=dtype))
        self.assert_close(out, kornia.geometry.conversions.quaternion_to_axis_angle(quaternion) / 2.0)

        out_negated = kornia.geometry.conversions.quaternion_exp_to_log(-quaternion)
        self.assert_close(
            out_negated,
            torch.tensor([-1.1828648814475096, -1.5771531752633463, -1.9714414690791828], device=device, dtype=dtype),
        )

        assert kornia.geometry.conversions.quaternion_exp_to_log(
            torch.zeros(2, 5, 4, device=device, dtype=dtype)
        ).shape == (2, 5, 3)

    def test_convention_log_of_exp_is_exact_below_pi_and_wraps_above(self, device, dtype):
        # Convention pin (domain fact of the map, not a defect): quaternion_exp_to_log composed
        # with quaternion_log_to_exp reproduces its input only for 0 < ||v|| < pi. Above pi the
        # rotation has passed a full turn (theta = 2 * ||v||) and the result wraps into
        # ||v|| - 2*pi, i.e. it comes back with the OPPOSITE sign, which the second case pins:
        # a caller doing exp/log arithmetic on large vectors must reduce the norm itself.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   the round-trip is the identity for ||v|| < pi -> [0.0, 0.0, 1.0]
        #   math.pi + 0.5 - 2 * math.pi -> -2.641592653589793
        exp_to_log = kornia.geometry.conversions.quaternion_exp_to_log
        log_to_exp = kornia.geometry.conversions.quaternion_log_to_exp

        below_pi = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        self.assert_close(exp_to_log(log_to_exp(below_pi)), below_pi)

        above_pi = torch.tensor([0.0, 0.0, 3.641592653589793], device=device, dtype=dtype)
        self.assert_close(
            exp_to_log(log_to_exp(above_pi)),
            torch.tensor([0.0, 0.0, -2.641592653589793], device=device, dtype=dtype),
        )

    def test_convention_log_of_exp_collapses_to_zero_at_pi_in_float64(self, device):
        # Convention pin (domain fact of the map): at exactly ||v|| = pi the exponential map lands
        # on (-1, 0, 0, 0), whose log genuinely IS the origin in this parametrisation, so the
        # round-trip collapses to ~0 instead of returning pi -- the one interior point where the
        # log/exp pair is not invertible. float64 is hardcoded and the dtype fixture dropped
        # because the collapse needs cos(||v||) to round to exactly -1 and the vector part to fall
        # below the eps clamp, which only happens at float64: at float32 the same input returns
        # -3.1415927410125732 (no collapse), at float16/bfloat16 3.140625. MPS is skipped visibly
        # because it has no float64 at all.
        # Snippet used to generate expected (stdlib only):
        #   math.cos(math.pi), math.sin(math.pi) -> (-1.0, 1.2246467991473532e-16)
        # Measured round-trip value at float64 (torch 2.9.1, cpu): 3.847341387443579e-08, i.e. an
        # error of 3.14 against the input, so atol 1e-7 pins the collapse without pinning the
        # residue itself.
        _skip_if_dtype_unavailable(device, torch.float64)

        at_pi = torch.tensor([0.0, 0.0, 3.141592653589793], device=device, dtype=torch.float64)

        out = kornia.geometry.conversions.quaternion_exp_to_log(
            kornia.geometry.conversions.quaternion_log_to_exp(at_pi)
        )

        self.assert_close(out, torch.zeros(3, device=device, dtype=torch.float64), atol=1e-7, rtol=0.0)

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="quaternion_exp_to_log does not normalise its input, so a non-unit quaternion gives "
        "a silently wrong log — kornia#3953",
        strict=True,
    )
    def test_convention_exp_to_log_normalizes_its_input_3953(self, device, dtype):
        # Intended behavior: the log of a quaternion depends only on the rotation it represents, so
        # a rescaled quaternion gives the same answer -- which is what its scale-safe siblings do
        # (quaternion_to_rotation_matrix normalises internally, quaternion_to_axis_angle is
        # homogeneous by construction; both are pinned above). quaternion_exp_to_log does neither:
        # it feeds the raw scalar part straight into acos, so q = (0.5, 0.5, 0, 0) -- the 90-degree
        # rotation about x, scaled by 1/sqrt(2) -- returns 1.0471975511965976 instead of
        # acos(1/sqrt(2)) = 0.7853981633974484, i.e. 33% too large, with no error and no warning.
        # Marked xfail(strict=True) so fixing #3953 makes this XPASS and forces the mark out.
        # Companion wart: test_wart_exp_to_log_ignores_the_quaternion_norm_3953; the
        # euler_from_quaternion half of the same issue is pinned in TestEulerFromQuaternion.
        quaternion = torch.tensor([0.5, 0.5, 0.0, 0.0], device=device, dtype=dtype)

        out = kornia.geometry.conversions.quaternion_exp_to_log(quaternion, eps=1e-8)

        assert_close(
            out,
            torch.tensor([0.7853981633974484, 0.0, 0.0], device=device, dtype=dtype),
            msg=_issue_msg("kornia#3953: quaternion_exp_to_log did not normalise its input"),
        )

    def test_wart_exp_to_log_ignores_the_quaternion_norm_3953(self, device, dtype):
        # Wart pin for kornia#3953, companion to the strict xfail above: assert the CURRENT
        # non-unit-input outputs. Two cells that discriminate the two plausible fix shapes:
        #   (1) q = (0.5, 0.5, 0, 0) returns 1.0471975511965976 -- flips under a "normalise the
        #       input" fix and under a "raise on non-unit input" fix alike;
        #   (2) q = (2, 0, 0, 0) returns the origin because the scalar part is clamped into
        #       [-1, 1] before the acos -- this does NOT flip under a normalising fix (the
        #       normalised input is the identity, whose log is the origin), so it is the cell that
        #       tells a normalising fix apart from a validating one.
        # If either fails, #3953 was (partly) fixed -- flip/remove the strict xfail above. NOT a
        # contract that non-unit input must keep these values. eps is passed explicitly so the
        # literals do not silently track a later change of the default.
        # Snippet used to generate expected (torch + stdlib, executed on cpu float64):
        #   quaternion_exp_to_log(torch.tensor([0.5, 0.5, 0., 0.], dtype=torch.float64), eps=1e-8)
        #     -> [1.0471975511965976, 0.0, 0.0]        (float32: 1.0471975803375244;
        #                                               float16/bfloat16: 1.046875)
        #   math.acos(0.5) -> 1.0471975511965979, and math.acos(1 / math.sqrt(2)) -> 0.7853981633974484
        #   quaternion_exp_to_log(torch.tensor([2., 0., 0., 0.], dtype=torch.float64), eps=1e-8)
        #     -> [0.0, 0.0, 0.0]
        if dtype == torch.float16:
            pytest.skip(
                "at float16 the second cell is NaN rather than the origin, because the default eps underflows and "
                "the zero vector part is divided by itself -- that is kornia#3966, pinned separately below"
            )

        exp_to_log = kornia.geometry.conversions.quaternion_exp_to_log

        scaled_down = exp_to_log(torch.tensor([0.5, 0.5, 0.0, 0.0], device=device, dtype=dtype), eps=1e-8)
        scaled_up = exp_to_log(torch.tensor([2.0, 0.0, 0.0, 0.0], device=device, dtype=dtype), eps=1e-8)

        assert_close(
            scaled_down,
            torch.tensor([1.0471975511965976, 0.0, 0.0], device=device, dtype=dtype),
            msg=_issue_msg("kornia#3953: quaternion_exp_to_log no longer takes the raw scalar part at face value"),
        )
        assert_close(
            scaled_up,
            torch.zeros(3, device=device, dtype=dtype),
            msg=_issue_msg("kornia#3953: the scalar-part clamp no longer sends an over-scaled quaternion to zero"),
        )

    def test_convention_exp_to_log_of_the_identity_is_the_origin_in_float16_3966(self, device):
        # Intended behavior: the log of the identity quaternion is the origin, at every dtype --
        # float32, float64 and bfloat16 all return [0, 0, 0]. float16 returns [nan, nan, nan]:
        # the default eps = 1e-8 is below float16's smallest subnormal (5.960464477539063e-08), so
        # torch.tensor(1e-8, dtype=float16) is exactly 0.0, the .clamp(min=eps) on the vector-part
        # norm is a no-op, and the log is acos(1) * 0 / 0. bfloat16 escapes only because its
        # exponent range is float32's (1e-8 survives as 1.0011717677116394e-08). float16 is
        # hardcoded and the dtype fixture dropped so this pin runs in every configuration, with a
        # visible skip where the device lacks the dtype -- otherwise a RuntimeError would satisfy
        # the raises=AssertionError mark instead of the assertion. Marked xfail(strict=True) so
        # fixing #3966 makes this XPASS and forces the mark out. Companion wart:
        # test_wart_float16_eps_underflow_makes_exp_to_log_nan_3966. Note that 3a's merged
        # test_wart_float16_underflowed_default_eps_flips_branches pins the same failure class
        # (a float default eps that float16 cannot represent) at a different site.
        _skip_if_dtype_unavailable(device, torch.float16)

        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float16)

        out = kornia.geometry.conversions.quaternion_exp_to_log(identity)

        assert_close(
            out,
            torch.zeros(3, device=device, dtype=torch.float16),
            msg=_issue_msg("kornia#3966: quaternion_exp_to_log of the float16 identity is not the origin"),
        )


class TestAngleAxisToRotationMatrix(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_axis_angle_gradcheck(self, batch_size, device, atol, rtol):
        dtype = torch.float64
        # generate input data
        axis_angle = torch.rand(batch_size, 3, device=device, dtype=dtype)
        eye_batch = eye_like(3, axis_angle)

        # apply transform
        rotation_matrix = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)

        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        self.assert_close(rotation_matrix_eye, eye_batch, atol=atol, rtol=rtol)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.axis_angle_to_rotation_matrix, (axis_angle,))

    def test_axis_angle_to_rotation_matrix(self, device, dtype, atol, rtol):
        rmat_1 = torch.tensor(
            (
                (-0.30382753, -0.95095137, -0.05814062),
                (-0.71581715, 0.26812278, -0.64476041),
                (0.62872461, -0.15427791, -0.76217038),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_1 = torch.tensor((1.50485376, -2.10737739, 0.7214174), device=device, dtype=dtype)

        rmat_2 = torch.tensor(
            (
                (0.6027768, -0.79275544, -0.09054801),
                (-0.67915707, -0.56931658, 0.46327563),
                (-0.41881476, -0.21775548, -0.88157628),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_2 = torch.tensor((-2.44916812, 1.18053411, 0.4085298), device=device, dtype=dtype)
        rmat = torch.stack((rmat_2, rmat_1), dim=0)
        rvec = torch.stack((rvec_2, rvec_1), dim=0)

        self.assert_close(kornia.geometry.conversions.axis_angle_to_rotation_matrix(rvec), rmat, atol=atol, rtol=rtol)

    def test_convention_positive_angle_about_z_maps_x_to_y(self, device, dtype):
        # Convention pin (covers quaternion_to_rotation_matrix too -- both routes to a rotation
        # matrix in this module must agree): rotations follow the right-hand rule, so a positive
        # angle about +z takes x_hat to y_hat and the matrix is
        # [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], NOT its transpose. Pinned at theta = 0.6 rad
        # rather than a quarter turn so that a transposed or sign-flipped implementation cannot
        # slip through on symmetry, and the mapped basis vector is asserted as well as the matrix
        # so the claim is stated the way a reader will use it.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.cos(0.6), math.sin(0.6) -> (0.8253356149096783, 0.5646424733950354)
        #   the same rotation as a quaternion, (cos(0.3), 0, 0, sin(0.3))
        #     -> (0.955336489125606, 0.0, 0.0, 0.29552020666133955)
        expected = torch.tensor(
            [
                [0.8253356149096783, -0.5646424733950354, 0.0],
                [0.5646424733950354, 0.8253356149096783, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        expected_x_maps_to = torch.tensor([0.8253356149096783, 0.5646424733950354, 0.0], device=device, dtype=dtype)
        x_hat = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)

        rot_from_axis_angle = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            torch.tensor([[0.0, 0.0, 0.6]], device=device, dtype=dtype)
        )[0]
        self.assert_close(rot_from_axis_angle, expected)
        self.assert_close(rot_from_axis_angle @ x_hat, expected_x_maps_to)

        rot_from_quaternion = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            torch.tensor([0.955336489125606, 0.0, 0.0, 0.29552020666133955], device=device, dtype=dtype)
        )
        self.assert_close(rot_from_quaternion, expected)
        self.assert_close(rot_from_quaternion @ x_hat, expected_x_maps_to)

    def test_convention_axis_angle_is_in_radians(self, device, dtype):
        # Convention pin: the axis-angle vector's magnitude is an angle in RADIANS. This is the
        # trap that separates this family from angle_to_rotation_matrix in the same module, which
        # reads DEGREES (see TestRadDegConversions.test_convention_angle_to_rotation_matrix_takes_
        # degrees) -- the two live a few hundred lines apart and neither says so in its signature.
        # pi/2 gives the quarter turn; feeding 90 in the belief that it is degrees gives cos/sin of
        # 90 radians instead, a rotation of roughly 152 degrees that is nowhere near a quarter turn.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.cos(math.pi / 2), math.sin(math.pi / 2) -> (6.123233995736766e-17, 1.0)
        #   math.cos(90.0), math.sin(90.0) -> (-0.4480736161291702, 0.8939966636005579)
        quarter_turn = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            torch.tensor([[0.0, 0.0, torch.pi / 2]], device=device, dtype=dtype)
        )[0]
        self.assert_close(
            quarter_turn,
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                device=device,
                dtype=dtype,
            ),
        )

        read_as_degrees = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            torch.tensor([[0.0, 0.0, 90.0]], device=device, dtype=dtype)
        )[0]
        self.assert_close(
            read_as_degrees,
            torch.tensor(
                [
                    [-0.4480736161291702, -0.8939966636005579, 0.0],
                    [0.8939966636005579, -0.4480736161291702, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            ),
        )

    def test_convention_returns_an_orthogonal_matrix_3947(self, device):
        # Regression test for kornia#3947. The function used to normalise the axis by
        # (theta + eps) with eps = 1e-6 hardcoded inside _compute_rotation_matrix, so the axis
        # was shrunk by eps/theta and what came back was a slightly scaled rotation. Measured at
        # theta = pi/2 about +z in float64 (torch 2.9.1, cpu) before the fix:
        #   det(R)           = 0.9999974535249636       (should be 1.0)
        #   max|R @ R.T - I| = 2.5464750363912714e-06   (should be ~1e-16)
        # and the error did not shrink with dtype -- float32 gave 2.5033950805664062e-06.
        # kornia's own quaternion route is the independent reference for the intended value:
        # quaternion_to_rotation_matrix(axis_angle_to_quaternion(v)) has det exactly 1.0 and
        # max|R @ R.T - I| exactly 0.0 on this input -- so the defect was in this function,
        # not in the angle.
        # float64 is hardcoded and the dtype fixture dropped so the literals mean one thing;
        # the skip is visible so that on MPS, which has no float64, a raw TypeError cannot
        # satisfy the assertions this test documents.
        _skip_if_dtype_unavailable(device, torch.float64)

        axis_angle = torch.tensor([[0.0, 0.0, torch.pi / 2]], device=device, dtype=torch.float64)

        rot = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)[0]

        identity = torch.eye(3, device=device, dtype=torch.float64)
        assert (rot @ rot.T - identity).abs().max().item() < 1e-12, (
            "kornia#3947: axis_angle_to_rotation_matrix did not return an orthogonal matrix"
        )
        assert abs(torch.linalg.det(rot).item() - 1.0) < 1e-12, (
            "kornia#3947: axis_angle_to_rotation_matrix returned a matrix whose determinant is not 1"
        )

    def test_convention_both_branches_are_orthogonal_3947(self, device):
        # Regression test for kornia#3947 covering BOTH branches of the function, which switch at
        # theta**2 > 1e-6 and were each broken for its own reason before the fix:
        #   (1) theta = pi/2 takes the general branch and the eps in the axis normalisation gave
        #       det = 0.9999974535249636 and max|R @ R.T - I| = 2.5464750363912714e-06;
        #   (2) theta = 1e-3 takes the low-angle branch, which returned the first-order Taylor
        #       matrix [[1, -rz, ry], [rz, 1, -rx], [-ry, rx, 1]] with det = 1 + theta**2 = 1.000001.
        # The low-angle branch now returns the second-order Taylor expansion R = I + [v]x + [v]x^2/2,
        # whose determinant is 1 + theta**4 / 4 -- at theta = 1e-3 that is 1 + 2.5e-13, a rotation
        # to the working precision -- so both branches must now pass the same orthogonality checks.
        # The general branch must also agree with the quaternion route on a generic (non-axis
        # aligned) rotation, which is where the eps defect showed up as an axis-dependent error.
        # float64 is hardcoded and the dtype fixture dropped because both cells are float64 facts:
        # at float32 the same theta = 1e-3 input has theta**2 = 1.0000001111620804e-06 and
        # falls into the *other* branch, so cell (2) would be exercising a different code path.
        # Snippet used to generate expected (torch only, executed on cpu float64):
        #   v = torch.tensor([[0., 0., math.pi / 2]], dtype=torch.float64)
        #   R = axis_angle_to_rotation_matrix(v)[0]
        #   torch.linalg.det(R).item()                                  -> 1.0
        #   (R @ R.T - torch.eye(3, dtype=torch.float64)).abs().max()   -> 0.0
        #   t = torch.tensor([[0., 0., 1e-3]], dtype=torch.float64)   # theta**2 == 1e-06 exactly
        #   axis_angle_to_rotation_matrix(t)[0].tolist()
        #     -> [[0.9999995, -0.001, 0.0], [0.001, 0.9999995, 0.0], [0.0, 0.0, 1.0]]
        #   torch.linalg.det(that).item()                               -> 1.00000000000025
        #   g = torch.tensor([[1., 2., 3.]], dtype=torch.float64) * 0.6 / math.sqrt(14.0)  # generic axis
        #   Rg = axis_angle_to_rotation_matrix(g)[0]
        #   torch.linalg.det(Rg).item()                                 -> 1.0
        #   Rq = quaternion_to_rotation_matrix(axis_angle_to_quaternion(g))[0]
        #   (Rg - Rq).abs().max().item()                                -> 4.440892098500626e-16
        _skip_if_dtype_unavailable(device, torch.float64)

        identity = torch.eye(3, device=device, dtype=torch.float64)

        general = axis_angle_to_rotation_matrix(
            torch.tensor([[0.0, 0.0, torch.pi / 2]], device=device, dtype=torch.float64)
        )[0]
        taylor = axis_angle_to_rotation_matrix(torch.tensor([[0.0, 0.0, 1e-3]], device=device, dtype=torch.float64))[0]
        generic = axis_angle_to_rotation_matrix(
            torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=torch.float64) * 0.6 / 14.0**0.5
        )[0]

        for rot in (general, taylor, generic):
            assert (rot @ rot.T - identity).abs().max().item() < 1e-12, (
                "kornia#3947: axis_angle_to_rotation_matrix did not return an orthogonal matrix"
            )
            assert abs(torch.linalg.det(rot).item() - 1.0) < 1e-12, (
                "kornia#3947: axis_angle_to_rotation_matrix returned a matrix whose determinant is not 1"
            )

        # the general branch must agree with the independent quaternion route (machine precision)
        quat_route = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            kornia.geometry.conversions.axis_angle_to_quaternion(
                torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=torch.float64) * 0.6 / 14.0**0.5
            )
        )[0]
        assert (generic - quat_route).abs().max().item() < 1e-12, (
            "kornia#3947: axis_angle_to_rotation_matrix disagrees with the quaternion route"
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="axis_angle_to_rotation_matrix accepts only rank-2 (N, 3) input despite its own "
        "guard message saying (*, 3) — kornia#3955",
        strict=True,
    )
    def test_convention_accepts_any_leading_batch_dimensions_3955(self, device):
        # Intended behavior: axis_angle_to_rotation_matrix accepts (*, 3) -- which is what its own
        # shape guard says in the message it raises ("Input size must be a (*, 3) tensor") and what
        # every sibling in this module does, including rotation_matrix_to_axis_angle (pinned by
        # TestRotationMatrixToAngleAxis.test_convention_accepts_any_leading_batch_dimensions).
        # It accepts only rank 2: the body does wxyz.unbind(dim=1) and .view(-1, 3, 3), so an
        # unbatched (3,) raises IndexError and any extra batch dimension raises ValueError from the
        # unbind. The asymmetry breaks composition -- aa2R(R2aa(R)) works for a (N, 3, 3) input and
        # for nothing else, which the last two assertions pin. Written through the shared
        # _runs_without_raising helper because the current behavior is a *raise*: a bare call would
        # let IndexError/ValueError escape, and the mark (raises=AssertionError) would then not
        # match, so the failure would be reported as an error rather than an XFAIL. Marked
        # xfail(strict=True) so fixing #3955 makes this XPASS and forces the mark out. Companion
        # wart: test_wart_only_rank_2_input_is_accepted_3955.
        # float32 is hardcoded and the dtype fixture dropped: all three rank errors come out of the
        # same shape-driven `wxyz.unbind(dim=1)` inside _compute_rotation_matrix, which no dtype can
        # change. NOT "before any arithmetic runs" -- theta2 = (aa * aa).sum(-1), the sqrt and the
        # axis division all execute first and succeed at every float dtype; it is that the
        # arithmetic ahead of the unbind is dtype-safe, so which ranks are accepted still cannot
        # depend on the dtype and the fixture only multiplied the cell count.
        axis_angle_to_rotation_matrix = kornia.geometry.conversions.axis_angle_to_rotation_matrix
        rotation_matrix_to_axis_angle = kornia.geometry.conversions.rotation_matrix_to_axis_angle

        unbatched = torch.tensor([0.0, 0.0, 0.6], device=device, dtype=torch.float32)
        rot = axis_angle_to_rotation_matrix(unbatched.reshape(1, 3))[0]

        assert _runs_without_raising(axis_angle_to_rotation_matrix, unbatched), (
            "kornia#3955: axis_angle_to_rotation_matrix rejects an unbatched (3,) input"
        )
        assert _runs_without_raising(axis_angle_to_rotation_matrix, unbatched.expand(2, 5, 3)), (
            "kornia#3955: axis_angle_to_rotation_matrix rejects a (2, 5, 3) input"
        )
        assert _runs_without_raising(axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle(rot)), (
            "kornia#3955: aa2R(R2aa(R)) fails for a (3, 3) rotation matrix"
        )
        assert _runs_without_raising(
            axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle(rot.expand(2, 5, 3, 3))
        ), "kornia#3955: aa2R(R2aa(R)) fails for a (2, 5, 3, 3) stack of rotation matrices"

    @pytest.mark.parametrize(
        ("shape", "error", "message"),
        [
            ((3,), IndexError, r"Dimension out of range"),
            ((2, 5, 3), ValueError, r"too many values to unpack"),
            ((1, 1, 3), ValueError, r"not enough values to unpack"),
        ],
        ids=["unbatched", "extra_batch_dim", "singleton_extra_batch_dim"],
    )
    def test_wart_only_rank_2_input_is_accepted_3955(self, device, shape, error, message):
        # Wart pin for kornia#3955, companion to the strict xfail above: assert the CURRENT failure
        # modes, matching on the message and not merely on the type, because the message is the
        # evidence that these are raw Python unpacking errors leaking out of the implementation
        # rather than kornia's own shape guard (whose message says "(*, 3)" and never fires here).
        # Matched on the distinguishing phrase only, not the full parenthesised detail: that detail
        # is PyTorch's and CPython's wording, so a reword upstream would flip these cells and be
        # misread as "#3955 was partly fixed". The phrase alone still separates the three failure
        # modes from each other and from kornia's own guard, which is all the evidence needs.
        # Three cells: all three raise at the SAME line -- wxyz.unbind(dim=1) in
        # _compute_rotation_matrix -- but with different error kinds, because
        # the rank differs: the unbatched (3,) case has no dim=1 to unbind and gets an IndexError,
        # while the two over-batched cases unbind successfully and fail on the 3-way assignment
        # with a ValueError. So a fix that only flattens the leading dimensions flips the last two
        # and leaves the first. The (1, 1, 3) and (2, 5, 3) cells do flip together under every fix
        # shape I could construct, but they are kept apart because they report *different* messages
        # today, and pinning only one of them would let the other change unnoticed.
        # If any cell fails, #3955 was (partly) fixed -- flip/remove the strict xfail above. NOT a
        # contract that these ranks must keep raising.
        # float32 is hardcoded and the dtype fixture dropped for the same reason as the xfail above:
        # every one of these errors comes from the same shape-driven unbind, and the arithmetic that
        # precedes it succeeds at every float dtype, so the dtype cannot change which ranks raise.
        # Snippet used to generate expected (torch only, executed on cpu float64):
        #   axis_angle_to_rotation_matrix(torch.zeros(3, dtype=torch.float64))
        #     -> IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
        #   axis_angle_to_rotation_matrix(torch.zeros(2, 5, 3, dtype=torch.float64))
        #     -> ValueError: too many values to unpack (expected 3)
        #   axis_angle_to_rotation_matrix(torch.zeros(1, 1, 3, dtype=torch.float64))
        #     -> ValueError: not enough values to unpack (expected 3, got 1)
        #   (the accepted ranks (1, 3) and (2, 3) return (1, 3, 3) and (2, 3, 3))
        with pytest.raises(error, match=message):
            kornia.geometry.conversions.axis_angle_to_rotation_matrix(
                torch.zeros(shape, device=device, dtype=torch.float32)
            )


class TestRotationMatrixToAngleAxis(BaseTester):
    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_rand_quaternion_gradcheck(self, batch_size, device, dtype, atol, rtol):
        # generate input data
        quaternion = torch.rand(batch_size, 4, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.normalize_quaternion(quaternion + 1e-6)
        rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion=quaternion)

        eye_batch = eye_like(3, rotation_matrix)
        rotation_matrix_eye = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
        # This didn't pass with atol=0.001, rtol=0.001 for float16 Cuda 11.2 GeForce 1080 Ti
        self.assert_close(rotation_matrix_eye, eye_batch, atol=atol * 10.0, rtol=rtol * 10.0)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device):
        dtype = torch.float64
        quaternion = torch.rand(batch_size, 4, device=device, dtype=dtype)
        quaternion = kornia.geometry.conversions.normalize_quaternion(quaternion + 1e-6)
        rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion=quaternion)
        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.rotation_matrix_to_axis_angle, (rotation_matrix,))

    def test_rotation_matrix_to_axis_angle(self, device, dtype, atol, rtol):
        rmat_1 = torch.tensor(
            (
                (-0.30382753, -0.95095137, -0.05814062),
                (-0.71581715, 0.26812278, -0.64476041),
                (0.62872461, -0.15427791, -0.76217038),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_1 = torch.tensor((1.50485376, -2.10737739, 0.7214174), device=device, dtype=dtype)

        rmat_2 = torch.tensor(
            (
                (0.6027768, -0.79275544, -0.09054801),
                (-0.67915707, -0.56931658, 0.46327563),
                (-0.41881476, -0.21775548, -0.88157628),
            ),
            device=device,
            dtype=dtype,
        )
        rvec_2 = torch.tensor((-2.44916812, 1.18053411, 0.4085298), device=device, dtype=dtype)
        rmat = torch.stack((rmat_2, rmat_1), dim=0)
        rvec = torch.stack((rvec_2, rvec_1), dim=0)

        self.assert_close(kornia.geometry.conversions.rotation_matrix_to_axis_angle(rmat), rvec, atol=atol, rtol=rtol)

    def test_convention_accepts_any_leading_batch_dimensions(self, device, dtype):
        # Convention pin (rotation_matrix_to_axis_angle has no test class under its own name; this
        # class is the one that exercises it): the shape contract is the full (*, 3, 3) -> (*, 3),
        # not the (N, 3, 3) -> (N, 3) its docstring states. An unbatched (3, 3) works -- that is
        # what its own doctest passes -- and so does any number of leading batch dimensions.
        # Expected is the true axis-angle vector computed with stdlib, not the function's output.
        # Snippet used to generate the matrix and expected (stdlib only):
        #   import math
        #   n = math.sqrt(14.0); axis = (1 / n, 2 / n, -3 / n); theta = math.radians(170.0)
        #   R = I + sin(theta) * K + (1 - cos(theta)) * K @ K    # Rodrigues, K = skew(axis)
        #   [theta * a for a in axis]
        #     -> [0.7929800678379483, 1.5859601356758966, -2.378940203513845]
        rot = torch.tensor(
            [
                [-0.8430357706541933, 0.4227722475733091, -0.3324970918358584],
                [0.14431568185875046, -0.4177198235801487, -0.8970413217671823],
                [-0.5181348023122309, -0.8042224665289962, 0.29114008820992554],
            ],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor(
            [0.7929800678379483, 1.5859601356758966, -2.378940203513845], device=device, dtype=dtype
        )

        unbatched = kornia.geometry.conversions.rotation_matrix_to_axis_angle(rot)
        assert unbatched.shape == (3,)
        self.assert_close(unbatched, expected)

        multi_batched = kornia.geometry.conversions.rotation_matrix_to_axis_angle(rot.expand(2, 5, 3, 3))
        assert multi_batched.shape == (2, 5, 3)
        self.assert_close(multi_batched[1, 4], expected)

    def test_convention_axis_angle_roundtrip_tolerance_is_1e_6_in_float64(self, device):
        # Convention pin: rotation_matrix_to_axis_angle composed with
        # axis_angle_to_rotation_matrix recovers the vector only to ~1e-6, and that floor does not
        # move with the dtype -- it is still ~1e-6 in float64, six orders worse than the machine
        # epsilon a reader would expect from a "round-trip" and eleven orders worse than the
        # quaternion leg (see TestAngleAxisToQuaternion.test_convention_axis_angle_quaternion_
        # roundtrip_is_exact_in_float64, which is exact at the same angles). Anyone comparing
        # rotations through this pair must budget 1e-6. float64 is hardcoded and the dtype fixture
        # dropped because the claim is precisely that float64 does NOT help; MPS is skipped visibly
        # because it has no float64 at all. The tolerance is the observed one and must not be
        # tightened.
        # Snippet used to generate the inputs (stdlib only):
        #   import math
        #   n = math.sqrt(14.0); axis = (1 / n, 2 / n, 3 / n)
        #   [[theta * a for a in axis] for theta in (1e-3, 0.7, 2.0, math.pi)]
        # Measured max |roundtrip - input| at those four thetas (torch 2.9.1, cpu float64):
        #   8.009844122083441e-07, 6.410323137862051e-07, 5.134006499929455e-07,
        #   5.387205765927661e-07 -- so atol 1e-6 clears the worst of them by 20%.
        _skip_if_dtype_unavailable(device, torch.float64)

        axis_angle = torch.tensor(
            [
                [0.0002672612419124244, 0.0005345224838248488, 0.0008017837257372733],
                [0.18708286933869706, 0.3741657386773941, 0.5612486080160912],
                [0.5345224838248488, 1.0690449676496976, 1.6035674514745464],
                [0.839625954181357, 1.679251908362714, 2.518877862544071],
            ],
            device=device,
            dtype=torch.float64,
        )

        roundtrip = kornia.geometry.conversions.rotation_matrix_to_axis_angle(
            kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)
        )

        self.assert_close(roundtrip, axis_angle, atol=1e-6, rtol=0.0)


class TestRadDegConversions(BaseTester):
    def test_pi(self):
        self.assert_close(kornia.constants.pi.item(), 3.141592)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_rad2deg(self, batch_shape, device, dtype):
        # generate input data
        x_rad = kornia.constants.pi * torch.rand(batch_shape, device=device, dtype=dtype)

        # convert radians/degrees
        x_deg = kornia.geometry.conversions.rad2deg(x_rad)
        x_deg_to_rad = kornia.geometry.conversions.deg2rad(x_deg)

        # compute error
        self.assert_close(x_rad, x_deg_to_rad)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_rad2deg_gradcheck(self, batch_shape, device):
        dtype = torch.float64
        x_rad = torch.rand(batch_shape, device=device, dtype=dtype)
        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.rad2deg, (x_rad,))

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_deg2rad(self, batch_shape, device, dtype, atol, rtol):
        # generate input data
        x_deg = 180.0 * torch.rand(batch_shape, device=device, dtype=dtype)

        # convert radians/degrees
        x_rad = kornia.geometry.conversions.deg2rad(x_deg)
        x_rad_to_deg = kornia.geometry.conversions.rad2deg(x_rad)

        self.assert_close(x_deg, x_rad_to_deg, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_deg2rad_gradcheck(self, batch_shape, device):
        x_deg = 180.0 * torch.rand(batch_shape, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.conversions.deg2rad, (x_deg,))

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="kornia.constants.pi is float32, so f64 loses ~7 digits (angle_to_rotation_matrix "
        "inherits it via deg2rad) — kornia#3937",
        strict=True,
    )
    @pytest.mark.parametrize(
        ("op_name", "arg", "expected"),
        [
            ("rad2deg", torch.pi, 180.0),
            ("deg2rad", 180.0, torch.pi),
            ("angle_to_rotation_matrix", 90.0, [[0.0, 1.0], [-1.0, 0.0]]),
        ],
    )
    def test_convention_float64_results_are_exact_3937(self, device, op_name, arg, expected):
        # Intended behavior: each op is exact to the precision of its input dtype, like
        # torch.rad2deg / torch.deg2rad; angle_to_rotation_matrix(90) is then the exact quarter
        # turn. It is not: all three multiply by kornia.constants.pi, a *float32* tensor merely
        # cast to the input dtype, so a float64 input carries a systematic ~2.8e-8 relative
        # error (#3937). float64 is hardcoded (like test_rad2deg_gradcheck above) because at
        # float32 the biased constant *is* the correctly rounded pi; MPS is skipped visibly
        # below because it has no float64 at all, so without the skip the xfail would be
        # satisfied by a TypeError instead of the precision assert it documents (hence also
        # raises=AssertionError on the mark). Marked xfail(strict=True) so fixing #3937 makes
        # every case XPASS and forces this mark out — a one-place edit.
        # Snippet used to generate expected (stdlib + torch):
        #   math.degrees(math.pi) == 180.0 and (180.0 * math.pi) / 180.0 == math.pi exactly
        #   kornia rad2deg(tensor(pi, f64)).item()   -> 179.99999499104382
        #   kornia deg2rad(tensor(180., f64)).item() -> 3.1415927410125732 (math.pi + 8.7e-08)
        #   kornia angle_to_rotation_matrix(tensor(90., f64)).flatten().tolist() ->
        #     [-4.371139000186241e-08, 0.999999999999999, -0.999999999999999, -4.371139e-08]
        # atol/rtol 1e-12 sits between the current ~4.4e-8 cosine error and the 6.123234e-17
        # an unbiased constant would give.
        if device.type == "mps":
            pytest.skip("MPS has no float64, and this pin is float64-only by construction")

        op = getattr(kornia.geometry.conversions, op_name)

        out = op(torch.tensor(arg, device=device, dtype=torch.float64))

        self.assert_close(out, torch.tensor(expected, device=device, dtype=torch.float64), atol=1e-12, rtol=1e-12)

    def test_convention_angle_to_rotation_matrix_takes_degrees(self, device, dtype):
        # Convention pin: angle_to_rotation_matrix reads its argument in DEGREES (not radians)
        # and returns [[cos, sin], [-sin, cos]] -- the transpose of the textbook math-frame CCW
        # matrix. Pinned on a small non-symmetric angle so a sign flip on the off-diagonal is
        # caught.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
        #   [[c, s], [-s, c]] -> [[0.8660254037844387, 0.49999999999999994],
        #                         [-0.49999999999999994, 0.8660254037844387]]
        out = kornia.geometry.conversions.angle_to_rotation_matrix(torch.tensor(30.0, device=device, dtype=dtype))
        expected = torch.tensor([[0.8660254, 0.5], [-0.5, 0.8660254]], device=device, dtype=dtype)
        self.assert_close(out, expected)

        # A radian-reading implementation would turn pi/2 into the quarter turn [[0, 1], [-1, 0]];
        # this one reads pi/2 as 1.5708 *degrees* and returns a near-identity matrix instead.
        # Snippet used to generate expected:
        #   c, s = math.cos(math.radians(math.pi / 2)), math.sin(math.radians(math.pi / 2))
        #   [[c, s], [-s, c]] -> [[0.9996242168385687, 0.027412134354665284], ...]
        out_rad = kornia.geometry.conversions.angle_to_rotation_matrix(
            torch.tensor(torch.pi / 2, device=device, dtype=dtype)
        )
        expected_rad = torch.tensor([[0.99962422, 0.02741213], [-0.02741213, 0.99962422]], device=device, dtype=dtype)
        self.assert_close(out_rad, expected_rad)

    @pytest.mark.parametrize(
        ("op_name", "arg", "expected"),
        [
            ("rad2deg", [1, 2, 3], [60.0, 120.0, 180.0]),
            ("deg2rad", [180, 90], [3.0, 1.5]),
            ("angle_to_rotation_matrix", [90], [[[0.07073720, 0.99749500], [-0.99749500, 0.07073720]]]),
        ],
    )
    def test_wart_integer_input_truncates_pi_to_3_3937(self, device, op_name, arg, expected):
        # Wart pins for #3937: assert the CURRENT broken outputs the docstring warnings document.
        # kornia.constants.pi is cast to the *integer* input dtype and truncates to 3, so rad2deg
        # divides by 3, deg2rad multiplies by 3 (90 degrees -> 1.5 radians), and the downstream
        # angle_to_rotation_matrix([90]) is nowhere near the quarter turn. If a case fails, #3937
        # was (partly) fixed -- update or remove the warnings in rad2deg, deg2rad and
        # angle_to_rotation_matrix and flip/remove the strict xfail above. NOT a contract that
        # int inputs must keep these values: what they *should* do (promote to float like
        # torch.rad2deg, or raise) is a maintainer decision, and a strict xfail asserting the
        # promoted-float answer would stay silently XFAIL forever if the fix chose to raise;
        # a wart pin flips loudly under either polarity.
        # Snippet used to generate expected (torch only):
        #   kornia rad2deg(torch.tensor([1, 2, 3])) -> tensor([ 60., 120., 180.]), dtype float32
        #     (torch.rad2deg gives [ 57.2958, 114.5916, 171.8873])
        #   kornia deg2rad(torch.tensor([180, 90])) -> tensor([3.0000, 1.5000]), dtype float32
        #     (torch.deg2rad gives [3.1416, 1.5708])
        #   kornia angle_to_rotation_matrix(torch.tensor([90])).flatten().tolist() ->
        #     [0.07073719799518585, 0.9974949955940247, -0.9974949955940247, 0.07073719799518585]
        #     (math.cos(1.5), math.sin(1.5) -> (0.0707372016677029, 0.9974949866040544))
        op = getattr(kornia.geometry.conversions, op_name)

        out = op(torch.tensor(arg, device=device))

        assert out.dtype == torch.float32
        self.assert_close(out, torch.tensor(expected, device=device, dtype=torch.float32), atol=1e-4, rtol=1e-4)


class TestPolCartConversions(BaseTester):
    def test_smoke(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        assert kornia.geometry.conversions.pol2cart(x, x) is not None
        assert kornia.geometry.conversions.cart2pol(x, x) is not None

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_pol2cart(self, batch_shape, device, dtype):
        # generate input data
        rho = torch.rand(batch_shape, dtype=dtype)
        phi = kornia.constants.pi * torch.rand(batch_shape, dtype=dtype)
        rho = rho.to(device)
        phi = phi.to(device)

        # convert pol/cart
        x_pol2cart, y_pol2cart = kornia.geometry.conversions.pol2cart(rho, phi)
        rho_pol2cart, phi_pol2cart = kornia.geometry.conversions.cart2pol(x_pol2cart, y_pol2cart, 0)

        self.assert_close(rho, rho_pol2cart)
        self.assert_close(phi, phi_pol2cart)

    @pytest.mark.parametrize("batch_shape", [(2, 3)])
    def test_gradcheck(self, batch_shape, device):
        rho = torch.rand(batch_shape, dtype=torch.float64, device=device)
        phi = kornia.constants.pi * torch.rand(batch_shape, dtype=torch.float64, device=device)
        self.gradcheck(kornia.geometry.conversions.pol2cart, (rho, phi))
        self.gradcheck(kornia.geometry.conversions.cart2pol, (rho, phi))

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_cart2pol(self, batch_shape, device, dtype):
        # generate input data
        x = torch.rand(batch_shape, dtype=dtype)
        y = torch.rand(batch_shape, dtype=dtype)
        x = x.to(device)
        y = y.to(device)

        # convert cart/pol
        rho_cart2pol, phi_cart2pol = kornia.geometry.conversions.cart2pol(x, y, 0)
        x_cart2pol, y_cart2pol = kornia.geometry.conversions.pol2cart(rho_cart2pol, phi_cart2pol)

        self.assert_close(x, x_cart2pol)
        self.assert_close(y, y_cart2pol)

    def test_convention_pol2cart_takes_rho_phi_returns_x_y(self, device, dtype):
        # Convention pin: pol2cart's argument order is (rho, phi) and its return order is
        # (x, y), with phi in RADIANS measured from the +x axis: x = rho*cos(phi),
        # y = rho*sin(phi). The literal is deliberately off-axis (3 != 4) so that swapping
        # either the arguments or the returns is caught.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   phi = math.atan2(4.0, 3.0)  # 0.9272952180016122 rad
        #   5.0 * math.cos(phi), 5.0 * math.sin(phi) -> (3.0000000000000004, 4.0)
        rho = torch.tensor(5.0, device=device, dtype=dtype)
        phi = torch.tensor(0.9272952180016122, device=device, dtype=dtype)

        x, y = kornia.geometry.conversions.pol2cart(rho, phi)

        self.assert_close(x, torch.tensor(3.0, device=device, dtype=dtype))
        self.assert_close(y, torch.tensor(4.0, device=device, dtype=dtype))

    def test_convention_cart2pol_takes_x_y_and_phi_is_atan2_y_x(self, device, dtype):
        # Convention pin: cart2pol's argument order is (x, y) and its return order is
        # (rho, phi), with phi = atan2(y, x) in radians -- zero on the +x axis and increasing
        # toward +y (which is clockwise *as displayed* under kornia's y-down image axes).
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   math.hypot(3.0, 4.0), math.atan2(4.0, 3.0) -> (5.0, 0.9272952180016122)
        #   math.atan2(1.0, 0.0) -> 1.5707963267948966  (atan2(x, y) would give 0.0 here)
        rho, phi = kornia.geometry.conversions.cart2pol(
            torch.tensor(3.0, device=device, dtype=dtype), torch.tensor(4.0, device=device, dtype=dtype)
        )
        self.assert_close(rho, torch.tensor(5.0, device=device, dtype=dtype))
        self.assert_close(phi, torch.tensor(0.9272952180016122, device=device, dtype=dtype))

        phi_y_axis = kornia.geometry.conversions.cart2pol(
            torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
        )[1]
        self.assert_close(phi_y_axis, torch.tensor(1.5707963267948966, device=device, dtype=dtype))

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="cart2pol returns sqrt(x**2 + y**2 + eps), biasing rho — kornia#3939",
        strict=True,
    )
    def test_convention_cart2pol_rho_is_the_exact_radius(self, device, dtype):
        # Intended behavior: rho is the Euclidean radius, so rho(0, 0) == 0. It currently is
        # not: eps is added *inside* the sqrt, so rho = sqrt(x**2 + y**2 + eps) and the origin
        # maps to sqrt(1e-8) = 1e-4 (see #3939; eps belongs in the gradient path, not the
        # value). Marked xfail(strict=True) so fixing #3939 makes this XPASS loudly.
        # Snippet used to generate expected (stdlib only):
        #   math.hypot(0.0, 0.0) -> 0.0 ; kornia cart2pol(0., 0.)[0].item() -> 0.0001
        if dtype == torch.float16:
            pytest.skip("float16 cannot represent the default eps=1e-8, so the bias is invisible there")

        rho = kornia.geometry.conversions.cart2pol(
            torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
        )[0]
        self.assert_close(rho, torch.tensor(0.0, device=device, dtype=dtype), atol=1e-6, rtol=0.0)

    def test_wart_rho_is_biased_by_eps_inside_the_sqrt_3939(self, device, dtype):
        # Wart pin for kornia#3939, companion to the strict xfail above: assert the CURRENT
        # biased rho. The xfail pins the intended rho(0, 0) == 0 but cannot flip under every fix
        # polarity -- the equally standard sqrt(clamp(x**2 + y**2, min=eps)) (the shape
        # normalize_pixel_coordinates already uses) also returns 1e-4 at the origin, leaving the
        # mark silently XFAIL with a stale reason string. So two cells are pinned: the origin,
        # rho = sqrt(eps) = 1e-4, which flips under a grad-only eps (rho 0) and under eps**2
        # inside the sqrt (rho 1e-8); and a sub-eps point x = 5e-5, whose x**2 = 2.5e-9 < eps
        # gives rho = sqrt(1.25e-8) ~ 1.118e-4, which additionally flips under the clamp shape
        # (rho 1e-4, 10.6 % below, outside rtol 1e-2). If either assert fails, #3939 was
        # (partly) fixed -- update or remove the warning in cart2pol and flip/remove the strict
        # xfail above. eps=1e-8 is passed explicitly so the pinned literals do not silently
        # track a later change to the default.
        # Snippet used to generate expected (torch only, executed at each pinned dtype):
        #   c2p = kornia.geometry.conversions.cart2pol
        #   c2p(torch.tensor(0., dtype=torch.float64), torch.tensor(0., dtype=torch.float64),
        #       eps=1e-8)[0] -> 0.0001                    (f32: 9.999999747378752e-05)
        #   c2p(torch.tensor(5e-5, dtype=torch.float64), torch.tensor(0., dtype=torch.float64),
        #       eps=1e-8)[0] -> 0.00011180339887498949    (f32: 0.00011180339788552374)
        # At bfloat16 the outputs land within 0.3 % of the literals (1.00136e-4, 1.12057e-4),
        # inside rtol 1e-2, so the pin holds there too.
        if dtype == torch.float16:
            pytest.skip("float16 cannot represent eps=1e-8, so rho is 0 at both pinned points and the bias invisible")

        zero = torch.tensor(0.0, device=device, dtype=dtype)

        rho_origin = kornia.geometry.conversions.cart2pol(zero, zero, eps=1e-8)[0]
        rho_sub_eps = kornia.geometry.conversions.cart2pol(
            torch.tensor(5e-5, device=device, dtype=dtype), zero, eps=1e-8
        )[0]

        self.assert_close(rho_origin, torch.tensor(1e-4, device=device, dtype=dtype), atol=0.0, rtol=1e-2)
        self.assert_close(
            rho_sub_eps, torch.tensor(1.1180339887498949e-4, device=device, dtype=dtype), atol=0.0, rtol=1e-2
        )

    def test_convention_positive_rotation_decreases_cart2pol_phi(self, device, dtype):
        # Cross-symbol convention pin: enforces the opposite-sense relation between
        # angle_to_rotation_matrix and cart2pol stated canonically in cart2pol's Convention
        # block (phi decreases by theta modulo 2*pi).
        # First case: no branch-cut crossing, so the raw difference is -theta itself.
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   -math.radians(30.0) -> -0.5235987755982988
        v = torch.tensor([3.0, 4.0], device=device, dtype=dtype)
        phi0 = kornia.geometry.conversions.cart2pol(v[0], v[1])[1]

        rot = kornia.geometry.conversions.angle_to_rotation_matrix(torch.tensor(30.0, device=device, dtype=dtype))
        v_rot = rot @ v
        phi1 = kornia.geometry.conversions.cart2pol(v_rot[0], v_rot[1])[1]

        expected_delta = torch.tensor(-0.5235987755982988, device=device, dtype=dtype)
        self.assert_close(phi1 - phi0, expected_delta)

        # Second case: crossing the -x branch cut, where the returned phi is re-wrapped into
        # [-pi, pi] and only the difference modulo 2*pi is -theta (the worked -170 + 30 example
        # lives in cart2pol's Convention block).
        # Snippet used to generate expected (stdlib only):
        #   import math
        #   5 * math.cos(math.radians(-170.0)), 5 * math.sin(math.radians(-170.0))
        #     -> (-4.92403876506104, -0.8682408883346514)
        #   math.radians(160.0) -> 2.792526803190927
        w = torch.tensor([-4.9240388, -0.8682409], device=device, dtype=dtype)
        phi0_cut = kornia.geometry.conversions.cart2pol(w[0], w[1])[1]

        w_rot = rot @ w
        phi1_cut = kornia.geometry.conversions.cart2pol(w_rot[0], w_rot[1])[1]

        self.assert_close(phi1_cut, torch.tensor(2.7925268, device=device, dtype=dtype))

        raw_delta = phi1_cut - phi0_cut
        wrapped_delta = torch.atan2(torch.sin(raw_delta), torch.cos(raw_delta))
        # The re-wrap atan2(sin, cos) adds two more transcendental roundings on top of the two
        # atan2 outputs it differences, overshooting the central per-dtype tolerances in the half
        # dtypes. Measured against the dtype-cast expected tensor the assert compares with
        # (-0.5234375 in both halves): |err| is 1.953125e-3 in float16 (wrapped -0.525390625;
        # central allowance atol 1e-3 + rtol 1e-3 * 0.52 = 1.52e-3) and 1.171875e-2 in bfloat16
        # (wrapped -0.53515625; allowance 1.19e-2 -- a 1.4 % margin that torch rounding drift
        # could erase). atol 2.4e-2 is ~2x the bfloat16 error; a sign-flipped or unwrapped delta
        # would still be off by >= 1.0.
        wrap_tol = {"atol": 2.4e-2, "rtol": 0.0} if dtype in (torch.float16, torch.bfloat16) else {}
        self.assert_close(wrapped_delta, expected_delta, **wrap_tol)


class TestConvertPointsToHomogeneous(BaseTester):
    def test_convert_points(self, device, dtype):
        # Convention pin: the homogeneous 1.0 is appended as the *last* component (the
        # non-symmetric rows catch a prepend or a component reversal).
        points_h = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [-1.0, -2.0, -1.0], [0.0, 1.0, -2.0]],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [1.0, 2.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 1.0],
                [2.0, 1.0, 0.0, 1.0],
                [-1.0, -2.0, -1.0, 1.0],
                [0.0, 1.0, -2.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_to_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_convert_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[2.0, 1.0, 0.0, 1.0]], [[0.0, 1.0, 2.0, 1.0]], [[0.0, 1.0, -2.0, 1.0]]], device=device, dtype=dtype
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_to_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_gradcheck(self, batch_shape, device):
        points_h = torch.rand(batch_shape, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_points_to_homogeneous, (points_h,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_points_to_homogeneous
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        self.assert_close(actual, expected)


class TestConvertAtoH(BaseTester):
    def test_convert_points(self, device, dtype):
        # Convention pin and its enforcement point: the (B, 2, 3) affine block is copied
        # verbatim into the top of the (B, 3, 3) result (no transpose, no reordering) and the
        # row [0, 0, 1] is appended at the *bottom*. The literal is non-symmetric so a
        # transpose is caught.
        # Snippet used to generate expected (torch only):
        #   convert_affinematrix_to_homography(torch.tensor([[[1., 2., 3.], [4., 5., 6.]]]))
        #     -> [[[1., 2., 3.], [4., 5., 6.], [0., 0., 1.]]]
        A = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], device=device, dtype=dtype)

        expected = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        H = kornia.geometry.conversions.convert_affinematrix_to_homography(A)
        self.assert_close(H, expected)

    @pytest.mark.parametrize("batch_shape", [(10, 2, 3), (16, 2, 3)])
    def test_gradcheck(self, batch_shape, device):
        points_h = torch.rand(batch_shape, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_affinematrix_to_homography, (points_h,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_affinematrix_to_homography
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        self.assert_close(actual, expected)

    def test_convention_homography3d_appends_bottom_row_0_0_0_1(self, device, dtype):
        # Convention pin (3-D sibling, which has no test class of its own): the (B, 3, 4) affine
        # block is copied verbatim and the row [0, 0, 0, 1] is appended at the bottom.
        # Snippet used to generate expected (by hand):
        #   [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] gains [0, 0, 0, 1]
        A = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.convert_affinematrix_to_homography3d(A)

        expected = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out, expected)


class TestConvertPointsFromHomogeneous(BaseTester):
    @pytest.mark.parametrize("batch_shape", [(2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3)])
    def test_cardinality(self, device, dtype, batch_shape):
        points_h = torch.rand(batch_shape, device=device, dtype=dtype)
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        assert points.shape == points.shape[:-1] + (2,)

    def test_points(self, device, dtype):
        # Convention pins: the [2., 1., 0.] row is the |w| <= eps case (default eps 1e-8) --
        # returned *unchanged*, not zeros, not inf, no exception. The negative-w rows pin that
        # the sign of w is preserved (no abs): [0., 1., -2.] -> [0., -0.5].
        points_h = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [-1.0, -2.0, -1.0], [0.0, 1.0, -2.0]],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [[1.0, 2.0], [0.0, 0.5], [2.0, 1.0], [1.0, 2.0], [0.0, -0.5]], device=device, dtype=dtype
        )

        # to euclidean
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_points_batch(self, device, dtype):
        # generate input data
        points_h = torch.tensor([[[2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, -2.0]]], device=device, dtype=dtype)

        expected = torch.tensor([[[2.0, 1.0]], [[0.0, 0.5]], [[0.0, -0.5]]], device=device, dtype=dtype)

        # to euclidean
        points = kornia.geometry.conversions.convert_points_from_homogeneous(points_h)
        self.assert_close(points, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_h = torch.ones(1, 10, 3, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_points_from_homogeneous, (points_h,))

    def test_gradcheck_zvec_zeros(self, device):
        # generate input data
        points_h = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.1], [2.0, 1.0, 0.1]], device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.convert_points_from_homogeneous, (points_h,), eps=1e-8)

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_h = torch.zeros(1, 2, 3, device=device, dtype=dtype)

        op = kornia.geometry.conversions.convert_points_from_homogeneous
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_h)
        expected = op(points_h)

        self.assert_close(actual, expected)

    def test_convention_divides_by_exactly_w(self, device, dtype):
        # For |w| > eps the point is divided by exactly w, matching OpenCV's
        # convertPointsFromHomogeneous (scale = fabs(w) > FLT_EPSILON ? 1./w : 1.), see
        # https://github.com/opencv/opencv/pull/14411/files. This used to divide by w + eps,
        # which made the result 33 % low here (#3938).
        # Snippet used to generate expected (by hand):
        #   2 / 2e-8, 4 / 2e-8 -> [1e8, 2e8]
        if dtype == torch.float16:
            pytest.skip("float16 underflows w=2e-8 to 0, which takes the |w| <= eps passthrough branch instead")

        points = torch.tensor([[2.0, 4.0, 2e-8]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.convert_points_from_homogeneous(points)

        expected = torch.tensor([[1e8, 2e8]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_division_is_exact_for_both_signs_of_w(self, device, dtype):
        # Companion to the test above: the same small |w| in both signs. The division used to
        # be by w + eps, which is not sign-aware, so it grew a small positive denominator and
        # shrank a small negative one -- 33 % low at w = +2e-8 and 100 % high at w = -2e-8,
        # errors in opposite directions for inputs that differ only in the sign of w (#3938).
        # w is a power of two just above the default eps (2 ** -26 = 1.4901161e-8 > 1e-8), so
        # the quotients are powers of two too and are exact in every dtype that can hold them.
        # That is what lets this assert with zero tolerance. eps is passed explicitly so the
        # literals do not silently track a later change to the default.
        # Snippet used to generate expected (by hand, all values exact in binary):
        #   2 / 2 ** -26 -> 2 ** 27 = 134217728.0,  4 / 2 ** -26 -> 2 ** 28 = 268435456.0
        if dtype == torch.float16:
            pytest.skip("float16 underflows w=2**-26 to 0 (the |w| <= eps passthrough branch) and overflows 2**27")

        cpfh = kornia.geometry.conversions.convert_points_from_homogeneous
        w = 2.0**-26

        out_pos = cpfh(torch.tensor([[2.0, 4.0, w]], device=device, dtype=dtype), eps=1e-8)
        out_neg = cpfh(torch.tensor([[2.0, 4.0, -w]], device=device, dtype=dtype), eps=1e-8)

        expected_pos = torch.tensor([[2.0**27, 2.0**28]], device=device, dtype=dtype)
        self.assert_close(out_pos, expected_pos, atol=0.0, rtol=0.0)
        self.assert_close(out_neg, -expected_pos, atol=0.0, rtol=0.0)

    def test_roundtrip_from_to_homogeneous_is_identity(self, device, dtype):
        # Oracle-free invariant: convert_points_to_homogeneous appends w = 1, so dividing by
        # exactly w must return the input untouched in any dtype. Dividing by w + eps instead
        # made this an identity only to ~1e-8 relative, worse than float32 even in float64 (#3938).
        points = torch.tensor([[1.5, -2.5], [0.0, 3.25], [-4.75, 0.125]], device=device, dtype=dtype)

        cpth = kornia.geometry.conversions.convert_points_to_homogeneous
        cpfh = kornia.geometry.conversions.convert_points_from_homogeneous

        self.assert_close(cpfh(cpth(points)), points, atol=0.0, rtol=0.0)

    def test_convention_the_masked_division_keeps_the_gradient_finite(self, device):
        # Companion to the forward pins above, for the half of the fix they cannot see. The old
        # `1.0 / (z_vec + eps)` is evaluated for EVERY point, including the ones the mask discards.
        # At w == -eps that denominator is exactly zero, so the reciprocal is inf; torch.where drops
        # that lane from the forward value but the backward pass still multiplies it by the zero
        # cotangent and 0 * inf is NaN. Measured on the unfixed code the gradient at this input is
        # [1., 1., nan]; the double-`where` makes it [1., 1., 0.].
        # w is the exact pole rather than merely a small value: a nearby w only makes the reciprocal
        # large, which no assertion on finiteness would catch. eps is passed explicitly so the input
        # tracks the guard rather than the default. float32 and float64 both represent -eps and eps
        # identically enough for their sum to hit exactly zero; float32 keeps the pin active on MPS,
        # which cannot represent float64.
        eps = 1e-8
        regression_dtype = torch.float32 if device.type == "mps" else torch.float64
        points = torch.tensor([[2.0, 4.0, -eps]], device=device, dtype=regression_dtype, requires_grad=True)

        kornia.geometry.conversions.convert_points_from_homogeneous(points, eps=eps).sum().backward()

        assert torch.isfinite(points.grad).all(), (
            "kornia#3938: convert_points_from_homogeneous divides by exactly zero at w == -eps, so "
            f"the discarded branch poisons the gradient: {points.grad.tolist()}"
        )


def _skip_if_mps_clamp_caching(device):
    # Runtime probe instead of a torch-version pin, so the skip retires itself on any torch
    # build where the two clamps below return different values.
    if device.type == "mps" and torch.equal(
        torch.zeros(2, device=device).clamp(1e-8), torch.zeros(2, device=device).clamp(1e-7)
    ):
        pytest.skip(
            "this torch build caches clamp's scalar min per shape/dtype on MPS -- first value wins "
            "(seen on torch 2.9.1): z = torch.zeros(2, device='mps'); z.clamp(1e-8) then z.clamp(1e-7) "
            "both return 9.99999993922529e-09, while the same pair on cpu returns 1e-08 then "
            "1.0000000116860974e-07. The clamped eps this pin measures is therefore set by whichever "
            "earlier test clamped first, which is a torch defect, not a kornia one"
        )


class TestNormalizePixelCoordinates(BaseTester):
    def test_tensor_bhw2(self, device, dtype, atol, rtol):
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.geometry.conversions.normalize_pixel_coordinates(grid, height, width)

        self.assert_close(grid_norm, expected, atol=atol, rtol=rtol)

    def test_list(self, device, dtype, atol, rtol):
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.geometry.conversions.normalize_pixel_coordinates(grid, height, width)

        self.assert_close(grid_norm, expected, atol=atol, rtol=rtol)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if device == torch.device("cpu"):
            pytest.skip("NormalizePixelCoordinates not working on CPU with dynamo!")

        op = kornia.geometry.conversions.normalize_pixel_coordinates
        op_optimized = torch_optimizer(op)

        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        actual = op_optimized(grid, height, width)
        expected = op(grid, height, width)

        self.assert_close(actual, expected)

    def test_convention_corner_aligned_formula(self, device, dtype):
        # Convention pin: normalize_pixel_coordinates maps x -> 2*x/(W - 1) - 1 (corner-aligned,
        # i.e. the align_corners=True convention). grid_sample's *default* align_corners=False
        # convention, (2*x + 1)/W - 1, would give [-0.75, -0.25, 0.75] for the same input.
        # Snippet used to generate expected (stdlib only, W = 4):
        #   [2 * x / (4 - 1) - 1 for x in (0.0, 1.0, 3.0)] -> [-1.0, -0.3333333333333333, 1.0]
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 4, 4)

        expected = torch.tensor([[-1.0, -1.0], [-0.33333333, -0.33333333], [1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_positional_order_is_height_then_width(self, device, dtype):
        # Convention pin: the positional signature is (pixel_coordinates, height, width), which is
        # the reverse of the per-point (x, y) -> (width, height) scaling order: slot 0 is scaled by
        # width and slot 1 by height. Calling with H and W swapped would give [[5.0, -0.3333]].
        # Snippet used to generate expected (stdlib only, H = 2, W = 4):
        #   2 * 3.0 / (4 - 1) - 1, 2 * 1.0 / (2 - 1) - 1 -> (1.0, 1.0)
        pts = torch.tensor([[3.0, 1.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 2, 4)

        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_output_is_not_clamped(self, device, dtype):
        # Convention pin: nothing is clamped to [-1, 1] -- out-of-image coordinates extrapolate
        # linearly past it.
        # Snippet used to generate expected (stdlib only, H = W = 4):
        #   2 * 10.0 / (4 - 1) - 1, 2 * 0.0 / (4 - 1) - 1 -> (5.666666666666666, -1.0)
        pts = torch.tensor([[10.0, 0.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 4, 4)

        expected = torch.tensor([[5.6666667, -1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_grid_sample_needs_align_corners_true(self, device, dtype):
        # Convention pin: feeding normalized coordinates to torch.nn.functional.grid_sample
        # requires align_corners=True to be passed explicitly. With it, the three normalized
        # pixel centres sample back the exact pixel values; grid_sample's own default
        # (align_corners=None -> False) instead places u = -1, -1/3, 1 at pixels
        # ((u + 1) * 4 - 1) / 2 = -0.5, 0.8333, 3.5, i.e. half a pixel outside the image at
        # both ends, so every sampled value would be wrong.
        # Snippet used to generate expected (stdlib only, W = 4, img = arange(16).view(4, 4)):
        #   img[0, 0], img[1, 1], img[3, 3] -> 0.0, 5.0, 15.0
        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], device=device, dtype=dtype)
        grid = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 4, 4).view(1, 1, 3, 2)

        sampled_aligned = torch.nn.functional.grid_sample(img, grid, align_corners=True).flatten()

        self.assert_close(sampled_aligned, torch.tensor([0.0, 5.0, 15.0], device=device, dtype=dtype))

    def test_convention_3d_component_order_is_depth_x_y(self, device, dtype):
        # Convention pin (normalize_pixel_coordinates3d has no test class of its own): the
        # component order is (d, x, y) -- depth first, then x scaled by width, then y scaled by
        # height. It is NOT (x, y, z): reading the same three numbers that way sends the point
        # out of range instead of to the far corner.
        # Snippet used to generate expected (stdlib only, D = 3, H = 5, W = 9):
        #   2 * 2 / (3 - 1) - 1, 2 * 8 / (9 - 1) - 1, 2 * 4 / (5 - 1) - 1 -> (1.0, 1.0, 1.0)
        #   the (x, y, z) reading [2, 4, 8] gives 2 * 2 / 2 - 1, 2 * 4 / 8 - 1, 2 * 8 / 4 - 1
        #                                      -> (1.0, 0.0, 3.0)
        far_corner = torch.tensor([[2.0, 8.0, 4.0]], device=device, dtype=dtype)
        out = kornia.geometry.conversions.normalize_pixel_coordinates3d(far_corner, 3, 5, 9)
        self.assert_close(out, torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=dtype))

        swapped = torch.tensor([[2.0, 4.0, 8.0]], device=device, dtype=dtype)
        out_swapped = kornia.geometry.conversions.normalize_pixel_coordinates3d(swapped, 3, 5, 9)
        self.assert_close(out_swapped, torch.tensor([[1.0, 0.0, 3.0]], device=device, dtype=dtype))

    def test_singleton_axes_map_to_center_and_keep_unit_extension(self, device, dtype):
        pts = torch.tensor([[0.0, 0.0], [0.25, -0.5]], device=device, dtype=dtype)
        out = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 1, 1)
        self.assert_close(out, pts, atol=0.0, rtol=0.0)

        pts3d = torch.tensor([[0.0, 0.0, 0.0], [0.25, -0.5, 0.75]], device=device, dtype=dtype)
        out3d = kornia.geometry.conversions.normalize_pixel_coordinates3d(pts3d, 1, 1, 1)
        self.assert_close(out3d, pts3d, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("bad_size", [0, -3], ids=["zero", "negative"])
    @pytest.mark.parametrize(
        ("op_name", "sizes"),
        [
            ("normalize_pixel_coordinates", (5, 7)),
            ("normalize_pixel_coordinates3d", (5, 7, 9)),
        ],
    )
    def test_non_positive_sizes_raise(self, op_name, sizes, bad_size, device, dtype):
        op = getattr(kornia.geometry.conversions, op_name)
        for index in range(len(sizes)):
            bad_sizes = list(sizes)
            bad_sizes[index] = bad_size
            coords = torch.zeros(1, len(sizes), device=device, dtype=dtype)
            with pytest.raises(ValueError, match="must be positive"):
                op(coords, *bad_sizes)

    def test_normalize_and_denormalize_trace_cross_singleton_boundary(self, device, dtype):
        class Convert(torch.nn.Module):
            def forward(self, image, coords):
                height, width = image.shape[-2], image.shape[-1]
                return (
                    kornia.geometry.conversions.normalize_pixel_coordinates(coords, height, width),
                    kornia.geometry.conversions.denormalize_pixel_coordinates(coords, height, width),
                )

        coords = torch.tensor([[0.0, 0.0], [0.5, 0.5]], device=device, dtype=dtype)
        for trace_height, runtime_height in ((2, 1), (1, 2)):
            example = torch.zeros(1, 1, trace_height, 4, device=device, dtype=dtype)
            runtime = torch.zeros(1, 1, runtime_height, 4, device=device, dtype=dtype)
            convert = Convert()
            traced = torch.jit.trace(convert, (example, coords))
            actual = traced(runtime, coords)
            expected = convert(runtime, coords)
            self.assert_close(actual[0], expected[0], atol=0.0, rtol=0.0)
            self.assert_close(actual[1], expected[1], atol=0.0, rtol=0.0)

        class Convert3d(torch.nn.Module):
            def forward(self, volume, coords):
                depth, height, width = volume.shape[-3], volume.shape[-2], volume.shape[-1]
                return (
                    kornia.geometry.conversions.normalize_pixel_coordinates3d(coords, depth, height, width),
                    kornia.geometry.conversions.denormalize_pixel_coordinates3d(coords, depth, height, width),
                )

        coords3d = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=device, dtype=dtype)
        for trace_depth, runtime_depth in ((2, 1), (1, 2)):
            example = torch.zeros(1, 1, trace_depth, 3, 4, device=device, dtype=dtype)
            runtime = torch.zeros(1, 1, runtime_depth, 3, 4, device=device, dtype=dtype)
            convert = Convert3d()
            traced = torch.jit.trace(convert, (example, coords3d))
            actual = traced(runtime, coords3d)
            expected = convert(runtime, coords3d)
            self.assert_close(actual[0], expected[0], atol=0.0, rtol=0.0)
            self.assert_close(actual[1], expected[1], atol=0.0, rtol=0.0)


def test_wart_default_eps_1e_8_backs_the_remaining_quoted_warning_numbers():
    # These two APIs still use eps numerically and quote outputs based on its default.
    for op_name in ("cart2pol", "convert_points_from_homogeneous"):
        op = getattr(kornia.geometry.conversions, op_name)
        assert inspect.signature(op).parameters["eps"].default == 1e-8, op_name


def test_wart_float16_underflowed_default_eps_flips_branches(device):
    # Wart pin for the float16 sentence of the #3939 warning and for the float16 sentence of
    # the convert_points_from_homogeneous Convention block. float16 is hardcoded (no dtype
    # fixture) so the pins run in every test configuration: the float16 legs of the tests
    # above are skipped because the default eps=1e-8 underflows to 0 there, which is exactly
    # the behavior pinned here. eps is left at its default on purpose
    # -- the underflow of the *default* is the claim. atol=rtol=0.0 because both claims are
    # exactness claims: with the float16 default tolerance (1e-3) the eps-biased
    # rho = 1e-4 of the other branch would still compare equal to 0.
    # Snippet used to generate expected (torch only, executed on cpu float16):
    #   cart2pol(torch.tensor(0., dtype=torch.float16), torch.tensor(0., dtype=torch.float16))[0]
    #     -> 0.0  (not sqrt(eps) = 1e-4: eps underflows the sum inside the sqrt)
    #   convert_points_from_homogeneous(torch.tensor([[2., 4., 2e-8]], dtype=torch.float16))
    #     -> [[2., 4.]]  (w underflows to 0 and takes the abs(w) <= eps passthrough branch)
    # The cart2pol half is narrowed rather than skipped whole: it reaches torch.atan2, whose CPU
    # float16 kernel ("atan2_cpu" not implemented for 'Half') only landed in PyTorch 2.2, while the
    # convert_points_from_homogeneous claim below has no such floor and stays pinned everywhere.
    if not (device.type == "cpu" and torch_version_lt(2, 2, 0)):
        zero = torch.tensor(0.0, device=device, dtype=torch.float16)
        rho = kornia.geometry.conversions.cart2pol(zero, zero)[0]
        assert_close(rho, zero, atol=0.0, rtol=0.0)

    out = kornia.geometry.conversions.convert_points_from_homogeneous(
        torch.tensor([[2.0, 4.0, 2e-8]], device=device, dtype=torch.float16)
    )
    assert_close(out, torch.tensor([[2.0, 4.0]], device=device, dtype=torch.float16), atol=0.0, rtol=0.0)


class TestDenormalizePixelCoordinates(BaseTester):
    def test_tensor_bhw2(self, device, dtype):
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )

        grid_norm = kornia.geometry.conversions.denormalize_pixel_coordinates(grid, height, width)

        self.assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_list(self, device, dtype):
        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )
        grid = grid.contiguous().view(-1, 2)

        expected = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(
            dtype=dtype
        )
        expected = expected.contiguous().view(-1, 2)

        grid_norm = kornia.geometry.conversions.denormalize_pixel_coordinates(grid, height, width)

        self.assert_close(grid_norm, expected, atol=1e-4, rtol=1e-4)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if device == torch.device("cpu"):
            pytest.xfail("DenormalizePixelCoordinates not working on CPU with dynamo!")

        op = kornia.geometry.conversions.denormalize_pixel_coordinates
        op_optimized = torch_optimizer(op)

        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device).to(
            dtype=dtype
        )

        actual = op_optimized(grid, height, width)
        expected = op(grid, height, width)

        self.assert_close(actual, expected)

    def test_convention_corner_aligned_inverse(self, device, dtype):
        # Convention pin: denormalize_pixel_coordinates is the corner-aligned inverse,
        # x = (W - 1) * (x_norm + 1) / 2, taken positionally as (coords, height, width) with
        # (x, y) points. grid_sample's align_corners=False convention, ((x_norm + 1) * W - 1)/2,
        # would give [[3.5, -0.5]] for the same input.
        # Snippet used to generate expected (stdlib only, H = 2, W = 4):
        #   (4 - 1) * (1.0 + 1) / 2, (2 - 1) * (-1.0 + 1) / 2 -> (3.0, 0.0)
        pts_norm = torch.tensor([[1.0, -1.0]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.denormalize_pixel_coordinates(pts_norm, 2, 4)

        expected = torch.tensor([[3.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_roundtrip_denormalize_of_normalize(self, device, dtype):
        # Convention pin: denormalize(normalize(p)) == p on a non-degenerate, non-square,
        # non-identity image size, so the two formulas are exact mutual inverses.
        # Snippet used to generate expected (by hand): the input itself.
        pts = torch.tensor([[1.0, 2.0], [3.0, 0.0]], device=device, dtype=dtype)

        norm = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 5, 7)
        out = kornia.geometry.conversions.denormalize_pixel_coordinates(norm, 5, 7)

        self.assert_close(out, pts)

    def test_convention_3d_component_order_and_roundtrip(self, device, dtype):
        # Convention pin (denormalize_pixel_coordinates3d has no test class of its own): same
        # (d, x, y) order as the 3-D normalizer, so the normalized origin maps to the per-axis
        # centres ((D - 1)/2, (W - 1)/2, (H - 1)/2), and the pair round-trips exactly.
        # Snippet used to generate expected (stdlib only, D = 3, H = 5, W = 9):
        #   (3 - 1) / 2, (9 - 1) / 2, (5 - 1) / 2 -> (1.0, 4.0, 2.0)
        centre = kornia.geometry.conversions.denormalize_pixel_coordinates3d(
            torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype), 3, 5, 9
        )
        self.assert_close(centre, torch.tensor([[1.0, 4.0, 2.0]], device=device, dtype=dtype))

        pts = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        norm = kornia.geometry.conversions.normalize_pixel_coordinates3d(pts, 3, 5, 9)
        out = kornia.geometry.conversions.denormalize_pixel_coordinates3d(norm, 3, 5, 9)
        self.assert_close(out, pts)

    def test_singleton_axes_are_exact_inverses(self, device, dtype):
        pts = torch.tensor([[0.0, 0.0], [0.25, -0.5]], device=device, dtype=dtype)
        norm = kornia.geometry.conversions.normalize_pixel_coordinates(pts, 1, 1)
        out = kornia.geometry.conversions.denormalize_pixel_coordinates(norm, 1, 1)
        self.assert_close(out, pts, atol=0.0, rtol=0.0)

        pts3d = torch.tensor([[0.0, 0.0, 0.0], [0.25, -0.5, 0.75]], device=device, dtype=dtype)
        norm3d = kornia.geometry.conversions.normalize_pixel_coordinates3d(pts3d, 1, 1, 1)
        out3d = kornia.geometry.conversions.denormalize_pixel_coordinates3d(norm3d, 1, 1, 1)
        self.assert_close(out3d, pts3d, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("bad_size", [0, -3], ids=["zero", "negative"])
    @pytest.mark.parametrize(
        ("op_name", "sizes"),
        [
            ("denormalize_pixel_coordinates", (5, 7)),
            ("denormalize_pixel_coordinates3d", (5, 7, 9)),
        ],
    )
    def test_non_positive_sizes_raise(self, op_name, sizes, bad_size, device, dtype):
        op = getattr(kornia.geometry.conversions, op_name)
        for index in range(len(sizes)):
            bad_sizes = list(sizes)
            bad_sizes[index] = bad_size
            coords = torch.zeros(1, len(sizes), device=device, dtype=dtype)
            with pytest.raises(ValueError, match="must be positive"):
                op(coords, *bad_sizes)

    @pytest.mark.parametrize(
        "op_name",
        [
            "normalize_pixel_coordinates",
            "denormalize_pixel_coordinates",
            "normalize_pixel_coordinates3d",
            "denormalize_pixel_coordinates3d",
        ],
    )
    def test_non_default_eps_warns_and_is_ignored(self, op_name, device, dtype):
        op = getattr(kornia.geometry.conversions, op_name)
        ndim = 3 if op_name.endswith("3d") else 2
        coords = torch.zeros(1, ndim, device=device, dtype=dtype)
        sizes = (1, 1, 1) if ndim == 3 else (1, 1)
        expected = op(coords, *sizes)
        with pytest.warns(FutureWarning, match="deprecated and ignored"):
            actual = op(coords, *sizes, eps=1.0)
        self.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("op_name", "args"),
    [
        ("normalize_pixel_coordinates", (1, 1)),
        ("denormalize_pixel_coordinates", (1, 1)),
        ("normalize_pixel_coordinates3d", (1, 1, 1)),
        ("denormalize_pixel_coordinates3d", (1, 1, 1)),
    ],
)
def test_pixel_coordinate_singleton_policy_scripts(op_name, args, device, dtype):
    op = getattr(kornia.geometry.conversions, op_name)
    coords = torch.zeros(1, len(args), device=device, dtype=dtype)
    scripted = torch.jit.script(op)
    assert_close(scripted(coords, *args), op(coords, *args), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("op_name", "ndim"),
    [
        ("normalize_pixel_coordinates", 2),
        ("denormalize_pixel_coordinates", 2),
        ("normalize_pixel_coordinates3d", 3),
        ("denormalize_pixel_coordinates3d", 3),
    ],
)
@pytest.mark.skipif(not dynamic_export_is_available(), reason=DYNAMIC_EXPORT_UNAVAILABLE_REASON)
def test_pixel_coordinate_export_crosses_singleton_boundary(op_name, ndim):
    op = getattr(kornia.geometry.conversions, op_name)

    class ExportCoordinates(torch.nn.Module):
        def forward(self, image, coords):
            if ndim == 2:
                return op(coords, image.shape[-2], image.shape[-1])
            return op(coords, image.shape[-3], image.shape[-2], image.shape[-1])

    image_shape = (1, 1, 2, 4) if ndim == 2 else (1, 1, 2, 3, 4)
    example = torch.zeros(image_shape)
    coords = torch.tensor([[0.0] * ndim, [0.5] * ndim])
    exported = torch.export.export(
        ExportCoordinates(),
        (example, coords),
        dynamic_shapes=({2: torch.export.Dim("singleton_axis", min=1, max=8)}, None),
    ).module()

    for runtime_size in (1, 5):
        runtime = torch.zeros(*image_shape[:2], runtime_size, *image_shape[3:])
        assert_close(exported(runtime, coords), ExportCoordinates()(runtime, coords), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("op_name", "sizes"),
    [
        ("normalize_pixel_coordinates", (4, 5)),
        ("denormalize_pixel_coordinates", (4, 5)),
        ("normalize_pixel_coordinates3d", (4, 5, 6)),
        ("denormalize_pixel_coordinates3d", (4, 5, 6)),
        ("normal_transform_pixel", (4, 5)),
        ("normal_transform_pixel3d", (4, 5, 6)),
    ],
)
@pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
def test_non_default_eps_does_not_break_fullgraph_compile(op_name, sizes):
    op = getattr(kornia.geometry, op_name)
    value = torch.zeros(1, len(sizes))
    if op_name.startswith("normal_transform"):

        def captured(tensor):
            return op(*sizes, eps=1e-6, device=tensor.device, dtype=tensor.dtype)

        expected = op(*sizes, device=value.device, dtype=value.dtype)
    else:

        def captured(tensor):
            return op(tensor, *sizes, eps=1e-6)

        expected = op(value, *sizes)

    actual = torch.compile(captured, fullgraph=True)(value)
    assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("is_3d", [False, True], ids=["2d", "3d"])
@pytest.mark.parametrize(
    ("dtype", "runtime_sizes"),
    [
        (torch.float32, (1, 5)),
        (torch.bfloat16, (1, 257)),
        (torch.float16, (1, 2049)),
    ],
    ids=["float32", "bfloat16-rounding-boundary", "float16-rounding-boundary"],
)
@pytest.mark.skipif(not dynamic_export_is_available(), reason=DYNAMIC_EXPORT_UNAVAILABLE_REASON)
def test_normal_transform_export_crosses_singleton_boundary(is_3d, dtype, runtime_sizes):
    class ExportTransform(torch.nn.Module):
        def forward(self, image):
            if is_3d:
                return kornia.geometry.normal_transform_pixel3d(
                    image.shape[-3],
                    image.shape[-2],
                    image.shape[-1],
                    device=image.device,
                    dtype=image.dtype,
                )
            return kornia.geometry.normal_transform_pixel(
                image.shape[-2], image.shape[-1], device=image.device, dtype=image.dtype
            )

    image_shape = (1, 1, 2, 3, 4) if is_3d else (1, 1, 2, 4)
    example = torch.zeros(image_shape, dtype=dtype)
    exported = torch.export.export(
        ExportTransform(),
        (example,),
        dynamic_shapes=({2: torch.export.Dim("singleton_axis", min=1, max=max(runtime_sizes) + 1)},),
    ).module()

    for runtime_size in runtime_sizes:
        runtime = torch.zeros(*image_shape[:2], runtime_size, *image_shape[3:], dtype=dtype)
        assert_close(exported(runtime), ExportTransform()(runtime), atol=0.0, rtol=0.0)


class TestNormalTransformPixel(BaseTester):
    # normal_transform_pixel and normal_transform_pixel3d have no test class of their own in this
    # file -- their existing coverage lives in tests/geometry/transform/test_homography_warper.py.
    # The convention pins live here, next to the pixel-coordinate family whose [-1, 1] convention
    # they share (an executed agreement, not a shared-code argument: see
    # test_convention_agrees_with_normalize_pixel_coordinates below).
    # NOTE: kornia#3904 (reserved) may extend this surface. EVERY literal in this class -- not only
    # the pins that repeat this line -- is built from the unconditional corner-aligned 2/(size - 1)
    # constants, so all of them would flip if #3904 made the normalization respect align_corners.
    # They record current default behavior; none of them is a ratified contract for that choice.

    def test_convention_returns_one_unbatched_matrix_in_the_ambient_default_dtype(self, device):
        # Convention pin: both helpers return exactly one matrix behind a leading axis of 1 --
        # (1, 3, 3) and (1, 4, 4) -- for every size; there is no batched form, and the sizes are
        # Python ints rather than tensors. With dtype=None the matrix is built by torch.tensor()
        # from Python floats, so its dtype is torch's AMBIENT default rather than float32
        # unconditionally: changing the process default changes the result. That is the mechanism
        # behind the float32 constants that leak into float64 homography pipelines (kornia#3958,
        # pinned in TestNormalizeHomography): normalize_homography calls these helpers without
        # passing dtype= through, so they materialise at the ambient default and are cast after.
        # The dtype fixture is dropped because the claim is about the *absence* of a dtype
        # argument, and the default is read back through torch.get_default_dtype() rather than
        # hardcoded, so the pin says "follows the ambient default" and not "is always float32".
        # The ambient-default leg runs on cpu whatever the device fixture says: the claim is about
        # which dtype is selected, not about placement, and MPS cannot represent float64 at all.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   normal_transform_pixel(4, 5).shape, .dtype      -> (1, 3, 3), torch.float32
        #   normal_transform_pixel3d(3, 5, 9).shape, .dtype -> (1, 4, 4), torch.float32
        #   torch.set_default_dtype(torch.float64)
        #   normal_transform_pixel(4, 5).dtype              -> torch.float64
        normal_transform_pixel = kornia.geometry.conversions.normal_transform_pixel
        normal_transform_pixel3d = kornia.geometry.conversions.normal_transform_pixel3d

        assert normal_transform_pixel(4, 5, device=device).shape == (1, 3, 3)
        assert normal_transform_pixel3d(3, 5, 9, device=device).shape == (1, 4, 4)
        assert normal_transform_pixel(4, 5, device=device).dtype == torch.get_default_dtype()
        assert normal_transform_pixel3d(3, 5, 9, device=device).dtype == torch.get_default_dtype()
        assert normal_transform_pixel(4, 5, device=device, dtype=torch.float16).dtype == torch.float16

        with _ambient_default_dtype(torch.float64):
            ambient_dtype = normal_transform_pixel(4, 5).dtype

        assert ambient_dtype == torch.float64, (
            "normal_transform_pixel no longer follows the ambient default dtype, so the kornia#3958 "
            "mechanism pinned in TestNormalizeHomography has changed"
        )

    def test_convention_corner_aligned_scale_is_two_over_size_minus_one(self, device):
        # Convention pin: the matrix is diag(2/(width - 1), 2/(height - 1), 1) with a -1 offset in
        # the last column, so pixel CENTRES 0 and size - 1 land exactly on -1 and +1. That is the
        # corner-aligned (align_corners=True) convention and it is applied unconditionally -- there
        # is no align_corners parameter, so the half-pixel convention cannot be selected:
        # grid_sample's default (align_corners=False) mapping (2*x + 1)/width - 1 would send the
        # same three columns to -0.8, 0.8, 0.0 instead of -1.0, 1.0, 0.0.
        # NOTE: covered by this class's kornia#3904 note above; recorded, not a ratified contract.
        # Sizes 3 and 5 are chosen because 2/(3 - 1) = 1.0 and 2/(5 - 1) = 0.5 are exact, so the
        # comparison runs at atol=rtol=0 and no nearby convention can satisfy it. float32 is
        # hardcoded and the dtype fixture dropped: the scales are pure Python floats computed
        # before any tensor exists (the wart matrix below documents the same mechanism), so a
        # non-float32 leg would only test torch's cast. Only the matrix is asserted: the pixel
        # maps in the snippet are pure arithmetic on the bitwise-pinned matrix and this test's own
        # constants, so a product assertion would add no failure surface against kornia.
        # Snippet used to generate expected (stdlib only, height = 3, width = 5):
        #   2 / (5 - 1), 2 / (3 - 1)               -> 0.5, 1.0
        #   [2 * x / (5 - 1) - 1 for x in (0, 4, 2)] -> [-1.0, 1.0, 0.0]
        #   [(2 * x + 1) / 5 - 1 for x in (0, 4, 2)] -> [-0.8, 0.8, 0.0]  (the half-pixel alternative)
        matrix = kornia.geometry.conversions.normal_transform_pixel(3, 5, device=device, dtype=torch.float32)

        expected = torch.tensor(
            [[[0.5, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32
        )
        self.assert_close(matrix, expected, atol=0.0, rtol=0.0)

    def test_convention_agrees_with_normalize_pixel_coordinates(self, device, dtype):
        # Convention pin: the matrix is the homogeneous form of normalize_pixel_coordinates -- the
        # same map, not merely a similar one -- and the agreement is EXACT only where the
        # arithmetic is. Two legs, and the dtype fixture is load-bearing across both.
        # Leg 1, sizes (3, 5): both scales (1.0 and 0.5) are exactly representable, so every route
        # and every dtype hits the same hardcoded literal at atol=rtol=0, on five points including
        # a half-pixel and an out-of-image one (neither route clamps).
        # Leg 2, sizes (2, 28): 2/(28 - 1) is not exactly representable in reduced precision, so
        # the two routes do diverge there and the dtype fixture is load-bearing -- each dtype gets
        # a different bound and a different measured gap, unlike the sizes in leg 1, where nothing
        # can diverge in any dtype. Note the two routes hold the SAME rounded scale; what differs
        # is where the rounding falls, so the divergence is not a property of the scale literal.
        # The helper multiplies and
        # subtracts elementwise in the working dtype, while applying the matrix is a matmul, whose
        # dot product is accumulated at higher precision and rounded once at the end.
        # What leg 2 asserts here is the dtype-scaled half of the docstring's claim -- the one that
        # runs everywhere rather than only where a kernel was measured: the disagreement stays
        # within 2 * the eps of the format the matmul rounds its INPUTS to -- finfo(dtype).eps on
        # every backend that evaluates the matmul at the working dtype, and the coarser format's eps
        # where it does not (TF32 on cuda; see _matmul_input_eps). The docstring's bound is scoped
        # the same way. That is a TOLERATED bound, not one derived from a portable
        # accuracy model (see the note at the assertion for what it does and does not promise),
        # computed from the dtype rather than hardcoded. The exact per-configuration
        # gaps are a kernel measurement and live in
        # test_wart_agreement_gap_at_2_28_is_a_kernel_measurement below, which is where the
        # docstring's figures are pinned and where an unmeasured configuration skips visibly.
        # What this pin does NOT decide: anything about non-corner-aligned callers such as
        # grid_sample(align_corners=False), which is pinned separately in
        # TestNormalizePixelCoordinates.
        # NOTE: covered by this class's kornia#3904 note above; recorded, not a ratified contract.
        # Snippet used to generate expected (leg 1, stdlib only, height = 3, width = 5):
        #   x -> 2 * x / 4 - 1 for x in (0, 4, 2, 1, 6) -> -1.0, 1.0, 0.0, -0.5, 2.0
        #   y -> 2 * y / 2 - 1 for y in (0, 2, 1, 0.5, 0) -> -1.0, 1.0, 0.0, -0.5, -1.0
        _skip_if_dtype_unavailable(device, dtype)
        pixels = torch.tensor(
            [[[0.0, 0.0], [4.0, 2.0], [2.0, 1.0], [1.0, 0.5], [6.0, 0.0]]], device=device, dtype=dtype
        )
        expected = torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-0.5, -0.5], [2.0, -1.0]], device=device, dtype=dtype
        )

        via_helper = kornia.geometry.conversions.normalize_pixel_coordinates(pixels, 3, 5)[0]
        matrix = kornia.geometry.conversions.normal_transform_pixel(3, 5, device=device, dtype=dtype)
        homogeneous = torch.cat([pixels[0], torch.ones_like(pixels[0][:, :1])], dim=-1)
        via_matrix = (matrix[0] @ homogeneous.transpose(0, 1)).transpose(0, 1)[:, :2]

        self.assert_close(via_helper, expected, atol=0.0, rtol=0.0)
        self.assert_close(via_matrix, expected, atol=0.0, rtol=0.0)

        largest_gap = _agreement_gap_at_2_28(device, dtype)
        # A tolerated bound, not a derived guarantee, and deliberately not called one rounding
        # step: eps IS the spacing at 1.0, so 2 * eps accepts two spacings there and four just
        # below it. What the constant is: roughly 2x headroom over the worst gap measured in any
        # configuration below, all of which are at most one eps (float16 is exactly eps on both
        # backends; every other nonzero cell is eps/2), scaled so no dtype inherits a bound sized
        # for another. Scaled by _matmul_input_eps rather than by finfo(dtype).eps directly,
        # because the matrix route is a matmul and a backend can be configured to round a matmul's
        # inputs below the working dtype (TF32 or bfloat16 for float32 on cuda) -- there the
        # coarser format IS the arithmetic, so the bound follows it instead of the pin going red on
        # a configuration change that touched no kornia code.
        # A failure is still not by itself a kornia regression: torch promises no common
        # accumulation precision across every backend, so it says re-derive the bound against the
        # configuration that produced it -- and record the measurement in the wart pin below,
        # keyed like the rest.
        tolerated_gap = 2 * _matmul_input_eps(device, dtype)

        assert largest_gap <= tolerated_gap, (
            f"the two routes now differ by {largest_gap!r} at (2, 28) in {dtype}, more than the "
            f"tolerated {tolerated_gap!r} (2 * the matmul's input eps) -- either they no longer differ only in where "
            "the rounding falls, or this configuration accumulates matmuls at a precision none of "
            "the measured ones used and needs a bound derived for it"
        )

    def test_wart_agreement_gap_at_2_28_is_a_kernel_measurement(self, device, dtype):
        # Wart pin for the exact figures normal_transform_pixel's agreement bullet quotes: at
        # (2, 28), where 2/(28 - 1) is not representable in reduced precision, how far apart the
        # two routes land is a property of the backend's and the build's matmul kernel, not of the
        # convention. Pinned exactly rather than as a bound -- the dtype-scaled bound is asserted by
        # test_convention_agrees_with_normalize_pixel_coordinates above -- because a bound would
        # let cpu float32/float64 regress from exact agreement to a small nonzero gap while the
        # docstring's "0.0" claim quietly became false.
        # Keyed by (torch version, backend, machine, dtype): every cell is a measurement of one
        # build, not a portable contract, so an unmeasured torch release must not silently inherit
        # an older release's kernel literal -- a future kernel reassociation could then fail this
        # pin on a release that was never measured, without any kornia regression.
        #   - backend, because the mps matmul rounds once where cpu does not (float32: 2**-24 vs 0)
        #   - machine, because every figure below was taken on macOS arm64 and nothing here has
        #     run on x86-64/MKL; an unmeasured platform must not inherit a kernel literal.
        #     What that costs, stated rather than left to be discovered: of the runners
        #     pr_test_cpu.yml uses, only macos-latest is arm64, so the ubuntu and windows legs
        #     skip every cell here, and no CI job sets KORNIA_TEST_DTYPE to float16/bfloat16, so
        #     the reduced-precision cells -- the only ones with a nonzero literal on cpu -- run
        #     on none of them. The x86-64 rows need one measurement on such a runner to become
        #     live; deriving them from the arm64 figures instead is exactly what this key exists
        #     to prevent. The portable half of the docstring's claim is asserted on every CI leg
        #     by test_convention_agrees_with_normalize_pixel_coordinates above.
        #   A cuda row, when one is measured, will need the float32 matmul precision mode in its
        #   key as well: --tf32 rounds a matmul's inputs to 10 mantissa bits and the matrix route
        #   is a matmul, while cpu is unaffected by that setting (executed, all four dtypes).
        #   - version, because the cpu bfloat16 kernel changed between the two torch versions
        #     executed (no divergence at all on 2.5.1, one bfloat16 step on 2.9.1) while every other
        #     cell reproduced identically on both -- reproducing does not exempt a cell from being
        #     keyed by version, since a later release could still change it
        # An unmeasured configuration skips visibly: the skip is a reminder to measure, not a
        # silent pass, and the dtype-scaled bound is still asserted by the pin above.
        # NOT a contract that these kernels must keep producing these numbers -- if a cell fails,
        # re-measure, update the cell here (this comment and the table below are where the figures
        # live; normal_transform_pixel's bullet states the contract and quotes none of them) and
        # check that the bullet's "agree in float32/float64; whether they agree at float16/bfloat16
        # is a property of the build" still holds -- the cpu bfloat16 row is the whole reason that
        # clause is build-scoped rather than absolute: 2.5.1 agrees at every size, 2.9.1 does not.
        # Snippet used to generate expected (torch + kornia, executed on macOS arm64 against both
        # torch 2.9.1 and torch 2.5.1; max|helper - matrix| over the full (2, 28) pixel grid):
        #   cpu 2.9.1: float64 -> 0.0   float32 -> 0.0   float16 -> 0.0009765625  bfloat16 -> 0.00390625
        #   cpu 2.5.1: float64 -> 0.0   float32 -> 0.0   float16 -> 0.0009765625  bfloat16 -> 0.0
        #   mps, both: float32 -> 5.960464477539063e-08 (2**-24)   float16 -> 0.0009765625
        #              bfloat16 -> 0.00390625            (float64 is unavailable on mps)
        # The full size sweep behind the docstring's "not at float16/bfloat16" clause is NOT
        # pinned -- 3364 size pairs per dtype is too slow for a unit test -- so its counts live in
        # this comment and nowhere else; the same snippet with the (2, 28) call in a double loop
        # over range(2, 60) reproduces them:
        #   cpu 2.9.1: float16 3328/3364 (worst 9.77e-04)   bfloat16 3315/3364 (worst 7.81e-03)
        #   cpu 2.5.1: float16 3328/3364 (worst 9.77e-04)   bfloat16    0/3364 (worst 0.0)
        _skip_if_dtype_unavailable(device, dtype)
        measured_gaps = {
            ("2.9.1", "cpu", "arm64", torch.float64): 0.0,
            ("2.5.1", "cpu", "arm64", torch.float64): 0.0,
            ("2.9.1", "cpu", "arm64", torch.float32): 0.0,
            ("2.5.1", "cpu", "arm64", torch.float32): 0.0,
            ("2.9.1", "cpu", "arm64", torch.float16): 0.0009765625,
            ("2.5.1", "cpu", "arm64", torch.float16): 0.0009765625,
            ("2.9.1", "cpu", "arm64", torch.bfloat16): 0.00390625,
            ("2.5.1", "cpu", "arm64", torch.bfloat16): 0.0,
            ("2.9.1", "mps", "arm64", torch.float32): 2.0**-24,
            ("2.5.1", "mps", "arm64", torch.float32): 2.0**-24,
            ("2.9.1", "mps", "arm64", torch.float16): 0.0009765625,
            ("2.5.1", "mps", "arm64", torch.float16): 0.0009765625,
            ("2.9.1", "mps", "arm64", torch.bfloat16): 0.00390625,
            ("2.5.1", "mps", "arm64", torch.bfloat16): 0.00390625,
        }
        machine = platform.machine()
        key = (torch_version(), device.type, machine, dtype)

        if key not in measured_gaps:
            pytest.skip(
                f"the (2, 28) agreement gap was not measured on {device.type} / {machine} / torch "
                f"{torch_version()} at {dtype}; no kernel literal to inherit"
            )
        expected_gap = measured_gaps[key]

        largest_gap = _agreement_gap_at_2_28(device, dtype)

        assert largest_gap == expected_gap, (
            f"the two routes now differ by {largest_gap!r} at (2, 28) on {device.type} / {machine} "
            f"in {dtype} under torch {torch_version()}, not {expected_gap!r} -- "
            "normal_transform_pixel's agreement bullet quotes this figure and must be re-measured"
        )

    def test_convention_3d_matrix_acts_on_x_y_z_one(self, device):
        # Convention pin: normal_transform_pixel3d(depth, height, width) returns a 4x4 whose
        # diagonal is (2/(width - 1), 2/(height - 1), 2/(depth - 1), 1) -- the matrix acts on
        # homogeneous (x, y, z, 1) with x scaled by WIDTH, y by HEIGHT and z by DEPTH, i.e. the
        # reverse of its own argument order -- and is corner-aligned like the 2-D form, so
        # (0, 0, 0) maps to (-1, -1, -1) and (width - 1, height - 1, depth - 1) to (1, 1, 1).
        # NOTE: covered by this class's kornia#3904 note above; recorded, not a ratified contract.
        # (depth, height, width) = (3, 5, 9) keeps 2/8, 2/4 and 2/2 exact, so the comparison runs
        # at atol=rtol=0. float32 is hardcoded and the dtype fixture dropped -- NOT for the 2-D
        # pin's pure-Python-floats reason: normal_transform_pixel3d computes its scales as tensor
        # arithmetic in the requested dtype (tr_mat[i, i] * 2.0 / denominator). These dyadic
        # scales are exact in every dtype, and the 3-D helper's dtype-dependent arithmetic stays
        # exercised by the dtype-parameterized component-order pin below. Only the matrix is
        # asserted: the corner maps in the
        # snippet are pure arithmetic on the bitwise-pinned matrix and this test's own constants.
        # Snippet used to generate expected (stdlib only):
        #   2 / (9 - 1), 2 / (5 - 1), 2 / (3 - 1) -> 0.25, 0.5, 1.0
        #   matrix @ (0, 0, 0, 1) -> (-1, -1, -1, 1);  matrix @ (8, 4, 2, 1) -> (1, 1, 1, 1)
        matrix = kornia.geometry.conversions.normal_transform_pixel3d(3, 5, 9, device=device, dtype=torch.float32)

        expected = torch.tensor(
            [[[0.25, 0.0, 0.0, -1.0], [0.0, 0.5, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=torch.float32,
        )
        self.assert_close(matrix, expected, atol=0.0, rtol=0.0)

    def test_convention_3d_component_order_permutes_normalize_pixel_coordinates3d(self, device, dtype):
        # Convention pin: normal_transform_pixel3d and normalize_pixel_coordinates3d order their
        # components differently, so a 3-D grid built for one is silently permuted by the other.
        # The matrix consumes (x, y, z); the helper consumes (d, x, y) (pinned in
        # TestNormalizePixelCoordinates.test_convention_3d_component_order_is_depth_x_y). The same
        # voxel therefore produces the same three numbers in two different slot orders --
        # helper[(1, 2, 0)] == matrix route -- which is what this pin asserts, at atol=rtol=0.
        # (depth, height, width) = (9, 5, 3) keeps every scale exact in every dtype.
        # Snippet used to generate expected (stdlib only, depth = 9, height = 5, width = 3):
        #   helper (d, x, y) = (7, 2, 1) -> 2*7/8 - 1, 2*2/2 - 1, 2*1/4 - 1 -> [0.75, 1.0, -0.5]
        #   matrix (x, y, z) = (2, 1, 7) -> 2*2/2 - 1, 2*1/4 - 1, 2*7/8 - 1 -> [1.0, -0.5, 0.75]
        _skip_if_dtype_unavailable(device, dtype)
        voxel_for_helper = torch.tensor([[[7.0, 2.0, 1.0]]], device=device, dtype=dtype)
        voxel_for_matrix = torch.tensor([[2.0], [1.0], [7.0], [1.0]], device=device, dtype=dtype)

        via_helper = kornia.geometry.conversions.normalize_pixel_coordinates3d(voxel_for_helper, 9, 5, 3)[0, 0]
        matrix = kornia.geometry.conversions.normal_transform_pixel3d(9, 5, 3, device=device, dtype=dtype)
        via_matrix = (matrix[0] @ voxel_for_matrix)[:3, 0]

        self.assert_close(via_helper, torch.tensor([0.75, 1.0, -0.5], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(via_matrix, torch.tensor([1.0, -0.5, 0.75], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(via_helper[[1, 2, 0]], via_matrix, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(
        ("ndim", "arg_name", "diagonal"),
        [
            ("2d", "height", [0.5, None]),
            ("2d", "width", [None, 1.0]),
            ("3d", "depth", [0.25, 0.5, None]),
            ("3d", "height", [0.25, None, 1.0]),
            ("3d", "width", [None, 0.5, 1.0]),
        ],
        ids=["2d-height", "2d-width", "3d-depth", "3d-height", "3d-width"],
    )
    def test_singleton_axis_maps_to_center(self, ndim, arg_name, diagonal, device):
        expected = list(diagonal)
        axis = diagonal.index(None)
        expected[axis] = 1.0

        if ndim == "2d":
            sizes = {"height": 3, "width": 5}
            sizes[arg_name] = 1
            matrix = kornia.geometry.conversions.normal_transform_pixel(
                sizes["height"], sizes["width"], device=device, dtype=torch.float32
            )
        else:
            sizes = {"depth": 3, "height": 5, "width": 9}
            sizes[arg_name] = 1
            matrix = kornia.geometry.conversions.normal_transform_pixel3d(
                sizes["depth"], sizes["height"], sizes["width"], device=device, dtype=torch.float32
            )

        scales = torch.stack([matrix[0, i, i] for i in range(len(expected))])

        self.assert_close(scales, torch.tensor(expected, device=device, dtype=torch.float32), atol=0.0, rtol=0.0)
        self.assert_close(matrix[0, axis, -1], torch.tensor(0.0, device=device), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(
        ("op_name", "sizes"), [("normal_transform_pixel", (3, 1)), ("normal_transform_pixel3d", (3, 1, 5))]
    )
    def test_non_default_eps_warns_and_is_ignored(self, op_name, sizes, device):
        op = getattr(kornia.geometry.conversions, op_name)
        default = op(*sizes, device=device, dtype=torch.float32)
        with pytest.warns(FutureWarning, match="deprecated and ignored"):
            overridden = op(*sizes, eps=1.0, device=device, dtype=torch.float32)
        self.assert_close(default, overridden, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("invalid_size", [0, -3], ids=["zero", "negative"])
    @pytest.mark.parametrize(
        ("ndim", "arg_name"),
        [("2d", "height"), ("2d", "width"), ("3d", "depth"), ("3d", "height"), ("3d", "width")],
    )
    def test_non_positive_size_raises(self, ndim, arg_name, invalid_size, device):
        if ndim == "2d":
            sizes = {"height": 3, "width": 5}
            sizes[arg_name] = invalid_size
            with pytest.raises(ValueError, match="Input image size must be positive"):
                kornia.geometry.conversions.normal_transform_pixel(sizes["height"], sizes["width"], device=device)
        else:
            sizes = {"depth": 3, "height": 5, "width": 9}
            sizes[arg_name] = invalid_size
            with pytest.raises(ValueError, match="Input image size must be positive"):
                kornia.geometry.conversions.normal_transform_pixel3d(
                    sizes["depth"], sizes["height"], sizes["width"], device=device
                )

    @pytest.mark.parametrize("ndim", ["2d", "3d"])
    @pytest.mark.parametrize(
        ("default_dtype", "size"), [(torch.float16, 2049), (torch.bfloat16, 257)], ids=["float16", "bfloat16"]
    )
    def test_half_default_dtype_does_not_round_the_size_under_tracing(self, ndim, default_dtype, size, device):
        # The graph-capture branch keeps the size arithmetic in at least float32 because a half
        # type cannot hold every practical image size. That promotion has to key off the dtype the
        # matrix is actually built in, not off the ``dtype`` ARGUMENT: with ``dtype=None`` the
        # matrix inherits torch.get_default_dtype(), which may itself be a half type, and reading
        # the argument alone leaves the size rounded in exactly the case the promotion exists for.
        # The coordinate helpers resolve the output dtype first and were always right here; these
        # two did not, so nothing else in this file covers the ``dtype=None`` + half-default cell.
        # Sizes are picked so the size ITSELF is unrepresentable in the default dtype AND the
        # rounding survives into the scale: float16 holds 2049 only as 2048, bfloat16 holds 257
        # only as 256. Merely-unrepresentable is not enough -- at float16 and size 3001 the two
        # paths compute 2/3000 and 2/2999, which round to the SAME float16, so that cell would
        # pass whether or not the promotion happens. 2049 and 257 are the smallest sizes at which
        # each dtype actually disagrees under torch.jit.trace.
        class NormalTransformPixel(torch.nn.Module):
            def forward(self, image):
                if ndim == "3d":
                    return kornia.geometry.conversions.normal_transform_pixel3d(
                        image.shape[-3], image.shape[-2], image.shape[-1], device=image.device
                    )
                return kornia.geometry.conversions.normal_transform_pixel(
                    image.shape[-2], image.shape[-1], device=image.device
                )

        previous = torch.get_default_dtype()
        torch.set_default_dtype(default_dtype)
        try:
            shape = (1, 1, 3, size, 5) if ndim == "3d" else (1, 1, size, 5)
            image = torch.zeros(*shape, device=device)
            traced = torch.jit.trace(NormalTransformPixel(), image)
            self.assert_close(traced(image), NormalTransformPixel()(image), atol=0.0, rtol=0.0)
        finally:
            torch.set_default_dtype(previous)

    def test_wart_integer_dtype_truncates_the_scale_to_zero_3959(self, device):
        # Wart pin for kornia#3959: the matrix is built by torch.tensor([...], dtype=dtype) from
        # Python floats, so an integer dtype truncates every scale below 1 to 0 and the function
        # returns a rank-deficient matrix -- no error, no warning. The 2-D result maps EVERY pixel
        # to the constant (-1, -1) -- recorded in the snippet as pure arithmetic on the
        # bitwise-pinned integer matrix, so it is not asserted separately (and no integer matmul
        # runs, which CUDA would not implement). The 3-D cell keeps depth = 2 so that its z scale, 2/1 = 2,
        # survives the truncation: a partial fix that only promotes float scales would still have
        # to flip the two zeroed axes.
        # There is deliberately NO companion strict xfail: the intended behavior is undecided
        # (promote to float, or raise), and an assertion-shaped xfail can express only the first.
        # kornia#3948, the same defect in axis_angle_to_quaternion, was settled by PROMOTING --
        # see TestAngleAxisToQuaternion.test_convention_integer_input_is_promoted_to_float_3948 --
        # so that is the precedent a fix here should follow, but it is precedent and not a decision
        # taken for this function, which is why the pin stays a wart rather than becoming an xfail.
        # If either cell fails, #3959 was (partly) fixed -- remove this pin. NOT a contract that an
        # integer dtype must keep returning a degenerate matrix.
        # The dtype fixture is dropped because the claim is about the dtype argument itself.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   normal_transform_pixel(4, 5, dtype=torch.int64)
        #     -> [[0, 0, -1], [0, 0, -1], [0, 0, 1]]
        #   normal_transform_pixel3d(2, 4, 5, dtype=torch.int64)
        #     -> diag [0, 0, 2]   (2/(5-1) and 2/(4-1) truncate, 2/(2-1) does not)
        #   matrix @ (x, y, 1) -> (-1, -1, 1) for every pixel -- e.g. (0, 0), (4, 3), (2, 1)
        matrix = kornia.geometry.conversions.normal_transform_pixel(4, 5, device=device, dtype=torch.int64)
        matrix3d = kornia.geometry.conversions.normal_transform_pixel3d(2, 4, 5, device=device, dtype=torch.int64)

        assert matrix[0].tolist() == [[0, 0, -1], [0, 0, -1], [0, 0, 1]], (
            "kornia#3959: an integer dtype no longer truncates the 2-D normalization scales to 0"
        )
        assert [matrix3d[0, i, i].item() for i in range(3)] == [0, 0, 2], (
            "kornia#3959: an integer dtype no longer truncates the 3-D normalization scales to 0"
        )


class TestNormalizeHomography(BaseTester):
    # normalize_homography, denormalize_homography and normalize_homography3d have no test class of
    # their own in this file -- their existing coverage lives in
    # tests/geometry/transform/test_homography_warper.py. The convention pins live here, next to
    # normal_transform_pixel, whose corner-aligned convention all three inherit.
    # The CONVENTION pins below (composition, direction, round-trip, batching, 3-D) use sizes of
    # the form 2**k + 1 (3, 5, 9, 17) so that every 2/(size - 1) is exact in every dtype and those
    # pins compare at atol=rtol=0. That also keeps them independent of kornia#3958 (the float32
    # constants leak, pinned separately below with non-dyadic sizes): a fix for #3958 must not flip
    # an ordering or direction pin. The exactness invariant is theirs alone -- the bug pins below
    # deliberately step outside it (the round-trip pin's non-dyadic (4, 5)/(8, 9) legs at
    # atol=32*eps, the #3960 shape-guard cells), so a new atol=0 pin belongs here
    # only at these sizes AND with a literal whose intermediates are exact. The invariant also
    # leans on the SHAPE of the normalization matrices -- upper-triangular with power-of-two
    # pivots -- surviving BOTH inverse routes actually in play (the functions do NOT share one):
    # normalize_homography inverts through _inverse_3x3_closed_form (cofactor arithmetic --
    # products and sums of dyadic values, then division by a power-of-two determinant, all
    # exact), while denormalize_homography and normalize_homography3d go through
    # _torch_inverse_cast (torch.linalg.inv in eager mode, rounding-free on such triangular
    # matrices, with a float32 upcast for half dtypes and a closed-form 3x3 fallback under
    # tracing -- each exactness-preserving on these values). A future non-triangular
    # normalization (a #3904 align_corners variant, say) voids the atol=0 claim on EVERY route
    # and needs a tolerance instead.
    # No pin asserts anything about kornia#3962 (no denormalize_homography3d, no
    # ColmapQTVecs_to_ARKitQTVecs) -- a missing symbol is a scope question, not a defect.
    # NOTE: kornia#3904 (reserved) may extend this surface. EVERY literal in this class -- the
    # composition, direction, round-trip, batching and 3-D pins as much as the #3957 singleton and
    # #3958 bug pins, and whether or not the pin repeats this line -- is built from the corner-aligned
    # 2/(size - 1) constants these three functions inherit from normal_transform_pixel, so a #3904
    # fix that made the normalization respect align_corners would flip all of them. They record
    # current default behavior; none of them ratifies that choice as contract. (The #3958 pins would
    # also flip on a #3958 fix, which is their point; the #3904 exposure is separate and additional.)

    def test_gradcheck(self, device):
        # The three functions are on the warp_perspective path and are differentiable in their
        # homography argument, and nothing in the suite checked that: neither this file nor
        # test_homography_warper.py had a gradcheck for any of them before this pin
        # (`grep -rn "normalize_homography" tests/ | grep gradcheck` was empty). Each is linear in
        # the homography -- a fixed matrix on each side -- so the gradient is the composition of
        # the two normalization matrices and any evaluation point checks the same thing; a
        # non-identity one is used anyway so that a route which silently ignored its input would
        # not still agree numerically. The dsize arguments are Python ints, so they are bound
        # through partial rather than passed as gradcheck inputs.
        # normal_transform_pixel and normal_transform_pixel3d get no gradcheck on purpose: they
        # take ints and return a constant matrix, so there is no input to differentiate.
        dtype = torch.float64
        homography = torch.tensor([[[1.0, 0.2, 3.0], [0.1, 1.0, -2.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        homography3d = torch.eye(4, device=device, dtype=dtype)[None] + 0.1

        self.gradcheck(
            partial(kornia.geometry.conversions.normalize_homography, dsize_src=(4, 5), dsize_dst=(8, 9)),
            (homography,),
        )
        self.gradcheck(
            partial(kornia.geometry.conversions.denormalize_homography, dsize_src=(4, 5), dsize_dst=(8, 9)),
            (homography,),
        )
        self.gradcheck(
            partial(kornia.geometry.conversions.normalize_homography3d, dsize_src=(2, 4, 5), dsize_dst=(3, 8, 9)),
            (homography3d,),
        )

    def test_convention_maps_normalized_src_to_normalized_dst(self, device, dtype):
        # Convention pin: the returned matrix has the SAME src -> dst direction as its input,
        # re-expressed in the two [-1, 1] frames -- it maps normalized src coordinates to
        # normalized dst coordinates, and it is size-aware: a +2 px shift in a 5-wide image becomes
        # a +1.0 shift in normalized units (2 px * 2/(5 - 1)).
        # Only the matrix is asserted: the pixel-level reading in the snippet -- src (1, 1)
        # normalized, pushed through the result, landing on the normalization of the dst pixel
        # (3, 1) -- is pure arithmetic on the bitwise-pinned matrix and this test's constants.
        # NOTE: covered by this class's kornia#3904 note above; recorded, not a ratified contract.
        # Snippet used to generate expected (stdlib only, height = 3, width = 5 on both sides):
        #   2 / (5 - 1) = 0.5, so the +2 px translation becomes 2 * 0.5 = 1.0
        #   src (1, 1) -> (2*1/4 - 1, 2*1/2 - 1) = (-0.5, 0.0)
        #   dst (3, 1) -> (2*3/4 - 1, 2*1/2 - 1) = ( 0.5, 0.0)
        _skip_if_dtype_unavailable(device, dtype)
        _skip_if_closed_form_inverse_unavailable(device, dtype)
        translate_two_px = torch.tensor(
            [[[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        normalized = kornia.geometry.conversions.normalize_homography(translate_two_px, (3, 5), (3, 5))

        expected = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        self.assert_close(normalized, expected, atol=0.0, rtol=0.0)

    def test_wart_identity_at_equal_sizes_is_exact_at_some_sizes_and_not_others(self, device):
        # Wart pin for the figures normalize_homography's kornia#3904 warning quotes. That warning
        # exists to say this function is NOT the cause of warp_perspective's 11.25 deviation under
        # align_corners=False, and it backs that up with how little this function itself deviates on
        # the identity at equal sizes. Both halves of that figure are pinned here, because the
        # tempting reading of it -- "exact where the 2/(size - 1) scale is dyadic" -- is FALSE:
        # measured over equal sizes 2..32 in float32 the deviation is 0 at 2, 3, 5, 8, 9, 10, 12,
        # 15..20, 22, 23, 24, 26, 28..32 and 5.96e-08 at 4, 6, 7, 11, 13, 14, 21, 25, 27, and 8
        # (scale 2/7) is exact while 4 (scale 2/3) is not. Which size lands where is a property of
        # the inverse-and-matmul chain, which is what the warning now says instead of a rule.
        # Two sizes are enough to hold that: (4, 4), the size the warning quotes and the one the
        # warp_perspective comparison uses, and (3, 3), an exact one -- a change that made the chain
        # exact everywhere, or inexact everywhere, moves one of them.
        # float32 is hardcoded and the dtype fixture dropped: 5.96e-08 IS 2**-24, a float32 rounding
        # step, so the figure is only meaningful in float32 and every other dtype would need its own.
        # NOT a contract that these sizes must keep these residuals -- the class header's #3904 note
        # covers the whole surface, and a #3904 fix is expected to move both cells.
        # Snippet used to generate expected (torch only, executed on cpu, torch 2.9.1):
        #   I = torch.eye(3)[None]
        #   (normalize_homography(I, (n, n), (n, n)) - I).abs().max()  for n = 4 -> 5.960464477539063e-08
        #                                                              for n = 3 -> 0.0
        _skip_if_dtype_unavailable(device, torch.float32)
        _skip_if_closed_form_inverse_unavailable(device, torch.float32)
        identity = torch.eye(3, device=device, dtype=torch.float32)[None]

        inexact = kornia.geometry.conversions.normalize_homography(identity, (4, 4), (4, 4))
        exact = kornia.geometry.conversions.normalize_homography(identity, (3, 3), (3, 3))

        assert (inexact - identity).abs().max().item() == 2.0**-24, (
            "normalize_homography's #3904 warning quotes 5.96e-08 for the identity at equal sizes "
            f"(4, 4); got {(inexact - identity).abs().max().item()!r}"
        )
        self.assert_close(exact, identity, atol=0.0, rtol=0.0)

    def test_convention_dsize_src_is_the_right_factor_and_dsize_dst_the_left(self, device, dtype):
        # Convention pin: normalize_homography(H, dsize_src, dsize_dst) composes
        # N(dsize_dst) @ H @ N(dsize_src)^-1 -- the source size drives the RIGHT (input) factor and
        # the destination size the LEFT (output) factor. Both sizes are asymmetric here ((3, 5) and
        # (5, 9)) and the same call with the two size arguments exchanged is pinned to its own,
        # different literal: that second literal is what makes this a direction pin rather than a
        # value pin, since the reversed composition is exactly the mistake a caller makes.
        # Snippet used to generate expected (torch only, executed on cpu float64 and float32):
        #   H = [[2, 0.5, 2], [-0.25, 1, 1], [0, 0, 1]]
        #   normalize_homography(H, (3, 5), (5, 9))
        #     -> [[1.0, 0.125, 0.625], [-0.25, 0.5, -0.25], [0.0, 0.0, 1.0]]
        #   normalize_homography(H, (5, 9), (3, 5))
        #     -> [[4.0, 0.5, 4.5], [-1.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
        _skip_if_dtype_unavailable(device, dtype)
        _skip_if_closed_form_inverse_unavailable(device, dtype)
        homography = torch.tensor(_DIRECTION_H, device=device, dtype=dtype)

        normalized = kornia.geometry.conversions.normalize_homography(homography, (3, 5), (5, 9))
        swapped = kornia.geometry.conversions.normalize_homography(homography, (5, 9), (3, 5))

        self.assert_close(
            normalized,
            torch.tensor([[[1.0, 0.125, 0.625], [-0.25, 0.5, -0.25], [0.0, 0.0, 1.0]]], device=device, dtype=dtype),
            atol=0.0,
            rtol=0.0,
        )
        self.assert_close(
            swapped,
            torch.tensor([[[4.0, 0.5, 4.5], [-1.0, 2.0, 1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype),
            atol=0.0,
            rtol=0.0,
        )

    def test_convention_denormalize_is_the_mirror_composition(self, device, dtype):
        # Convention pin: denormalize_homography(H, dsize_src, dsize_dst) composes
        # N(dsize_dst)^-1 @ H @ N(dsize_src) -- the exact mirror of normalize_homography, with the
        # same argument roles (source size on the right, destination size on the left) and the two
        # normalization matrices exchanged. Pinned on the same asymmetric input and sizes as the
        # normalize pin above, so the two literals can be read side by side; they differ, which is
        # what rules out the two functions being the same map.
        # Snippet used to generate expected (torch only, executed on cpu float64 and float32):
        #   H = [[2, 0.5, 2], [-0.25, 1, 1], [0, 0, 1]]
        #   denormalize_homography(H, (3, 5), (5, 9))
        #     -> [[4.0, 2.0, 2.0], [-0.25, 2.0, 2.5], [0.0, 0.0, 1.0]]
        _skip_if_dtype_unavailable(device, dtype)
        homography = torch.tensor(_DIRECTION_H, device=device, dtype=dtype)

        denormalized = kornia.geometry.conversions.denormalize_homography(homography, (3, 5), (5, 9))

        self.assert_close(
            denormalized,
            torch.tensor([[[4.0, 2.0, 2.0], [-0.25, 2.0, 2.5], [0.0, 0.0, 1.0]]], device=device, dtype=dtype),
            atol=0.0,
            rtol=0.0,
        )

    def test_convention_normalize_and_denormalize_round_trip(self, device, dtype):
        # Convention pin: the two functions are mutual inverses, in BOTH compositions -- each is
        # executed on its own here rather than one being inferred from the other -- on a general
        # projective homography whose bottom row is non-zero.
        # Two legs. With dyadic sizes ((3, 5) and (5, 9)) every normalization constant is exact,
        # and THIS literal is chosen so that every intermediate product and sum is exactly
        # representable too (exact constants alone do not make the round trip bitwise -- that is a
        # property of the whole computation, not of the sizes or of the entries; the
        # triangular-inverse mechanism in the class header is part of it), so the round trip
        # returns the input bit for bit and that leg runs at atol=rtol=0. Cross-file:
        # TestHomographyWarper::test_consistency in
        # tests/geometry/transform/test_homography_warper.py (back-pointer there) exercises the
        # same high-level mutual-inverse invariant, on its own literal, sizes and tolerances. With
        # non-dyadic sizes ((4, 5) and (8, 9)) the constants are rounded and the round trip is only
        # approximate; its tolerance is sized from the mechanism rather than from a measurement --
        # each entry passes through four matrix products and TWO 3x3 inverses, one per function and
        # by different routines (normalize_homography inverts N_src with _inverse_3x3_closed_form,
        # denormalize_homography inverts N_dst with _torch_inverse_cast), so the error is a small
        # multiple of eps times the largest entry (max |H| = 4) -- giving atol = 32 * eps and
        # rtol = 8 * eps. Both compositions are asserted at those sizes, not just one: they do not
        # land on the same figure, and quoting one of them for the other is the mistake a
        # single-leg pin invites. Measured float32 deviations, sample points against a bound of
        # 3.8e-06: 1.19e-07 for denormalize(normalize(H)) and 2.38e-07 for the reverse.
        # Snippet used to generate expected (torch only, executed on cpu at every dtype):
        #   H = [[1.25, 0.25, 4], [-0.5, 0.75, 2], [0.0625, 0.125, 1]]
        #   denormalize_homography(normalize_homography(H, (3, 5), (5, 9)), (3, 5), (5, 9)) == H  (bitwise)
        #   normalize_homography(denormalize_homography(H, (3, 5), (5, 9)), (3, 5), (5, 9)) == H  (bitwise)
        _skip_if_dtype_unavailable(device, dtype)
        _skip_if_closed_form_inverse_unavailable(device, dtype)
        normalize_homography = kornia.geometry.conversions.normalize_homography
        denormalize_homography = kornia.geometry.conversions.denormalize_homography
        homography = torch.tensor(_ROUND_TRIP_H, device=device, dtype=dtype)

        denorm_of_norm = denormalize_homography(normalize_homography(homography, (3, 5), (5, 9)), (3, 5), (5, 9))
        norm_of_denorm = normalize_homography(denormalize_homography(homography, (3, 5), (5, 9)), (3, 5), (5, 9))

        self.assert_close(denorm_of_norm, homography, atol=0.0, rtol=0.0)
        self.assert_close(norm_of_denorm, homography, atol=0.0, rtol=0.0)

        eps = torch.finfo(dtype).eps
        non_dyadic = denormalize_homography(normalize_homography(homography, (4, 5), (8, 9)), (4, 5), (8, 9))
        non_dyadic_reverse = normalize_homography(denormalize_homography(homography, (4, 5), (8, 9)), (4, 5), (8, 9))

        self.assert_close(non_dyadic, homography, atol=32.0 * eps, rtol=8.0 * eps)
        self.assert_close(non_dyadic_reverse, homography, atol=32.0 * eps, rtol=8.0 * eps)

    # The two pins below are the negative half of the round-trip claim above, which the pin above
    # cannot express because it only ever runs the configuration that DOES come back bitwise:
    # denormalize_homography's bullet says dyadic sizes on their own are neither necessary nor
    # sufficient for a bitwise round trip, and each half of that needs a counterexample or it is an
    # unfalsifiable hedge. Both inputs are hardcoded rather than searched for, and both are
    # deterministic -- no rand, so neither pin can go quiet on a lucky seed. They are split into two
    # tests, and each hardcodes ONE dtype with the fixture dropped, because each half IS a statement
    # about one dtype: the same identity at (4, 5) -> (8, 9) in float32 misses through
    # denormalize(normalize(.)) and returns bitwise through the reverse, so a dtype-parameterized
    # form of either half would be false on the other dtype. Split rather than one test with two
    # dtypes so that on a backend without float64 (mps) the sufficiency half still reports as run.
    # NEITHER is a contract that these particular inputs must keep landing on these verdicts -- if a
    # cell flips, re-derive the bullet's "neither necessary nor sufficient" from whatever the new
    # counterexamples are.

    def test_convention_dyadic_sizes_are_not_sufficient_for_a_bitwise_round_trip(self, device):
        # The same dyadic sizes (3, 5) -> (5, 9) as the round-trip pin above, with entries that are
        # NOT dyadic, miss in float32 through BOTH legs. This is what makes that pin's "property of
        # the whole computation, not of the sizes" sentence load-bearing: nothing about the sizes
        # changed between this case and the bitwise one, only H.
        # Snippet used to generate expected (torch only, executed on cpu, torch 2.9.1):
        #   H = [[1.1, 0.2, 3], [-0.3, 0.9, 1], [0.05, 0.1, 1]] (float32)
        #   torch.equal(denormalize_homography(normalize_homography(H, (3,5), (5,9)), (3,5), (5,9)), H) -> False
        #   torch.equal(normalize_homography(denormalize_homography(H, (3,5), (5,9)), (3,5), (5,9)), H) -> False
        _skip_if_dtype_unavailable(device, torch.float32)
        _skip_if_closed_form_inverse_unavailable(device, torch.float32)
        normalize_homography = kornia.geometry.conversions.normalize_homography
        denormalize_homography = kornia.geometry.conversions.denormalize_homography
        not_dyadic_entried = torch.tensor(
            [[[1.1, 0.2, 3.0], [-0.3, 0.9, 1.0], [0.05, 0.1, 1.0]]], device=device, dtype=torch.float32
        )

        forward = denormalize_homography(normalize_homography(not_dyadic_entried, (3, 5), (5, 9)), (3, 5), (5, 9))
        reverse = normalize_homography(denormalize_homography(not_dyadic_entried, (3, 5), (5, 9)), (3, 5), (5, 9))

        assert not torch.equal(forward, not_dyadic_entried), (
            "dyadic sizes now make denormalize(normalize(H)) bitwise for a non-dyadic-entried H in "
            "float32, so denormalize_homography's 'not sufficient' half has lost its counterexample"
        )
        assert not torch.equal(reverse, not_dyadic_entried), (
            "dyadic sizes now make normalize(denormalize(H)) bitwise for a non-dyadic-entried H in "
            "float32, so denormalize_homography's 'not sufficient' half has lost its counterexample"
        )

    def test_convention_dyadic_sizes_are_not_necessary_for_a_bitwise_round_trip(self, device):
        # The float64 identity comes back bitwise through BOTH legs at the non-dyadic
        # (4, 5) -> (8, 9) -- the exact size pair where the round-trip pin above has to fall back to
        # a 32 * eps tolerance for a general H.
        # Snippet used to generate expected (torch only, executed on cpu, torch 2.9.1):
        #   I = torch.eye(3)[None] (float64), sizes (4, 5) -> (8, 9)
        #   torch.equal(denormalize_homography(normalize_homography(I, (4,5), (8,9)), (4,5), (8,9)), I) -> True
        #   torch.equal(normalize_homography(denormalize_homography(I, (4,5), (8,9)), (4,5), (8,9)), I) -> True
        _skip_if_dtype_unavailable(device, torch.float64)
        _skip_if_closed_form_inverse_unavailable(device, torch.float64)
        normalize_homography = kornia.geometry.conversions.normalize_homography
        denormalize_homography = kornia.geometry.conversions.denormalize_homography
        identity = torch.eye(3, device=device, dtype=torch.float64)[None]

        forward = denormalize_homography(normalize_homography(identity, (4, 5), (8, 9)), (4, 5), (8, 9))
        reverse = normalize_homography(denormalize_homography(identity, (4, 5), (8, 9)), (4, 5), (8, 9))

        assert torch.equal(forward, identity), (
            "the float64 identity no longer survives denormalize(normalize(.)) bitwise at the "
            "non-dyadic (4, 5) -> (8, 9), so denormalize_homography's 'not necessary' half has "
            "lost its counterexample"
        )
        assert torch.equal(reverse, identity), (
            "the float64 identity no longer survives normalize(denormalize(.)) bitwise at the "
            "non-dyadic (4, 5) -> (8, 9), so denormalize_homography's 'not necessary' half has "
            "lost its counterexample"
        )

    def test_convention_batch_is_per_sample(self, device, dtype):
        # Convention pin: both functions are per-sample -- the result for one batch element does not
        # depend on the others, bit for bit -- and the leading batch dimension is preserved. Pinned
        # on a batch of two different homographies, comparing element 1 of the batched call against
        # the single-element call.
        _skip_if_dtype_unavailable(device, dtype)
        _skip_if_closed_form_inverse_unavailable(device, dtype)
        normalize_homography = kornia.geometry.conversions.normalize_homography
        denormalize_homography = kornia.geometry.conversions.denormalize_homography
        batch = torch.tensor([_DIRECTION_H[0], _ROUND_TRIP_H[0]], device=device, dtype=dtype)

        normalized = normalize_homography(batch, (3, 5), (5, 9))
        denormalized = denormalize_homography(batch, (3, 5), (5, 9))

        assert normalized.shape == (2, 3, 3)
        self.assert_close(normalized[1], normalize_homography(batch[1:], (3, 5), (5, 9))[0], atol=0.0, rtol=0.0)
        self.assert_close(denormalized[1], denormalize_homography(batch[1:], (3, 5), (5, 9))[0], atol=0.0, rtol=0.0)

    def test_convention_3d_dsize_is_depth_height_width(self, device, dtype):
        # Convention pin: normalize_homography3d takes (depth, height, width) size tuples, composes
        # them the same way as the 2-D function (source size on the right, destination size on the
        # left) and returns a 4x4 acting on homogeneous (x, y, z, 1). Both the call and the call
        # with the two size arguments exchanged are pinned, so the pin is about direction and not
        # only about values; the sizes are asymmetric in all three axes ((3, 5, 9) and (5, 9, 17)),
        # so a (width, height, depth) reading of either tuple would give a different matrix.
        # Snippet used to generate expected (torch only, executed on cpu float64 and float32):
        #   H = [[2, 0.5, 0.25, 2], [-0.25, 1, 0.5, 1], [0.125, -0.5, 2, 3], [0, 0, 0, 1]]
        #   normalize_homography3d(H, (3, 5, 9), (5, 9, 17))
        #     -> [[1.0, 0.125, 0.03125, 0.40625], [-0.25, 0.5, 0.125, -0.375],
        #         [0.25, -0.5, 1.0, 1.25], [0.0, 0.0, 0.0, 1.0]]
        #   normalize_homography3d(H, (5, 9, 17), (3, 5, 9))
        #     -> [[4.0, 0.5, 0.125, 4.125], [-1.0, 2.0, 0.5, 1.0],
        #         [1.0, -2.0, 4.0, 5.0], [0.0, 0.0, 0.0, 1.0]]
        _skip_if_dtype_unavailable(device, dtype)
        homography = torch.tensor(
            [
                [
                    [2.0, 0.5, 0.25, 2.0],
                    [-0.25, 1.0, 0.5, 1.0],
                    [0.125, -0.5, 2.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        normalized = kornia.geometry.conversions.normalize_homography3d(homography, (3, 5, 9), (5, 9, 17))
        swapped = kornia.geometry.conversions.normalize_homography3d(homography, (5, 9, 17), (3, 5, 9))

        self.assert_close(
            normalized,
            torch.tensor(
                [
                    [
                        [1.0, 0.125, 0.03125, 0.40625],
                        [-0.25, 0.5, 0.125, -0.375],
                        [0.25, -0.5, 1.0, 1.25],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                device=device,
                dtype=dtype,
            ),
            atol=0.0,
            rtol=0.0,
        )
        self.assert_close(
            swapped,
            torch.tensor(
                [
                    [
                        [4.0, 0.5, 0.125, 4.125],
                        [-1.0, 2.0, 0.5, 1.0],
                        [1.0, -2.0, 4.0, 5.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                device=device,
                dtype=dtype,
            ),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="the normalization matrices are built at the ambient default dtype and cast to the "
        "input afterwards, so a float64 caller gets float32-rounded constants — kornia#3958",
        strict=True,
    )
    def test_convention_float64_input_gets_float64_normalization_constants_3958(self, device):
        # Intended behavior: a float64 homography is normalized with float64 constants, so the
        # entries carry float64 accuracy. They do not: normalize_homography calls
        # normal_transform_pixel() without passing dtype= through, so the constants materialise at
        # the ambient default (float32) and are cast to float64 afterwards, leaving about eight
        # significant digits. For sizes (4, 4) -> (6, 6) the (0, 0) entry is mathematically
        # 2/(6 - 1) * (4 - 1)/2 = 0.6 exactly, and any float64-native evaluation lands within an ulp
        # of it; the tolerance 1e-12 sits four orders above float64 noise and three below the
        # deviation the current implementation produces.
        # Non-dyadic sizes are required here: with 2**k + 1 sizes the float32 constants are exact
        # and there is nothing to leak, which is why the ordering pins above use them and this pin
        # does not. float64 is hardcoded and the dtype fixture dropped because the claim is a
        # float64 claim, and the skip is visible so that on MPS, which has no float64, a raw
        # TypeError cannot satisfy the raises=AssertionError mark instead of the assertion.
        # Marked xfail(strict=True) so fixing #3958 makes this XPASS and forces the mark out.
        # Companion wart: test_wart_float32_constants_leak_into_float64_results_3958.
        _skip_if_dtype_unavailable(device, torch.float64)

        identity = torch.eye(3, device=device, dtype=torch.float64)[None]

        normalized = kornia.geometry.conversions.normalize_homography(identity, (4, 4), (6, 6))

        assert abs(normalized[0, 0, 0].item() - 0.6) < 1e-12, (
            "kornia#3958: normalize_homography did not use float64 normalization constants"
        )

    def test_wart_float32_constants_leak_into_float64_results_3958(self, device):
        # Wart pin for kornia#3958, companion to the strict xfail above: assert the CURRENT
        # float32-rounded entries in a float64 result. Four cells:
        #   (1) normalize_homography, whose src factor is inverted by the closed-form 3x3 inverse;
        #   (2) denormalize_homography, whose dst factor is inverted by _torch_inverse_cast instead
        #       -- a separate code path that could be fixed on its own;
        #   (3) normalize_homography3d, which calls the 3-D helper and is a third call site;
        #   (4) the control that proves the cause is the missing dtype= pass-through and not an
        #       epsilon or a rounding choice: with the ambient default dtype set to float64 the
        #       same call returns the float64-native value 0.6000000000000001, because the helper
        #       now materialises in float64 before the cast. Cell (4) also fails if the helpers stop
        #       reading the ambient default, which is the other half of the same mechanism.
        # If any cell fails, #3958 was (partly) fixed -- flip/remove the strict xfail above. NOT a
        # contract that float64 callers must keep receiving float32-rounded constants.
        # atol 1e-10 pins the MAGNITUDE of the deviation, which is what the docstring warning
        # promises ("the magnitude -- half the mantissa gone -- is the point ... rather than the
        # digits"): it sits an order below the ~8.9e-09 deviation being discriminated (so a fix
        # still flips these cells red) and six above the ~1.1e-16 ulp of the entries, so no
        # backend's reassociation of the matmul-and-inverse chain can flip them. float64 is
        # hardcoded for the same reason as the xfail above.
        # Snippet used to generate expected (torch only, executed on cpu float64):
        #   normalize_homography(eye(3, float64), (4, 4), (6, 6))[0, 0]     -> 0.5999999910593036
        #   denormalize_homography(eye(3, float64), (4, 4), (6, 6))[0, 0]   -> 1.6666666915019348
        #   normalize_homography3d(eye(4, float64), (4, 4, 4), (6, 6, 6))[0, 0] -> 0.5999999910593036
        #   with torch.set_default_dtype(torch.float64):
        #     normalize_homography(eye(3, float64), (4, 4), (6, 6))[0, 0]   -> 0.6000000000000001
        _skip_if_dtype_unavailable(device, torch.float64)

        normalize_homography = kornia.geometry.conversions.normalize_homography
        identity = torch.eye(3, device=device, dtype=torch.float64)[None]
        identity3d = torch.eye(4, device=device, dtype=torch.float64)[None]

        normalized = normalize_homography(identity, (4, 4), (6, 6))[0, 0, 0]
        denormalized = kornia.geometry.conversions.denormalize_homography(identity, (4, 4), (6, 6))[0, 0, 0]
        normalized3d = kornia.geometry.conversions.normalize_homography3d(identity3d, (4, 4, 4), (6, 6, 6))[0, 0, 0]

        with _ambient_default_dtype(torch.float64):
            with_float64_default = normalize_homography(identity, (4, 4), (6, 6))[0, 0, 0]

        assert_close(
            normalized,
            torch.tensor(0.5999999910593036, device=device, dtype=torch.float64),
            atol=1e-10,
            rtol=0.0,
            msg=_issue_msg("kornia#3958: normalize_homography no longer rounds its constants to float32"),
        )
        assert_close(
            denormalized,
            torch.tensor(1.6666666915019348, device=device, dtype=torch.float64),
            atol=1e-10,
            rtol=0.0,
            msg=_issue_msg("kornia#3958: denormalize_homography no longer rounds its constants to float32"),
        )
        assert_close(
            normalized3d,
            torch.tensor(0.5999999910593036, device=device, dtype=torch.float64),
            atol=1e-10,
            rtol=0.0,
            msg=_issue_msg("kornia#3958: normalize_homography3d no longer rounds its constants to float32"),
        )
        assert_close(
            with_float64_default,
            torch.tensor(0.6000000000000001, device=device, dtype=torch.float64),
            atol=1e-10,
            rtol=0.0,
            msg=_issue_msg("kornia#3958: the ambient default dtype no longer decides the constants' precision"),
        )

    def test_wart_integer_input_raises_or_nans_by_backend_3959(self, device):
        # Wart pin for kornia#3959's homography reach, companion to normalize_homography's
        # integer-input warning: the normalization matrices are cast to the input's int64 by
        # .to(input), truncating their scales to zero (the truncation itself is pinned in
        # test_wart_integer_dtype_truncates_the_scale_to_zero_3959 above), and the downstream
        # failure differs by backend. normalize_homography dies in the FINAL CHAIN MATMUL with a
        # RuntimeError -- the closed-form inverse does not raise, it silently promotes the
        # truncated int64 matrix to an all-nan float32 one, and the int64-vs-float32 matmul then
        # rejects the mix. denormalize_homography inverts the other matrix and by the other route
        # -- _torch_inverse_cast, i.e. torch.linalg.inv -- which dies on the zero diagonal of the
        # truncated matrix before any matmul runs.
        # Both legs are gated by capability PROBES rather than a device-name list: each probe IS
        # the mechanism its leg claims, so a future backend that behaves like cpu is covered instead
        # of skipped. Executed: cpu rejects the mixed batched matmul and reports a singular
        # inverse as torch.linalg.LinAlgError; mps accepts the matmul (hence the all-nan float32
        # result) and reports the singular inverse as a plain RuntimeError. The matmul probe must
        # be BATCHED -- the unbatched 2-D mixed form raises on both backends, so it would
        # discriminate nothing.
        # No message text is asserted -- torch may reword either message (the snippet quotes them
        # as samples). The exception TYPE plus the module-level _raised_by_a_kornia_guard is what
        # carries the claim instead: both raising legs assert that the failure came from downstream
        # arithmetic and NOT from a kornia guard, so a dtype guard added by a #3959 fix flips these
        # cells loudly in any style -- including a literal `raise RuntimeError(...)`, which the
        # type test alone could not tell apart from today's matmul failure. That second assert is
        # also what machine-checks the mechanism sentence above: it fails if the raise moves off
        # the chain matmul / linalg.inv statements into a guard.
        # If any cell fails, #3959 was (partly) fixed -- update or remove the warning. NOT a
        # contract that integer input must keep failing this way.
        # Snippet used to generate expected (torch only, executed on cpu and mps, torch 2.9.1):
        #   normalize_homography(torch.eye(3, dtype=torch.int64)[None], (4, 5), (8, 9))
        #     cpu -> RuntimeError: expected scalar type Long but found Float
        #            raised at conversions.py's
        #            dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
        #     mps -> all-nan float32 matrix, no error
        #   _inverse_3x3_closed_form(normal_transform_pixel(4, 5).to(torch.int64))
        #     cpu -> no exception; returns an all-nan float32 matrix
        #   denormalize_homography(torch.eye(3, dtype=torch.int64)[None], (4, 5), (8, 9))
        #     cpu -> torch.linalg.LinAlgError: linalg.inv: (Batch element 0): The diagonal
        #            element 2 is zero, the inversion could not be completed
        #     mps -> RuntimeError (linalg.inv on a singular matrix is an internal assert there)
        identity = torch.eye(3, device=device, dtype=torch.int64)[None]

        try:
            _ = (
                torch.eye(3, device=device, dtype=torch.int64)[None]
                @ torch.eye(3, device=device, dtype=torch.float32)[None]
            )
        except RuntimeError:
            mixed_matmul_rejected = True
        else:
            mixed_matmul_rejected = False

        if mixed_matmul_rejected:
            with pytest.raises(RuntimeError) as normalize_err:
                kornia.geometry.conversions.normalize_homography(identity, (4, 5), (8, 9))
            assert not _raised_by_a_kornia_guard(normalize_err.value), (
                "kornia#3959: normalize_homography now rejects integer input in a guard of its own "
                "-- update or remove the warning"
            )
        else:
            out = kornia.geometry.conversions.normalize_homography(identity, (4, 5), (8, 9))
            assert out.dtype == torch.float32, "kornia#3959: the accepting int64 result is no longer float32"
            assert out.isnan().all(), "kornia#3959: the accepting int64 result is no longer all-nan"

        try:
            torch.linalg.inv(torch.zeros(1, 3, 3, device=device, dtype=torch.float32))
        except RuntimeError as err:
            singular_inverse_error = type(err)
        else:
            singular_inverse_error = None

        if singular_inverse_error is None:
            pytest.skip("backend does not raise on a singular linalg.inv; #3959's denormalize leg has no failure here")

        with pytest.raises(singular_inverse_error) as denormalize_err:
            kornia.geometry.conversions.denormalize_homography(identity, (4, 5), (8, 9))

        assert not _raised_by_a_kornia_guard(denormalize_err.value), (
            "kornia#3959: denormalize_homography now rejects integer input in a guard of its own "
            "-- update or remove the warning"
        )

    @pytest.mark.parametrize(("op_name", "wrong_size"), _WRONG_SIZE_CASES, ids=_WRONG_SIZE_IDS)
    def test_convention_wrong_sized_matrices_are_rejected_by_the_shape_guard_3960(self, op_name, wrong_size, device):
        # Convention: a matrix of the wrong size is rejected by the function's own shape guard,
        # with a message naming the argument -- the same thing the guard already did for a rank-2
        # input of the wrong shape. Until kornia#3960 was fixed the guard read
        # `len(shape) == 3 or shape[-2:] == (3, 3)`, and the `or` meant any rank-3 tensor passed
        # whatever its trailing shape was, so a (1, 4, 4) input to the 3x3 functions (and a
        # (1, 3, 3) input to the 4x4 one) reached the matmul and died there with a message naming
        # neither the argument nor the expected shape. The guard now reads
        # `ndim in (2, 3) and shape[-2:] == (3, 3)`.
        # Three cells because the three functions carry three separate copies of the guard.
        # The body classifies the exception rather than asserting a type: the fix is a plain
        # `raise ValueError(...)` today, but a later rewrite to KORNIA_CHECK_SHAPE would raise a
        # ShapeError and a type test would fail on a change that keeps the convention. float32 is
        # hardcoded and the dtype fixture dropped because a shape guard runs before any arithmetic
        # and cannot depend on the dtype.
        # Classified by the module-level _raised_by_a_kornia_guard, so a guard in ANY style --
        # including a literal `raise RuntimeError(...)`, which no type test could tell apart from
        # the old matmul failure -- counts as guarded. An exception raised at an arithmetic
        # statement counts as unguarded, so a regression that reopens #3960 fails this pin instead
        # of passing on the raise alone. The cells come from the shared _WRONG_SIZE_CASES table.
        op = getattr(kornia.geometry.conversions, op_name)
        wrong = torch.eye(wrong_size, device=device, dtype=torch.float32)[None]

        try:
            op(wrong, *_homography_sizes(op_name))
        except Exception as err:
            guarded = _raised_by_a_kornia_guard(err)
        else:
            guarded = False

        assert guarded, f"kornia#3960: {op_name} did not reject a ({wrong_size}, {wrong_size}) matrix in its guard"

    def test_convention_normalize_homography3d_shape_error_names_bx4x4_3960(self, device):
        # The message half of kornia#3960, and the reason it is pinned apart from the cells above:
        # normalize_homography3d is a 4x4 function whose guard used to report "must be a Bx3x3
        # tensor", so a fix that only tightened the guard's *condition* would have left the wrong
        # noun in place and the cells above would still have passed. This input is rejected by the
        # guard on either side of the fix -- only the noun changed -- which is why it asserts the
        # message text where the cells above deliberately assert none.
        # Snippet used to generate expected (torch only, executed on cpu float32):
        #   normalize_homography3d(torch.zeros(4, 5), (2, 4, 5), (3, 8, 9))
        #     -> ValueError: Input dst_pix_trans_src_pix must be a Bx4x4 tensor. Got torch.Size([4, 5])
        with pytest.raises(ValueError, match="must be a Bx4x4 tensor"):
            kornia.geometry.conversions.normalize_homography3d(
                torch.zeros(4, 5, device=device, dtype=torch.float32), (2, 4, 5), (3, 8, 9)
            )

    @pytest.mark.parametrize("op_name", _HOMOGRAPHY_OP_NAMES)
    def test_convention_the_guard_accepts_rank_2_and_3_and_rejects_higher_3960(self, op_name, device):
        # The rank half of the #3960 guard rewrite, and the two cases the wrong-size cells above
        # cannot see -- they only ever pass rank-3 inputs.
        #   (1) An UNBATCHED matrix of the right size is accepted. The old `or` guard let it through
        #       by its second clause, so this is behavior the fix had to PRESERVE, not behavior it
        #       introduced: a narrower fix reading `ndim == 3 and shape[-2:] == (N, N)` would have
        #       broken it, and nothing else in this file covers the unbatched path through these
        #       three functions. The returned shape is (1, N, N), not (N, N) -- the composition
        #       broadcasts against normal_transform_pixel's leading 1 and the input's missing batch
        #       axis is not restored. Asserted as-is because it is what the old guard's callers
        #       already got; kornia#3957's batch-axis policy is pinned elsewhere.
        #   (2) A rank-4 input is now REJECTED, which the old `or` guard accepted through its second
        #       clause: `normalize_homography(eye(3).expand(2, 1, 3, 3), (4, 5), (8, 9))` used to
        #       broadcast all the way through and return a (2, 1, 3, 3) matrix. That is the
        #       tightening #3960 asks for -- all three functions document a Bx3x3/Bx4x4 argument --
        #       and it is pinned here so the break is deliberate rather than incidental.
        # The guard is what is under test, so the accepting leg asserts only the returned shape;
        # the values are pinned by the numerical tests above. float32 is hardcoded and the dtype
        # fixture dropped for the same reason as the cells above.
        op = getattr(kornia.geometry.conversions, op_name)
        size = 4 if op_name.endswith("3d") else 3
        sizes = _homography_sizes(op_name)
        eye = torch.eye(size, device=device, dtype=torch.float32)

        assert op(eye, *sizes).shape == (1, size, size)
        assert op(eye[None], *sizes).shape == (1, size, size)

        with pytest.raises(ValueError, match="dst_pix_trans_src_pix"):
            op(eye.expand(2, 1, size, size), *sizes)

    def test_singleton_dsize_produces_finite_homographies(self, device):
        identity = torch.eye(3, device=device, dtype=torch.float32)[None]
        identity3d = torch.eye(4, device=device, dtype=torch.float32)[None]

        outputs = [
            kornia.geometry.conversions.normalize_homography(identity, (4, 1), (4, 5)),
            kornia.geometry.conversions.normalize_homography(identity, (4, 5), (4, 1)),
            kornia.geometry.conversions.denormalize_homography(identity, (4, 1), (4, 5)),
            kornia.geometry.conversions.denormalize_homography(identity, (4, 5), (4, 1)),
            kornia.geometry.conversions.normalize_homography3d(identity3d, (1, 4, 5), (3, 8, 9)),
            kornia.geometry.conversions.normalize_homography3d(identity3d, (3, 8, 9), (1, 4, 5)),
        ]
        assert all(torch.isfinite(output).all() for output in outputs)

    def test_one_pixel_output_is_finite_3957(self, device):
        image = torch.arange(25.0, device=device, dtype=torch.float32).view(1, 1, 5, 5)
        identity = torch.eye(3, device=device, dtype=torch.float32)[None]

        warped = kornia.geometry.transform.warp_perspective(image, identity, (1, 4), align_corners=True)
        cropped = kornia.geometry.transform.crop_by_transform_mat(image, identity, (1, 4), align_corners=True)
        control = kornia.geometry.transform.warp_perspective(image, identity, (2, 4), align_corners=True)

        assert torch.isfinite(warped).all()
        assert torch.isfinite(cropped).all()
        self.assert_close(warped[0, 0, 0], control[0, 0, 0], atol=0.0, rtol=0.0)
        self.assert_close(cropped[0, 0, 0], control[0, 0, 0], atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(("trace_height", "runtime_height"), [(5, 1), (1, 5)])
    def test_one_pixel_output_trace_crosses_singleton_source_boundary(self, trace_height, runtime_height, device):
        class WarpPerspective(torch.nn.Module):
            def forward(self, image, transform):
                return kornia.geometry.transform.warp_perspective(image, transform, (1, 4), align_corners=True)

        identity = torch.eye(3, device=device, dtype=torch.float32)[None]
        example = torch.arange(float(trace_height * 5), device=device).view(1, 1, trace_height, 5)
        runtime = torch.arange(float(runtime_height * 5), device=device).view(1, 1, runtime_height, 5)
        traced = torch.jit.trace(WarpPerspective(), (example, identity))
        self.assert_close(traced(runtime, identity), WarpPerspective()(runtime, identity), atol=0.0, rtol=0.0)


class TestProjectPoints(BaseTester):
    def test_smoke(self, device, dtype):
        point_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        point_3d = torch.zeros(2, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 2)

    def test_smoke_batch_multi(self, device, dtype):
        point_3d = torch.zeros(2, 4, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, 4, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 4, 2)

    def test_project_and_unproject(self, device, dtype):
        point_3d = torch.tensor([[10.0, 2.0, 30.0]], device=device, dtype=dtype)
        depth = point_3d[..., -1:]
        camera_matrix = torch.tensor(
            [[[2746.0, 0.0, 991.0], [0.0, 2748.0, 619.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        point_3d_hat = kornia.geometry.camera.unproject_points(point_2d, depth, camera_matrix)
        self.assert_close(point_3d, point_3d_hat, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # TODO: point [0, 0, 0] crashes
        points_3d = torch.ones(1, 3, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.camera.project_points, (points_3d, camera_matrix))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.camera.project_points
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_3d, camera_matrix)
        expected = op(points_3d, camera_matrix)

        self.assert_close(actual, expected)


class TestDenormalizePointsWithIntrinsics(BaseTester):
    def test_smoke(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        points_norm = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        points_2d = torch.zeros(2, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 2)

    def test_smoke_batch_n(self, device, dtype):
        points_2d = torch.zeros(2, 9, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 9, 2)

    def test_toy(self, device, dtype):
        point_2d = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.denormalize_points_with_intrinsics
        expected = torch.tensor([[192.0, 192.0]], device=device, dtype=dtype)
        self.assert_close(op(point_2d, camera_matrix), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_2d = torch.zeros(1, 2, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.denormalize_points_with_intrinsics, (points_2d, camera_matrix))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.conversions.denormalize_points_with_intrinsics
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_2d, camera_matrix)
        expected = op(points_2d, camera_matrix)

        self.assert_close(actual, expected)

    def test_convention_maps_normalized_camera_points_to_pixels(self, device, dtype):
        # Convention pin: the input is in *normalized camera* coordinates and the output is in
        # *pixel* coordinates, using the pinhole layout u = x * fx + cx, v = y * fy + cy.
        # fx != fy and cx != cy so a transposed or swapped read of K is caught.
        # Snippet used to generate expected (stdlib only, fx=100, fy=200, cx=320, cy=240):
        #   1.0 * 100 + 320, 1.0 * 200 + 240 -> (420.0, 440.0)
        points_norm = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.denormalize_points_with_intrinsics(points_norm, camera_matrix)

        expected = torch.tensor([[420.0, 440.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)


class TestNormalizePointsWithIntrinsics(BaseTester):
    def test_smoke(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        points_norm = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        points_2d = torch.zeros(2, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 2)

    def test_smoke_batch_n(self, device, dtype):
        points_2d = torch.zeros(2, 10, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        points_norm = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)
        assert points_norm.shape == (2, 10, 2)

    def test_norm_unnorm(self, device, dtype):
        point_2d = torch.tensor([[128.0, 128.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        back = kornia.geometry.conversions.denormalize_points_with_intrinsics
        point_2d_norm = op(point_2d, camera_matrix)
        point_2d_hat = back(point_2d_norm, camera_matrix)
        self.assert_close(point_2d, point_2d_hat, atol=1e-4, rtol=1e-4)

    def test_toy(self, device, dtype):
        point_2d = torch.tensor([[192.0, 192.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[64.0, 0.0, 128.0], [0.0, 64.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        out = op(point_2d, camera_matrix)
        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_2d = torch.zeros(1, 2, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.conversions.normalize_points_with_intrinsics, (points_2d, camera_matrix))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.conversions.normalize_points_with_intrinsics
        op_optimized = torch_optimizer(op)

        actual = op_optimized(points_2d, camera_matrix)
        expected = op(points_2d, camera_matrix)

        self.assert_close(actual, expected)

    def test_convention_intrinsics_layout_fx_fy_cx_cy(self, device, dtype):
        # Convention pin: K is the standard row-major pinhole matrix, fx = K[..., 0, 0],
        # fy = K[..., 1, 1], cx = K[..., 0, 2], cy = K[..., 1, 2], and the point is (u, v) in
        # pixels, so x = (u - cx)/fx, y = (v - cy)/fy. fx != fy and cx != cy, so swapping the
        # two focal lengths would give [[0.5, 2.0]] and a transposed K would not divide at all.
        # Snippet used to generate expected (stdlib only, fx=100, fy=200, cx=320, cy=240):
        #   (420.0 - 320) / 100, (440.0 - 240) / 200 -> (1.0, 1.0)
        points_2d = torch.tensor([[420.0, 440.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)

        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_skew_term_is_ignored(self, device, dtype):
        # Convention pin: only the diagonal fx, fy and the [:2, 2] column of K are read -- the
        # skew entry K[..., 0, 1] is silently ignored, so a skewed K gives the same answer as
        # the skew-free one. A skew-aware implementation would return (1.0 - 7/100, 1.0).
        # Snippet used to generate expected (stdlib only): identical to the skew-free result,
        #   (420.0 - 320) / 100, (440.0 - 240) / 200 -> (1.0, 1.0)
        points_2d = torch.tensor([[420.0, 440.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 7.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        out = kornia.geometry.conversions.normalize_points_with_intrinsics(points_2d, camera_matrix)

        expected = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected)


class TestRt2Extrinsics(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        R = torch.rand(batch_size, 3, 3, dtype=dtype, device=device)
        t = torch.rand(batch_size, 3, 1, dtype=dtype, device=device)

        Rt = Rt_to_matrix4x4(R, t)
        assert Rt.shape == (batch_size, 4, 4)

        R2, t2 = matrix4x4_to_Rt(Rt)
        assert R2.shape == (batch_size, 3, 3)
        assert t2.shape == (batch_size, 3, 1)

        self.assert_close(R, R2, rtol=1e-4, atol=1e-5)
        self.assert_close(t, t2, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [5])
    def test_gradcheck(self, batch_size, device):
        R = torch.rand(batch_size, 3, 3, dtype=torch.float64, device=device)
        t = torch.rand(batch_size, 3, 1, dtype=torch.float64, device=device)
        self.gradcheck(kornia.geometry.conversions.Rt_to_matrix4x4, (R, t))

    # Every literal in the convention pins below uses the same asymmetric rotation
    # R = [[0, 0, 1], [1, 0, 0], [0, 1, 0]] -- the 120-degree turn about (1, 1, 1), a proper
    # rotation (det = +1) that differs from its own transpose -- with t = (1, 2, 3). Its entries are
    # 0 and 1, so every product below is exact in every dtype and the pins compare at atol=rtol=0;
    # an identity or a symmetric rotation would make the direction and transpose claims unfalsifiable.

    def test_convention_translation_is_the_last_column_under_a_0_0_0_1_row(self, device):
        # Convention pin: Rt_to_matrix4x4 places R in the top-left 3x3 block, t in the LAST COLUMN
        # (not the bottom row, which is the other packing convention in use), and appends
        # [0, 0, 0, 1]. Pinned as one hardcoded 4x4 so that a transposed or bottom-row packing
        # cannot satisfy it. float32 is hardcoded and the dtype fixture dropped for the same reason
        # as the split-back pin below: packing is pure concatenation of exactly representable
        # values -- no arithmetic touches any element -- so no other-dtype leg can fail where
        # float32 passes.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   Rt_to_matrix4x4([[0,0,1],[1,0,0],[0,1,0]], [1,2,3])
        #     -> [[0, 0, 1, 1], [1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 0, 1]]
        rotation, translation = _asymmetric_pose(device, torch.float32)

        extrinsics = Rt_to_matrix4x4(rotation, translation)

        expected = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=torch.float32,
        )
        self.assert_close(extrinsics, expected, atol=0.0, rtol=0.0)

    def test_convention_matrix_maps_x_to_r_x_plus_t(self, device):
        # Convention pin: the packed matrix computes x_out = R @ x_in + t, established by applying
        # it to points rather than by reading the argument names. The origin maps to t itself, so
        # under the camtoworld reading the rest of this family uses (pinned in
        # TestCamtoworldRtToPoseRt.test_convention_camtoworld_t_is_the_camera_centre) t is the
        # camera centre in world coordinates -- Rt_to_matrix4x4 itself is frame-agnostic and packs
        # whatever (R, t) it is given; what it is NOT is a world-to-camera packing, which would
        # send the origin to -R.T @ t.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   M @ (0, 0, 0, 1) -> [1, 2, 3, 1]           == t
        #   M @ (1, 0, 0, 1) -> [1, 3, 3, 1]           == R[:, 0] + t = (0, 1, 0) + (1, 2, 3)
        # float32 is hardcoded and the dtype fixture dropped: Rt_to_matrix4x4 is pure
        # concatenation, so only the test-side matmul would run per-dtype -- torch's arithmetic,
        # not kornia's.
        rotation, translation = _asymmetric_pose(device, torch.float32)

        extrinsics = Rt_to_matrix4x4(rotation, translation)[0]

        origin = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        unit_x = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)

        self.assert_close(
            extrinsics @ origin,
            torch.tensor([1.0, 2.0, 3.0, 1.0], device=device, dtype=torch.float32),
            atol=0.0,
            rtol=0.0,
        )
        self.assert_close(
            extrinsics @ unit_x,
            torch.tensor([1.0, 3.0, 3.0, 1.0], device=device, dtype=torch.float32),
            atol=0.0,
            rtol=0.0,
        )

    def test_convention_matrix4x4_to_Rt_splits_back_and_ignores_the_bottom_row(self, device):
        # Convention pin: matrix4x4_to_Rt returns (R (B, 3, 3), t (B, 3, 1)) sliced out of the same
        # positions Rt_to_matrix4x4 wrote them to. Rt -> 4x4 -> Rt is therefore bitwise for any
        # (R, t); the OTHER direction is bitwise only for a CANONICAL extrinsics matrix, one whose
        # bottom row is already [0, 0, 0, 1]. It reads only the top three rows: a projective
        # (non-affine) bottom row is silently dropped, and packing the pieces back re-imposes the
        # canonical [0, 0, 0, 1], so a matrix carrying [9, 9, 9, 9] does not survive the trip
        # (executed below). That makes the pair lossy for anything but a rigid transform, which is
        # the claim this pin fixes. float32 is hardcoded and the dtype fixture dropped: the round
        # trip is pure slicing and concatenation of exactly representable values -- no arithmetic
        # touches any element -- so no other-dtype leg can fail where float32 passes and the
        # fixture only multiplied cells.
        # Not asserted, per this file's rule that a comparison which could only flip together with
        # one already made is left out: the two recovered shapes (assert_close checks shape itself,
        # and the two calls below run against the (1, 3, 3)/(1, 3, 1) tensors _asymmetric_pose
        # returns) and the canonical re-pack
        # Rt_to_matrix4x4(recovered_R, recovered_t) == extrinsics (pure cat/pad over tensors the
        # two lines above already pin bitwise to what extrinsics was built from). The PROJECTIVE
        # re-pack at the end IS asserted -- the rebuilt [0, 0, 0, 1] bottom row is what nothing
        # else here constrains.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   matrix4x4_to_Rt(M) -> (R, t) equal to the inputs of Rt_to_matrix4x4, bitwise
        #   matrix4x4_to_Rt(M with bottom row [9, 9, 9, 9]) -> the same (R, t)
        #   Rt_to_matrix4x4(that R, t)[0, 3] -> [0, 0, 0, 1]
        rotation, translation = _asymmetric_pose(device, torch.float32)
        extrinsics = Rt_to_matrix4x4(rotation, translation)

        recovered_R, recovered_t = matrix4x4_to_Rt(extrinsics)

        self.assert_close(recovered_R, rotation, atol=0.0, rtol=0.0)
        self.assert_close(recovered_t, translation, atol=0.0, rtol=0.0)

        projective = extrinsics.clone()
        projective[0, 3] = torch.tensor([9.0, 9.0, 9.0, 9.0], device=device, dtype=torch.float32)
        projective_R, projective_t = matrix4x4_to_Rt(projective)

        self.assert_close(projective_R, rotation, atol=0.0, rtol=0.0)
        self.assert_close(projective_t, translation, atol=0.0, rtol=0.0)
        self.assert_close(
            Rt_to_matrix4x4(projective_R, projective_t)[0, 3],
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32),
            atol=0.0,
            rtol=0.0,
        )

    def test_convention_matrix4x4_to_Rt_returns_views_of_its_input(self, device):
        # Convention pin: the two returned tensors are VIEWS of the input extrinsics, not copies --
        # writing into the returned R in place rewrites the caller's matrix. Pinned by observing the
        # mutation rather than by comparing data_ptr(), so the pin stays true to what a caller can
        # actually notice. float32 is hardcoded and the dtype fixture dropped: whether the returned
        # tensors alias the input is a property of the slicing, not of the element type, so no
        # other-dtype leg can fail where float32 passes and the fixture only multiplied cells.
        # What this pin does NOT decide: whether the views are contiguous (they are not, today),
        # whether any particular later kornia version must keep aliasing, or what a copy-returning
        # implementation should do; it records the aliasing so the Convention block that documents
        # it stays honest. The t leg is asserted separately because R and t are two independent
        # slices and a fix could copy one and not the other.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   R, t = matrix4x4_to_Rt(M); R.mul_(0.) -> M[:, :3, :3] becomes all zeros
        #   t.mul_(0.)                            -> M[:, :3, 3] becomes all zeros
        rotation, translation = _asymmetric_pose(device, torch.float32)
        extrinsics = Rt_to_matrix4x4(rotation, translation)

        aliased_R, aliased_t = matrix4x4_to_Rt(extrinsics)
        aliased_R.mul_(0.0)

        self.assert_close(
            extrinsics[0, :3, :3], torch.zeros(3, 3, device=device, dtype=torch.float32), atol=0.0, rtol=0.0
        )
        self.assert_close(
            extrinsics[0, :3, 3], torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float32), atol=0.0, rtol=0.0
        )

        aliased_t.mul_(0.0)

        self.assert_close(extrinsics[0, :3, 3], torch.zeros(3, device=device, dtype=torch.float32), atol=0.0, rtol=0.0)

    def test_wart_int64_Rt_raises_while_4x4_and_pose_forms_accept_3959(self, device):
        # Wart pin for the int64 dichotomy this family documents: Rt_to_matrix4x4 and the two
        # frame functions built on it raise RuntimeError on an int64 (R, t) -- the appended float
        # homogeneous row cannot be cast to Long -- while the _4x4 forms and the
        # transpose-negate-multiply pose pair accept int64 and return int64. Part of the
        # kornia#3959 integer-dtype family; without this pin, a torch promotion change in cat/pad
        # or a dtype guard added to one side would invalidate the docstring warnings with nothing
        # flipping red. RuntimeError is asserted by type only (the message is torch's and may be
        # reworded; a sample is quoted in the snippet), and each raising leg additionally asserts
        # through the module-level _raised_by_a_kornia_guard that the failure came from downstream
        # arithmetic rather than from a guard -- so a #3959 dtype guard flips these legs loudly in
        # any style, a bare `raise RuntimeError(...)` included.
        # Maintenance map, if any leg fails (#3959 was partly fixed, or torch promotion changed).
        # Five docstrings carry int64 text, two of them the carriers and three pointers into them:
        #   carriers: Rt_to_matrix4x4, camtoworld_graphics_to_vision_4x4
        #   pointers: camtoworld_graphics_to_vision_Rt, camtoworld_vision_to_graphics_4x4,
        #             camtoworld_vision_to_graphics_Rt
        # Edit the two carriers, then re-check that the three pointers still describe what they
        # point at. camtoworld_to_worldtocam_Rt is deliberately NOT on this list: its int64
        # acceptance is exercised below but its docstring carries no integer text (only the #3961
        # non-orthogonal-R warning), so there is nothing there to update.
        # NOT a contract that the raising side must keep raising.
        # The accepting legs run only where the backend implements integer batched matmul (probed
        # visibly below): PyTorch 2.9.1 implements it for no integer dtype on CUDA -- source-
        # derived from that release's aten/src/ATen/native/cuda/Blas.cpp, not executed here, and
        # the probe rather than this comment is what decides -- so there the accepting side is a
        # torch limitation, not a kornia guard, and the warnings scope their accept-and-return-
        # int64 sentences to cpu/mps for the same reason. The raising legs run everywhere -- they
        # raise RuntimeError on every backend, whichever operation dies first.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   Rt_to_matrix4x4(eye(3, int64)[None], ones(1, 3, 1, int64))
        #     -> RuntimeError: result type Float can't be cast to the desired output type Long
        #   camtoworld_graphics_to_vision_4x4(eye(4, int64)[None]).dtype   -> torch.int64
        #   camtoworld_to_worldtocam_Rt(eye(3, int64)[None], ones(1, 3, 1, int64))
        #     -> (int64, int64)
        rotation = torch.eye(3, device=device, dtype=torch.int64)[None]
        translation = torch.ones(1, 3, 1, device=device, dtype=torch.int64)
        extrinsics = torch.eye(4, device=device, dtype=torch.int64)[None]

        for op in (Rt_to_matrix4x4, camtoworld_graphics_to_vision_Rt, camtoworld_vision_to_graphics_Rt):
            with pytest.raises(RuntimeError) as excinfo:
                op(rotation, translation)
            assert not _raised_by_a_kornia_guard(excinfo.value), (
                f"kornia#3959: {op.__name__} now rejects int64 (R, t) in a guard of its own -- "
                "update the docstring warnings named in the map above"
            )

        try:
            torch.eye(2, device=device, dtype=torch.int64)[None] @ torch.eye(2, device=device, dtype=torch.int64)[None]
        except RuntimeError:
            pytest.skip("backend implements no integer batched matmul; the accepting side of #3959 is cpu/mps-scoped")

        for op in (camtoworld_graphics_to_vision_4x4, camtoworld_vision_to_graphics_4x4):
            assert op(extrinsics).dtype == torch.int64, (
                "kornia#3959: the _4x4 side of the int64 dichotomy no longer accepts integer input"
            )

        for op in (camtoworld_to_worldtocam_Rt, worldtocam_to_camtoworld_Rt):
            pose_R, pose_t = op(rotation, translation)
            assert pose_R.dtype == torch.int64 and pose_t.dtype == torch.int64, (
                "kornia#3959: the pose pair no longer accepts integer input"
            )

    @pytest.mark.parametrize(
        ("op_name", "shapes"),
        [
            ("Rt_to_matrix4x4", ((3, 3), (1, 3, 1))),
            ("Rt_to_matrix4x4", ((1, 3, 3), (1, 3))),
            ("Rt_to_matrix4x4", ((1, 3, 3), (1, 1, 3))),
            ("Rt_to_matrix4x4", ((1, 4, 4), (1, 3, 1))),
            ("Rt_to_matrix4x4", ((2, 1, 3, 3), (1, 3, 1))),
            ("matrix4x4_to_Rt", ((4, 4),)),
            ("matrix4x4_to_Rt", ((2, 1, 4, 4),)),
            ("matrix4x4_to_Rt", ((1, 3, 3),)),
        ],
        ids=[
            "pack-unbatched-R",
            "pack-flat-t",
            "pack-transposed-t",
            "pack-4x4-R",
            "pack-extra-batch-dim",
            "split-unbatched",
            "split-extra-batch-dim",
            "split-3x3",
        ],
    )
    def test_convention_shapes_are_strictly_batched(self, op_name, shapes, device):
        # Convention pin: both functions go through KORNIA_CHECK_SHAPE and accept exactly
        # (B, 3, 3) + (B, 3, 1) and (B, 4, 4) -- no unbatched form, no (B, 3) translation, no
        # transposed (B, 1, 3) translation, no extra leading batch dimensions. This is the strict
        # end of the family: camtoworld_to_worldtocam_Rt broadcasts a (1, 3, 1) translation across
        # a (2, 3, 3) rotation where Rt_to_matrix4x4 raises, which the Convention blocks record.
        # The batch-mismatch case is a torch.cat RuntimeError, pinned in
        # test_convention_batch_sizes_must_match below. Assertion policy and the float32 hardcoding
        # are documented once on the shared _assert_strictly_batched helper.
        _assert_strictly_batched(op_name, shapes, device)

    def test_convention_batch_sizes_must_match(self, device):
        # Convention pin: Rt_to_matrix4x4 does NOT broadcast -- a batch of 2 rotations with a single
        # translation raises inside torch.cat instead of repeating the translation. RuntimeError is
        # asserted by type only: the message is entirely PyTorch's wording and may be reworded (a
        # sample is quoted in the snippet), same rule as the int64 pin above.
        # float32 is hardcoded for the same reason as the pin above.
        # Snippet used to generate expected (torch only, executed on cpu float32):
        #   Rt_to_matrix4x4(torch.eye(3).expand(2, 3, 3), torch.ones(1, 3, 1))
        #     -> RuntimeError: Sizes of tensors must match except in dimension 2. Expected size 2
        #        but got size 1 for tensor number 1 in the list.
        rotation = torch.eye(3, device=device, dtype=torch.float32).expand(2, 3, 3)
        translation = torch.ones(1, 3, 1, device=device, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            Rt_to_matrix4x4(rotation, translation)


class TestCamtoworldGraphicsToVision(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        t_vis = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R_vis = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        K_vis = Rt_to_matrix4x4(R_vis, t_vis)
        K_graf = camtoworld_vision_to_graphics_4x4(K_vis)

        expected = torch.tensor(
            [[0, 0, -1, 2], [0, -1, 0, 3], [-1, 0, 0, 4], [0, 0, 0, 1]], device=device, dtype=dtype
        )[None].repeat(batch_size, 1, 1)

        self.assert_close(K_graf, expected, rtol=1e-4, atol=1e-5)
        R_graf, t_graf = camtoworld_vision_to_graphics_Rt(R_vis, t_vis)
        expected_R = torch.tensor([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], device=device, dtype=dtype)[None].repeat(
            batch_size, 1, 1
        )
        expected_t = torch.tensor([2, 3, 4], device=device, dtype=dtype).reshape(1, 3, 1).repeat(batch_size, 1, 1)

        self.assert_close(t_graf, expected_t, rtol=1e-4, atol=1e-5)
        self.assert_close(R_graf, expected_R, rtol=1e-4, atol=1e-5)

        Kvis_back = camtoworld_graphics_to_vision_4x4(K_graf)
        self.assert_close(Kvis_back, K_vis, rtol=1e-4, atol=1e-5)

        R_vis_back, t_vis_back = camtoworld_graphics_to_vision_Rt(R_graf, t_graf)
        self.assert_close(R_vis_back, R_vis, rtol=1e-4, atol=1e-5)
        self.assert_close(t_vis_back, t_vis, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device):
        t_vis = torch.tensor([2, 3, 4], device=device, dtype=torch.float64).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=torch.float64)[None]
        R_vis = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        K_vis = Rt_to_matrix4x4(R_vis, t_vis)
        self.gradcheck(camtoworld_graphics_to_vision_4x4, (K_vis,))
        self.gradcheck(camtoworld_vision_to_graphics_4x4, (K_vis,))

    # The convention pins below use the same asymmetric pose as TestRt2Extrinsics --
    # R = [[0, 0, 1], [1, 0, 0], [0, 1, 0]] (a proper rotation that is not its own transpose) with
    # t = (1, 2, 3) -- because an identity or a symmetric pose is invariant under the very flip
    # being pinned. All entries are 0, 1 or small integers, so every literal is exact in every dtype.

    def test_convention_flip_right_multiplies_by_diag_1_minus1_minus1_1(self, device, dtype):
        # Convention pin: the conversion is extrinsics @ diag(1, -1, -1, 1) -- a RIGHT
        # multiplication, i.e. a change of the CAMERA-side basis. Two consequences are pinned
        # because both are what a caller gets wrong: columns 1 and 2 of the rotation flip sign while
        # column 0 is untouched, and the translation column is left ALONE. The left-multiplied
        # alternative, diag(1, -1, -1, 1) @ extrinsics, negates the translation instead -- exactly
        # what happens when a *worldtocam* matrix is passed to these functions, a silent error
        # since the shapes match. That contrast is recorded in the snippet below rather than
        # asserted: it is test-side arithmetic on an already-pinned matrix with no kornia call on
        # its path, so per this file's rule it could only flip on a torch matmul regression.
        # The determinant of the rotation block stays +1: two axes flip, not one, so handedness is
        # preserved and this is a rotation rather than a reflection.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   M = [[0, 0, 1, 1], [1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 0, 1]]
        #   camtoworld_graphics_to_vision_4x4(M)
        #     -> [[0, 0, -1, 1], [1, 0, 0, 2], [0, -1, 0, 3], [0, 0, 0, 1]]
        #   diag(1, -1, -1, 1) @ M
        #     -> [[0, 0, 1, 1], [-1, 0, 0, -2], [0, -1, 0, -3], [0, 0, 0, 1]]
        _skip_if_dtype_unavailable(device, dtype)
        rotation, translation = _asymmetric_pose(device, dtype)
        extrinsics = Rt_to_matrix4x4(rotation, translation)

        flipped = camtoworld_graphics_to_vision_4x4(extrinsics)

        expected = torch.tensor(
            [[[0.0, 0.0, -1.0, 1.0], [1.0, 0.0, 0.0, 2.0], [0.0, -1.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(flipped, expected, atol=0.0, rtol=0.0)

        flipped_R, flipped_t = camtoworld_graphics_to_vision_Rt(rotation, translation)

        self.assert_close(flipped_t, translation, atol=0.0, rtol=0.0)
        self.assert_close(flipped_R, flipped[:, :3, :3], atol=0.0, rtol=0.0)
        _assert_proper_rotation(flipped_R)

    @pytest.mark.parametrize(
        ("op_name", "shapes"),
        [
            ("camtoworld_graphics_to_vision_4x4", ((4, 4),)),
            ("camtoworld_graphics_to_vision_4x4", ((2, 1, 4, 4),)),
            ("camtoworld_graphics_to_vision_4x4", ((1, 3, 3),)),
            ("camtoworld_vision_to_graphics_4x4", ((4, 4),)),
            ("camtoworld_graphics_to_vision_Rt", ((3, 3), (1, 3, 1))),
            ("camtoworld_graphics_to_vision_Rt", ((1, 3, 3), (1, 3))),
            ("camtoworld_vision_to_graphics_Rt", ((1, 4, 4), (1, 3, 1))),
        ],
        ids=[
            "g2v-4x4-unbatched",
            "g2v-4x4-extra-batch-dim",
            "g2v-4x4-3x3",
            "v2g-4x4-unbatched",
            "g2v-Rt-unbatched-R",
            "g2v-Rt-flat-t",
            "v2g-Rt-4x4-R",
        ],
    )
    def test_convention_shapes_are_strictly_batched(self, op_name, shapes, device):
        # Convention pin: the four conversions accept exactly (B, 4, 4) and (B, 3, 3) + (B, 3, 1) --
        # no unbatched form and no extra leading batch dimensions, the same strictness as
        # Rt_to_matrix4x4, which the Rt variants call. Assertion policy and the float32 hardcoding
        # are documented once on the shared _assert_strictly_batched helper.
        _assert_strictly_batched(op_name, shapes, device)

    def test_convention_graphics_is_y_up_and_vision_is_y_down(self, device, dtype):
        # Convention pin: the two frames the docstrings name, read off the columns of the converted
        # identity pose. For a camtoworld matrix, column i is the world direction of camera axis i,
        # so converting the identity graphics pose gives the vision axes in graphics coordinates:
        # +x is shared, +y is negated (up -> down) and +z is negated (backwards -> forwards).
        # Graphics (OpenGL): [+x, +y, +z] == [right, up, backwards], the camera looks down -z.
        # Vision (OpenCV):   [+x, +y, +z] == [right, down, forwards], the camera looks down +z.
        # Asserted as the full diag(1, -1, -1, 1) matrix rather than per-column reads -- strictly
        # stronger: the translation column and the bottom row are covered too.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   camtoworld_graphics_to_vision_4x4(torch.eye(4)[None]) -> diag(1, -1, -1, 1)
        _skip_if_dtype_unavailable(device, dtype)
        identity = torch.eye(4, device=device, dtype=dtype)[None]

        vision_axes = camtoworld_graphics_to_vision_4x4(identity)[0]

        self.assert_close(
            vision_axes,
            torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device, dtype=dtype)),
            atol=0.0,
            rtol=0.0,
        )

    def test_convention_all_four_functions_are_the_same_involution(self, device, dtype):
        # Convention pin: diag(1, -1, -1, 1) is its own inverse, so graphics_to_vision and
        # vision_to_graphics are the IDENTICAL map -- the direction in the name is documentation for
        # the reader, not behavior -- and applying either one twice returns the input value-exactly
        # (atol=rtol=0, no rounding). VALUE-exactly and not bitwise: assert_close at atol=rtol=0
        # treats -0.0 and +0.0 as equal, and the flip does not preserve the sign of a zero, which
        # test_convention_involution_normalises_signed_zero below pins on the raw words. Both
        # halves are executed here rather than argued from the shared implementation: the two 4x4
        # functions are compared on the same input, the Rt pair is compared against the 4x4 pair,
        # and each round trip is run.
        # Practical consequence recorded by this pin: nothing in the API can detect that a pose was
        # already converted, so a double conversion is silently a no-op.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   camtoworld_vision_to_graphics_4x4(M) == camtoworld_graphics_to_vision_4x4(M)  (bitwise)
        #   camtoworld_graphics_to_vision_4x4(camtoworld_graphics_to_vision_4x4(M)) == M  (bitwise)
        _skip_if_dtype_unavailable(device, dtype)
        rotation, translation = _asymmetric_pose(device, dtype)
        extrinsics = Rt_to_matrix4x4(rotation, translation)

        to_vision = camtoworld_graphics_to_vision_4x4(extrinsics)
        to_graphics = camtoworld_vision_to_graphics_4x4(extrinsics)

        self.assert_close(to_graphics, to_vision, atol=0.0, rtol=0.0)
        self.assert_close(camtoworld_graphics_to_vision_4x4(to_vision), extrinsics, atol=0.0, rtol=0.0)
        self.assert_close(camtoworld_vision_to_graphics_4x4(to_vision), extrinsics, atol=0.0, rtol=0.0)

        vision_R, vision_t = camtoworld_graphics_to_vision_Rt(rotation, translation)
        graphics_R, graphics_t = camtoworld_vision_to_graphics_Rt(rotation, translation)

        self.assert_close(graphics_R, vision_R, atol=0.0, rtol=0.0)
        self.assert_close(graphics_t, vision_t, atol=0.0, rtol=0.0)
        self.assert_close(Rt_to_matrix4x4(vision_R, vision_t), to_vision, atol=0.0, rtol=0.0)

    def test_convention_involution_normalises_signed_zero(self, device):
        # Convention pin for the one place the involution above is exact in VALUE but not in BITS,
        # which the docstring states and no assert_close can check: assert_close at atol=rtol=0
        # compares -0.0 equal to +0.0, so the pin above would pass either way and the docstring's
        # "value-exact, not bitwise" clause would be unfalsifiable. Compared on the raw words
        # instead, through an int32 view of the float32 buffer.
        # The mechanism, and why it is not specific to the two flipped columns: the flip is
        # extrinsics @ diag(1, -1, -1, 1), a matmul, so every output entry is a sum of four
        # products. A -0.0 input entry contributes -0.0 to that sum and the other three terms
        # contribute +0.0, and IEEE-754 round-to-nearest gives -0.0 + 0.0 = +0.0. So EVERY signed
        # zero in the input is normalised to +0.0, in any column, on the FIRST application already,
        # and a second application cannot restore it.
        # float32 is hardcoded and the dtype fixture dropped: the subject is a sign bit, the
        # comparison needs a fixed-width integer view to read it, and the summation that loses the
        # sign is the same in every dtype.
        # NOT a contract that the flip must clear these sign bits -- if a future implementation
        # multiplies elementwise instead of through a matmul, -0.0 * -1.0 = +0.0 and -0.0 * 1.0
        # stays -0.0, so the flipped columns would flip signs and the rest would be preserved.
        # Then this pin fails and the docstring clause is what gets re-derived.
        # The input is built from a nested literal rather than by assigning -0.0 into a zeros
        # tensor: on mps (torch 2.9.1, executed) a Python-scalar assignment writes +0.0, so the
        # item-assignment form would silently build an input with no signed zeros in it and the pin
        # would assert nothing there. The guard below is what makes that failure loud, not silent.
        # Snippet used to generate expected (torch only, executed on cpu, torch 2.9.1):
        #   M = torch.tensor([[[0., -0., 0., 0.], [0.] * 4, [0.] * 4, [0.] * 4]])
        #   torch.signbit(camtoworld_graphics_to_vision_4x4(M))[0, 0, 1] -> False
        signed_zeros = torch.tensor(
            [
                [
                    [0.0, -0.0, 0.0, 0.0],
                    [-0.0, 0.0, -0.0, 0.0],
                    [0.0, -0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=torch.float32,
        )
        assert torch.signbit(signed_zeros).sum().item() == 4, "the input lost its signed zeros before the call"

        once = camtoworld_graphics_to_vision_4x4(signed_zeros)
        twice = camtoworld_graphics_to_vision_4x4(once)

        self.assert_close(twice, signed_zeros, atol=0.0, rtol=0.0)
        assert not torch.equal(twice.view(torch.int32), signed_zeros.view(torch.int32)), (
            "the involution is now bitwise on signed zeros -- camtoworld_graphics_to_vision_4x4's "
            "involution bullet says value-exact but not bitwise and needs re-deriving"
        )
        assert torch.signbit(once).sum().item() == 0, "a signed zero survived the first application"
        assert torch.signbit(twice).sum().item() == 0, "a signed zero reappeared after the second application"


class TestCamtoworldRtToPoseRt(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_everything(self, batch_size, device, dtype):
        # generate input data
        t = torch.tensor([2, 3, 4], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=dtype)[None]
        R = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)

        Rp, tp = camtoworld_to_worldtocam_Rt(R, t)

        expected_Rp = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=device, dtype=dtype)[None].repeat(
            batch_size, 1, 1
        )
        expected_tp = torch.tensor([4, -3, -2], device=device, dtype=dtype).view(1, 3, 1).repeat(batch_size, 1, 1)
        self.assert_close(Rp, expected_Rp, rtol=1e-4, atol=1e-5)
        self.assert_close(tp, expected_tp, rtol=1e-4, atol=1e-5)

        Rback, tback = worldtocam_to_camtoworld_Rt(Rp, tp)
        self.assert_close(Rback, R, rtol=1e-4, atol=1e-5)
        self.assert_close(tback, t, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [4])
    def test_gradcheck(self, batch_size, device):
        t = torch.tensor([2, 3, 4], device=device, dtype=torch.float64).view(1, 3, 1).repeat(batch_size, 1, 1)
        angles = torch.tensor([0, kornia.pi / 2.0, 0.0], device=device, dtype=torch.float64)[None]
        R = kornia.geometry.axis_angle_to_rotation_matrix(angles).repeat(batch_size, 1, 1)
        self.gradcheck(camtoworld_to_worldtocam_Rt, (R, t))
        self.gradcheck(worldtocam_to_camtoworld_Rt, (R, t))

    # The convention pins below reuse the asymmetric pose of TestRt2Extrinsics --
    # R = [[0, 0, 1], [1, 0, 0], [0, 1, 0]], t = (1, 2, 3) -- so that R.T differs from R and the
    # transpose claim is falsifiable; every literal is exact in every dtype.

    def test_convention_is_the_rigid_inverse_r_transposed_and_minus_r_transposed_t(self, device, dtype):
        # Convention pin: both functions compute exactly (R.T, -R.T @ t) -- the RIGID inverse, built
        # from a transpose, with no matrix inverse anywhere. Checked against a literal computed by
        # hand rather than against torch.inverse, so the pin does not assume the very property it
        # is asserting: for this R, R.T @ t permutes (1, 2, 3) to (2, 3, 1) and the result is its
        # negation. One assert per output: the hand literal IS R.T of _ASYMMETRIC_R, so a second
        # comparison against rotation.transpose(1, 2) could only ever flip together with it and is
        # not made. Same for t's shape -- the assert_close against a (1, 3, 1) literal checks it.
        # Snippet used to generate expected (stdlib only):
        #   R.T          = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        #   R.T @ t      = (2, 3, 1)
        #   -R.T @ t     = (-2, -3, -1)
        _skip_if_dtype_unavailable(device, dtype)
        rotation, translation = _asymmetric_pose(device, dtype)

        inverted_R, inverted_t = camtoworld_to_worldtocam_Rt(rotation, translation)

        self.assert_close(
            inverted_R,
            torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], device=device, dtype=dtype),
            atol=0.0,
            rtol=0.0,
        )
        self.assert_close(
            inverted_t, torch.tensor([[[-2.0], [-3.0], [-1.0]]], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )

    def test_convention_both_directions_are_the_same_function(self, device, dtype):
        # Convention pin: camtoworld_to_worldtocam_Rt and worldtocam_to_camtoworld_Rt are the same
        # formula under two names -- bitwise identical outputs on the same input -- so the direction
        # lives in the caller's head, not in the call. Consequences pinned here: applying either one
        # twice returns the input, and the two compose to the input in EITHER order. All three round
        # trips are executed separately rather than one being inferred from another.
        # This holds because R is a rotation; the same round trip fails for a non-orthogonal R,
        # which test_wart_non_orthogonal_rotation_is_transposed_not_inverted_3961 pins.
        _skip_if_dtype_unavailable(device, dtype)
        rotation, translation = _asymmetric_pose(device, dtype)

        forward_R, forward_t = camtoworld_to_worldtocam_Rt(rotation, translation)
        backward_R, backward_t = worldtocam_to_camtoworld_Rt(rotation, translation)

        self.assert_close(backward_R, forward_R, atol=0.0, rtol=0.0)
        self.assert_close(backward_t, forward_t, atol=0.0, rtol=0.0)

        twice_R, twice_t = camtoworld_to_worldtocam_Rt(forward_R, forward_t)
        round_R, round_t = worldtocam_to_camtoworld_Rt(forward_R, forward_t)

        self.assert_close(twice_R, rotation, atol=0.0, rtol=0.0)
        self.assert_close(twice_t, translation, atol=0.0, rtol=0.0)
        self.assert_close(round_R, rotation, atol=0.0, rtol=0.0)
        self.assert_close(round_t, translation, atol=0.0, rtol=0.0)

    def test_convention_camtoworld_t_is_the_camera_centre(self, device, dtype):
        # Convention pin: what t MEANS on each side, established by applying the packed 4x4 matrices
        # to points. On the camtoworld side the camera origin maps to t, so t is the camera centre
        # in world coordinates; on the worldtocam (Colmap) side the same centre maps to the origin,
        # so the returned t' = -R.T @ t is the world-to-camera translation and not a second camera
        # centre. The composition of the two matrices is the identity.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   M  @ (0, 0, 0, 1) -> [1, 2, 3, 1]      (the camera centre)
        #   Mi @ (1, 2, 3, 1) -> [0, 0, 0, 1]
        #   Mi @ M            -> the identity
        _skip_if_dtype_unavailable(device, dtype)
        rotation, translation = _asymmetric_pose(device, dtype)

        camtoworld = Rt_to_matrix4x4(rotation, translation)[0]
        worldtocam = Rt_to_matrix4x4(*camtoworld_to_worldtocam_Rt(rotation, translation))[0]

        origin = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
        centre = torch.tensor([1.0, 2.0, 3.0, 1.0], device=device, dtype=dtype)

        self.assert_close(camtoworld @ origin, centre, atol=0.0, rtol=0.0)
        self.assert_close(worldtocam @ centre, origin, atol=0.0, rtol=0.0)
        self.assert_close(worldtocam @ camtoworld, torch.eye(4, device=device, dtype=dtype), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(
        ("op_name", "shapes"),
        [
            ("camtoworld_to_worldtocam_Rt", ((3, 3), (1, 3, 1))),
            ("camtoworld_to_worldtocam_Rt", ((1, 3, 3), (1, 3))),
            ("camtoworld_to_worldtocam_Rt", ((2, 1, 3, 3), (1, 3, 1))),
            ("worldtocam_to_camtoworld_Rt", ((3, 3), (1, 3, 1))),
            ("worldtocam_to_camtoworld_Rt", ((1, 3, 3), (1, 3))),
        ],
        ids=["c2w-unbatched-R", "c2w-flat-t", "c2w-extra-batch-dim", "w2c-unbatched-R", "w2c-flat-t"],
    )
    def test_convention_shapes_are_strictly_batched(self, op_name, shapes, device):
        # Convention pin: both functions accept exactly (B, 3, 3) + (B, 3, 1) -- no unbatched form,
        # no (B, 3) translation, no extra leading batch dimensions. What they do NOT enforce is a
        # matching batch size: a (1, 3, 1) translation broadcasts across a (2, 3, 3) rotation here,
        # where Rt_to_matrix4x4 raises (pinned in TestRt2Extrinsics). That asymmetry is recorded in
        # the Convention blocks and is deliberately not pinned as a contract.
        # Assertion policy and the float32 hardcoding are documented once on the shared
        # _assert_strictly_batched helper.
        _assert_strictly_batched(op_name, shapes, device)

    def test_wart_non_orthogonal_rotation_is_transposed_not_inverted_3961(self, device, dtype):
        # Wart pin for kornia#3961: R is ASSUMED orthogonal and the assumption is never checked, so
        # for any other matrix the result is a transpose that is not an inverse -- silently, with no
        # error and no warning. Three cells, each a different observable of the same root:
        #   (1) the returned rotation is exactly R.T even though R.T is not R^-1 here;
        #   (2) composing the two 4x4 matrices misses the identity by 3.0, i.e. not by a rounding
        #       amount -- this is what a caller notices as "my poses drifted";
        #   (3) even the round trip breaks, the translation coming back 9.0 away from the input;
        #       cell (3) is kept separate from (2) because a fix that validated only the rotation
        #       would leave the translation error in place for a caller who ignores the raise.
        # There is deliberately NO companion strict xfail: the intended behavior is undecided -- a
        # fix could raise on a non-orthogonal R, or fall back to a true inverse -- and an
        # assertion-shaped xfail could express only one of those and would stay silently XFAIL if
        # the other were chosen. Same shape as the #3959 and #3957 wart pins in this file.
        # If any cell fails, #3961 was (partly) fixed -- remove this pin. NOT a contract that a
        # non-orthogonal R must keep producing these numbers.
        # R = [[1, 0.5, 0], [0, 1, 0], [0, 0, 2]] has det = 2 and dyadic entries, so every literal
        # below is exact in every dtype.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   camtoworld_to_worldtocam_Rt(R, (1, 2, 3))
        #     -> R' = [[1, 0, 0], [0.5, 1, 0], [0, 0, 2]] (= R.T), t' = (-1, -2.5, -6)
        #   max|Rt_to_matrix4x4(R', t') @ Rt_to_matrix4x4(R, t) - I| -> 3.0
        #   worldtocam_to_camtoworld_Rt(R', t')[1] -> (2.25, 2.5, 12.0), i.e. 9.0 from (1, 2, 3)
        _skip_if_dtype_unavailable(device, dtype)
        rotation = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]], device=device, dtype=dtype)
        translation = torch.tensor([[[1.0], [2.0], [3.0]]], device=device, dtype=dtype)

        inverted_R, inverted_t = camtoworld_to_worldtocam_Rt(rotation, translation)
        composed = Rt_to_matrix4x4(inverted_R, inverted_t)[0] @ Rt_to_matrix4x4(rotation, translation)[0]
        _, round_trip_t = worldtocam_to_camtoworld_Rt(inverted_R, inverted_t)

        self.assert_close(inverted_R, rotation.transpose(1, 2), atol=0.0, rtol=0.0)
        self.assert_close(
            inverted_t, torch.tensor([[[-1.0], [-2.5], [-6.0]]], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )
        assert (composed - torch.eye(4, device=device, dtype=dtype)).abs().max().item() == 3.0, (
            "kornia#3961: a non-orthogonal rotation no longer misses the identity by 3.0"
        )
        self.assert_close(
            round_trip_t, torch.tensor([[[2.25], [2.5], [12.0]]], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )


class TestCARKitToColmap(BaseTester):
    def test_everything(self, device, dtype):
        # generate input data
        t = torch.tensor([1, 0, 0], device=device, dtype=dtype).view(1, 3, 1)
        ang_deg = torch.tensor([45, 60.0, 0.0], device=device, dtype=dtype)[None]
        ang_rad = kornia.geometry.conversions.deg2rad(ang_deg)
        qvec = kornia.geometry.axis_angle_to_quaternion(ang_rad)

        q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(qvec, t)

        angles_colmap = kornia.geometry.conversions.quaternion_to_axis_angle(q_colmap)
        angles_colmap = kornia.geometry.conversions.rad2deg(angles_colmap)
        expected_angles = torch.tensor([[116.8870620728, 0.0, -71.7524719238]], device=device, dtype=dtype)
        expected_t = torch.tensor([[[-0.5256], [0.3558], [0.7727]]], device=device, dtype=dtype)

        self.assert_close(angles_colmap, expected_angles, rtol=1e-4, atol=1e-5)
        self.assert_close(t_colmap, expected_t, rtol=1e-4, atol=1e-5)

    def test_convention_quaternion_order_is_w_x_y_z_on_both_sides(self, device, dtype):
        # Convention pin: the real part comes FIRST on both sides of this function -- the input
        # qvec is read as (w, x, y, z), and the returned q_colmap is (w, x, y, z) too, which is
        # Colmap's images.txt order (QW QX QY QZ).
        # Two legs, one per side, because the two are independent claims:
        #   (in)  [1, 0, 0, 0] is the identity rotation, so the result is the pure frame flip. The
        #         (x, y, z, w) impostor of the same rotation, [0, 0, 0, 1], is passed as the
        #         contrast: kornia reads it as a 180-degree turn about z and returns a different
        #         quaternion AND a different translation, so a caller who forgets to reorder is not
        #         merely off by a sign.
        #   (out) the returned quaternion is fed back through quaternion_to_rotation_matrix and
        #         compared against the hand-computed R_colmap = (I @ diag(1, -1, -1)).T =
        #         diag(1, -1, -1); reading the output as (x, y, z, w) would make it the identity
        #         instead.
        # Caller obligation this pin CANNOT check, because Apple's types are not importable here:
        # ARKit's simd_quatf.vector is (ix, iy, iz, r) = xyzw, so a pose read straight from ARKit
        # must be reordered to wxyz before it is passed in.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   ARKitQTVecs_to_ColmapQTVecs([1, 0, 0, 0], [1, 2, 3]) -> [0, 1, 0, 0], [-1, 2, 3]
        #   ARKitQTVecs_to_ColmapQTVecs([0, 0, 0, 1], [1, 2, 3]) -> [0, 0, 1, 0], [1, -2, 3]
        #   quaternion_to_rotation_matrix([0, 1, 0, 0])          -> diag(1, -1, -1)
        _skip_if_dtype_unavailable(device, dtype)
        identity_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        identity_xyzw_impostor = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
        translation = torch.tensor([[[1.0], [2.0], [3.0]]], device=device, dtype=dtype)

        q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(identity_wxyz, translation)
        q_impostor, t_impostor = ARKitQTVecs_to_ColmapQTVecs(identity_xyzw_impostor, translation)

        self.assert_close(q_colmap, torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype))
        self.assert_close(t_colmap, torch.tensor([[[-1.0], [2.0], [3.0]]], device=device, dtype=dtype))
        self.assert_close(q_impostor, torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device, dtype=dtype))
        self.assert_close(t_impostor, torch.tensor([[[1.0], [-2.0], [3.0]]], device=device, dtype=dtype))

        rotation_from_output = kornia.geometry.conversions.quaternion_to_rotation_matrix(q_colmap)

        self.assert_close(
            rotation_from_output,
            torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]], device=device, dtype=dtype),
        )

    def test_convention_worked_literal_matches_the_hand_computation(self, device, dtype):
        # Convention pin: the docstring's own example, reproduced against a hand computation done
        # outside kornia (wxyz -> R by the standard formula, right-multiply by diag(1, -1, -1),
        # transpose, then negate-and-rotate the translation). The rotation is pinned as a matrix
        # rather than only as a quaternion because the quaternion sign is not part of the contract
        # (see the trace-zero pin below), and det(R_colmap) = +1 is asserted: the flip negates two
        # axes, not one, so handedness is PRESERVED despite the graphics/vision framing.
        # Snippet used to generate expected (stdlib only, q = [0, 1, 0, 1] wxyz, t = (1, 1, 1)):
        #   R_cg      = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
        #   R_colmap  = (R_cg @ diag(1, -1, -1)).T = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        #   t_colmap  = -R_colmap @ (1, 1, 1) = (-1, -1, 1)
        #   q_colmap  = [0.7071067811865476, 0.0, 0.7071067811865475, 0.0]
        #   det(R_colmap) = 1.0
        _skip_if_dtype_unavailable(device, dtype)
        qvec = torch.tensor(_ARKIT_WORKED_QVEC, device=device, dtype=dtype)
        tvec = torch.tensor(_ARKIT_WORKED_TVEC, device=device, dtype=dtype)

        q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(qvec, tvec)
        rotation = kornia.geometry.conversions.quaternion_to_rotation_matrix(q_colmap)

        self.assert_close(q_colmap, torch.tensor([[0.70710678, 0.0, 0.70710678, 0.0]], device=device, dtype=dtype))
        self.assert_close(t_colmap, torch.tensor([[[-1.0], [-1.0], [1.0]]], device=device, dtype=dtype))
        self.assert_close(
            rotation,
            torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]], device=device, dtype=dtype),
        )
        _assert_proper_rotation(rotation)

    def test_convention_output_quaternion_sign_is_not_canonical(self, device, dtype):
        # Convention pin: compare ROTATIONS, never raw quaternion components. At trace = 0 the
        # final rotation_matrix_to_quaternion step takes a branch that returns the negative-w half
        # of the double cover, so a perfectly ordinary input comes back with w < 0. Both halves
        # encode the same rotation; the sign is an implementation artefact, not information.
        # Same claim, one function further down the pipeline, as
        # TestRotationMatrixToQuaternion.test_convention_w_is_not_canonicalised_to_non_negative.
        # The second, fully asymmetric hand-computed literal of this function lives here: the input
        # q = [0.5, 0.5, 0.5, 0.5] with t = (1, 2, 3) is exact in every dtype and its R_colmap has
        # trace 0, so it exercises the branch the docstring example does not.
        # Snippet used to generate expected (stdlib only, q = [0.5]*4 wxyz, t = (1, 2, 3)):
        #   R_colmap = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]   (trace 0)
        #   t_colmap = -R_colmap @ (1, 2, 3) = (-2, 3, 1)
        #   kornia returns q = [-0.5, -0.5, -0.5, 0.5], i.e. the negative-w representative
        _skip_if_dtype_unavailable(device, dtype)
        qvec = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device, dtype=dtype)
        tvec = torch.tensor([[[1.0], [2.0], [3.0]]], device=device, dtype=dtype)

        q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(qvec, tvec)
        rotation = kornia.geometry.conversions.quaternion_to_rotation_matrix(q_colmap)

        self.assert_close(
            rotation,
            torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]], device=device, dtype=dtype),
        )
        self.assert_close(t_colmap, torch.tensor([[[-2.0], [3.0], [1.0]]], device=device, dtype=dtype))
        # The [-0.5, ...] literal itself carries the sign claim -- w within tolerance of -0.5 is
        # necessarily negative, so a separate sign assert could never fail on its own. The point
        # stands: the sign is an artefact either way; compare rotations, never raw components.
        self.assert_close(q_colmap, torch.tensor([[-0.5, -0.5, -0.5, 0.5]], device=device, dtype=dtype))

    def test_convention_input_quaternion_is_normalized_and_double_covered(self, device, dtype):
        # Convention pin: the input quaternion does not have to be unit -- the pipeline normalizes
        # it -- and q and -q describe the same rotation, so both give the same pose. Two legs
        # because they are independent: scaling exercises the normalizer, negating exercises the
        # double cover. What this pin does NOT decide is the zero quaternion, which is absorbed by
        # the normalizer's eps guard and silently produces a plausible pose (documented, not pinned:
        # the intended answer there is a maintainer decision, tracked in kornia#3952 -- this is the
        # downstream reach of that issue's sub-eps clamp).
        # Snippet used to generate expected (torch only, executed on cpu float32):
        #   ARKitQTVecs_to_ColmapQTVecs(q * 1000, t) == ARKitQTVecs_to_ColmapQTVecs(q, t)  (0.0 diff)
        #   ARKitQTVecs_to_ColmapQTVecs(-q, t)       == ARKitQTVecs_to_ColmapQTVecs(q, t)  (0.0 diff)
        _skip_if_dtype_unavailable(device, dtype)
        qvec = torch.tensor(_ARKIT_WORKED_QVEC, device=device, dtype=dtype)
        tvec = torch.tensor(_ARKIT_WORKED_TVEC, device=device, dtype=dtype)

        q_reference, t_reference = ARKitQTVecs_to_ColmapQTVecs(qvec, tvec)
        q_scaled, t_scaled = ARKitQTVecs_to_ColmapQTVecs(qvec * 1000.0, tvec)
        q_negated, t_negated = ARKitQTVecs_to_ColmapQTVecs(-qvec, tvec)

        self.assert_close(q_scaled, q_reference)
        self.assert_close(t_scaled, t_reference)
        self.assert_close(q_negated, q_reference)
        self.assert_close(t_negated, t_reference)

    def test_wart_zero_quaternion_is_absorbed_or_nans_by_dtype_3952(self, device, dtype):
        # Wart pin for the downstream reach of kornia#3952 into this function, companion to its
        # zero-quaternion warning: the all-zero input is not rejected, and what it produces instead
        # SPLITS BY DTYPE, which is the half the warning got wrong before this pin existed.
        #   float64/float32/bfloat16: normalize_quaternion's ‖q‖ < eps clamp absorbs the zero, the
        #     internal rotation comes back as the identity, and the call returns exactly what the
        #     IDENTITY quaternion returns -- a plausible pose, silently, for an input that is not a
        #     rotation at all.
        #   float16: the default eps = 1e-12 underflows to 0 there (bfloat16's wider exponent keeps
        #     it: 1.0018652574217413e-12), so the clamp is a no-op, the normalisation divides 0 by 0
        #     and the whole pose is NaN. Same underflow class as kornia#3966.
        # The first two legs assert against the IDENTITY input's own output rather than against a
        # literal, so the claim the docstring makes -- "the same answer as the identity input" --
        # is what runs, at every dtype and on every backend, with no constant to re-measure per
        # build. The third leg pins the one value the warning quotes outright, t = (-1, 1, 1),
        # which is exact in every dtype here; without it the first two would still pass if BOTH
        # routes moved together.
        # If the non-float16 leg fails, #3952 was (partly) fixed, or this function grew a guard --
        # check which. If the float16 leg fails, the eps reaching normalize_quaternion is no longer
        # underflowing, which is the #3966 half. NOT a contract that either is correct.
        # Snippet used to generate expected (torch only, executed on cpu):
        #   ARKitQTVecs_to_ColmapQTVecs(torch.zeros(1, 4), torch.ones(1, 3, 1))
        #     -> q [0., 1., 0., 0.], t [-1., 1., 1.]         (float64 q_x: 1.0000000012499999)
        #   the same call at float16 -> q [nan] * 4, t [nan] * 3
        _skip_if_dtype_unavailable(device, dtype)
        zeros = torch.zeros(1, 4, device=device, dtype=dtype)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        tvec = torch.tensor(_ARKIT_WORKED_TVEC, device=device, dtype=dtype)

        q_zero, t_zero = ARKitQTVecs_to_ColmapQTVecs(zeros, tvec)

        if dtype == torch.float16:
            assert torch.isnan(q_zero).all() and torch.isnan(t_zero).all(), _issue_msg(
                "kornia#3952/#3966: the float16 zero quaternion no longer underflows to an all-NaN pose"
            )
            return

        q_identity, t_identity = ARKitQTVecs_to_ColmapQTVecs(identity, tvec)

        assert_close(
            q_zero,
            q_identity,
            atol=0.0,
            rtol=0.0,
            msg=_issue_msg("kornia#3952: the zero quaternion no longer gives the identity input's rotation"),
        )
        assert_close(
            t_zero,
            t_identity,
            atol=0.0,
            rtol=0.0,
            msg=_issue_msg("kornia#3952: the zero quaternion no longer gives the identity input's translation"),
        )
        assert_close(
            t_zero,
            torch.tensor([[[-1.0], [1.0], [1.0]]], device=device, dtype=dtype),
            atol=0.0,
            rtol=0.0,
            msg=_issue_msg("kornia#3952: the absorbed zero quaternion no longer produces the documented pose"),
        )

    def test_convention_input_is_camtoworld_graphics_and_output_is_worldtocam_vision(self, device, dtype):
        # Convention pin: the frames, which the docstring never states -- it calls the output "the
        # camera-to-world transformation, expected by Colmap", and the output is world-to-camera.
        # Executed as an equality against the composition of the two conversions this function is
        # built from, each of them pinned separately in this file: the input is a CAMTOWORLD pose in
        # the GRAPHICS frame (y up, -z forward), and the output is a WORLDTOCAM pose in the VISION
        # frame (y down, +z forward), i.e. Colmap's images.txt convention.
        # The two-step reference is written out here rather than compared against a single literal
        # so that the pin names the frames it is asserting; the literals themselves are pinned by
        # test_convention_worked_literal_matches_the_hand_computation.
        _skip_if_dtype_unavailable(device, dtype)
        qvec = torch.tensor(_ARKIT_WORKED_QVEC, device=device, dtype=dtype)
        tvec = torch.tensor(_ARKIT_WORKED_TVEC, device=device, dtype=dtype)

        q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(qvec, tvec)

        camtoworld_graphics_R = kornia.geometry.conversions.quaternion_to_rotation_matrix(qvec)
        camtoworld_vision_R, camtoworld_vision_t = camtoworld_graphics_to_vision_Rt(camtoworld_graphics_R, tvec)
        worldtocam_vision_R, worldtocam_vision_t = camtoworld_to_worldtocam_Rt(camtoworld_vision_R, camtoworld_vision_t)

        self.assert_close(
            kornia.geometry.conversions.quaternion_to_rotation_matrix(q_colmap),
            worldtocam_vision_R,
            atol=1e-5,
            rtol=0.0,
        )
        self.assert_close(t_colmap, worldtocam_vision_t, atol=0.0, rtol=0.0)

    def test_convention_output_shapes_and_per_sample_batching(self, device, dtype):
        # Convention pin: the outputs are q (B, 4) and t (B, 3, 1) -- the translation keeps its
        # trailing singleton axis, guaranteed by an explicit reshape -- and the conversion is
        # per-sample: element 1 of a batched call equals the single-element call, bitwise. The
        # two shape asserts are dtype-invariant, but the bitwise per-sample comparison runs the
        # whole quaternion pipeline on a batched and an unbatched path, so the dtype fixture
        # stays: each dtype's arithmetic could break the equality on its own.
        _skip_if_dtype_unavailable(device, dtype)
        batch_q = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]], device=device, dtype=dtype)
        batch_t = torch.tensor([[[1.0], [1.0], [1.0]], [[1.0], [2.0], [3.0]]], device=device, dtype=dtype)

        q_colmap, t_colmap = ARKitQTVecs_to_ColmapQTVecs(batch_q, batch_t)
        single_q, single_t = ARKitQTVecs_to_ColmapQTVecs(batch_q[1:], batch_t[1:])

        assert q_colmap.shape == (2, 4)
        assert t_colmap.shape == (2, 3, 1)
        self.assert_close(q_colmap[1:], single_q, atol=0.0, rtol=0.0)
        self.assert_close(t_colmap[1:], single_t, atol=0.0, rtol=0.0)

    def test_convention_shapes_are_strictly_batched(self, device):
        # Convention pin: the inputs are strictly qvec (B, 4) and tvec (B, 3, 1) -- an unbatched
        # (4,) quaternion is rejected, and so is a flat (B, 3) translation. The two rejections come
        # from different guards, which the pin keeps visible: both ShapeErrors are raised by
        # camtoworld_graphics_to_vision_Rt's OWN KORNIA_CHECK_SHAPE guards (traceback-verified:
        # ARKitQTVecs_to_ColmapQTVecs -> camtoworld_graphics_to_vision_Rt -> KORNIA_CHECK_SHAPE;
        # Rt_to_matrix4x4 is never reached -- its guards do not back these rejections and cannot be
        # deduplicated on the strength of this pin), while a (B, 3) quaternion is caught even
        # earlier by quaternion_to_rotation_matrix's own "(*, 4)" ValueError.
        # float32 is hardcoded and the dtype fixture dropped because these guards run before any
        # arithmetic and cannot depend on the dtype.
        # Snippet used to generate expected (torch only, executed on cpu float32):
        #   ARKitQTVecs_to_ColmapQTVecs(torch.zeros(4), torch.ones(1, 3, 1))   -> ShapeError
        #   ARKitQTVecs_to_ColmapQTVecs(torch.zeros(1, 4), torch.ones(1, 3))   -> ShapeError
        #   ARKitQTVecs_to_ColmapQTVecs(torch.zeros(1, 3), torch.ones(1, 3, 1))
        #     -> ValueError: Input must be a tensor of shape (*, 4). Got torch.Size([1, 3])
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        translation = torch.ones(1, 3, 1, device=device, dtype=torch.float32)

        with pytest.raises(ShapeError):
            ARKitQTVecs_to_ColmapQTVecs(quaternion[0], translation)
        with pytest.raises(ShapeError):
            ARKitQTVecs_to_ColmapQTVecs(quaternion, translation[..., 0])
        with pytest.raises(ValueError, match=r"shape \(\*, 4\)"):
            ARKitQTVecs_to_ColmapQTVecs(quaternion[:, :3], translation)

    def test_wart_float64_output_quaternion_is_not_unit_3951(self, device):
        # Wart pin for the downstream reach of kornia#3951: this function ends its pipeline with
        # rotation_matrix_to_quaternion, whose eps is added INSIDE the sqrt, so a float64 ARKit call
        # returns a quaternion that is not unit -- a Colmap consumer that validates QW QX QY QZ sees
        # it. The root cause is pinned in TestRotationMatrixToQuaternion
        # (test_convention_returns_a_unit_quaternion_3951 and its companion wart); this cell pins
        # only that the defect reaches the public ARKit entry point, so it flips together with them.
        # The [0, 1, 0, 0] shape of the output is CORRECT, not a component shift: for an identity
        # input the composed map is (I @ diag(1, -1, -1)).T = diag(1, -1, -1), a 180-degree turn
        # about x. The defect is the magnitude alone.
        # If this fails, #3951 was fixed -- remove this pin together with the two in
        # TestRotationMatrixToQuaternion. NOT a contract that the output must stay non-unit.
        # float64 is hardcoded and the dtype fixture dropped because a 1.25e-09 inflation is
        # invisible at float32 and below (the same reason the root-cause pins hardcode it), and the
        # skip is visible so MPS, which has no float64, reports a skip rather than a TypeError.
        # atol 1e-11 sits two orders below the inflation and five above the float64 ulp of 1.0.
        # Snippet used to generate expected (torch only, executed on cpu float64):
        #   ARKitQTVecs_to_ColmapQTVecs(tensor([[1., 0., 0., 0.]], float64), ones(1, 3, 1, float64))
        #     -> q = [0.0, 1.0000000012499999, 0.0, 0.0],  ||q|| - 1 = 1.2499998813808588e-09
        _skip_if_dtype_unavailable(device, torch.float64)

        qvec = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float64)
        tvec = torch.ones(1, 3, 1, device=device, dtype=torch.float64)

        q_colmap, _ = ARKitQTVecs_to_ColmapQTVecs(qvec, tvec)

        assert_close(
            q_colmap,
            torch.tensor([[0.0, 1.0000000012499999, 0.0, 0.0]], device=device, dtype=torch.float64),
            atol=1e-11,
            rtol=0.0,
            msg=_issue_msg("kornia#3951: the float64 ARKit output quaternion is no longer inflated"),
        )
        # No separate ||q|| != 1 assert: the literal above pins ||q|| - 1 = 1.25e-09 at atol=1e-11,
        # which already implies non-unit by two orders -- a second, weaker assert could never fail
        # while the pin holds.


class TestEulerFromQuaternion(BaseTester):
    def test_smoke(self, device, dtype):
        q = Quaternion.random(batch_size=1)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        assert roll.shape == pitch.shape
        assert pitch.shape == yaw.shape

    @pytest.mark.parametrize("batch_size", ((1, 3, 4)))
    def test_cardinality(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size=batch_size)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        assert roll.shape[0] == batch_size
        assert pitch.shape[0] == batch_size
        assert yaw.shape[0] == batch_size

    def test_exception(self, device, dtype):
        q = Quaternion.random(batch_size=2)
        q = q.to(device, dtype)
        with pytest.raises(Exception):
            euler_from_quaternion(q.w, torch.rand(1), q.y, q.z)

    def test_gradcheck(self, device):
        q = Quaternion.random(batch_size=1).to(device, torch.float64)
        self.gradcheck(euler_from_quaternion, (q.w, q.x, q.y, q.z))

    @pytest.mark.skipif(
        torch_version() in {"2.0.1", "2.1.2", "2.2.2", "2.3.1"} and sys.version_info.minor == 8,
        reason="Not working on 2.0",
    )
    def test_dynamo(self, device, dtype, torch_optimizer):
        q = Quaternion.random(batch_size=1)
        q = q.to(device, dtype)
        op = euler_from_quaternion
        op_optimized = torch_optimizer(op)
        self.assert_close(op(q.w, q.x, q.y, q.z), op_optimized(q.w, q.x, q.y, q.z))

    def test_forth_and_back(self, device, dtype):
        q = Quaternion.random(batch_size=2)
        q = q.to(device, dtype)
        roll, pitch, yaw = euler_from_quaternion(q.w, q.x, q.y, q.z)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        # TODO: check hwo to prevent getting inverted angles sometimes
        self.assert_close(q.w.abs(), qw.abs())
        self.assert_close(q.x.abs(), qx.abs())
        self.assert_close(q.y.abs(), qy.abs())
        self.assert_close(q.z.abs(), qz.abs())

    def test_convention_roll_is_x_pitch_is_y_yaw_is_z(self, device, dtype):
        # Convention pin: the three returned angles are (roll, pitch, yaw) in that order, and they
        # are rotations about x, y and z respectively -- a rotation about a single axis puts its
        # angle in exactly one slot and leaves the other two at zero, which no permutation of the
        # naming could reproduce. The return is a TUPLE of three separate tensors, not a stacked
        # (*, 3) tensor, so it cannot be indexed or sliced like one; that is pinned first.
        # The angle is 0.6 rad rather than a quarter turn so the pin stays far from the pitch =
        # +-pi/2 gimbal lock where this function does not recover the input at all.
        # Snippet used to generate the inputs (stdlib only):
        #   import math
        #   for each axis: q = (cos(0.3), sin(0.3) * axis) with 0.3 = theta / 2
        #     cos(0.3), sin(0.3) -> (0.955336489125606, 0.29552020666133955)
        w = torch.tensor(0.955336489125606, device=device, dtype=dtype)
        s = torch.tensor(0.29552020666133955, device=device, dtype=dtype)
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        expected_angle = torch.tensor(0.6, device=device, dtype=dtype)

        about_x = euler_from_quaternion(w, s, zero, zero)
        assert isinstance(about_x, tuple)
        assert len(about_x) == 3
        self.assert_close(about_x[0], expected_angle)
        self.assert_close(about_x[1], zero)
        self.assert_close(about_x[2], zero)

        about_y = euler_from_quaternion(w, zero, s, zero)
        self.assert_close(about_y[0], zero)
        self.assert_close(about_y[1], expected_angle)
        self.assert_close(about_y[2], zero)

        about_z = euler_from_quaternion(w, zero, zero, s)
        self.assert_close(about_z[0], zero)
        self.assert_close(about_z[1], zero)
        self.assert_close(about_z[2], expected_angle)

    def test_convention_euler_and_quaternion_are_mutual_inverses(self, device, dtype):
        # Convention pin: away from gimbal lock, euler_from_quaternion and quaternion_from_euler
        # invert each other exactly -- the same three angles come back, with their signs, and so
        # do the same four quaternion coefficients. Pinned at |pitch| = 0.7 < pi/4-ish and three
        # distinct non-symmetric angles so neither a permutation nor a sign flip survives. (At
        # pitch = +-pi/2 the pair is NOT a mutual inverse; that failure is out of scope here.)
        # Snippet used to generate expected (stdlib only):
        #   the round-trip is the identity on (roll, pitch, yaw) = (0.3, 0.7, 1.1)
        #   quaternion_from_euler(0.3, 0.7, 1.1) at float64 ->
        #     [0.8186292656554958, -0.057539988180335386, 0.3624200943552256, 0.44179967222724353]
        #   which is qz (x) qy (x) qx with qa = (cos(a/2), sin(a/2) * axis) -- see
        #   TestQuaternionFromEuler.test_convention_composition_is_rz_ry_rx
        roll = torch.tensor(0.3, device=device, dtype=dtype)
        pitch = torch.tensor(0.7, device=device, dtype=dtype)
        yaw = torch.tensor(1.1, device=device, dtype=dtype)

        quaternion = quaternion_from_euler(roll, pitch, yaw)
        roll_back, pitch_back, yaw_back = euler_from_quaternion(*quaternion)

        self.assert_close(roll_back, roll)
        self.assert_close(pitch_back, pitch)
        self.assert_close(yaw_back, yaw)

        quaternion_back = quaternion_from_euler(roll_back, pitch_back, yaw_back)
        for component, component_back in zip(quaternion, quaternion_back):
            self.assert_close(component_back, component)

    @pytest.mark.parametrize(
        ("zero_sign", "expected_roll"),
        [(-0.0, -torch.pi), (0.0, torch.pi)],
        ids=["negative_zero_gives_minus_pi", "positive_zero_gives_plus_pi"],
    )
    def test_convention_roll_range_is_closed_with_signed_zero_endpoint(self, device, zero_sign, expected_roll):
        # Convention pin for the CLOSED [-pi, pi] range documented on euler_from_quaternion: roll
        # comes from atan2, which returns the +-pi endpoint EXACTLY, its sign taken from the
        # signed zero of the first argument. The input is the half-turn about x, where that
        # argument is 2 * (w*x + y*z) = +-0.0 tracking the sign of w, and the second argument is
        # 1 - 2 * (x**2 + y**2) = -1 exactly, so IEEE 754 atan2(+-0.0, -1.0) mandates the result:
        # a bitwise fact, not a rounding accident. A range check written from the half-open
        # (-pi, pi] form (roll > -pi) fails on the negative-zero input. float64 is hardcoded and
        # the dtype fixture dropped because the docstring quotes the float64 endpoints; the
        # signed-zero sign rule itself is dtype-independent.
        # Snippet used to generate expected (torch only, executed on cpu float64):
        #   euler_from_quaternion(w=-0.0, x=1.0, y=0.0, z=-0.0)[0] -> -3.141592653589793 == -pi
        #   euler_from_quaternion(w=0.0,  x=1.0, y=0.0, z=0.0)[0]  ->  3.141592653589793 == +pi
        _skip_if_dtype_unavailable(device, torch.float64)

        w = torch.tensor([zero_sign], device=device, dtype=torch.float64)
        x = torch.tensor([1.0], device=device, dtype=torch.float64)
        y = torch.tensor([0.0], device=device, dtype=torch.float64)
        z = torch.tensor([zero_sign], device=device, dtype=torch.float64)

        roll, _, _ = euler_from_quaternion(w, x, y, z)

        assert roll.item() == expected_roll, (
            f"roll for the w = {zero_sign} half-turn is {roll.item()}, not the exact {expected_roll} endpoint"
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="euler_from_quaternion has no gimbal-lock branch, so at pitch = ±pi/2 the returned "
        "triple does not represent the input rotation — kornia#3950",
        strict=True,
    )
    def test_convention_roundtrip_holds_at_gimbal_lock_3950(self, device):
        # Intended behavior: euler_from_quaternion returns *a* triple representing the input
        # rotation. At pitch = ±pi/2 (gimbal lock) roll and yaw are individually undetermined --
        # only their sum or difference is -- so no library can return the input triple back, but a
        # correct implementation still returns a triple whose rotation matrix is the input's, which
        # is what this pin asserts. There is no gimbal-lock branch at all: roll and yaw come from
        # atan2 of two quantities that both cancel there, and the result is simply wrong. For
        # (roll, pitch, yaw) = (0.1, pi/2, 0.2) in float64 the reconstructed rotation is far from
        # the input -- by a margin that varies with rounding, so no figure is quoted here -- and
        # random (roll, yaw) at pitch = +pi/2 fail the same way, while |pitch| < pi/4 round trips
        # to rounding. float64 is hardcoded and the dtype fixture
        # dropped because the returned triple is wildly dtype-dependent here (see the companion
        # wart), and the skip is visible so a raw TypeError on MPS, which has no float64, cannot
        # satisfy the raises=AssertionError mark instead of the assertion. Marked xfail(strict=True)
        # so fixing #3950 makes this XPASS and forces the mark out. Companion wart:
        # test_wart_gimbal_lock_returns_a_wrong_triple_3950.
        _skip_if_dtype_unavailable(device, torch.float64)

        roll = torch.tensor(0.1, device=device, dtype=torch.float64)
        pitch = torch.tensor(torch.pi / 2, device=device, dtype=torch.float64)
        yaw = torch.tensor(0.2, device=device, dtype=torch.float64)

        quaternion = quaternion_from_euler(roll, pitch, yaw)
        roundtrip = quaternion_from_euler(*euler_from_quaternion(*quaternion))

        rot_in = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.stack(quaternion))
        rot_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.stack(roundtrip))
        assert (rot_in - rot_back).abs().max().item() < 1e-12, (
            "kornia#3950: the euler triple returned at pitch = pi/2 does not represent the input rotation"
        )

    @pytest.mark.parametrize("sign", [1.0, -1.0], ids=["pitch_plus_pi_over_2", "pitch_minus_pi_over_2"])
    def test_wart_gimbal_lock_returns_a_wrong_triple_3950(self, device, sign):
        # Wart pin for kornia#3950, companion to the strict xfail above. It pins the two facts about
        # gimbal lock that are STABLE, and deliberately pins no exact triple.
        #
        # Pinning the returned triples themselves ((pi/2, pi/2, pi/2) at +pi/2 and (0, -pi/2, pi/2)
        # at -pi/2 on this build) is not an option: those values are not reproducible. roll and yaw
        # come from atan2 of two quantities that cancel to ~1e-17 there, so which way they
        # cancel is decided by rounding. Perturbing the input pitch by a single ulp on this very
        # build changes the +pi/2 triple to (0.15500, pi/2, 0.35877) at -2 ulp and to
        # (-3.12597, pi/2, -3.07917) at +1 ulp; a review on torch 2.12.0 saw different triples again
        # on the unperturbed input. Kornia declares torch>=2.0.0, so pinning any one of them makes
        # the suite red on builds the pin was never measured against, for a value that is not the
        # defect being tracked.
        #
        # What this pin asserts instead:
        #   1. pitch_back is +-pi/2 to a tolerance -- the asin saturates. Note this fact SURVIVES a
        #      fix to #3950 (a correct gimbal-lock branch still reports pitch = +-pi/2); it is
        #      pinned as the structural claim the strict xfail above does not make, not as a defect
        #      indicator.
        #   2. the round-tripped rotation is far from the input -- this is the defect, and the half
        #      of this pin that flips when #3950 is fixed.
        #
        # Assertion 1 is a TOLERANCE and not exact equality, which matters. At gimbal lock the asin
        # argument is 1 - O(eps), and asin(1 - d) ~= pi/2 - sqrt(2d), so one ulp of slack in the
        # argument amplifies to a sqrt-scale error in the output: sqrt(2 * eps_f64) = 2.107e-08.
        # Whether the argument rounds to exactly 1.0 or to one ulp below is decided by the last bit
        # of the sin/cos computation upstream, so it moves between torch builds, backends and
        # vectorisation paths. torch 2.12.0 reports -1.5707963057214724 for the -pi/2 cell, which is
        # pi/2 - 2.1073424116835326e-08 -- agreeing with sqrt(2 * eps) to eight significant figures.
        # Exact equality here is therefore red on 2.12; the tolerance is what keeps the cell green.
        # Reproducing the 2.12 value locally needs the right probe: perturbing the input pitch, or any component
        # by a single ulp, does NOT move pitch_back at all (a +-1 ulp sweep over all four quaternion
        # components returns one distinct value, as does a +-200 ulp input-pitch sweep). Perturbing
        # w -- the component the saturation actually depends on -- by two or more ulps walks up the
        # same sqrt-scale family and reproduces the 2.12.0 figure bit-for-bit: w - 2 ulp gives
        # -1.5707963057214724 on the -pi/2 cell, and w - 3 ulp gives +1.5707963057214724 on the
        # +pi/2 cell. The rule for the next pin here is therefore to perturb the intermediate the
        # branch depends on, and to go wider than one ulp -- not to assume the input is the probe.
        # Tolerance sizing stays mechanism-based rather than sampled: dev = sqrt(2 * k * eps) for an
        # argument k ulps below 1.0, so 1e-6 is only reached at k ~ 2250, far beyond the one-to-few
        # ulps a cross-build rounding difference can move it, and far below any real defect, which
        # would move pitch by O(1). Note the deviation is NOT bounded by the values above: pushing w
        # to -20000 ulp reaches 2.5e-06 and does cross the tolerance. That is not a realistic
        # rounding difference, but it is why the sizing argument is the mechanism and the measured
        # figures here (2.1e-08 at w - 2 ulp, 5.4e-08 at w - 10 ulp) are sample points, not bounds.
        #
        # The 1e-9 floor in 2 is likewise a chosen threshold with margin, NOT a measured bound:
        # a correct gimbal-lock branch round trips to ~1e-16, and widening the probe drives the
        # sample minimum steadily toward 0, which is why no sampled extremum is quoted as a bound.
        #
        # Two cells, one per sign, because a fix could plausibly add a gimbal-lock branch for one
        # sign only or get the roll/yaw split sign wrong; the strict xfail above only covers +pi/2,
        # so the -pi/2 cell here is the only coverage of that sign. If either cell fails, #3950 was
        # (partly) fixed -- flip/remove the strict xfail above. NOT a contract that the current
        # output is correct: any triple whose rotation matrix matches the input is an acceptable
        # replacement, and such a triple would fail assertion 2 as intended.
        # float64 is hardcoded and the dtype fixture dropped because the round-trip margin is a
        # float64 fact; the skip is visible so a raw TypeError on MPS, which has no float64, cannot
        # pass for the assertion.
        _skip_if_dtype_unavailable(device, torch.float64)

        pitch_in = sign * torch.pi / 2
        quaternion = quaternion_from_euler(
            torch.tensor(0.1, device=device, dtype=torch.float64),
            torch.tensor(pitch_in, device=device, dtype=torch.float64),
            torch.tensor(0.2, device=device, dtype=torch.float64),
        )

        roll_back, pitch_back, yaw_back = euler_from_quaternion(*quaternion)

        assert abs(pitch_back.item() - pitch_in) < 1e-6, (
            f"kornia#3950: pitch no longer saturates to {pitch_in} at gimbal lock (got {pitch_back.item()!r})"
        )

        roundtrip = quaternion_from_euler(roll_back, pitch_back, yaw_back)
        rot_in = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.stack(quaternion))
        rot_back = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.stack(roundtrip))
        error = (rot_in - rot_back).abs().max().item()

        assert error > 1e-9, (
            f"kornia#3950: the triple returned at pitch = {pitch_in} now reproduces the input "
            f"rotation to {error} -- the gimbal-lock defect looks fixed"
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="euler_from_quaternion does not normalise its input, so a non-unit quaternion gives "
        "a silently wrong triple — kornia#3953",
        strict=True,
    )
    def test_convention_euler_from_quaternion_normalizes_its_input_3953(self, device, dtype):
        # Intended behavior: the euler angles of a quaternion depend only on the rotation it
        # represents, so rescaling the quaternion must not change them -- which is what the
        # scale-safe siblings do (quaternion_to_rotation_matrix normalises internally and returns a
        # bit-identical matrix for 2q; quaternion_to_axis_angle is homogeneous by construction).
        # euler_from_quaternion does not normalise and does not check: feeding 2q returns
        # [1.6560585860248003, 1.5707963267948966, 2.1048169977173687] instead of the (0.3, 0.7,
        # 1.1) the unit quaternion gives -- and note the middle component, which is exactly pi/2:
        # the unnormalised argument saturates the asin, so a merely-scaled input is reported as
        # gimbal-locked. Marked xfail(strict=True) so fixing #3953 makes this XPASS and forces the
        # mark out. Companion wart: test_wart_euler_from_quaternion_ignores_the_norm_3953; the
        # quaternion_exp_to_log half of the same issue is pinned in TestQuaternionExpToLog.
        roll = torch.tensor(0.3, device=device, dtype=dtype)
        pitch = torch.tensor(0.7, device=device, dtype=dtype)
        yaw = torch.tensor(1.1, device=device, dtype=dtype)
        quaternion = quaternion_from_euler(roll, pitch, yaw)

        out = torch.stack(euler_from_quaternion(*[2.0 * component for component in quaternion]))

        assert_close(
            out,
            torch.stack((roll, pitch, yaw)),
            msg=_issue_msg("kornia#3953: euler_from_quaternion did not normalise its input"),
        )

    def test_wart_euler_from_quaternion_ignores_the_norm_3953(self, device, dtype):
        # Wart pin for kornia#3953, companion to the strict xfail above: assert the CURRENT triple
        # for a scaled-up quaternion. This is a separate cell from the quaternion_exp_to_log cells
        # in TestQuaternionExpToLog because the two functions are independent code paths and a fix
        # to one leaves the other broken, which would leave the other strict xfail silently XFAIL.
        # If it fails, the euler half of #3953 was fixed -- flip/remove the strict xfail above.
        # NOT a contract that a scaled quaternion must keep producing this triple.
        # Snippet used to generate expected (torch only, executed on cpu float64):
        #   t = lambda x: torch.tensor(x, dtype=torch.float64)
        #   q = quaternion_from_euler(t(0.3), t(0.7), t(1.1))
        #     -> [0.8186292656554958, -0.057539988180335386, 0.3624200943552256, 0.44179967222724353]
        #   [x.item() for x in euler_from_quaternion(*q)]
        #     -> [0.2999999999999999, 0.6999999999999998, 1.0999999999999999]     (the unit input)
        #   [x.item() for x in euler_from_quaternion(*[2 * c for c in q])]
        #     -> [1.6560585860248003, 1.5707963267948966, 2.1048169977173687]
        #   (float32: [1.656058430671692, 1.5707963705062866, 2.1048169136047363];
        #    float16:  [1.6572265625, 1.5703125, 2.10546875];
        #    bfloat16: [1.6484375, 1.5703125, 2.109375] -- all within the dtype's default tolerance
        #    of the float64 literals below)
        quaternion = quaternion_from_euler(
            torch.tensor(0.3, device=device, dtype=dtype),
            torch.tensor(0.7, device=device, dtype=dtype),
            torch.tensor(1.1, device=device, dtype=dtype),
        )

        out = torch.stack(euler_from_quaternion(*[2.0 * component for component in quaternion]))

        assert_close(
            out,
            torch.tensor([1.6560585860248003, 1.5707963267948966, 2.1048169977173687], device=device, dtype=dtype),
            msg=_issue_msg("kornia#3953: euler_from_quaternion no longer ignores the quaternion norm"),
        )


class TestQuaternionFromEuler(BaseTester):
    def test_smoke(self, device, dtype):
        roll, pitch, yaw = torch.rand(3, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        assert qw.shape == qx.shape
        assert qx.shape == qy.shape
        assert qy.shape == qz.shape

    @pytest.mark.parametrize("batch_size", ((1, 3, 4)))
    def test_cardinality(self, device, dtype, batch_size):
        roll, pitch, yaw = torch.rand(3, batch_size, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        assert qw.shape[0] == batch_size
        assert qx.shape[0] == batch_size
        assert qy.shape[0] == batch_size
        assert qz.shape[0] == batch_size

    def test_exception(self, device, dtype):
        _, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)
        with pytest.raises(Exception):
            quaternion_from_euler(torch.rand(1), pitch, yaw)

    def test_gradcheck(self, device):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(quaternion_from_euler, (roll, pitch, yaw))

    def test_dynamo(self, device, dtype, torch_optimizer):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)

        op = quaternion_from_euler
        op_optimized = torch_optimizer(op)

        actual = op_optimized(roll, pitch, yaw)
        expected = op(roll, pitch, yaw)

        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1])
        self.assert_close(actual[2], expected[2])

    def test_forth_and_back(self, device, dtype):
        roll, pitch, yaw = torch.rand(3, 2, device=device, dtype=dtype)
        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        roll_new, pitch_new, yaw_new = euler_from_quaternion(qw, qx, qy, qz)
        self.assert_close(roll, roll_new)
        self.assert_close(pitch, pitch_new)
        self.assert_close(yaw, yaw_new)

    def test_values(self, device, dtype):
        # num_samples = 5
        # data = 2 * torch.rand(3, num_samples, device=device, dtype=dtype) - 1
        # roll, pitch, yaw = torch.pi * data
        roll = torch.tensor(
            [2.6518599987, 0.0612506270, 1.2417907715, 2.8829660416, -1.9961174726], device=device, dtype=dtype
        )

        pitch = torch.tensor(
            [2.3267219067, -2.7309591770, -1.4011553526, -2.1962766647, 2.1454355717], device=device, dtype=dtype
        )

        yaw = torch.tensor(
            [-0.8856627345, 0.2605336905, 0.4579202533, -1.3095731735, 0.6096843481], device=device, dtype=dtype
        )

        euler_expected = torch.tensor(
            [
                [-0.4897327125, 0.8148705959, 2.2559301853],
                [-3.0803420544, -0.4106334746, -2.8810589314],
                [1.2417914867, -1.4011553526, 0.4579201937],
                [-0.2586266696, -0.9453159571, 1.8320195675],
                [1.1454752684, 0.9961569905, -2.5319085121],
            ],
            device=device,
            dtype=dtype,
        )

        qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
        euler = euler_from_quaternion(qw, qx, qy, qz)
        euler = torch.stack(euler, -1)

        self.assert_close(euler, euler_expected, 1e-4, 1e-4)

        # this test is passing: pip install transforms3d
        # import transforms3d as tf3
        # out = [tf3.euler.euler2quat(roll[i], pitch[i], yaw[i]) for i in range(num_samples)]
        # out = torch.tensor(out, device=device, dtype=dtype)
        # self.assert_close(torch.stack((qw, qx, qy, qz), -1), out)

        # out = [tf3.euler.quat2euler((qw[i], qx[i], qy[i], qz[i])) for i in range(num_samples)]
        # out = torch.tensor(out, device=device, dtype=dtype)

    def test_convention_composition_is_rz_ry_rx(self, device, dtype):
        # Convention pin: "XYZ convention" in the docstring does not say whether the three
        # rotations are applied about the fixed axes or about the axes carried along by the body,
        # and the four candidate products differ enormously. The actual composition is
        #   R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        # i.e. extrinsic X -> Y -> Z about the FIXED axes (equivalently intrinsic Z-Y'-X''), with
        # roll about x, pitch about y and yaw about z. Three distinct non-symmetric angles are
        # required: a symmetric or single-axis input cannot separate the four candidates. Measured
        # max |R - candidate| at (0.3, 0.7, 1.1) in float64:
        #   Rz@Ry@Rx 0.0 (1.11e-16 when the product is built from math.cos/math.sin literals),
        #   Rx@Ry@Rz 0.6404683155788216, Ry@Rz@Rx 0.5484888138736672, Rx@Rz@Ry 0.2503184512807922.
        # The three rejected products are asserted to stay above 0.2 so the discrimination itself
        # is executable rather than a claim in a comment; at bfloat16, the coarsest dtype run, the
        # accepted product is still within 3.90625e-03 and the nearest rejected one at 0.25.
        # The return is a TUPLE of four separate tensors, not a stacked (*, 4) tensor; pinned first.
        # Snippet used to generate the elementary matrices (stdlib only):
        #   import math
        #   math.cos(0.3), math.sin(0.3) -> (0.955336489125606, 0.29552020666133955)
        #   math.cos(0.7), math.sin(0.7) -> (0.7648421872844885, 0.644217687237691)
        #   math.cos(1.1), math.sin(1.1) -> (0.4535961214255773, 0.8912073600614354)
        rot_x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.955336489125606, -0.29552020666133955],
                [0.0, 0.29552020666133955, 0.955336489125606],
            ],
            device=device,
            dtype=dtype,
        )
        rot_y = torch.tensor(
            [
                [0.7648421872844885, 0.0, 0.644217687237691],
                [0.0, 1.0, 0.0],
                [-0.644217687237691, 0.0, 0.7648421872844885],
            ],
            device=device,
            dtype=dtype,
        )
        rot_z = torch.tensor(
            [
                [0.4535961214255773, -0.8912073600614354, 0.0],
                [0.8912073600614354, 0.4535961214255773, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )

        quaternion = quaternion_from_euler(
            torch.tensor(0.3, device=device, dtype=dtype),
            torch.tensor(0.7, device=device, dtype=dtype),
            torch.tensor(1.1, device=device, dtype=dtype),
        )
        assert isinstance(quaternion, tuple)
        assert len(quaternion) == 4

        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.stack(quaternion))

        self.assert_close(rot, rot_z @ rot_y @ rot_x)

        assert (rot - rot_x @ rot_y @ rot_z).abs().max() > 0.2
        assert (rot - rot_y @ rot_z @ rot_x).abs().max() > 0.2
        assert (rot - rot_x @ rot_z @ rot_y).abs().max() > 0.2


@pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
def test_vector_to_skew_symmetric_matrix(batch_size, device, dtype):
    if batch_size is None:
        vector = torch.rand(3, device=device, dtype=dtype)
    else:
        vector = torch.rand((batch_size, 3), device=device, dtype=dtype)
    skew_symmetric_matrix = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(vector)
    assert skew_symmetric_matrix.shape[-1] == 3
    assert skew_symmetric_matrix.shape[-2] == 3
    z = torch.zeros_like(vector[..., 0])
    assert_close(skew_symmetric_matrix[..., 0, 0], z)
    assert_close(skew_symmetric_matrix[..., 1, 1], z)
    assert_close(skew_symmetric_matrix[..., 2, 2], z)
    assert_close(skew_symmetric_matrix[..., 0, 1], -vector[..., 2])
    assert_close(skew_symmetric_matrix[..., 1, 0], vector[..., 2])
    assert_close(skew_symmetric_matrix[..., 0, 2], vector[..., 1])
    assert_close(skew_symmetric_matrix[..., 2, 0], -vector[..., 1])
    assert_close(skew_symmetric_matrix[..., 1, 2], -vector[..., 0])
    assert_close(skew_symmetric_matrix[..., 2, 1], vector[..., 0])

    # Convention's enforcement point: [v]x @ x == cross(v, x) -- the vector is the LEFT factor
    # of the cross product, NOT cross(x, v), which is the negation.
    # Snippet used to generate expected (stdlib only):
    #   v, x = (1, 2, 3), (4, 5, 6)
    #   cross(v, x) = (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) -> (-3, 6, -3)
    v = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
    x = torch.tensor([4.0, 5.0, 6.0], device=device, dtype=dtype)
    skew = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(v)
    expected_cross = torch.tensor([-3.0, 6.0, -3.0], device=device, dtype=dtype)
    assert_close(skew @ x, expected_cross)


class TestAxisAngleToRotationMatrix:
    def test_identity_rotation(self):
        aa = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
        R = axis_angle_to_rotation_matrix(aa)
        Id = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        assert torch.allclose(R, Id, atol=1e-6)

    def test_90deg_x_axis(self):
        aa = torch.tensor([[torch.pi / 2, 0.0, 0.0]], dtype=torch.float64)
        R = axis_angle_to_rotation_matrix(aa).squeeze(0)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(R, expected, atol=1e-6)

    def test_180deg_y_axis(self):
        aa = torch.tensor([[0.0, torch.pi, 0.0]], dtype=torch.float64)
        R = axis_angle_to_rotation_matrix(aa).squeeze(0)
        expected = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(R, expected, atol=1e-6)

    def test_batched_input(self):
        aa = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [torch.pi / 2, 0.0, 0.0],
                [0.0, torch.pi, 0.0],
            ],
            dtype=torch.float64,
        )
        R = axis_angle_to_rotation_matrix(aa)
        assert R.shape == (3, 3, 3)


# Module-level pins for kornia#3956: the four deprecated aliases of this module
# (angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis, quaternion_to_angle_axis,
# angle_axis_to_quaternion) were never the site of the defect -- the root cause was
# _emit_deprecation_warning in kornia/core/_compat.py, which wrapped its warnings.warn call in
# warnings.simplefilter("always", DeprecationWarning) / simplefilter("default", DeprecationWarning)
# and so rewrote the PROCESS-GLOBAL filter list on every call. Every deprecated symbol in kornia
# was therefore affected, which is why these live at module level rather than in one symbol's
# class, following the module-level wart precedent above; the emitter itself is pinned in
# tests/utils/test_deprecated.py, and these four cells are what keeps the convention attached to
# the aliases rather than to one caller of the decorator. Both pins run inside
# warnings.catch_warnings() so that a regression's filter mutation cannot leak into the rest of the
# suite: that bug was contagious across tests, and an unisolated pin would silently disarm every
# other test's warning discipline.
# Both pins parametrize over _DEPRECATED_ALIAS_NAMES_AND_ARGS, the projection of the module-level
# _DEPRECATED_ALIASES table down to the two columns they use -- neither is about what the alias
# forwards TO, only about the warning it emits on the way -- so both the list of aliases and the
# projection of it are maintained in exactly one place, and neither signature carries a parameter
# it never reads.


@pytest.mark.parametrize(("alias_name", "arg"), _DEPRECATED_ALIAS_NAMES_AND_ARGS, ids=_DEPRECATED_ALIAS_IDS)
def test_convention_deprecated_alias_warning_can_be_escalated_to_an_error_3956(alias_name, arg):
    # Convention: a DeprecationWarning emitted by kornia obeys the caller's warning filters, so a
    # project running under -W error::DeprecationWarning (or pytest's filterwarnings = error) sees
    # the call fail and can find its deprecated usages. Until kornia#3956 was fixed it did not:
    # _emit_deprecation_warning installed simplefilter("always", DeprecationWarning) immediately
    # before warnings.warn, which overrode the caller's "error" entry, so the warning was printed
    # and execution continued.
    # The escalated DeprecationWarning is caught by type rather than through the shared
    # _runs_without_raising helper: that helper treats *any* exception as the awaited raise, so an
    # unrelated TypeError from the alias would set escalated=True and pass the body under a name
    # that reads as "escalation works". Catching DeprecationWarning specifically lets any other
    # exception propagate and be reported as an error instead. (The #3955 call sites keep the broad
    # helper on purpose: there an unrelated exception makes the assertion *fail*, which is already
    # the correct report.)
    # Four cells, one per alias, because all four are separate @deprecated call sites: the fix is
    # in _emit_deprecation_warning, but a future decorator that emits its own warning would have to
    # honor this too.
    alias = getattr(kornia.geometry.conversions, alias_name)
    tensor = torch.tensor(arg)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        try:
            alias(tensor)
        except DeprecationWarning:
            escalated = True
        else:
            escalated = False

    assert escalated, f"kornia#3956: {alias_name} did not raise under simplefilter('error', DeprecationWarning)"


@pytest.mark.parametrize(("alias_name", "arg"), _DEPRECATED_ALIAS_NAMES_AND_ARGS, ids=_DEPRECATED_ALIAS_IDS)
def test_convention_deprecated_alias_leaves_the_global_warning_filters_alone_3956(alias_name, arg):
    # The other half of kornia#3956, and the half the escalation pin above cannot see: a call must
    # leave warnings.filters exactly as it found it. Before the fix a single alias call pushed two
    # entries of its own -- 'always' before the warn and 'default' from the finally clause -- so
    # even a caller that never escalated had its process-global warning config rewritten, and the
    # 'default' entry outlived the call. Pinned separately because a half fix that dropped only the
    # "always" would restore escalation while still clobbering the caller's filters.
    # Starts from an EMPTY filter list rather than the ambient one so the assertion is exact rather
    # than a length comparison, and runs inside warnings.catch_warnings() so neither the reset nor
    # any mutation a regression reintroduces can leak into the rest of the suite.
    alias = getattr(kornia.geometry.conversions, alias_name)
    tensor = torch.tensor(arg)

    with warnings.catch_warnings():
        warnings.resetwarnings()
        assert warnings.filters == [], "the filter list was not empty before the call, so the pin below is not clean"

        alias(tensor)

        after = list(warnings.filters)

    assert after == [], f"kornia#3956: {alias_name} mutated the global DeprecationWarning filters; got {after}"

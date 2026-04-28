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
"""Tests for AugmentationSequential.compile() — PR-G4.

Smoke-tests for pipeline-wide torch.compile opt-in.  The tests intentionally
avoid asserting on exact numerical outputs because torch.compile may alter
floating-point order-of-operations within the allowed tolerance.  Shape and
call-count stability are the primary invariants.

Known compile-time limitations today (pre PR-G1):
  - ColorJiggle uses a data-dependent list index (transforms[idx]) which
    causes GuardOnDataDependentSymNode with the Inductor backend.
  - int(to_apply.sum().item()) in base.py triggers a graph break.

These are documented as xfail where relevant.  Tests that must pass use
backend="eager" which bypasses Inductor's symbolic-shape analysis.
"""

from __future__ import annotations

import pytest
import torch

import kornia.augmentation as K

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_simple_pipeline(device: torch.device) -> K.AugmentationSequential:
    """HFlip + ColorJiggle + Normalize — a common image-classification pipeline.

    Note: ColorJiggle uses a data-dependent list index that prevents the Inductor
    backend from compiling without graph breaks.
    """
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
        K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406], device=device),
            std=torch.tensor([0.229, 0.224, 0.225], device=device),
        ),
    ).to(device)


def _make_deterministic_pipeline(device: torch.device) -> K.AugmentationSequential:
    """Normalize only — deterministic, no random state, friendlier to fullgraph=True."""
    return K.AugmentationSequential(
        K.Normalize(
            mean=torch.tensor([0.5, 0.5, 0.5], device=device),
            std=torch.tensor([0.2, 0.2, 0.2], device=device),
        ),
    ).to(device)


def _make_hflip_normalize_pipeline(device: torch.device) -> K.AugmentationSequential:
    """HFlip + Normalize — no data-dependent indexing, cleaner for compile tests."""
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0),
        K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406], device=device),
            std=torch.tensor([0.229, 0.224, 0.225], device=device),
        ),
    ).to(device)


# ---------------------------------------------------------------------------
# Case 1: compile() returns a callable
# ---------------------------------------------------------------------------


def test_compile_returns_callable():
    """compile() must return something callable — regardless of CUDA availability."""
    device = torch.device("cpu")
    aug = _make_simple_pipeline(device)
    compiled = aug.compile()
    assert callable(compiled), "compile() must return a callable"


# ---------------------------------------------------------------------------
# Case 2: output shape matches eager — backend="eager" avoids Inductor issues
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_str",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))],
)
def test_compile_output_shape_matches_eager(device_str):
    """Compiled pipeline (backend=eager) output shape must match the eager forward pass."""
    device = torch.device(device_str)
    aug = _make_hflip_normalize_pipeline(device)

    x = torch.rand(2, 3, 64, 64, device=device)

    eager_out = aug(x.clone())
    eager_shape = eager_out.shape if isinstance(eager_out, torch.Tensor) else eager_out[0].shape

    # backend="eager" bypasses Inductor; avoids symbolic-shape issues on CPU
    compiled = aug.compile(backend="eager")
    compiled_out = compiled(x.clone())
    compiled_shape = compiled_out.shape if isinstance(compiled_out, torch.Tensor) else compiled_out[0].shape

    assert compiled_shape == eager_shape, f"Shape mismatch: compiled {compiled_shape} vs eager {eager_shape}"


# ---------------------------------------------------------------------------
# Case 3: second call does not error (kernel is cached / reused)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_str",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))],
)
def test_compile_second_call_stable(device_str):
    """Calling the compiled pipeline twice on identically-shaped inputs must not error."""
    device = torch.device(device_str)
    aug = _make_hflip_normalize_pipeline(device)
    compiled = aug.compile(backend="eager")

    x = torch.rand(2, 3, 64, 64, device=device)

    out1 = compiled(x)
    out2 = compiled(x)

    shape1 = out1.shape if isinstance(out1, torch.Tensor) else out1[0].shape
    shape2 = out2.shape if isinstance(out2, torch.Tensor) else out2[0].shape

    assert shape1 == shape2, "Second call produced different output shape"


# ---------------------------------------------------------------------------
# Case 4: CPU smoke test with backend="eager" — always runs, no CUDA needed
# ---------------------------------------------------------------------------


def test_compile_cpu_smoke():
    """Pipeline-wide compile with backend=eager must not raise on CPU."""
    device = torch.device("cpu")
    aug = _make_simple_pipeline(device)
    compiled = aug.compile(fullgraph=False, mode="default", dynamic=False, backend="eager")
    x = torch.rand(1, 3, 32, 32)
    out = compiled(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Case 5: default fullgraph=False (most lenient), backend="eager"
# ---------------------------------------------------------------------------


def test_compile_default_args_eager_backend():
    """compile() with backend=eager and default fullgraph=False must succeed."""
    device = torch.device("cpu")
    aug = _make_hflip_normalize_pipeline(device)
    compiled = aug.compile(backend="eager")
    x = torch.rand(2, 3, 48, 48)
    out = compiled(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Case 5b: default args with Inductor — documents current known failure mode
#
# ColorJiggle uses `transforms[idx]` with a data-dependent idx which triggers
# GuardOnDataDependentSymNode in Inductor.  This is the exact failure the
# PR-G1 / augmentation refactor work needs to resolve.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Inductor backend fails on ColorJiggle's data-dependent list indexing "
        "(transforms[idx]) — GuardOnDataDependentSymNode. "
        "Will pass once color_jitter.py is refactored per PR-G1."
    ),
)
def test_compile_inductor_colojiggle_known_issue():
    """Documents the known GuardOnDataDependentSymNode failure with ColorJiggle + Inductor."""
    device = torch.device("cpu")
    aug = _make_simple_pipeline(device)
    # No backend="eager" override — use the default Inductor backend
    compiled = aug.compile(fullgraph=False)
    x = torch.rand(2, 3, 48, 48)
    out = compiled(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Case 6: fullgraph=True on a deterministic (Normalize-only) pipeline
#
# Post PR-G1 (host-sync removal) this should pass without graph breaks.
# Until then it may raise; we mark it xfail-strict=False so it is
# informative but non-blocking.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason=(
        "fullgraph=True may raise due to residual host-sync branches in base.py. "
        "Expected to pass after PR-G1 host-sync removal is complete."
    ),
)
def test_compile_fullgraph_deterministic_pipeline():
    """fullgraph=True on a Normalize-only pipeline — target: passes post PR-G1."""
    device = torch.device("cpu")
    aug = _make_deterministic_pipeline(device)
    compiled = aug.compile(fullgraph=True)
    x = torch.rand(2, 3, 64, 64)
    eager_out = aug(x.clone())
    compiled_out = compiled(x.clone())
    assert compiled_out.shape == eager_out.shape
    torch.testing.assert_close(compiled_out, eager_out, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Case 7: kwargs forwarded to torch.compile
# ---------------------------------------------------------------------------


def test_compile_kwargs_forwarded():
    """Extra kwargs (e.g. backend='eager') must be forwarded to torch.compile."""
    device = torch.device("cpu")
    aug = _make_hflip_normalize_pipeline(device)
    compiled = aug.compile(backend="eager")
    x = torch.rand(2, 3, 32, 32)
    out = compiled(x)
    assert out.shape == x.shape

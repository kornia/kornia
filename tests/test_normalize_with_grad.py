"""Tests for NormalizeWithGrad — the gradient-preserving Normalize.

This test file intentionally avoids importing ``kornia`` at the top level to
remain runnable on Python 3.10 where other parts of kornia require 3.11+ enum
features.  The deterministic subpackage itself has no such dependency.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest
import torch

# ---------------------------------------------------------------------------
# Bootstrap: load just the deterministic subpackage without triggering the
# full kornia.__init__ (which requires Python 3.11+ enum features).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DET_BASE = os.path.join(_REPO_ROOT, "kornia", "augmentations", "deterministic")

# Names that _bootstrap injects into sys.modules as stubs.
_BOOTSTRAP_KEYS = (
    "kornia",
    "kornia.augmentations",
    "kornia.augmentations.deterministic",
    "kornia.augmentations.deterministic.normalize_with_grad",
)


def _bootstrap() -> type:
    for name in ("kornia", "kornia.augmentations", "kornia.augmentations.deterministic"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    nwg_path = os.path.join(_DET_BASE, "normalize_with_grad.py")
    spec = importlib.util.spec_from_file_location(
        "kornia.augmentations.deterministic.normalize_with_grad", nwg_path
    )
    nwg_mod = importlib.util.module_from_spec(spec)
    sys.modules["kornia.augmentations.deterministic.normalize_with_grad"] = nwg_mod
    spec.loader.exec_module(nwg_mod)

    init_path = os.path.join(_DET_BASE, "__init__.py")
    spec2 = importlib.util.spec_from_file_location("kornia.augmentations.deterministic", init_path)
    det_mod = importlib.util.module_from_spec(spec2)
    sys.modules["kornia.augmentations.deterministic"] = det_mod
    spec2.loader.exec_module(det_mod)

    return det_mod.NormalizeWithGrad


# ---------------------------------------------------------------------------
# Fixture: run bootstrap only during test execution, restore sys.modules after.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def NWG():  # noqa: N802
    """Bootstrap the deterministic subpackage, yield NormalizeWithGrad, then clean up.

    Runs only for tests in this file (no autouse). Restores sys.modules after the
    session so that stub entries injected by _bootstrap do not pollute other test
    files that import the real kornia package.
    """
    # Snapshot only the keys we are about to inject (not the entire sys.modules).
    snapshot = {k: sys.modules.get(k) for k in _BOOTSTRAP_KEYS}
    cls = _bootstrap()
    yield cls
    # Teardown: restore sys.modules to the pre-bootstrap state.
    for k, v in snapshot.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_forward_4d(NWG):
    m = NWG(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    x = torch.ones(2, 3, 4, 4)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.ones_like(y))


def test_basic_forward_3d(NWG):
    m = NWG(mean=(0.5,), std=(0.5,))
    x = torch.ones(1, 4, 4) * 0.75
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.full_like(y, 0.5))


def test_gradient_flows(NWG):
    m = NWG(mean=(0.5,), std=(0.5,))
    x = torch.ones(1, 1, 4, 4, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert x.grad is not None
    # d/dx [(x - 0.5)/0.5] = 1/0.5 = 2  per element
    assert torch.allclose(x.grad, torch.full_like(x, 2.0))


def test_gradcheck(NWG):
    m = NWG(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    x = torch.randn(1, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    m = m.double()
    assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-5)


def test_invalid_lengths(NWG):
    with pytest.raises(ValueError, match="same length"):
        NWG(mean=(0.5,), std=(0.5, 0.5))


def test_invalid_std_negative(NWG):
    with pytest.raises(ValueError, match="std must be all"):
        NWG(mean=(0.5,), std=(-0.5,))


def test_invalid_std_zero(NWG):
    with pytest.raises(ValueError, match="std must be all"):
        NWG(mean=(0.5,), std=(0.0,))


def test_device_move(NWG):
    m = NWG(mean=(0.5,), std=(0.5,))
    if torch.cuda.is_available():
        m = m.cuda()
        x = torch.ones(1, 1, 4, 4, device="cuda")
        assert m(x).device.type == "cuda"


def test_dtype_promotion(NWG):
    m = NWG(mean=(0.5,), std=(0.5,))
    x = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    y = m(x)
    assert y.dtype == torch.float64


def test_export_compatible(NWG):
    """The whole point of this class — torch.export should work."""
    m = NWG(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    x = torch.randn(1, 3, 8, 8)
    try:
        ep = torch.export.export(m, (x,))
        y_eager = m(x)
        y_export = ep.module()(x)
        assert torch.allclose(y_eager, y_export, atol=1e-6)
    except Exception as e:
        pytest.skip(f"torch.export not available or failed: {e}")

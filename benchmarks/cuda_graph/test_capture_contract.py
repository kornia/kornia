"""Pytest entry point: parametrized over discovered transforms; FAIL if capture fails.

Skipped entirely if CUDA unavailable.
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.cuda_graph.per_transform import _bench_one, discover_transforms

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

_TRANSFORMS = discover_transforms()


@pytest.mark.parametrize(
    "name,factory",
    _TRANSFORMS,
    ids=[name for name, _ in _TRANSFORMS],
)
def test_transform_captures_into_cuda_graph(name: str, factory) -> None:  # type: ignore[type-arg]
    r = _bench_one(factory, name, n_replays=10, warmup=5)
    if r.capture_status == "FAILED":
        pytest.fail(f"{name} failed CUDA Graph capture: {r.capture_error}")

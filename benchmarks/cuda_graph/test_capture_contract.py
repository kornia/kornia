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

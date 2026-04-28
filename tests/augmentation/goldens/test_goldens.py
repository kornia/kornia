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

# Licensed under the Apache License, Version 2.0 (the "License").
"""Test that current transform outputs match snapshotted goldens.

Goldens are checked in under tests/augmentation/goldens/data/.
A failure means the transform's behavior CHANGED -- either intentional
(needs explicit golden update via ``python -m tests.augmentation.goldens._generate``)
or a regression (must be fixed).

CG-3 gate: refactor PRs run this suite to prove no numerical change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import kornia.augmentation as K

DATA_DIR = Path(__file__).parent / "data"
INDEX = DATA_DIR / "_index.json"

if not INDEX.exists():
    pytest.skip(
        "Goldens not generated yet. Run: .venv/bin/python -m tests.augmentation.goldens._generate",
        allow_module_level=True,
    )

_files: list[str] = json.loads(INDEX.read_text())["files"]


@pytest.mark.parametrize("filename", _files)
def test_golden(filename: str) -> None:
    """Assert transform output matches the checked-in golden snapshot."""
    p = DATA_DIR / filename
    # allow_pickle required for object-dtype arrays (tuple-returning transforms).
    # These .npz files are repo-controlled checked-in fixtures, not external input.
    z = np.load(p, allow_pickle=True)

    name = str(z["transform_class"])
    seed = int(z["seed"])
    expected = z["output"]

    cls = getattr(K, name, None)
    if cls is None:
        pytest.skip(f"{name} no longer in kornia.augmentation public API")

    # np.load returns memory-mapped (read-only) arrays; copy before converting to
    # torch tensors to avoid "non-writable buffer" warnings and incorrect state restores.
    x = torch.from_numpy(z["input"].copy())

    # Restore the RNG state captured right after x was generated so that
    # the transform's forward pass sees identical random draws to those at
    # snapshot time.  Older goldens without rng_state fall back to manual_seed.
    if "rng_state" in z:
        rng_state_np = z["rng_state"].copy()  # must be writable for set_rng_state
        torch.set_rng_state(torch.from_numpy(rng_state_np))
    else:
        torch.manual_seed(seed)

    transform = cls()
    with torch.no_grad():
        y = transform(x)

    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = np.asarray(
            [yi.cpu().numpy() if isinstance(yi, torch.Tensor) else yi for yi in y],
            dtype=object,
        )

    if expected.dtype == object or y_np.dtype == object:
        # Tuple-returning transforms -- compare element-wise approximately
        for a, b in zip(expected, y_np):
            np.testing.assert_allclose(
                np.asarray(a, dtype=np.float32),
                np.asarray(b, dtype=np.float32),
                atol=1e-5,
                rtol=1e-4,
                err_msg=f"Golden mismatch for {name} (seed={seed}, file={filename})",
            )
    else:
        np.testing.assert_allclose(
            y_np,
            expected,
            atol=1e-5,
            rtol=1e-4,
            err_msg=f"Golden mismatch for {name} (seed={seed}, file={filename})",
        )

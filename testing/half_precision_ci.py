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

# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from kornia.core.exceptions import BaseError

ISSUE_URL = "https://github.com/kornia/kornia/issues/4153"
_MANIFEST_DIR = Path(__file__).with_name("half_precision_xfails")
_EXCEPTION_TYPES: dict[str, type[BaseException]] = {
    "AssertionError": AssertionError,
    "KeyError": KeyError,
    "NotImplementedError": NotImplementedError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "kornia.core.exceptions.BaseError": BaseError,
}


def load_known_failures(dtype: str, manifest_dir: Path | None = None) -> dict[str, type[BaseException]]:
    """Load exact test node IDs and their expected exception types for a CPU half dtype."""
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError(f"unsupported half-precision dtype: {dtype}")

    path = (manifest_dir or _MANIFEST_DIR) / f"cpu_{dtype}.txt"
    failures: dict[str, type[BaseException]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        try:
            exception_name, nodeid = line.split("\t", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: expected '<exception>\\t<node ID>'") from error
        if exception_name not in _EXCEPTION_TYPES:
            raise ValueError(f"{path}:{line_number}: unknown exception type: {exception_name}")
        if f"cpu-{dtype}" not in nodeid:
            raise ValueError(f"{path}:{line_number}: node ID does not select cpu-{dtype}: {nodeid}")
        if nodeid in failures:
            raise ValueError(f"{path}:{line_number}: duplicate node ID: {nodeid}")
        failures[nodeid] = _EXCEPTION_TYPES[exception_name]
    return failures


def mark_known_failures(items: Sequence[Any], dtypes: Sequence[str], manifest_dir: Path | None = None) -> None:
    """Strictly xfail the complete known-failure set for selected CPU half dtypes."""
    failures: dict[str, type[BaseException]] = {}
    for dtype in dtypes:
        dtype_failures = load_known_failures(dtype, manifest_dir)
        overlap = failures.keys() & dtype_failures.keys()
        if overlap:
            raise ValueError(f"duplicate node IDs across manifests: {sorted(overlap)!r}")
        failures.update(dtype_failures)

    collected = {item.nodeid for item in items}
    missing = failures.keys() - collected
    if missing:
        preview = "\n".join(sorted(missing)[:10])
        raise ValueError(f"{len(missing)} known half-precision failures were not collected:\n{preview}")

    reason = f"Known Linux CPU half-precision failure tracked in {ISSUE_URL}"
    for item in items:
        if exception_type := failures.get(item.nodeid):
            item.add_marker(pytest.mark.xfail(reason=reason, raises=exception_type, strict=True))

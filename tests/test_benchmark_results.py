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

"""CI shim for benchmark result schema validation (collected by CI via tests/ directory)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("results_schema", REPO_ROOT / "benchmarks" / "results_schema.py")
results_schema = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(results_schema)


def test_all_committed_results_are_valid() -> None:
    """Every committed benchmark result JSON must pass schema validation."""
    results_root = REPO_ROOT / "benchmarks" / "results"
    for path in sorted(results_root.rglob("*.json")):
        assert results_schema.validate_result(path) == [], f"{path} invalid"


def test_all_committed_artefacts_are_valid() -> None:
    """Comparison artefacts (A/B and cross-release raw JSON next to a report) pass the content rules."""
    paths = sorted((REPO_ROOT / "benchmarks").glob("*/*_results/*.json"))
    assert paths, "no comparison artefacts matched benchmarks/*/*_results/*.json; update the glob with the layout"
    for path in paths:
        assert results_schema.validate_artefact(path) == [], f"{path} invalid"

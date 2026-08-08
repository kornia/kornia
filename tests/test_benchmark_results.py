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

"""Shim for benchmark result schema validation tests (collected by CI via tests/ directory)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# Load results_schema_test from benchmarks/ directory
spec = importlib.util.spec_from_file_location(
    "results_schema_test",
    os.path.join(Path(__file__).resolve().parent.parent, "benchmarks", "results_schema_test.py"),
)
results_schema_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results_schema_test)


def test_all_committed_results_are_valid() -> None:
    """Proxy test that ensures all committed benchmark results pass validation."""
    results_schema_test.test_all_committed_results_are_valid()

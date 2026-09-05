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

"""Unit tests for the benchmark result file schema validator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_schema


def _valid_payload() -> dict:
    return {
        "metadata": {
            "timestamp_utc": "2026-08-08T12:00:00+00:00",
            "git_commit": "c670e2ab",
            "platform": "macOS-26.5.1-arm64-arm-64bit",
            "python": "3.11.0",
            "torch": "2.9.1",
            "kornia": "0.9.0rc1",
            "device": "cpu",
            "load": {"load_avg_1m": 2.0, "cpu_count": 8},
        },
        "results": [
            {"op": "sobel", "backend": "kornia (eager)", "batch": 1, "median_us": 10.0, "throughput_per_s": 1e5}
        ],
    }


def test_valid_file_passes(tmp_path) -> None:
    p = tmp_path / "0.9.0rc1" / "filters--test-box--cpu.json"
    p.parent.mkdir()
    p.write_text(json.dumps(_valid_payload()))
    assert results_schema.validate_result(p) == []


def test_absolute_path_in_metadata_flagged(tmp_path) -> None:
    """A committed file must not disclose a home directory or machine layout (README privacy rule)."""
    p = tmp_path / "0.9.0rc1" / "filters--test-box--cpu.json"
    p.parent.mkdir()
    for leaked in ("/home/someone/dev/kornia/kornia/__init__.py", "C:\\Users\\someone\\kornia", "\\\\host\\share"):
        payload = _valid_payload()
        payload["metadata"]["kornia_module"] = leaked
        p.write_text(json.dumps(payload))
        assert any("privacy" in e for e in results_schema.validate_result(p)), leaked


def test_checkout_relative_module_path_passes(tmp_path) -> None:
    payload = _valid_payload()
    payload["metadata"]["kornia_module"] = "kornia/__init__.py"
    p = tmp_path / "0.9.0rc1" / "filters--test-box--cpu.json"
    p.parent.mkdir()
    p.write_text(json.dumps(payload))
    assert results_schema.validate_result(p) == []


def test_device_mismatch_flagged(tmp_path) -> None:
    p = tmp_path / "0.9.0rc1" / "filters--test-box--cuda.json"
    p.parent.mkdir()
    p.write_text(json.dumps(_valid_payload()))  # metadata says cpu
    assert any("device" in e for e in results_schema.validate_result(p))


def test_version_dir_mismatch_flagged(tmp_path) -> None:
    p = tmp_path / "9.9.9" / "filters--test-box--cpu.json"
    p.parent.mkdir()
    p.write_text(json.dumps(_valid_payload()))
    assert any("version" in e for e in results_schema.validate_result(p))


def test_missing_metric_fields_flagged(tmp_path) -> None:
    p = tmp_path / "0.9.0rc1" / "filters--test-box--cpu.json"
    p.parent.mkdir()
    payload = _valid_payload()
    # Remove required metric fields
    payload["results"][0].pop("median_us", None)
    p.write_text(json.dumps(payload))
    assert any("median_us" in e for e in results_schema.validate_result(p))
    # Also test missing throughput_per_s
    payload = _valid_payload()
    payload["results"][0].pop("throughput_per_s", None)
    p.write_text(json.dumps(payload))
    assert any("throughput_per_s" in e for e in results_schema.validate_result(p))


def test_all_committed_results_are_valid() -> None:
    results_root = Path(__file__).parent / "results"
    for path in sorted(results_root.rglob("*.json")):
        assert results_schema.validate_result(path) == [], f"{path} invalid"


def test_missing_load_flagged_for_release_snapshot(tmp_path) -> None:
    p = tmp_path / "0.9.0rc1" / "filters--test-box--cpu.json"
    p.parent.mkdir()
    payload = _valid_payload()
    del payload["metadata"]["load"]
    p.write_text(json.dumps(payload))
    assert any("load" in e for e in results_schema.validate_result(p))


def test_artefact_skips_layout_rules_and_optional_load(tmp_path) -> None:
    """An A/B artefact names the compared revision, not a machine, and may predate ``load``."""
    p = tmp_path / "graf_results" / "base-cuda.json"
    p.parent.mkdir()
    payload = _valid_payload()
    del payload["metadata"]["load"]
    p.write_text(json.dumps(payload))
    assert results_schema.validate_result(p) != []  # the same file is not a release snapshot
    assert results_schema.validate_artefact(p) == []


def test_artefact_keeps_content_rules(tmp_path) -> None:
    p = tmp_path / "graf_results" / "base-cuda.json"
    p.parent.mkdir()
    for mutate, needle in (
        (lambda m: m["metadata"].pop("git_commit"), "git_commit"),
        (lambda m: m["metadata"].__setitem__("kornia_module", "/home/someone/kornia"), "privacy"),
        (lambda m: m["metadata"].__setitem__("load", {"load_avg_1m": "busy"}), "privacy"),
        (lambda m: m["results"][0].pop("median_us"), "median_us"),
        (lambda m: m["results"].clear(), "non-empty"),
    ):
        payload = _valid_payload()
        mutate(payload)
        p.write_text(json.dumps(payload))
        assert any(needle in e for e in results_schema.validate_artefact(p)), needle


def test_all_committed_artefacts_are_valid() -> None:
    paths = sorted((Path(__file__).parent).glob("*/*_results/*.json"))
    assert paths, "no comparison artefacts matched benchmarks/*/*_results/*.json; update the glob with the layout"
    for path in paths:
        assert results_schema.validate_artefact(path) == [], f"{path} invalid"


def test_non_object_payload_rejected(tmp_path) -> None:
    p = tmp_path / "0.9.0rc1" / "filters--test-box--cpu.json"
    p.parent.mkdir()
    p.write_text(json.dumps(["metadata", "results"]))
    errors = results_schema.validate_result(p)
    assert errors and "object" in errors[0]


def test_non_object_metadata_rejected(tmp_path) -> None:
    p = tmp_path / "0.9.0rc1" / "filters--test-box--cpu.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"metadata": "oops", "results": []}))
    errors = results_schema.validate_result(p)
    assert errors and "metadata must be an object" in errors[0]


def test_bool_batch_rejected(tmp_path) -> None:
    p = tmp_path / "0.9.0rc1" / "filters--test-box--cpu.json"
    p.parent.mkdir()
    payload = _valid_payload()
    payload["results"][0]["batch"] = True
    p.write_text(json.dumps(payload))
    assert any("batch" in e for e in results_schema.validate_result(p))

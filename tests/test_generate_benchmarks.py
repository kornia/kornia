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

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "generate_benchmarks", Path(__file__).parents[1] / "docs" / "generate_benchmarks.py"
)
generate_benchmarks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_benchmarks)


def _seed(tmp_path: Path) -> Path:
    payload = {
        "metadata": {
            "timestamp_utc": "2026-08-08T12:00:00+00:00",
            "git_commit": "c670e2ab",
            "platform": "macOS-26.5.1-arm64-arm-64bit",
            "python": "3.11.0",
            "torch": "2.9.1",
            "kornia": "0.9.0rc1",
            "device": "mps",
            "load": {"load_avg_1m": 2.0},
        },
        "results": [
            {"op": "ColorJiggle", "backend": "kornia (eager)", "batch": 32, "throughput_per_s": 317.0},
            {"op": "ColorJiggle", "backend": "albumentations", "batch": 32, "throughput_per_s": 2526.0},
        ],
    }
    d = tmp_path / "0.9.0rc1"
    d.mkdir(exist_ok=True)
    (d / "augmentation--apple-m3--mps.json").write_text(json.dumps(payload))
    return tmp_path


def test_render_page_contains_table_and_metadata(tmp_path: Path) -> None:
    rst = generate_benchmarks.render_page(_seed(tmp_path))
    assert "Performance" in rst and "ColorJiggle" in rst
    assert "317" in rst and "2526" in rst  # numbers from the JSON, not hand-written
    assert "apple-m3" in rst and "2026-08-08" in rst  # machine + date disclosed
    assert "close other applications" in rst  # hygiene box present


def test_latest_version_orders_rc_before_final() -> None:
    assert generate_benchmarks.latest_version(["0.9.0rc1", "0.9.0", "0.8.3"]) == "0.9.0"


def test_render_page_empty_results_dir(tmp_path: Path) -> None:
    rst = generate_benchmarks.render_page(tmp_path)
    assert "No benchmark results" in rst  # page still builds, honestly empty


def test_refresh_llms_replaces_only_marker_block(tmp_path: Path) -> None:
    llms = tmp_path / "llms-full.txt"
    llms.write_text("before\n<!-- BENCH:BEGIN -->\nold\n<!-- BENCH:END -->\nafter\n")
    generate_benchmarks.refresh_llms(llms, _seed(tmp_path))
    text = llms.read_text()
    assert text.startswith("before\n") and text.endswith("after\n")
    assert "old" not in text and "ColorJiggle" not in text  # digest is per-suite headline, not per-op dump
    assert "0.9.0rc1" in text and "apple-m3" in text


def test_refresh_llms_idempotent(tmp_path: Path) -> None:
    llms = tmp_path / "llms-full.txt"
    llms.write_text("x\n<!-- BENCH:BEGIN -->\n<!-- BENCH:END -->\ny\n")
    generate_benchmarks.refresh_llms(llms, _seed(tmp_path))
    first = llms.read_text()
    generate_benchmarks.refresh_llms(llms, _seed(tmp_path))
    assert llms.read_text() == first


def test_refresh_llms_missing_markers_raises(tmp_path: Path) -> None:
    llms = tmp_path / "llms-full.txt"
    llms.write_text("no markers here\n")
    with pytest.raises(RuntimeError):
        generate_benchmarks.refresh_llms(llms, _seed(tmp_path))

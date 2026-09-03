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


def _seed_two_axis(tmp_path: Path) -> Path:
    """A suite that sweeps a second axis at a fixed batch, like ``feature-laf-ops``."""
    payload = {
        "metadata": {
            "timestamp_utc": "2026-09-03T12:00:00+00:00",
            "git_commit": "d415faa0",
            "platform": "Linux-6.18-x86_64",
            "python": "3.13.0",
            "torch": "2.14.0",
            "kornia": "0.9.0rc1",
            "device": "cpu",
            "load": {"load_avg_1m": 1.0},
            "units": "LAFs/s",
        },
        "results": [
            {"op": "make_upright", "backend": "kornia (eager)", "batch": 1, "n_lafs": 2000, "throughput_per_s": 20.0},
            {"op": "make_upright", "backend": "kornia (eager)", "batch": 1, "n_lafs": 20000, "throughput_per_s": 33.0},
            {"op": "make_upright", "backend": "kornia (eager)", "batch": 8, "n_lafs": 2000, "throughput_per_s": 31.0},
        ],
    }
    d = tmp_path / "0.9.0rc1"
    d.mkdir(exist_ok=True)
    (d / "feature-laf-ops--box--cpu.json").write_text(json.dumps(payload))
    return tmp_path


def test_render_page_keeps_configs_that_share_a_batch(tmp_path: Path) -> None:
    rst = generate_benchmarks.render_page(_seed_two_axis(tmp_path))
    # One cell per config, in order. Keying rows on (op, batch) alone collapses the two B=1
    # configs and drops a value outright; a bare `"20" in rst` would not catch that, because the
    # 2026-09-03 timestamp above the table contains "20" whatever the table renders.
    assert [ln.strip()[2:] for ln in rst.splitlines() if ln.startswith("     - ")] == [
        "kornia (eager)",
        "20",
        "33",
        "31",
    ]
    assert "n_lafs=2000" in rst and "n_lafs=20000" in rst
    assert "throughput in LAFs/s" in rst  # the item is a LAF, not an image


def test_render_page_omits_constant_config_fields(tmp_path: Path) -> None:
    rst = generate_benchmarks.render_page(_seed(tmp_path))
    assert "ColorJiggle @ 32" in rst  # batch-only suites keep the short label
    assert "throughput in items/s" in rst  # no units key -> the existing default


def test_digest_disambiguates_configs_that_share_a_batch(tmp_path: Path) -> None:
    llms = tmp_path / "llms-full.txt"
    llms.write_text("a\n<!-- BENCH:BEGIN -->\n<!-- BENCH:END -->\nb\n")
    generate_benchmarks.refresh_llms(llms, _seed_two_axis(tmp_path))
    line = next(ln for ln in llms.read_text().splitlines() if ln.startswith("- feature-laf-ops"))
    assert "make_upright@1,n_lafs=20000" in line  # fastest row named by its full config
    assert "LAFs/s" in line


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
    assert "old" not in text  # marker block content was replaced, not appended to
    # digest is one headline line per result file, naming the actual op/batch that was
    # fastest/slowest so the numbers can't be misread as a blanket per-backend gap
    digest_lines = [ln for ln in text.splitlines() if ln.startswith("- augmentation")]
    assert len(digest_lines) == 1
    assert "ColorJiggle@32" in digest_lines[0]
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


def test_refresh_llms_swapped_markers_raises(tmp_path: Path) -> None:
    llms = tmp_path / "llms-full.txt"
    llms.write_text("before\n<!-- BENCH:END -->\nold\n<!-- BENCH:BEGIN -->\nafter\n")
    with pytest.raises(RuntimeError):
        generate_benchmarks.refresh_llms(llms, _seed(tmp_path))


def test_committed_digest_is_fresh(tmp_path: Path) -> None:
    """The committed llms-full.txt digest must already match committed results.

    Guards against the committed benchmark digest going stale relative to committed
    results: if this fails, re-run `python docs/generate_benchmarks.py --refresh-llms`
    and commit the result.
    """
    results_root = generate_benchmarks.RESULTS
    if not any(results_root.rglob("*.json")):
        pytest.skip("no committed benchmark results")
    committed = generate_benchmarks.LLMS_FULL
    copy = tmp_path / "llms-full.txt"
    copy.write_text(committed.read_text())
    generate_benchmarks.refresh_llms(copy, results_root)
    assert copy.read_bytes() == committed.read_bytes()


def test_latest_version_tolerates_digitless_dirs() -> None:
    assert generate_benchmarks.latest_version(["unknown", "0.9.0rc1"]) == "0.9.0rc1"

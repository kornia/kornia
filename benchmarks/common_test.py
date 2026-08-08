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

"""Unit tests for the shared benchmark methodology utilities."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import git_commit, run_batch_sweep, run_metadata, save_json, time_us


def test_time_us_returns_median_and_spread():
    median, iqr = time_us(lambda: sum(range(100)), min_run_time=0.05)
    assert median > 0 and not math.isnan(median)
    assert iqr >= 0


def test_time_us_failure_is_nan():
    def boom():
        raise RuntimeError("expected")

    median, iqr = time_us(boom, min_run_time=0.05)
    assert math.isnan(median) and math.isnan(iqr)


def test_time_us_accepts_sync_callable():
    calls = []
    median, _ = time_us(lambda: None, min_run_time=0.05, sync=lambda: calls.append(1))
    assert not math.isnan(median)
    assert calls  # sync ran inside the timed region


def test_run_metadata_records_environment():
    meta = run_metadata(torch.device("cpu"))
    assert meta["torch"] == torch.__version__
    assert meta["git_commit"]
    for key in ("timestamp_utc", "platform", "python", "kornia", "torch_num_threads", "device"):
        assert key in meta


def test_git_commit_is_short_hash():
    assert len(git_commit()) >= 7


def test_save_json_round_trip_sanitizes_non_finite(tmp_path):
    rows = [{"op": "x", "median_us": float("nan"), "throughput_per_s": float("inf"), "iqr_us": float("-inf")}]
    path = save_json(tmp_path / "run.json", {"a": 1}, rows)

    def reject_constant(name):
        raise AssertionError(f"non-strict JSON token in output: {name}")

    payload = json.loads(path.read_text(), parse_constant=reject_constant)
    assert payload["metadata"] == {"a": 1}
    assert payload["results"][0]["median_us"] is None
    assert payload["results"][0]["throughput_per_s"] is None
    assert payload["results"][0]["iqr_us"] is None


def test_run_metadata_records_optional_baseline_versions():
    meta = run_metadata(torch.device("cpu"))
    for key in ("opencv", "torchvision", "albumentations", "kornia_rs", "pillow"):
        assert key in meta  # None when not installed — key must still be present


def test_run_batch_sweep_rows_and_skip_cells(capsys):
    def build(b):
        return {"opA": {"fast": lambda: None, "missing": None}}, {}

    rows = run_batch_sweep([1, 2], build, ["fast", "missing"], row_fields=lambda b: {"size": 8}, min_run_time=0.05)
    assert [r["batch"] for r in rows] == [1, 2]  # only 'fast' produces rows
    assert rows[0]["op"] == "opA" and rows[0]["backend"] == "fast" and rows[0]["size"] == 8
    assert rows[0]["median_us"] > 0 and rows[0]["throughput_per_s"] > 0
    out = capsys.readouterr().out
    assert "batch=1" in out and "batch=2" in out
    assert "-" in out  # the skip cell


def test_run_batch_sweep_syncs_torch_backends_only():
    synced_during: list[str] = []
    current = {"name": ""}

    def build(b):
        def make(name):
            def fn():
                current["name"] = name

            return fn

        return {"opA": {"kornia (eager)": make("kornia (eager)"), "kornia-rs": make("kornia-rs")}}, {}

    run_batch_sweep(
        [1],
        build,
        ["kornia (eager)", "kornia-rs"],
        row_fields=lambda b: {},
        sync=lambda: synced_during.append(current["name"]),
        min_run_time=0.05,
    )
    assert "kornia (eager)" in synced_during  # torch backend gets the device sync
    assert "kornia-rs" not in synced_during  # CPU-only backend must not be synced


def test_run_batch_sweep_reports_warmup_failures(capsys):  # 'compile' in a test NAME gets deselected by conftest
    def build(b):
        return {"opA": {"fast": lambda: None}}, {"opA": "RuntimeError"}

    run_batch_sweep([1], build, ["fast"], row_fields=lambda b: {}, min_run_time=0.05)
    assert "torch.compile warmup failed" in capsys.readouterr().out

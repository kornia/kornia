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
import common
from common import git_commit, run_batch_sweep, run_metadata, save_json, time_us, versions_line


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
    assert len(git_commit().removesuffix("-dirty")) >= 7


def test_git_commit_marks_a_modified_checkout(monkeypatch):
    """A run from an edited tree is not reproducible from the hash, so the hash says so."""
    import common

    calls = []

    def fake(cmd, text=True):
        calls.append(cmd)
        return "abc1234\n" if "rev-parse" in cmd else " M kornia/feature/laf.py\n"

    monkeypatch.setattr(common.subprocess, "check_output", fake)
    assert common.git_commit() == "abc1234-dirty"
    assert "-uno" in calls[1]  # untracked files (a contributed result file) must not count


def test_git_commit_clean_checkout_is_bare_hash(monkeypatch):
    import common

    monkeypatch.setattr(
        common.subprocess, "check_output", lambda cmd, text=True: "abc1234\n" if "rev-parse" in cmd else "\n"
    )
    assert common.git_commit() == "abc1234"


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
    measured = [r for r in rows if r.get("error") is None]
    assert [r["batch"] for r in measured] == [1, 2]
    assert measured[0]["op"] == "opA" and measured[0]["backend"] == "fast" and measured[0]["size"] == 8
    assert measured[0]["median_us"] > 0 and measured[0]["throughput_per_s"] > 0
    # the unavailable backend is recorded, not dropped, so the JSON shows the gap
    skipped = [r for r in rows if r["backend"] == "missing"]
    assert len(skipped) == 2
    assert all(r["median_us"] is None and r["throughput_per_s"] is None for r in skipped)
    assert all(r["error"] == "unavailable" for r in skipped)
    out = capsys.readouterr().out
    assert "batch=1" in out and "batch=2" in out
    assert "-" in out  # the skip cell


def test_run_batch_sweep_accepts_non_batch_configs(capsys):
    def build(config):
        return {"opA": {"fast": lambda: None}}, {}

    rows = run_batch_sweep(
        [(2, 100)],
        build,
        ["fast"],
        row_fields=lambda c: {"batch": c[0], "n": c[1]},
        label_fn=lambda c: f"B={c[0]} N={c[1]}",
        items_fn=lambda c: c[0] * c[1],
        units="LAFs/s",
        min_run_time=0.05,
    )
    assert rows[0]["batch"] == 2 and rows[0]["n"] == 100  # row_fields overrides the config object
    per_call_s = rows[0]["median_us"] * 1e-6
    assert math.isclose(rows[0]["throughput_per_s"], 200 / per_call_s)  # items_fn drives it, not the config
    out = capsys.readouterr().out
    assert "B=2 N=100" in out and "(LAFs/s)" in out


def test_versions_line_reports_stack_and_gaps():
    line = versions_line({"torch": "2.9.1", "kornia": "0.9.0", "torchvision": None})
    assert line.startswith("#")
    assert "torch 2.9.1" in line and "kornia 0.9.0" in line
    assert "torchvision -" in line  # missing libs shown as '-', never dropped


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


def test_run_batch_sweep_records_warmup_failure_in_rows():  # 'compile' in a NAME gets deselected
    def build(b):
        return {"opA": {"kornia (eager)": lambda: None, "kornia (compiled)": None}}, {"opA": "InductorError"}

    rows = run_batch_sweep(
        [1], build, ["kornia (eager)", "kornia (compiled)"], row_fields=lambda b: {}, min_run_time=0.05
    )
    failed = next(r for r in rows if r["backend"] == "kornia (compiled)")
    assert failed["error"] == "InductorError"  # the exception type reaches the JSON, not just stdout
    assert failed["median_us"] is None and failed["throughput_per_s"] is None


def test_collect_load_metrics_aggregate_only() -> None:
    from common import collect_load_metrics

    m = collect_load_metrics()
    assert set(m) == {
        "load_avg_1m",
        "load_avg_5m",
        "load_avg_15m",
        "cpu_count",
        "mem_total_bytes",
        "mem_available_bytes",
    }
    # privacy: values are numbers or None — never strings that could carry process names
    assert all(v is None or isinstance(v, (int, float)) for v in m.values())


def test_machine_slug_prefers_cuda_device() -> None:
    from common import machine_slug

    assert machine_slug({"cuda_device": "NVIDIA GeForce RTX 4080", "machine": "x86_64"}) == ("nvidia-geforce-rtx-4080")


def test_machine_slug_override_wins() -> None:
    from common import machine_slug

    assert machine_slug({"cuda_device": "NVIDIA L4"}, override="box-a") == "box-a"


def test_canonical_result_name() -> None:
    from common import canonical_result_name

    meta = {"cuda_device": "NVIDIA L4", "device": "cuda:0", "machine": "x86_64"}
    assert canonical_result_name(meta, "augmentation") == "augmentation--nvidia-l4--cuda.json"


def test_contribute_result_writes_canonical_path(tmp_path) -> None:
    meta = {"kornia": "0.9.0rc1", "device": "cpu", "machine": "arm64", "load": {"load_avg_1m": 1.0}}
    out = common.contribute_result(tmp_path, "filters", meta, [{"op": "sobel", "batch": 1}], slug_override="test-box")
    assert out == tmp_path / "0.9.0rc1" / "filters--test-box--cpu.json"
    payload = json.loads(out.read_text())
    assert payload["metadata"]["load"]["load_avg_1m"] == 1.0
    assert payload["results"][0]["op"] == "sobel"


def test_print_preflight_warns_on_high_load(capsys) -> None:
    common.print_preflight({"load_avg_1m": 64.0, "cpu_count": 8, "mem_total_bytes": 100, "mem_available_bytes": 5})
    outp = capsys.readouterr().out
    assert "close other applications" in outp
    assert "WARNING" in outp  # load1 > cpu_count and available < 10% both trip it


def test_machine_slug_override_is_slugified() -> None:
    from common import machine_slug

    assert machine_slug({"machine": "x86_64"}, override="My Box! #2") == "my-box-2"

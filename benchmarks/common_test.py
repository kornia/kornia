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

import torch
from common import git_commit, run_metadata, save_json, time_us


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


def test_save_json_round_trip_sanitizes_nan(tmp_path):
    path = save_json(tmp_path / "run.json", {"a": 1}, [{"op": "x", "median_us": float("nan")}])
    payload = json.loads(path.read_text())  # strict parse — a bare NaN token would fail here
    assert payload["metadata"] == {"a": 1}
    assert payload["results"][0]["median_us"] is None

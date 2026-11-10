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

"""Budget accounting for the HEB RANSAC benchmark."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.geometry import heb


def test_legacy_estimator_requires_exact_budget(monkeypatch):
    def legacy_ransac(*args, **kwargs):
        if "max_samples" in kwargs:
            raise TypeError("unexpected keyword argument 'max_samples'")
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(heb, "RANSAC", legacy_ransac)
    args = Namespace(threshold=1.0, confidence=1.0, score="msac")
    config = {"batch_size": 600, "budget": 1000, "max_lo_iters": 0, "prosac": False}
    with pytest.raises(SystemExit, match="budget must be divisible by batch size"):
        heb.make_estimator(args, config, 0)

    config["budget"] = 1200
    assert heb.make_estimator(args, config, 0).max_iter == 2


def test_reported_samples_exclude_truncated_tail(monkeypatch):
    matches = np.zeros((6, 10), dtype=np.float32)
    matches[:, :2] = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2]]
    matches[:, 2:4] = matches[:, :2]
    matches[:, 8] = 0.1
    monkeypatch.setattr(heb, "setup_run", lambda *args, **kwargs: (torch.device("cpu"), torch.float32, None))
    monkeypatch.setattr(heb, "start_run", lambda *args, **kwargs: {})
    monkeypatch.setattr(heb, "load_pairs", lambda args: {"pair": {"matches": matches}})
    rows = []
    monkeypatch.setattr(heb, "finish_run", lambda args, name, meta, result: rows.extend(result))
    args = Namespace(
        h5="unused.h5",
        pairs=1,
        pair_seed=0,
        snn=0.8,
        threshold=1.0,
        confidence=1.0,
        score="msac",
        min_run_time=0.01,
        batches="400",
        budgets="1000",
        lo="0",
        prosac="off",
        seeds="0",
        timing_pairs=0,
    )

    heb.run(args)

    assert rows[0]["batches"] == 3
    assert rows[0]["sampled_sets"] == 1000

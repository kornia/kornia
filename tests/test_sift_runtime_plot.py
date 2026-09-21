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
"""Validate SIFT chart data without requiring Matplotlib or running benchmarks."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "plot_sift_runtime", REPO_ROOT / "benchmarks" / "feature" / "plot_sift_runtime.py"
)
sift_plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sift_plot)


@pytest.fixture
def document() -> dict[str, Any]:
    """Return independent benchmark data with complete comparison metadata."""
    return {
        "metadata": {
            "num_features": 4096,
            "python": "3.13.5",
            "torch": "2.14.0+cu130",
            "cuda_device": "NVIDIA GeForce RTX 4090",
            "input_sha256": "a" * 64,
            "torch_num_threads": 1,
            "opencv_num_threads": 1,
        },
        "results": [
            {
                "op": "SIFTFeatureScaleSpace",
                "series": "current",
                "device": "cuda",
                "batch": 4,
                "height": 640,
                "width": 800,
                "median_us": 8000.0,
                "iqr_us": 800.0,
                "features_per_image": [4096] * 4,
                "peak_extra_cuda_bytes": 1048576,
            }
        ],
    }


def write_document(tmp_path: Path, document: dict[str, Any], name: str = "run.json") -> Path:
    """Write a test result using the benchmark metadata/results envelope."""
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize("batch, expected", [(1, (8.0, 0.8)), (4, (2.0, 0.2)), (8, (1.0, 0.1))])
def test_normalize_batch_latency(batch: int, expected: tuple[float, float]) -> None:
    """Both median and IQR are converted from batch microseconds to ms/image."""
    assert sift_plot.normalize_timing({"batch": batch, "median_us": 8000, "iqr_us": 800}) == expected


@pytest.mark.parametrize("median", [0, -1, math.nan, math.inf, -math.inf, None, True, "8000"])
def test_reject_invalid_median(document: dict[str, Any], median: Any) -> None:
    with pytest.raises(ValueError, match="median_us"):
        sift_plot.normalize_timing(document["results"][0] | {"median_us": median})


@pytest.mark.parametrize("batch", [0, -1, 1.5, True, None, "4"])
def test_reject_invalid_batch(document: dict[str, Any], batch: Any) -> None:
    with pytest.raises(ValueError, match="batch"):
        sift_plot.normalize_timing(document["results"][0] | {"batch": batch})


@pytest.mark.parametrize("spread", [-1, math.nan, math.inf, None, True, "800"])
def test_reject_invalid_iqr(document: dict[str, Any], spread: Any) -> None:
    with pytest.raises(ValueError, match="iqr_us"):
        sift_plot.normalize_timing(document["results"][0] | {"iqr_us": spread})


def test_zero_iqr_is_valid(document: dict[str, Any]) -> None:
    assert sift_plot.normalize_timing(document["results"][0] | {"iqr_us": 0}) == (2.0, 0.0)


def test_load_normalized_rows(tmp_path: Path, document: dict[str, Any]) -> None:
    assert sift_plot.load_rows([write_document(tmp_path, document)]) == {("current", "cuda", 4): (2.0, 0.2)}


def test_reject_duplicate_measurements(tmp_path: Path, document: dict[str, Any]) -> None:
    path = write_document(tmp_path, document)
    with pytest.raises(ValueError, match="duplicate measurement"):
        sift_plot.load_rows([path, path])


@pytest.mark.parametrize("height,width", [(800, 640), (320, 800), (640, 400)])
def test_reject_wrong_image_dimensions(tmp_path: Path, document: dict[str, Any], height: int, width: int) -> None:
    document["results"][0].update(height=height, width=width)
    with pytest.raises(ValueError, match="height=640, width=800"):
        sift_plot.load_rows([write_document(tmp_path, document)])


@pytest.mark.parametrize("count", [None, 2048, "4096"])
def test_reject_wrong_feature_budget(tmp_path: Path, document: dict[str, Any], count: Any) -> None:
    document["metadata"]["num_features"] = count
    with pytest.raises(ValueError, match="num_features=4096"):
        sift_plot.load_rows([write_document(tmp_path, document)])


def test_reject_opencv_on_cuda(tmp_path: Path, document: dict[str, Any]) -> None:
    document["results"][0].update(op="OpenCV SIFT", series="OpenCV", batch=1)
    with pytest.raises(ValueError, match="OpenCV must be CPU batch one"):
        sift_plot.load_rows([write_document(tmp_path, document)])


def test_hardware_subtitle(tmp_path: Path, document: dict[str, Any]) -> None:
    path = write_document(tmp_path, document)
    assert sift_plot.describe_environment([path], "Intel Core i7-14700K") == (
        "NVIDIA GeForce RTX 4090 · Intel Core i7-14700K · PyTorch 2.14.0"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("python", "3.12.0"),
        ("input_sha256", "b" * 64),
        ("torch_num_threads", 2),
        ("opencv_num_threads", 2),
        ("cuda_device", "NVIDIA GeForce RTX 3090"),
        ("torch", "2.13.0+cu130"),
    ],
)
def test_reject_mixed_environments(tmp_path: Path, document: dict[str, Any], key: str, value: Any) -> None:
    first = write_document(tmp_path, document)
    other = copy.deepcopy(document)
    other["metadata"][key] = value
    second = write_document(tmp_path, other, "other.json")
    with pytest.raises(ValueError, match=key):
        sift_plot.describe_environment([first, second])


@pytest.mark.parametrize("key", ["python", "input_sha256", "torch_num_threads", "opencv_num_threads", "torch"])
def test_reject_missing_environment_metadata(tmp_path: Path, document: dict[str, Any], key: str) -> None:
    del document["metadata"][key]
    with pytest.raises(ValueError, match="metadata"):
        sift_plot.describe_environment([write_document(tmp_path, document)])


def test_caption_discloses_actual_output_counts(tmp_path: Path, document: dict[str, Any]) -> None:
    document["results"].append(
        document["results"][0]
        | {"op": "OpenCV SIFT", "series": "OpenCV", "device": "cpu", "batch": 1, "features_per_image": [2678]}
    )
    caption = sift_plot.describe_outputs([write_document(tmp_path, document)])
    assert caption == (
        "Kornia: 4096 nonzero LAF/descriptor slots/image; OpenCV: 2678 keypoints/image. "
        "RootSIFT enabled; native detection/filtering differ."
    )


def test_caption_reports_output_count_range(tmp_path: Path, document: dict[str, Any]) -> None:
    document["results"][0]["features_per_image"] = [0, 2048, 4000, 4096]
    caption = sift_plot.describe_outputs([write_document(tmp_path, document)])
    assert "Kornia: 0\N{EN DASH}4096 nonzero LAF/descriptor slots/image" in caption


@pytest.mark.parametrize("counts", [None, [], [4096], [4096] * 5, [-1] * 4, [True] * 4, [1.5] * 4])
def test_reject_invalid_output_counts(tmp_path: Path, document: dict[str, Any], counts: Any) -> None:
    document["results"][0]["features_per_image"] = counts
    with pytest.raises(ValueError, match="count"):
        sift_plot.describe_outputs([write_document(tmp_path, document)])

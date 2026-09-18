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

"""Benchmark measurements must use the intended checkout and thread count."""

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location("benchmark_common", Path(__file__).parents[1] / "benchmarks/common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


def _load_filter_flagship():
    root = Path(__file__).parents[1]
    script = root / "benchmarks/filters/flagship.py"
    spec = importlib.util.spec_from_file_location("filter_flagship", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    benchmark_dir = str(script.parent.parent)
    sys.path.insert(0, benchmark_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(benchmark_dir)
    return module


@pytest.mark.device_agnostic
def test_warm_up_cpu_is_bounded():
    start = time.perf_counter()
    common.warm_up_cpu(0.05)
    assert time.perf_counter() - start < 2.0


@pytest.mark.device_agnostic
def test_timing_preserves_configured_threads():
    previous = torch.get_num_threads()
    observed = []
    try:
        torch.set_num_threads(2)
        median, spread = common.time_us(lambda: observed.append(torch.get_num_threads()), min_run_time=0.01)
        assert median > 0 and spread >= 0
        assert observed and set(observed) == {2}
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous)


@pytest.mark.device_agnostic
def test_filter_benchmark_imports_its_checkout(tmp_path):
    # A different installation is visible when Python runs a script directly:
    # the script directory replaces the current directory on sys.path.
    shadow = tmp_path / "kornia"
    shadow.mkdir()
    (shadow / "__init__.py").write_text('raise RuntimeError("imported the installed Kornia instead of the checkout")')
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks/filters/flagship.py"
    code = (
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        f"runpy.run_path({str(script)!r}, run_name='benchmark_import_test')\n"
        "import kornia\n"
        "from pathlib import Path\n"
        f"assert Path(kornia.__file__).resolve().parent == Path({str(root / 'kornia')!r})\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed test code and interpreter; no shell or external input.
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.device_agnostic
def test_filter_benchmark_operation_selection():
    flagship = _load_filter_flagship()

    assert flagship.parse_ops("motion_blur, otsu_threshold") == frozenset({"motion_blur", "otsu_threshold"})
    assert flagship.parse_ops("") is None
    with pytest.raises(argparse.ArgumentTypeError, match=r"unknown operation.*Available:.*otsu_threshold"):
        flagship.parse_ops("not_a_filter")


@pytest.mark.device_agnostic
def test_filter_benchmark_selection_compiles_only_requested_row(monkeypatch):
    flagship = _load_filter_flagship()
    compiled = []

    def fake_compile(fn):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(flagship.torch, "compile", fake_compile)
    ops, failures = flagship.build_ops(
        1,
        5,
        5,
        torch.device("cpu"),
        torch.float32,
        True,
        None,
        None,
        None,
        None,
        None,
        selected_ops=frozenset({"otsu_threshold"}),
    )

    assert list(ops) == ["otsu_threshold"]
    assert len(compiled) == 1
    assert not failures


@pytest.mark.device_agnostic
def test_filter_benchmark_missing_baselines_are_not_timed():
    flagship = _load_filter_flagship()
    ops, _ = flagship.build_ops(
        1,
        9,
        11,
        torch.device("cpu"),
        torch.float32,
        False,
        None,
        None,
        None,
        None,
        None,
        selected_ops=frozenset(set(flagship.AVAILABLE_OPS) - {"canny"}),
    )
    for row in ops.values():
        # A callable returning None would be timed as a very fast, fake baseline.
        for backend in ("scikit-image", "albumentations", "torchvision v2"):
            assert row.get(backend) is None


@pytest.mark.device_agnostic
def test_filter_benchmark_skimage_spatial_axes_and_output():
    skf = pytest.importorskip("skimage.filters")
    skr = pytest.importorskip("skimage.restoration")
    flagship = _load_filter_flagship()
    ops, _ = flagship.build_ops(
        2,
        9,
        11,
        torch.device("cpu"),
        torch.float32,
        False,
        None,
        None,
        None,
        None,
        None,
        selected_ops=frozenset(
            {
                "gaussian_blur2d",
                "median_blur",
                "otsu_threshold",
                "unsharp_mask",
                "sobel",
                "laplacian",
                "box_blur",
                "bilateral_blur",
                "bilateral_blur_grayscale",
            }
        ),
        skf=skf,
        skr=skr,
    )
    for name in ("gaussian_blur2d", "median_blur"):
        expected = ops[name]["kornia (eager)"]()
        actual = torch.from_numpy(np.stack(ops[name]["scikit-image"]())).permute(0, 3, 1, 2)
        # Compare borders too: channel mixing or SciPy's reflect/mirror mismatch
        # can otherwise hide behind a superficially correct interior.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    rng = np.random.default_rng(0)
    images = [(rng.random((9, 11, 3)) * 255).astype(np.uint8).astype(np.float32) / 255 for _ in range(2)]
    for image, actual in zip(images, ops["unsharp_mask"]["scikit-image"]()):
        blurred = skf.gaussian(image, sigma=1.5, mode="reflect", preserve_range=True, channel_axis=2)
        np.testing.assert_allclose(actual, 2 * image - blurred, rtol=1e-5, atol=1e-6)

    gray_images = [image.mean(-1) for image in images]
    outputs = ops["otsu_threshold"]["scikit-image"]()
    assert len(outputs) == 2
    for gray, (filtered, threshold) in zip(gray_images, outputs):
        np.testing.assert_array_equal(filtered, np.where(gray > threshold, gray, 0))

    # These baselines intentionally use different operators or integer rounding.
    # Invoke directly so timing's exception-to-NaN handling cannot hide API drift.
    for name in ("sobel", "laplacian", "box_blur", "bilateral_blur"):
        outputs = ops[name]["scikit-image"]()
        assert len(outputs) == 2
        for output in outputs:
            assert output.shape == (9, 11, 3)
            assert np.isfinite(output).all()
    for output in ops["bilateral_blur_grayscale"]["scikit-image"]():
        assert output.shape == (9, 11)
        assert np.isfinite(output).all()


@pytest.mark.device_agnostic
def test_filter_benchmark_albumentations_fixed_parameters():
    albumentations = pytest.importorskip("albumentations")
    flagship = _load_filter_flagship()
    ops, _ = flagship.build_ops(
        2,
        9,
        11,
        torch.device("cpu"),
        torch.float32,
        False,
        None,
        albumentations,
        None,
        None,
        None,
        selected_ops=frozenset({"unsharp_mask", "motion_blur"}),
    )
    for row in ops.values():
        first, second = row["albumentations"](), row["albumentations"]()
        assert len(first) == 2
        for actual, repeated in zip(first, second):
            assert actual.shape == (9, 11, 3)
            assert actual.dtype == np.uint8
            np.testing.assert_array_equal(actual, repeated)


@pytest.mark.device_agnostic
@pytest.mark.parametrize(
    "name,api,channels,dtype",
    [
        ("box_blur", "box_blur", 3, np.uint8),
        ("median_blur", "median_blur", 3, np.uint8),
        ("sobel", "sobel", 3, np.float32),
        ("bilateral_blur_grayscale", "bilateral_filter", 1, np.uint8),
    ],
)
def test_filter_benchmark_kornia_rs(name, api, channels, dtype):
    pytest.importorskip("kornia_rs")
    flagship = _load_filter_flagship()
    ops, _ = flagship.build_ops(
        2,
        9,
        11,
        torch.device("cpu"),
        torch.float32,
        False,
        None,
        None,
        None,
        None,
        None,
        selected_ops=frozenset({name}),
    )
    callback = ops[name].get("kornia-rs")
    if flagship.krs_fn(api) is None:
        assert callback is None
        return
    outputs = callback()
    assert len(outputs) == 2
    for output in outputs:
        assert output.shape == (9, 11, channels)
        assert output.dtype == dtype
        assert np.isfinite(output).all()

    if name == "median_blur":
        # kornia-rs uses replicated borders and independent RGB medians.
        rng = np.random.default_rng(0)
        for output in outputs:
            image = (rng.random((9, 11, 3)) * 255).astype(np.uint8)
            padded = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode="edge")
            windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5), axis=(0, 1))
            np.testing.assert_array_equal(output, np.median(windows, axis=(-1, -2)))


@pytest.mark.device_agnostic
def test_filter_benchmark_kornia_rs_missing_api(monkeypatch):
    flagship = _load_filter_flagship()
    monkeypatch.setattr(flagship, "krs_fn", lambda name: None)
    ops, _ = flagship.build_ops(
        1,
        9,
        11,
        torch.device("cpu"),
        torch.float32,
        False,
        None,
        None,
        None,
        None,
        None,
        selected_ops=frozenset({"box_blur", "median_blur", "sobel", "bilateral_blur_grayscale"}),
    )
    assert len(ops) == 4
    assert all(row.get("kornia-rs") is None for row in ops.values())

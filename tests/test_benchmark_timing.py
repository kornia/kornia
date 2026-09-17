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

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location("benchmark_common", Path(__file__).parents[1] / "benchmarks/common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


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

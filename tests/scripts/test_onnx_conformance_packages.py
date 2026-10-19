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

import re
import subprocess
import sys
from pathlib import Path

import pytest

from testing import onnx_conformance

# `.github/scripts/` is neither a package nor on the path; see tests/scripts/test_check_torch_env.py for why this
# stays a per-module insert.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".github" / "scripts"))

from onnx_conformance_packages import STABLE_CORE, select

REPO = Path(__file__).parent.parent.parent
SCRIPT = REPO / ".github" / "scripts" / "onnx_conformance_packages.py"

pytestmark = pytest.mark.device_agnostic


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (["kornia/geometry/transform/imgwarp.py", "tests/geometry/test_imgwarp.py"], "geometry"),
        (["kornia/filters/gaussian.py", "kornia/color/hsv.py", "kornia/color/lab.py"], "color,filters"),
        (["kornia/core/check.py"], "all"),
        (["kornia/utils/helpers.py"], "all"),
        (["kornia/onnx/sequential.py"], "all"),
        (["kornia/__init__.py"], "all"),
        (["docs/export_support/cases_misc.py"], "all"),
        (["testing/known_failure_xfails/onnx_conformance_torch2.14.txt"], "all"),
        (["tests/onnx/test_conformance.py"], "all"),
        (["uv.lock"], "all"),
        (["pixi.toml"], "all"),
        (["pyproject.toml"], "all"),
        (["conftest.py"], "all"),
        (["docs/source/get-started/onnx.rst", "README.md"], ""),
        (["kornia/contrib/face_detection.py", "kornia/models/sam/model.py"], ""),
        (["tests/geometry/test_homography.py"], ""),
        ([], ""),
    ],
)
def test_select(changed, expected):
    assert select(changed) == expected


def test_cli_reads_stdin():
    result = subprocess.run(  # noqa: S603 -- the current test interpreter on the script under test.
        [sys.executable, str(SCRIPT)],
        input="kornia/feature/laf.py\nkornia/enhance/adjust.py\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "enhance,feature\n"


def test_stable_core_matches_the_test_and_the_stability_page():
    assert STABLE_CORE == onnx_conformance.STABLE_CORE
    page = (REPO / "docs" / "source" / "get-started" / "stability.rst").read_text(encoding="utf-8")
    paragraph = page.split("**Stable core**", 1)[1].split("\n\n", 1)[0]
    assert set(re.findall(r"``kornia\.(\w+)``", paragraph)) == set(STABLE_CORE)
